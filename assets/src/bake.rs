//! Baked textures (§17): the half of the texture bake that the runtime and the
//! `bake` tool share, which is the content key and the file format.
//!
//! The bake turns each (image, colour space) a scene uses into a BC7 mip chain,
//! stored under a key derived from the decoded pixels. The runtime computes the
//! same key and uploads the baked chain when it finds one, falling back to raw
//! RGBA8 when it doesn't. Open sources stay the truth, and the cache is
//! disposable.

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::TextureData;

/// Bump when the bake's output changes meaning (mip filter, encoder settings,
/// file layout): stale cache entries then simply aren't found.
pub const BAKE_VERSION: u32 = 1;
const MAGIC: [u8; 4] = *b"FBTX";

/// Content key of one texture as the bake sees it. Keyed on the *decoded*
/// pixels, so it's the same however the image was stored (file, GLB buffer,
/// data URI), plus the colour space, since the same pixels baked as sRGB and
/// as data are different outputs.
pub fn texture_key(tex: &TextureData, srgb: bool) -> u64 {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    h.update(&BAKE_VERSION.to_le_bytes());
    h.update(&tex.width.to_le_bytes());
    h.update(&tex.height.to_le_bytes());
    h.update(&[srgb as u8]);
    h.update(&tex.pixels);
    h.digest()
}

/// Where a key's baked file lives in the cache directory.
pub fn baked_path(dir: &Path, key: u64) -> PathBuf {
    dir.join(format!("{key:016x}.bc7"))
}

/// Bytes of BC7 blocks for one mip level: 16 per 4×4 block, partial blocks
/// rounded up, which is exactly what Vulkan expects for a BC7 level.
pub fn bc7_level_size(width: u32, height: u32, level: u32) -> usize {
    let w = (width >> level).max(1);
    let h = (height >> level).max(1);
    (w.div_ceil(4) * h.div_ceil(4) * 16) as usize
}

/// A baked BC7 mip chain: level 0 first, down to 1×1.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedTexture {
    pub srgb: bool,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Vec<u8>>,
}

impl BakedTexture {
    /// Thin header + raw GPU-ready payloads (§17): magic, version, sRGB flag,
    /// width, height, level count, then each level's length and blocks.
    /// Written to a temporary name and renamed, so an interrupted bake never
    /// leaves a truncated file that a later run would trust.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let tmp = path.with_extension("tmp");
        {
            let mut f = io::BufWriter::new(std::fs::File::create(&tmp)?);
            f.write_all(&MAGIC)?;
            for v in [
                BAKE_VERSION,
                self.srgb as u32,
                self.width,
                self.height,
                self.levels.len() as u32,
            ] {
                f.write_all(&v.to_le_bytes())?;
            }
            for level in &self.levels {
                f.write_all(&(level.len() as u32).to_le_bytes())?;
                f.write_all(level)?;
            }
            f.flush()?;
        }
        std::fs::rename(tmp, path)
    }

    /// Read and *validate* a baked file: a wrong magic, a version from another
    /// bake, or a level whose size doesn't match its dimensions is an error,
    /// and the caller falls back to the raw texture.
    pub fn read(path: &Path) -> io::Result<Self> {
        let mut data = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut data)?;
        let mut r = Cursor { data: &data, at: 0 };
        if r.take(4)? != MAGIC {
            return Err(invalid("not a baked texture"));
        }
        let (version, srgb, width, height, count) =
            (r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?);
        if version != BAKE_VERSION {
            return Err(invalid("baked by another version"));
        }
        let mut levels = Vec::with_capacity(count as usize);
        for level in 0..count {
            let len = r.u32()? as usize;
            if len != bc7_level_size(width, height, level) {
                return Err(invalid("level size doesn't match its dimensions"));
            }
            levels.push(r.take(len)?.to_vec());
        }
        Ok(Self {
            srgb: srgb != 0,
            width,
            height,
            levels,
        })
    }
}

fn invalid(why: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, why.to_string())
}

/// Bounds-checked reads over a baked file.
struct Cursor<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let s = self
            .data
            .get(self.at..self.at + n)
            .ok_or_else(|| invalid("truncated"))?;
        self.at += n;
        Ok(s)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tex(fill: u8) -> TextureData {
        TextureData {
            pixels: vec![fill; 8 * 8 * 4],
            width: 8,
            height: 8,
        }
    }

    #[test]
    fn keys_are_content_and_colour_space_addressed() {
        assert_eq!(texture_key(&tex(1), true), texture_key(&tex(1), true));
        assert_ne!(texture_key(&tex(1), true), texture_key(&tex(2), true));
        // Same pixels, different colour space: different bakes.
        assert_ne!(texture_key(&tex(1), true), texture_key(&tex(1), false));
    }

    #[test]
    fn bc7_level_sizes_round_partial_blocks_up() {
        assert_eq!(bc7_level_size(2048, 2048, 0), 2048 * 2048); // 1 byte/pixel
        assert_eq!(bc7_level_size(8, 8, 1), 16); // 4x4: one block
        assert_eq!(bc7_level_size(8, 8, 3), 16); // 1x1 still costs a block
        assert_eq!(bc7_level_size(6, 2, 0), 32); // 2 blocks by 1
    }

    #[test]
    fn baked_files_round_trip_and_reject_damage() {
        let dir = std::env::temp_dir().join(format!("feather_bake_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let baked = BakedTexture {
            srgb: true,
            width: 8,
            height: 4,
            levels: (0..4)
                .map(|l| vec![l as u8; bc7_level_size(8, 4, l)])
                .collect(),
        };
        let path = baked_path(&dir, 0x1234);
        baked.write(&path).unwrap();
        assert_eq!(BakedTexture::read(&path).unwrap(), baked);

        // A truncated file is rejected, not trusted.
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(BakedTexture::read(&path).is_err());
        // So is a level whose size doesn't match its dimensions.
        let wrong = BakedTexture {
            levels: vec![vec![0; 3]],
            ..baked
        };
        wrong.write(&path).unwrap();
        assert!(BakedTexture::read(&path).is_err());
    }
}
