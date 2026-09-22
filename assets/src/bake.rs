//! Baked assets (§17): the half of the bake that the runtime and the `bake`
//! tool share, which is the content keys and the file formats.
//!
//! The bake turns each (image, colour space) a scene uses into a BC7 mip chain,
//! and each mesh into an optimised vertex order plus a LOD chain, stored under
//! keys derived from the decoded content. The runtime computes the same keys
//! and uses a baked file when it finds one, falling back to the raw asset when
//! it doesn't. Open sources stay the truth, and the cache is disposable.
//!
//! Layout under the bake root: `tex/<key>.bc7` and `mesh/<key>.fbm`.

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::{MeshData, TextureData, Vertex};

/// Subdirectory of the bake root holding baked textures.
pub const TEX_DIR: &str = "tex";
/// Subdirectory of the bake root holding baked meshes.
pub const MESH_DIR: &str = "mesh";

/// Bump when the bake's output changes meaning (mip filter, encoder settings,
/// file layout): stale cache entries then simply aren't found.
pub const BAKE_VERSION: u32 = 2;
const MAGIC: [u8; 4] = *b"FBTX";

/// What a texture is used as, which decides how it's baked.
///
/// Normal maps are `Data`, i.e. three-channel BC7, on purpose. BC5 (X, Y with Z
/// rebuilt in the shader) was measured on the detail scene's four normal maps
/// and rejected: the rock and grass maps hold inward-pointing and non-unit
/// texels (25% and 24% of them) that only a third channel reproduces, so BC5
/// moved their mean error from 1.1° to 17.8° (rocks) and 1.4° to 5.2° (grass),
/// while gaining under 0.1° on the clean bust map. Same VRAM either way.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TexKind {
    /// Base colour: BC7, sampled as sRGB.
    Color,
    /// Linear data (metallic-roughness, normal maps): BC7, UNORM.
    Data,
}

impl TexKind {
    /// Sampled with sRGB decoding (only base colour).
    pub fn srgb(self) -> bool {
        self == TexKind::Color
    }

    fn tag(self) -> u32 {
        match self {
            TexKind::Data => 0,
            TexKind::Color => 1,
        }
    }

    fn from_tag(tag: u32) -> Option<Self> {
        [TexKind::Data, TexKind::Color]
            .into_iter()
            .find(|k| k.tag() == tag)
    }
}

/// Content key of one texture as the bake sees it: the image's **encoded**
/// bytes (`source_key`), so the runtime can find the bake without decoding
/// anything, plus the kind, since the same image baked as sRGB colour and as
/// linear data gives different outputs.
pub fn texture_key(tex: &TextureData, kind: TexKind) -> u64 {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    h.update(&BAKE_VERSION.to_le_bytes());
    h.update(&kind.tag().to_le_bytes());
    h.update(&tex.width.to_le_bytes());
    h.update(&tex.height.to_le_bytes());
    h.update(&tex.source_key.to_le_bytes());
    h.digest()
}

/// Where a key's baked file lives in the cache directory.
pub fn baked_path(dir: &Path, key: u64) -> PathBuf {
    dir.join(format!("{key:016x}.bc7"))
}

/// Bump when the mesh bake's output changes meaning (simplifier settings, LOD
/// rules, file layout). Separate from `BAKE_VERSION` so re-tuning LODs doesn't
/// invalidate the (slow) texture bake.
pub const MESH_BAKE_VERSION: u32 = 1;
/// Meshes under this many triangles bake to LOD0 only: their cost is the
/// draw, not the triangles. Shared so the runtime only suggests baking meshes
/// that would gain LODs.
pub const MIN_LOD_TRIS: usize = 256;
const MESH_MAGIC: [u8; 4] = *b"FBMS";

/// Content key of one mesh: its vertices and indices exactly as the loader
/// produced them. The material isn't part of it, so a shape reused with
/// several materials bakes once.
pub fn mesh_key(mesh: &MeshData) -> u64 {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    h.update(&MESH_BAKE_VERSION.to_le_bytes());
    h.update(&(mesh.vertices.len() as u64).to_le_bytes());
    h.update(pod_bytes(&mesh.vertices));
    h.update(pod_bytes(&mesh.indices));
    h.digest()
}

/// Where a key's baked mesh lives in the mesh directory.
pub fn baked_mesh_path(dir: &Path, key: u64) -> PathBuf {
    dir.join(format!("{key:016x}.fbm"))
}

fn pod_bytes<T: Copy>(s: &[T]) -> &[u8] {
    // Vertex is repr(C) f32s and u32 has no padding: every byte is initialised.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// One level of detail: an index list into the shared vertex array.
#[derive(Debug, Clone, PartialEq)]
pub struct Lod {
    /// Geometric error of this level against the original, in mesh-local
    /// units (0 for LOD0). The runtime projects it to decide when the level
    /// is good enough.
    pub error: f32,
    pub indices: Vec<u32>,
}

/// A baked mesh: vertices reordered for fetch locality, shared by every LOD,
/// and LODs from full detail (0) to coarsest.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedMesh {
    pub vertices: Vec<Vertex>,
    pub lods: Vec<Lod>,
}

impl BakedMesh {
    /// Magic, version, vertex count, raw `Vertex` array, LOD count, then per
    /// LOD its error, index count and indices. Temp + rename like textures.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let tmp = path.with_extension("tmp");
        {
            let mut f = io::BufWriter::new(std::fs::File::create(&tmp)?);
            f.write_all(&MESH_MAGIC)?;
            f.write_all(&MESH_BAKE_VERSION.to_le_bytes())?;
            f.write_all(&(self.vertices.len() as u32).to_le_bytes())?;
            f.write_all(pod_bytes(&self.vertices))?;
            f.write_all(&(self.lods.len() as u32).to_le_bytes())?;
            for lod in &self.lods {
                f.write_all(&lod.error.to_le_bytes())?;
                f.write_all(&(lod.indices.len() as u32).to_le_bytes())?;
                f.write_all(pod_bytes(&lod.indices))?;
            }
            f.flush()?;
        }
        std::fs::rename(tmp, path)
    }

    /// Read and *validate* a baked mesh. Anything the renderer would trust
    /// blindly is checked: sizes, whole triangles, every index in range, at
    /// least one LOD, and errors that never decrease (the selection walks the
    /// chain assuming coarser means worse). Any failure is an error and the
    /// caller uses the raw mesh.
    pub fn read(path: &Path) -> io::Result<Self> {
        let mut data = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut data)?;
        let mut r = Cursor { data: &data, at: 0 };
        if r.take(4)? != MESH_MAGIC {
            return Err(invalid("not a baked mesh"));
        }
        if r.u32()? != MESH_BAKE_VERSION {
            return Err(invalid("baked by another version"));
        }
        let vertex_count = r.u32()? as usize;
        let raw = r.take(vertex_count * std::mem::size_of::<Vertex>())?;
        // 8 little-endian f32s per vertex: pos, normal, uv.
        let (floats, _) = raw.as_chunks::<4>();
        let vertices: Vec<Vertex> = floats
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| {
                let f = v.map(f32::from_le_bytes);
                Vertex {
                    pos: [f[0], f[1], f[2]],
                    normal: [f[3], f[4], f[5]],
                    uv: [f[6], f[7]],
                }
            })
            .collect();
        let lod_count = r.u32()?;
        if lod_count == 0 {
            return Err(invalid("no LODs"));
        }
        let mut lods: Vec<Lod> = Vec::with_capacity(lod_count as usize);
        for _ in 0..lod_count {
            let error = f32::from_le_bytes(r.take(4)?.try_into().unwrap());
            let count = r.u32()? as usize;
            if count == 0 || !count.is_multiple_of(3) {
                return Err(invalid("LOD isn't whole triangles"));
            }
            let indices: Vec<u32> = r
                .take(count * 4)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            if indices.iter().any(|&i| i as usize >= vertex_count) {
                return Err(invalid("index out of range"));
            }
            let prev = lods.last().map_or(0.0, |l| l.error);
            if !error.is_finite() || error < prev {
                return Err(invalid("LOD errors must be finite and non-decreasing"));
            }
            lods.push(Lod { error, indices });
        }
        if r.at != data.len() {
            return Err(invalid("trailing bytes"));
        }
        Ok(Self { vertices, lods })
    }
}

/// Bytes of one BC7 mip level: 16 per 4×4 block, partial blocks rounded up,
/// which is exactly what Vulkan expects.
pub fn bc7_level_size(width: u32, height: u32, level: u32) -> usize {
    let w = (width >> level).max(1);
    let h = (height >> level).max(1);
    (w.div_ceil(4) * h.div_ceil(4) * 16) as usize
}

/// A baked BC7 mip chain, level 0 first, down to 1×1.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedTexture {
    pub kind: TexKind,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Vec<u8>>,
}

impl BakedTexture {
    /// Thin header + raw GPU-ready payloads (§17): magic, version, kind,
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
                self.kind.tag(),
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
        let (version, kind, width, height, count) =
            (r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?);
        if version != BAKE_VERSION {
            return Err(invalid("baked by another version"));
        }
        let kind = TexKind::from_tag(kind).ok_or_else(|| invalid("unknown texture kind"))?;
        let mut levels = Vec::with_capacity(count as usize);
        for level in 0..count {
            let len = r.u32()? as usize;
            if len != bc7_level_size(width, height, level) {
                return Err(invalid("level size doesn't match its dimensions"));
            }
            levels.push(r.take(len)?.to_vec());
        }
        Ok(Self {
            kind,
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
        TextureData::from_rgba8(vec![fill; 8 * 8 * 4], 8, 8)
    }

    #[test]
    fn keys_are_content_and_kind_addressed() {
        let k = |t: &TextureData, kind| texture_key(t, kind);
        assert_eq!(k(&tex(1), TexKind::Color), k(&tex(1), TexKind::Color));
        assert_ne!(k(&tex(1), TexKind::Color), k(&tex(2), TexKind::Color));
        // Same image, different use: different bakes.
        assert_ne!(k(&tex(1), TexKind::Color), k(&tex(1), TexKind::Data));
        // Keyed on the source, not on decoding: two textures with the same
        // encoded bytes share a key without either being decoded.
        let png = |c: u8| {
            let mut out = Vec::new();
            image::RgbImage::from_pixel(2, 2, image::Rgb([c, 0, 0]))
                .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
                .unwrap();
            TextureData::from_encoded(out).unwrap()
        };
        let (a, b) = (png(7), png(7));
        assert_eq!(k(&a, TexKind::Data), k(&b, TexKind::Data));
        assert_ne!(k(&a, TexKind::Data), k(&png(8), TexKind::Data));
        assert!(!a.is_decoded() && !b.is_decoded());
    }

    #[test]
    fn bc7_level_sizes_round_partial_blocks_up() {
        assert_eq!(bc7_level_size(2048, 2048, 0), 2048 * 2048); // 1 byte/pixel
        assert_eq!(bc7_level_size(8, 8, 1), 16); // 4x4: one block
        assert_eq!(bc7_level_size(8, 8, 3), 16); // 1x1 still costs a block
        assert_eq!(bc7_level_size(6, 2, 0), 32); // 2 blocks by 1
    }

    fn mesh() -> BakedMesh {
        let v = |x: f32| Vertex {
            pos: [x, 0.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            uv: [x, 0.5],
        };
        BakedMesh {
            vertices: (0..4).map(|i| v(i as f32)).collect(),
            lods: vec![
                Lod {
                    error: 0.0,
                    indices: vec![0, 1, 2, 0, 2, 3],
                },
                Lod {
                    error: 0.25,
                    indices: vec![0, 1, 3],
                },
            ],
        }
    }

    #[test]
    fn mesh_keys_ignore_material_but_not_geometry() {
        let m = mesh();
        let data = |indices: Vec<u32>| MeshData {
            vertices: m.vertices.clone(),
            indices,
            material: crate::Material::default(),
        };
        let a = data(vec![0, 1, 2]);
        let mut b = data(vec![0, 1, 2]);
        b.material.roughness = 0.1;
        assert_eq!(mesh_key(&a), mesh_key(&b));
        assert_ne!(mesh_key(&a), mesh_key(&data(vec![0, 2, 1])));
    }

    #[test]
    fn baked_meshes_round_trip_and_reject_damage() {
        let dir = std::env::temp_dir().join(format!("feather_bake_mesh_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = baked_mesh_path(&dir, 0x42);
        mesh().write(&path).unwrap();
        assert_eq!(BakedMesh::read(&path).unwrap(), mesh());

        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(BakedMesh::read(&path).is_err(), "truncated");
        let mut extra = bytes.clone();
        extra.push(0);
        std::fs::write(&path, &extra).unwrap();
        assert!(BakedMesh::read(&path).is_err(), "trailing bytes");
        let mut magic = bytes.clone();
        magic[0] = b'X';
        std::fs::write(&path, &magic).unwrap();
        assert!(BakedMesh::read(&path).is_err(), "bad magic");

        for (why, damage) in [
            ("index out of range", {
                let mut m = mesh();
                m.lods[1].indices[2] = 4;
                m
            }),
            ("decreasing error", {
                let mut m = mesh();
                m.lods[1].error = -1.0;
                m
            }),
            ("NaN error", {
                let mut m = mesh();
                m.lods[1].error = f32::NAN;
                m
            }),
            ("partial triangle", {
                let mut m = mesh();
                m.lods[1].indices.pop();
                m
            }),
            ("no LODs", {
                let mut m = mesh();
                m.lods.clear();
                m
            }),
        ] {
            damage.write(&path).unwrap();
            assert!(BakedMesh::read(&path).is_err(), "{why} was accepted");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn baked_files_round_trip_and_reject_damage() {
        let dir = std::env::temp_dir().join(format!("feather_bake_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let baked = BakedTexture {
            kind: TexKind::Color,
            width: 8,
            height: 4,
            levels: (0..4)
                .map(|l| vec![l as u8; bc7_level_size(8, 4, l)])
                .collect(),
        };
        let path = baked_path(&dir, 0x1234);
        for kind in [TexKind::Color, TexKind::Data] {
            let b = BakedTexture {
                kind,
                ..baked.clone()
            };
            b.write(&path).unwrap();
            assert_eq!(BakedTexture::read(&path).unwrap(), b);
        }
        // An unknown kind tag is rejected (it's the u32 after magic + version).
        baked.write(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[8] = 9;
        std::fs::write(&path, &bytes).unwrap();
        assert!(BakedTexture::read(&path).is_err());
        baked.write(&path).unwrap();

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
