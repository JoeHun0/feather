//! Texture bake (§17): turn every texture a scene uses into a BC7 mip chain in a
//! content-addressed cache that the runtime uploads instead of raw RGBA8.
//!
//!     cargo run --release -p feather-bake -- SCENE.gltf|.glb... [--out DIR]
//!
//! For each (image, colour space) pair a scene actually uses (base colour as
//! sRGB, normal and metallic-roughness as data), the bake builds mips on the
//! CPU (averaging sRGB in linear space, like the runtime's GPU blits), then
//! BC7-encodes every level with Intel's ISPC encoder. Output goes to
//! `<key>.bc7` under `--out` (default `scratch/bake/tex`), keyed exactly as
//! the runtime keys it (`feather_assets::bake::texture_key`). It's
//! incremental: pairs whose file exists are skipped. Open sources stay the
//! truth; the cache can be deleted at any time.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use feather_assets::bake::{baked_path, texture_key, BakedTexture};
use feather_assets::{load_gltf_scene, TextureData};

fn main() {
    let mut out = PathBuf::from("scratch/bake/tex");
    let mut scenes = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = args.next().expect("--out needs a directory").into(),
            _ => scenes.push(a),
        }
    }
    if scenes.is_empty() {
        eprintln!("usage: feather-bake SCENE.gltf|.glb... [--out DIR]");
        std::process::exit(2);
    }
    std::fs::create_dir_all(&out).expect("create the bake directory");

    // Every (texture, colour space) pair in use, deduped by content key across
    // all scenes.
    let mut jobs: HashMap<u64, (Arc<TextureData>, bool)> = HashMap::new();
    for scene in &scenes {
        let data = load_gltf_scene(scene).unwrap_or_else(|e| panic!("{scene}: {e}"));
        for m in data.meshes.iter().map(|m| &m.material) {
            for (tex, srgb) in [
                (&m.base_color_texture, true),
                (&m.normal_texture, false),
                (&m.metallic_roughness_texture, false),
            ] {
                if let Some(t) = tex {
                    jobs.entry(texture_key(t, srgb))
                        .or_insert((t.clone(), srgb));
                }
            }
        }
    }
    let todo: Vec<(u64, Arc<TextureData>, bool)> = jobs
        .into_iter()
        .filter(|(key, _)| !baked_path(&out, *key).exists())
        .map(|(key, (t, srgb))| (key, t, srgb))
        .collect();
    let cached = count_cached(&out, &scenes);
    println!("{} to bake, {cached} already cached", todo.len());

    // Textures in parallel; each is independent.
    let t0 = Instant::now();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let bytes: usize = std::thread::scope(|s| {
        let chunks: Vec<_> = todo.chunks(todo.len().div_ceil(threads).max(1)).collect();
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let out = &out;
                s.spawn(move || {
                    chunk
                        .iter()
                        .map(|(key, tex, srgb)| {
                            let baked = bake(tex, *srgb);
                            baked
                                .write(&baked_path(out, *key))
                                .expect("write baked texture");
                            baked.levels.iter().map(Vec::len).sum::<usize>()
                        })
                        .sum::<usize>()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    println!(
        "baked {} textures ({:.1} MB of BC7) in {:.1} s into {}",
        todo.len(),
        bytes as f64 / 1_048_576.0,
        t0.elapsed().as_secs_f32(),
        out.display()
    );
}

/// How many of the scenes' pairs were already baked (for the summary line).
fn count_cached(out: &Path, scenes: &[String]) -> usize {
    let mut keys = std::collections::HashSet::new();
    for scene in scenes {
        let data = load_gltf_scene(scene).expect("scene loaded above");
        for m in data.meshes.iter().map(|m| &m.material) {
            for (tex, srgb) in [
                (&m.base_color_texture, true),
                (&m.normal_texture, false),
                (&m.metallic_roughness_texture, false),
            ] {
                if let Some(t) = tex {
                    keys.insert(texture_key(t, srgb));
                }
            }
        }
    }
    keys.into_iter()
        .filter(|k| baked_path(out, *k).exists())
        .count()
}

/// One texture's BC7 mip chain.
fn bake(tex: &TextureData, srgb: bool) -> BakedTexture {
    let settings = intel_tex_2::bc7::alpha_basic_settings();
    let levels = mip_chain(tex, srgb)
        .iter()
        .map(|level| {
            let padded = pad_to_blocks(level);
            intel_tex_2::bc7::compress_blocks(
                &settings,
                &intel_tex_2::RgbaSurface {
                    data: &padded.pixels,
                    width: padded.width,
                    height: padded.height,
                    stride: padded.width * 4,
                },
            )
        })
        .collect();
    BakedTexture {
        srgb,
        width: tex.width,
        height: tex.height,
        levels,
    }
}

/// Every level from full size down to 1×1, each a 2×2 box filter of the one
/// above (edges clamped for odd sizes). sRGB colour is averaged in *linear*
/// space: averaging the encoded values would darken every mip, the same bug
/// the runtime's GPU blit path avoids.
fn mip_chain(tex: &TextureData, srgb: bool) -> Vec<TextureData> {
    let mut chain = vec![tex.clone()];
    while let Some(prev) = chain.last().filter(|t| t.width > 1 || t.height > 1) {
        let (w, h) = ((prev.width / 2).max(1), (prev.height / 2).max(1));
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let taps = [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dy)| {
                    let sx = (2 * x + dx).min(prev.width - 1);
                    let sy = (2 * y + dy).min(prev.height - 1);
                    ((sy * prev.width + sx) * 4) as usize
                });
                for c in 0..4 {
                    let colour = srgb && c < 3; // alpha is always linear
                    let sum: f32 = taps
                        .iter()
                        .map(|&i| {
                            let v = prev.pixels[i + c];
                            if colour {
                                srgb_to_linear(v)
                            } else {
                                v as f32 / 255.0
                            }
                        })
                        .sum();
                    let avg = sum / 4.0;
                    pixels.push(if colour {
                        linear_to_srgb(avg)
                    } else {
                        (avg * 255.0).round() as u8
                    });
                }
            }
        }
        chain.push(TextureData {
            pixels,
            width: w,
            height: h,
        });
    }
    chain
}

fn srgb_to_linear(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(l: f32) -> u8 {
    let c = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (c.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// BC7 encodes whole 4×4 blocks. Pad a level up to them by repeating the edge,
/// so the padding can't bleed a foreign colour into the real texels' blocks.
fn pad_to_blocks(t: &TextureData) -> TextureData {
    let (w, h) = (t.width.div_ceil(4) * 4, t.height.div_ceil(4) * 4);
    if (w, h) == (t.width, t.height) {
        return t.clone();
    }
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let i = ((y.min(t.height - 1) * t.width + x.min(t.width - 1)) * 4) as usize;
            pixels.extend_from_slice(&t.pixels[i..i + 4]);
        }
    }
    TextureData {
        pixels,
        width: w,
        height: h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use feather_assets::bake::bc7_level_size;

    fn tex(width: u32, height: u32, px: impl Fn(u32, u32) -> [u8; 4]) -> TextureData {
        let mut pixels = Vec::new();
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&px(x, y));
            }
        }
        TextureData {
            pixels,
            width,
            height,
        }
    }

    #[test]
    fn srgb_mips_average_light_not_encoded_values() {
        // A black/white checker: its true average brightness is 50% *light*,
        // which is 188 in sRGB. Averaging the encoded bytes would give 128, a
        // visibly darker mip.
        let checker = tex(2, 2, |x, y| {
            if (x + y) % 2 == 0 {
                [0, 0, 0, 255]
            } else {
                [255, 255, 255, 255]
            }
        });
        assert_eq!(
            &mip_chain(&checker, true)[1].pixels[..4],
            &[188, 188, 188, 255]
        );
        // Data textures (normal, metallic-roughness) average the values as-is.
        assert_eq!(
            &mip_chain(&checker, false)[1].pixels[..4],
            &[128, 128, 128, 255]
        );
    }

    #[test]
    fn chains_end_at_one_by_one() {
        for (w, h, levels) in [(8, 8, 4), (8, 2, 4), (5, 3, 3), (1, 1, 1)] {
            let chain = mip_chain(&tex(w, h, |_, _| [9, 9, 9, 255]), false);
            assert_eq!(chain.len(), levels, "{w}x{h}");
            let last = chain.last().unwrap();
            assert_eq!((last.width, last.height), (1, 1), "{w}x{h}");
        }
    }

    #[test]
    fn baked_levels_have_exactly_the_blocks_vulkan_expects() {
        // Odd, non-square, not a multiple of 4: every level must still be
        // exactly the size the runtime (and Vulkan) will copy.
        let t = tex(10, 6, |x, y| [x as u8 * 20, y as u8 * 40, 7, 255]);
        let baked = bake(&t, true);
        assert_eq!(baked.levels.len(), 4);
        for (l, level) in baked.levels.iter().enumerate() {
            assert_eq!(level.len(), bc7_level_size(10, 6, l as u32), "level {l}");
        }
    }
}
