//! Asset bake (§17): textures to BC7 mip chains and meshes to an optimised
//! vertex order plus a LOD chain, in a content-addressed cache the runtime
//! uses instead of the raw assets.
//!
//!     cargo run --release -p feather-bake -- SCENE.gltf|.glb... [--out DIR]
//!
//! Textures: for each (image, colour space) pair a scene actually uses (base
//! colour as sRGB, normal and metallic-roughness as data), the bake builds mips
//! on the CPU (averaging sRGB in linear space, like the runtime's GPU blits),
//! then BC7-encodes every level with Intel's ISPC encoder.
//!
//! Meshes: see `mesh.rs` (meshoptimizer vertex-cache + fetch order, LODs).
//!
//! Output goes under `--out` (default `scratch/bake`): `tex/<key>.bc7` and
//! `mesh/<key>.fbm`, keyed exactly as the runtime keys them. It's
//! incremental: anything whose file exists is skipped. Open sources stay the
//! truth; the cache can be deleted at any time.

mod mesh;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use feather_assets::bake::{
    baked_mesh_path, baked_path, mesh_key, texture_key, BakedTexture, MESH_DIR, TEX_DIR,
};
use feather_assets::{load_gltf_scene, MeshData, TextureData};

fn main() {
    let mut out = PathBuf::from("scratch/bake");
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
    let (tex_dir, mesh_dir) = (out.join(TEX_DIR), out.join(MESH_DIR));
    std::fs::create_dir_all(&tex_dir).expect("create the texture bake directory");
    std::fs::create_dir_all(&mesh_dir).expect("create the mesh bake directory");

    // Every (texture, colour space) pair and every mesh in use, deduped by
    // content key across all scenes.
    let mut textures: HashMap<u64, (Arc<TextureData>, bool)> = HashMap::new();
    let mut meshes: HashMap<u64, MeshData> = HashMap::new();
    for scene in &scenes {
        let data = load_gltf_scene(scene).unwrap_or_else(|e| panic!("{scene}: {e}"));
        for m in &data.meshes {
            let mat = &m.material;
            for (tex, srgb) in [
                (&mat.base_color_texture, true),
                (&mat.normal_texture, false),
                (&mat.metallic_roughness_texture, false),
            ] {
                if let Some(t) = tex {
                    textures
                        .entry(texture_key(t, srgb))
                        .or_insert((t.clone(), srgb));
                }
            }
            meshes.entry(mesh_key(m)).or_insert_with(|| m.clone());
        }
    }

    let (tex_todo, tex_cached): (Vec<_>, Vec<_>) = textures
        .into_iter()
        .partition(|(key, _)| !baked_path(&tex_dir, *key).exists());
    println!(
        "textures: {} to bake, {} already cached",
        tex_todo.len(),
        tex_cached.len()
    );
    let t0 = Instant::now();
    let bytes: usize = parallel(&tex_todo, |(key, (tex, srgb))| {
        let baked = bake(tex, *srgb);
        baked
            .write(&baked_path(&tex_dir, *key))
            .expect("write baked texture");
        baked.levels.iter().map(Vec::len).sum::<usize>()
    })
    .into_iter()
    .sum();
    println!(
        "baked {} textures ({:.1} MB of BC7) in {:.1} s into {}",
        tex_todo.len(),
        bytes as f64 / 1_048_576.0,
        t0.elapsed().as_secs_f32(),
        tex_dir.display()
    );

    let (mesh_todo, mesh_cached): (Vec<_>, Vec<_>) = meshes
        .into_iter()
        .partition(|(key, _)| !baked_mesh_path(&mesh_dir, *key).exists());
    println!(
        "meshes: {} to bake, {} already cached",
        mesh_todo.len(),
        mesh_cached.len()
    );
    let t0 = Instant::now();
    let mut reports = parallel(&mesh_todo, |(key, m)| {
        let baked = mesh::bake_mesh(m);
        baked
            .write(&baked_mesh_path(&mesh_dir, *key))
            .expect("write baked mesh");
        let tris: Vec<String> = baked
            .lods
            .iter()
            .map(|l| (l.indices.len() / 3).to_string())
            .collect();
        let errors: Vec<String> = baked.lods[1..]
            .iter()
            .map(|l| format!("{:.2e}", l.error))
            .collect();
        (
            m.indices.len() / 3,
            format!(
                "  {key:016x}: tris {} | error {}",
                tris.join(" > "),
                if errors.is_empty() {
                    "-".into()
                } else {
                    errors.join(" ")
                }
            ),
        )
    });
    // Biggest first: those are the ones worth reading.
    reports.sort_by_key(|r| std::cmp::Reverse(r.0));
    for (_, line) in &reports {
        println!("{line}");
    }
    println!(
        "baked {} meshes in {:.1} s into {}",
        mesh_todo.len(),
        t0.elapsed().as_secs_f32(),
        mesh_dir.display()
    );
}

/// `f` over `items` on every core, results in input order.
fn parallel<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let f = &f;
    std::thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(items.len().div_ceil(threads).max(1))
            .map(|chunk| s.spawn(move || chunk.iter().map(f).collect::<Vec<R>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    })
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
