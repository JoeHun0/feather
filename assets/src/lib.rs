//! Assets: runtime blob loaders, handle tables, bake format definitions.
//!
//! For now this owns the CPU-side mesh representation shared with the renderer
//! (`Vertex` / `MeshData`), a procedural sphere, and a minimal glTF mesh loader.
//! Everything here is Vulkan-free by design (§22) — the renderer uploads these
//! buffers to the GPU.

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

use glam::{Mat4, Vec3};

/// Interleaved static vertex. Matches the renderer's vertex attribute layout
/// (position at offset 0, normal at offset 12, uv at offset 24). `#[repr(C)]` so
/// a slice uploads straight into a device-local vertex buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// Decoded RGBA8 image (sRGB base-color). Owned CPU pixels the renderer uploads.
#[derive(Clone, Debug)]
pub struct TextureData {
    pub pixels: Vec<u8>, // RGBA8, row-major, width*height*4 bytes
    pub width: u32,
    pub height: u32,
}

/// Surface parameters for a mesh (glTF metallic-roughness aligned). Factor
/// colors are **linear**; `base_color_texture` is sRGB; `normal_texture` and
/// `metallic_roughness_texture` are linear data (uploaded `_UNORM`). MR packing
/// follows glTF: green = roughness, blue = metallic.
#[derive(Clone, Debug)]
pub struct Material {
    pub base_color: [f32; 4], // linear RGBA factor
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3], // linear RGB
    pub normal_scale: f32,
    pub base_color_texture: Option<TextureData>,
    pub normal_texture: Option<TextureData>,
    pub metallic_roughness_texture: Option<TextureData>,
}

impl Default for Material {
    fn default() -> Self {
        // White dielectric, no textures.
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0, 0.0, 0.0],
            normal_scale: 1.0,
            base_color_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
        }
    }
}

/// CPU-side mesh: interleaved vertices + a 32-bit index buffer, plus the surface
/// material. One shared representation for both procedural and loaded geometry.
#[derive(Clone, Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material: Material,
}

impl MeshData {
    /// Axis-aligned bounds `(min, max)`. Zeros for an empty mesh.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        if self.vertices.is_empty() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &self.vertices {
            let p = Vec3::from(v.pos);
            min = min.min(p);
            max = max.max(p);
        }
        (min, max)
    }

    /// UV sphere with per-vertex normals (the previous hardcoded demo mesh).
    pub fn uv_sphere(stacks: u32, slices: u32, radius: f32) -> Self {
        let mut vertices = Vec::with_capacity(((stacks + 1) * (slices + 1)) as usize);
        for i in 0..=stacks {
            let phi = std::f32::consts::PI * i as f32 / stacks as f32; // 0..pi
            let (sp, cp) = phi.sin_cos();
            for j in 0..=slices {
                let theta = std::f32::consts::TAU * j as f32 / slices as f32; // 0..2pi
                let (st, ct) = theta.sin_cos();
                let n = [sp * ct, cp, sp * st];
                vertices.push(Vertex {
                    pos: [n[0] * radius, n[1] * radius, n[2] * radius],
                    normal: n,
                    uv: [j as f32 / slices as f32, i as f32 / stacks as f32],
                });
            }
        }
        let mut indices = Vec::with_capacity((stacks * slices * 6) as usize);
        let stride = slices + 1;
        for i in 0..stacks {
            for j in 0..slices {
                let a = i * stride + j;
                let b = a + stride;
                indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
            }
        }
        Self { vertices, indices, material: Material::default() }
    }

    /// Unit-ish cube with per-face (flat) normals: 24 vertices, 36 indices.
    pub fn cube(size: f32) -> Self {
        let h = size * 0.5;
        // (normal, four CCW corners) per face; winding is irrelevant (cull NONE).
        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            ([0.0, 0.0, 1.0], [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]]), // +Z
            ([0.0, 0.0, -1.0], [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]]), // -Z
            ([1.0, 0.0, 0.0], [[h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h]]), // +X
            ([-1.0, 0.0, 0.0], [[-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h]]), // -X
            ([0.0, 1.0, 0.0], [[-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]]), // +Y
            ([0.0, -1.0, 0.0], [[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]]), // -Y
        ];
        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);
        let face_uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        for (normal, corners) in faces {
            let base = vertices.len() as u32;
            for (k, pos) in corners.into_iter().enumerate() {
                vertices.push(Vertex {
                    pos,
                    normal,
                    uv: face_uv[k],
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        Self { vertices, indices, material: Material::default() }
    }
}

/// One placement in a loaded scene: which mesh to draw and where. Several nodes
/// referencing the same mesh is the common case (props repeated across a level),
/// and is exactly what the renderer's instanced path wants.
pub struct SceneNode {
    /// Index into [`SceneData::meshes`].
    pub mesh: usize,
    /// The node's world transform, with the whole parent chain applied.
    pub transform: Mat4,
}

/// A glTF scene decomposed for the ECS: meshes in **local** space, plus one node
/// per placement. Deliberately *not* merged into a single mesh — per-primitive
/// materials survive, and repeated meshes instance instead of being duplicated.
pub struct SceneData {
    /// One entry per (glTF mesh, primitive) pair actually referenced, each with
    /// its own `material`. Local space: the node transform is on the node.
    pub meshes: Vec<MeshData>,
    pub nodes: Vec<SceneNode>,
}

/// Load a glTF / GLB **scene**: every referenced mesh primitive becomes its own
/// `MeshData` (keeping its own material), and every node referencing one becomes a
/// [`SceneNode`] carrying that node's world transform.
///
/// Primitives are deduplicated by `(mesh index, primitive index)`, so a mesh used
/// by many nodes is stored once and drawn as many instances.
///
/// Handled: positions (required), normals (computed if absent), UVs (0 if
/// absent), indices (generated if absent), the node hierarchy and its transforms,
/// triangle primitives, .glb / external / data: URI buffers and images, and each
/// primitive's base-color + normal + metallic-roughness textures (with factors +
/// normal scale).
///
/// Deferred: tangents, skinning, animation, morph targets, non-triangle
/// primitives, 16-/32-bit image formats, and `extras` gameplay data (§18).
pub fn load_gltf_scene(path: impl AsRef<Path>) -> Result<SceneData, Box<dyn Error>> {
    // import resolves all buffers (blob / external / data URI) and decodes images.
    let (doc, buffers, images) = gltf::import(path)?;

    let mut out = SceneData {
        meshes: Vec::new(),
        nodes: Vec::new(),
    };
    // (gltf mesh index, primitive index) -> index into out.meshes.
    let mut seen: HashMap<(usize, usize), usize> = HashMap::new();

    match doc.default_scene().or_else(|| doc.scenes().next()) {
        Some(scene) => {
            for node in scene.nodes() {
                walk_node(
                    &node,
                    Mat4::IDENTITY,
                    &buffers,
                    &images,
                    &mut seen,
                    &mut out,
                );
            }
        }
        // No scene graph: place every mesh at the origin.
        None => {
            for mesh in doc.meshes() {
                add_mesh_nodes(
                    &mesh,
                    Mat4::IDENTITY,
                    &buffers,
                    &images,
                    &mut seen,
                    &mut out,
                );
            }
        }
    }

    if out.nodes.is_empty() {
        return Err("glTF contained no triangle mesh geometry".into());
    }
    Ok(out)
}

fn read_material(m: &gltf::Material, images: &[gltf::image::Data]) -> Material {
    let pbr = m.pbr_metallic_roughness();
    let tex = |idx: usize| images.get(idx).and_then(to_rgba8);
    let base_color_texture = pbr
        .base_color_texture()
        .and_then(|i| tex(i.texture().source().index()));
    let metallic_roughness_texture = pbr
        .metallic_roughness_texture()
        .and_then(|i| tex(i.texture().source().index()));
    let (normal_texture, normal_scale) = match m.normal_texture() {
        Some(nt) => (tex(nt.texture().source().index()), nt.scale()),
        None => (None, 1.0),
    };
    Material {
        base_color: pbr.base_color_factor(), // linear RGBA
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        emissive: m.emissive_factor(), // linear RGB
        normal_scale,
        base_color_texture,
        normal_texture,
        metallic_roughness_texture,
    }
}

/// Normalize a decoded glTF image to RGBA8. Returns None for formats not handled
/// yet (16-/32-bit), in which case the material falls back to its color factor.
fn to_rgba8(img: &gltf::image::Data) -> Option<TextureData> {
    use gltf::image::Format;
    let (width, height) = (img.width, img.height);
    let n = (width as usize) * (height as usize) * 4;
    let px = &img.pixels;
    let pixels = match img.format {
        Format::R8G8B8A8 => px.clone(),
        Format::R8G8B8 => {
            let mut out = Vec::with_capacity(n);
            for c in px.chunks_exact(3) {
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
            out
        }
        Format::R8 => {
            let mut out = Vec::with_capacity(n);
            for &g in px {
                out.extend_from_slice(&[g, g, g, 255]);
            }
            out
        }
        Format::R8G8 => {
            let mut out = Vec::with_capacity(n);
            for c in px.chunks_exact(2) {
                out.extend_from_slice(&[c[0], c[0], c[0], c[1]]);
            }
            out
        }
        _ => return None,
    };
    Some(TextureData {
        pixels,
        width,
        height,
    })
}

fn walk_node(
    node: &gltf::Node,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    seen: &mut HashMap<(usize, usize), usize>,
    out: &mut SceneData,
) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        add_mesh_nodes(&mesh, world, buffers, images, seen, out);
    }
    for child in node.children() {
        walk_node(&child, world, buffers, images, seen, out);
    }
}

/// Emit one [`SceneNode`] per triangle primitive of `mesh` at `world`, registering
/// each primitive's geometry once (subsequent references reuse the same index, so
/// repeated props instance rather than duplicating vertices).
fn add_mesh_nodes(
    mesh: &gltf::Mesh,
    world: Mat4,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    seen: &mut HashMap<(usize, usize), usize>,
    out: &mut SceneData,
) {
    for prim in mesh.primitives() {
        if prim.mode() != gltf::mesh::Mode::Triangles {
            continue;
        }
        let key = (mesh.index(), prim.index());
        let slot = match seen.get(&key) {
            Some(&i) => i,
            None => {
                let Some(data) = primitive_mesh(&prim, buffers, images) else {
                    continue; // no positions
                };
                out.meshes.push(data);
                let i = out.meshes.len() - 1;
                seen.insert(key, i);
                i
            }
        };
        out.nodes.push(SceneNode {
            mesh: slot,
            transform: world,
        });
    }
}

/// Read one primitive into a local-space `MeshData` with its own material.
fn primitive_mesh(
    prim: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
) -> Option<MeshData> {
    let reader = prim.reader(|b| Some(&buffers[b.index()][..]));
    let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();
    let indices: Vec<u32> = match reader.read_indices() {
        Some(idx) => idx.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };
    let normals: Vec<[f32; 3]> = match reader.read_normals() {
        Some(n) => n.collect(),
        None => compute_normals(&positions, &indices),
    };
    let uvs: Option<Vec<[f32; 2]>> = reader.read_tex_coords(0).map(|t| t.into_f32().collect());

    let vertices = positions
        .iter()
        .enumerate()
        .map(|(i, p)| Vertex {
            pos: *p,
            normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
            uv: uvs
                .as_ref()
                .and_then(|u| u.get(i).copied())
                .unwrap_or([0.0, 0.0]),
        })
        .collect();

    Some(MeshData {
        vertices,
        indices,
        material: read_material(&prim.material(), images),
    })
}

/// Area-weighted per-vertex normals from a triangle list (used when a primitive
/// ships no normals).
fn compute_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if ia >= positions.len() || ib >= positions.len() || ic >= positions.len() {
            continue;
        }
        let a = Vec3::from(positions[ia]);
        let b = Vec3::from(positions[ib]);
        let c = Vec3::from(positions[ic]);
        let face = (b - a).cross(c - a); // length proportional to triangle area
        acc[ia] += face;
        acc[ib] += face;
        acc[ic] += face;
    }
    acc.iter().map(|v| v.normalize_or_zero().to_array()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal glTF fixture: one triangle mesh referenced by three nodes, one of
    /// which is a child, so a single file covers per-primitive meshes, mesh
    /// deduplication and transform inheritance. Written to disk rather than
    /// committed as a binary, since `scratch/` is gitignored.
    fn write_fixture(dir: &std::path::Path) -> std::path::PathBuf {
        let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut bin = Vec::new();
        for f in positions {
            bin.extend_from_slice(&f.to_le_bytes());
        }
        std::fs::write(dir.join("tri.bin"), &bin).unwrap();

        // node 1 parents node 2, so node 2's world translation is (0, 2, 3).
        let gltf = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [ { "nodes": [0, 1] } ],
  "nodes": [
    { "mesh": 0, "translation": [1, 0, 0] },
    { "mesh": 0, "translation": [0, 2, 0], "children": [2] },
    { "mesh": 0, "translation": [0, 0, 3] }
  ],
  "meshes": [ { "primitives": [ { "attributes": { "POSITION": 0 } } ] } ],
  "accessors": [ {
    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
    "min": [0, 0, 0], "max": [1, 1, 0]
  } ],
  "bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 36 } ],
  "buffers": [ { "byteLength": 36, "uri": "tri.bin" } ]
}"#;
        let path = dir.join("tri.gltf");
        std::fs::write(&path, gltf).unwrap();
        path
    }

    fn fixture_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("feather_gltf_{}_{}", name, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn scene_dedups_meshes_and_accumulates_transforms() {
        let dir = fixture_dir("scene");
        let scene = load_gltf_scene(write_fixture(&dir)).unwrap();

        // Three nodes reference ONE primitive: it must be stored once and placed
        // three times, which is what lets the renderer instance it.
        assert_eq!(scene.meshes.len(), 1, "primitive should be deduplicated");
        assert_eq!(scene.nodes.len(), 3, "one placement per referencing node");
        assert!(scene.nodes.iter().all(|n| n.mesh == 0));

        let origins: Vec<Vec3> = scene
            .nodes
            .iter()
            .map(|n| n.transform.transform_point3(Vec3::ZERO))
            .collect();
        assert!(origins.contains(&Vec3::new(1.0, 0.0, 0.0)));
        assert!(origins.contains(&Vec3::new(0.0, 2.0, 0.0)));
        // The child's transform is its parent's composed with its own.
        assert!(
            origins.contains(&Vec3::new(0.0, 2.0, 3.0)),
            "child should inherit the parent transform, got {origins:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn scene_meshes_stay_in_local_space() {
        let dir = fixture_dir("local");
        let scene = load_gltf_scene(write_fixture(&dir)).unwrap();

        // Geometry must NOT have the node transforms baked in — that is precisely
        // what would break instancing of a mesh used at several places.
        let verts: Vec<[f32; 3]> = scene.meshes[0].vertices.iter().map(|v| v.pos).collect();
        assert_eq!(
            verts,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
        // Indices are generated when the primitive ships none.
        assert_eq!(scene.meshes[0].indices, vec![0, 1, 2]);

        std::fs::remove_dir_all(&dir).ok();
    }
}
