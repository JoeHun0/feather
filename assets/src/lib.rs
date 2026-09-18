//! Assets: runtime blob loaders, handle tables, bake format definitions.
//!
//! For now this owns the CPU-side mesh representation shared with the renderer
//! (`Vertex` / `MeshData`), a procedural sphere, and a minimal glTF mesh loader.
//! Everything here is Vulkan-free by design (§22) — the renderer uploads these
//! buffers to the GPU.

use std::error::Error;
use std::path::Path;

use glam::{Mat3, Mat4, Vec3};

/// Interleaved static vertex. Matches the renderer's vertex attribute layout
/// (position at offset 0, normal at offset 12). `#[repr(C)]` so a slice uploads
/// straight into a device-local vertex buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
}

/// CPU-side mesh: interleaved vertices + a 32-bit index buffer. One shared
/// representation for both procedural and loaded geometry.
#[derive(Clone, Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
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
        Self { vertices, indices }
    }
}

/// Load a glTF / GLB file and merge every mesh primitive (across the node
/// hierarchy, node transforms applied) into a single `MeshData`.
///
/// Handled: positions (required), normals (computed if absent), indices
/// (generated if absent), node transforms, triangle primitives, the `.glb`
/// binary blob, and external `.bin` buffers.
///
/// Deferred: materials, textures, UVs, tangents, skinning, animation, morph
/// targets, non-triangle primitives, and `data:` URI buffers (export as `.glb`
/// or with an external `.bin` for now).
pub fn load_gltf(path: impl AsRef<Path>) -> Result<MeshData, Box<dyn Error>> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let gltf = gltf::Gltf::from_slice(&bytes)?;
    let blob = gltf.blob.clone();

    // Resolve every buffer declared by the document to raw bytes.
    let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(gltf.buffers().count());
    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {
                let b = blob
                    .as_ref()
                    .ok_or("glTF references the GLB binary blob but none is present")?;
                buffers.push(b.clone());
            }
            gltf::buffer::Source::Uri(uri) => {
                if uri.starts_with("data:") {
                    return Err("data: URI buffers are not supported yet; \
                                export as .glb or with an external .bin"
                        .into());
                }
                let dir = path.parent().unwrap_or_else(|| Path::new("."));
                buffers.push(std::fs::read(dir.join(uri))?);
            }
        }
    }

    let mut out = MeshData::default();
    match gltf.default_scene().or_else(|| gltf.scenes().next()) {
        Some(scene) => {
            for node in scene.nodes() {
                accumulate_node(&node, Mat4::IDENTITY, &buffers, &mut out);
            }
        }
        // No scene graph: take mesh geometry directly, untransformed.
        None => {
            for mesh in gltf.meshes() {
                append_mesh(&mesh, Mat4::IDENTITY, &buffers, &mut out);
            }
        }
    }

    if out.vertices.is_empty() {
        return Err("glTF contained no triangle mesh geometry".into());
    }
    Ok(out)
}

fn accumulate_node(node: &gltf::Node, parent: Mat4, buffers: &[Vec<u8>], out: &mut MeshData) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        append_mesh(&mesh, world, buffers, out);
    }
    for child in node.children() {
        accumulate_node(&child, world, buffers, out);
    }
}

fn append_mesh(mesh: &gltf::Mesh, world: Mat4, buffers: &[Vec<u8>], out: &mut MeshData) {
    // Normals transform by the inverse-transpose (correct under non-uniform scale).
    let normal_mat = Mat3::from_mat4(world).inverse().transpose();
    for prim in mesh.primitives() {
        if prim.mode() != gltf::mesh::Mode::Triangles {
            continue;
        }
        let reader = prim.reader(|b| buffers.get(b.index()).map(|v| v.as_slice()));
        let positions: Vec<[f32; 3]> = match reader.read_positions() {
            Some(p) => p.collect(),
            None => continue,
        };
        let indices: Vec<u32> = match reader.read_indices() {
            Some(idx) => idx.into_u32().collect(),
            None => (0..positions.len() as u32).collect(),
        };
        let normals: Vec<[f32; 3]> = match reader.read_normals() {
            Some(n) => n.collect(),
            None => compute_normals(&positions, &indices),
        };

        let base = out.vertices.len() as u32;
        for (i, p) in positions.iter().enumerate() {
            let wp = world.transform_point3(Vec3::from(*p));
            let n = normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
            let wn = (normal_mat * Vec3::from(n)).normalize_or_zero();
            out.vertices.push(Vertex {
                pos: wp.to_array(),
                normal: wn.to_array(),
            });
        }
        out.indices.extend(indices.iter().map(|i| base + *i));
    }
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
