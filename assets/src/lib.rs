//! Assets: runtime blob loaders, handle tables, bake format definitions.
//!
//! For now this owns the CPU-side mesh representation shared with the renderer
//! (`Vertex` / `MeshData`), a procedural sphere, and a minimal glTF mesh loader.
//! Everything here is Vulkan-free by design (§22) — the renderer uploads these
//! buffers to the GPU.

pub mod bake;

use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use glam::{Mat4, Vec3};

/// Interleaved static vertex. Matches the renderer's vertex attribute layout
/// (position at offset 0, normal at offset 12, uv at offset 24). `#[repr(C)]` so
/// a slice uploads straight into a device-local vertex buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// A texture image, held **encoded** (the PNG/JPEG bytes as stored) and decoded
/// to RGBA8 only on first use (§17). A baked texture is keyed on `source_key`
/// and uploaded straight from the bake, so its pixels are never decoded.
/// Materials hold it by `Arc`: one image used by many materials is decoded and
/// uploaded once, and the renderer dedups by the `Arc`'s identity.
#[derive(Clone)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    /// xxh3 of the encoded bytes (or of the pixels, for `from_rgba8`): the
    /// content half of the bake key.
    pub source_key: u64,
    encoded: Vec<u8>,
    decoded: OnceLock<Option<Vec<u8>>>,
}

impl std::fmt::Debug for TextureData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextureData")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("source_key", &format_args!("{:016x}", self.source_key))
            .field("decoded", &self.is_decoded())
            .finish()
    }
}

impl TextureData {
    /// From encoded image bytes (PNG or JPEG). Reads only the header, for the
    /// size; `None` if it isn't a readable image.
    pub fn from_encoded(encoded: Vec<u8>) -> Option<Self> {
        let (width, height) = image::ImageReader::new(std::io::Cursor::new(&encoded))
            .with_guessed_format()
            .ok()?
            .into_dimensions()
            .ok()?;
        Some(Self {
            width,
            height,
            source_key: xxhash_rust::xxh3::xxh3_64(&encoded),
            encoded,
            decoded: OnceLock::new(),
        })
    }

    /// From pixels already decoded to RGBA8 (procedural or test textures),
    /// keyed on those pixels.
    pub fn from_rgba8(pixels: Vec<u8>, width: u32, height: u32) -> Self {
        assert_eq!(pixels.len(), (width * height * 4) as usize, "RGBA8 size");
        let mut h = xxhash_rust::xxh3::Xxh3::new();
        h.update(&width.to_le_bytes());
        h.update(&height.to_le_bytes());
        h.update(&pixels);
        Self {
            width,
            height,
            source_key: h.digest(),
            encoded: Vec::new(),
            decoded: OnceLock::from(Some(pixels)),
        }
    }

    /// RGBA8, row-major, `width * height * 4` bytes; decoded on the first
    /// call. `None` if the bytes don't decode (or decode to another size than
    /// the header said): callers treat it like a missing texture.
    pub fn pixels(&self) -> Option<&[u8]> {
        self.decoded
            .get_or_init(|| {
                let rgba = image::load_from_memory(&self.encoded).ok()?.to_rgba8();
                (rgba.dimensions() == (self.width, self.height)).then(|| rgba.into_raw())
            })
            .as_deref()
    }

    /// Whether `pixels` has run (or the texture was made from pixels).
    pub fn is_decoded(&self) -> bool {
        self.decoded.get().is_some()
    }
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
    pub base_color_texture: Option<Arc<TextureData>>,
    pub normal_texture: Option<Arc<TextureData>>,
    pub metallic_roughness_texture: Option<Arc<TextureData>>,
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

/// The gameplay half of a node (§18): which prefab to spawn, plus its
/// parameters, read from the glTF node's `extras`.
///
/// Params stay as raw JSON with typed accessors rather than a fixed struct, so
/// that a new prefab is a new spawn function rather than a format change —
/// which is §18's whole point.
#[derive(Clone, Debug)]
pub struct PrefabSpec {
    pub id: String,
    pub params: serde_json::Value,
}

impl PrefabSpec {
    pub fn f32(&self, key: &str) -> Option<f32> {
        self.params.get(key)?.as_f64().map(|v| v as f32)
    }

    pub fn bool(&self, key: &str) -> Option<bool> {
        self.params.get(key)?.as_bool()
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        self.params.get(key)?.as_str()
    }

    pub fn vec3(&self, key: &str) -> Option<Vec3> {
        let a = self.params.get(key)?.as_array()?;
        if a.len() < 3 {
            return None;
        }
        let c: Vec<f32> = a
            .iter()
            .filter_map(|v| v.as_f64())
            .map(|v| v as f32)
            .collect();
        (c.len() == 3).then(|| Vec3::new(c[0], c[1], c[2]))
    }
}

/// One placement in a loaded scene: what to draw and where, plus any prefab the
/// node carries. Several nodes referencing the same mesh is the common case
/// (props repeated across a level), and is exactly what the renderer's instanced
/// path wants.
pub struct SceneNode {
    /// Index into [`SceneData::meshes`], or `None` for a **marker**: a node with
    /// no geometry that exists only to place a prefab (a spawn point, say).
    pub mesh: Option<usize>,
    /// The node's world transform, with the whole parent chain applied.
    pub transform: Mat4,
    /// §18 gameplay data from the node's `extras`, when it has any.
    pub prefab: Option<PrefabSpec>,
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
/// Nodes carrying §18 `extras` gain a [`PrefabSpec`]; a node with a prefab but no
/// mesh is emitted as a **marker** (`mesh: None`).
///
/// Deferred: tangents, skinning, animation, morph targets, non-triangle
/// primitives, and 16-/32-bit image formats.
pub fn load_gltf_scene(path: impl AsRef<Path>) -> Result<SceneData, Box<dyn Error>> {
    // Buffers (blob / external / data URI) are resolved here; images are only
    // read as encoded bytes, and decoded later if and when something needs
    // their pixels (§17).
    let path = path.as_ref();
    let gltf::Gltf {
        document: doc,
        blob,
    } = gltf::Gltf::open(path)?;
    let base = path.parent().unwrap_or(Path::new(""));
    let buffers = gltf::import_buffers(&doc, Some(base), blob)?;
    let images = ImageCache {
        doc: &doc,
        buffers: &buffers,
        base,
        by_index: RefCell::new(HashMap::new()),
        by_source: RefCell::new(HashMap::new()),
    };

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
                    // No scene graph means no nodes, so no prefab data either.
                    None,
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

/// A scene's glTF images, each read **at most once** and shared by every
/// material that uses it. Before sharing, every primitive converted its
/// material's textures afresh: 12 images became 81 conversions (13.5 s of a
/// debug load) and 81 uploads.
struct ImageCache<'a> {
    doc: &'a gltf::Document,
    buffers: &'a [gltf::buffer::Data],
    base: &'a Path,
    by_index: RefCell<HashMap<usize, Option<Arc<TextureData>>>>,
    /// Two images with identical bytes share one texture too.
    by_source: RefCell<HashMap<u64, Arc<TextureData>>>,
}

impl ImageCache<'_> {
    /// Keyed by glTF *image*, not texture, so two textures sampling one image
    /// also share it. An unreadable image warns and yields `None`: the
    /// material falls back to its factors.
    fn get(&self, index: usize) -> Option<Arc<TextureData>> {
        self.by_index
            .borrow_mut()
            .entry(index)
            .or_insert_with(|| {
                let image = self.doc.images().nth(index)?;
                let tex = encoded_image(&image, self.base, self.buffers)
                    .and_then(TextureData::from_encoded);
                let Some(tex) = tex else {
                    eprintln!("[scene] image {index} isn't a readable PNG/JPEG; using the material's factors");
                    return None;
                };
                let shared = self
                    .by_source
                    .borrow_mut()
                    .entry(tex.source_key)
                    .or_insert_with(|| Arc::new(tex))
                    .clone();
                Some(shared)
            })
            .clone()
    }
}

/// An image's encoded bytes, wherever the glTF keeps them: a buffer view, a
/// `data:` URI or a file next to the glTF.
fn encoded_image(
    image: &gltf::Image,
    base: &Path,
    buffers: &[gltf::buffer::Data],
) -> Option<Vec<u8>> {
    match image.source() {
        gltf::image::Source::View { view, .. } => {
            let data = &buffers.get(view.buffer().index())?.0;
            data.get(view.offset()..view.offset() + view.length())
                .map(<[u8]>::to_vec)
        }
        gltf::image::Source::Uri { uri, .. } => {
            if let Some(data) = uri.strip_prefix("data:") {
                let (_, b64) = data.split_once(";base64,")?;
                base64::decode(b64).ok()
            } else {
                let rel = uri
                    .strip_prefix("file://")
                    .or_else(|| uri.strip_prefix("file:"))
                    .unwrap_or(uri);
                let rel = urlencoding::decode(rel).ok()?;
                std::fs::read(base.join(&*rel)).ok()
            }
        }
    }
}

fn read_material(m: &gltf::Material, images: &ImageCache) -> Material {
    let pbr = m.pbr_metallic_roughness();
    let tex = |idx: usize| images.get(idx);
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

/// Read a node's §18 prefab data from its `extras`.
///
/// **Lenient on purpose**: anything unparseable warns and is dropped rather than
/// failing the import. An authoring typo should cost you one prop, not the whole
/// level — and `extras` is a free-form escape hatch that other tools write into,
/// so foreign shapes are expected rather than exceptional.
fn read_prefab(node: &gltf::Node) -> Option<PrefabSpec> {
    let raw = node.extras().as_ref()?;
    let value: serde_json::Value = match serde_json::from_str(raw.get()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "node {}: extras is not valid JSON ({e}); ignored",
                node_label(node)
            );
            return None;
        }
    };
    // No `prefab` key is not an error: extras is shared with other tooling.
    let id = value.get("prefab")?;
    let Some(id) = id.as_str() else {
        eprintln!(
            "node {}: extras.prefab is not a string; ignored",
            node_label(node)
        );
        return None;
    };
    Some(PrefabSpec {
        id: id.to_string(),
        params: value
            .get("params")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    })
}

fn node_label(node: &gltf::Node) -> String {
    match node.name() {
        Some(n) => format!("{} (#{})", n, node.index()),
        None => format!("#{}", node.index()),
    }
}

fn walk_node(
    node: &gltf::Node,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    images: &ImageCache,
    seen: &mut HashMap<(usize, usize), usize>,
    out: &mut SceneData,
) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    let prefab = read_prefab(node);
    match node.mesh() {
        Some(mesh) => add_mesh_nodes(&mesh, world, prefab, buffers, images, seen, out),
        // A node with no geometry but a prefab is a marker — the spawn points and
        // trigger volumes §18 wants. Without a prefab it is just a transform in
        // the hierarchy, already applied to its children, so nothing to emit.
        None => {
            if prefab.is_some() {
                out.nodes.push(SceneNode {
                    mesh: None,
                    transform: world,
                    prefab,
                });
            }
        }
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
    prefab: Option<PrefabSpec>,
    buffers: &[gltf::buffer::Data],
    images: &ImageCache,
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
        // Every primitive of the node inherits its prefab: one glTF node with a
        // multi-primitive mesh is still one *thing*.
        out.nodes.push(SceneNode {
            mesh: Some(slot),
            transform: world,
            prefab: prefab.clone(),
        });
    }
}

/// Read one primitive into a local-space `MeshData` with its own material.
fn primitive_mesh(
    prim: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    images: &ImageCache,
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

    /// Fixture for §18 `extras`: a mesh node with a prefab, a **mesh-less
    /// marker**, a node whose extras are the wrong shape, and a plain node with
    /// none — so one file covers every branch of `read_prefab`.
    fn write_prefab_fixture(dir: &std::path::Path) -> std::path::PathBuf {
        let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut bin = Vec::new();
        for f in positions {
            bin.extend_from_slice(&f.to_le_bytes());
        }
        std::fs::write(dir.join("tri.bin"), &bin).unwrap();

        let gltf = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [ { "nodes": [0, 1, 2, 3, 4] } ],
  "nodes": [
    { "mesh": 0, "translation": [1, 0, 0],
      "extras": { "prefab": "no_collide", "params": { "flag": true } } },
    { "translation": [5, 6, 7],
      "extras": { "prefab": "player_start", "params": { "yaw": 90.0 } } },
    { "mesh": 0, "translation": [2, 0, 0] },
    { "mesh": 0, "translation": [3, 0, 0], "extras": { "prefab": 42 } },
    { "translation": [9, 9, 9] }
  ],
  "meshes": [ { "primitives": [ { "attributes": { "POSITION": 0 } } ] } ],
  "accessors": [ {
    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
    "min": [0, 0, 0], "max": [1, 1, 0]
  } ],
  "bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 36 } ],
  "buffers": [ { "byteLength": 36, "uri": "tri.bin" } ]
}"#;
        let path = dir.join("prefabs.gltf");
        std::fs::write(&path, gltf).unwrap();
        path
    }

    #[test]
    fn extras_become_prefabs_and_markers() {
        let dir = fixture_dir("prefabs");
        let scene = load_gltf_scene(write_prefab_fixture(&dir)).unwrap();

        // 3 mesh nodes + 1 marker. The plain mesh-less node emits nothing: it is
        // only a transform, already folded into any children.
        assert_eq!(scene.nodes.len(), 4, "got {:?}", scene.nodes.len());

        let marker = scene
            .nodes
            .iter()
            .find(|n| n.mesh.is_none())
            .expect("mesh-less marker should be emitted");
        let spec = marker.prefab.as_ref().expect("marker carries a prefab");
        assert_eq!(spec.id, "player_start");
        assert_eq!(spec.f32("yaw"), Some(90.0));
        assert_eq!(
            marker.transform.transform_point3(Vec3::ZERO),
            Vec3::new(5.0, 6.0, 7.0)
        );

        let tagged = scene
            .nodes
            .iter()
            .find(|n| n.prefab.as_ref().is_some_and(|p| p.id == "no_collide"))
            .expect("mesh node keeps its prefab");
        assert_eq!(tagged.mesh, Some(0));
        assert_eq!(tagged.prefab.as_ref().unwrap().bool("flag"), Some(true));

        // Exactly one prefab-less mesh node, and the malformed one (prefab: 42)
        // must be among them — dropped, not fatal, and not silently accepted.
        let plain = scene
            .nodes
            .iter()
            .filter(|n| n.mesh.is_some() && n.prefab.is_none())
            .count();
        assert_eq!(plain, 2, "malformed extras should be ignored, not fatal");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn prefab_params_are_typed_and_forgiving() {
        let spec = PrefabSpec {
            id: "x".into(),
            params: serde_json::json!({ "yaw": 45, "on": false, "c": [1.0, 2.0, 3.0] }),
        };
        assert_eq!(spec.f32("yaw"), Some(45.0));
        assert_eq!(spec.bool("on"), Some(false));
        assert_eq!(spec.vec3("c"), Some(Vec3::new(1.0, 2.0, 3.0)));
        // Wrong type or missing key yields None rather than panicking.
        assert_eq!(spec.f32("on"), None);
        assert_eq!(spec.vec3("yaw"), None);
        assert_eq!(spec.f32("nope"), None);
    }

    fn fixture_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("feather_gltf_{}_{}", name, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn materials_sharing_an_image_share_one_conversion() {
        // Three primitives, three materials. Materials 0 and 1 use two
        // different glTF *textures* that both sample image 0; material 2
        // uses image 1. Each image must be converted once and shared.
        let dir = fixture_dir("shared_image");
        let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let bin: Vec<u8> = positions.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(dir.join("tri.bin"), &bin).unwrap();
        let red = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";
        let blue = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYPj/HwADAgH/5ncLrgAAAABJRU5ErkJggg==";
        let gltf = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [ {{ "nodes": [0] }} ],
  "nodes": [ {{ "mesh": 0 }} ],
  "meshes": [ {{ "primitives": [
    {{ "attributes": {{ "POSITION": 0 }}, "material": 0 }},
    {{ "attributes": {{ "POSITION": 0 }}, "material": 1 }},
    {{ "attributes": {{ "POSITION": 0 }}, "material": 2 }}
  ] }} ],
  "materials": [
    {{ "pbrMetallicRoughness": {{ "baseColorTexture": {{ "index": 0 }} }} }},
    {{ "pbrMetallicRoughness": {{ "baseColorTexture": {{ "index": 1 }} }} }},
    {{ "pbrMetallicRoughness": {{ "baseColorTexture": {{ "index": 2 }} }} }}
  ],
  "textures": [ {{ "source": 0 }}, {{ "source": 0 }}, {{ "source": 1 }} ],
  "images": [
    {{ "uri": "data:image/png;base64,{red}" }},
    {{ "uri": "data:image/png;base64,{blue}" }}
  ],
  "accessors": [ {{
    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
    "min": [0, 0, 0], "max": [1, 1, 0]
  }} ],
  "bufferViews": [ {{ "buffer": 0, "byteOffset": 0, "byteLength": 36 }} ],
  "buffers": [ {{ "byteLength": 36, "uri": "tri.bin" }} ]
}}"#
        );
        let path = dir.join("shared.gltf");
        std::fs::write(&path, gltf).unwrap();
        let scene = load_gltf_scene(&path).unwrap();
        let tex: Vec<Arc<TextureData>> = scene
            .meshes
            .iter()
            .map(|m| m.material.base_color_texture.clone().expect("textured"))
            .collect();
        assert_eq!(tex.len(), 3);
        assert!(Arc::ptr_eq(&tex[0], &tex[1]), "one image, one conversion");
        assert!(
            !Arc::ptr_eq(&tex[0], &tex[2]),
            "distinct images stay distinct"
        );
        // Nothing decodes until asked: this is what lets a baked texture skip
        // its JPEG decode entirely.
        assert!(tex.iter().all(|t| !t.is_decoded()));
        assert_eq!((tex[0].width, tex[0].height), (1, 1));
        // 1x1 RGB PNGs, expanded to opaque RGBA as before.
        assert_eq!(tex[0].pixels(), Some(&[255, 0, 0, 255][..]));
        assert_eq!(tex[2].pixels(), Some(&[0, 0, 255, 255][..]));
        assert!(tex[0].is_decoded() && !tex[2].encoded.is_empty());
        assert_ne!(tex[0].source_key, tex[2].source_key);
    }

    #[test]
    fn scene_dedups_meshes_and_accumulates_transforms() {
        let dir = fixture_dir("scene");
        let scene = load_gltf_scene(write_fixture(&dir)).unwrap();

        // Three nodes reference ONE primitive: it must be stored once and placed
        // three times, which is what lets the renderer instance it.
        assert_eq!(scene.meshes.len(), 1, "primitive should be deduplicated");
        assert_eq!(scene.nodes.len(), 3, "one placement per referencing node");
        assert!(scene.nodes.iter().all(|n| n.mesh == Some(0)));

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
