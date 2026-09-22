//! Instanced, lit mesh renderer. Several meshes live in one shared vertex +
//! index buffer (each a vertex offset plus one index range per LOD, §17);
//! per-frame instances get a LOD per view and are sorted by (mesh, LOD) so each
//! pair draws as one contiguous run (`cmd_draw_indexed` with `firstInstance`). Per-instance data is
//! `{ model, material_id }`; the fragment shader reads the material from a
//! resident materials SSBO and applies a directional light. Meshes and materials
//! come in as Vulkan-free `assets` types.

use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

use feather_assets::bake::{
    baked_mesh_path, baked_path, mesh_key, texture_key, BakedMesh, BakedTexture, TexKind, MESH_DIR,
    MIN_LOD_TRIS, TEX_DIR,
};
use feather_assets::{Material, MeshData, TextureData, Vertex};
use feather_gfx::{Buffer, Image, MappedBuffer, Renderer, FRAMES_IN_FLIGHT, SHADOW_CASCADES};
use glam::{Mat4, Vec3, Vec4};

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

/// Cap on punctual lights visible in one frame (§12). Clamped rather than
/// grown: the buffer is sized once, and silently dropping past the cap is
/// better than a resize mid-frame.
pub const MAX_LIGHTS: usize = 128;
/// Light-cluster grid (§12): screen tiles × exponential depth slices. Must
/// match `cluster.comp` and `mesh.frag` (a test checks).
const CLUSTER_X: usize = 16;
const CLUSTER_Y: usize = 9;
const CLUSTER_Z: usize = 24;
const CLUSTER_COUNT: usize = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
/// Each cluster is a bitmask over the frame's light list, one bit per light.
/// With `MAX_LIGHTS` capped this is exact — no per-cluster overflow to clamp —
/// and small: 4 words × 3456 clusters = 55 KB per frame.
const CLUSTER_WORDS: usize = MAX_LIGHTS / 32;
/// Threads per cluster-assignment workgroup; one thread per cluster.
const CLUSTER_GROUP: usize = 64;

/// The camera as the light clusters need it. Must describe the same projection
/// the geometry pass renders with, or fragments look up the wrong cluster.
#[derive(Clone, Copy)]
pub struct ClusterView {
    /// World → view (right-handed, looking down -Z).
    pub view: Mat4,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    /// Width / height of the projection.
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    /// Viewport height in pixels, for projecting LOD error to the screen (§17).
    pub viewport_height: f32,
}

/// Where each material's textures live in the bindless array (§9).
struct TexturePlan {
    /// Unique (image, kind) pairs to upload, in slot order from
    /// `FIRST_TEXTURE_SLOT`.
    uploads: Vec<(Arc<TextureData>, TexKind)>,
    /// Per material: base colour, normal, metallic-roughness slots.
    slots: Vec<[u32; 3]>,
    /// Texture references across all materials, before dedup.
    references: usize,
    /// References that fell back to a default because the array was full.
    overflow: usize,
}

/// Bytes of an RGBA8 texture with its full mip chain (what the raw path uploads).
fn rgba8_chain_bytes(width: u32, height: u32) -> usize {
    let (mut w, mut h, mut total) = (width.max(1), height.max(1), 0usize);
    loop {
        total += (w * h * 4) as usize;
        if w == 1 && h == 1 {
            return total;
        }
        (w, h) = ((w / 2).max(1), (h / 2).max(1));
    }
}

/// Slots 0 (white) and 1 (flat normal) are the resident defaults.
const FIRST_TEXTURE_SLOT: usize = 2;

/// Assign bindless slots, one per **unique image × kind**. The key is the
/// `Arc`'s identity (the loader shares one per image) plus the kind, since the
/// same pixels sampled as base colour (sRGB) and as data (UNORM: normal and
/// metallic-roughness maps) need two image views. Pure, so it is testable
/// without a GPU.
///
/// `capacity` is the most slots the array may have: the device's limit, not a
/// fixed constant (§9). References past it fall back to the defaults.
fn plan_texture_slots(materials: &[Material], capacity: usize) -> TexturePlan {
    let mut uploads: Vec<(Arc<TextureData>, TexKind)> = Vec::new();
    let mut index: HashMap<(*const TextureData, TexKind), u32> = HashMap::new();
    let (mut references, mut overflow) = (0, 0);
    let mut slot = |tex: &Option<Arc<TextureData>>, kind: TexKind, default: u32| {
        let Some(t) = tex else {
            return default;
        };
        references += 1;
        let key = (Arc::as_ptr(t), kind);
        if let Some(&s) = index.get(&key) {
            return s;
        }
        if FIRST_TEXTURE_SLOT + uploads.len() >= capacity {
            overflow += 1;
            return default;
        }
        let s = (FIRST_TEXTURE_SLOT + uploads.len()) as u32;
        uploads.push((t.clone(), kind));
        index.insert(key, s);
        s
    };
    let slots = materials
        .iter()
        .map(|m| {
            [
                slot(&m.base_color_texture, TexKind::Color, 0),
                slot(&m.normal_texture, TexKind::Data, 1),
                slot(&m.metallic_roughness_texture, TexKind::Data, 0),
            ]
        })
        .collect();
    TexturePlan {
        uploads,
        slots,
        references,
        overflow,
    }
}

/// Handle to a mesh registered with the renderer, in `new`'s input order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MeshId(pub u32);

/// One level of detail: a range of the shared index buffer.
#[derive(Clone, Copy, Debug)]
struct LodSlice {
    first_index: u32,
    index_count: u32,
    /// Geometric error against LOD0 in mesh-local units (0 for LOD0);
    /// non-decreasing along a mesh's chain.
    error: f32,
}

/// Where one mesh lives inside the shared vertex/index buffers. `vertex_offset`
/// is added to each (mesh-local) index at draw time, so per-mesh indices stay
/// 0-based. Every LOD indexes the same vertices. An unbaked mesh has one LOD.
#[derive(Clone, Debug)]
struct MeshSlice {
    vertex_offset: i32,
    lods: Vec<LodSlice>,
    /// Local bounding sphere, for projecting LOD error.
    center: Vec3,
    radius: f32,
}

/// Most LODs a mesh keeps; they're packed into 4 bits of the run sort key.
const MAX_LODS: usize = 16;
/// Largest LOD error allowed on screen, in pixels. Below one pixel a coarser
/// level is indistinguishable from the full mesh, up to shading.
const LOD_PIXEL_ERROR: f32 = 1.0;

/// How a view turns an instance into an acceptable LOD error (§17).
#[derive(Clone, Copy, Debug)]
enum LodRule {
    /// Perspective camera: error projected to pixels at the instance's
    /// nearest point.
    Screen {
        eye: Vec3,
        /// Pixels per world unit at distance 1: `height / (2 tan(fov/2))`.
        px_per_unit: f32,
        near: f32,
    },
    /// Shadow cascade: detail finer than one shadow texel can't show, at any
    /// distance (the ortho has no perspective).
    Texel(f32),
}

impl LodRule {
    /// The largest mesh-local error acceptable for an instance with `model`
    /// and local bounding sphere (`center`, `radius`).
    fn budget(self, model: &Mat4, center: Vec3, radius: f32) -> f32 {
        // World units per mesh unit: the largest axis scale, so the error is
        // never under-estimated on a stretched instance.
        let scale = model
            .x_axis
            .truncate()
            .length()
            .max(model.y_axis.truncate().length())
            .max(model.z_axis.truncate().length())
            .max(1e-12);
        match self {
            LodRule::Screen {
                eye,
                px_per_unit,
                near,
            } => {
                let centre = model.transform_point3(center);
                let distance = (centre.distance(eye) - radius * scale).max(near);
                LOD_PIXEL_ERROR * distance / (px_per_unit * scale)
            }
            LodRule::Texel(texel) => texel / scale,
        }
    }
}

/// The coarsest LOD whose error fits `budget` (LOD0 always does).
fn pick_lod(lods: &[LodSlice], budget: f32) -> usize {
    lods.iter().rposition(|l| l.error <= budget).unwrap_or(0)
}

/// Bounding sphere of a vertex set: the AABB centre and the farthest vertex.
fn bounding_sphere(vertices: &[Vertex]) -> (Vec3, f32) {
    let Some(first) = vertices.first() else {
        return (Vec3::ZERO, 0.0);
    };
    let (lo, hi) = vertices.iter().fold(
        (Vec3::from(first.pos), Vec3::from(first.pos)),
        |(lo, hi), v| (lo.min(Vec3::from(v.pos)), hi.max(Vec3::from(v.pos))),
    );
    let center = (lo + hi) * 0.5;
    let radius = vertices
        .iter()
        .map(|v| Vec3::from(v.pos).distance(center))
        .fold(0.0, f32::max);
    (center, radius)
}

/// Triangles and LOD choices of one frame, for `--bench` (§17). Triangles
/// are counted per instance submitted, before any GPU culling.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameStats {
    pub main_tris: u64,
    pub shadow_tris: u64,
    /// Instances drawn at each LOD, camera view.
    pub main_lods: [u32; MAX_LODS],
    /// Instances drawn at each LOD, summed over the shadow cascades.
    pub shadow_lods: [u32; MAX_LODS],
}

/// GPU material record (§5): std430, 64 bytes, indexed by `material_id`.
/// `tex.x` = base-color texture slot into the bindless array (0 = white).
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuMaterial {
    base_color_factor: [f32; 4],
    emissive: [f32; 4], // rgb = emissive, a = metallic
    params: [f32; 4],   // x = roughness, y = normal_scale, z = occlusion, w = alpha_cutoff
    tex: [u32; 4],      // x = base color, y = normal, z = metallic-roughness, w reserved
}

impl GpuMaterial {
    fn from_material(m: &Material, slots: [u32; 3]) -> Self {
        Self {
            base_color_factor: m.base_color,
            emissive: [m.emissive[0], m.emissive[1], m.emissive[2], m.metallic],
            params: [m.roughness, m.normal_scale, 1.0, 0.5],
            tex: [slots[0], slots[1], slots[2], 0],
        }
    }
}

/// Per-instance data uploaded to the storage buffer (§6: std430, 80 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct InstanceData {
    pub model: Mat4,
    pub material_id: u32,
    _pad: [u32; 3],
}

impl InstanceData {
    pub fn new(model: Mat4, material_id: u32) -> Self {
        Self {
            model,
            material_id,
            _pad: [0; 3],
        }
    }
}

const INSTANCE_SIZE: u64 = std::mem::size_of::<InstanceData>() as u64; // 80

/// Per-frame globals UBO (§9 groundwork). Carries the sun's light-space matrix
/// and shadow sampling params — the mat4 won't fit the already-96 B push constant.
/// Bound at set 0 binding 4 (fragment). std140-friendly: mat4 + vec4 = 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Globals {
    /// One light-space matrix per cascade (§11).
    light_view_proj: [[f32; 16]; SHADOW_CASCADES],
    /// World units per shadow texel, per cascade — scales the normal-offset
    /// bias so it stays exactly one texel wide whatever the cascade covers.
    texel_world: [f32; 4],
    /// x = number of live entries in the lights SSBO; y/z/w spare.
    light_params: [f32; 4],
    // x = shadow-map texel size (1/dim), y = depth bias (light-space depth), z/w -
    shadow_params: [f32; 4],
    /// World → view, for the cluster lookup (§12).
    view: [f32; 16],
    /// x = near, y = far, z = CLUSTER_Z / ln(far/near) (depth → slice scale).
    cluster_params: [f32; 4],
    /// x = tan(fov_x / 2), y = tan(fov_y / 2): view-space slope at the screen edge.
    cluster_proj: [f32; 4],
}

/// One punctual light, std430, 32 bytes (§12).
///
/// §12's layout also reserves `dir_cone` and `type` for spot lights. Left out
/// rather than padded in: unused fields cost bandwidth every frame, and adding
/// them later is a struct plus a shader edit.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuLight {
    /// xyz = world position, w = radius (the distance at which it reaches zero).
    pub pos_radius: [f32; 4],
    /// rgb = linear colour × intensity (premultiplied on the CPU, which frees
    /// the alpha channel), a = source radius: the light's physical size, for
    /// the sphere-light specular in `mesh.frag`.
    pub radiance_source: [f32; 4],
}

/// Per-cascade setup the app computes and both passes consume.
#[derive(Clone, Copy)]
pub struct CascadeSetup {
    /// Light-space view-projection for this cascade.
    pub view_proj: Mat4,
    /// World units covered by one shadow texel in this cascade.
    pub texel_world: f32,
}

/// A contiguous run of same-(mesh, LOD) instances. `prepare_frame` sorts +
/// records these once per frame; the shadow and main passes each replay them.
#[derive(Clone, Copy)]
struct Run {
    first_index: u32,
    index_count: u32,
    vertex_offset: i32,
    run_start: u32,
    run_len: u32,
}

fn as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
    }
}

pub struct MeshRenderer {
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // Depth-only pipeline for the sun shadow pass (§11); reuses `layout`/`sets`.
    shadow_pipeline: vk::Pipeline,
    // Depth-only prepass into the main depth buffer (§10), so the opaque pass
    // shades each pixel once. Shares `mesh.vert` with the main pipeline.
    depth_pipeline: vk::Pipeline,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    instance_buffers: Vec<MappedBuffer>,
    // Per-frame globals UBO (light-space matrix + shadow params), binding 4.
    globals_buffers: Vec<MappedBuffer>,
    light_buffers: Vec<MappedBuffer>,
    // Per-frame light-cluster bitmasks (§12), binding 6. Written by
    // `cluster_pipeline`, read by the main pass; GPU-only, so device-local.
    // Held for its lifetime: the descriptor sets reference it. Freed on drop.
    #[allow(dead_code)]
    cluster_buffers: Vec<Buffer>,
    cluster_pipeline: vk::Pipeline,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    // Held for its lifetime: the descriptor sets reference it. Freed on drop.
    #[allow(dead_code)]
    materials_buffer: Buffer,
    // Bindless texture array + its sampler. Images free on drop; sampler in Drop.
    #[allow(dead_code)]
    textures: Vec<Image>,
    sampler: vk::Sampler,
    slices: Vec<MeshSlice>,
    // Reused each frame to gather sorted instances before the SSBO upload.
    scratch: Vec<InstanceData>,
    // Reused each frame: instances keyed by (mesh, LOD) for sorting.
    keyed: Vec<(u32, InstanceData)>,
    /// `false` = always LOD0 (`--no-lod`), keeping the baked vertex order.
    lod_enabled: bool,
    stats: FrameStats,
    // Per-mesh runs recorded by `prepare_frame`, replayed by each pass. Both index
    // one instance SSBO laid out as [main instances | shadow instances]; the run
    // `run_start` is the firstInstance offset into that concatenation.
    main_runs: Vec<Run>,
    /// Caster runs per cascade — each cascade culls separately.
    shadow_runs: Vec<Vec<Run>>,
    /// This frame's visible lights, staged by `prepare_frame`.
    lights: Vec<GpuLight>,
    // This frame's globals (light matrix + shadow params), written in draw_shadow.
    globals: Globals,
    // Shadow-map texel size (1/dim), for the PCF offset in the globals UBO.
    shadow_texel: f32,
    max_instances: u32,
}

impl MeshRenderer {
    /// Registers `meshes` into shared vertex/index buffers and uploads a resident
    /// `materials` table (indexed by `material_id` on each instance). Returns the
    /// renderer and a `MeshId` per input mesh (same order). Both slices must be
    /// non-empty.
    pub fn new(
        renderer: &Renderer,
        meshes: &[MeshData],
        materials: &[Material],
        max_instances: u32,
        bake_dir: Option<&std::path::Path>,
    ) -> (Self, Vec<MeshId>) {
        let device = renderer.device();

        // Merge every mesh into one vertex + one index buffer; record each slice.
        // A baked mesh (§17) brings its optimised vertex order and LOD chain;
        // otherwise the loader's mesh goes in as its only LOD. Never an error.
        let mesh_dir = bake_dir.map(|d| d.join(MESH_DIR));
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut slices: Vec<MeshSlice> = Vec::with_capacity(meshes.len());
        let mut ids: Vec<MeshId> = Vec::with_capacity(meshes.len());
        let (mut baked_meshes, mut lod_count, mut would_gain) = (0usize, 0usize, 0usize);
        for (i, mesh) in meshes.iter().enumerate() {
            let baked = mesh_dir
                .as_deref()
                .and_then(|d| BakedMesh::read(&baked_mesh_path(d, mesh_key(mesh))).ok());
            let (verts, lods): (&[Vertex], Vec<(f32, &[u32])>) = match &baked {
                Some(b) => {
                    baked_meshes += 1;
                    let lods = b.lods.iter().take(MAX_LODS);
                    (
                        &b.vertices,
                        lods.map(|l| (l.error, &l.indices[..])).collect(),
                    )
                }
                None => {
                    if mesh.indices.len() / 3 >= MIN_LOD_TRIS {
                        would_gain += 1;
                    }
                    (&mesh.vertices, vec![(0.0, &mesh.indices[..])])
                }
            };
            let (center, radius) = bounding_sphere(verts);
            let mut lod_slices = Vec::with_capacity(lods.len());
            for (error, lod) in lods {
                lod_slices.push(LodSlice {
                    first_index: indices.len() as u32,
                    index_count: lod.len() as u32,
                    error,
                });
                // Per-mesh indices stay 0-based; vertex_offset rebases them at draw.
                indices.extend_from_slice(lod);
            }
            lod_count += lod_slices.len();
            slices.push(MeshSlice {
                vertex_offset: vertices.len() as i32,
                lods: lod_slices,
                center,
                radius,
            });
            ids.push(MeshId(i as u32));
            vertices.extend_from_slice(verts);
        }
        eprintln!(
            "[mesh] {} meshes ({baked_meshes} baked, {lod_count} LODs): {:.1} MB vertices, {:.1} MB indices",
            meshes.len(),
            std::mem::size_of_val(&vertices[..]) as f64 / 1_048_576.0,
            std::mem::size_of_val(&indices[..]) as f64 / 1_048_576.0,
        );
        // Only meshes big enough to get LODs; worded to stay true for the
        // app's built-in meshes, which no scene contains and no bake reaches.
        if would_gain > 0 && bake_dir.is_some() {
            eprintln!(
                "[mesh] {would_gain} meshes of {MIN_LOD_TRIS}+ triangles have no LODs (not baked; `feather-bake SCENE...` bakes a scene's meshes)"
            );
        }
        let tex_dir = bake_dir.map(|d| d.join(TEX_DIR));

        let vertex_buffer = renderer
            .create_device_local_buffer(as_bytes(&vertices), vk::BufferUsageFlags::VERTEX_BUFFER);
        let index_buffer = renderer
            .create_device_local_buffer(as_bytes(&indices), vk::BufferUsageFlags::INDEX_BUFFER);

        // Bindless texture array. Fixed defaults: slot 0 = white (base color / MR
        // fallback — white samples to 1.0, so factors pass through), slot 1 = flat
        // normal (0,0,1). Each material's real textures take the next slots.
        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        // Trilinear over the full mip chain the textures now
                        // carry, plus anisotropy so surfaces at grazing angles
                        // (the ground ahead) stay sharp instead of blurring.
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .max_lod(vk::LOD_CLAMP_NONE)
                        .anisotropy_enable(renderer.max_anisotropy().is_some())
                        .max_anisotropy(renderer.max_anisotropy().unwrap_or(1.0).min(16.0))
                        .address_mode_u(vk::SamplerAddressMode::REPEAT)
                        .address_mode_v(vk::SamplerAddressMode::REPEAT)
                        .address_mode_w(vk::SamplerAddressMode::REPEAT),
                    None,
                )
                .expect("texture sampler")
        };
        let mut textures: Vec<Image> = vec![
            renderer.create_texture(&[255, 255, 255, 255], 1, 1, true), // 0: white (sRGB)
            renderer.create_texture(&[128, 128, 255, 255], 1, 1, false), // 1: flat normal (UNORM)
        ];
        // Each unique image uploads once, however many materials use it.
        // The shadow map is the one other combined sampler in the set.
        let capacity = renderer.max_bindless_textures().saturating_sub(1);
        let plan = plan_texture_slots(materials, capacity);
        let (mut baked_count, mut bytes) = (0usize, 0usize);
        for (t, kind) in &plan.uploads {
            // A baked BC7 chain (§17) when the bake has one for exactly this
            // image in this kind and the device can sample BC. It's found by
            // the source bytes' key, so a baked texture is never decoded.
            // Otherwise raw RGBA8 with GPU-built mips; never an error.
            let baked = tex_dir
                .as_deref()
                .filter(|_| renderer.texture_compression_bc())
                .and_then(|dir| BakedTexture::read(&baked_path(dir, texture_key(t, *kind))).ok())
                .filter(|b| b.kind == *kind && (b.width, b.height) == (t.width, t.height));
            let image = match &baked {
                Some(b) => {
                    baked_count += 1;
                    bytes += b.levels.iter().map(Vec::len).sum::<usize>();
                    renderer.create_texture_bc7(&b.levels, b.width, b.height, kind.srgb())
                }
                None => match t.pixels() {
                    Some(pixels) => {
                        bytes += rgba8_chain_bytes(t.width, t.height);
                        renderer.create_texture(pixels, t.width, t.height, kind.srgb())
                    }
                    // Undecodable: a white 1×1 stands in (the factors still
                    // apply). A broken normal map shades oddly rather than
                    // failing the load.
                    None => {
                        eprintln!("[mesh] a {kind:?} texture doesn't decode; using white");
                        renderer.create_texture(&[255; 4], 1, 1, kind.srgb())
                    }
                },
            };
            textures.push(image);
        }
        let references = plan.references;
        let raw = plan.uploads.len() - baked_count;
        // Decoded images (shared across kinds, so counted once each): with a
        // complete bake this is 0, i.e. no JPEG was decoded at all.
        let mut decoded: Vec<*const TextureData> = plan
            .uploads
            .iter()
            .filter(|(t, _)| t.is_decoded())
            .map(|(t, _)| Arc::as_ptr(t))
            .collect();
        decoded.sort();
        decoded.dedup();
        eprintln!(
            "[mesh] {} textures uploaded for {references} references ({baked_count} baked BC7, {raw} raw; {} images decoded), {:.1} MB",
            plan.uploads.len(),
            decoded.len(),
            bytes as f64 / 1_048_576.0
        );
        if raw > 0 && bake_dir.is_some() {
            eprintln!(
                "[mesh] {raw} textures aren't baked: `feather-bake SCENE...` compresses them ~4x"
            );
        }
        if plan.overflow > 0 {
            eprintln!(
                "[mesh] {} texture references past the device's {capacity} texture slots use the defaults",
                plan.overflow
            );
        }
        let tex_slots = plan.slots;
        // The runtime-sized `textures[]` binding holds exactly these: the two
        // defaults plus one per unique image, with no padding.
        let texture_count = textures.len() as u32;

        // Resident material table (§5). Uploaded once; indexed by material_id.
        let gpu_materials: Vec<GpuMaterial> = materials
            .iter()
            .zip(&tex_slots)
            .map(|(m, &slots)| GpuMaterial::from_material(m, slots))
            .collect();
        let materials_buffer = renderer.create_device_local_buffer(
            as_bytes(&gpu_materials),
            vk::BufferUsageFlags::STORAGE_BUFFER,
        );

        let instance_buffers: Vec<MappedBuffer> = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                renderer.create_host_visible_buffer(
                    max_instances as u64 * INSTANCE_SIZE,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                )
            })
            .collect();
        // Per-frame globals UBO (light-space matrix + shadow params).
        let light_buffers: Vec<MappedBuffer> = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                renderer.create_host_visible_buffer(
                    (MAX_LIGHTS * std::mem::size_of::<GpuLight>()) as vk::DeviceSize,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                )
            })
            .collect();
        // Zero-filled once so a mask is never uninitialised, though every
        // frame's dispatch overwrites all of it before the main pass reads it.
        let cluster_zeros = vec![0u8; CLUSTER_COUNT * CLUSTER_WORDS * 4];
        let cluster_buffers: Vec<Buffer> = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                renderer.create_device_local_buffer(
                    &cluster_zeros,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                )
            })
            .collect();
        let globals_buffers: Vec<MappedBuffer> = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                renderer.create_host_visible_buffer(
                    std::mem::size_of::<Globals>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                )
            })
            .collect();
        // Shadow map view + comparison sampler are stable for the renderer's life
        // (fixed-size, never recreated), so the binding is written once below.
        let shadow_view = renderer.shadow_view();
        let shadow_sampler = renderer.shadow_sampler();
        let shadow_texel = 1.0 / renderer.shadow_extent().width as f32;

        let bindings = [
            // binding 0: per-frame instances (vertex stage).
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            // binding 1: resident materials (fragment stage).
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // binding 2: bindless texture array (fragment stage), sized to this
            // level's textures; the shader declares it runtime-sized.
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(texture_count)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // binding 3: sun shadow map, comparison-sampled (fragment stage).
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // binding 4: per-frame globals UBO — light matrix + shadow params,
            // plus the camera the cluster pass builds its grid from.
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE),
            // binding 5: per-frame punctual lights (§12) — assigned to clusters
            // in compute, shaded in fragment.
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE),
            // binding 6: per-frame light-cluster bitmasks (§12).
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE),
        ];
        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .expect("descriptor set layout")
        };

        let pool_sizes = [
            // instances + materials + lights + cluster masks.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4 * FRAMES_IN_FLIGHT as u32),
            // the texture array + the shadow map, per set.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count((texture_count + 1) * FRAMES_IN_FLIGHT as u32),
            // the globals UBO, per set.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(FRAMES_IN_FLIGHT as u32),
        ];
        let pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(FRAMES_IN_FLIGHT as u32)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .expect("descriptor pool")
        };

        let layouts = vec![set_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
                .expect("allocate descriptor sets")
        };
        // Texture array image infos (resident; the same for every frame's set).
        // Every slot is bound — used slots to their texture, the rest to white.
        let tex_infos: Vec<vk::DescriptorImageInfo> = textures
            .iter()
            .map(|t| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(t.view)
                    .sampler(sampler)
            })
            .collect();
        for (i, &set) in sets.iter().enumerate() {
            // binding 0 -> this frame's instance buffer; binding 1 -> the shared
            // resident materials buffer; binding 2 -> the shared texture array.
            let inst_info = [vk::DescriptorBufferInfo::default()
                .buffer(instance_buffers[i].handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let mat_info = [vk::DescriptorBufferInfo::default()
                .buffer(materials_buffer.handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            // binding 3 -> the shared shadow map; binding 4 -> this frame's globals.
            let shadow_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(shadow_view)
                .sampler(shadow_sampler)];
            let globals_info = [vk::DescriptorBufferInfo::default()
                .buffer(globals_buffers[i].handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            // binding 5 -> this frame's punctual lights.
            let light_info = [vk::DescriptorBufferInfo::default()
                .buffer(light_buffers[i].handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            // binding 6 -> this frame's cluster masks.
            let cluster_info = [vk::DescriptorBufferInfo::default()
                .buffer(cluster_buffers[i].handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&inst_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&mat_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&tex_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&shadow_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&globals_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&light_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&cluster_info),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        let vert = load_shader(&device, spv!("mesh.vert"));
        let frag = load_shader(&device, spv!("mesh.frag"));
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(c"main"),
        ];

        let vbindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let vattrs = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(12), // after pos
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(24), // after pos + normal
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vbindings)
            .vertex_attribute_descriptions(&vattrs);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        // Must match the HDR/depth targets' sample count (the geometry-pass knob).
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(renderer.samples());
        // Opaque pass runs after the depth prepass: depth is already laid down, so
        // test against it and don't write. LESS_OR_EQUAL rather than EQUAL as cheap
        // insurance — early-Z still rejects everything strictly behind.
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
        let blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);

        let set_layouts = [set_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(96)]; // mat4 view_proj + vec4 light_dir + vec4 camera_pos
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push_ranges),
                    None,
                )
                .expect("pipeline layout")
        };

        // Geometry now renders into the offscreen HDR target, not the swapchain.
        let color_formats = [renderer.hdr_format()];
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(renderer.depth_format());

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)
                .expect("graphics pipeline")[0]
        };

        // ---- Depth prepass pipeline (§10): lays down opaque depth so the main
        // pass can early-Z out occluded fragments instead of shading them twice.
        // Deliberately reuses `mesh.vert` — the SAME module as the main pass — so
        // gl_Position is bit-identical and the LESS_OR_EQUAL test above always
        // passes for the surface that won the prepass. A separate depth-only
        // shader (e.g. shadow.vert) associates its matrix multiply differently,
        // which would round differently and punch holes in the image.
        let depth_stages = [vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(c"main")];
        let prepass_depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
        // This runs *inside* the geometry pass, which has one colour attachment, so
        // the pipeline must declare the same attachment count — writing no colour
        // is expressed with an empty write mask, not by omitting the attachment.
        // Depth target, sample count and vertex input all match the main pass (they
        // must agree for the prepass depths to match).
        let depth_blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::empty())
            .blend_enable(false)];
        let depth_color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&depth_blend_attachment);
        let mut depth_rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(renderer.depth_format());
        let depth_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&depth_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&prepass_depth)
            .color_blend_state(&depth_color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut depth_rendering);
        let depth_pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[depth_pipeline_info], None)
                .map_err(|(_, e)| e)
                .expect("depth prepass pipeline")[0]
        };

        // ---- Shadow (depth-only) pipeline: renders instances from the sun into
        // the shadow map. Vertex-only, no color attachment; reuses `layout` (its
        // vertex shader reads only the instance SSBO + a 64 B light-matrix push).
        let shadow_vert = load_shader(&device, spv!("shadow.vert"));
        let shadow_stages = [vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shadow_vert)
            .name(c"main")];
        let shadow_dyn_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::DEPTH_BIAS,
        ];
        let shadow_dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&shadow_dyn_states);
        // Front-face cull + slope-scaled depth bias (set dynamically per pass)
        // reduce shadow acne and peter-panning. Depth clamp pancakes casters
        // (§11): anything nearer the sun than the cascade's near plane lands on
        // it at depth 0 instead of being clipped, so it still shadows everything
        // behind it. Exact per fragment, unlike clamping z in the vertex shader.
        let shadow_raster = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(renderer.depth_clamp())
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::FRONT)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(true)
            .line_width(1.0);
        let shadow_ms = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let shadow_depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
        let mut shadow_rendering = vk::PipelineRenderingCreateInfo::default()
            .depth_attachment_format(renderer.shadow_format());
        // shadow.vert consumes only position (location 0); describing just that
        // attribute avoids "attribute not consumed" validation warnings.
        let shadow_vattrs = [vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)];
        let shadow_vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vbindings)
            .vertex_attribute_descriptions(&shadow_vattrs);
        let shadow_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shadow_stages)
            .vertex_input_state(&shadow_vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&shadow_raster)
            .multisample_state(&shadow_ms)
            .depth_stencil_state(&shadow_depth)
            .dynamic_state(&shadow_dynamic_state)
            .layout(layout)
            .push_next(&mut shadow_rendering);
        let shadow_pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[shadow_pipeline_info], None)
                .map_err(|(_, e)| e)
                .expect("shadow pipeline")[0]
        };

        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
            device.destroy_shader_module(shadow_vert, None);
        }

        // Light-cluster assignment (§12), the engine's first compute pipeline.
        // Reuses `layout`: it binds the same set 0 (globals, lights, masks), and
        // declares no push constants, so the graphics-only push range is inert.
        let cluster_comp = load_shader(&device, spv!("cluster.comp"));
        let cluster_ci = vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(cluster_comp)
                    .name(c"main"),
            )
            .layout(layout);
        let cluster_pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[cluster_ci], None)
                .map_err(|(_, e)| e)
                .expect("cluster pipeline")[0]
        };
        unsafe { device.destroy_shader_module(cluster_comp, None) };

        let renderer = Self {
            device,
            layout,
            pipeline,
            shadow_pipeline,
            depth_pipeline,
            set_layout,
            pool,
            sets,
            instance_buffers,
            globals_buffers,
            light_buffers,
            cluster_buffers,
            cluster_pipeline,
            vertex_buffer,
            index_buffer,
            materials_buffer,
            textures,
            sampler,
            slices,
            scratch: Vec::new(),
            keyed: Vec::new(),
            lod_enabled: true,
            stats: FrameStats::default(),
            main_runs: Vec::new(),
            shadow_runs: (0..SHADOW_CASCADES).map(|_| Vec::new()).collect(),
            lights: Vec::new(),
            globals: Globals::default(),
            shadow_texel,
            max_instances,
        };
        (renderer, ids)
    }

    /// Re-point every frame's shadow-map descriptor (binding 3) at a new view —
    /// used after the shadow map is resized by the quality setting (§13).
    ///
    /// **The caller must have waited for the device to go idle.** Unlike
    /// `TonemapPass::update`, which refreshes one frame's descriptor after that
    /// frame's fence, this rewrites *all* of them at once; that is only sound
    /// when nothing is in flight. It is a rare, settings-change-only path.
    /// Re-point at a resized shadow map. `dim` matters: `shadow_params.x` is the
    /// PCF tap offset in UV, so leaving it stale after a resize would collapse
    /// the 3x3 taps into a single texel (too small) or smear them (too large).
    pub fn set_shadow_map(&mut self, view: vk::ImageView, sampler: vk::Sampler, dim: u32) {
        self.shadow_texel = 1.0 / dim as f32;
        for &set in &self.sets {
            let info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view)
                .sampler(sampler)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info);
            unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        }
    }

    /// CPU-only per-frame prep (call once, before `draw_frame`): sort each culled
    /// list by mesh, stage them into one instance array as `[main | shadow]`,
    /// record each pass's per-mesh runs, and stash this frame's globals (light
    /// matrix + shadow params). No GPU buffers are touched here — the actual upload
    /// happens in `draw_shadow`, after the frame fence, to avoid racing the
    /// in-flight GPU read of the per-frame buffers.
    ///
    /// `main` is the camera-visible set, `shadow` the light-frustum set (§8); an
    /// entity visible to both appears once in each region.
    /// Sort and stage this frame's instances. `shadows` carries one caster list
    /// per cascade (each culled against its own ortho), so they land in the SSBO
    /// as separate regions and each cascade replays only its own runs.
    pub fn prepare_frame(
        &mut self,
        main: &mut [(MeshId, InstanceData)],
        shadows: &mut [Vec<(MeshId, InstanceData)>],
        cascades: &[CascadeSetup],
        lights: &[GpuLight],
        camera: &ClusterView,
    ) {
        // Clamp rather than overflow: the buffer is sized once at startup.
        self.lights.clear();
        self.lights
            .extend_from_slice(&lights[..lights.len().min(MAX_LIGHTS)]);
        if lights.len() > MAX_LIGHTS {
            eprintln!(
                "[light] {} visible lights exceeds MAX_LIGHTS ({MAX_LIGHTS}); dropping the rest",
                lights.len()
            );
        }
        self.scratch.clear();
        self.main_runs.clear();
        for runs in &mut self.shadow_runs {
            runs.clear();
        }

        let cap = self.max_instances as usize;
        // LOD rules per view (§17): pixels for the camera, texels per cascade.
        // `None` pins everything to LOD0 (`--no-lod`).
        let screen = LodRule::Screen {
            eye: camera.view.inverse().w_axis.truncate(),
            px_per_unit: camera.viewport_height / (2.0 * (camera.fov_y * 0.5).tan()),
            near: camera.near,
        };
        let enabled = self.lod_enabled;
        let mut stats = FrameStats::default();
        // main region at offset 0, then one region per cascade. `run_start` is the
        // firstInstance base into the concatenated SSBO.
        let mut runs = Runs {
            slices: &self.slices,
            keyed: &mut self.keyed,
            scratch: &mut self.scratch,
        };
        let main_stats = runs.build(main, enabled.then_some(screen), &mut self.main_runs, 0);
        stats.main_tris = main_stats.tris;
        stats.main_lods = main_stats.lods;
        let mut base = main_stats.instances;
        for (cascade, casters) in shadows.iter_mut().enumerate().take(SHADOW_CASCADES) {
            let rule = cascades
                .get(cascade)
                .map(|c| LodRule::Texel(c.texel_world))
                .filter(|_| enabled);
            let st = runs.build(casters, rule, &mut self.shadow_runs[cascade], base);
            stats.shadow_tris += st.tris;
            for (total, n) in stats.shadow_lods.iter_mut().zip(st.lods) {
                *total += n;
            }
            base += st.instances;
        }
        self.stats = stats;
        // Guard the shared buffer's capacity (main + every cascade could, worst
        // case, exceed it); drop the tail of scratch and any runs past the cap.
        if self.scratch.len() > cap {
            self.scratch.truncate(cap);
            self.main_runs
                .retain(|r| r.run_start + r.run_len <= cap as u32);
            for runs in &mut self.shadow_runs {
                runs.retain(|r| r.run_start + r.run_len <= cap as u32);
            }
        }

        let mut globals = Globals {
            light_view_proj: [[0.0; 16]; SHADOW_CASCADES],
            texel_world: [0.0; 4],
            light_params: [self.lights.len() as f32, 0.0, 0.0, 0.0],
            shadow_params: [self.shadow_texel, SHADOW_DEPTH_BIAS, 0.0, 0.0],
            view: camera.view.to_cols_array(),
            cluster_params: [
                camera.near,
                camera.far,
                CLUSTER_Z as f32 / (camera.far / camera.near).ln(),
                0.0,
            ],
            cluster_proj: {
                let tan_y = (camera.fov_y * 0.5).tan();
                [tan_y * camera.aspect, tan_y, 0.0, 0.0]
            },
        };
        for (i, c) in cascades.iter().enumerate().take(SHADOW_CASCADES) {
            globals.light_view_proj[i] = c.view_proj.to_cols_array();
            globals.texel_world[i] = c.texel_world;
        }
        self.globals = globals;
    }

    /// Record the sun shadow pass: depth-only draws of the prepared runs from the
    /// light's point of view. Runs first in the frame (after the fence wait), so
    /// it also performs this frame's per-frame buffer uploads — the instance SSBO
    /// and the globals UBO the main pass then consumes.
    pub fn draw_shadow(
        &self,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        frame: usize,
        cascade: usize,
    ) {
        // Uploads ride in the first cascade only. This is called once per
        // cascade, and re-writing the buffers mid-pass would both waste
        // bandwidth and race the draws already recording against them.
        if cascade == 0 {
            // Safe to write now: draw_frame waited on this frame index's fence.
            self.instance_buffers[frame].write(as_bytes(&self.scratch));
            self.globals_buffers[frame].write(as_bytes(std::slice::from_ref(&self.globals)));
            self.light_buffers[frame].write(as_bytes(&self.lights));
        }
        let Some(runs) = self.shadow_runs.get(cascade) else {
            return;
        };

        // shadow.vert reads this cascade's light matrix from the push constant
        // (offset 0, 64 B), so the shader itself needs no cascade awareness.
        let lvp = self.globals.light_view_proj[cascade];
        let push_bytes = unsafe { std::slice::from_raw_parts(lvp.as_ptr() as *const u8, 64) };
        unsafe {
            self.device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                push_bytes,
            );
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.shadow_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[self.sets[frame]],
                &[],
            );
            // Slope-scaled depth bias to push casters away from the light and kill
            // acne. (constant_factor, clamp, slope_factor).
            self.device.cmd_set_depth_bias(cmd, 2.0, 0.0, 3.0);
            self.set_viewport_scissor(cmd, extent);
            self.bind_geometry(cmd);
            self.draw_runs(cmd, runs);
        }
    }

    /// Record the light-cluster assignment (§12): one thread per cluster builds
    /// its view-space AABB and tests every light in this frame's list against
    /// it, writing a bitmask. Always dispatched, even with no lights, so the
    /// main pass never reads a stale mask that names lights no longer present.
    /// Must be recorded after `draw_shadow` (which uploads the globals + lights)
    /// and outside a render pass; `draw_frame`'s cluster slot guarantees both.
    pub fn dispatch_clusters(&self, cmd: vk::CommandBuffer, frame: usize) {
        let groups = CLUSTER_COUNT.div_ceil(CLUSTER_GROUP) as u32;
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.cluster_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.sets[frame]],
                &[],
            );
            self.device.cmd_dispatch(cmd, groups, 1, 1);
        }
    }

    /// Record the depth prepass (§10): the camera-visible runs, depth only, into
    /// the geometry pass's depth attachment. Run before `draw_main`, which then
    /// tests LESS_OR_EQUAL with depth-write off and shades each pixel once.
    /// Uses the same push constant and culled set as `draw_main`.
    pub fn draw_depth_prepass(
        &self,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        frame: usize,
        view_proj: Mat4,
        light_dir: Vec4,
        camera_pos: glam::Vec3,
    ) {
        unsafe {
            self.push_view_constants(cmd, view_proj, light_dir, camera_pos);
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.depth_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[self.sets[frame]],
                &[],
            );
            self.set_viewport_scissor(cmd, extent);
            self.bind_geometry(cmd);
            self.draw_runs(cmd, &self.main_runs);
        }
    }

    /// Record the main lit pass. Sorts/uploads happen in `upload_instances`; this
    /// replays the runs into the HDR target with the full PBR + shadow pipeline.
    pub fn draw_main(
        &self,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        frame: usize,
        view_proj: Mat4,
        light_dir: Vec4,
        camera_pos: glam::Vec3,
    ) {
        unsafe {
            self.push_view_constants(cmd, view_proj, light_dir, camera_pos);
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[self.sets[frame]],
                &[],
            );
            self.set_viewport_scissor(cmd, extent);
            self.bind_geometry(cmd);
            self.draw_runs(cmd, &self.main_runs);
        }
    }

    /// Push the shared 96 B view constants: mat4 view_proj + vec4 light_dir +
    /// vec4 camera_pos. Identical for the depth prepass and the main pass.
    unsafe fn push_view_constants(
        &self,
        cmd: vk::CommandBuffer,
        view_proj: Mat4,
        light_dir: Vec4,
        camera_pos: glam::Vec3,
    ) {
        let mut push = [0f32; 24];
        push[..16].copy_from_slice(&view_proj.to_cols_array());
        push[16..20].copy_from_slice(&light_dir.to_array());
        push[20..23].copy_from_slice(&camera_pos.to_array());
        let push_bytes = std::slice::from_raw_parts(push.as_ptr() as *const u8, 96);
        self.device.cmd_push_constants(
            cmd,
            self.layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );
    }

    unsafe fn set_viewport_scissor(&self, cmd: vk::CommandBuffer, extent: vk::Extent2D) {
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        self.device.cmd_set_viewport(cmd, 0, &[viewport]);
        self.device.cmd_set_scissor(cmd, 0, &[scissor]);
    }

    /// Triangles and LOD mix submitted by the last `prepare_frame`.
    pub fn frame_stats(&self) -> FrameStats {
        self.stats
    }

    /// `false` pins every instance to LOD0 (`--no-lod`): the A/B that
    /// separates the baked vertex order from the LOD win.
    pub fn set_lod_enabled(&mut self, enabled: bool) {
        self.lod_enabled = enabled;
    }

    unsafe fn bind_geometry(&self, cmd: vk::CommandBuffer) {
        self.device
            .cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer.handle], &[0]);
        self.device
            .cmd_bind_index_buffer(cmd, self.index_buffer.handle, 0, vk::IndexType::UINT32);
    }

    /// One `cmd_draw_indexed` per contiguous same-mesh run. `firstInstance` =
    /// `run_start`, so `gl_InstanceIndex` indexes the concatenated instance SSBO.
    unsafe fn draw_runs(&self, cmd: vk::CommandBuffer, runs: &[Run]) {
        for r in runs {
            self.device.cmd_draw_indexed(
                cmd,
                r.index_count,
                r.run_len,
                r.first_index,
                r.vertex_offset,
                r.run_start,
            );
        }
    }
}

/// What one view's run building produced.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RunStats {
    instances: u32,
    tris: u64,
    lods: [u32; MAX_LODS],
}

/// The per-frame state run building shares across views.
struct Runs<'a> {
    slices: &'a [MeshSlice],
    keyed: &'a mut Vec<(u32, InstanceData)>,
    scratch: &'a mut Vec<InstanceData>,
}

impl Runs<'_> {
    /// Pick each item's LOD under `rule` (`None` = LOD0), sort by (mesh, LOD),
    /// append the instances to `scratch`, and record one run per contiguous
    /// (mesh, LOD) group with `firstInstance = base + local_start`. `base`
    /// must equal `scratch.len()` at entry.
    fn build(
        &mut self,
        items: &[(MeshId, InstanceData)],
        rule: Option<LodRule>,
        runs: &mut Vec<Run>,
        base: u32,
    ) -> RunStats {
        let mut stats = RunStats::default();
        self.keyed.clear();
        self.keyed.extend(items.iter().filter_map(|(mesh, inst)| {
            // Unknown meshes are dropped here, as the draw loop always did.
            let slice = self.slices.get(mesh.0 as usize)?;
            let lod = rule.map_or(0, |r| {
                pick_lod(
                    &slice.lods,
                    r.budget(&inst.model, slice.center, slice.radius),
                )
            });
            Some(((mesh.0 << 4) | lod as u32, *inst))
        }));
        // Unstable sort is fine; draw order within a run doesn't matter
        // (opaque + depth test).
        self.keyed.sort_unstable_by_key(|(key, _)| *key);
        self.scratch
            .extend(self.keyed.iter().map(|(_, inst)| *inst));

        let count = self.keyed.len();
        let mut first = 0usize;
        while first < count {
            let key = self.keyed[first].0;
            let mut run = 1usize;
            while first + run < count && self.keyed[first + run].0 == key {
                run += 1;
            }
            let slice = &self.slices[(key >> 4) as usize];
            let lod = (key & 0xf) as usize;
            let l = slice.lods[lod];
            runs.push(Run {
                first_index: l.first_index,
                index_count: l.index_count,
                vertex_offset: slice.vertex_offset,
                run_start: base + first as u32,
                run_len: run as u32,
            });
            stats.tris += (l.index_count / 3) as u64 * run as u64;
            stats.lods[lod] += run as u32;
            first += run;
        }
        stats.instances = count as u32;
        stats
    }
}

/// Small constant bias in light-space depth, added in the shader on top of the
/// rasterizer slope bias, to finish off shadow acne.
const SHADOW_DEPTH_BIAS: f32 = 0.0015;

impl Drop for MeshRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline(self.shadow_pipeline, None);
            self.device.destroy_pipeline(self.depth_pipeline, None);
            self.device.destroy_pipeline(self.cluster_pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

fn load_shader(device: &ash::Device, bytes: &[u8]) -> vk::ShaderModule {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).expect("read SPIR-V");
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe {
        device
            .create_shader_module(&info, None)
            .expect("create shader module")
    }
}

// A cluster's bitmask must cover every light exactly.
const _: () = assert!(MAX_LIGHTS.is_multiple_of(32));

#[cfg(test)]
mod tests {
    use super::*;

    fn tex(n: u8) -> Arc<TextureData> {
        Arc::new(TextureData::from_rgba8(vec![n, n, n, 255], 1, 1))
    }

    #[test]
    fn texture_slots_dedup_by_image_and_kind() {
        let shared = tex(1);
        let other = tex(2);
        let with = |base: Option<&Arc<TextureData>>, mr: Option<&Arc<TextureData>>| Material {
            base_color_texture: base.cloned(),
            metallic_roughness_texture: mr.cloned(),
            ..Material::default()
        };
        let normal = Material {
            normal_texture: Some(shared.clone()),
            ..Material::default()
        };
        let materials = [
            with(Some(&shared), None),
            with(Some(&shared), None), // same image again: same slot
            with(Some(&other), None),  // a distinct image: its own slot
            with(None, Some(&shared)), // same pixels as data (UNORM): new slot
            with(None, None),          // no textures: the defaults
            normal,                    // as a normal map it's data too: shared
        ];
        let plan = plan_texture_slots(&materials, 1024);
        assert_eq!(plan.uploads.len(), 3);
        assert_eq!(plan.references, 5);
        assert_eq!(plan.overflow, 0);
        let s = &plan.slots;
        assert_eq!(s[0][0], s[1][0], "shared image shares a slot");
        assert_ne!(s[0][0], s[2][0], "distinct images get distinct slots");
        assert_ne!(s[3][2], s[0][0], "sRGB and UNORM views are separate");
        assert_eq!(s[4], [0, 1, 0], "untextured materials use the defaults");
        assert_eq!(s[5][1], s[3][2], "normal and MR maps are both UNORM data");
        let kinds: Vec<TexKind> = plan.uploads.iter().map(|(_, k)| *k).collect();
        assert_eq!(kinds, [TexKind::Color, TexKind::Color, TexKind::Data]);
        assert!(s[..4].iter().flatten().all(|&x| x == 0 || x == 1 || x >= 2));
    }

    #[test]
    fn texture_overflow_is_counted_and_falls_back() {
        // More distinct images than slots: the excess falls back to the
        // defaults and is reported, rather than vanishing silently.
        // A small capacity stands in for a device limit.
        let capacity = 16;
        let materials: Vec<Material> = (0..capacity + 5)
            .map(|i| Material {
                base_color_texture: Some(tex(i as u8)),
                ..Material::default()
            })
            .collect();
        let plan = plan_texture_slots(&materials, capacity);
        assert_eq!(plan.uploads.len(), capacity - FIRST_TEXTURE_SLOT);
        assert_eq!(plan.overflow, 7);
        assert_eq!(plan.slots.last().unwrap()[0], 0, "overflow uses white");
    }

    fn lods(errors: &[f32]) -> Vec<LodSlice> {
        errors
            .iter()
            .enumerate()
            .map(|(i, &error)| LodSlice {
                first_index: i as u32 * 300,
                index_count: 300 >> i,
                error,
            })
            .collect()
    }

    fn screen() -> LodRule {
        LodRule::Screen {
            eye: Vec3::ZERO,
            px_per_unit: 1000.0,
            near: 0.1,
        }
    }

    #[test]
    fn pick_lod_takes_the_coarsest_level_within_budget() {
        let l = lods(&[0.0, 0.01, 0.02, 0.08]);
        assert_eq!(pick_lod(&l, 0.0), 0);
        assert_eq!(pick_lod(&l, 0.015), 1);
        assert_eq!(pick_lod(&l, 0.02), 2);
        assert_eq!(pick_lod(&l, 1.0), 3);
        // A negative budget can't happen, but must still give LOD0.
        assert_eq!(pick_lod(&l, -1.0), 0);
    }

    #[test]
    fn screen_budget_is_one_pixel_and_grows_with_distance() {
        let at = |z: f32, scale: f32| {
            let model = Mat4::from_translation(Vec3::new(0.0, 0.0, -z))
                * Mat4::from_scale(Vec3::splat(scale));
            screen().budget(&model, Vec3::ZERO, 0.5)
        };
        // Unit scale, 10.5 away, radius 0.5: nearest point at 10 units, where one
        // pixel is 10 / 1000 = 0.01 world units.
        assert!((at(10.5, 1.0) - 0.01).abs() < 1e-6, "{}", at(10.5, 1.0));
        assert!(at(100.0, 1.0) > at(10.0, 1.0));
        // A twice-as-big instance at the same distance must keep finer detail
        // (in mesh units the budget halves, and more: its sphere is nearer).
        assert!(at(20.0, 2.0) < at(20.0, 1.0) / 2.0);
        // Inside the sphere: clamped to the near plane, i.e. LOD0 territory.
        assert!(at(0.2, 1.0) <= 0.1 / 1000.0 + 1e-9);
        // Non-uniform scale uses the largest axis.
        let stretched = Mat4::from_translation(Vec3::new(0.0, 0.0, -50.0))
            * Mat4::from_scale(Vec3::new(1.0, 4.0, 1.0));
        let b = screen().budget(&stretched, Vec3::ZERO, 0.0);
        assert!((b - 50.0 / (1000.0 * 4.0)).abs() < 1e-7, "{b}");
    }

    #[test]
    fn texel_budget_ignores_distance() {
        let rule = LodRule::Texel(0.05);
        let near = Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0));
        let far = Mat4::from_translation(Vec3::new(0.0, 0.0, -500.0));
        assert_eq!(rule.budget(&near, Vec3::ZERO, 1.0), 0.05);
        assert_eq!(rule.budget(&far, Vec3::ZERO, 1.0), 0.05);
        let big = Mat4::from_scale(Vec3::splat(5.0));
        assert!((rule.budget(&big, Vec3::ZERO, 1.0) - 0.01).abs() < 1e-7);
    }

    #[test]
    fn runs_split_by_mesh_and_lod() {
        let slices = vec![
            MeshSlice {
                vertex_offset: 0,
                lods: lods(&[0.0, 0.01, 0.1]),
                center: Vec3::ZERO,
                radius: 0.5,
            },
            MeshSlice {
                vertex_offset: 100,
                lods: lods(&[0.0]),
                center: Vec3::ZERO,
                radius: 0.5,
            },
        ];
        let inst = |z: f32| InstanceData::new(Mat4::from_translation(Vec3::new(0.0, 0.0, -z)), 0);
        // Mesh 0 at 5 (LOD0: budget 0.0045), 20 and 30 (LOD1), 500 (LOD2);
        // mesh 1 far away (its only LOD); mesh 7 doesn't exist.
        let items = vec![
            (MeshId(0), inst(500.0)),
            (MeshId(0), inst(20.0)),
            (MeshId(1), inst(900.0)),
            (MeshId(0), inst(5.0)),
            (MeshId(7), inst(5.0)),
            (MeshId(0), inst(30.0)),
        ];
        let (mut keyed, mut scratch, mut runs) = (Vec::new(), Vec::new(), Vec::new());
        let mut b = Runs {
            slices: &slices,
            keyed: &mut keyed,
            scratch: &mut scratch,
        };
        let st = b.build(&items, Some(screen()), &mut runs, 10);
        let got: Vec<(u32, u32, i32, u32, u32)> = runs
            .iter()
            .map(|r| {
                (
                    r.first_index,
                    r.index_count,
                    r.vertex_offset,
                    r.run_start,
                    r.run_len,
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (0, 300, 0, 10, 1),   // mesh 0 LOD0
                (300, 150, 0, 11, 2), // mesh 0 LOD1 x2
                (600, 75, 0, 13, 1),  // mesh 0 LOD2
                (0, 300, 100, 14, 1), // mesh 1 LOD0
            ]
        );
        assert_eq!(st.instances, 5, "the unknown mesh is dropped");
        assert_eq!(scratch.len(), 5);
        assert_eq!(st.tris, 100 + 2 * 50 + 25 + 100);
        assert_eq!(&st.lods[..3], &[2, 2, 1]);

        // No rule: everything at LOD0.
        let (mut keyed, mut scratch, mut runs) = (Vec::new(), Vec::new(), Vec::new());
        let mut b = Runs {
            slices: &slices,
            keyed: &mut keyed,
            scratch: &mut scratch,
        };
        let st = b.build(&items, None, &mut runs, 0);
        assert_eq!(st.lods[0], 5);
        assert_eq!(runs.len(), 2);
    }

    /// The cluster grid and light cap are repeated as GLSL constants in both
    /// shaders. A mismatch still compiles, and fragments then silently read
    /// masks built for a different grid.
    #[test]
    fn shader_cluster_constants_match() {
        let shaders = [
            ("cluster.comp", include_str!("../shaders/cluster.comp")),
            ("mesh.frag", include_str!("../shaders/mesh.frag")),
        ];
        let consts = [
            ("CLUSTER_X", CLUSTER_X),
            ("CLUSTER_Y", CLUSTER_Y),
            ("CLUSTER_Z", CLUSTER_Z),
            ("MAX_LIGHTS", MAX_LIGHTS),
        ];
        for (name, src) in shaders {
            for (c, v) in consts {
                let decl = format!("const uint {c} = {v};");
                assert!(src.contains(&decl), "{name} lacks `{decl}`");
            }
        }
        let group = format!("local_size_x = {CLUSTER_GROUP}");
        assert!(
            shaders[0].1.contains(&group),
            "cluster.comp lacks `{group}`"
        );
    }
}
