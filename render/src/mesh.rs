//! Instanced, lit mesh renderer. Several meshes live in one shared vertex +
//! index buffer (each a `{first_index, index_count, vertex_offset}` slice);
//! per-frame instances are sorted by mesh so each mesh draws as one contiguous
//! run (`cmd_draw_indexed` with `firstInstance`). Per-instance data is
//! `{ model, material_id }`; the fragment shader reads the material from a
//! resident materials SSBO and applies a directional light. Meshes and materials
//! come in as Vulkan-free `assets` types.

use ash::vk;
use feather_assets::{Material, MeshData, Vertex};
use feather_gfx::{Buffer, Image, MappedBuffer, Renderer, FRAMES_IN_FLIGHT};
use glam::{Mat4, Vec4};

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

/// Fixed size of the bindless texture array (`textures[]` in the shader). Slot 0
/// is a resident white default; unused slots also point at it.
const MAX_TEXTURES: usize = 64;

/// Handle to a mesh registered with the renderer, in `new`'s input order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MeshId(pub u32);

/// Where one mesh lives inside the shared vertex/index buffers. `vertex_offset`
/// is added to each (mesh-local) index at draw time, so per-mesh indices stay
/// 0-based.
#[derive(Clone, Copy)]
struct MeshSlice {
    first_index: u32,
    index_count: u32,
    vertex_offset: i32,
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
    light_view_proj: [f32; 16],
    // x = shadow-map texel size (1/dim), y = depth bias (light-space depth), z/w -
    shadow_params: [f32; 4],
}

/// A contiguous run of same-mesh instances. `upload_instances` sorts + records
/// these once per frame; the shadow and main passes each replay them.
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
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    instance_buffers: Vec<MappedBuffer>,
    // Per-frame globals UBO (light-space matrix + shadow params), binding 4.
    globals_buffers: Vec<MappedBuffer>,
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
    // Per-mesh runs recorded by `prepare_frame`, replayed by both passes.
    runs: Vec<Run>,
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
    ) -> (Self, Vec<MeshId>) {
        let device = renderer.device();

        // Merge every mesh into one vertex + one index buffer; record each slice.
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut slices: Vec<MeshSlice> = Vec::with_capacity(meshes.len());
        let mut ids: Vec<MeshId> = Vec::with_capacity(meshes.len());
        for (i, mesh) in meshes.iter().enumerate() {
            slices.push(MeshSlice {
                first_index: indices.len() as u32,
                index_count: mesh.indices.len() as u32,
                vertex_offset: vertices.len() as i32,
            });
            ids.push(MeshId(i as u32));
            vertices.extend_from_slice(&mesh.vertices);
            // Per-mesh indices stay 0-based; vertex_offset rebases them at draw.
            indices.extend_from_slice(&mesh.indices);
        }

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
        // Upload each material's real textures into the next free slots.
        let tex_slots: Vec<[u32; 3]> = {
            let mut upload =
                |tex: &Option<feather_assets::TextureData>, srgb: bool, default: u32| match tex {
                    Some(t) if textures.len() < MAX_TEXTURES => {
                        let slot = textures.len() as u32;
                        textures.push(renderer.create_texture(&t.pixels, t.width, t.height, srgb));
                        slot
                    }
                    _ => default,
                };
            materials
                .iter()
                .map(|m| {
                    [
                        upload(&m.base_color_texture, true, 0),
                        upload(&m.normal_texture, false, 1),
                        upload(&m.metallic_roughness_texture, false, 0),
                    ]
                })
                .collect()
        };

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
            // binding 2: bindless texture array (fragment stage).
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_TEXTURES as u32)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // binding 3: sun shadow map, comparison-sampled (fragment stage).
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // binding 4: per-frame globals UBO — light matrix + shadow params.
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
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
            // instances + materials.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2 * FRAMES_IN_FLIGHT as u32),
            // the texture array + the shadow map, per set.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(((MAX_TEXTURES + 1) * FRAMES_IN_FLIGHT) as u32),
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
        let tex_infos: Vec<vk::DescriptorImageInfo> = (0..MAX_TEXTURES)
            .map(|i| {
                let view = textures.get(i).unwrap_or(&textures[0]).view;
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(view)
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
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
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
        // reduce shadow acne and peter-panning.
        let shadow_raster = vk::PipelineRasterizationStateCreateInfo::default()
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

        let renderer = Self {
            device,
            layout,
            pipeline,
            shadow_pipeline,
            set_layout,
            pool,
            sets,
            instance_buffers,
            globals_buffers,
            vertex_buffer,
            index_buffer,
            materials_buffer,
            textures,
            sampler,
            slices,
            scratch: Vec::new(),
            runs: Vec::new(),
            globals: Globals::default(),
            shadow_texel,
            max_instances,
        };
        (renderer, ids)
    }

    /// CPU-only per-frame prep (call once, before `draw_frame`): sort `items` by
    /// mesh, stage the sorted instances + per-mesh runs, and stash this frame's
    /// globals (light matrix + shadow params). No GPU buffers are touched here —
    /// the actual uploads happen in `draw_shadow`, after the frame fence, to avoid
    /// racing the in-flight GPU read of the per-frame buffers.
    pub fn prepare_frame(&mut self, items: &mut [(MeshId, InstanceData)], light_view_proj: Mat4) {
        // Contiguous runs per mesh -> one draw each. Unstable sort is fine; draw
        // order within a mesh doesn't matter (opaque + depth test).
        items.sort_unstable_by_key(|(mesh, _)| mesh.0);

        let count = items.len().min(self.max_instances as usize);
        self.scratch.clear();
        self.scratch
            .extend(items[..count].iter().map(|(_, inst)| *inst));

        self.runs.clear();
        let mut first = 0usize;
        while first < count {
            let mesh = items[first].0;
            let mut run = 1usize;
            while first + run < count && items[first + run].0 == mesh {
                run += 1;
            }
            if let Some(slice) = self.slices.get(mesh.0 as usize) {
                self.runs.push(Run {
                    first_index: slice.first_index,
                    index_count: slice.index_count,
                    vertex_offset: slice.vertex_offset,
                    run_start: first as u32,
                    run_len: run as u32,
                });
            }
            first += run;
        }

        self.globals = Globals {
            light_view_proj: light_view_proj.to_cols_array(),
            shadow_params: [self.shadow_texel, SHADOW_DEPTH_BIAS, 0.0, 0.0],
        };
    }

    /// Record the sun shadow pass: depth-only draws of the prepared runs from the
    /// light's point of view. Runs first in the frame (after the fence wait), so
    /// it also performs this frame's per-frame buffer uploads — the instance SSBO
    /// and the globals UBO the main pass then consumes.
    pub fn draw_shadow(&self, cmd: vk::CommandBuffer, extent: vk::Extent2D, frame: usize) {
        // Safe to write now: draw_frame waited on this frame index's fence.
        self.instance_buffers[frame].write(as_bytes(&self.scratch));
        self.globals_buffers[frame].write(as_bytes(std::slice::from_ref(&self.globals)));

        // shadow.vert reads the light matrix from the push constant (offset 0, 64 B).
        let lvp = self.globals.light_view_proj;
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
            self.draw_runs(cmd);
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
        // Push constant (96 B): mat4 view_proj + vec4 light_dir + vec4 camera_pos.
        let mut push = [0f32; 24];
        push[..16].copy_from_slice(&view_proj.to_cols_array());
        push[16..20].copy_from_slice(&light_dir.to_array());
        push[20..23].copy_from_slice(&camera_pos.to_array());
        let push_bytes = unsafe { std::slice::from_raw_parts(push.as_ptr() as *const u8, 96) };

        unsafe {
            self.device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                push_bytes,
            );
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
            self.draw_runs(cmd);
        }
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

    unsafe fn bind_geometry(&self, cmd: vk::CommandBuffer) {
        self.device
            .cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer.handle], &[0]);
        self.device
            .cmd_bind_index_buffer(cmd, self.index_buffer.handle, 0, vk::IndexType::UINT32);
    }

    /// One `cmd_draw_indexed` per contiguous same-mesh run recorded by
    /// `upload_instances`. `firstInstance` = run start, so `gl_InstanceIndex`
    /// indexes the sorted instance SSBO.
    unsafe fn draw_runs(&self, cmd: vk::CommandBuffer) {
        for r in &self.runs {
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

/// Small constant bias in light-space depth, added in the shader on top of the
/// rasterizer slope bias, to finish off shadow acne.
const SHADOW_DEPTH_BIAS: f32 = 0.0015;

impl Drop for MeshRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline(self.shadow_pipeline, None);
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
