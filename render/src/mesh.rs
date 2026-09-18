//! Instanced, lit mesh renderer. Several meshes live in one shared vertex +
//! index buffer (each a `{first_index, index_count, vertex_offset}` slice);
//! per-frame instances are sorted by mesh so each mesh draws as one contiguous
//! run (`cmd_draw_indexed` with `firstInstance`). Per-instance data is
//! `{ model, color }` from a storage buffer; a directional light is applied in
//! the fragment shader. Meshes come in as Vulkan-free `assets::MeshData`.

use ash::vk;
use feather_assets::{MeshData, Vertex};
use feather_gfx::{Buffer, MappedBuffer, Renderer, FRAMES_IN_FLIGHT};
use glam::{Mat4, Vec4};

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

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

/// Per-instance data uploaded to the storage buffer (std430: 80 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct InstanceData {
    pub model: Mat4,
    pub color: Vec4,
}

const INSTANCE_SIZE: u64 = std::mem::size_of::<InstanceData>() as u64; // 80

fn as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
    }
}

pub struct MeshRenderer {
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    instance_buffers: Vec<MappedBuffer>,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    slices: Vec<MeshSlice>,
    // Reused each frame to gather sorted instances before the SSBO upload.
    scratch: Vec<InstanceData>,
    max_instances: u32,
}

impl MeshRenderer {
    /// Registers `meshes` into shared vertex/index buffers. Returns the renderer
    /// and a `MeshId` per input mesh (same order). `meshes` must be non-empty.
    pub fn new(renderer: &Renderer, meshes: &[MeshData], max_instances: u32) -> (Self, Vec<MeshId>) {
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

        let instance_buffers: Vec<MappedBuffer> = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                renderer.create_host_visible_buffer(
                    max_instances as u64 * INSTANCE_SIZE,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                )
            })
            .collect();

        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .expect("descriptor set layout")
        };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(FRAMES_IN_FLIGHT as u32)];
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
        for (i, &set) in sets.iter().enumerate() {
            let info = [vk::DescriptorBufferInfo::default()
                .buffer(instance_buffers[i].handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info);
            unsafe { device.update_descriptor_sets(&[write], &[]) };
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
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
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
            .size(80)]; // mat4 view_proj + vec4 light_dir
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
        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
        }

        let renderer = Self {
            device,
            layout,
            pipeline,
            set_layout,
            pool,
            sets,
            instance_buffers,
            vertex_buffer,
            index_buffer,
            slices,
            scratch: Vec::new(),
            max_instances,
        };
        (renderer, ids)
    }

    /// Draws all `items` (mesh + instance). Sorts them by mesh in place, uploads
    /// the instances in that order, then emits one `cmd_draw_indexed` per
    /// contiguous same-mesh run.
    pub fn draw(
        &mut self,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        frame: usize,
        view_proj: Mat4,
        light_dir: Vec4,
        items: &mut [(MeshId, InstanceData)],
    ) {
        // Contiguous runs per mesh -> one draw each. Unstable sort is fine; draw
        // order within a mesh doesn't matter (opaque + depth test).
        items.sort_unstable_by_key(|(mesh, _)| mesh.0);

        let count = (items.len()).min(self.max_instances as usize);
        self.scratch.clear();
        self.scratch
            .extend(items[..count].iter().map(|(_, inst)| *inst));
        self.instance_buffers[frame].write(as_bytes(&self.scratch));

        // Push constant: mat4 view_proj (16 f32) + vec4 light_dir (4 f32) = 80 bytes.
        let mut push = [0f32; 20];
        push[..16].copy_from_slice(&view_proj.to_cols_array());
        push[16..].copy_from_slice(&light_dir.to_array());
        let push_bytes = unsafe { std::slice::from_raw_parts(push.as_ptr() as *const u8, 80) };

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

            self.device
                .cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer.handle], &[0]);
            self.device
                .cmd_bind_index_buffer(cmd, self.index_buffer.handle, 0, vk::IndexType::UINT32);

            // One draw per contiguous run of the same mesh. firstInstance is the
            // run's start, so gl_InstanceIndex indexes the sorted instance SSBO.
            let mut first = 0usize;
            while first < count {
                let mesh = items[first].0;
                let mut run = 1usize;
                while first + run < count && items[first + run].0 == mesh {
                    run += 1;
                }
                if let Some(slice) = self.slices.get(mesh.0 as usize) {
                    self.device.cmd_draw_indexed(
                        cmd,
                        slice.index_count,
                        run as u32,
                        slice.first_index,
                        slice.vertex_offset,
                        first as u32,
                    );
                }
                first += run;
            }
        }
    }
}

impl Drop for MeshRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
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
