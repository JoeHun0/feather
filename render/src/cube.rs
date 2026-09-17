//! Cube pipeline, now backed by real vertex/index buffers (vk-mem) and
//! depth-tested. Geometry is still a hardcoded cube; loaded meshes come next.

use ash::vk;
use feather_gfx::{Buffer, Renderer};
use glam::Mat4;

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
}

const RED: [f32; 3] = [1.0, 0.0, 0.0];
const GREEN: [f32; 3] = [0.0, 1.0, 0.0];
const BLUE: [f32; 3] = [0.0, 0.0, 1.0];
const YELLOW: [f32; 3] = [1.0, 1.0, 0.0];
const CYAN: [f32; 3] = [0.0, 1.0, 1.0];
const MAGENTA: [f32; 3] = [1.0, 0.0, 1.0];

// 6 faces x 4 corners, each corner wound so the quad (0,1,2,0,2,3) faces outward.
#[rustfmt::skip]
const VERTICES: [Vertex; 24] = [
    // +X (red)
    Vertex { pos: [ 0.5,-0.5,-0.5], color: RED }, Vertex { pos: [ 0.5, 0.5,-0.5], color: RED },
    Vertex { pos: [ 0.5, 0.5, 0.5], color: RED }, Vertex { pos: [ 0.5,-0.5, 0.5], color: RED },
    // -X (green)
    Vertex { pos: [-0.5,-0.5,-0.5], color: GREEN }, Vertex { pos: [-0.5,-0.5, 0.5], color: GREEN },
    Vertex { pos: [-0.5, 0.5, 0.5], color: GREEN }, Vertex { pos: [-0.5, 0.5,-0.5], color: GREEN },
    // +Y (blue)
    Vertex { pos: [-0.5, 0.5,-0.5], color: BLUE }, Vertex { pos: [-0.5, 0.5, 0.5], color: BLUE },
    Vertex { pos: [ 0.5, 0.5, 0.5], color: BLUE }, Vertex { pos: [ 0.5, 0.5,-0.5], color: BLUE },
    // -Y (yellow)
    Vertex { pos: [-0.5,-0.5,-0.5], color: YELLOW }, Vertex { pos: [ 0.5,-0.5,-0.5], color: YELLOW },
    Vertex { pos: [ 0.5,-0.5, 0.5], color: YELLOW }, Vertex { pos: [-0.5,-0.5, 0.5], color: YELLOW },
    // +Z (cyan)
    Vertex { pos: [-0.5,-0.5, 0.5], color: CYAN }, Vertex { pos: [ 0.5,-0.5, 0.5], color: CYAN },
    Vertex { pos: [ 0.5, 0.5, 0.5], color: CYAN }, Vertex { pos: [-0.5, 0.5, 0.5], color: CYAN },
    // -Z (magenta)
    Vertex { pos: [-0.5,-0.5,-0.5], color: MAGENTA }, Vertex { pos: [-0.5, 0.5,-0.5], color: MAGENTA },
    Vertex { pos: [ 0.5, 0.5,-0.5], color: MAGENTA }, Vertex { pos: [ 0.5,-0.5,-0.5], color: MAGENTA },
];

#[rustfmt::skip]
const INDICES: [u16; 36] = [
     0, 1, 2,  0, 2, 3,   // +X
     4, 5, 6,  4, 6, 7,   // -X
     8, 9,10,  8,10,11,   // +Y
    12,13,14, 12,14,15,   // -Y
    16,17,18, 16,18,19,   // +Z
    20,21,22, 20,22,23,   // -Z
];

fn as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
    }
}

pub struct CubePipeline {
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
}

impl CubePipeline {
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device();
        let vertex_buffer =
            renderer.create_device_local_buffer(as_bytes(&VERTICES), vk::BufferUsageFlags::VERTEX_BUFFER);
        let index_buffer =
            renderer.create_device_local_buffer(as_bytes(&INDICES), vk::BufferUsageFlags::INDEX_BUFFER);

        let vert = load_shader(&device, spv!("cube.vert"));
        let frag = load_shader(&device, spv!("cube.frag"));

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

        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attributes = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(12), // after pos: 3 * f32
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);

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
            .cull_mode(vk::CullModeFlags::BACK)
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

        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(64)];
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges),
                    None,
                )
                .expect("create pipeline layout")
        };

        let color_formats = [renderer.color_format()];
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
                .expect("create graphics pipeline")[0]
        };

        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
        }

        Self {
            device,
            layout,
            pipeline,
            vertex_buffer,
            index_buffer,
        }
    }

    pub fn draw(&self, cmd: vk::CommandBuffer, extent: vk::Extent2D, mvp: Mat4) {
        let cols = mvp.to_cols_array();
        let bytes = unsafe { std::slice::from_raw_parts(cols.as_ptr() as *const u8, 64) };
        unsafe {
            self.device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytes,
            );
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

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
            self.device.cmd_bind_index_buffer(
                cmd,
                self.index_buffer.handle,
                0,
                vk::IndexType::UINT16,
            );
            self.device.cmd_draw_indexed(cmd, INDICES.len() as u32, 1, 0, 0, 0);
        }
    }
}

impl Drop for CubePipeline {
    fn drop(&mut self) {
        // vertex_buffer / index_buffer free themselves (RAII) after this runs.
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
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
