//! Sky background pass. A far-plane fullscreen triangle drawn in the geometry
//! pass **after** the opaque meshes (§10 step 6), filling the background with the
//! procedural sky the mesh shader reflects for IBL. It depth-tests against the
//! geometry (LESS_OR_EQUAL, no write), so it shades only pixels the geometry
//! didn't cover rather than the whole screen and then being overdrawn.
//!
//! No descriptors — the inverse view-projection, camera position, and light
//! direction come in through a push constant.

use ash::vk;
use feather_gfx::Renderer;
use glam::{Mat4, Vec3, Vec4};

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

pub struct SkyPass {
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl SkyPass {
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device();

        let vert = load_shader(&device, spv!("fullscreen.vert"));
        let frag = load_shader(&device, spv!("sky.frag"));
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

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
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
        // Must match the geometry pass's HDR/depth sample count.
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(renderer.samples());
        // Drawn after the opaque geometry (§10 step 6). The triangle sits on the far
        // plane (z = 1.0), so LESS_OR_EQUAL keeps it only where the depth buffer is
        // still the cleared 1.0 — background pixels — and rejects it wherever
        // geometry wrote nearer depth. That way the sky shades only what it is
        // actually visible through, instead of the whole screen. No depth write:
        // the sky is a backdrop, not an occluder.
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
        let blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);

        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(96)]; // mat4 inv_view_proj + vec4 camera_pos + vec4 light_dir
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges),
                    None,
                )
                .expect("sky pipeline layout")
        };

        // Must match the geometry pass attachments: HDR color + depth.
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
                .expect("sky pipeline")[0]
        };
        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
        }

        Self {
            device,
            layout,
            pipeline,
        }
    }

    /// Fills the *background* with the sky. Call inside the geometry pass **after**
    /// the depth prepass and opaque draws — the depth test then rejects it wherever
    /// geometry is, so only visible background pixels are shaded.
    /// `inv_view_proj` is the inverse of the same matrix the meshes use.
    pub fn draw(
        &self,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        inv_view_proj: Mat4,
        camera_pos: Vec3,
        light_dir: Vec4,
    ) {
        let mut push = [0f32; 24];
        push[..16].copy_from_slice(&inv_view_proj.to_cols_array());
        push[16..19].copy_from_slice(&camera_pos.to_array());
        push[20..].copy_from_slice(&light_dir.to_array());
        let push_bytes = unsafe { std::slice::from_raw_parts(push.as_ptr() as *const u8, 96) };

        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                push_bytes,
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
            self.device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }
}

impl Drop for SkyPass {
    fn drop(&mut self) {
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
