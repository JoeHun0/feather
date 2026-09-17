//! Tonemap pass. A fullscreen triangle that samples the engine's linear HDR
//! scene-color target, applies exposure + an ACES curve, and writes the sRGB
//! swapchain. This is the seam the whole "baseline look" hangs off: bloom, SSR
//! and transparents all read the resolved HDR here later.
//!
//! It owns one descriptor set per frame-in-flight pointing at the HDR image.
//! The HDR view changes on resize, so `update` re-points the descriptor when the
//! handle changes (cheap: skipped when unchanged).

use ash::vk;
use feather_gfx::{Renderer, FRAMES_IN_FLIGHT};

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

pub struct TonemapPass {
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    // Currently-bound HDR view per set; used to skip redundant descriptor writes.
    bound_views: Vec<vk::ImageView>,
}

impl TonemapPass {
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device();

        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .expect("tonemap set layout")
        };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(FRAMES_IN_FLIGHT as u32)];
        let pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(FRAMES_IN_FLIGHT as u32)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .expect("tonemap descriptor pool")
        };

        let layouts = vec![set_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
                .expect("allocate tonemap sets")
        };

        let vert = load_shader(&device, spv!("fullscreen.vert"));
        let frag = load_shader(&device, spv!("tonemap.frag"));
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

        // No vertex input: the vertex shader generates the triangle from the index.
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
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false);
        let blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);

        let set_layouts = [set_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(4)]; // float exposure
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push_ranges),
                    None,
                )
                .expect("tonemap pipeline layout")
        };

        // Writes the swapchain; no depth attachment in this pass.
        let color_formats = [renderer.color_format()];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

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
                .expect("tonemap pipeline")[0]
        };
        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
        }

        Self {
            device,
            layout,
            pipeline,
            set_layout,
            pool,
            sets,
            bound_views: vec![vk::ImageView::null(); FRAMES_IN_FLIGHT],
        }
    }

    /// Point this frame's descriptor at the current HDR view. No-op if unchanged
    /// (the view only changes on resize). Safe to call each frame: `draw_frame`
    /// waits on the frame fence before recording, so this set is idle here.
    pub fn update(&mut self, frame: usize, hdr_view: vk::ImageView, sampler: vk::Sampler) {
        if self.bound_views[frame] == hdr_view {
            return;
        }
        let info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(hdr_view)
            .sampler(sampler)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.sets[frame])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        self.bound_views[frame] = hdr_view;
    }

    pub fn draw(&self, cmd: vk::CommandBuffer, extent: vk::Extent2D, frame: usize, exposure: f32) {
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[self.sets[frame]],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                &exposure.to_ne_bytes(),
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

impl Drop for TonemapPass {
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
