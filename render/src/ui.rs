//! Immediate-mode 2D overlay (§19's "lightweight custom HUD renderer"). One
//! pipeline drawing alpha-blended textured quads into the swapchain **after**
//! tonemap/FXAA, i.e. in LDR/sRGB, which is where §13 and §19 both put UI.
//!
//! Text comes from `font::Font`: DejaVu Sans Mono rasterized once at startup
//! into an R8 coverage atlas (see font.rs for the license). The atlas has a
//! blank padding border around every glyph, so LINEAR sampling stays clean at
//! any scale — rectangles and text still share the one atlas/draw, the solid
//! texel included. This is deliberately minimal — enough for a pause menu and
//! a future HUD, without committing to egui (which §19 reserves for the *dev*
//! UI).

use ash::vk;
use feather_gfx::{Image, MappedBuffer, Renderer, FRAMES_IN_FLIGHT};

use crate::font::Font;

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

/// Raster size the font is cached at; `size / raster_em` is the screen scale.
const RASTER_EM: u32 = 18;
const MAX_QUADS: usize = 1024;

#[repr(C)]
#[derive(Clone, Copy)]
struct UiVertex {
    pos: [f32; 2], // NDC
    uv: [f32; 2],
    color: [f32; 4],
}

const VERTEX_SIZE: u64 = std::mem::size_of::<UiVertex>() as u64;

fn as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

pub struct UiPass {
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    buffers: Vec<MappedBuffer>,
    #[allow(dead_code)]
    atlas: Image,
    sampler: vk::Sampler,
    verts: Vec<UiVertex>,
    screen: [f32; 2],
    font: Font,
}

impl UiPass {
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device();

        // Rasterize the bundled font once, replicate coverage into RGBA8 so
        // the existing create_texture upload path can be reused.
        let font = Font::rasterize(include_bytes!("../fonts/DejaVuSansMono.ttf"), RASTER_EM)
            .expect("the bundled UI font rasterizes");
        let mut pixels = vec![0u8; font.width * font.height * 4];
        for (i, &cov) in font.atlas.iter().enumerate() {
            pixels[i * 4..i * 4 + 4].copy_from_slice(&[255, 255, 255, cov]);
        }
        let atlas = renderer.create_texture(&pixels, font.width as u32, font.height as u32, false);
        // LINEAR: the atlas's per-glyph padding borders make sure no
        // neighbour bleeds in, and small text stays smooth.
        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .expect("ui sampler")
        };

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
                .expect("ui set layout")
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
                .expect("ui descriptor pool")
        };
        let layouts = vec![set_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
                .expect("allocate ui sets")
        };
        // The atlas never changes, so bind it once for every frame's set.
        for &set in &sets {
            let info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(atlas.view)
                .sampler(sampler)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info);
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }

        let buffers: Vec<MappedBuffer> = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                renderer.create_host_visible_buffer(
                    MAX_QUADS as u64 * 6 * VERTEX_SIZE,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                )
            })
            .collect();

        let vert = load_shader(&device, spv!("ui.vert"));
        let frag = load_shader(&device, spv!("ui.frag"));
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
            .stride(VERTEX_SIZE as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let vattrs = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(8),
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .binding(0)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(16),
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
        // Drawn into the swapchain, which is always single-sample.
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false);
        // Standard straight-alpha blending over whatever the post chain left.
        let blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);
        let set_layouts = [set_layout];
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )
                .expect("ui pipeline layout")
        };
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
                .expect("ui pipeline")[0]
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
            buffers,
            atlas,
            sampler,
            verts: Vec::new(),
            screen: [1.0, 1.0],
            font,
        }
    }

    /// Start a frame's overlay. Coordinates passed to `rect`/`text` afterwards are
    /// in pixels with the origin at the top-left.
    pub fn begin(&mut self, extent_w: u32, extent_h: u32) {
        self.verts.clear();
        self.screen = [extent_w.max(1) as f32, extent_h.max(1) as f32];
    }

    pub fn is_empty(&self) -> bool {
        self.verts.is_empty()
    }

    /// The font driving `text`, for callers that lay text out before drawing
    /// (`menu_layout` sizes its rows from the same advances the GPU will use).
    pub fn font(&self) -> &Font {
        &self.font
    }

    /// Width in pixels that `text` will occupy at `size` pixels per em.
    pub fn text_width(&self, text: &str, size: f32) -> f32 {
        self.font.width(text, size)
    }

    /// One line's pixel height at `size` pixels per em.
    pub fn text_height(&self, size: f32) -> f32 {
        self.font.height(size)
    }

    fn quad(&mut self, x: f32, y: f32, w: f32, h: f32, uv0: [f32; 2], uv1: [f32; 2], c: [f32; 4]) {
        if self.verts.len() + 6 > MAX_QUADS * 6 {
            return; // silently drop rather than overrun the frame's buffer
        }
        let (sw, sh) = (self.screen[0], self.screen[1]);
        // Pixels (top-left origin) -> NDC. Vulkan's NDC y already points down.
        let ndc = |px: f32, py: f32| [px / sw * 2.0 - 1.0, py / sh * 2.0 - 1.0];
        let (p0, p1) = (ndc(x, y), ndc(x + w, y + h));
        let v = |p: [f32; 2], uv: [f32; 2]| UiVertex {
            pos: p,
            uv,
            color: c,
        };
        let tl = v([p0[0], p0[1]], [uv0[0], uv0[1]]);
        let tr = v([p1[0], p0[1]], [uv1[0], uv0[1]]);
        let br = v([p1[0], p1[1]], [uv1[0], uv1[1]]);
        let bl = v([p0[0], p1[1]], [uv0[0], uv1[1]]);
        self.verts.extend_from_slice(&[tl, tr, br, tl, br, bl]);
    }

    /// Filled rectangle. `color` is **linear** — this pass writes an _SRGB target.
    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        // Sample the centre of the solid cell, so filtering cannot reach a
        // neighbouring glyph.
        let u = self.font.solid;
        self.quad(x, y, w, h, u, u, color);
    }

    /// Draw `text` with the top of its line box at `(x, y)`, at `size` pixels
    /// per em. `color` is linear. Characters outside printable ASCII are
    /// skipped (see font.rs).
    pub fn text(&mut self, x: f32, y: f32, size: f32, color: [f32; 4], text: &str) {
        let s = size / self.font.raster_em;
        let baseline = y + self.font.ascent * s;
        let mut pen = x;
        for ch in text.chars() {
            let Some(g) = self.font.glyph(ch).copied() else {
                continue;
            };
            if g.w > 0.0 {
                self.quad(
                    pen + g.min_x * s,
                    baseline + g.min_y * s,
                    g.w * s,
                    g.h * s,
                    [g.u0, g.v0],
                    [g.u1, g.v1],
                    color,
                );
            }
            pen += g.advance * s;
        }
    }

    /// Upload this frame's geometry and record the draw. Call inside the UI pass.
    pub fn draw(&self, cmd: vk::CommandBuffer, extent: vk::Extent2D, frame: usize) {
        if self.verts.is_empty() {
            return;
        }
        self.buffers[frame].write(as_bytes(&self.verts));
        unsafe {
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
                .cmd_bind_vertex_buffers(cmd, 0, &[self.buffers[frame].handle], &[0]);
            self.device.cmd_draw(cmd, self.verts.len() as u32, 1, 0, 0);
        }
    }
}

impl Drop for UiPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
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
