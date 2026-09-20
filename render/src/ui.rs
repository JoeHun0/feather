//! Immediate-mode 2D overlay (§19's "lightweight custom HUD renderer"). One
//! pipeline drawing alpha-blended textured quads into the swapchain **after**
//! tonemap/FXAA, i.e. in LDR/sRGB, which is where §13 and §19 both put UI.
//!
//! The atlas is a 5x7 pixel font plus a single solid texel, so rectangles and
//! text share one draw. Nearest sampling keeps the font crisp when scaled up.
//! This is deliberately minimal — enough for a pause menu and a future HUD,
//! without committing to egui (which §19 reserves for the *dev* UI).

use ash::vk;
use feather_gfx::{Image, MappedBuffer, Renderer, FRAMES_IN_FLIGHT};

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

const GLYPH_W: usize = 5;
const GLYPH_H: usize = 7;
/// Slot 0 is solid, slot 1 is blank, then A-Z, then 0-9.
const SLOTS: usize = 38;
const MAX_QUADS: usize = 1024;

/// 5x7 glyphs, one byte per row, bit 4 = leftmost pixel. Verified legible by
/// rendering the table to ASCII rather than by eye on screen.
const GLYPHS: [[u8; GLYPH_H]; SLOTS] = [
    [0x1F; GLYPH_H],                            // solid block (rectangles)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // space
    [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11], // A
    [0x1E, 0x11, 0x1E, 0x11, 0x11, 0x11, 0x1E], // B
    [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E], // C
    [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E], // D
    [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F], // E
    [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10], // F
    [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E], // G
    [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11], // H
    [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1F], // I
    [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C], // J
    [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11], // K
    [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F], // L
    [0x11, 0x1B, 0x15, 0x11, 0x11, 0x11, 0x11], // M
    [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11], // N
    [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E], // O
    [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10], // P
    [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D], // Q
    [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11], // R
    [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E], // S
    [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04], // T
    [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E], // U
    [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04], // V
    [0x11, 0x11, 0x11, 0x11, 0x15, 0x1B, 0x11], // W
    [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11], // X
    [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04], // Y
    [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F], // Z
    [0x0E, 0x13, 0x15, 0x19, 0x11, 0x11, 0x0E], // 0
    [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x1F], // 1
    [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F], // 2
    [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E], // 3
    [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02], // 4
    [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E], // 5
    [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E], // 6
    [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08], // 7
    [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E], // 8
    [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C], // 9
];

fn slot_of(c: char) -> usize {
    match c.to_ascii_uppercase() {
        'A'..='Z' => 2 + (c.to_ascii_uppercase() as usize - 'A' as usize),
        '0'..='9' => 28 + (c as usize - '0' as usize),
        _ => 1, // blank
    }
}

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
}

impl UiPass {
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device();

        // Atlas: SLOTS glyphs side by side, coverage replicated into RGBA8 so the
        // existing create_texture upload path can be reused.
        let aw = SLOTS * GLYPH_W;
        let mut pixels = vec![0u8; aw * GLYPH_H * 4];
        for (slot, glyph) in GLYPHS.iter().enumerate() {
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..GLYPH_W {
                    let on = bits & (1 << (GLYPH_W - 1 - col)) != 0;
                    let idx = (row * aw + slot * GLYPH_W + col) * 4;
                    pixels[idx..idx + 4].copy_from_slice(&[
                        255,
                        255,
                        255,
                        if on { 255 } else { 0 },
                    ]);
                    if !on {
                        pixels[idx] = 0;
                    }
                }
            }
        }
        let atlas = renderer.create_texture(&pixels, aw as u32, GLYPH_H as u32, false);
        // Nearest: a pixel font scaled up should stay crisp, and it avoids
        // bleeding between neighbouring glyphs in the atlas.
        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
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

    /// Width in pixels that `text` will occupy at `px` per font pixel.
    pub fn text_width(text: &str, px: f32) -> f32 {
        (text.chars().count() * (GLYPH_W + 1)) as f32 * px
    }

    pub fn text_height(px: f32) -> f32 {
        GLYPH_H as f32 * px
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
        // Sample the middle of the solid slot, so filtering cannot reach a
        // neighbouring glyph.
        let aw = (SLOTS * GLYPH_W) as f32;
        let u = 2.5 / aw;
        self.quad(
            x,
            y,
            w,
            h,
            [u, 0.5 / GLYPH_H as f32],
            [u, 0.5 / GLYPH_H as f32],
            color,
        );
    }

    /// Draw `text` with its top-left at `(x, y)`, one font pixel = `px` screen
    /// pixels. `color` is linear.
    pub fn text(&mut self, x: f32, y: f32, px: f32, color: [f32; 4], text: &str) {
        let aw = (SLOTS * GLYPH_W) as f32;
        let mut cx = x;
        for ch in text.chars() {
            let slot = slot_of(ch) as f32;
            let u0 = slot * GLYPH_W as f32 / aw;
            let u1 = (slot + 1.0) * GLYPH_W as f32 / aw;
            self.quad(
                cx,
                y,
                GLYPH_W as f32 * px,
                GLYPH_H as f32 * px,
                [u0, 0.0],
                [u1, 1.0],
                color,
            );
            cx += (GLYPH_W + 1) as f32 * px;
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
