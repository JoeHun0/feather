//! Bloom (§13): physically based, after Jimenez 2014 ("next-generation post
//! processing in Call of Duty: Advanced Warfare"). No threshold: the whole
//! HDR image is filtered down a mip chain and back up, and the tonemap mixes
//! the result in at a small strength, so things glow in proportion to how
//! bright they are and dark ones stay crisp.
//!
//! Compute over the engine-owned bloom chain (`Renderer::bloom_mips`), which
//! `draw_frame` hands over in GENERAL layout:
//! - down: HDR → level 0 → … → level N−1, a 13-tap filter (the first step a
//!   Karis average, against fireflies);
//! - up: level N−1 → … → level 0, each level adding a tent-filtered copy of
//!   the one below it, so level 0 ends as the sum of all N levels.
//!
//! Every step has its own descriptor set per frame in flight, re-pointed when
//! the views change (resize), like `TonemapPass::update`.

use ash::vk;
use feather_gfx::{Renderer, BLOOM_MAX_MIPS, FRAMES_IN_FLIGHT};

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

/// How strongly the tonemap mixes bloom in: the reference value, a veil
/// rather than a glare.
pub const BLOOM_STRENGTH: f32 = 0.04;

/// Down steps plus up steps, the most a chain can need.
const MAX_STEPS: usize = 2 * BLOOM_MAX_MIPS - 1;

/// Workgroup side, matching `local_size_x/y` in both shaders.
const GROUP: u32 = 8;

pub struct BloomPass {
    device: ash::Device,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    down: vk::Pipeline,
    up: vk::Pipeline,
    pool: vk::DescriptorPool,
    /// `sets[frame][step]`.
    sets: Vec<Vec<vk::DescriptorSet>>,
    /// What each frame's sets point at: the HDR view, then the level views.
    bound: Vec<Vec<vk::ImageView>>,
}

/// One step of the chain: read `src`, write `dst` (level indices; `None` as
/// the source is the HDR image).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Step {
    up: bool,
    src: Option<usize>,
    dst: usize,
}

/// The steps for a chain of `levels` levels, in recording order.
fn steps(levels: usize) -> Vec<Step> {
    let down = (0..levels).map(|i| Step {
        up: false,
        src: i.checked_sub(1),
        dst: i,
    });
    let up = (0..levels.saturating_sub(1)).rev().map(|i| Step {
        up: true,
        src: Some(i + 1),
        dst: i,
    });
    down.chain(up).collect()
}

impl BloomPass {
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device();
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .expect("bloom set layout")
        };
        let sets_total = (FRAMES_IN_FLIGHT * MAX_STEPS) as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(sets_total),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(sets_total),
        ];
        let pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(sets_total)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .expect("bloom descriptor pool")
        };
        let sets = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                let layouts = vec![set_layout; MAX_STEPS];
                unsafe {
                    device
                        .allocate_descriptor_sets(
                            &vk::DescriptorSetAllocateInfo::default()
                                .descriptor_pool(pool)
                                .set_layouts(&layouts),
                        )
                        .expect("allocate bloom sets")
                }
            })
            .collect();

        // vec2 src_texel + uint karis (padded to 16).
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        let set_layouts = [set_layout];
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push_ranges),
                    None,
                )
                .expect("bloom pipeline layout")
        };
        let compute = |bytes: &[u8], what: &str| {
            let module = load_shader(&device, bytes);
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(module)
                        .name(c"main"),
                )
                .layout(layout);
            let pipeline = unsafe {
                device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                    .map_err(|(_, e)| e)
                    .expect(what)[0]
            };
            unsafe { device.destroy_shader_module(module, None) };
            pipeline
        };
        let down = compute(spv!("bloom_down.comp"), "bloom down pipeline");
        let up = compute(spv!("bloom_up.comp"), "bloom up pipeline");

        Self {
            device,
            set_layout,
            layout,
            down,
            up,
            pool,
            sets,
            bound: vec![Vec::new(); FRAMES_IN_FLIGHT],
        }
    }

    /// Point this frame's step sets at the current HDR view and bloom levels.
    /// No-op when nothing changed (views only change on resize). Safe each
    /// frame: `draw_frame` waits on the frame fence before recording.
    pub fn update(
        &mut self,
        frame: usize,
        hdr_view: vk::ImageView,
        levels: &[vk::ImageView],
        sampler: vk::Sampler,
    ) {
        let key: Vec<vk::ImageView> = std::iter::once(hdr_view)
            .chain(levels.iter().copied())
            .collect();
        if self.bound[frame] == key {
            return;
        }
        for (n, step) in steps(levels.len()).iter().enumerate() {
            let src = match step.src {
                None => [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(hdr_view)
                    .sampler(sampler)],
                Some(i) => [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(levels[i])
                    .sampler(sampler)],
            };
            let dst = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(levels[step.dst])];
            let set = self.sets[frame][n];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&src),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&dst),
            ];
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }
        self.bound[frame] = key;
    }

    /// Record the chain. `window` is the HDR image's size, `levels` the
    /// bloom levels' sizes (as `Renderer::bloom_mips` gives them). Each step
    /// waits for the one before (a compute → compute barrier).
    pub fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        frame: usize,
        window: vk::Extent2D,
        levels: &[vk::Extent2D],
    ) {
        let size = |e: vk::Extent2D| [1.0 / e.width as f32, 1.0 / e.height as f32];
        for (n, step) in steps(levels.len()).iter().enumerate() {
            let src = step.src.map_or(window, |i| levels[i]);
            let dst = levels[step.dst];
            let [tx, ty] = size(src);
            let karis = u32::from(step.src.is_none());
            let mut push = [0u8; 16];
            push[0..4].copy_from_slice(&tx.to_ne_bytes());
            push[4..8].copy_from_slice(&ty.to_ne_bytes());
            push[8..12].copy_from_slice(&karis.to_ne_bytes());
            let pipeline = if step.up { self.up } else { self.down };
            unsafe {
                if n > 0 {
                    // The previous step's writes, before this one reads them
                    // (and, going up, before it reads and rewrites its level).
                    let barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        );
                    self.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                }
                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[self.sets[frame][n]],
                    &[],
                );
                self.device.cmd_push_constants(
                    cmd,
                    self.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push,
                );
                self.device.cmd_dispatch(
                    cmd,
                    dst.width.div_ceil(GROUP),
                    dst.height.div_ceil(GROUP),
                    1,
                );
            }
        }
    }
}

impl Drop for BloomPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.down, None);
            self.device.destroy_pipeline(self.up, None);
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

#[cfg(test)]
mod tests {
    use super::*;

    // The reference's filter weights, which the shaders must declare.
    const BOX_CENTRE: f32 = 0.5;
    const BOX_CORNER: f32 = 0.125;
    const TENT: [f32; 3] = [0.25, 0.125, 0.0625]; // centre, edge, corner

    /// A single-channel image, row-major.
    #[derive(Clone)]
    struct Img {
        w: usize,
        h: usize,
        px: Vec<f32>,
    }

    impl Img {
        /// Bilinear, clamp-to-edge, at `uv` (0..1, texel centres at +0.5):
        /// what the shaders' linear sampler does.
        fn sample(&self, u: f32, v: f32) -> f32 {
            let x = (u * self.w as f32 - 0.5).clamp(0.0, self.w as f32 - 1.0);
            let y = (v * self.h as f32 - 0.5).clamp(0.0, self.h as f32 - 1.0);
            let (x0, y0) = (x.floor() as usize, y.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
            let (fx, fy) = (x - x0 as f32, y - y0 as f32);
            let at = |x: usize, y: usize| self.px[y * self.w + x];
            let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
            let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
            top * (1.0 - fy) + bottom * fy
        }

        fn sum(&self) -> f32 {
            self.px.iter().sum()
        }
    }

    /// `bloom_down.comp`, one channel, with luma = the value.
    fn down(src: &Img, w: usize, h: usize, karis: bool) -> Img {
        let (tx, ty) = (1.0 / src.w as f32, 1.0 / src.h as f32);
        let mut px = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                let t = |dx: f32, dy: f32| src.sample(u + dx * tx, v + dy * ty);
                let (a, b, c) = (t(-2.0, 2.0), t(0.0, 2.0), t(2.0, 2.0));
                let (d, e, f) = (t(-2.0, 0.0), t(0.0, 0.0), t(2.0, 0.0));
                let (g, hh, i) = (t(-2.0, -2.0), t(0.0, -2.0), t(2.0, -2.0));
                let (j, k, l, m) = (t(-1.0, 1.0), t(1.0, 1.0), t(-1.0, -1.0), t(1.0, -1.0));
                let boxes = [
                    (j + k + l + m) / 4.0,
                    (a + b + d + e) / 4.0,
                    (b + c + e + f) / 4.0,
                    (d + e + g + hh) / 4.0,
                    (e + f + hh + i) / 4.0,
                ];
                let (mut sum, mut total) = (0.0, 0.0);
                for (n, bx) in boxes.iter().enumerate() {
                    let mut wgt = if n == 0 { BOX_CENTRE } else { BOX_CORNER };
                    if karis {
                        wgt /= 1.0 + bx;
                    }
                    sum += bx * wgt;
                    total += wgt;
                }
                px[y * w + x] = sum / total;
            }
        }
        Img { w, h, px }
    }

    /// `bloom_up.comp`: `dst += tent(src)`.
    fn up(src: &Img, dst: &mut Img) {
        let (tx, ty) = (1.0 / src.w as f32, 1.0 / src.h as f32);
        for y in 0..dst.h {
            for x in 0..dst.w {
                let (u, v) = (
                    (x as f32 + 0.5) / dst.w as f32,
                    (y as f32 + 0.5) / dst.h as f32,
                );
                let t = |dx: f32, dy: f32| src.sample(u + dx * tx, v + dy * ty);
                let add = t(0.0, 0.0) * TENT[0]
                    + (t(-1.0, 0.0) + t(1.0, 0.0) + t(0.0, -1.0) + t(0.0, 1.0)) * TENT[1]
                    + (t(-1.0, -1.0) + t(1.0, -1.0) + t(-1.0, 1.0) + t(1.0, 1.0)) * TENT[2];
                dst.px[y * dst.w + x] += add;
            }
        }
    }

    /// The whole chain on `hdr` with `levels` levels, as the passes run it:
    /// level 0 after the up passes, divided by the level count as the
    /// tonemap does.
    fn bloom(hdr: &Img, levels: usize) -> Img {
        let mut chain: Vec<Img> = Vec::new();
        for step in steps(levels) {
            if step.up {
                let src = chain[step.src.unwrap()].clone();
                up(&src, &mut chain[step.dst]);
            } else {
                let src = step.src.map_or(hdr, |i| &chain[i]);
                let (w, h) = ((src.w / 2).max(1), (src.h / 2).max(1));
                let next = down(src, w, h, step.src.is_none());
                chain.push(next);
            }
        }
        let mut out = chain.swap_remove(0);
        for p in &mut out.px {
            *p /= levels as f32;
        }
        out
    }

    #[test]
    fn steps_go_down_the_chain_then_back_up() {
        let s = steps(3);
        let got: Vec<(bool, Option<usize>, usize)> =
            s.iter().map(|s| (s.up, s.src, s.dst)).collect();
        assert_eq!(
            got,
            [
                (false, None, 0),
                (false, Some(0), 1),
                (false, Some(1), 2),
                (true, Some(2), 1),
                (true, Some(1), 0),
            ]
        );
        assert_eq!(steps(BLOOM_MAX_MIPS).len(), MAX_STEPS);
        assert_eq!(steps(1).len(), 1);
    }

    /// Both filters' weights sum to 1, so a flat image blooms to itself: no
    /// brightening and no halo anywhere, edges included (the sampler clamps).
    /// That is what makes mixing bloom in energy-conserving.
    #[test]
    fn a_flat_image_blooms_to_itself() {
        assert!((BOX_CENTRE + 4.0 * BOX_CORNER - 1.0).abs() < 1e-6);
        assert!((TENT[0] + 4.0 * TENT[1] + 4.0 * TENT[2] - 1.0).abs() < 1e-6);
        let flat = Img {
            w: 64,
            h: 40,
            px: vec![3.0; 64 * 40],
        };
        let b = bloom(&flat, 4);
        assert!(
            b.px.iter().all(|&p| (p - 3.0).abs() < 1e-4),
            "{:?}",
            &b.px[..8]
        );
    }

    /// A single bright pixel spreads into a wide, smooth glow that falls off
    /// with distance, and the chain roughly keeps its energy (bilinear
    /// resampling and the Karis weighting shift it a little).
    #[test]
    fn a_bright_pixel_spreads_and_keeps_its_energy() {
        let (w, h) = (128, 128);
        let mut px = vec![0.0; w * h];
        px[64 * w + 64] = 100.0;
        let hdr = Img { w, h, px };
        let b = bloom(&hdr, 5);
        // Level 0 is half size: 64 x 64, the pixel near (32, 32).
        let at = |x: usize, y: usize| b.px[y * b.w + x];
        assert!(
            at(32, 32) > at(36, 32) && at(36, 32) > at(44, 32),
            "no falloff"
        );
        assert!(at(44, 32) > 0.0, "the glow doesn't reach 12 texels");
        // Level 0 is a quarter of the pixels, so compare per-area energy.
        let energy = b.sum() * 4.0;
        // The Karis weighting takes energy from a lone bright pixel on
        // purpose (that is what stops fireflies); without it the chain keeps
        // it within a few percent.
        let plain = {
            let (w2, h2) = (w / 2, h / 2);
            let l0 = down(&hdr, w2, h2, false);
            l0.sum() * 4.0
        };
        assert!(
            (plain - 100.0).abs() < 5.0,
            "plain downsample energy {plain}"
        );
        assert!(
            energy > 0.0 && energy <= 100.0 * 1.05,
            "chain energy {energy}"
        );
    }

    /// The shaders' weights are the reference's.
    #[test]
    fn the_shaders_use_the_reference_weights() {
        let down = include_str!("../shaders/bloom_down.comp");
        let up = include_str!("../shaders/bloom_up.comp");
        for decl in [
            format!("const float BOX_CENTRE = {BOX_CENTRE:?};"),
            format!("const float BOX_CORNER = {BOX_CORNER:?};"),
        ] {
            assert!(down.contains(&decl), "bloom_down.comp lacks `{decl}`");
        }
        for decl in [
            format!("const float TENT_CENTRE = {:?};", TENT[0]),
            format!("const float TENT_EDGE = {:?};", TENT[1]),
            format!("const float TENT_CORNER = {:?};", TENT[2]),
        ] {
            assert!(up.contains(&decl), "bloom_up.comp lacks `{decl}`");
        }
    }
}
