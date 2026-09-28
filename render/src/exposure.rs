//! Auto-exposure (§13): the eye adapting to the scene's brightness.
//!
//! Two compute passes over the HDR image, after bloom:
//! - `exposure_histogram.comp`: a histogram of log2 luminance over a
//!   quarter-resolution sample grid, each sample weighted by how near the
//!   screen's centre it is (centre-weighted metering: what you look at drives
//!   exposure);
//! - `exposure_average.comp`: the weighted mean between the `LOW_PCT` and
//!   `HIGH_PCT` percentiles, adapted towards over time (faster to bright than
//!   to dark), and turned into an exposure, `KEY / 2^adapted`, clamped to the
//!   level's range.
//!
//! The result stays on the GPU, in a persistent state buffer the tonemap
//! reads (no CPU round trip, so no frames of lag). Each frame also copies it
//! into a small host-visible slot, which the app reads after that frame's
//! fence: the bench's evidence that exposure responds.

use ash::vk;
use feather_gfx::{Buffer, MappedBuffer, Renderer, FRAMES_IN_FLIGHT};

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

/// Histogram bins (bin 0 is black). The shaders' other constants (the log
/// range, percentiles, speeds, `KEY`, centre weighting) live in the shaders,
/// pinned by the reference in this file's tests.
const BINS: usize = 256;
/// Sample grid: one sample per this many pixels a side.
const GRID_STEP: u32 = 4;

/// The persistent state, as the shaders declare it (std430).
#[repr(C)]
struct State {
    histogram: [u32; BINS],
    adapted: f32,
    exposure: f32,
    target: f32,
    frames: u32,
}

/// One frame's copy of the result, for the CPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExposureReadback {
    /// The exposure the tonemap multiplies by (before compensation).
    pub exposure: f32,
    /// This frame's metered log2 luminance, and the adapted one.
    pub target: f32,
    pub adapted: f32,
    _pad: f32,
}

/// What the average pass is told each frame.
#[derive(Clone, Copy, Debug)]
pub struct ExposureParams {
    /// Seconds since the last frame (the app clamps it).
    pub dt: f32,
    /// The level's range for the exposure (`environment`).
    pub min: f32,
    pub max: f32,
    /// Snap to the metered value instead of adapting: a level's first frame.
    pub reset: bool,
}

pub struct ExposurePass {
    device: ash::Device,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    histogram: vk::Pipeline,
    average: vk::Pipeline,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    bound: Vec<vk::ImageView>,
    state: Buffer,
    readback: Vec<MappedBuffer>,
}

impl ExposurePass {
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device();
        let bindings = [
            (0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            (1, vk::DescriptorType::STORAGE_BUFFER),
            (2, vk::DescriptorType::STORAGE_BUFFER),
        ]
        .map(|(b, ty)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .expect("exposure set layout")
        };
        let frames = FRAMES_IN_FLIGHT as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(frames),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2 * frames),
        ];
        let pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(frames)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .expect("exposure descriptor pool")
        };
        let layouts = vec![set_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
                .expect("allocate exposure sets")
        };

        // Zeroed: `frames` = 0 makes the first average snap.
        let state = renderer.create_device_local_buffer(
            &vec![0u8; std::mem::size_of::<State>()],
            vk::BufferUsageFlags::STORAGE_BUFFER,
        );
        let readback: Vec<MappedBuffer> = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                renderer.create_host_visible_buffer(
                    std::mem::size_of::<ExposureReadback>() as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                )
            })
            .collect();
        // The state and readback bindings never change; the HDR view does.
        for (frame, &set) in sets.iter().enumerate() {
            let state_info = [vk::DescriptorBufferInfo::default()
                .buffer(state.handle)
                .range(vk::WHOLE_SIZE)];
            let readback_info = [vk::DescriptorBufferInfo::default()
                .buffer(readback[frame].handle)
                .range(vk::WHOLE_SIZE)];
            let writes = [(1, &state_info), (2, &readback_info)].map(|(b, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            });
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

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
                .expect("exposure pipeline layout")
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
        let histogram = compute(spv!("exposure_histogram.comp"), "exposure histogram");
        let average = compute(spv!("exposure_average.comp"), "exposure average");

        Self {
            device,
            set_layout,
            layout,
            histogram,
            average,
            pool,
            sets,
            bound: vec![vk::ImageView::null(); FRAMES_IN_FLIGHT],
            state,
            readback,
        }
    }

    /// The state buffer, which the tonemap reads its exposure from.
    pub fn state_buffer(&self) -> vk::Buffer {
        self.state.handle
    }

    /// The result this frame slot's last use wrote. Only meaningful after
    /// that frame's fence (`draw_frame` waits on it before calling the
    /// closures) and once the pass has run in that slot.
    pub fn last(&self, frame: usize) -> ExposureReadback {
        let mut out = ExposureReadback::default();
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                (&mut out as *mut ExposureReadback).cast::<u8>(),
                std::mem::size_of::<ExposureReadback>(),
            )
        };
        self.readback[frame].read(bytes);
        out
    }

    /// Point this frame's set at the current HDR view (it changes on
    /// resize). Safe each frame: `draw_frame` waits on the frame fence first.
    pub fn update(&mut self, frame: usize, hdr_view: vk::ImageView, sampler: vk::Sampler) {
        if self.bound[frame] == hdr_view {
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
        self.bound[frame] = hdr_view;
    }

    /// Record both passes over an HDR image of `extent`, with their
    /// barriers: the state is shared across frames in flight, so it first
    /// waits for the previous frame's compute writes and tonemap reads
    /// (§21's rule), and last makes this frame's writes visible to the
    /// tonemap and the readback to the host.
    pub fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        frame: usize,
        extent: vk::Extent2D,
        params: ExposureParams,
    ) {
        let grid = [
            extent.width.div_ceil(GRID_STEP),
            extent.height.div_ceil(GRID_STEP),
        ];
        let aspect = extent.width as f32 / extent.height.max(1) as f32;
        let barrier = |src_stage, src, dst_stage, dst| unsafe {
            let b = vk::MemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst);
            self.device.cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[b],
                &[],
                &[],
            );
        };
        let rw = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
        barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            rw,
        );
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.sets[frame]],
                &[],
            );
            let mut push = [0u8; 16];
            push[0..4].copy_from_slice(&grid[0].to_ne_bytes());
            push[4..8].copy_from_slice(&grid[1].to_ne_bytes());
            push[8..12].copy_from_slice(&aspect.to_ne_bytes());
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.histogram);
            self.device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            self.device
                .cmd_dispatch(cmd, grid[0].div_ceil(16), grid[1].div_ceil(16), 1);
        }
        barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            rw,
        );
        unsafe {
            let mut push = [0u8; 16];
            push[0..4].copy_from_slice(&params.dt.to_ne_bytes());
            push[4..8].copy_from_slice(&params.min.to_ne_bytes());
            push[8..12].copy_from_slice(&params.max.to_ne_bytes());
            push[12..16].copy_from_slice(&u32::from(params.reset).to_ne_bytes());
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.average);
            self.device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            self.device.cmd_dispatch(cmd, 1, 1, 1);
        }
        barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::HOST,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::HOST_READ,
        );
    }
}

impl Drop for ExposurePass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.histogram, None);
            self.device.destroy_pipeline(self.average, None);
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

    // ---- The reference: the shaders' constants and maths, transcribed. ----

    /// The log2 luminance range the histogram covers.
    const LOG_MIN: f32 = -10.0;
    const LOG_MAX: f32 = 6.0;
    /// The metered average ignores the darkest and brightest tails.
    const LOW_PCT: f32 = 0.1;
    const HIGH_PCT: f32 = 0.9;
    /// Adaptation speed, in EV per second: to a brighter scene, and to a
    /// darker.
    const SPEED_TO_BRIGHT: f32 = 3.0;
    const SPEED_TO_DARK: f32 = 1.0;
    /// Exposure = `KEY / 2^adapted`.
    const KEY: f32 = 0.55;
    /// Centre weighting: `exp(-r² / σ²)`, r in half-heights from the centre.
    const CENTRE_SIGMA: f32 = 0.6;
    const WEIGHT_MAX: u32 = 64;

    fn bin_of(lum: f32) -> usize {
        if lum < LOG_MIN.exp2() {
            return 0;
        }
        let t = ((lum.log2() - LOG_MIN) / (LOG_MAX - LOG_MIN)).clamp(0.0, 1.0);
        1 + (t * 254.0) as usize
    }

    fn bin_log(b: usize) -> f32 {
        if b == 0 {
            return LOG_MIN;
        }
        LOG_MIN + ((b - 1) as f32 + 0.5) / 254.0 * (LOG_MAX - LOG_MIN)
    }

    /// `(u, v)` in 0..1 on a screen of `aspect`.
    fn centre_weight(u: f32, v: f32, aspect: f32) -> u32 {
        let (dx, dy) = ((u - 0.5) * 2.0 * aspect, (v - 0.5) * 2.0);
        let w = (-(dx * dx + dy * dy) / (CENTRE_SIGMA * CENTRE_SIGMA)).exp();
        1 + (w * (WEIGHT_MAX - 1) as f32) as u32
    }

    /// The trimmed weighted mean log2 luminance of a histogram.
    fn metered(bins: &[u32; BINS]) -> Option<f32> {
        let total: f32 = bins.iter().map(|&c| c as f32).sum();
        let (lo, hi) = (total * LOW_PCT, total * HIGH_PCT);
        let (mut cum, mut sum, mut kept) = (0.0f32, 0.0, 0.0);
        for (b, &c) in bins.iter().enumerate() {
            let c = c as f32;
            let k = ((cum + c).min(hi) - cum.max(lo)).max(0.0);
            sum += k * bin_log(b);
            kept += k;
            cum += c;
        }
        (kept > 0.0).then(|| sum / kept)
    }

    /// One frame of adaptation, from `adapted` towards `target`.
    fn adapt(adapted: f32, target: f32, dt: f32) -> f32 {
        let speed = if target > adapted {
            SPEED_TO_BRIGHT
        } else {
            SPEED_TO_DARK
        };
        adapted + (target - adapted) * (1.0 - (-dt * speed).exp())
    }

    fn exposure(adapted: f32, min: f32, max: f32) -> f32 {
        (KEY / adapted.exp2()).clamp(min, max)
    }

    fn histogram(lums: &[f32]) -> [u32; BINS] {
        let mut h = [0u32; BINS];
        for &l in lums {
            h[bin_of(l)] += 1;
        }
        h
    }

    // ---- Tests ----

    #[test]
    fn luminance_round_trips_through_its_bin() {
        let width = (LOG_MAX - LOG_MIN) / 254.0;
        for l in [0.002f32, 0.05, 0.18, 1.0, 7.5, 40.0] {
            let back = bin_log(bin_of(l));
            assert!(
                (back - l.log2()).abs() <= width,
                "{l}: bin {} → {back}",
                bin_of(l)
            );
        }
        assert_eq!(bin_of(0.0), 0);
        assert_eq!(bin_of(1e-6), 0);
        assert_eq!(
            bin_of(1e9),
            BINS - 1,
            "past the top clamps into the last bin"
        );
    }

    /// A uniform image meters as itself; a few very bright pixels (fewer
    /// than the top `1 - HIGH_PCT`) don't move it, and neither do a few black
    /// ones.
    #[test]
    fn the_trimmed_mean_ignores_the_tails() {
        let width = (LOG_MAX - LOG_MIN) / 254.0;
        let flat = metered(&histogram(&[0.3; 1000])).unwrap();
        assert!((flat - 0.3f32.log2()).abs() <= width, "{flat}");
        let mut lums = vec![0.3; 1000];
        lums[..50].fill(60.0); // the sun, 5% of the frame
        lums[50..100].fill(0.0); // black, 5%
        let tails = metered(&histogram(&lums)).unwrap();
        assert!((tails - flat).abs() < 1e-3, "{tails} vs {flat}");
        assert_eq!(
            metered(&[0; BINS]),
            None,
            "nothing metered, nothing to adapt to"
        );
    }

    /// Adaptation reaches the target, faster to bright than to dark, and
    /// hardly depends on the frame rate.
    #[test]
    fn adaptation_converges_and_is_frame_rate_independent() {
        let run = |from: f32, to: f32, dt: f32, secs: f32| {
            let mut a = from;
            for _ in 0..(secs / dt).round() as usize {
                a = adapt(a, to, dt);
            }
            a
        };
        assert!((run(-3.0, 1.0, 1.0 / 60.0, 10.0) - 1.0).abs() < 1e-3);
        // Half a second into a 4 EV step: brightening has gone much further
        // (3.1 EV against 1.6 at 3 and 1 EV/s).
        let to_bright = run(-2.0, 2.0, 1.0 / 60.0, 0.5) + 2.0;
        let to_dark = 2.0 - run(2.0, -2.0, 1.0 / 60.0, 0.5);
        assert!(to_bright > 1.5 * to_dark, "{to_bright} vs {to_dark}");
        // 30 fps and 144 fps land in the same place after a second.
        let slow = run(-2.0, 2.0, 1.0 / 30.0, 1.0);
        let fast = run(-2.0, 2.0, 1.0 / 144.0, 1.0);
        assert!((slow - fast).abs() < 1e-3, "{slow} vs {fast}");
    }

    #[test]
    fn exposure_maps_the_key_and_clamps() {
        assert!((exposure(KEY.log2(), 0.1, 10.0) - 1.0).abs() < 1e-5);
        assert_eq!(exposure(-12.0, 0.125, 8.0), 8.0, "very dark: capped");
        assert_eq!(exposure(8.0, 0.125, 8.0), 0.125, "very bright: capped");
    }

    #[test]
    fn centre_weighting_falls_off_but_never_to_zero() {
        let aspect = 16.0 / 9.0;
        let c = centre_weight(0.5, 0.5, aspect);
        assert_eq!(c, WEIGHT_MAX);
        assert!(centre_weight(0.6, 0.5, aspect) < c);
        assert!(centre_weight(0.9, 0.5, aspect) < centre_weight(0.6, 0.5, aspect));
        assert_eq!(centre_weight(0.0, 0.0, aspect), 1, "corners still count");
    }

    /// The shaders declare the reference's constants and the state layout.
    #[test]
    fn the_shaders_match_the_reference() {
        let hist = include_str!("../shaders/exposure_histogram.comp");
        let avg = include_str!("../shaders/exposure_average.comp");
        let f = |name: &str, v: f32| format!("const float {name} = {v:?};");
        for decl in [
            f("LOG_MIN", LOG_MIN),
            f("LOG_MAX", LOG_MAX),
            f("CENTRE_SIGMA", CENTRE_SIGMA),
            format!("const uint WEIGHT_MAX = {WEIGHT_MAX}u;"),
        ] {
            assert!(
                hist.contains(&decl),
                "exposure_histogram.comp lacks `{decl}`"
            );
        }
        for decl in [
            f("LOG_MIN", LOG_MIN),
            f("LOG_MAX", LOG_MAX),
            f("LOW_PCT", LOW_PCT),
            f("HIGH_PCT", HIGH_PCT),
            f("SPEED_TO_BRIGHT", SPEED_TO_BRIGHT),
            f("SPEED_TO_DARK", SPEED_TO_DARK),
            f("KEY", KEY),
        ] {
            assert!(avg.contains(&decl), "exposure_average.comp lacks `{decl}`");
        }
        // The state: 256 bins then four 4-byte fields, in both shaders and
        // the tonemap (which reads `exposure`).
        let tonemap = include_str!("../shaders/tonemap.frag");
        for (name, src) in [("histogram", hist), ("average", avg), ("tonemap", tonemap)] {
            assert!(
                src.contains(&format!(
                    "uint histogram[{BINS}];\n    float adapted;\n    float exposure;"
                )),
                "{name}: state layout differs"
            );
        }
        assert_eq!(std::mem::size_of::<State>(), BINS * 4 + 16);
        assert_eq!(std::mem::size_of::<ExposureReadback>(), 16);
        assert_eq!(GRID_STEP, 4);
    }
}
