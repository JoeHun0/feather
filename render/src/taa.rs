//! Temporal anti-aliasing (§13): each frame's camera is jittered by a
//! sub-pixel offset, and a compute pass blends the frame into a history that
//! follows the camera, so over a few frames every pixel averages samples
//! from across its area.
//!
//! One pass, `taa.comp`, between the main pass and bloom, per pixel:
//! - the 3×3 neighbourhood of the current (resolved) HDR image, in a
//!   compressed space (`compress`: colour over 1 + its luma, after exposure,
//!   so one bright sample can't dominate) and YCoCg, for its mean and
//!   standard deviation;
//! - where the pixel was last frame, from the depth of the neighbourhood's
//!   nearest texel (so an edge moves with its foreground) and the camera's
//!   previous and current (unjittered) view-projection;
//! - the history there, by a 5-tap Catmull-Rom (sharper than bilinear),
//!   clipped towards the mean to a box of `TAA_GAMMA` deviations (so what
//!   the neighbourhood rules out, a disocclusion, a ghost, goes);
//! - blended with the current pixel by `TAA_ALPHA`.
//!
//! Motion is the camera's only: nothing else moves in a level (§26). Things
//! that do move (the orb demo) blur back into their neighbourhood's box.
//!
//! `reference` is the shader step for step, on the CPU, for the tests.

use ash::vk;
use feather_gfx::{Renderer, FRAMES_IN_FLIGHT};
use glam::{Mat4, Vec2, Vec4};

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

/// Jitter phases: the Halton (2, 3) sequence repeats after this many frames.
pub const TAA_PHASES: u32 = 8;
/// How much of each new frame goes into the history (the shader's copy is
/// the one used; a test pins them together).
#[cfg(test)]
pub const TAA_ALPHA: f32 = 0.1;
/// The history is clipped to the neighbourhood's mean ± this many standard
/// deviations.
#[cfg(test)]
pub const TAA_GAMMA: f32 = 1.0;

/// Element `index` (from 1) of the Halton sequence in `base`, in [0, 1).
pub fn halton(mut index: u32, base: u32) -> f32 {
    let (mut f, mut r) = (1.0, 0.0);
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// Frame `frame`'s jitter in pixels, x right and y down, within ±0.5.
pub fn jitter(frame: u64) -> Vec2 {
    let i = (frame % TAA_PHASES as u64) as u32 + 1;
    Vec2::new(halton(i, 2), halton(i, 3)) - 0.5
}

/// `view_proj` moved on screen by `jitter_px` pixels (x right, y down) in an
/// image of `size` pixels: a clip-space shear, so depth is untouched.
pub fn jittered(view_proj: Mat4, jitter_px: Vec2, size: Vec2) -> Mat4 {
    let d = 2.0 * jitter_px / size;
    let shift = Mat4::from_cols(Vec4::X, Vec4::Y, Vec4::Z, Vec4::new(d.x, d.y, 0.0, 1.0));
    shift * view_proj
}

/// Takes a clip-space point of this frame (unjittered) to where it was in the
/// last one: `prev · cur⁻¹`. The identity for a camera that didn't move.
pub fn reprojection(prev_view_proj: Mat4, view_proj: Mat4) -> Mat4 {
    prev_view_proj * view_proj.inverse()
}

/// The pass's push constants.
#[derive(Clone, Copy, Debug)]
pub struct TaaPush {
    /// `reprojection(prev, cur)`.
    pub reproject: Mat4,
    /// False on the first frame after a reset: take the current frame.
    pub history_valid: bool,
    /// The exposure the tonemap applies: the metered one times this when
    /// `auto_exposure`, else this.
    pub exposure: f32,
    pub auto_exposure: bool,
}

impl TaaPush {
    /// `mat4 reproject; vec4 params` (x = history valid, y = exposure,
    /// z = auto exposure).
    pub fn floats(&self) -> [f32; 20] {
        let mut out = [0.0; 20];
        out[..16].copy_from_slice(&self.reproject.to_cols_array());
        out[16] = self.history_valid as u32 as f32;
        out[17] = self.exposure;
        out[18] = self.auto_exposure as u32 as f32;
        out
    }
}

/// What a frame's set points at; rebound when any of it changes.
type Bound = [u64; 5];

/// The resolve pass and its per-frame descriptor sets.
pub struct TaaPass {
    device: ash::Device,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    bound: Vec<Option<Bound>>,
}

impl TaaPass {
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device();
        let bindings = [
            (0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            (1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            (2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            (3, vk::DescriptorType::STORAGE_IMAGE),
            (4, vk::DescriptorType::STORAGE_BUFFER),
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
                .expect("taa set layout")
        };
        let frames = FRAMES_IN_FLIGHT as u32;
        let pool_sizes = [
            (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 3),
            (vk::DescriptorType::STORAGE_IMAGE, 1),
            (vk::DescriptorType::STORAGE_BUFFER, 1),
        ]
        .map(|(ty, n)| {
            vk::DescriptorPoolSize::default()
                .ty(ty)
                .descriptor_count(n * frames)
        });
        let pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(frames)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .expect("taa descriptor pool")
        };
        let layouts = vec![set_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
                .expect("allocate taa sets")
        };
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(80)];
        let set_layouts = [set_layout];
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push_ranges),
                    None,
                )
                .expect("taa pipeline layout")
        };
        let words =
            ash::util::read_spv(&mut std::io::Cursor::new(spv!("taa.comp"))).expect("read taa spv");
        let module = unsafe {
            device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .expect("taa shader module")
        };
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
                .expect("taa pipeline")[0]
        };
        unsafe { device.destroy_shader_module(module, None) };
        Self {
            device,
            set_layout,
            layout,
            pipeline,
            pool,
            sets,
            bound: vec![None; FRAMES_IN_FLIGHT],
        }
    }

    /// Point this frame's set at this frame's images, if any changed: the
    /// resolved HDR (`current`, SHADER_READ_ONLY), the single-sample depth
    /// (read-only), last frame's history (`history`, SHADER_READ_ONLY,
    /// sampled bilinearly) and this frame's (`out`, GENERAL), and the
    /// auto-exposure state. Safe each frame: `draw_frame` waits on the frame
    /// fence before the closures run.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        frame: usize,
        current: vk::ImageView,
        depth: vk::ImageView,
        history: vk::ImageView,
        out: vk::ImageView,
        exposure_state: vk::Buffer,
        linear: vk::Sampler,
        nearest: vk::Sampler,
    ) {
        use ash::vk::Handle;
        let key = [
            current.as_raw(),
            depth.as_raw(),
            history.as_raw(),
            out.as_raw(),
            exposure_state.as_raw(),
        ];
        if self.bound[frame] == Some(key) {
            return;
        }
        let sampled = |view, layout, sampler| {
            [vk::DescriptorImageInfo::default()
                .image_layout(layout)
                .image_view(view)
                .sampler(sampler)]
        };
        let current_info = sampled(current, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, nearest);
        let depth_info = sampled(
            depth,
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            nearest,
        );
        let history_info = sampled(history, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, linear);
        let out_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(out)];
        let state_info = [vk::DescriptorBufferInfo::default()
            .buffer(exposure_state)
            .range(vk::WHOLE_SIZE)];
        let set = self.sets[frame];
        fn image<'a>(
            set: vk::DescriptorSet,
            binding: u32,
            ty: vk::DescriptorType,
            info: &'a [vk::DescriptorImageInfo; 1],
        ) -> vk::WriteDescriptorSet<'a> {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding)
                .descriptor_type(ty)
                .image_info(info)
        }
        let writes = [
            image(
                set,
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                &current_info,
            ),
            image(
                set,
                1,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                &depth_info,
            ),
            image(
                set,
                2,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                &history_info,
            ),
            image(set, 3, vk::DescriptorType::STORAGE_IMAGE, &out_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&state_info),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        self.bound[frame] = Some(key);
    }

    /// Record the resolve over an image of `extent`. `draw_frame` puts the
    /// images in their layouts first and hands the result on after.
    pub fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        frame: usize,
        extent: vk::Extent2D,
        push: &TaaPush,
    ) {
        let bytes: Vec<u8> = push.floats().iter().flat_map(|v| v.to_ne_bytes()).collect();
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.sets[frame]],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &bytes,
            );
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device
                .cmd_dispatch(cmd, extent.width.div_ceil(8), extent.height.div_ceil(8), 1);
        }
    }
}

impl Drop for TaaPass {
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

/// `taa.comp` on the CPU, function for function, for the tests.
#[cfg(test)]
pub mod reference {
    use super::{TaaPush, TAA_ALPHA, TAA_GAMMA};
    use glam::{IVec2, Vec2, Vec3, Vec4};

    /// An RGB image, row by row.
    #[derive(Clone, Debug)]
    pub struct Image {
        pub width: i32,
        pub height: i32,
        pub pixels: Vec<Vec3>,
    }

    impl Image {
        pub fn new(width: i32, height: i32, f: impl Fn(i32, i32) -> Vec3) -> Self {
            let pixels = (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .map(|(x, y)| f(x, y))
                .collect();
            Self {
                width,
                height,
                pixels,
            }
        }

        /// `texelFetch`, clamped to the edge.
        pub fn fetch(&self, p: IVec2) -> Vec3 {
            let p = p.clamp(IVec2::ZERO, IVec2::new(self.width - 1, self.height - 1));
            self.pixels[(p.y * self.width + p.x) as usize]
        }

        /// `texture` with a linear, clamp-to-edge sampler at `uv`.
        pub fn bilinear(&self, uv: Vec2) -> Vec3 {
            let q = uv * Vec2::new(self.width as f32, self.height as f32) - 0.5;
            let base = q.floor();
            let f = q - base;
            let b = base.as_ivec2();
            let row = |y| {
                self.fetch(IVec2::new(b.x, y))
                    .lerp(self.fetch(IVec2::new(b.x + 1, y)), f.x)
            };
            row(b.y).lerp(row(b.y + 1), f.y)
        }
    }

    pub fn luma(c: Vec3) -> f32 {
        c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
    }

    /// Exposed colour over 1 + its luma: bounded, so a lone bright sample
    /// weighs as much as a bright one, not thousands of times more.
    pub fn compress(c: Vec3, exposure: f32) -> Vec3 {
        let s = c.max(Vec3::ZERO) * exposure;
        s / (1.0 + luma(s))
    }

    /// `compress`'s inverse.
    pub fn expand(t: Vec3, exposure: f32) -> Vec3 {
        t / (1.0 - luma(t)).max(1e-4) / exposure
    }

    pub fn to_ycocg(c: Vec3) -> Vec3 {
        Vec3::new(
            0.25 * c.x + 0.5 * c.y + 0.25 * c.z,
            0.5 * c.x - 0.5 * c.z,
            -0.25 * c.x + 0.5 * c.y - 0.25 * c.z,
        )
    }

    pub fn from_ycocg(y: Vec3) -> Vec3 {
        Vec3::new(y.x + y.y - y.z, y.x + y.z, y.x - y.y - y.z)
    }

    /// `h` pulled towards `mean` onto the box of half-size `extent` round it,
    /// if outside; unchanged inside.
    pub fn clip_to_box(h: Vec3, mean: Vec3, extent: Vec3) -> Vec3 {
        let v = h - mean;
        let units = (v / extent.max(Vec3::splat(1e-5))).abs();
        let m = units.max_element();
        if m > 1.0 {
            mean + v / m
        } else {
            h
        }
    }

    /// Catmull-Rom weights for a tap `f` ∈ [0, 1) past texel 1 of 0..3.
    pub fn catmull_rom_weights(f: f32) -> [f32; 4] {
        [
            f * (-0.5 + f * (1.0 - 0.5 * f)),
            1.0 + f * f * (-2.5 + 1.5 * f),
            f * (0.5 + f * (2.0 - 1.5 * f)),
            f * f * (-0.5 + 0.5 * f),
        ]
    }

    /// A bicubic Catmull-Rom sample of `img` at `uv` in five bilinear taps
    /// (the corner ones left out, the rest renormalised).
    pub fn catmull_rom(img: &Image, uv: Vec2) -> Vec3 {
        let size = Vec2::new(img.width as f32, img.height as f32);
        let q = uv * size;
        let t1 = (q - 0.5).floor() + 0.5;
        let f = q - t1;
        let wx = catmull_rom_weights(f.x);
        let wy = catmull_rom_weights(f.y);
        let w12 = Vec2::new(wx[1] + wx[2], wy[1] + wy[2]);
        let t0 = (t1 - 1.0) / size;
        let t3 = (t1 + 2.0) / size;
        let t12 = (t1 + Vec2::new(wx[2], wy[2]) / w12) / size;
        let taps = [
            (Vec2::new(t12.x, t0.y), w12.x * wy[0]),
            (Vec2::new(t0.x, t12.y), wx[0] * w12.y),
            (Vec2::new(t12.x, t12.y), w12.x * w12.y),
            (Vec2::new(t3.x, t12.y), wx[3] * w12.y),
            (Vec2::new(t12.x, t3.y), w12.x * wy[3]),
        ];
        let mut sum = Vec3::ZERO;
        let mut weight = 0.0;
        for (uv, w) in taps {
            sum += img.bilinear(uv) * w;
            weight += w;
        }
        sum / weight
    }

    /// One frame: `current` (this frame's image, jittered), its `depth`
    /// (row by row) and `history` (last frame's output) into this frame's
    /// output.
    pub fn resolve(current: &Image, depth: &[f32], history: &Image, push: &TaaPush) -> Image {
        let size = IVec2::new(current.width, current.height);
        let sizef = size.as_vec2();
        let exposure = push.exposure;
        Image::new(current.width, current.height, |x, y| {
            let p = IVec2::new(x, y);
            // The neighbourhood: its moments in compressed YCoCg, and its
            // nearest depth.
            let (mut m1, mut m2) = (Vec3::ZERO, Vec3::ZERO);
            let mut nearest = (p, f32::INFINITY);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let q = (p + IVec2::new(dx, dy)).clamp(IVec2::ZERO, size - 1);
                    let c = to_ycocg(compress(current.fetch(q), exposure));
                    m1 += c;
                    m2 += c * c;
                    let d = depth[(q.y * size.x + q.x) as usize];
                    if d < nearest.1 {
                        nearest = (q, d);
                    }
                }
            }
            let mean = m1 / 9.0;
            let var = (m2 / 9.0 - mean * mean).max(Vec3::ZERO);
            let sigma = Vec3::new(var.x.sqrt(), var.y.sqrt(), var.z.sqrt());
            let centre = to_ycocg(compress(current.fetch(p), exposure));

            // Where the nearest texel was last frame: its motion, moved to p.
            let (q, d) = nearest;
            let ndc = (q.as_vec2() + 0.5) / sizef * 2.0 - 1.0;
            let prev = push.reproject * Vec4::new(ndc.x, ndc.y, d, 1.0);
            let motion = Vec2::new(prev.x, prev.y) / prev.w - ndc;
            let uv = (p.as_vec2() + 0.5) / sizef + motion * 0.5;

            let inside = uv.cmpge(Vec2::ZERO).all() && uv.cmple(Vec2::ONE).all();
            let out = if push.history_valid && inside {
                let h = to_ycocg(compress(catmull_rom(history, uv), exposure));
                let h = clip_to_box(h, mean, sigma * TAA_GAMMA);
                h.lerp(centre, TAA_ALPHA)
            } else {
                centre
            };
            expand(from_ycocg(out), exposure)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::reference::*;
    use super::*;
    use glam::{IVec2, Vec3};

    const SIZE: Vec2 = Vec2::new(1920.0, 1080.0);

    fn camera(eye: Vec3, yaw: f32) -> Mat4 {
        let fwd = Vec3::new(yaw.cos(), 0.0, yaw.sin());
        let mut proj = Mat4::perspective_rh(1.0, SIZE.x / SIZE.y, 0.1, 200.0);
        proj.y_axis.y *= -1.0;
        proj * Mat4::look_to_rh(eye, fwd, Vec3::Y)
    }

    fn to_pixels(view_proj: Mat4, p: Vec3) -> (Vec2, f32) {
        let c = view_proj * p.extend(1.0);
        let ndc = c.truncate() / c.w;
        ((Vec2::new(ndc.x, ndc.y) + 1.0) * 0.5 * SIZE, ndc.z)
    }

    #[test]
    fn jitter_covers_the_pixel_evenly() {
        let js: Vec<Vec2> = (0..TAA_PHASES as u64).map(jitter).collect();
        for (i, a) in js.iter().enumerate() {
            assert!(a.abs().max_element() < 0.5, "{a} leaves the pixel");
            for b in &js[i + 1..] {
                assert!(a.distance(*b) > 0.05, "{a} and {b} nearly repeat");
            }
        }
        let mean = js.iter().sum::<Vec2>() / js.len() as f32;
        assert!(mean.abs().max_element() < 0.07, "mean {mean} is off centre");
        // It cycles.
        assert_eq!(jitter(3), jitter(3 + TAA_PHASES as u64));
        assert_eq!(halton(1, 2), 0.5);
        assert_eq!(halton(3, 3), 1.0 / 9.0);
    }

    #[test]
    fn jittering_moves_the_image_by_the_offset_and_keeps_depth() {
        let vp = camera(Vec3::new(1.0, 2.0, 3.0), 0.7);
        let j = Vec2::new(0.3125, -0.1875);
        let point = Vec3::new(1.0, 2.5, 3.0) + Vec3::new(0.7f32.cos(), 0.0, 0.7f32.sin()) * 9.0;
        let (a, za) = to_pixels(vp, point);
        let (b, zb) = to_pixels(jittered(vp, j, SIZE), point);
        assert!(
            (b - a - j).abs().max_element() < 1e-3,
            "moved {} not {j}",
            b - a
        );
        assert_eq!(za, zb);
        // Control: no jitter, no move.
        let (c, _) = to_pixels(jittered(vp, Vec2::ZERO, SIZE), point);
        assert!((c - a).abs().max_element() < 1e-4);
    }

    #[test]
    fn reprojection_finds_last_frames_pixel() {
        let prev = camera(Vec3::new(0.0, 1.6, 0.0), 0.3);
        let cur = camera(Vec3::new(0.2, 1.6, -0.1), 0.34);
        let point = Vec3::new(8.0, 1.0, 3.5);
        let (now, depth) = to_pixels(cur, point);
        let (then, _) = to_pixels(prev, point);
        let ndc = now / SIZE * 2.0 - 1.0;
        let c = reprojection(prev, cur) * Vec4::new(ndc.x, ndc.y, depth, 1.0);
        let found = (Vec2::new(c.x, c.y) / c.w + 1.0) * 0.5 * SIZE;
        assert!((found - then).length() < 0.01, "{found} vs {then}");
        assert!(
            (then - now).length() > 5.0,
            "the camera should have moved it"
        );
        // A still camera maps every point to itself.
        let still = reprojection(cur, cur);
        assert!(still.abs_diff_eq(Mat4::IDENTITY, 1e-4));
    }

    #[test]
    fn colour_transforms_round_trip() {
        for c in [
            Vec3::new(0.2, 0.5, 0.9),
            Vec3::new(40.0, 3.0, 0.01),
            Vec3::ZERO,
        ] {
            assert!((from_ycocg(to_ycocg(c)) - c).abs().max_element() < 1e-5);
            let back = expand(compress(c, 1.4), 1.4);
            assert!((back - c).abs().max_element() < 1e-3 * (1.0 + c.max_element()));
            assert!(luma(compress(c, 1.4)) < 1.0);
        }
    }

    #[test]
    fn catmull_rom_is_exact_at_centres_and_on_ramps() {
        for f in [0.0, 0.25, 0.5, 0.9] {
            let w = catmull_rom_weights(f);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        let ramp = Image::new(16, 12, |x, y| Vec3::new(x as f32, y as f32, 1.0));
        let size = Vec2::new(16.0, 12.0);
        // At texel centres it is the texel; between them along a row, the
        // ramp. Off both axes it's close: the five taps leave out the corners.
        for (q, want, within) in [
            (Vec2::new(5.5, 7.5), Vec3::new(5.0, 7.0, 1.0), 1e-4),
            (Vec2::new(6.2, 4.5), Vec3::new(5.7, 4.0, 1.0), 1e-4),
            (Vec2::new(6.2, 4.9), Vec3::new(5.7, 4.4, 1.0), 1e-2),
        ] {
            let got = catmull_rom(&ramp, q / size);
            assert!((got - want).abs().max_element() < within, "{got} vs {want}");
        }
        // Sharper than bilinear on a spike: it keeps more of the peak halfway.
        let spike = Image::new(9, 9, |x, y| Vec3::splat((x == 4 && y == 4) as u32 as f32));
        let uv = Vec2::new(4.75, 4.5) / 9.0;
        assert!(catmull_rom(&spike, uv).x > spike.bilinear(uv).x);
    }

    #[test]
    fn clipping_keeps_the_inside_and_pulls_in_the_outside() {
        let (mean, ext) = (Vec3::new(0.5, 0.0, 0.1), Vec3::new(0.2, 0.1, 0.1));
        let inside = Vec3::new(0.6, 0.05, 0.0);
        assert_eq!(clip_to_box(inside, mean, ext), inside);
        let outside = Vec3::new(1.3, 0.0, 0.1);
        let c = clip_to_box(outside, mean, ext);
        assert!((c - Vec3::new(0.7, 0.0, 0.1)).length() < 1e-6, "{c}");
        // On the line to the mean.
        let o = Vec3::new(0.9, 0.3, 0.1);
        let c = clip_to_box(o, mean, ext);
        assert!((c - mean).cross(o - mean).length() < 1e-6);
        assert!(((c - mean) / ext).abs().max_element() <= 1.0 + 1e-6);
    }

    /// A still camera over an edge at x = 3.3 px: each frame samples the
    /// scene at pixel centres moved by the jitter.
    fn edge_frames(frames: u64, jitter_on: bool) -> Vec<Image> {
        let edge = 3.3;
        let depth = vec![0.5; 8 * 3];
        let push = TaaPush {
            reproject: Mat4::IDENTITY,
            history_valid: false,
            exposure: 1.0,
            auto_exposure: false,
        };
        let mut out = Vec::new();
        let mut history = Image::new(8, 3, |_, _| Vec3::ZERO);
        for f in 0..frames {
            let j = if jitter_on { jitter(f) } else { Vec2::ZERO };
            // The content moves with the jitter, so pixel x sees x + 0.5 - j.
            let current = Image::new(8, 3, |x, _| {
                Vec3::splat(((x as f32 + 0.5 - j.x) < edge) as u32 as f32)
            });
            let push = TaaPush {
                history_valid: f > 0,
                ..push
            };
            history = resolve(&current, &depth, &history, &push);
            out.push(history.clone());
        }
        out
    }

    #[test]
    fn a_still_edge_converges_to_its_coverage() {
        let frames = edge_frames(96, true);
        // Pixel 3 spans x = 3..4 and the edge is at 3.3. Of the jitter's
        // samples, those left of the edge are its coverage; the average is
        // taken in compressed space, which weighs the dark ones up.
        let hits = (0..TAA_PHASES as u64)
            .filter(|&f| 3.5 - jitter(f).x < 3.3)
            .count();
        let coverage = hits as f32 / TAA_PHASES as f32;
        let want = expand(compress(Vec3::ONE, 1.0) * coverage, 1.0).x;
        let last: Vec<f32> = frames[88..]
            .iter()
            .map(|f| f.fetch(IVec2::new(3, 1)).x)
            .collect();
        let mean = last.iter().sum::<f32>() / last.len() as f32;
        assert!(coverage > 0.0 && coverage < 1.0);
        assert!(
            (mean - want).abs() < 0.03,
            "edge pixel settles at {mean}, not {want}"
        );
        for v in &last {
            assert!(*v > 0.02 && *v < 0.5, "edge pixel swings to {v}");
        }
        // Fully covered and empty pixels stay put.
        let f = frames.last().unwrap();
        assert!((f.fetch(IVec2::new(1, 1)).x - 1.0).abs() < 1e-3);
        assert!(f.fetch(IVec2::new(6, 1)).x.abs() < 1e-3);
        // Control: without the jitter the edge stays hard (pixel 3's centre,
        // 3.5, is past it).
        let hard = edge_frames(96, false);
        assert_eq!(hard.last().unwrap().fetch(IVec2::new(3, 1)).x, 0.0);
    }

    /// A pan across a sine pattern at `speed` px a frame, resolved with the
    /// given reprojection (the true one, or a still camera's); the mean
    /// error against the true image after `frames`.
    fn pan_error(speed: f32, frames: u32, reproject_right: bool) -> f32 {
        let (w, h) = (48, 4);
        let scene = |x: f32| 0.5 + 0.4 * (x * std::f32::consts::TAU / 16.0).sin();
        let depth = vec![0.5; (w * h) as usize];
        let image = |n: u32| {
            Image::new(w, h, |x, _| {
                Vec3::splat(scene(x as f32 + 0.5 + n as f32 * speed))
            })
        };
        // The camera pans right, so a pixel's content was `speed` px further
        // right last frame: a clip-space shift of 2·speed/width.
        let shift = if reproject_right {
            2.0 * speed / w as f32
        } else {
            0.0
        };
        let reproject = Mat4::from_cols(Vec4::X, Vec4::Y, Vec4::Z, Vec4::new(shift, 0.0, 0.0, 1.0));
        let mut history = image(0);
        for n in 1..=frames {
            let push = TaaPush {
                reproject,
                history_valid: true,
                exposure: 1.0,
                auto_exposure: false,
            };
            history = resolve(&image(n), &depth, &history, &push);
        }
        let truth = image(frames);
        // Away from the edges, where the history runs out.
        let row = (6..w - 6).map(|x| {
            (history.fetch(IVec2::new(x, 1)) - truth.fetch(IVec2::new(x, 1)))
                .x
                .abs()
        });
        row.clone().sum::<f32>() / row.count() as f32
    }

    #[test]
    fn a_pan_keeps_its_image() {
        let right = pan_error(0.37, 60, true);
        assert!(right < 0.03, "pan drifts by {right}");
        // Control: a history that doesn't follow the camera smears.
        let wrong = pan_error(0.37, 60, false);
        assert!(
            wrong > 3.0 * right && wrong > 0.03,
            "unfollowed error {wrong}"
        );
    }

    #[test]
    fn the_shader_declares_the_reference_constants() {
        let src = include_str!("../shaders/taa.comp");
        for decl in [
            format!("const float TAA_ALPHA = {TAA_ALPHA:?};"),
            format!("const float TAA_GAMMA = {TAA_GAMMA:?};"),
            "vec3 compress(vec3 c, float exposure) {".to_string(),
            "return s / (1.0 + luma(s));".to_string(),
            "return t / max(1.0 - luma(t), 1e-4) / exposure;".to_string(),
            "vec3 units = abs(v / max(extent, vec3(1e-5)));".to_string(),
            "vec2 motion = prev.xy / prev.w - ndc;".to_string(),
        ] {
            assert!(src.contains(&decl), "taa.comp lacks `{decl}`");
        }
        assert_eq!(
            std::mem::size_of_val(
                &TaaPush {
                    reproject: Mat4::IDENTITY,
                    history_valid: true,
                    exposure: 1.0,
                    auto_exposure: false,
                }
                .floats()
            ),
            80
        );
    }
}
