//! Ground-truth ambient occlusion (§13, GTAO): contact shadowing of the
//! ambient light within `AO_RADIUS` of each pixel, from the depth prepass.
//!
//! Two compute passes between the prepass and the main pass:
//! - `gtao.comp`: per pixel, `AO_SLICES` slices × `AO_STEPS` steps each way
//!   for the horizons, integrated against the normal (rebuilt from depth), with
//!   slice angle and step offset jittered by 4×4 patterns;
//! - `gtao_denoise.comp`: a depth-aware 4×4 blur, which holds each jitter
//!   variant exactly once, so the noise averages out.
//!
//! `mesh.frag` then multiplies the ambient (diffuse with a multi-bounce fit,
//! specular with a specular-occlusion fit) by the result. The large scale is
//! the baked sky visibility's job; this adds what it's too coarse for.
//!
//! `reference` is both shaders step for step, on the CPU, so the tests can
//! run them over depth buffers of known scenes.

use ash::vk;
use feather_gfx::{Renderer, FRAMES_IN_FLIGHT};

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

/// The camera terms both passes need (their push constants).
#[derive(Clone, Copy, Debug)]
pub struct AoProjection {
    pub near: f32,
    pub far: f32,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    /// Width over height.
    pub aspect: f32,
}

impl AoProjection {
    /// `[near·r, r, tan(fov_x/2), tan(fov_y/2), px per m at 1 m, 0, 0, 0]`
    /// with `r = far / (near − far)`, for an image `height` pixels tall.
    pub fn push(&self, height: u32) -> [f32; 8] {
        let r = self.far / (self.near - self.far);
        let tan_y = (self.fov_y * 0.5).tan();
        [
            self.near * r,
            r,
            tan_y * self.aspect,
            tan_y,
            height as f32 / (2.0 * tan_y),
            0.0,
            0.0,
            0.0,
        ]
    }
}

/// The two GTAO compute passes and their per-frame descriptor sets.
pub struct AoPass {
    device: ash::Device,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    gtao: vk::Pipeline,
    denoise: vk::Pipeline,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    /// The `Renderer::targets_generation` each frame's set points at.
    bound: Vec<Option<u64>>,
}

impl AoPass {
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device();
        let bindings = [
            (0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            (1, vk::DescriptorType::STORAGE_IMAGE),
            (2, vk::DescriptorType::STORAGE_IMAGE),
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
                .expect("ao set layout")
        };
        let frames = FRAMES_IN_FLIGHT as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(frames),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
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
                .expect("ao descriptor pool")
        };
        let layouts = vec![set_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
                .expect("allocate ao sets")
        };
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(32)];
        let set_layouts = [set_layout];
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push_ranges),
                    None,
                )
                .expect("ao pipeline layout")
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
        let gtao = compute(spv!("gtao.comp"), "gtao pipeline");
        let denoise = compute(spv!("gtao_denoise.comp"), "gtao denoise pipeline");
        Self {
            device,
            set_layout,
            layout,
            gtao,
            denoise,
            pool,
            sets,
            bound: vec![None; FRAMES_IN_FLIGHT],
        }
    }

    /// Point this frame's set at the renderer's current depth and AO images,
    /// if they've been recreated since (`generation`). Safe each frame:
    /// `draw_frame` waits on the frame fence before the closures run, and
    /// this is called before the set is bound.
    pub fn update(
        &mut self,
        frame: usize,
        generation: u64,
        depth_view: vk::ImageView,
        [raw_view, ao_view]: [vk::ImageView; 2],
        sampler: vk::Sampler,
    ) {
        if self.bound[frame] == Some(generation) {
            return;
        }
        let depth_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
            .image_view(depth_view)
            .sampler(sampler)];
        let storage = |view| {
            [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(view)]
        };
        let (raw_info, ao_info) = (storage(raw_view), storage(ao_view));
        let set = self.sets[frame];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&depth_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&raw_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&ao_info),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        self.bound[frame] = Some(generation);
    }

    /// Record both passes over an image of `extent`, with the barrier
    /// between them. `draw_frame` puts the images in GENERAL first and makes
    /// the result visible to fragment shaders after.
    pub fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        frame: usize,
        extent: vk::Extent2D,
        proj: AoProjection,
    ) {
        let push = proj.push(extent.height);
        let bytes: Vec<u8> = push.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let groups = (extent.width.div_ceil(8), extent.height.div_ceil(8));
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
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.gtao);
            self.device.cmd_dispatch(cmd, groups.0, groups.1, 1);
            let b = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[b],
                &[],
                &[],
            );
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.denoise);
            self.device.cmd_dispatch(cmd, groups.0, groups.1, 1);
        }
    }
}

impl Drop for AoPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.gtao, None);
            self.device.destroy_pipeline(self.denoise, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

fn load_shader(device: &ash::Device, bytes: &[u8]) -> vk::ShaderModule {
    let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).expect("read spv");
    unsafe {
        device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .expect("shader module")
    }
}

/// The shaders' reference, step for step, and their constants: what the
/// tests run over depth buffers of known scenes, and pin the shaders to.
#[cfg(test)]
mod reference {
    /// Slices per pixel; the 4×4 jitter makes it 16× that after the denoise.
    pub const AO_SLICES: usize = 2;
    /// Steps each way along a slice.
    pub const AO_STEPS: usize = 4;
    /// How far occluders count, in metres. The sky visibility's cells are 0.5 m
    /// and it samples a cell off the surface, so this covers what it can't see.
    pub const AO_RADIUS: f32 = 0.8;
    /// The last fraction of the radius over which an occluder fades out.
    pub const AO_FALLOFF: f32 = 0.6;
    /// Below this screen radius (far away) a pixel is left open.
    pub const AO_MIN_RADIUS_PX: f32 = 2.0;
    /// The screen radius is clamped to this, which bounds the cost up close.
    pub const AO_MAX_RADIUS_PX: f32 = 200.0;
    /// The first step is at least this many pixels out, so it never samples the
    /// pixel itself.
    pub const AO_MIN_STEP_PX: f32 = 1.3;
    /// The denoise ignores neighbours this far off the centre's distance,
    /// relatively.
    pub const AO_DENOISE_DEPTH: f32 = 0.1;
    /// Jimenez et al. 2016's multi-bounce fit: `(a, b, c)` each `k·albedo + d`
    /// as `[k, d]`. mesh.frag's `ao_multibounce` (a test checks).
    pub const AO_MULTIBOUNCE: [[f32; 2]; 3] =
        [[2.0404, -0.3324], [-4.7951, 0.6417], [2.7552, 0.6903]];

    /// A 4×4 ordered-dither index, 0–15.
    pub fn bayer4(x: i32, y: i32) -> f32 {
        const M: [f32; 16] = [
            0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0,
        ];
        M[((y & 3) * 4 + (x & 3)) as usize]
    }

    /// The cosine-weighted visibility of one slice between horizon angles `h0`
    /// and `h1` (from the view vector) for a normal at angle `n` in it:
    /// `∫ cos(θ − n)·|sin θ| dθ` over `[h0, h1]`, in closed form.
    pub fn slice_arc(h0: f32, h1: f32, n: f32) -> f32 {
        let (sin_n, cos_n) = n.sin_cos();
        0.25 * (-(2.0 * h0 - n).cos() + cos_n + 2.0 * h0 * sin_n)
            + 0.25 * (-(2.0 * h1 - n).cos() + cos_n + 2.0 * h1 * sin_n)
    }

    /// Jimenez et al. 2016: visibility `v` with the light that bounces between
    /// occluders of this `albedo` put back, per channel.
    pub fn multibounce(v: f32, albedo: [f32; 3]) -> [f32; 3] {
        let [a, b, c] = AO_MULTIBOUNCE;
        albedo.map(|al| {
            let (a, b, c) = (a[0] * al + a[1], b[0] * al + b[1], c[0] * al + c[1]);
            v.max(((v * a + b) * v + c) * v)
        })
    }

    /// Lagarde and de Rousiers 2014: how much of the specular ambient survives
    /// visibility `ao` at this view angle and roughness.
    pub fn specular_occlusion(ndv: f32, ao: f32, roughness: f32) -> f32 {
        ((ndv + ao).powf((-16.0 * roughness - 1.0).exp2()) - 1.0 + ao).clamp(0.0, 1.0)
    }

    /// A depth buffer (0 near, 1 far/sky) and the projection it was made with:
    /// what the passes read. Row-major, `width × height`.
    pub struct DepthImage<'a> {
        pub width: i32,
        pub height: i32,
        pub depth: &'a [f32],
        pub push: [f32; 8],
    }

    impl DepthImage<'_> {
        fn raw(&self, x: i32, y: i32) -> f32 {
            let x = x.clamp(0, self.width - 1);
            let y = y.clamp(0, self.height - 1);
            self.depth[(y * self.width + x) as usize]
        }

        pub fn dist(&self, depth: f32) -> f32 {
            self.push[0] / (depth + self.push[1])
        }

        /// gtao.comp's `view_pos`: x right, y down, z the distance ahead.
        pub fn view_pos(&self, x: i32, y: i32) -> [f32; 3] {
            let (cx, cy) = (x.clamp(0, self.width - 1), y.clamp(0, self.height - 1));
            let d = self.dist(self.raw(cx, cy));
            let nx = (cx as f32 + 0.5) / self.width as f32 * 2.0 - 1.0;
            let ny = (cy as f32 + 0.5) / self.height as f32 * 2.0 - 1.0;
            [nx * self.push[2] * d, ny * self.push[3] * d, d]
        }

        /// gtao.comp's `view_pos_at`: at screen point `q` (pixels, centres on
        /// +0.5), depth interpolated between the four texels round it, as
        /// `textureGather` with a clamp-to-edge sampler finds them.
        pub fn view_pos_at(&self, q: [f32; 2]) -> [f32; 3] {
            let (x0, y0) = ((q[0] - 0.5).floor(), (q[1] - 0.5).floor());
            let (fx, fy) = (q[0] - 0.5 - x0, q[1] - 0.5 - y0);
            let (x0, y0) = (x0 as i32, y0 as i32);
            let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
            let top = lerp(self.raw(x0, y0), self.raw(x0 + 1, y0), fx);
            let bottom = lerp(self.raw(x0, y0 + 1), self.raw(x0 + 1, y0 + 1), fx);
            let d = self.dist(lerp(top, bottom, fy));
            let nx = q[0] / self.width as f32 * 2.0 - 1.0;
            let ny = q[1] / self.height as f32 * 2.0 - 1.0;
            [nx * self.push[2] * d, ny * self.push[3] * d, d]
        }

        /// gtao.comp's `depth_normal`, at pixel (x, y).
        fn depth_normal(&self, x: i32, y: i32) -> glam::Vec3 {
            let pos = |x, y| glam::Vec3::from(self.view_pos(x, y));
            let p = pos(x, y);
            let (l, r, u, d) = (pos(x - 1, y), pos(x + 1, y), pos(x, y - 1), pos(x, y + 1));
            let right = x == 0 || (x < self.width - 1 && (r.z - p.z).abs() < (p.z - l.z).abs());
            let down = y == 0 || (y < self.height - 1 && (d.z - p.z).abs() < (p.z - u.z).abs());
            let dx = if right { r - p } else { p - l };
            let dy = if down { d - p } else { p - u };
            dy.cross(dx).normalize()
        }

        /// gtao.comp for one pixel.
        pub fn gtao(&self, x: i32, y: i32) -> f32 {
            use glam::Vec3;
            use std::f32::consts::{FRAC_PI_2, PI};
            if self.raw(x, y) >= 1.0 {
                return 1.0;
            }
            let pos = |x, y| Vec3::from(self.view_pos(x, y));
            let p = pos(x, y);
            let radius_px = AO_RADIUS * self.push[4] / p.z;
            if radius_px < AO_MIN_RADIUS_PX {
                return 1.0;
            }
            let radius_px = radius_px.min(AO_MAX_RADIUS_PX);
            let n_vec = self.depth_normal(x, y);
            let v = (-p).normalize();
            let falloff_mul = -1.0 / (AO_FALLOFF * AO_RADIUS);
            let falloff_add = 1.0 / AO_FALLOFF;
            let slice_noise = (bayer4(x, y) + 0.5) / 16.0;
            let step_noise = (bayer4(y, x) + 0.5) / 16.0;
            let min_step = AO_MIN_STEP_PX / radius_px;
            let centre = [x as f32 + 0.5, y as f32 + 0.5];
            let at = |q: [f32; 2]| Vec3::from(self.view_pos_at(q));
            let b1 = Vec3::Y.cross(v).normalize();
            let b2 = v.cross(b1);
            let mut visibility = 0.0;
            for s in 0..AO_SLICES {
                let psi = (s as f32 + slice_noise) / AO_SLICES as f32 * PI;
                let ortho = psi.cos() * b1 + psi.sin() * b2;
                let dir = (ortho.truncate() * p.z - p.truncate() * ortho.z).normalize();
                let axis = ortho.cross(v);
                let proj_n = n_vec - axis * n_vec.dot(axis);
                let proj_len = proj_n.length();
                let cos_n = (proj_n.dot(v) / proj_len).clamp(0.0, 1.0);
                let n = ortho.dot(proj_n).signum() * cos_n.acos();
                let (low0, low1) = ((n + FRAC_PI_2).cos(), (n - FRAC_PI_2).cos());
                let (mut hc0, mut hc1) = (low0, low1);
                for j in 0..AO_STEPS {
                    let t = (j as f32 + step_noise) / AO_STEPS as f32;
                    let t = t * t + min_step;
                    let (ox, oy) = (dir.x * (t * radius_px), dir.y * (t * radius_px));
                    let d0 = at([centre[0] + ox, centre[1] + oy]) - p;
                    let d1 = at([centre[0] - ox, centre[1] - oy]) - p;
                    let (l0, l1) = (d0.length(), d1.length());
                    let w0 = (l0 * falloff_mul + falloff_add).clamp(0.0, 1.0);
                    let w1 = (l1 * falloff_mul + falloff_add).clamp(0.0, 1.0);
                    let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
                    hc0 = hc0.max(mix(low0, (d0 / l0.max(1e-6)).dot(v), w0));
                    hc1 = hc1.max(mix(low1, (d1 / l1.max(1e-6)).dot(v), w1));
                }
                let h0 = -hc1.clamp(-1.0, 1.0).acos();
                let h1 = hc0.clamp(-1.0, 1.0).acos();
                let h0 = n + (h0 - n).max(-FRAC_PI_2);
                let h1 = n + (h1 - n).min(FRAC_PI_2);
                visibility += proj_len * slice_arc(h0, h1, n);
            }
            (visibility / AO_SLICES as f32).max(0.0)
        }

        /// gtao_denoise.comp for one pixel, over `raw` (gtao's output, the same
        /// size).
        pub fn denoise(&self, raw: &[f32], x: i32, y: i32) -> f32 {
            use glam::Vec3;
            if self.raw(x, y) >= 1.0 {
                return 1.0;
            }
            let p = Vec3::from(self.view_pos(x, y));
            let n = self.depth_normal(x, y);
            let tolerance = AO_DENOISE_DEPTH * p.z;
            let (mut sum, mut total) = (0.0, 0.0);
            for dy in -1..=2 {
                for dx in -1..=2 {
                    let qx = (x + dx).clamp(0, self.width - 1);
                    let qy = (y + dy).clamp(0, self.height - 1);
                    let off = (Vec3::from(self.view_pos(qx, qy)) - p).dot(n).abs();
                    let w = (1.0 - off / tolerance).max(0.0);
                    sum += w * raw[(qy * self.width + qx) as usize];
                    total += w;
                }
            }
            (sum / total).min(1.0)
        }

        /// Both passes over the whole image.
        pub fn ambient_occlusion(&self) -> Vec<f32> {
            let raw: Vec<f32> = (0..self.height)
                .flat_map(|y| (0..self.width).map(move |x| (x, y)))
                .map(|(x, y)| self.gtao(x, y))
                .collect();
            (0..self.height)
                .flat_map(|y| (0..self.width).map(move |x| (x, y)))
                .map(|(x, y)| self.denoise(&raw, x, y))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::reference::*;
    use super::*;
    use glam::{Mat4, Vec3, Vec4Swizzles};
    use std::f32::consts::FRAC_PI_2;

    const W: i32 = 160;
    const H: i32 = 90;

    /// The app's camera (`Look::view_proj`): glam's perspective with y
    /// flipped for Vulkan.
    fn camera(eye: Vec3, target: Vec3) -> (Mat4, AoProjection) {
        let proj = AoProjection {
            near: 0.1,
            far: 200.0,
            fov_y: 60f32.to_radians(),
            aspect: W as f32 / H as f32,
        };
        let mut p = Mat4::perspective_rh(proj.fov_y, proj.aspect, proj.near, proj.far);
        p.y_axis.y *= -1.0;
        (p * Mat4::look_at_rh(eye, target, Vec3::Y), proj)
    }

    /// A depth buffer for `view_proj`, from `hit`: a ray's (origin,
    /// direction) to the world point it hits first, if any.
    fn render(view_proj: Mat4, hit: impl Fn(Vec3, Vec3) -> Option<Vec3>) -> Vec<f32> {
        let inv = view_proj.inverse();
        let mut depth = Vec::with_capacity((W * H) as usize);
        for y in 0..H {
            for x in 0..W {
                let nx = (x as f32 + 0.5) / W as f32 * 2.0 - 1.0;
                let ny = (y as f32 + 0.5) / H as f32 * 2.0 - 1.0;
                let a = inv.project_point3(Vec3::new(nx, ny, 0.0));
                let b = inv.project_point3(Vec3::new(nx, ny, 1.0));
                depth.push(match hit(a, (b - a).normalize()) {
                    Some(p) => {
                        let c = view_proj * p.extend(1.0);
                        c.z / c.w
                    }
                    None => 1.0,
                });
            }
        }
        depth
    }

    /// The nearest hit of a ray on the plane `n·p = k`, if ahead and
    /// accepted by `ok`.
    fn plane(o: Vec3, d: Vec3, n: Vec3, k: f32, ok: impl Fn(Vec3) -> bool) -> Option<(f32, Vec3)> {
        let t = (k - n.dot(o)) / n.dot(d);
        let p = o + d * t;
        (t > 0.0 && ok(p)).then_some((t, p))
    }

    fn nearest(hits: impl IntoIterator<Item = Option<(f32, Vec3)>>) -> Option<Vec3> {
        hits.into_iter()
            .flatten()
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|h| h.1)
    }

    /// A floor, and optionally a wall across it at z = −4 facing +z.
    fn floor_and_wall(wall: bool) -> (Vec<f32>, Mat4, AoProjection) {
        let (vp, proj) = camera(Vec3::new(0.3, 1.6, 0.0), Vec3::new(0.0, 0.2, -4.0));
        let depth = render(vp, |o, d| {
            nearest([
                plane(o, d, Vec3::Y, 0.0, |p| !wall || p.z > -4.0),
                if wall {
                    plane(o, d, Vec3::Z, -4.0, |p| p.y > 0.0)
                } else {
                    None
                },
            ])
        });
        (depth, vp, proj)
    }

    /// Each pixel's world position, where it has one.
    fn world(depth: &[f32], vp: Mat4) -> Vec<Option<Vec3>> {
        let inv = vp.inverse();
        (0..H)
            .flat_map(|y| (0..W).map(move |x| (x, y)))
            .map(|(x, y)| {
                let d = depth[(y * W + x) as usize];
                (d < 1.0).then(|| {
                    let nx = (x as f32 + 0.5) / W as f32 * 2.0 - 1.0;
                    let ny = (y as f32 + 0.5) / H as f32 * 2.0 - 1.0;
                    inv.project_point3(Vec3::new(nx, ny, d))
                })
            })
            .collect()
    }

    #[test]
    fn view_positions_come_back_from_depth() {
        let (vp, proj) = camera(Vec3::ZERO, -Vec3::Z);
        let depth = vec![0.0; (W * H) as usize];
        let img = DepthImage {
            width: W,
            height: H,
            depth: &depth,
            push: proj.push(H as u32),
        };
        // Points in view space project to a depth; the reference turns that
        // depth back into the same distance, and a pixel's direction into
        // the same x, y (screen axes: y down).
        for &(x, y, z) in &[(0.3, -0.2, -2.0), (-1.0, 0.5, -7.5), (4.0, 3.0, -150.0)] {
            let c = vp * glam::Vec4::new(x, y, z, 1.0);
            let d = c.z / c.w;
            assert!(
                (img.dist(d) - -z).abs() < 1e-3 * -z,
                "{d} -> {}",
                img.dist(d)
            );
            let ndc = c.xy() / c.w;
            let back = Vec3::new(ndc.x * img.push[2] * -z, ndc.y * img.push[3] * -z, -z);
            assert!((back - Vec3::new(x, -y, -z)).length() < 1e-4 * -z, "{back}");
        }
    }

    #[test]
    fn slice_arc_is_the_cosine_weighted_visibility_integral() {
        // ∫ cos(θ − n)·|sin θ| dθ over [h0, h1], by Simpson's rule.
        let simpson = |h0: f32, h1: f32, n: f32| {
            let steps = 2000;
            let h = (h1 - h0) / steps as f32;
            let f = |t: f32| (t - n).cos() * t.sin().abs();
            let mut s = f(h0) + f(h1);
            for i in 1..steps {
                s += f(h0 + i as f32 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
            }
            s * h / 3.0
        };
        for &n in &[0.0, 0.4, -0.9, 1.2] {
            for &(a, b) in &[(1.0, 1.0), (0.3, 0.8), (0.0, 0.5), (0.9, 0.0)] {
                let h0 = n - FRAC_PI_2 * a;
                let h1 = n + FRAC_PI_2 * b;
                let (h0, h1) = (h0.min(0.0), h1.max(0.0));
                let want = simpson(h0, h1, n);
                assert!(
                    (slice_arc(h0, h1, n) - want).abs() < 1e-4,
                    "n {n}, [{h0}, {h1}]"
                );
            }
        }
        // Face on and open: all of it; a wall at the view vector: half.
        assert!((slice_arc(-FRAC_PI_2, FRAC_PI_2, 0.0) - 1.0).abs() < 1e-6);
        assert!((slice_arc(-FRAC_PI_2, 0.0, 0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn an_open_floor_is_unoccluded() {
        let (depth, _, proj) = floor_and_wall(false);
        let img = DepthImage {
            width: W,
            height: H,
            depth: &depth,
            push: proj.push(H as u32),
        };
        let ao = img.ambient_occlusion();
        let floor: Vec<f32> = ao
            .iter()
            .zip(&depth)
            .filter(|(_, &d)| d < 1.0)
            .map(|(&a, _)| a)
            .collect();
        let min = floor.iter().copied().fold(1.0, f32::min);
        let mean = floor.iter().sum::<f32>() / floor.len() as f32;
        // The screen's edge pixels, whose denoise window is clamped, a
        // little less.
        let inner = (4..H - 4)
            .flat_map(|y| (4..W - 4).map(move |x| (y * W + x) as usize))
            .filter(|&i| depth[i] < 1.0)
            .map(|i| ao[i])
            .fold(1.0, f32::min);
        assert!(floor.len() > 1000);
        assert!(
            inner > 0.99 && min > 0.95 && mean > 0.999,
            "min {min} ({inner} inside), mean {mean}"
        );
    }

    #[test]
    fn a_wall_filling_the_view_is_open_to_the_screens_edges() {
        // Face on from 1.5 m: the view vector tilts up to 49° off the normal
        // towards the corners.
        let (vp, proj) = camera(Vec3::new(0.0, 1.0, 1.5), Vec3::new(0.0, 1.0, 0.0));
        let depth = render(vp, |o, d| nearest([plane(o, d, Vec3::Z, 0.0, |_| true)]));
        let img = DepthImage {
            width: W,
            height: H,
            depth: &depth,
            push: proj.push(H as u32),
        };
        let ao = img.ambient_occlusion();
        let min = ao.iter().copied().fold(1.0, f32::min);
        assert!(depth.iter().all(|&d| d < 1.0));
        assert!(min > 0.98, "min {min}");
    }

    #[test]
    fn a_slanted_wall_filling_the_view_is_open_to_the_screens_edges() {
        // Seen at 40°, so the view vector is off the normal everywhere, and
        // up to 86° off it at the screen's edge.
        let (vp, proj) = camera(Vec3::new(0.84, 1.0, 1.0), Vec3::new(0.0, 1.0, 0.0));
        let depth = render(vp, |o, d| nearest([plane(o, d, Vec3::Z, 0.0, |_| true)]));
        let img = DepthImage {
            width: W,
            height: H,
            depth: &depth,
            push: proj.push(H as u32),
        };
        let ao = img.ambient_occlusion();
        let min = ao.iter().copied().fold(1.0, f32::min);
        let mean = ao.iter().sum::<f32>() / ao.len() as f32;
        // The screen's edge on the grazing side (86° there), a little less:
        // its last column, and the next, whose denoise reaches it.
        let inner = (0..H)
            .flat_map(|y| (2..W - 2).map(move |x| (y * W + x) as usize))
            .map(|i| ao[i])
            .fold(1.0, f32::min);
        assert!(depth.iter().all(|&d| d < 1.0));
        assert!(
            inner > 0.99 && min > 0.9 && mean > 0.999,
            "min {min} ({inner} inside), mean {mean}"
        );
    }

    #[test]
    fn a_wall_darkens_the_floor_at_its_foot_and_nowhere_else() {
        let (depth, vp, proj) = floor_and_wall(true);
        let img = DepthImage {
            width: W,
            height: H,
            depth: &depth,
            push: proj.push(H as u32),
        };
        let ao = img.ambient_occlusion();
        let (mut foot, mut far) = (Vec::new(), Vec::new());
        for (a, p) in ao.iter().zip(world(&depth, vp)) {
            let Some(p) = p else { continue };
            // On the floor, by how far it is from the wall; on the wall, by
            // how high.
            let from_corner = if p.y.abs() < 1e-3 { p.z + 4.0 } else { p.y };
            if from_corner < 0.15 {
                foot.push(*a);
            } else if from_corner > 2.0 * AO_RADIUS {
                far.push(*a);
            }
        }
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        assert!(
            foot.len() > 20 && far.len() > 1000,
            "{} {}",
            foot.len(),
            far.len()
        );
        // A 90° inside corner sees half its hemisphere at the crease.
        assert!(
            mean(&foot) < 0.8 && mean(&foot) > 0.4,
            "foot {}",
            mean(&foot)
        );
        assert!(mean(&far) > 0.99, "far {}", mean(&far));
    }

    #[test]
    fn the_denoise_keeps_to_its_own_side_of_an_edge() {
        // Left half near (raw 0), right half far (raw 1).
        let (w, h) = (16, 8);
        let (_, proj) = camera(Vec3::ZERO, -Vec3::Z);
        let depth: Vec<f32> = (0..w * h)
            .map(|i| if i % w < w / 2 { 0.5 } else { 0.99 })
            .collect();
        let raw: Vec<f32> = depth
            .iter()
            .map(|&d| if d < 0.9 { 0.0 } else { 1.0 })
            .collect();
        let img = DepthImage {
            width: w,
            height: h,
            depth: &depth,
            push: proj.push(h as u32),
        };
        for y in 0..h {
            for x in 0..w {
                let want = raw[(y * w + x) as usize];
                assert!((img.denoise(&raw, x, y) - want).abs() < 1e-6, "({x}, {y})");
            }
        }
        // A flat run of one value stays that value.
        let flat = vec![0.4; (w * h) as usize];
        assert!((img.denoise(&flat, 3, 3) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn every_4x4_window_holds_each_jitter_once() {
        for (ox, oy) in [(0, 0), (1, 2), (3, 3), (-1, 5)] {
            let mut seen: Vec<f32> = (0..16)
                .map(|i| bayer4(ox + i % 4, oy + i / 4))
                .chain((0..16).map(|i| bayer4(oy + i / 4, ox + i % 4) + 100.0))
                .collect();
            seen.sort_by(f32::total_cmp);
            let want: Vec<f32> = (0..16)
                .map(|i| i as f32)
                .chain((0..16).map(|i| i as f32 + 100.0))
                .collect();
            assert_eq!(seen, want);
        }
    }

    #[test]
    fn the_ambient_fits_leave_open_surfaces_alone() {
        for albedo in [[0.0; 3], [0.2, 0.5, 0.9], [1.0; 3]] {
            // The fit's coefficients sum to 1 ± 1e-4.
            let open = multibounce(1.0, albedo);
            assert!(open.iter().all(|&c| (c - 1.0).abs() < 1e-3), "{open:?}");
            // Never darker than the visibility itself...
            let mb = multibounce(0.5, albedo);
            assert!(mb.iter().all(|&c| (0.5..=1.0).contains(&c)), "{mb:?}");
        }
        // ...and brighter surfaces bounce more back.
        assert!(multibounce(0.5, [0.9; 3])[0] > multibounce(0.5, [0.1; 3])[0]);
        for ndv in [0.05, 0.5, 1.0] {
            for rough in [0.04, 0.5, 1.0] {
                assert!((specular_occlusion(ndv, 1.0, rough) - 1.0).abs() < 1e-6);
                assert_eq!(specular_occlusion(ndv, 0.0, rough), 0.0);
            }
        }
    }

    /// The denoise rebuilds positions and normals exactly as gtao.comp does,
    /// or its surface weights would disagree with the AO they blur.
    #[test]
    fn the_passes_share_view_pos_and_depth_normal() {
        let body = |src: &str, sig: &str| -> String {
            let start = src.find(sig).unwrap_or_else(|| panic!("no {sig}"));
            let len = src[start..].find("\n}\n").expect("function end") + 3;
            src[start..start + len].to_string()
        };
        let (a, b) = (
            include_str!("../shaders/gtao.comp"),
            include_str!("../shaders/gtao_denoise.comp"),
        );
        for sig in [
            "vec3 view_pos(ivec2 p)",
            "vec3 depth_normal(ivec2 p, vec3 P)",
        ] {
            assert_eq!(body(a, sig), body(b, sig), "{sig}");
        }
    }

    /// The shaders repeat the reference's constants. A mismatch compiles
    /// and shades subtly wrong.
    #[test]
    fn the_shaders_declare_the_reference_constants() {
        let gtao = include_str!("../shaders/gtao.comp");
        for decl in [
            format!("const int AO_SLICES = {AO_SLICES};"),
            format!("const int AO_STEPS = {AO_STEPS};"),
            format!("const float AO_RADIUS = {AO_RADIUS:?};"),
            format!("const float AO_FALLOFF = {AO_FALLOFF:?};"),
            format!("const float AO_MIN_RADIUS_PX = {AO_MIN_RADIUS_PX:?};"),
            format!("const float AO_MAX_RADIUS_PX = {AO_MAX_RADIUS_PX:?};"),
            format!("const float AO_MIN_STEP_PX = {AO_MIN_STEP_PX:?};"),
        ] {
            assert!(gtao.contains(&decl), "gtao.comp lacks `{decl}`");
        }
        let denoise = include_str!("../shaders/gtao_denoise.comp");
        let decl = format!("const float AO_DENOISE_DEPTH = {AO_DENOISE_DEPTH:?};");
        assert!(denoise.contains(&decl), "gtao_denoise.comp lacks `{decl}`");
        let mesh = include_str!("../shaders/mesh.frag");
        for (name, [k, d]) in ["a", "b", "c"].iter().zip(AO_MULTIBOUNCE) {
            let sign = if d < 0.0 { '-' } else { '+' };
            let line = format!("vec3 {name} = {k:?} * albedo {sign} {:?};", d.abs());
            assert!(mesh.contains(&line), "mesh.frag lacks `{line}`");
        }
    }
}
