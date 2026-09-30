//! Punctual lights (§12): the component, their per-frame extract, and
//! test-only reference copies of `mesh.frag`'s light maths.

use crate::components::{Hanging, Transform};
use crate::rope;
use bevy_ecs::prelude::*;
use feather_render::{Frustum, GpuLight};
use glam::{Mat4, Vec3};

/// A punctual light (§12). Position comes from the entity's `Transform`, so a
/// light can ride on geometry (a lamp that emits) or on a bare marker node.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    /// Linear colour.
    pub color: Vec3,
    pub intensity: f32,
    /// Distance at which the light reaches exactly zero.
    pub radius: f32,
    /// Physical size of the emitter, in metres (§12's sphere-light specular).
    /// Zero is a true point, which makes a near-singular highlight on smooth
    /// metal; the default is bulb-sized. Clamped to `[0, radius]`.
    pub source_radius: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: Vec3::ONE,
            intensity: 12.0,
            radius: 10.0,
            source_radius: 0.1,
        }
    }
}

/// Collect this frame's visible lights (§12).
///
/// Culled by **sphere**, not point: a light whose centre is off-screen still
/// lights what is on-screen if its radius reaches in, which a naive point test
/// gets wrong and which shows up as lights popping at the screen edge.
///
/// A light carried by something on a rope moves with it: it's placed with the
/// item's own matrix, interpolated by `alpha` like the item is drawn.
pub fn extract_lights(world: &mut World, frustum: &Frustum, alpha: f32) -> Vec<GpuLight> {
    let mut out = Vec::new();
    let mut push = |pos: Vec3, l: &PointLight| {
        if frustum.contains_sphere(pos, l.radius) {
            let c = l.color * l.intensity;
            out.push(GpuLight {
                pos_radius: [pos.x, pos.y, pos.z, l.radius],
                radiance_source: [c.x, c.y, c.z, l.source_radius],
            });
        }
    };
    let mut q = world.query::<(&Transform, &PointLight)>();
    for (t, l) in q.iter(world) {
        push(t.0.transform_point3(Vec3::ZERO), l);
    }
    let mut qh = world.query::<&Hanging>();
    for h in qh.iter(world) {
        if let Some((l, at)) = &h.light {
            let local = h.item.map_or(Mat4::IDENTITY, |(_, _, local)| local);
            let item = rope::item_matrix(&h.rope.interpolated(alpha), local);
            push(item.transform_point3(*at), l);
        }
    }
    out
}

/// Windowed inverse-square falloff — a **reference copy** of what `mesh.frag`
/// does. The window is what makes a light reach *exactly* zero at `radius`
/// instead of being clipped mid-gradient, which would show as a visible sphere
/// edge across the ground.
///
/// Test-only, and worth being clear about its limits: it pins the properties the
/// curve must have (zero at and past the radius, monotonic, finite at d = 0),
/// but it is a second implementation, so it cannot catch the shader drifting
/// away from it. Shading itself is still only verifiable on screen.
#[cfg(test)]
pub fn light_attenuation(distance: f32, radius: f32) -> f32 {
    if radius <= 0.0 || distance >= radius {
        return 0.0;
    }
    let t = (distance / radius).powi(4);
    let window = (1.0 - t).clamp(0.0, 1.0);
    window * window / (distance * distance + 1.0)
}

/// Scalar specular of one light as `mesh.frag`'s `punctual()` computes it
/// (F = 1, no falloff), treating the light as a sphere of `src` (Karis's
/// representative point). A **reference copy** like `light_attenuation`, with
/// the same limit: it pins the maths, not the shader.
#[cfg(test)]
pub fn sphere_light_specular(n: Vec3, v: Vec3, delta: Vec3, src: f32, roughness: f32) -> f32 {
    let dist = delta.length();
    let r = 2.0 * v.dot(n) * n - v; // reflect(-v, n)
    let center_to_ray = delta.dot(r) * r - delta;
    let t = (src / center_to_ray.length().max(1e-6)).clamp(0.0, 1.0);
    let ls = (delta + center_to_ray * t).normalize();
    let ndl = n.dot(ls).max(0.0);
    if ndl <= 0.0 {
        return 0.0;
    }
    let h = (v + ls).normalize();
    let ndv = n.dot(v).max(1e-4);
    let a = roughness * roughness;
    let a_wide = (a + src / (2.0 * dist.max(1e-4))).clamp(0.0, 1.0);
    let d = ggx_d(n.dot(h).max(0.0), a) * (a / a_wide).powi(2);
    d * smith_g(ndv, ndl, roughness) / (4.0 * ndv * ndl + 1e-4) * ndl
}

/// The same term for an infinitesimal point light, as `punctual()` was before
/// the sphere-light change — the baseline `src = 0` must reproduce.
#[cfg(test)]
pub fn point_light_specular(n: Vec3, v: Vec3, delta: Vec3, roughness: f32) -> f32 {
    let l = delta.normalize();
    let ndl = n.dot(l).max(0.0);
    if ndl <= 0.0 {
        return 0.0;
    }
    let h = (v + l).normalize();
    let ndv = n.dot(v).max(1e-4);
    let d = ggx_d(n.dot(h).max(0.0), roughness * roughness);
    d * smith_g(ndv, ndl, roughness) / (4.0 * ndv * ndl + 1e-4) * ndl
}

#[cfg(test)]
pub fn ggx_d(ndh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    a2 / (std::f32::consts::PI * d * d)
}

#[cfg(test)]
pub fn smith_g(ndv: f32, ndl: f32, rough: f32) -> f32 {
    let k = (rough + 1.0).powi(2) / 8.0;
    ndv / (ndv * (1.0 - k) + k) * (ndl / (ndl * (1.0 - k) + k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::rand01;

    #[test]
    fn sphere_light_with_zero_source_is_a_point_light() {
        // Random normals, view directions in the upper hemisphere, light offsets
        // and roughness: src = 0 must be exactly the old point light.
        let unit = |seed: u32| {
            let v = Vec3::new(
                rand01(seed) - 0.5,
                rand01(seed + 1) - 0.5,
                rand01(seed + 2) - 0.5,
            );
            v.normalize_or(Vec3::Y)
        };
        for i in 0..500u32 {
            let n = unit(i * 11);
            let mut v = unit(i * 11 + 3);
            if v.dot(n) < 0.0 {
                v = -v;
            }
            let delta = unit(i * 11 + 6) * (0.5 + 10.0 * rand01(i * 11 + 9));
            let rough = 0.04 + 0.96 * rand01(i * 11 + 10);
            let sphere = sphere_light_specular(n, v, delta, 0.0, rough);
            let point = point_light_specular(n, v, delta, rough);
            assert!(
                (sphere - point).abs() <= 1e-4 * point.abs().max(1.0),
                "case {i}: sphere {sphere} vs point {point}"
            );
        }
    }

    /// Smooth metal, viewed exactly along the light's mirror direction, 5 m
    /// away, at 45°.
    fn mirror_setup() -> (Vec3, Vec3, Vec3) {
        let n = Vec3::Y;
        let delta = Vec3::new(1.0, 1.0, 0.0).normalize() * 5.0;
        let l = delta.normalize();
        let v = 2.0 * l.dot(n) * n - l;
        (n, v, delta)
    }

    #[test]
    fn sphere_light_removes_the_smooth_metal_singularity() {
        let (n, v, delta) = mirror_setup();
        let peak = |src: f32| sphere_light_specular(n, v, delta, src, 0.04);
        let point = peak(0.0);
        // The singularity is real: a point light on roughness-0.04 metal peaks
        // far above anything a sized source produces...
        assert!(point.is_finite() && point > 1000.0, "point peak {point}");
        // ...and a size tames it, monotonically.
        let mut prev = point;
        for src in [0.02, 0.05, 0.1, 0.25, 0.7] {
            let p = peak(src);
            assert!(p.is_finite() && p > 0.0);
            assert!(p < prev, "peak rose at src {src}: {p} >= {prev}");
            prev = p;
        }
        assert!(
            point / peak(0.1) > 10.0,
            "a bulb-sized source should cut it >10x"
        );
    }

    #[test]
    fn sphere_light_highlight_matches_the_source_size() {
        // Turning the view by θ turns the reflection ray by θ. While that ray
        // still passes through the sphere the lobe is flat-topped (the
        // representative point lies on the ray), so the highlight is a disc.
        // Its half-width at half maximum should be about the sphere's angular
        // radius, asin(src / d): a mirror sphere's highlight matches the
        // lamp's own reflection.
        let (n, v, delta) = mirror_setup();
        let half_width = |src: f32| {
            let peak = sphere_light_specular(n, v, delta, src, 0.04);
            (1..20_000)
                .map(|i| i as f32 * 0.001_f32.to_radians())
                .find(|&a| {
                    let vr = glam::Quat::from_rotation_z(a) * v;
                    sphere_light_specular(n, vr, delta, src, 0.04) < peak * 0.5
                })
                .expect("the highlight ends somewhere")
        };
        let mut prev = 0.0;
        for src in [0.05, 0.1, 0.25, 0.7] {
            let w = half_width(src);
            let angular = (src / delta.length()).asin();
            assert!(w > prev, "highlight did not widen at src {src}");
            assert!(
                (0.9..1.3).contains(&(w / angular)),
                "src {src}: half-width {:.3}° vs source {:.3}°",
                w.to_degrees(),
                angular.to_degrees()
            );
            prev = w;
        }
    }

    #[test]
    fn sphere_light_roughly_conserves_energy() {
        // The (α/α')² normalisation exists to keep a sized light about as bright
        // overall as the point it replaces. Integrate the specular lobe over view
        // directions around the mirror direction and compare. This pins the two
        // ways it goes wrong: widening twice (D at α' as well) keeps ~1% of the
        // energy, and a src/3d widening overshoots ~3x on smooth metal.
        let (n, _, delta) = mirror_setup();
        let l = delta.normalize();
        let vm = 2.0 * l.dot(n) * n - l;
        let t1 = vm.cross(Vec3::Z).normalize();
        let t2 = vm.cross(t1);
        let energy = |src: f32, rough: f32, cap_deg: f32| {
            let (nt, nphi) = (600, 64);
            let cap = cap_deg.to_radians();
            let mut sum = 0.0;
            for i in 0..nt {
                let th = (i as f32 + 0.5) / nt as f32 * cap;
                for j in 0..nphi {
                    let ph = (j as f32 + 0.5) / nphi as f32 * std::f32::consts::TAU;
                    let v = vm * th.cos() + (t1 * ph.cos() + t2 * ph.sin()) * th.sin();
                    if v.dot(n) <= 0.0 {
                        continue;
                    }
                    let w = th.sin() * (cap / nt as f32) * (std::f32::consts::TAU / nphi as f32);
                    sum += sphere_light_specular(n, v, delta, src, rough) * w;
                }
            }
            sum
        };
        for (rough, cap, lo, hi) in [(0.04, 40.0, 0.9, 1.6), (0.3, 89.0, 0.85, 1.2)] {
            let point = energy(0.0, rough, cap);
            for src in [0.1, 0.25, 0.7] {
                let ratio = energy(src, rough, cap) / point;
                assert!(
                    (lo..hi).contains(&ratio),
                    "rough {rough} src {src}: {ratio:.2}x the point light's energy"
                );
            }
        }
    }

    #[test]
    fn attenuation_reaches_zero_at_the_radius() {
        let r = 10.0f32;
        // Exactly zero at and past the radius: otherwise the cutoff lands
        // mid-gradient and reads as a sphere edge on the ground.
        assert_eq!(light_attenuation(r, r), 0.0);
        assert_eq!(light_attenuation(r + 1.0, r), 0.0);
        assert_eq!(light_attenuation(100.0, r), 0.0);
        // Finite at the centre rather than exploding.
        assert!(light_attenuation(0.0, r).is_finite());
        assert!(light_attenuation(0.0, r) > 0.0);
        // Monotonically decreasing.
        let mut prev = f32::INFINITY;
        for i in 0..=100 {
            let d = r * i as f32 / 100.0;
            let a = light_attenuation(d, r);
            assert!(a <= prev + 1e-6, "attenuation rose at d={d}");
            assert!(a >= 0.0);
            prev = a;
        }
        // A degenerate radius lights nothing instead of dividing by zero.
        assert_eq!(light_attenuation(1.0, 0.0), 0.0);
    }
}
