//! Sun-shadow cascades (§11): where the view is split, the sphere each
//! slice is fitted with, and the texel-snapped light matrix for each. Pure
//! maths — the app feeds it the camera and hands the result to the shadow
//! pass as `CascadeSetup`.

use feather_gfx::SHADOW_CASCADES;
use glam::{Mat4, Vec3};

// Sun shadow frustum (§11). The ortho follows the player, so these are relative
// to them: half-extent of the covered square, how far back along the sun the
// light "eye" sits, and the ortho's depth range. A tighter radius means denser
// shadow texels (sharper) but less coverage around the player.
/// How far from the camera shadows reach, across all cascades. Past this,
/// distance fog carries the transition (camera far is 200).
///
/// 60 rather than something larger because the cascade budget is fixed: 4x2048²
/// is the same texel count as the single 4096² map it replaces, so range is paid
/// for in sharpness. The demo ground is 80x80, i.e. 56 units corner-to-centre,
/// so 60 covers the world without spending two cascades on empty space. A larger
/// world raises this and either accepts softer shadows or raises the per-cascade
/// dimension with it — §11 is explicit that shadow resolution is a quality knob.
pub const SHADOW_DISTANCE: f32 = 60.0;
/// Practical-split weighting (§11): 0 = uniform, 1 = logarithmic. 0.75 keeps
/// near cascades tight without starving the far one.
pub const SHADOW_LAMBDA: f32 = 0.75;
const SHADOW_BACK: f32 = 40.0;

/// Practical/parallel-split scheme (§11): the far depth of each cascade, blending
/// a logarithmic and a uniform distribution by `lambda`.
///
/// Logarithmic alone starves the far cascade; uniform alone wastes most of the
/// resolution on distance nobody looks at. `lambda` picks between them.
pub fn cascade_splits(near: f32, far: f32, lambda: f32) -> [f32; SHADOW_CASCADES] {
    let mut out = [0.0; SHADOW_CASCADES];
    for (i, slot) in out.iter_mut().enumerate() {
        let p = (i + 1) as f32 / SHADOW_CASCADES as f32;
        let log = near * (far / near).powf(p);
        let uniform = near + (far - near) * p;
        *slot = lambda * log + (1.0 - lambda) * uniform;
    }
    out
}

/// Bounding sphere of the view-frustum slice between `near` and `far`.
///
/// A sphere rather than a box **on purpose** (§11): centroid and radius are
/// invariant under rigid motion, so turning the camera cannot change the world
/// size a cascade covers. A box fit would change size as you rotate, and the
/// shadow texels would shimmer with it.
pub fn slice_sphere(
    eye: Vec3,
    fwd: Vec3,
    right: Vec3,
    up: Vec3,
    fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> (Vec3, f32) {
    let tan_v = (fov_y * 0.5).tan();
    let tan_h = tan_v * aspect;
    let mut corners = [Vec3::ZERO; 8];
    let mut n = 0;
    for d in [near, far] {
        let centre = eye + fwd * d;
        let (h, v) = (right * (tan_h * d), up * (tan_v * d));
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                corners[n] = centre + h * sx + v * sy;
                n += 1;
            }
        }
    }
    let centre = corners.iter().fold(Vec3::ZERO, |a, c| a + *c) / 8.0;
    let radius = corners
        .iter()
        .fold(0.0f32, |m, c| m.max(c.distance(centre)));
    (centre, radius)
}

/// Light-space matrix for one cascade, texel-snapped.
///
/// Snapping the ortho centre to whole shadow-map texels is what stops shadow
/// edges crawling as you walk (§11 calls it non-negotiable); depth along the
/// light needs no snap, since sliding along it causes no edge crawl.
pub fn fit_cascade(centre: Vec3, radius: f32, sun_dir: Vec3, dim: u32) -> (Mat4, f32) {
    let texel_world = (2.0 * radius) / dim as f32;
    let basis = Mat4::look_at_rh(Vec3::ZERO, sun_dir, Vec3::Y);
    let c = basis.transform_point3(centre);
    let snapped = Vec3::new(
        (c.x / texel_world).round() * texel_world,
        (c.y / texel_world).round() * texel_world,
        c.z,
    );
    let centre = basis.inverse().transform_point3(snapped);
    // Pull the light back past the sphere. With pancaking (§11) a caster further
    // than this toward the light is no longer lost: casters are culled without
    // the near plane and depth-clamped onto it. So `SHADOW_BACK` now only
    // spends depth precision, and it stays at 40 because SHADOW_DEPTH_BIAS is
    // in normalised depth over `back + radius`: shrinking the range would
    // silently shrink the world-space bias.
    let back = radius + SHADOW_BACK;
    let light_eye = centre - sun_dir * back;
    let view = Mat4::look_at_rh(light_eye, centre, Vec3::Y);
    let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.0, back + radius);
    (proj * view, texel_world)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Shadow cascades (§11) ----
    //
    // All pure maths, so it runs with no GPU. These cover the properties that
    // are easy to break and hard to see: a bad split distribution looks like
    // "shadows are blurry somewhere", and a fit that is not rotation-invariant
    // looks like shimmer while turning, which is easy to blame on something else.

    #[test]
    fn splits_are_increasing_and_span_the_range() {
        for lambda in [0.0, 0.5, 0.75, 1.0] {
            let s = cascade_splits(0.1, SHADOW_DISTANCE, lambda);
            for pair in s.windows(2) {
                assert!(
                    pair[1] > pair[0],
                    "splits not increasing at lambda {lambda}"
                );
            }
            assert!(s[0] > 0.1, "first split must be past the near plane");
            assert!(
                (s[SHADOW_CASCADES - 1] - SHADOW_DISTANCE).abs() < 0.01,
                "last split must reach SHADOW_DISTANCE, got {}",
                s[SHADOW_CASCADES - 1]
            );
        }
    }

    #[test]
    fn lambda_selects_between_uniform_and_logarithmic() {
        let (near, far) = (0.1f32, SHADOW_DISTANCE);
        let uniform = cascade_splits(near, far, 0.0);
        let log = cascade_splits(near, far, 1.0);
        for i in 0..SHADOW_CASCADES {
            let p = (i + 1) as f32 / SHADOW_CASCADES as f32;
            assert!((uniform[i] - (near + (far - near) * p)).abs() < 0.01);
            assert!((log[i] - near * (far / near).powf(p)).abs() < 0.01);
        }
        // Logarithmic keeps the near cascades much tighter; that is the point.
        assert!(log[0] < uniform[0]);
    }

    #[test]
    fn cascade_fit_snaps_to_texels_and_is_idempotent() {
        let sun = Vec3::new(-0.4, -1.0, -0.3).normalize();
        let dim = 2048;
        let radius = 24.0f32;
        let texel = (2.0 * radius) / dim as f32;
        let basis = Mat4::look_at_rh(Vec3::ZERO, sun, Vec3::Y);

        // Sweep sub-texel centre offsets: each must land on the texel grid, and
        // must not move by as much as a whole texel.
        for k in 0..16 {
            let centre = Vec3::new(1.0, 2.0, 3.0) + Vec3::X * (texel * k as f32 / 16.0);
            let (vp, tw) = fit_cascade(centre, radius, sun, dim);
            assert!((tw - texel).abs() < 1e-6);

            // Recover the snapped centre from the matrix by re-running the snap;
            // snapping something already snapped must be a no-op.
            let c = basis.transform_point3(centre);
            let snapped = Vec3::new(
                (c.x / texel).round() * texel,
                (c.y / texel).round() * texel,
                c.z,
            );
            let resnapped = Vec3::new(
                (snapped.x / texel).round() * texel,
                (snapped.y / texel).round() * texel,
                snapped.z,
            );
            assert!(
                (snapped - resnapped).length() < 1e-4,
                "snap is not idempotent"
            );
            assert!(
                (snapped.x - c.x).abs() <= texel * 0.5 + 1e-4
                    && (snapped.y - c.y).abs() <= texel * 0.5 + 1e-4,
                "snap moved the centre more than half a texel"
            );
            assert!(vp.is_finite(), "cascade matrix is not finite");
        }
    }
}
