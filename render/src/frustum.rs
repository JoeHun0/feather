//! View-frustum culling (§8): the planes of a view-proj matrix and the
//! bounding spheres tested against them.

use glam::{Mat4, Vec3, Vec4};

/// View frustum as six inward-facing planes (Gribb–Hartmann, from a view-proj
/// matrix). Used per view (§8): the camera for the main pass, the sun light ortho
/// for the shadow pass. Matrix-agnostic, so it works for the camera's y-flipped
/// projection and the light ortho alike.
pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    pub fn from_view_proj(m: &Mat4) -> Self {
        // glam is column-major: element (row r, col c) = cols[c * 4 + r].
        let c = m.to_cols_array();
        let row = |r: usize| Vec4::new(c[r], c[4 + r], c[8 + r], c[12 + r]);
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
        // Vulkan clip z ∈ [0,1]: the near plane is r2 (not r3 + r2).
        let raw = [
            r3 + r0, // left
            r3 - r0, // right
            r3 + r1, // bottom
            r3 - r1, // top
            r2,      // near
            r3 - r2, // far
        ];
        let mut planes = [Vec4::ZERO; 6];
        for (i, p) in raw.into_iter().enumerate() {
            let len = p.truncate().length();
            planes[i] = if len > 0.0 { p / len } else { p };
        }
        Self { planes }
    }

    /// The same frustum with its near plane dropped, for culling **shadow
    /// casters** (§11 pancaking). Geometry nearer the sun than a cascade's near
    /// plane still shadows the cascade: the shadow pipeline's depth clamp
    /// flattens it onto depth 0 instead of clipping it. An ortho's side planes
    /// are parallel to the light, so anything outside them could never shadow
    /// the box and stays culled; only the near plane is at fault.
    pub fn without_near(mut self) -> Self {
        self.planes[4] = Vec4::new(0.0, 0.0, 0.0, 1.0); // every point passes
        self
    }

    /// True unless the sphere is entirely behind some plane (i.e. culled).
    pub fn contains_sphere(&self, center: Vec3, radius: f32) -> bool {
        self.planes
            .iter()
            .all(|p| p.x * center.x + p.y * center.y + p.z * center.z + p.w >= -radius)
    }
}

/// World bounding sphere of `local` under `model`: the centre transforms with it,
/// and the radius scales by the model's largest axis scale — conservative under
/// non-uniform scale, which is what culling wants.
pub fn world_sphere(model: &Mat4, local: (Vec3, f32)) -> (Vec3, f32) {
    let s = model
        .x_axis
        .truncate()
        .length()
        .max(model.y_axis.truncate().length())
        .max(model.z_axis.truncate().length());
    (model.transform_point3(local.0), local.1 * s)
}
