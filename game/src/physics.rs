//! rapier as one raw ECS resource (§15), and each collider's surface (§20).

use crate::components::FIXED_DT;
use crate::Surface;
use bevy_ecs::prelude::*;
use glam::{Mat4, Quat, Vec3};
use rapier3d::prelude::{
    BroadPhaseBvh, CCDSolver, Collider, ColliderBuilder, ColliderHandle, ColliderSet,
    ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet, NarrowPhase,
    PhysicsPipeline, Pose, RigidBodyBuilder, RigidBodyHandle, RigidBodySet, Rotation, Vector,
};

/// Gravity on dynamic bodies (§15), m/s². Only they feel it: fixed and
/// kinematic bodies ignore world gravity, and the player's controller keeps
/// its own, gamier `GRAVITY`.
pub const BODY_GRAVITY: f32 = 9.81;
/// rapier has no rolling resistance, so a lying barrel nudged on flat ground
/// would roll on for good. A little damping settles bodies the way friction
/// against the air and the ground would.
pub const BODY_LINEAR_DAMPING: f32 = 0.1;
pub const BODY_ANGULAR_DAMPING: f32 = 0.8;

/// The workspace pins glam 0.29 but rapier 0.35 builds on its own (newer) glam, so
/// `Vec3` and rapier's `Vector` are distinct types. Convert at the boundary.
pub fn to_rapier(v: Vec3) -> Vector {
    Vector::new(v.x, v.y, v.z)
}

pub fn from_rapier(v: Vector) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

pub fn quat_to_rapier(q: Quat) -> Rotation {
    Rotation::from_xyzw(q.x, q.y, q.z, q.w)
}

pub fn quat_from_rapier(q: Rotation) -> Quat {
    Quat::from_xyzw(q.x, q.y, q.z, q.w)
}

/// All rapier state as one raw ECS resource (§15: raw rapier, not bevy_rapier).
/// It holds the fixed level colliders, the player's position-based kinematic
/// body (the controller owns its vertical velocity, so world gravity never
/// touches it), and the `dynamic` props, the only bodies gravity moves.
#[derive(Resource)]
pub struct Physics {
    pub pipeline: PhysicsPipeline,
    pub params: IntegrationParameters,
    pub islands: IslandManager,
    pub broad_phase: BroadPhaseBvh,
    pub narrow_phase: NarrowPhase,
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub ccd: CCDSolver,
}

impl Default for Physics {
    fn default() -> Self {
        Self::new()
    }
}

impl Physics {
    pub fn new() -> Self {
        Self {
            pipeline: PhysicsPipeline::new(),
            params: IntegrationParameters {
                dt: FIXED_DT,
                ..Default::default()
            },
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd: CCDSolver::new(),
        }
    }

    /// Advance the rapier world by one `FIXED_DT`. This is also what refreshes the
    /// broad-phase BVH the character controller's shape-casts query, so colliders
    /// added since the last step are invisible to the controller until this runs.
    pub fn step(&mut self) {
        self.pipeline.step(
            Vector::new(0.0, -BODY_GRAVITY, 0.0),
            &self.params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd,
            &(),
            &(),
        );
    }

    /// A fixed triangle-mesh collider for loaded scene geometry. `verts` must
    /// already be in **world** space with the collider left at the identity: glTF
    /// nodes routinely carry non-uniform scale, which a rapier `Pose` cannot
    /// express. Returns `None` (and logs) for degenerate meshes rather than
    /// bringing the level down.
    pub fn add_static_trimesh(
        &mut self,
        verts: Vec<Vector>,
        tris: Vec<[u32; 3]>,
    ) -> Option<ColliderHandle> {
        match ColliderBuilder::trimesh(verts, tris) {
            Ok(b) => Some(self.colliders.insert(b)),
            Err(e) => {
                eprintln!("scene collider skipped: {e}");
                None
            }
        }
    }

    /// A static convex hull of world-space points (§15 collision proxies).
    /// `None` when there is no hull (too few or collinear points).
    pub fn add_static_hull(&mut self, points: &[Vector]) -> Option<ColliderHandle> {
        ColliderBuilder::convex_hull(points).map(|b| self.colliders.insert(b))
    }

    /// A fixed axis-aligned box collider (static level geometry).
    pub fn add_static_box(&mut self, center: Vec3, size: Vec3) -> ColliderHandle {
        let h = size * 0.5;
        self.colliders
            .insert(ColliderBuilder::cuboid(h.x, h.y, h.z).translation(to_rapier(center)))
    }

    /// A fixed cuboid round the local box `lo..hi` placed by `transform`:
    /// its rotation turns the cuboid and its scale stretches it. `None` when
    /// the matrix is sheared (non-uniform scale under a rotation), which a
    /// cuboid can't follow.
    pub fn add_static_obb(
        &mut self,
        transform: Mat4,
        lo: Vec3,
        hi: Vec3,
    ) -> Option<ColliderHandle> {
        let (scale, rot, _) = transform.to_scale_rotation_translation();
        let rebuilt =
            Mat4::from_scale_rotation_translation(scale, rot, transform.w_axis.truncate());
        let tolerance = 1e-5 * scale.abs().max_element().max(1.0);
        if !rebuilt.abs_diff_eq(transform, tolerance) {
            return None;
        }
        let h = (hi - lo) * 0.5 * scale.abs();
        let center = transform.transform_point3((lo + hi) * 0.5);
        Some(
            self.colliders.insert(
                ColliderBuilder::cuboid(h.x, h.y, h.z)
                    .position(Pose::from_parts(to_rapier(center), quat_to_rapier(rot))),
            ),
        )
    }

    /// A dynamic body (§15) at `pos`/`rot`, colliding as the convex hull of
    /// `points`, which are in the body's own frame (scaled, not rotated).
    /// `None` when the points have no hull.
    pub fn add_dynamic_hull(
        &mut self,
        points: &[Vector],
        pos: Vec3,
        rot: Quat,
        mass: Option<f32>,
        density: f32,
    ) -> Option<(RigidBodyHandle, ColliderHandle)> {
        let collider = ColliderBuilder::convex_hull(points)?;
        Some(self.add_dynamic(collider, pos, rot, mass, density))
    }

    /// A dynamic body (§15) at `pos`/`rot` with `collider`, which is in the
    /// body's own frame. Its mass is `mass` kg when given, else the shape's
    /// volume times `density` (kg/m³). CCD keeps a knocked prop from passing
    /// through thin walls.
    pub fn add_dynamic(
        &mut self,
        collider: ColliderBuilder,
        pos: Vec3,
        rot: Quat,
        mass: Option<f32>,
        density: f32,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let collider = match mass {
            Some(m) => collider.mass(m),
            None => collider.density(density),
        };
        let body = self.bodies.insert(
            RigidBodyBuilder::dynamic()
                .pose(Pose::from_parts(to_rapier(pos), quat_to_rapier(rot)))
                .linear_damping(BODY_LINEAR_DAMPING)
                .angular_damping(BODY_ANGULAR_DAMPING)
                .ccd_enabled(true),
        );
        let collider = self
            .colliders
            .insert_with_parent(collider, body, &mut self.bodies);
        (body, collider)
    }

    /// Take a body and its colliders out of the world.
    pub fn remove_body(&mut self, body: RigidBodyHandle) {
        self.bodies.remove(
            body,
            &mut self.islands,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            true,
        );
    }
}

/// A collider's §20 surface lives in its rapier `user_data`, as the
/// surface's index. Concrete is 0, rapier's default, so a collider nobody
/// tagged is concrete. `user_data` has no other use yet; if something comes
/// to need collider → entity, the entity belongs there and the surface moves
/// to a component.
pub fn tag_surface(c: &mut Collider, surface: Surface) {
    c.user_data = surface.index() as u128;
}

pub fn collider_surface(c: &Collider) -> Surface {
    Surface::from_index(usize::try_from(c.user_data).unwrap_or(usize::MAX))
}
