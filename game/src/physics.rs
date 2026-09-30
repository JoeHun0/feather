//! rapier as one raw ECS resource (§15), and each collider's surface (§20).

use crate::components::FIXED_DT;
use crate::Surface;
use bevy_ecs::prelude::*;
use glam::Vec3;
use rapier3d::prelude::{
    BroadPhaseBvh, CCDSolver, Collider, ColliderBuilder, ColliderHandle, ColliderSet,
    ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet, NarrowPhase,
    PhysicsPipeline, RigidBodySet, Vector,
};

/// The workspace pins glam 0.29 but rapier 0.35 builds on its own (newer) glam, so
/// `Vec3` and rapier's `Vector` are distinct types. Convert at the boundary.
pub fn to_rapier(v: Vec3) -> Vector {
    Vector::new(v.x, v.y, v.z)
}

pub fn from_rapier(v: Vector) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// All rapier state as one raw ECS resource (§15: raw rapier, not bevy_rapier).
/// There is no gravity here — the character controller is kinematic and the
/// player owns its vertical velocity — so the world only ever holds fixed level
/// colliders plus the player's position-based kinematic body.
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
            Vector::ZERO,
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
