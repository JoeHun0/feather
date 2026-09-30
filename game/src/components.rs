//! ECS components and resources (§1), and the plain systems over them: the
//! orb demo's drift and the ropes' fixed step.

use crate::lights::PointLight;
use crate::rope;
use bevy_ecs::prelude::*;
use feather_render::MeshId;
use glam::{Mat4, Vec3};
use rapier3d::prelude::ColliderHandle;

pub const GRID: i32 = 10; // GRID^3 entities
pub const BOUND: f32 = 9.0;

/// The fixed step (§4): the schedule advances in whole steps of this, and the
/// app's frame loop interpolates the remainder.
pub const FIXED_DT: f32 = 1.0 / 60.0;

#[derive(Component)]
pub struct Position(pub Vec3);
/// Position at the start of the latest completed fixed step (the other endpoint
/// extract interpolates from). Kept in the same wrapped space as `Position`.
#[derive(Component)]
pub struct PrevPosition(pub Vec3);
#[derive(Component)]
pub struct Velocity(pub Vec3);
/// Accumulated Y rotation (radians), advanced by `Spin` each fixed step.
#[derive(Component)]
pub struct Rotation(pub f32);
/// `Rotation` at the start of the latest completed fixed step.
#[derive(Component)]
pub struct PrevRotation(pub f32);
#[derive(Component)]
pub struct Spin(pub f32); // angular velocity, rad/s
/// Per-entity local scale, applied inside the model matrix. Drifting demo
/// entities use a uniform scale; static level pieces use non-uniform sizes.
#[derive(Component)]
pub struct Scale(pub Vec3);
/// World transform for **static** scene geometry loaded from glTF (§18). glTF
/// nodes carry arbitrary rotations, which the demo's
/// `Position`/`Rotation(yaw)`/`Scale` triple cannot express. Static, so there is
/// no prev/curr pair — nothing to interpolate.
#[derive(Component)]
pub struct Transform(pub Mat4);
#[derive(Component)]
pub struct Mesh(pub MeshId);
#[derive(Component)]
pub struct Material(pub u32); // material_id into the renderer's material table
/// Opt-out marker: this entity is not rendered into the sun shadow map (§11).
/// Everything casts by default. The demo's flat ground carries it — a flat slab
/// casts nothing useful (nothing is beneath it) yet rasterizes the *entire*
/// shadow map, which dominates that pass's fill cost. This is deliberately
/// per-entity rather than a rule about "ground": terrain with relief has to cast
/// (hills shadow valleys), and then it simply doesn't carry this marker.
#[derive(Component)]
pub struct NoShadowCast;
/// This entity's collider in the rapier world (§15: entities carry their
/// handles). Static level geometry for now — nothing reads it back yet, since
/// statics never move, but it is the handle a future sync system would use.
#[derive(Component)]
pub struct ColliderRef(#[allow(dead_code)] pub ColliderHandle);
#[derive(Resource, Default)]
pub struct FrameCount(pub u64);

/// Number of shared palette materials generated for procedural meshes.
pub const PALETTE: u32 = 24;

/// Mesh registry slots that always exist, before any loaded scene's meshes.
/// `App::new` spawns the orb demo against SPHERE/CUBE, and the level pieces use
/// LEVEL_CUBE, so these indices must match the order they are registered in.
pub const MESH_SPHERE: u32 = 0;
pub const MESH_CUBE: u32 = 1;
pub const MESH_LEVEL_CUBE: u32 = 2;
/// The unit cylinder a rope's segments are drawn with (§15, `rope`).
pub const MESH_ROPE: u32 = 3;
pub const MESH_BUILTIN_COUNT: u32 = 4;

/// Something hanging on a rope (§15): the simulated rope, and the item on its
/// end (mesh, material, and its authored rotation and scale).
#[derive(Component)]
pub struct Hanging {
    pub rope: rope::Rope,
    pub item: Option<(MeshId, u32, Mat4)>,
    pub casts: bool,
    /// A light carried by the item (§12), at a point in the item's own
    /// space: a lantern's flame.
    pub light: Option<(PointLight, Vec3)>,
}

/// The level's wind (§13's `environment`), which ropes sway in.
#[derive(Resource, Default)]
pub struct LevelWind(pub rope::Wind);

/// Every rope, one fixed step on. Sim time is the step count, not the wall
/// clock, so a run replays the same.
pub fn ropes(mut q: Query<&mut Hanging>, wind: Res<LevelWind>, frame: Res<FrameCount>) {
    let t = frame.0 as f32 * FIXED_DT;
    for mut h in &mut q {
        h.rope.step(FIXED_DT, t, &wind.0);
    }
}

/// One fixed step: snapshot the current state into `Prev*`, then advance
/// position by velocity and the spin angle by angular velocity. Runs at
/// `FIXED_DT`, so motion is now framerate-independent.
pub fn integrate(
    mut q: Query<(
        &mut Position,
        &mut PrevPosition,
        &Velocity,
        &mut Rotation,
        &mut PrevRotation,
        &Spin,
    )>,
) {
    for (mut pos, mut prev, vel, mut rot, mut prev_rot, spin) in &mut q {
        prev.0 = pos.0;
        prev_rot.0 = rot.0;
        pos.0 += vel.0 * FIXED_DT;
        rot.0 += spin.0 * FIXED_DT;
        // Toroidal wrap. Shifting `prev` by the same amount keeps the segment
        // extract interpolates over local, so no full-width streak on the wrap
        // frame. (The angle accumulates unbounded — its matrix is periodic, so
        // it needs no wrap.)
        wrap_axis(&mut pos.0.x, &mut prev.0.x);
        wrap_axis(&mut pos.0.y, &mut prev.0.y);
        wrap_axis(&mut pos.0.z, &mut prev.0.z);
    }
}

/// Wrap one axis into `[-BOUND, BOUND]`, moving `prev` with it so the current↔
/// prev delta the renderer interpolates stays small across the seam.
pub fn wrap_axis(cur: &mut f32, prev: &mut f32) {
    if *cur > BOUND {
        *cur -= 2.0 * BOUND;
        *prev -= 2.0 * BOUND;
    } else if *cur < -BOUND {
        *cur += 2.0 * BOUND;
        *prev += 2.0 * BOUND;
    }
}

pub fn tick(mut frame: ResMut<FrameCount>) {
    frame.0 += 1;
}

pub fn rand01(seed: u32) -> f32 {
    let mut h = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    h ^= h >> 15;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    (h & 0x00ff_ffff) as f32 / 0x0100_0000 as f32
}
