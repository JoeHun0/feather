//! Test helpers more than one module's tests use: a level of boxes with
//! the player on it, the fixed tick the schedule runs, and a scene node.

use crate::controller::{player_readback, player_target, InputState, Player, GROUND_Y};
use crate::physics::Physics;
use glam::{Mat4, Vec3};

/// Ground (same 80x80x1 box as the app) plus `boxes` as `(center, size)`, and a
/// player standing at `start`. Mirrors the app's setup, including the warm-up
/// step that publishes the level to the broad-phase BVH.
pub fn setup(boxes: &[(Vec3, Vec3)], start: Vec3) -> (Physics, Player) {
    let mut physics = Physics::new();
    let player = Player::new(&mut physics, start);
    physics.add_static_box(
        Vec3::new(0.0, GROUND_Y - 0.5, 0.0),
        Vec3::new(80.0, 1.0, 80.0),
    );
    for &(c, s) in boxes {
        physics.add_static_box(c, s);
    }
    physics.step();
    (physics, player)
}

/// One fixed tick, mirroring the schedule's bracket: ECS→rapier, the step,
/// then rapier→ECS. Drives the same functions the systems do.
pub fn step(p: &mut Player, ph: &mut Physics, wish: Vec3, jump: bool, vgo: f32, noclip: bool) {
    let input = InputState {
        wish,
        jump,
        fire: false,
        vertical: vgo,
        noclip,
    };
    player_target(p, ph, &input);
    ph.step();
    player_readback(p, ph);
}

pub fn run(p: &mut Player, ph: &mut Physics, ticks: u32, wish: Vec3) {
    for _ in 0..ticks {
        step(p, ph, wish, false, 0.0, false);
    }
}

/// A scene node carrying the cube mesh, and a prefab if `prefab` names one.
pub fn node(prefab: Option<&str>, params: serde_json::Value) -> feather_assets::SceneNode {
    feather_assets::SceneNode {
        mesh: Some(0),
        transform: Mat4::IDENTITY,
        prefab: prefab.map(|id| feather_assets::PrefabSpec {
            id: id.to_string(),
            params,
        }),
    }
}
