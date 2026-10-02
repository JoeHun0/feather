//! The first-person controller (§15): the player's fixed-step state, the
//! render-rate look, and the systems that bracket the physics step.

use crate::components::FIXED_DT;
use crate::physics::{collider_surface, from_rapier, to_rapier, Physics};
use crate::Surface;
use bevy_ecs::prelude::*;
use glam::{Mat4, Vec3};
use rapier3d::control::{
    CharacterAutostep, CharacterCollision, CharacterLength, KinematicCharacterController,
};
use rapier3d::geometry::ContactManifold;
use rapier3d::parry::bounding_volume::BoundingVolume;
use rapier3d::parry::query::{DefaultQueryDispatcher, PersistentQueryDispatcher, ShapeCastOptions};
use rapier3d::parry::shape::{Ball, Shape};
use rapier3d::prelude::{
    ColliderBuilder, ColliderHandle, Pose, QueryFilter, QueryPipeline, RigidBodyBuilder,
    RigidBodyHandle, Vector,
};

// First-person controller (§15). The controller owns vertical velocity: gravity
// each tick, zeroed when grounded, jump on the press edge when grounded. rapier's
// `KinematicCharacterController` applies no gravity of its own. Look is
// render-rate; body position is fixed-step + interpolated.
pub const GROUND_Y: f32 = -9.0; // top surface of the ground collider (base of the orb column)
pub const EYE_HEIGHT: f32 = 1.6;
pub const PLAYER_RADIUS: f32 = 0.35; // capsule radius
pub const PLAYER_HEIGHT: f32 = 1.8; // total height, feet to crown (capsule caps included)
pub const CAPSULE_HALF_HEIGHT: f32 = (PLAYER_HEIGHT - 2.0 * PLAYER_RADIUS) * 0.5; // `capsule_y` arg: half the straight section
pub const MOVE_SPEED: f32 = 8.0; // target ground speed, units/s
pub const MOVE_ACCEL: f32 = 14.0; // how fast horizontal velocity chases the target
pub const GRAVITY: f32 = 26.0;
pub const JUMP_SPEED: f32 = 9.0; // ~1.5 units apex
/// What the player weighs to a dynamic prop it walks into (§15), kg: the
/// blocked motion is shared out by the two masses.
pub const PLAYER_MASS: f32 = 80.0;
/// The most the player can push with, N: about what feet hold on the
/// ground. A prop whose friction on the ground is more stays put (§15).
pub const PUSH_FORCE: f32 = 600.0;
pub const FLY_SPEED: f32 = 14.0; // noclip movement speed
/// How far below a grounded player the surface probe looks (§20). The
/// controller holds the capsule ~2 cm off the ground and counts contacts
/// within ~7 cm as grounded, so 10 cm finds whatever it counted.
pub const GROUND_PROBE: f32 = 0.1;
/// How close a downward-facing surface must be to count as touching the
/// capsule, for the overhang clip in `player_target` (§15), and so how far
/// short of one a grounded player stops. The controller keeps the capsule
/// ~2 cm off everything.
pub const OVERHANG_REACH: f32 = 0.05;

/// The player's simulation state, as a component on the player entity (§1: the
/// `World` is the single source of truth). `pos` is the feet position — advanced
/// at the fixed step, then interpolated for the camera. `vel` is owned by the
/// controller: gravity and jump live here, not in a physics solver. The body is a
/// position-based kinematic rigid body with a capsule collider in `Physics`;
/// `controller` resolves each tick's desired motion against the level.
///
/// Look angles are deliberately *not* here — they update at render rate, so they
/// live in [`Look`].
#[derive(Component)]
pub struct Player {
    pub pos: Vec3,
    pub prev_pos: Vec3,
    pub vel: Vec3,
    pub on_ground: bool,
    /// What the feet were last on (§20): refreshed every grounded tick, kept
    /// while airborne, so the steps after a landing use the new ground.
    pub surface: Surface,
    /// How many times the controller's slide hit something this tick, a
    /// diagnostic (§15). rapier's slide loop gives up after `SLIDE_PASSES`.
    pub slide_hits: u32,
    /// Wedged under a shallow overhang (§15): a grounded tick's slide gave
    /// up, so the overhang clip takes in shallow undersides too, for as long
    /// as it keeps changing the motion.
    pub wedged: bool,
    pub body: RigidBodyHandle,
    pub collider: ColliderHandle,
    pub controller: KinematicCharacterController,
}

/// Camera projection. The vertical FOV is a setting (`GraphicsSettings::fov_y`,
/// default `DEFAULT_FOV_DEG`); the projection, the light clusters (§12), the LOD
/// budget (§17) and the cascade fit (§11) all read it from there each frame,
/// since they must describe the same frustum or fragments read the wrong
/// cluster.
pub const DEFAULT_FOV_DEG: u32 = 60;
/// What GAMEPLAY > FIELD OF VIEW steps through, in vertical degrees.
pub const FOV_PRESETS: [u32; 5] = [50, 60, 70, 80, 90];
pub const CAMERA_NEAR: f32 = 0.1;
pub const CAMERA_FAR: f32 = 200.0;

/// Look angles, updated at **render rate** for responsive aim (§14/§15) — unlike
/// [`Player`], which is fixed-step. Separate component so the two rates don't get
/// tangled.
#[derive(Component, Clone, Copy)]
pub struct Look {
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for Look {
    fn default() -> Self {
        Self::new()
    }
}

impl Look {
    pub fn new() -> Self {
        Self {
            yaw: -std::f32::consts::FRAC_PI_2, // looking -Z
            pitch: 0.0,
        }
    }

    /// Full look direction (yaw + pitch), for the view matrix.
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp)
    }

    /// Horizontal (yaw-only) forward + right — the walking basis.
    pub fn ground_basis(&self) -> (Vec3, Vec3) {
        let (sy, cy) = self.yaw.sin_cos();
        let fwd = Vec3::new(cy, 0.0, sy); // unit: cy^2 + sy^2 = 1
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        (fwd, right)
    }

    /// Full camera basis (forward, right, up) — the frame the view frustum's
    /// corners are built in, for fitting shadow cascades.
    pub fn camera_basis(&self) -> (Vec3, Vec3, Vec3) {
        let fwd = self.forward();
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        (fwd, right, right.cross(fwd).normalize_or_zero())
    }

    /// World → view (right-handed, looking down -Z).
    pub fn view(&self, eye: Vec3) -> Mat4 {
        Mat4::look_to_rh(eye, self.forward(), Vec3::Y)
    }

    pub fn view_proj(&self, eye: Vec3, aspect: f32, fov_y: f32) -> Mat4 {
        let mut proj = Mat4::perspective_rh(fov_y, aspect, CAMERA_NEAR, CAMERA_FAR);
        proj.y_axis.y *= -1.0;
        proj * self.view(eye)
    }
}

/// Gameplay input for one fixed tick (§14): abstract actions, not raw keys. The
/// app fills this at render rate (building `wish` needs [`Look`]'s yaw); the fixed
/// step consumes it and clears the latched edges, so one press feeds exactly one
/// tick.
#[derive(Resource, Default)]
pub struct InputState {
    pub wish: Vec3,    // desired horizontal move direction (unit or zero)
    pub jump: bool,    // latched press edge
    pub fire: bool,    // latched press edge (LMB, §14)
    pub vertical: f32, // noclip vertical axis (+up/-down)
    pub noclip: bool,
}

impl InputState {
    /// Publish a frame's pressed edge (§14): OR it into the tick input and
    /// clear the frame's flag. The consumer system clears `fire` after its
    /// tick, so one press fires exactly one shot; frames that run no fixed
    /// step hold the edge until one does (same contract as `jump`).
    pub fn latch(&mut self, edge: &mut bool) {
        self.fire |= *edge;
        *edge = false;
    }
}

impl Player {
    /// Create the player with its feet at `feet`, registering the kinematic body
    /// and capsule in `physics`.
    pub fn new(physics: &mut Physics, feet: Vec3) -> Self {
        let body = physics.bodies.insert(
            RigidBodyBuilder::kinematic_position_based()
                .translation(to_rapier(feet + Self::CENTER)),
        );
        let collider = physics.colliders.insert_with_parent(
            ColliderBuilder::capsule_y(CAPSULE_HALF_HEIGHT, PLAYER_RADIUS),
            body,
            &mut physics.bodies,
        );
        Self {
            pos: feet,
            prev_pos: feet,
            vel: Vec3::ZERO,
            on_ground: true, // feet start resting on the ground
            surface: Surface::default(),
            slide_hits: 0,
            wedged: false,
            body,
            collider,
            controller: KinematicCharacterController {
                // Step over lips up to 0.4 units while grounded.
                autostep: Some(CharacterAutostep {
                    max_height: CharacterLength::Absolute(0.4),
                    min_width: CharacterLength::Absolute(0.2),
                    include_dynamic_bodies: false,
                }),
                ..Default::default()
            },
        }
    }

    /// Feet → capsule center (the body's origin).
    pub const CENTER: Vec3 = Vec3::new(0.0, PLAYER_HEIGHT * 0.5, 0.0);
}

/// **ECS → rapier** (§15's first sync half). Snapshot `pos` for interpolation,
/// work out this tick's desired motion (the controller owns velocity: chase the
/// target speed, gravity, jump on the press edge), let the
/// `KinematicCharacterController` shape-cast it against the level, and hand the
/// result to rapier as the kinematic body's next position.
///
/// Deliberately a plain function rather than a system body, so the physics tests
/// can drive the real logic directly. `player_target_sys` is the thin wrapper.
pub fn player_target(p: &mut Player, physics: &mut Physics, input: &InputState) {
    let (wish, jump, vgo, noclip) = (input.wish, input.jump, input.vertical, input.noclip);
    p.prev_pos = p.pos;
    // What the slide ran into that a push can move: dynamic props (§15).
    let mut pushes: Vec<CharacterCollision> = Vec::new();

    let motion = if noclip {
        // Free flight: velocity follows input directly, no gravity or collision.
        let dir = wish + Vec3::new(0.0, vgo, 0.0);
        p.vel = dir.normalize_or_zero() * FLY_SPEED;
        p.on_ground = false;
        p.slide_hits = 0;
        p.vel * FIXED_DT
    } else {
        // Horizontal velocity chases the target speed; vertical is gravity + jump.
        let target = wish * MOVE_SPEED;
        let t = (MOVE_ACCEL * FIXED_DT).min(1.0);
        p.vel.x += (target.x - p.vel.x) * t;
        p.vel.z += (target.z - p.vel.z) * t;
        if p.on_ground {
            // Grounded: no fall speed, and no gravity either. Feeding a grounded
            // capsule a downward push each tick presses it into the controller's
            // contact offset and makes the slide step intermittently return zero
            // horizontal motion; `snap_to_ground` keeps it planted instead.
            p.vel.y = 0.0;
        } else {
            p.vel.y -= GRAVITY * FIXED_DT;
        }
        if jump && p.on_ground {
            p.vel.y = JUMP_SPEED;
        }

        let queries = physics.broad_phase.as_query_pipeline(
            physics.narrow_phase.query_dispatcher(),
            &physics.bodies,
            &physics.colliders,
            QueryFilter::default().exclude_rigid_body(p.body), // don't collide with ourselves
        );
        let shape = physics.colliders[p.collider].shape();
        let at = Pose::from_translation(to_rapier(p.pos + Player::CENTER));
        // Walking into something that hangs lower than the player's head (a
        // tree's canopy, a roof's edge) wedges the capsule between the ground
        // and that thing's underside. rapier's slide then bounces between the
        // two for all 20 of its passes, every tick, without moving (§15). So
        // first bend the motion along every underside it would bring the head
        // within `OVERHANG_REACH` of: blocked head-on, it then asks for
        // nothing. That includes undersides it only reaches during this
        // tick, arriving or sliding from one canopy facet onto the next, not
        // just the ones it already touches.
        // Only on the ground and not on a jump: in the air there's no floor
        // to wedge against, and sliding up over undersides is how hopping at
        // a tree climbs its branches, which clipping would stop.
        let wish = p.vel * FIXED_DT;
        let desired = if p.on_ground && p.vel.y <= 0.0 {
            around_overhangs(&queries, p.pos, wish, p.wedged)
        } else {
            wish
        };
        let mut hits = 0;
        let moved =
            p.controller
                .move_shape(FIXED_DT, &queries, shape, &at, to_rapier(desired), |c| {
                    hits += 1;
                    let parent = physics.colliders[c.handle].parent();
                    if parent.is_some_and(|b| physics.bodies[b].is_dynamic()) {
                        pushes.push(c);
                    }
                });
        p.slide_hits = hits;
        let actual = from_rapier(moved.translation);

        // Velocity follows what actually happened, so a blocked axis loses its
        // speed instead of pushing into the obstacle every tick. The controller
        // applies no gravity, so vertical stays ours: zero it when grounded or
        // when the head hit something on the way up.
        p.on_ground = moved.grounded;
        p.wedged = p.on_ground && (hits >= SLIDE_PASSES || (p.wedged && desired != wish));
        p.vel.x = actual.x / FIXED_DT;
        p.vel.z = actual.z / FIXED_DT;
        if p.on_ground || (desired.y > 0.0 && actual.y < desired.y - 1e-4) {
            p.vel.y = 0.0;
        }
        // What is underfoot (§20)? The controller knows it is grounded but
        // not on what, so sweep the capsule's bottom sphere a little way
        // down. Unlike a ray from the centre, that finds a ledge the capsule
        // rests on by its rim; unlike the whole capsule, it doesn't also test
        // the trunk it's leaning on (§15 has the numbers). No hit (grounded
        // on a near-vertical contact) keeps the last surface.
        if p.on_ground {
            let feet = Ball::new(PLAYER_RADIUS);
            let at = Pose::from_translation(to_rapier(p.pos + actual + Vec3::Y * PLAYER_RADIUS));
            let options = ShapeCastOptions {
                max_time_of_impact: GROUND_PROBE,
                target_distance: 0.0,
                stop_at_penetration: false,
                compute_impact_geometry_on_penetration: false,
            };
            if let Some((ground, _)) = queries.cast_shape(&at, -Vector::Y, &feet, options) {
                p.surface = collider_surface(&physics.colliders[ground]);
            }
        }
        actual
    };

    push_props(physics, &pushes);

    physics.bodies[p.body]
        .set_next_kinematic_translation(to_rapier(p.pos + Player::CENTER + motion));
}

/// Push the dynamic props the slide ran into (§15). Each gets the impulse
/// that would share the blocked part of the motion out by the two masses,
/// at the point the capsule touched it, but all of them together no more
/// than `PUSH_FORCE` for a tick. rapier's own
/// `solve_character_collision_impulses` has no such cap: it gives every
/// contact point the whole share every tick, and walking into a 5000 kg
/// box moved it 0.7 m in 2 s.
pub fn push_props(physics: &mut Physics, pushes: &[CharacterCollision]) {
    let mut budget = PUSH_FORCE * FIXED_DT;
    for c in pushes {
        let Some(handle) = physics.colliders[c.handle].parent() else {
            continue;
        };
        // The controller's shape-cast reports the hit from the prop's side,
        // in world space: `normal1` points out of the prop at the player, and
        // `witness1` is the touched point on its surface.
        let n = -c.hit.normal1;
        let point = c.hit.witness1;
        let body = &mut physics.bodies[handle];
        let blocked = c.translation_remaining.dot(n) / FIXED_DT;
        let dv = blocked - body.velocity_at_point(point).dot(n);
        if dv <= 0.0 {
            continue;
        }
        let share = body.mass() * PLAYER_MASS / (body.mass() + PLAYER_MASS);
        let j = (dv * share).min(budget);
        body.apply_impulse_at_point(n * j, point, true);
        budget -= j;
        if budget <= 0.0 {
            break;
        }
    }
}

/// How many passes rapier's slide makes before it gives up (§15).
pub const SLIDE_PASSES: u32 = 20;

/// Every downward-facing surface (an overhang, §15) within `OVERHANG_REACH`
/// plus `ahead` of the head of the player whose feet are at `feet` (and if
/// `wedged`, the shallow ones touching it too), as the unit horizontal
/// direction out of it and how far the player may still move towards it
/// horizontally before the head is `OVERHANG_REACH` away (zero for one
/// already that close). Only the capsule's top sphere can meet a surface
/// that faces down (anywhere on the straight part, the contact normal is
/// horizontal), so the test uses that sphere alone: nothing below head height
/// costs anything. Read from contact manifolds the way rapier's own ground
/// test does.
pub fn overhangs_near(
    queries: &QueryPipeline,
    feet: Vec3,
    ahead: f32,
    wedged: bool,
) -> Vec<(Vec3, f32)> {
    let head = Ball::new(PLAYER_RADIUS);
    let at = Pose::from_translation(to_rapier(feet + Vec3::Y * (PLAYER_HEIGHT - PLAYER_RADIUS)));
    let reach = OVERHANG_REACH + ahead;
    let aabb = head.compute_aabb(&at).loosened(reach);
    let mut manifolds: Vec<ContactManifold> = Vec::new();
    let mut out = Vec::new();
    for (_, collider) in queries.intersect_aabb_conservative(aabb) {
        manifolds.clear();
        let pos12 = at.inv_mul(collider.position());
        let _ = DefaultQueryDispatcher.contact_manifolds(
            &pos12,
            &head,
            collider.shape(),
            reach,
            &mut manifolds,
            &mut None,
        );
        for m in &manifolds {
            // Out of the collider, towards the capsule.
            let n = from_rapier(-(at.rotation * m.local_n1));
            let side = Vec3::new(n.x, 0.0, n.z);
            let Some(dist) = m.points.iter().map(|c| c.dist).reduce(f32::min) else {
                continue;
            };
            // Facing down by more than ~12°, with enough of a side to clip
            // against: a flat ceiling can't wedge the capsule. Shallower
            // undersides wedge it too, but clipping them up front held
            // players whom the slide used to carry along a tree trunk, so
            // they count only once it's wedged, and only touching (§15).
            // Moving horizontally by `k` towards the surface closes the gap
            // along its normal by `k * side.length()`.
            let steep = n.y < -0.2;
            let shallow = wedged && n.y < -0.02 && dist <= OVERHANG_REACH;
            if dist <= reach && (steep || shallow) && side.length() > 0.1 {
                // Aim a leg 1 cm inside the reach rather than at its edge,
                // so the next look finds that underside touching, even on a
                // curve, which comes closer less than its plane predicts.
                let room = if dist <= OVERHANG_REACH {
                    0.0
                } else {
                    (dist - OVERHANG_REACH + 0.01) / side.length()
                };
                out.push((side.normalize(), room));
            }
        }
    }
    out
}

/// How many times one tick's motion may bend round an overhang (§15). When
/// they run out, the rest of the motion is dropped: it stops short, it
/// doesn't wedge.
pub const OVERHANG_BENDS: usize = 3;

/// The grounded `motion` of the player whose feet are at `feet`, bent round
/// the overhangs on its way (§15): each leg goes as far as the head may go
/// before it's `OVERHANG_REACH` from the next underside ahead, and the rest
/// is then clipped against that underside, now touching, like the ones
/// touching from the start (`wedged`: see `Player`). The legs are summed
/// into one move for the controller, which still does the real collision;
/// the straight line cuts inside a bend round a curved canopy by only
/// millimetres.
pub fn around_overhangs(queries: &QueryPipeline, feet: Vec3, motion: Vec3, wedged: bool) -> Vec3 {
    let mut done = Vec3::new(0.0, motion.y, 0.0);
    let mut left = Vec3::new(motion.x, 0.0, motion.z);
    for _ in 0..OVERHANG_BENDS {
        let walls = overhangs_near(queries, feet + done, left.length(), wedged);
        let (go, rest) = clip_horizontal(left, &walls);
        done += go;
        left = rest;
        if left == Vec3::ZERO {
            break;
        }
    }
    done
}

/// Horizontal `motion` kept out of `walls` (unit horizontal normals pointing
/// out of them, and how far it may still go into each), as the part to move
/// now and the part left for after the next bend. Into a wall it already
/// touches (no room), the part that goes into it is taken out, so it slides
/// along it; if what's left still goes into one, as in a corner between two,
/// nothing is left. Into one still some way off, it goes as far as the room
/// allows, keeping its direction: steering there would send it sideways along
/// something it hasn't reached, and on a rounded canopy that sideways drift
/// feeds itself until the player slides off round it.
pub fn clip_horizontal(mut motion: Vec3, walls: &[(Vec3, f32)]) -> (Vec3, Vec3) {
    let into = |m: Vec3, n: Vec3| -(m.x * n.x + m.z * n.z);
    let touching = walls.iter().filter(|w| w.1 <= 0.0).map(|w| w.0);
    for n in touching.clone() {
        let d = into(motion, n);
        if d > 0.0 {
            motion.x += n.x * d;
            motion.z += n.z * d;
        }
    }
    if touching.clone().any(|n| into(motion, n) > 1e-6) {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let mut part = 1.0f32;
    for &(n, room) in walls.iter().filter(|w| w.1 > 0.0) {
        let d = into(motion, n);
        if d > room {
            part = part.min(room / d);
        }
    }
    if part >= 1.0 {
        return (motion, Vec3::ZERO);
    }
    (motion * part, motion * (1.0 - part))
}

/// **rapier → ECS** (§15's second sync half): after `Physics::step`, write the
/// body's stepped position back into the component.
pub fn player_readback(p: &mut Player, physics: &Physics) {
    p.pos = from_rapier(physics.bodies[p.body].translation()) - Player::CENTER;
}

/// ECS→rapier system: run the controller for every player entity and push its
/// kinematic target, then clear the latched jump edge so one press feeds exactly
/// one tick.
pub fn player_target_sys(
    mut q: Query<&mut Player>,
    mut input: ResMut<InputState>,
    mut physics: ResMut<Physics>,
) {
    for mut p in &mut q {
        player_target(&mut p, &mut physics, &input);
    }
    input.jump = false;
}

/// Advance the rapier world one fixed tick — the step the two sync systems bracket.
pub fn physics_step_sys(mut physics: ResMut<Physics>) {
    physics.step();
}

/// rapier→ECS system: write stepped body positions back into components.
pub fn player_readback_sys(mut q: Query<&mut Player>, physics: Res<Physics>) {
    for mut p in &mut q {
        player_readback(&mut p, &physics);
    }
}

/// The player's health (the HUD's bar). Nothing drains it yet — the kill
/// plane kills outright — but damage sources land here.
#[derive(Component)]
pub struct Health {
    pub current: f32,
    pub max: f32,
}

/// Where a killed player comes back: the `player_start` marker's spot (§18),
/// or the hardcoded spawn when the scene places none. Inserted by
/// `build_world`.
#[derive(Resource)]
pub struct SpawnPoint {
    pub pos: Vec3,
    pub yaw: Option<f32>,
}

/// The kill plane's height: below it, the player dies and respawns. From the
/// `environment` marker's `kill_y` (level.rs), default `GROUND_Y - 4` — under
/// the demo slab, over anything a level digs on purpose. A level that brings
/// its own ground (the zone) sets it explicitly.
#[derive(Resource)]
pub struct KillPlane(pub f32);

/// §15's kill plane, as a plain fn so the physics tests drive it directly.
/// Below `kill_y` the player dies: back to the spawn point, velocity, look
/// and health reset. Noclip is the caller's check — free flight goes
/// wherever.
pub fn kill_plane(
    p: &mut Player,
    look: &mut Look,
    health: &mut Health,
    physics: &mut Physics,
    spawn: &SpawnPoint,
    kill_y: f32,
) {
    if p.pos.y >= kill_y {
        return;
    }
    p.pos = spawn.pos;
    p.prev_pos = spawn.pos;
    p.vel = Vec3::ZERO;
    p.on_ground = false;
    physics.bodies[p.body].set_next_kinematic_translation(to_rapier(spawn.pos + Player::CENTER));
    if let Some(yaw) = spawn.yaw {
        look.yaw = yaw;
        look.pitch = 0.0;
    }
    health.current = health.max;
}

/// After the physics bracket (pos is fresh): the kill plane, unless noclip —
/// flying below the world on purpose is not dying.
pub fn kill_plane_sys(
    mut q: Query<(&mut Player, &mut Look, &mut Health)>,
    input: Res<InputState>,
    spawn: Res<SpawnPoint>,
    kill: Res<KillPlane>,
    mut physics: ResMut<Physics>,
) {
    if input.noclip {
        return;
    }
    for (mut p, mut look, mut health) in &mut q {
        kill_plane(&mut p, &mut look, &mut health, &mut physics, &spawn, kill.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::tag_surface;
    use crate::prefab::{build_collider, ColliderKind};
    use crate::testing::{run, setup, step};
    use feather_assets::MeshData;

    const START: Vec3 = Vec3::new(0.0, GROUND_Y, 8.0);

    #[test]
    fn latches_one_fire_edge() {
        let mut input = InputState::default();
        let mut edge = true;
        input.latch(&mut edge);
        assert!(input.fire, "press latched for the fixed step");
        assert!(!edge, "frame flag cleared: a second publish can't re-fire");
        edge = false;
        input.latch(&mut edge);
        assert!(input.fire, "held until the consuming tick clears it");
    }

    #[test]
    fn falling_past_the_kill_plane_respawns_at_spawn() {
        let (mut ph, mut p) = setup(&[], START);
        let mut look = Look {
            yaw: 5.0,
            pitch: -1.0,
        };
        let mut health = Health {
            current: 55.0,
            max: 100.0,
        };
        let spawn = SpawnPoint {
            pos: Vec3::new(1.0, GROUND_Y, -3.0),
            yaw: Some(1.25),
        };
        p.pos.y = GROUND_Y - 20.0; // out of the world
        kill_plane(
            &mut p,
            &mut look,
            &mut health,
            &mut ph,
            &spawn,
            GROUND_Y - 4.0,
        );
        assert_eq!(p.pos, spawn.pos);
        assert_eq!(p.prev_pos, spawn.pos);
        assert_eq!(p.vel, Vec3::ZERO);
        assert!(!p.on_ground);
        assert_eq!((look.yaw, look.pitch), (1.25, 0.0), "look resets to spawn");
        assert_eq!(health.current, health.max);
        // The body itself teleports on the next step, so the readback agrees.
        ph.step();
        player_readback(&mut p, &ph);
        assert_eq!(p.pos, spawn.pos);
    }

    #[test]
    fn above_the_kill_plane_is_left_alone() {
        let (mut ph, mut p) = setup(&[], START);
        let mut look = Look::new();
        let mut health = Health {
            current: 55.0,
            max: 100.0,
        };
        let spawn = SpawnPoint {
            pos: Vec3::new(1.0, GROUND_Y, -3.0),
            yaw: Some(1.25),
        };
        kill_plane(
            &mut p,
            &mut look,
            &mut health,
            &mut ph,
            &spawn,
            GROUND_Y - 4.0,
        );
        assert_eq!(p.pos, START, "standing on the ground, untouched");
        assert_eq!(health.current, 55.0, "no heal without a death");
        assert_eq!(look.yaw, Look::new().yaw);
    }

    #[test]
    fn noclip_flies_below_the_kill_plane() {
        let (ph, p) = setup(&[], START);
        let mut world = World::new();
        world.insert_resource(ph);
        world.insert_resource(SpawnPoint {
            pos: Vec3::new(1.0, GROUND_Y, -3.0),
            yaw: Some(1.25),
        });
        world.insert_resource(KillPlane(GROUND_Y - 4.0));
        world.insert_resource(InputState {
            noclip: true,
            ..Default::default()
        });
        let e = world
            .spawn((
                p,
                Look::new(),
                Health {
                    current: 100.0,
                    max: 100.0,
                },
            ))
            .id();
        world.get_mut::<Player>(e).unwrap().pos.y = GROUND_Y - 50.0;
        let mut schedule = Schedule::default();
        schedule.add_systems(kill_plane_sys);
        schedule.run(&mut world);
        assert_eq!(
            world.get::<Player>(e).unwrap().pos.y,
            GROUND_Y - 50.0,
            "noclip is exempt: flying below the world on purpose"
        );
    }

    #[test]
    fn settles_on_ground() {
        let (mut ph, mut p) = setup(&[], START);
        run(&mut p, &mut ph, 90, Vec3::ZERO);
        assert!(p.on_ground);
        assert!((p.pos.y - GROUND_Y).abs() < 0.03, "y = {}", p.pos.y);
        assert_eq!(p.vel.y, 0.0);
    }

    #[test]
    fn walks_at_target_speed() {
        let (mut ph, mut p) = setup(&[], START);
        run(&mut p, &mut ph, 90, -Vec3::Z);
        assert!(p.on_ground);
        assert!((p.vel.z + MOVE_SPEED).abs() < 0.1, "vz = {}", p.vel.z);
        assert!(p.pos.z < 8.0 - 8.0, "z = {}", p.pos.z); // > 8 units covered in 1.5 s
        assert!((p.pos.y - GROUND_Y).abs() < 0.03);
        // The controller leaves a few mm of sideways drift; anything visible is a bug.
        assert!(p.pos.x.abs() < 0.02, "x = {}", p.pos.x);
    }

    #[test]
    fn stopped_by_box() {
        // 2-wide box centered on z=0 spans z in [-1, 1]; player walks -Z from z=8.
        let b = (Vec3::new(0.0, GROUND_Y + 1.0, 0.0), Vec3::splat(2.0));
        let (mut ph, mut p) = setup(&[b], START);
        run(&mut p, &mut ph, 240, -Vec3::Z);
        let gap = p.pos.z - (1.0 + PLAYER_RADIUS);
        assert!((0.0..0.05).contains(&gap), "gap = {gap}, z = {}", p.pos.z);
        assert!(p.vel.z.abs() < 0.5, "vz = {}", p.vel.z);
        assert!((p.pos.y - GROUND_Y).abs() < 0.03);
    }

    #[test]
    fn slides_along_box_face() {
        // Walk diagonally into the box's +Z face: -Z is blocked, +X slides.
        let b = (Vec3::new(0.0, GROUND_Y + 1.0, 0.0), Vec3::splat(2.0));
        let (mut ph, mut p) = setup(&[b], Vec3::new(-0.5, GROUND_Y, 1.6));
        let wish = Vec3::new(1.0, 0.0, -1.0).normalize();
        run(&mut p, &mut ph, 12, wish);
        // Held off the face, but slid along it.
        assert!(p.pos.z >= 1.0 + PLAYER_RADIUS - 0.01, "z = {}", p.pos.z);
        assert!(p.pos.x > 0.3, "x = {}", p.pos.x);
        // Once clear of the +X edge the -Z half of the input is free again.
        run(&mut p, &mut ph, 108, wish);
        assert!(
            p.pos.x > 1.0 + PLAYER_RADIUS && p.pos.z < -1.0,
            "{:?}",
            p.pos
        );
    }

    fn jump_trace(jump_ticks: &[u32], total: u32) -> (f32, Player) {
        let (mut ph, mut p) = setup(&[], START);
        run(&mut p, &mut ph, 5, Vec3::ZERO);
        let mut apex = 0.0f32;
        for t in 0..total {
            step(
                &mut p,
                &mut ph,
                Vec3::ZERO,
                jump_ticks.contains(&t),
                0.0,
                false,
            );
            apex = apex.max(p.pos.y - GROUND_Y);
        }
        (apex, p)
    }

    #[test]
    fn jump_reaches_apex_and_lands() {
        let (apex, p) = jump_trace(&[0], 120);
        // v^2 / 2g = 1.56, plus a little from discrete integration.
        assert!((1.4..1.75).contains(&apex), "apex = {apex}");
        assert!(p.on_ground);
        assert!((p.pos.y - GROUND_Y).abs() < 0.03, "y = {}", p.pos.y);
    }

    #[test]
    fn no_air_jump() {
        let (single, _) = jump_trace(&[0], 120);
        let (double, _) = jump_trace(&[0, 12, 20], 120);
        assert!((single - double).abs() < 1e-4, "{single} vs {double}");
    }

    #[test]
    fn jumps_onto_box() {
        // 1.0-high, 12-deep box (z in [-10, 2]): apex (1.6) clears it, so jumping
        // into its face lands on top. Deep enough not to walk off the far side.
        let b = (
            Vec3::new(0.0, GROUND_Y + 0.5, -4.0),
            Vec3::new(3.0, 1.0, 12.0),
        );
        let (mut ph, mut p) = setup(&[b], Vec3::new(0.0, GROUND_Y, 2.7));
        run(&mut p, &mut ph, 5, Vec3::ZERO);
        step(&mut p, &mut ph, -Vec3::Z, true, 0.0, false);
        run(&mut p, &mut ph, 60, -Vec3::Z);
        assert!(p.on_ground);
        assert!((p.pos.y - (GROUND_Y + 1.0)).abs() < 0.03, "y = {}", p.pos.y);
        assert!(p.pos.z < 1.6, "z = {}", p.pos.z); // actually on the box, not beside it
    }

    #[test]
    fn head_bump_cancels_upward_velocity() {
        // Slab whose underside is 2.0 above the ground; jump apex would reach 1.6+1.8.
        let slab = (
            Vec3::new(0.0, GROUND_Y + 2.25, 8.0),
            Vec3::new(4.0, 0.5, 4.0),
        );
        let (mut ph, mut p) = setup(&[slab], START);
        run(&mut p, &mut ph, 5, Vec3::ZERO);
        let mut crown = f32::MIN; // height of the head above the ground
        for t in 0..120 {
            step(&mut p, &mut ph, Vec3::ZERO, t == 0, 0.0, false);
            crown = crown.max(p.pos.y + PLAYER_HEIGHT - GROUND_Y);
        }
        assert!(crown <= 2.0 + 1e-3, "crown = {crown}");
        assert!(p.on_ground);
        assert!((p.pos.y - GROUND_Y).abs() < 0.03);
    }

    #[test]
    fn steps_over_low_lip_but_not_tall_wall() {
        // 0.3-high lip at z in [-0.5, 0.5]: auto-stepped, no jump needed.
        let lip = (
            Vec3::new(0.0, GROUND_Y + 0.15, 0.0),
            Vec3::new(6.0, 0.3, 1.0),
        );
        let (mut ph, mut p) = setup(&[lip], START);
        run(&mut p, &mut ph, 150, -Vec3::Z);
        assert!(p.pos.z < -1.5, "stuck at z = {}", p.pos.z);
        assert!((p.pos.y - GROUND_Y).abs() < 0.03, "y = {}", p.pos.y);
    }

    /// Walk from z = `from` in `wish` towards something that hangs lower
    /// than the player's head: a 1.5 m ball, as a hull or a trimesh, whose
    /// bottom is 1.3 m up at the origin (a tree's canopy, say). Returns where
    /// the player ended, how many ticks it was held (under 1 mm of progress),
    /// the most slide hits any of those ticks took, and the most any tick
    /// took.
    fn push_into_low_canopy(kind: ColliderKind, wish: Vec3, from: f32) -> (Vec3, u32, u32, u32) {
        let (mut ph, _) = setup(&[], Vec3::new(30.0, GROUND_Y, 30.0));
        let at = Mat4::from_translation(Vec3::new(0.0, GROUND_Y + 1.3 + 1.5, 0.0));
        build_collider(&mut ph, &MeshData::uv_sphere(16, 24, 1.5), None, at, kind).expect("canopy");
        let mut p = Player::new(&mut ph, Vec3::new(0.0, GROUND_Y, from));
        ph.step();
        let (mut held, mut worst, mut worst_any) = (0, 0, 0);
        for _ in 0..120 {
            let before = p.pos;
            step(&mut p, &mut ph, wish, false, 0.0, false);
            worst_any = worst_any.max(p.slide_hits);
            if ((p.pos - before) * Vec3::new(1.0, 0.0, 1.0)).length() < 0.001 {
                held += 1;
                worst = worst.max(p.slide_hits);
            }
        }
        (p.pos, held, worst, worst_any)
    }

    /// The stall of §15: pushing into a low overhang wedges the capsule
    /// between the ground and the overhang's underside, and rapier's slide
    /// took its full 20 passes every tick, alternating between the two,
    /// without moving. The player must still be held there, cheaply.
    #[test]
    fn a_low_overhang_holds_the_player_cheaply() {
        for kind in [ColliderKind::Hull, ColliderKind::Mesh] {
            let (end, held, worst, _) = push_into_low_canopy(kind, -Vec3::Z, 3.0);
            assert!(
                held > 60,
                "{kind:?}: not held at the canopy ({held} ticks, z {})",
                end.z
            );
            assert!(end.z > 1.0, "{kind:?}: got under the canopy, z = {}", end.z);
            assert!(worst <= 3, "{kind:?}: {worst} slide hits in a tick");
        }
    }

    /// Reaching a low overhang during a tick wedged the capsule just like
    /// pushing into it (§15): the clip only knew the undersides the head
    /// touched when the tick began, so arriving at one, or sliding from one
    /// of its facets onto the next, ran rapier's slide to all 20 passes.
    /// From starts spread over one tick's travel, head-on and glancing, no
    /// tick may take more than 3.
    #[test]
    fn reaching_a_low_overhang_mid_tick_is_cheap() {
        let glancing = Vec3::new(0.5, 0.0, -0.866);
        for kind in [ColliderKind::Hull, ColliderKind::Mesh] {
            for wish in [-Vec3::Z, glancing] {
                for k in 0..8 {
                    let from = 3.0 + k as f32 * MOVE_SPEED * FIXED_DT / 8.0;
                    let (_, _, _, worst) = push_into_low_canopy(kind, wish, from);
                    assert!(
                        worst <= 3,
                        "{kind:?} {wish:?} from z {from}: {worst} slide hits in a tick"
                    );
                }
            }
        }
    }

    /// Walk at `yaw` degrees off -Z into a 4 m wide wall leaning 9° over
    /// the player (a fat trunk's bulge, say), with a box standing against
    /// its +X half if `corner`. Returns where the player ended and how many
    /// ticks rapier's slide gave up.
    fn push_along_leaning_wall(corner: bool, yaw: f32) -> (Vec3, u32) {
        let (mut ph, _) = setup(&[], Vec3::new(30.0, GROUND_Y, 30.0));
        let lean = Mat4::from_translation(Vec3::new(0.0, GROUND_Y + 1.5, 0.0))
            * Mat4::from_rotation_x(9f32.to_radians())
            * Mat4::from_scale(Vec3::new(4.0, 3.0, 0.3));
        build_collider(
            &mut ph,
            &MeshData::cube(1.0),
            None,
            lean,
            ColliderKind::Hull,
        )
        .expect("wall");
        if corner {
            ph.add_static_box(
                Vec3::new(1.5, GROUND_Y + 1.5, 1.5),
                Vec3::new(1.0, 3.0, 3.0),
            );
        }
        let mut p = Player::new(&mut ph, Vec3::new(0.0, GROUND_Y, 2.0));
        ph.step();
        let (s, c) = yaw.to_radians().sin_cos();
        let mut gave_up = 0;
        for _ in 0..120 {
            step(&mut p, &mut ph, Vec3::new(s, 0.0, -c), false, 0.0, false);
            if p.slide_hits >= SLIDE_PASSES {
                gave_up += 1;
            }
        }
        (p.pos, gave_up)
    }

    /// A shallow underside (9° past vertical, under the clip's ~12°) wedges
    /// the capsule against the ground too, once the push along it is
    /// blocked: rapier's slide ran all 20 passes on 83 of 120 ticks in the
    /// corner, and 67 at the wall's end, where it stuck. The clip leaves
    /// shallow undersides to the slide until it gives up once (clipping
    /// them up front stopped slides along trunks, §15), then takes them in.
    #[test]
    fn a_shallow_overhang_wedges_at_most_once() {
        let (end, gave_up) = push_along_leaning_wall(true, 60.0);
        assert!(gave_up <= 1, "corner: the slide gave up {gave_up} times");
        assert!(end.x < 1.0, "corner: not held, x = {}", end.x);
        let (end, gave_up) = push_along_leaning_wall(false, 20.0);
        assert!(gave_up <= 1, "open end: the slide gave up {gave_up} times");
        assert!(end.x > 2.5, "open end: stuck at x = {}", end.x);
    }

    /// Hopping at a tree climbs it, jump by jump, over the stubs sticking out
    /// of its trunk. Two triangles of one of the nature scene's Kenney trees
    /// (CC0) are enough to show it: a sliver of trunk with a stub at ~1 m,
    /// and a branch's flat top above. (Found by recording every triangle a
    /// climb touched, then dropping each one the result didn't need.) The
    /// overhang clip must leave jumps alone: applied in the air as well, it
    /// kept the player on the ground here (§15).
    #[test]
    fn hopping_at_a_tree_climbs_its_branches() {
        const TREE: [[[f32; 3]; 3]; 2] = [
            [
                [0.014, -0.03, -2.076],
                [0.08, 0.987, -2.224],
                [-0.007, 2.807, -2.2],
            ],
            [
                [-0.143, 2.807, -1.377],
                [0.905, 2.807, -2.85],
                [0.806, 2.807, -1.811],
            ],
        ];
        let vertex = |pos: [f32; 3]| feather_assets::Vertex {
            pos,
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
        };
        let tree = MeshData {
            vertices: TREE.iter().flatten().map(|&p| vertex(p)).collect(),
            indices: (0..TREE.len() as u32 * 3).collect(),
            material: Default::default(),
        };
        // Recorded with the feet 3 cm up, on the scene's floor tiles.
        let (mut ph, _) = setup(&[], Vec3::new(30.0, GROUND_Y, 30.0));
        let at = Mat4::from_translation(Vec3::new(0.0, GROUND_Y + 0.03, 0.0));
        build_collider(&mut ph, &tree, None, at, ColliderKind::Mesh).expect("tree");
        let mut p = Player::new(&mut ph, Vec3::new(0.0, GROUND_Y, 0.0));
        ph.step();
        let mut stood = f32::MIN;
        for _ in 0..300 {
            let jump = p.on_ground;
            step(&mut p, &mut ph, -Vec3::Z, jump, 0.0, false);
            if p.on_ground {
                stood = stood.max(p.pos.y - GROUND_Y);
            }
        }
        assert!(
            stood > 0.9,
            "never got onto a branch: stood at most {stood:.2} m up"
        );
    }

    /// Glancing off the same overhang slides past it, as along a wall.
    #[test]
    fn a_low_overhang_is_slid_along() {
        let wish = Vec3::new(0.5, 0.0, -0.866); // 30° off the canopy's centre
        for kind in [ColliderKind::Hull, ColliderKind::Mesh] {
            let (end, held, worst, _) = push_into_low_canopy(kind, wish, 3.0);
            assert!(
                end.z < -1.5,
                "{kind:?}: didn't get past, stopped at {end:?}"
            );
            assert!(
                worst <= 3,
                "{kind:?}: {worst} slide hits in a tick ({held} held)"
            );
        }
    }

    #[test]
    fn noclip_flies_through_geometry_then_resumes() {
        let b = (Vec3::new(0.0, GROUND_Y + 1.0, 0.0), Vec3::splat(2.0));
        let (mut ph, mut p) = setup(&[b], START);
        for _ in 0..90 {
            step(&mut p, &mut ph, -Vec3::Z, false, 0.0, true);
        }
        // Straight through the box.
        assert!(p.pos.z < 0.0, "z = {}", p.pos.z);
        // Leaving noclip in the open: gravity takes over and it settles.
        run(&mut p, &mut ph, 120, Vec3::ZERO);
        assert!(p.on_ground);
        assert!((p.pos.y - GROUND_Y).abs() < 0.05, "y = {}", p.pos.y);
    }

    // ---- §20 surfaces underfoot ----

    /// Walk -Z from the concrete ground over a pad (4 x 4 m, 5 cm tall, so
    /// autostep takes it) and off its far side, and list what the probe said,
    /// merging repeats.
    fn surfaces_walking_over_a_pad(tag: Option<Surface>) -> Vec<Surface> {
        let (mut ph, mut p) = setup(&[], START);
        let pad = ph.add_static_box(
            Vec3::new(0.0, GROUND_Y + 0.025, 2.0),
            Vec3::new(4.0, 0.05, 4.0),
        );
        if let Some(s) = tag {
            tag_surface(&mut ph.colliders[pad], s);
        }
        ph.step();
        let mut seen = vec![p.surface];
        for _ in 0..150 {
            step(&mut p, &mut ph, -Vec3::Z, false, 0.0, false);
            if seen.last() != Some(&p.surface) {
                seen.push(p.surface);
            }
        }
        assert!(
            p.pos.z < -2.0,
            "should have crossed the pad, z = {}",
            p.pos.z
        );
        seen
    }

    #[test]
    fn the_probe_names_the_surface_underfoot() {
        use Surface::{Concrete, Wood};
        assert_eq!(
            surfaces_walking_over_a_pad(Some(Wood)),
            [Concrete, Wood, Concrete]
        );
        // The same walk over an untagged pad: concrete, like the ground.
        assert_eq!(surfaces_walking_over_a_pad(None), [Concrete]);
    }

    /// Standing on a ledge by the capsule's rim, centre out over the drop:
    /// sweeping the capsule's bottom sphere finds the ledge where a ray from
    /// the centre finds nothing, which is why the probe is a shape cast.
    #[test]
    fn the_probe_finds_a_ledge_under_the_rim() {
        let (mut ph, _) = setup(&[], Vec3::new(20.0, GROUND_Y, 20.0));
        // A 0.3 m block whose +Z face is at z = 0; the player 0.2 m past it.
        let block = ph.add_static_box(
            Vec3::new(0.0, GROUND_Y + 0.15, -1.0),
            Vec3::new(4.0, 0.3, 2.0),
        );
        tag_surface(&mut ph.colliders[block], Surface::Wood);
        let mut p = Player::new(&mut ph, Vec3::new(0.0, GROUND_Y + 0.32, 0.2));
        ph.step();
        for _ in 0..30 {
            step(&mut p, &mut ph, Vec3::ZERO, false, 0.0, false);
        }
        assert!(
            p.on_ground && p.pos.y > GROUND_Y + 0.2,
            "fell off: y = {}",
            p.pos.y
        );
        assert_eq!(p.surface, Surface::Wood);
        let queries = ph.broad_phase.as_query_pipeline(
            ph.narrow_phase.query_dispatcher(),
            &ph.bodies,
            &ph.colliders,
            QueryFilter::default().exclude_rigid_body(p.body),
        );
        let ray = rapier3d::prelude::Ray::new(to_rapier(p.pos), -Vector::Y);
        assert!(
            queries.cast_ray(&ray, GROUND_PROBE, true).is_none(),
            "the centre is over the block, so this isn't the rim case"
        );
    }
}
