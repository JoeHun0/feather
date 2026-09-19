//! Milestone 2+: a drifting field of ~1000 lit ECS entities. Several meshes
//! share one vertex/index buffer; each entity picks a mesh at random, and
//! extract sorts instances by mesh so each draws as one batched run. Shading
//! reads a **material** per instance (`material_id` → a materials SSBO), whose
//! base color can be a **texture** sampled from a bindless array: loaded glTF
//! meshes use their file's base-color texture/factor, procedural sphere/cube
//! instances use a shared palette. Meshes are a procedural sphere + cube by
//! default, or one
//! per glTF/GLB path on the CLI (`cargo run -- a.glb b.glb`), each auto-fitted
//! to the grid. A directional light with a Cook-Torrance **PBR** BRDF plus
//! analytic environment ambient (IBL) shades in
//! linear space into an HDR target, over a procedural **sky background**,
//! which a tonemap pass resolves to the sRGB swapchain. **WASD + mouse** walk a
//! first-person controller (Space jump, `V` toggles noclip fly); `[` / `]`
//! adjust exposure; Esc quits.
//!
//! Simulation runs on a **fixed timestep** (accumulator; `FIXED_DT`), decoupled
//! from the render rate. Entity sim state (position + spin angle) and the
//! **player** (a rapier `KinematicCharacterController` — gravity, jump, capsule
//! vs. static colliders) are double-buffered (`Prev*` / current); extract and the
//! camera **interpolate** by `alpha = accumulator / FIXED_DT`, so motion is
//! smooth at any framerate and sim speed no longer scales with it. rapier state
//! lives in a raw ECS `Resource` (§15) and is stepped once per fixed tick. The
//! drifting orbs now hang above a ground box with a few obstacle boxes to walk
//! among.

use std::time::Instant;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ExecutorKind;
use feather_assets::MeshData;
use feather_gfx::{Renderer, SHADOW_DIM};
use feather_platform::winit;
use feather_render::{InstanceData, MeshId, MeshRenderer, SkyPass, TonemapPass};
use glam::{Mat4, Vec3, Vec4};
use rapier3d::control::{CharacterAutostep, CharacterLength, KinematicCharacterController};
use rapier3d::prelude::{
    BroadPhaseBvh, CCDSolver, ColliderBuilder, ColliderHandle, ColliderSet, ImpulseJointSet,
    IntegrationParameters, IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline, Pose,
    QueryFilter, RigidBodyBuilder, RigidBodyHandle, RigidBodySet, Vector,
};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const GRID: i32 = 10; // GRID^3 entities
const MAX_INSTANCES: u32 = 8192;
const BOUND: f32 = 9.0;

// Fixed-timestep sim (§4). The schedule advances in whole `FIXED_DT` steps from
// an accumulator; render interpolates the remainder. `MAX_FRAME_TIME` clamps a
// single frame's real delta and `MAX_STEPS` caps steps per frame — together the
// spiral-of-death guard when the app stalls (debugger break, lost focus).
const FIXED_DT: f32 = 1.0 / 60.0;
const MAX_FRAME_TIME: f32 = 0.25;
const MAX_STEPS: u32 = 8;

// First-person controller (§15). The controller owns vertical velocity: gravity
// each tick, zeroed when grounded, jump on the press edge when grounded. rapier's
// `KinematicCharacterController` applies no gravity of its own. Look is
// render-rate; body position is fixed-step + interpolated.
const GROUND_Y: f32 = -9.0; // top surface of the ground collider (base of the orb column)
const EYE_HEIGHT: f32 = 1.6;
const PLAYER_RADIUS: f32 = 0.35; // capsule radius
const PLAYER_HEIGHT: f32 = 1.8; // total height, feet to crown (capsule caps included)
const CAPSULE_HALF_HEIGHT: f32 = (PLAYER_HEIGHT - 2.0 * PLAYER_RADIUS) * 0.5; // `capsule_y` arg: half the straight section
const MOVE_SPEED: f32 = 8.0; // target ground speed, units/s
const MOVE_ACCEL: f32 = 14.0; // how fast horizontal velocity chases the target
const GRAVITY: f32 = 26.0;
const JUMP_SPEED: f32 = 9.0; // ~1.5 units apex
const FLY_SPEED: f32 = 14.0; // noclip movement speed

// Frustum culling (§8). A fitted mesh lives in a unit cube (sphere radius ≤
// √3/2 ≈ 0.87), scaled by the entity's Scale — a conservative cull radius.
const CULL_SPHERE_K: f32 = 0.87;

// Sun shadow frustum (§11). The ortho follows the player, so these are relative
// to them: half-extent of the covered square, how far back along the sun the
// light "eye" sits, and the ortho's depth range. A tighter radius means denser
// shadow texels (sharper) but less coverage around the player.
const SHADOW_RADIUS: f32 = 16.0;
const SHADOW_BACK: f32 = 40.0;
const SHADOW_DEPTH: f32 = 80.0;

// ---- ECS data ----

#[derive(Component)]
struct Position(Vec3);
/// Position at the start of the latest completed fixed step (the other endpoint
/// extract interpolates from). Kept in the same wrapped space as `Position`.
#[derive(Component)]
struct PrevPosition(Vec3);
#[derive(Component)]
struct Velocity(Vec3);
/// Accumulated Y rotation (radians), advanced by `Spin` each fixed step.
#[derive(Component)]
struct Rotation(f32);
/// `Rotation` at the start of the latest completed fixed step.
#[derive(Component)]
struct PrevRotation(f32);
#[derive(Component)]
struct Spin(f32); // angular velocity, rad/s
/// Per-entity local scale, applied inside the model matrix. Drifting demo
/// entities use a uniform scale; static level pieces use non-uniform sizes.
#[derive(Component)]
struct Scale(Vec3);
#[derive(Component)]
struct Mesh(MeshId);
#[derive(Component)]
struct Material(u32); // material_id into the renderer's material table
/// Opt-out marker: this entity is not rendered into the sun shadow map (§11).
/// Everything casts by default. The demo's flat ground carries it — a flat slab
/// casts nothing useful (nothing is beneath it) yet rasterizes the *entire*
/// shadow map, which dominates that pass's fill cost. This is deliberately
/// per-entity rather than a rule about "ground": terrain with relief has to cast
/// (hills shadow valleys), and then it simply doesn't carry this marker.
#[derive(Component)]
struct NoShadowCast;
/// This entity's collider in the rapier world (§15: entities carry their
/// handles). Static level geometry for now — nothing reads it back yet, since
/// statics never move, but it is the handle a future sync system would use.
#[derive(Component)]
struct ColliderRef(#[allow(dead_code)] ColliderHandle);
#[derive(Resource, Default)]
struct FrameCount(u64);

/// Number of shared palette materials generated for procedural meshes.
const PALETTE: u32 = 24;

/// What to register with the renderer. Resolved into `MeshData` at startup; the
/// index in the source list is the mesh's `MeshId`.
enum MeshSource {
    Sphere,
    Cube,
    Gltf(String),
}

/// One fixed step: snapshot the current state into `Prev*`, then advance
/// position by velocity and the spin angle by angular velocity. Runs at
/// `FIXED_DT`, so motion is now framerate-independent.
fn integrate(
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
fn wrap_axis(cur: &mut f32, prev: &mut f32) {
    if *cur > BOUND {
        *cur -= 2.0 * BOUND;
        *prev -= 2.0 * BOUND;
    } else if *cur < -BOUND {
        *cur += 2.0 * BOUND;
        *prev += 2.0 * BOUND;
    }
}

fn tick(mut frame: ResMut<FrameCount>) {
    frame.0 += 1;
}

fn rand01(seed: u32) -> f32 {
    let mut h = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    h ^= h >> 15;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    (h & 0x00ff_ffff) as f32 / 0x0100_0000 as f32
}

// ---- Physics (rapier, §15) ----

/// The workspace pins glam 0.29 but rapier 0.35 builds on its own (newer) glam, so
/// `Vec3` and rapier's `Vector` are distinct types. Convert at the boundary.
fn to_rapier(v: Vec3) -> Vector {
    Vector::new(v.x, v.y, v.z)
}

fn from_rapier(v: Vector) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// All rapier state as one raw ECS resource (§15: raw rapier, not bevy_rapier).
/// There is no gravity here — the character controller is kinematic and the
/// player owns its vertical velocity — so the world only ever holds fixed level
/// colliders plus the player's position-based kinematic body.
#[derive(Resource)]
struct Physics {
    pipeline: PhysicsPipeline,
    params: IntegrationParameters,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd: CCDSolver,
}

impl Physics {
    fn new() -> Self {
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
    fn step(&mut self) {
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

    /// A fixed axis-aligned box collider (static level geometry).
    fn add_static_box(&mut self, center: Vec3, size: Vec3) -> ColliderHandle {
        let h = size * 0.5;
        self.colliders
            .insert(ColliderBuilder::cuboid(h.x, h.y, h.z).translation(to_rapier(center)))
    }
}

// ---- First-person player + input ----

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
struct Player {
    pos: Vec3,
    prev_pos: Vec3,
    vel: Vec3,
    on_ground: bool,
    body: RigidBodyHandle,
    collider: ColliderHandle,
    controller: KinematicCharacterController,
}

/// Look angles, updated at **render rate** for responsive aim (§14/§15) — unlike
/// [`Player`], which is fixed-step. Separate component so the two rates don't get
/// tangled.
#[derive(Component, Clone, Copy)]
struct Look {
    yaw: f32,
    pitch: f32,
}

impl Look {
    fn new() -> Self {
        Self {
            yaw: -std::f32::consts::FRAC_PI_2, // looking -Z
            pitch: 0.0,
        }
    }

    /// Full look direction (yaw + pitch), for the view matrix.
    fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp)
    }

    /// Horizontal (yaw-only) forward + right — the walking basis.
    fn ground_basis(&self) -> (Vec3, Vec3) {
        let (sy, cy) = self.yaw.sin_cos();
        let fwd = Vec3::new(cy, 0.0, sy); // unit: cy^2 + sy^2 = 1
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        (fwd, right)
    }

    fn view_proj(&self, eye: Vec3, aspect: f32) -> Mat4 {
        let view = Mat4::look_to_rh(eye, self.forward(), Vec3::Y);
        let mut proj = Mat4::perspective_rh(60f32.to_radians(), aspect, 0.1, 200.0);
        proj.y_axis.y *= -1.0;
        proj * view
    }
}

/// Gameplay input for one fixed tick (§14): abstract actions, not raw keys. The
/// app fills this at render rate (building `wish` needs [`Look`]'s yaw); the fixed
/// step consumes it and clears the latched jump edge, so one press feeds exactly
/// one tick.
#[derive(Resource, Default)]
struct InputState {
    wish: Vec3,    // desired horizontal move direction (unit or zero)
    jump: bool,    // latched press edge
    vertical: f32, // noclip vertical axis (+up/-down)
    noclip: bool,
}

impl Player {
    /// Create the player with its feet at `feet`, registering the kinematic body
    /// and capsule in `physics`.
    fn new(physics: &mut Physics, feet: Vec3) -> Self {
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
    const CENTER: Vec3 = Vec3::new(0.0, PLAYER_HEIGHT * 0.5, 0.0);
}

/// **ECS → rapier** (§15's first sync half). Snapshot `pos` for interpolation,
/// work out this tick's desired motion (the controller owns velocity: chase the
/// target speed, gravity, jump on the press edge), let the
/// `KinematicCharacterController` shape-cast it against the level, and hand the
/// result to rapier as the kinematic body's next position.
///
/// Deliberately a plain function rather than a system body, so the physics tests
/// can drive the real logic directly. `player_target_sys` is the thin wrapper.
fn player_target(p: &mut Player, physics: &mut Physics, input: &InputState) {
    let (wish, jump, vgo, noclip) = (input.wish, input.jump, input.vertical, input.noclip);
    p.prev_pos = p.pos;

    let motion = if noclip {
        // Free flight: velocity follows input directly, no gravity or collision.
        let dir = wish + Vec3::new(0.0, vgo, 0.0);
        p.vel = dir.normalize_or_zero() * FLY_SPEED;
        p.on_ground = false;
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

        let desired = p.vel * FIXED_DT;
        let queries = physics.broad_phase.as_query_pipeline(
            physics.narrow_phase.query_dispatcher(),
            &physics.bodies,
            &physics.colliders,
            QueryFilter::default().exclude_rigid_body(p.body), // don't collide with ourselves
        );
        let moved = p.controller.move_shape(
            FIXED_DT,
            &queries,
            physics.colliders[p.collider].shape(),
            &Pose::from_translation(to_rapier(p.pos + Player::CENTER)),
            to_rapier(desired),
            |_| {},
        );
        let actual = from_rapier(moved.translation);

        // Velocity follows what actually happened, so a blocked axis loses its
        // speed instead of pushing into the obstacle every tick. The controller
        // applies no gravity, so vertical stays ours: zero it when grounded or
        // when the head hit something on the way up.
        p.on_ground = moved.grounded;
        p.vel.x = actual.x / FIXED_DT;
        p.vel.z = actual.z / FIXED_DT;
        if p.on_ground || (desired.y > 0.0 && actual.y < desired.y - 1e-4) {
            p.vel.y = 0.0;
        }
        actual
    };

    physics.bodies[p.body]
        .set_next_kinematic_translation(to_rapier(p.pos + Player::CENTER + motion));
}

/// **rapier → ECS** (§15's second sync half): after `Physics::step`, write the
/// body's stepped position back into the component.
fn player_readback(p: &mut Player, physics: &Physics) {
    p.pos = from_rapier(physics.bodies[p.body].translation()) - Player::CENTER;
}

/// ECS→rapier system: run the controller for every player entity and push its
/// kinematic target, then clear the latched jump edge so one press feeds exactly
/// one tick.
fn player_target_sys(
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
fn physics_step_sys(mut physics: ResMut<Physics>) {
    physics.step();
}

/// rapier→ECS system: write stepped body positions back into components.
fn player_readback_sys(mut q: Query<&mut Player>, physics: Res<Physics>) {
    for mut p in &mut q {
        player_readback(&mut p, &physics);
    }
}

#[derive(Default)]
struct Input {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    jump: bool, // latched on the Space press edge; consumed by the fixed step
    mouse_dx: f32,
    mouse_dy: f32,
}

struct App {
    mesh: Option<MeshRenderer>,
    tonemap: Option<TonemapPass>,
    sky: Option<SkyPass>,
    renderer: Option<Renderer>,
    window: Option<Window>,
    world: World,
    schedule: Schedule,
    /// The player entity in `world` — its sim state is the `Player` component,
    /// its look angles the `Look` component (§1: the World owns simulation state).
    player: Entity,
    input: Input,
    noclip: bool,
    light_dir: Vec4,
    exposure: f32,
    // Meshes to register (decided up front); index == MeshId.
    sources: Vec<MeshSource>,
    // Per-mesh transform that centers + unit-scales it into the demo grid.
    fits: Vec<Mat4>,
    last_frame: Instant,
    // Fixed-timestep accumulator: real time not yet consumed by a sim step,
    // carried across frames. Its fraction of FIXED_DT is the render alpha.
    accumulator: f32,
}

impl App {
    fn new(sources: Vec<MeshSource>) -> Self {
        let mut world = World::new();
        world.insert_resource(FrameCount::default());
        let mut physics = Physics::new();
        // Feet on the ground, looking -Z. The player is a normal ECS entity: sim
        // state in `Player`, render-rate angles in `Look` (§15).
        let body = Player::new(&mut physics, Vec3::new(0.0, GROUND_Y, 8.0));
        world.insert_resource(physics);
        world.insert_resource(InputState::default());
        let player = world.spawn((body, Look::new())).id();

        let mesh_count = sources.len().max(1) as u32;
        let half = (GRID as f32 - 1.0) / 2.0;
        for i in 0..(GRID * GRID * GRID) {
            let (x, y, z) = (i % GRID, (i / GRID) % GRID, i / (GRID * GRID));
            let pos = Vec3::new(x as f32 - half, y as f32 - half, z as f32 - half) * 1.6;
            let u = i as u32;
            let vel = Vec3::new(
                rand01(u * 3) - 0.5,
                rand01(u * 3 + 1) - 0.5,
                rand01(u * 3 + 2) - 0.5,
            ) * 1.5;
            let spin = (rand01(u * 7 + 11) - 0.5) * 3.0;
            // Assign a mesh at random among those registered.
            let mesh = ((rand01(u * 17 + 5) * mesh_count as f32) as u32).min(mesh_count - 1);
            // glTF meshes use their own file material (id = PALETTE + mesh index);
            // procedural meshes pick a random shared palette material.
            let material = if matches!(sources.get(mesh as usize), Some(MeshSource::Gltf(_))) {
                PALETTE + mesh
            } else {
                (rand01(u * 23 + 7) * PALETTE as f32) as u32 % PALETTE
            };
            world.spawn((
                Position(pos),
                PrevPosition(pos), // prev == curr on frame 0: first interp is a no-op
                Velocity(vel),
                Rotation(0.0),
                PrevRotation(0.0),
                Spin(spin),
                Scale(Vec3::splat(0.6)),
                Mesh(MeshId(mesh)),
                Material(material),
            ));
        }

        let mut schedule = Schedule::default();
        schedule.set_executor_kind(ExecutorKind::MultiThreaded);
        schedule.add_systems((integrate, tick));
        // §15's coupling: ECS -> rapier, the step, then rapier -> ECS. Chained so
        // the bracket order is explicit (they all touch `Physics`, so bevy_ecs
        // would serialise them regardless).
        schedule.add_systems((player_target_sys, physics_step_sys, player_readback_sys).chain());

        let light = Vec3::new(-0.4, -1.0, -0.3).normalize();
        let now = Instant::now();
        Self {
            mesh: None,
            tonemap: None,
            sky: None,
            renderer: None,
            window: None,
            world,
            schedule,
            player,
            input: Input::default(),
            noclip: false,
            light_dir: Vec4::new(light.x, light.y, light.z, 0.0),
            exposure: 1.0,
            sources,
            fits: Vec::new(),
            last_frame: now,
            accumulator: 0.0,
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        if let Some(r) = &self.renderer {
            r.wait_idle();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = event_loop
            .create_window(feather_platform::window_attributes("feather — first person"))
            .expect("create window");
        window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
            .ok();
        window.set_cursor_visible(false);

        let size = window.inner_size();
        let renderer = Renderer::new(&window, size.width, size.height).expect("create renderer");

        // Resolve every source into MeshData (loading glTF, generating
        // procedurals). A failed glTF is replaced with a sphere so mesh indices
        // stay aligned with `sources` (and thus with spawned Mesh ids).
        let mut meshes: Vec<MeshData> = Vec::with_capacity(self.sources.len());
        for src in &self.sources {
            let mesh = match src {
                MeshSource::Sphere => MeshData::uv_sphere(16, 24, 0.5),
                MeshSource::Cube => MeshData::cube(1.0),
                MeshSource::Gltf(path) => match feather_assets::load_gltf(path) {
                    Ok(m) => {
                        eprintln!(
                            "loaded {path}: {} vertices, {} indices",
                            m.vertices.len(),
                            m.indices.len()
                        );
                        m
                    }
                    Err(e) => {
                        eprintln!("failed to load {path}: {e} (using sphere)");
                        MeshData::uv_sphere(16, 24, 0.5)
                    }
                },
            };
            meshes.push(mesh);
        }
        if meshes.is_empty() {
            meshes.push(MeshData::uv_sphere(16, 24, 0.5));
        }
        // Append a unit cube used for all static level geometry (ground + boxes).
        // Its MeshId is the last slot; `fits` below normalizes it to a unit cube
        // centered at the origin, so a per-entity `Scale` yields exact box sizes.
        let level_cube_id = meshes.len() as u32;
        meshes.push(MeshData::cube(1.0));
        self.fits = meshes.iter().map(fit_transform).collect();

        // Material table: shared palette first (indices 0..PALETTE), then each
        // mesh's own material (index PALETTE + mesh_id) — matches the ids that
        // App::new assigned to entities.
        let mut materials: Vec<feather_assets::Material> =
            (0..PALETTE).map(palette_material).collect();
        materials.extend(meshes.iter().map(|m| m.material.clone()));

        // Two dedicated level materials, appended after everything else.
        let ground_mat = materials.len() as u32;
        materials.push(level_material([0.20, 0.21, 0.23], 0.95));
        let box_mat = materials.len() as u32;
        materials.push(level_material([0.45, 0.22, 0.14], 0.7));

        // Static level. The ground and the obstacle boxes are each rendered as a
        // scaled unit cube and given a matching cuboid collider (`spawn_static`).
        // Level pieces carry no Velocity/Spin, so `integrate` skips them and they
        // never wrap.
        let ground = spawn_static(
            &mut self.world,
            Vec3::new(0.0, GROUND_Y - 0.5, 0.0),
            Vec3::new(80.0, 1.0, 80.0),
            level_cube_id,
            ground_mat,
        );
        // This ground is a flat slab: it casts nothing useful but would rasterize
        // the whole shadow map. It still *receives* shadows (receiving is sampling
        // the map, not being in it). Relief terrain would drop this marker.
        self.world.entity_mut(ground).insert(NoShadowCast);
        // (x, half-height as y, z), size — y offset keeps each box resting on the
        // ground (center = GROUND_Y + size.y/2).
        let boxes = [
            (Vec3::new(-3.0, 0.75, 2.0), Vec3::new(1.5, 1.5, 1.5)),
            (Vec3::new(3.5, 1.0, -1.0), Vec3::new(2.0, 2.0, 2.0)),
            (Vec3::new(0.0, 0.5, -4.5), Vec3::new(3.0, 1.0, 1.0)),
            (Vec3::new(-5.0, 1.5, -3.0), Vec3::new(1.0, 3.0, 1.0)),
            (Vec3::new(5.0, 0.5, 4.0), Vec3::new(1.0, 1.0, 4.0)),
        ];
        for (offset, size) in boxes {
            let center = Vec3::new(offset.x, GROUND_Y + offset.y, offset.z);
            spawn_static(&mut self.world, center, size, level_cube_id, box_mat);
        }
        // One step so the broad-phase BVH the character controller shape-casts
        // against contains the level before the first fixed tick.
        self.world.resource_mut::<Physics>().step();

        let (mesh, _ids) = MeshRenderer::new(&renderer, &meshes, &materials, MAX_INSTANCES);
        let tonemap = TonemapPass::new(&renderer);
        let sky = SkyPass::new(&renderer);

        self.mesh = Some(mesh);
        self.tonemap = Some(tonemap);
        self.sky = Some(sky);
        self.renderer = Some(renderer);
        self.window = Some(window);
    }

    fn device_event(&mut self, _e: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.input.mouse_dx += dx as f32;
            self.input.mouse_dy += dy as f32;
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => self.input.forward = pressed,
                        KeyCode::KeyS => self.input.back = pressed,
                        KeyCode::KeyA => self.input.left = pressed,
                        KeyCode::KeyD => self.input.right = pressed,
                        KeyCode::Space => {
                            // Latch the press edge for jump; also drives noclip up.
                            if pressed && !self.input.up {
                                self.input.jump = true;
                            }
                            self.input.up = pressed;
                        }
                        KeyCode::ControlLeft => self.input.down = pressed,
                        // V toggles noclip (free flight) for inspecting the scene.
                        KeyCode::KeyV if pressed => self.noclip = !self.noclip,
                        // Exposure control (showcases the HDR/tonemap pipeline).
                        KeyCode::BracketLeft if pressed => {
                            self.exposure = (self.exposure * 0.8).max(0.05);
                        }
                        KeyCode::BracketRight if pressed => {
                            self.exposure = (self.exposure * 1.25).min(16.0);
                        }
                        KeyCode::Escape if pressed => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                // Look updates at render rate for responsive aim (§15).
                let sens = 0.0025;
                let look = {
                    let mut look = self
                        .world
                        .get_mut::<Look>(self.player)
                        .expect("player has Look");
                    look.yaw += self.input.mouse_dx * sens;
                    look.pitch = (look.pitch - self.input.mouse_dy * sens).clamp(-1.54, 1.54);
                    *look
                };
                self.input.mouse_dx = 0.0;
                self.input.mouse_dy = 0.0;

                // Desired horizontal move direction from WASD, in the yaw plane.
                let (fwd, right) = look.ground_basis();
                let mut wish = Vec3::ZERO;
                if self.input.forward {
                    wish += fwd;
                }
                if self.input.back {
                    wish -= fwd;
                }
                if self.input.right {
                    wish += right;
                }
                if self.input.left {
                    wish -= right;
                }
                let wish = wish.normalize_or_zero();
                // Noclip vertical axis (Space up / Ctrl down); ignored when grounded.
                let vgo = (self.input.up as i32 - self.input.down as i32) as f32;

                // Publish this frame's abstract input (§14); the fixed step consumes
                // it. The jump edge is latched here and cleared by the controller
                // system, so a press still feeds exactly one tick.
                {
                    let mut state = self.world.resource_mut::<InputState>();
                    state.wish = wish;
                    state.vertical = vgo;
                    state.noclip = self.noclip;
                    state.jump |= self.input.jump;
                }
                self.input.jump = false;

                // Fixed-timestep sim: consume the accumulator in whole FIXED_DT
                // steps. The schedule advances gameplay and brackets the rapier step
                // with the two sync systems (§15). The frame delta is clamped and
                // MAX_STEPS caps catch-up per frame (spiral-of-death guard);
                // leftover beyond the cap is dropped.
                self.accumulator += dt.min(MAX_FRAME_TIME);
                let mut steps = 0;
                while self.accumulator >= FIXED_DT && steps < MAX_STEPS {
                    self.schedule.run(&mut self.world);
                    self.accumulator -= FIXED_DT;
                    steps += 1;
                }
                // How far we are into the next step, in [0,1): the render alpha.
                let alpha = (self.accumulator / FIXED_DT).clamp(0.0, 1.0);

                // Camera + sun matrices first — the extract loop culls against them.
                // Camera: interpolate the player's body, offset to eye height.
                // Copy the pair out before the extract query re-borrows the World.
                let (p_prev, p_pos) = {
                    let p = self.world.get::<Player>(self.player).expect("player body");
                    (p.prev_pos, p.pos)
                };
                let eye = p_prev.lerp(p_pos, alpha) + Vec3::new(0.0, EYE_HEIGHT, 0.0);
                let size = self.window.as_ref().unwrap().inner_size();
                let aspect = size.width as f32 / size.height.max(1) as f32;
                let view_proj = look.view_proj(eye, aspect);
                let inv_view_proj = view_proj.inverse();
                let light_dir = self.light_dir;
                let camera_pos = eye;

                // Sun shadow matrix (§11): a tight ortho that **follows the player**,
                // so shadows exist wherever you walk instead of only near the origin.
                // It is centered on the player's body, not the view direction, so
                // turning never disturbs the shadow map — only walking moves it, and
                // the texel snap below keeps that from crawling.
                let sun_dir = self.light_dir.truncate().normalize_or_zero();
                let body = eye - Vec3::Y * EYE_HEIGHT;
                // Rotation-only light basis (world -> light space), for the snap.
                let light_basis = Mat4::look_at_rh(Vec3::ZERO, sun_dir, Vec3::Y);
                // Texel-snap the ortho center to whole shadow-map texels — §11 calls
                // this non-negotiable: without it the shadow edges crawl every frame
                // as the center slides a fraction of a texel.
                let world_per_texel = (2.0 * SHADOW_RADIUS) / SHADOW_DIM as f32;
                let c = light_basis.transform_point3(body);
                let snapped = Vec3::new(
                    (c.x / world_per_texel).round() * world_per_texel,
                    (c.y / world_per_texel).round() * world_per_texel,
                    c.z, // depth along the light needs no snap (no edge crawl)
                );
                let light_center = light_basis.inverse().transform_point3(snapped);
                let light_eye = light_center - sun_dir * SHADOW_BACK;
                let light_view = Mat4::look_at_rh(light_eye, light_center, Vec3::Y);
                let light_proj = Mat4::orthographic_rh(
                    -SHADOW_RADIUS,
                    SHADOW_RADIUS,
                    -SHADOW_RADIUS,
                    SHADOW_RADIUS,
                    0.1,
                    SHADOW_DEPTH,
                );
                let light_view_proj = light_proj * light_view;

                // Per-view frustum culling (§8): bounding sphere vs six planes, once
                // per view. Camera frustum trims the main pass; the light ortho trims
                // the shadow pass (groundwork — everything is inside it today).
                let camera_frustum = Frustum::from_view_proj(&view_proj);
                let light_frustum = Frustum::from_view_proj(&light_view_proj);

                // Extract: interpolate each entity's sim state (prev -> curr) by
                // alpha, build its model matrix, and route it to the camera-visible
                // set (main pass) and/or the light-visible set (shadow pass). This is
                // the sim<->render seam; interpolation lives here per §4. Static level
                // pieces lack Velocity/Spin so `integrate` skips them; prev == curr.
                let fits = &self.fits;
                let mesh_max = fits.len().saturating_sub(1);
                let cap = (GRID * GRID * GRID) as usize + 8;
                let mut main_items: Vec<(MeshId, InstanceData)> = Vec::with_capacity(cap);
                let mut shadow_items: Vec<(MeshId, InstanceData)> = Vec::with_capacity(cap);
                let mut q = self.world.query::<(
                    &Position,
                    &PrevPosition,
                    &Rotation,
                    &PrevRotation,
                    &Scale,
                    &Mesh,
                    &Material,
                    Option<&NoShadowCast>,
                )>();
                for (p, pp, r, pr, scale, mesh, material, no_cast) in q.iter(&self.world) {
                    let id = (mesh.0 .0 as usize).min(mesh_max);
                    let pos = pp.0.lerp(p.0, alpha);
                    let angle = pr.0 + (r.0 - pr.0) * alpha;
                    let model = Mat4::from_translation(pos)
                        * Mat4::from_rotation_y(angle)
                        * Mat4::from_scale(scale.0)
                        * fits[id];
                    let item = (MeshId(id as u32), InstanceData::new(model, material.0));
                    // Conservative world bounding sphere: `fit_transform` normalizes
                    // the mesh into a unit cube (sphere <= sqrt(3)/2), then Scale sizes
                    // it. Over-inclusive by design — never culls a visible object.
                    let radius = CULL_SPHERE_K * scale.0.max_element();
                    if camera_frustum.contains_sphere(pos, radius) {
                        main_items.push(item);
                    }
                    if no_cast.is_none() && light_frustum.contains_sphere(pos, radius) {
                        shadow_items.push(item);
                    }
                }

                let exposure = self.exposure;
                if let (Some(r), Some(m), Some(tm), Some(sky)) = (
                    self.renderer.as_mut(),
                    self.mesh.as_mut(),
                    self.tonemap.as_mut(),
                    self.sky.as_ref(),
                ) {
                    // HDR view/sampler are stable except across resize; capture
                    // before the mutable draw_frame borrow, refresh in `update`.
                    let hdr_view = r.hdr_view();
                    let hdr_sampler = r.hdr_sampler();
                    // CPU prep once (sort + stage instances/globals); the shadow and
                    // main passes then replay them. Uploads happen in draw_shadow,
                    // after the frame fence.
                    m.prepare_frame(&mut main_items, &mut shadow_items, light_view_proj);
                    r.draw_frame(
                        // Shadow pass: sun depth map (also flushes this frame's buffers).
                        |cmd, extent, frame| m.draw_shadow(cmd, extent, frame),
                        // Geometry (§10): depth prepass, then the lit opaque pass
                        // (each pixel shaded once), then the sky depth-tested into
                        // whatever background is left.
                        |cmd, extent, frame| {
                            m.draw_depth_prepass(
                                cmd, extent, frame, view_proj, light_dir, camera_pos,
                            );
                            m.draw_main(cmd, extent, frame, view_proj, light_dir, camera_pos);
                            sky.draw(cmd, extent, inv_view_proj, camera_pos, light_dir);
                        },
                        |cmd, extent, frame| {
                            tm.update(frame, hdr_view, hdr_sampler);
                            tm.draw(cmd, extent, frame, exposure);
                        },
                    );
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

/// Spawn one static level piece: rendered through the normal instanced path but
/// carrying no `Velocity`/`Spin`, so `integrate` skips it (it never moves or
/// wraps). `prev == curr` makes it a no-op through the interpolation path. Also
/// registers a matching fixed cuboid collider with `Physics` (it must already be
/// a resource).
fn spawn_static(world: &mut World, center: Vec3, size: Vec3, mesh: u32, material: u32) -> Entity {
    let collider = world.resource_mut::<Physics>().add_static_box(center, size);
    world
        .spawn((
            Position(center),
            PrevPosition(center),
            Rotation(0.0),
            PrevRotation(0.0),
            Scale(size),
            Mesh(MeshId(mesh)),
            Material(material),
            ColliderRef(collider),
        ))
        .id()
}

/// A plain untextured material for level geometry. Base color is treated as
/// already-linear (the values here are low, so no sRGB decode needed).
fn level_material(base_linear: [f32; 3], roughness: f32) -> feather_assets::Material {
    feather_assets::Material {
        base_color: [base_linear[0], base_linear[1], base_linear[2], 1.0],
        metallic: 0.0,
        roughness,
        emissive: [0.0, 0.0, 0.0],
        normal_scale: 1.0,
        base_color_texture: None,
        normal_texture: None,
        metallic_roughness_texture: None,
    }
}

/// View frustum as six inward-facing planes (Gribb–Hartmann, from a view-proj
/// matrix). Used per view (§8): the camera for the main pass, the sun light ortho
/// for the shadow pass. Matrix-agnostic, so it works for the camera's y-flipped
/// projection and the light ortho alike.
struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    fn from_view_proj(m: &Mat4) -> Self {
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

    /// True unless the sphere is entirely behind some plane (i.e. culled).
    fn contains_sphere(&self, center: Vec3, radius: f32) -> bool {
        self.planes
            .iter()
            .all(|p| p.x * center.x + p.y * center.y + p.z * center.z + p.w >= -radius)
    }
}

/// Center the mesh at the origin and scale its largest extent to ~1 unit, so an
/// arbitrarily-sized glTF drops into the demo grid at the same scale as the
/// sphere. Applied as the innermost factor of each instance's model matrix.
fn fit_transform(mesh: &MeshData) -> Mat4 {
    let (min, max) = mesh.bounds();
    let center = (min + max) * 0.5;
    let extent = (max - min).max_element().max(1e-4);
    Mat4::from_scale(Vec3::splat(1.0 / extent)) * Mat4::from_translation(-center)
}

/// A varied shared material for the procedural demo. Base color is generated in
/// sRGB then stored linear (the renderer shades in linear space).
fn palette_material(k: u32) -> feather_assets::Material {
    let srgb_to_linear = |c: f32| {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let r = 0.1 + 0.85 * rand01(k * 13 + 1);
    let g = 0.1 + 0.85 * rand01(k * 13 + 2);
    let b = 0.1 + 0.85 * rand01(k * 13 + 3);
    feather_assets::Material {
        base_color: [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), 1.0],
        metallic: if rand01(k * 13 + 4) > 0.7 { 1.0 } else { 0.0 },
        roughness: 0.3 + 0.6 * rand01(k * 13 + 5),
        emissive: [0.0, 0.0, 0.0],
        normal_scale: 1.0,
        base_color_texture: None,
        normal_texture: None,
        metallic_roughness_texture: None,
    }
}

fn main() {
    // One MeshId per CLI path (`cargo run -- a.glb b.glb`). With no args, a
    // procedural sphere + cube so multi-mesh batching is visible out of the box.
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let sources = if paths.is_empty() {
        vec![MeshSource::Sphere, MeshSource::Cube]
    } else {
        paths.into_iter().map(MeshSource::Gltf).collect()
    };

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(sources);
    event_loop.run_app(&mut app).expect("run app");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ground (same 80x80x1 box as the app) plus `boxes` as `(center, size)`, and a
    /// player standing at `start`. Mirrors the app's setup, including the warm-up
    /// step that publishes the level to the broad-phase BVH.
    fn setup(boxes: &[(Vec3, Vec3)], start: Vec3) -> (Physics, Player) {
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
    fn step(p: &mut Player, ph: &mut Physics, wish: Vec3, jump: bool, vgo: f32, noclip: bool) {
        let input = InputState {
            wish,
            jump,
            vertical: vgo,
            noclip,
        };
        player_target(p, ph, &input);
        ph.step();
        player_readback(p, ph);
    }

    fn run(p: &mut Player, ph: &mut Physics, ticks: u32, wish: Vec3) {
        for _ in 0..ticks {
            step(p, ph, wish, false, 0.0, false);
        }
    }

    const START: Vec3 = Vec3::new(0.0, GROUND_Y, 8.0);

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
}
