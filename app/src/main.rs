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
//! **player** (a kinematic FPS controller — gravity, jump, box/ground collision)
//! are double-buffered (`Prev*` / current); extract and the camera
//! **interpolate** by `alpha = accumulator / FIXED_DT`, so motion is smooth at
//! any framerate and sim speed no longer scales with it. The controller is
//! hand-rolled groundwork for rapier (§15). The drifting orbs now hang above a
//! ground plane with a few obstacle boxes to walk among.

use std::time::Instant;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ExecutorKind;
use feather_assets::MeshData;
use feather_gfx::Renderer;
use feather_platform::winit;
use feather_render::{InstanceData, MeshId, MeshRenderer, SkyPass, TonemapPass};
use glam::{Mat4, Vec3, Vec4};
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
// each tick, zeroed when grounded, jump on the press edge when grounded. Look is
// render-rate; body position is fixed-step + interpolated.
const GROUND_Y: f32 = -9.0; // top surface of the ground plane (base of the orb column)
const EYE_HEIGHT: f32 = 1.6;
const PLAYER_RADIUS: f32 = 0.35; // half-width of the collision box (x/z)
const PLAYER_HEIGHT: f32 = 1.8;
const MOVE_SPEED: f32 = 8.0; // target ground speed, units/s
const MOVE_ACCEL: f32 = 14.0; // how fast horizontal velocity chases the target
const GRAVITY: f32 = 26.0;
const JUMP_SPEED: f32 = 9.0; // ~1.5 units apex
const FLY_SPEED: f32 = 14.0; // noclip movement speed

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

// ---- First-person player + input ----

/// Axis-aligned box collider for static level geometry.
#[derive(Clone, Copy)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn from_center_size(center: Vec3, size: Vec3) -> Self {
        let h = size * 0.5;
        Self {
            min: center - h,
            max: center + h,
        }
    }
}

/// The player. `pos` is the feet position — simulated at the fixed step, then
/// interpolated for the camera. `yaw`/`pitch` are look angles updated at render
/// rate (responsive aim, §15). `vel` is owned by the controller: gravity and
/// jump live here, not in a physics solver.
struct Player {
    pos: Vec3,
    prev_pos: Vec3,
    vel: Vec3,
    yaw: f32,
    pitch: f32,
    on_ground: bool,
}

impl Player {
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

/// Advance the player one fixed step (§15's `step_gameplay` + `physics.step`,
/// hand-rolled until rapier lands). `wish` is the desired horizontal move
/// direction (unit or zero); `vgo` is the noclip vertical axis (+up/-down).
fn step_player(p: &mut Player, wish: Vec3, jump: bool, vgo: f32, noclip: bool, colliders: &[Aabb]) {
    p.prev_pos = p.pos;

    if noclip {
        // Free flight: velocity follows input directly, no gravity or collision.
        let dir = wish + Vec3::new(0.0, vgo, 0.0);
        p.vel = dir.normalize_or_zero() * FLY_SPEED;
        p.pos += p.vel * FIXED_DT;
        p.on_ground = false;
        return;
    }

    // Horizontal velocity chases the target speed; vertical is gravity + jump.
    let target = wish * MOVE_SPEED;
    let t = (MOVE_ACCEL * FIXED_DT).min(1.0);
    p.vel.x += (target.x - p.vel.x) * t;
    p.vel.z += (target.z - p.vel.z) * t;
    p.vel.y -= GRAVITY * FIXED_DT;
    if jump && p.on_ground {
        p.vel.y = JUMP_SPEED;
        p.on_ground = false;
    }

    p.pos += p.vel * FIXED_DT;
    resolve_collisions(p, colliders);
}

/// Ground-plane clamp + push-out against each static box along its axis of least
/// penetration. Sets `on_ground` and zeroes the blocked velocity component. A
/// discrete (non-swept) resolver — fine at fixed 60 Hz and modest speeds; rapier
/// replaces it later.
fn resolve_collisions(p: &mut Player, colliders: &[Aabb]) {
    p.on_ground = false;

    // Ground plane.
    if p.pos.y <= GROUND_Y {
        p.pos.y = GROUND_Y;
        if p.vel.y < 0.0 {
            p.vel.y = 0.0;
        }
        p.on_ground = true;
    }

    // Static boxes. Recompute the player box each iteration (pos mutates).
    for c in colliders {
        let pmin = p.pos + Vec3::new(-PLAYER_RADIUS, 0.0, -PLAYER_RADIUS);
        let pmax = p.pos + Vec3::new(PLAYER_RADIUS, PLAYER_HEIGHT, PLAYER_RADIUS);
        let ox = pmax.x.min(c.max.x) - pmin.x.max(c.min.x);
        let oy = pmax.y.min(c.max.y) - pmin.y.max(c.min.y);
        let oz = pmax.z.min(c.max.z) - pmin.z.max(c.min.z);
        if ox <= 0.0 || oy <= 0.0 || oz <= 0.0 {
            continue; // separated on some axis
        }
        if ox <= oy && ox <= oz {
            let pc = (pmin.x + pmax.x) * 0.5;
            let cc = (c.min.x + c.max.x) * 0.5;
            p.pos.x += if pc < cc { -ox } else { ox };
            p.vel.x = 0.0;
        } else if oz <= oy {
            let pc = (pmin.z + pmax.z) * 0.5;
            let cc = (c.min.z + c.max.z) * 0.5;
            p.pos.z += if pc < cc { -oz } else { oz };
            p.vel.z = 0.0;
        } else {
            let pc = (pmin.y + pmax.y) * 0.5;
            let cc = (c.min.y + c.max.y) * 0.5;
            if pc < cc {
                p.pos.y -= oy; // clipped head on an underside
            } else {
                p.pos.y += oy; // landed on top
                p.on_ground = true;
            }
            p.vel.y = 0.0;
        }
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
    player: Player,
    input: Input,
    noclip: bool,
    // Static level colliders (boxes); the ground is a plane at GROUND_Y.
    colliders: Vec<Aabb>,
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
            player: Player {
                pos: Vec3::new(0.0, GROUND_Y, 8.0), // feet on the ground, looking -Z
                prev_pos: Vec3::new(0.0, GROUND_Y, 8.0),
                vel: Vec3::ZERO,
                yaw: -std::f32::consts::FRAC_PI_2,
                pitch: 0.0,
                on_ground: true,
            },
            input: Input::default(),
            noclip: false,
            colliders: Vec::new(),
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

        // Static level. The ground is rendered as a wide flat box but collided as
        // the GROUND_Y plane (jitter-free resting); the obstacle boxes are both
        // rendered and added as AABB colliders. Level pieces carry no Velocity/
        // Spin, so `integrate` skips them and they never wrap.
        spawn_static(
            &mut self.world,
            Vec3::new(0.0, GROUND_Y - 0.5, 0.0),
            Vec3::new(80.0, 1.0, 80.0),
            level_cube_id,
            ground_mat,
        );
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
            self.colliders.push(Aabb::from_center_size(center, size));
        }

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
                self.player.yaw += self.input.mouse_dx * sens;
                self.player.pitch =
                    (self.player.pitch - self.input.mouse_dy * sens).clamp(-1.54, 1.54);
                self.input.mouse_dx = 0.0;
                self.input.mouse_dy = 0.0;

                // Desired horizontal move direction from WASD, in the yaw plane.
                let (fwd, right) = self.player.ground_basis();
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

                // Fixed-timestep sim: consume the accumulator in whole FIXED_DT
                // steps. Each step advances the ECS schedule and the player
                // controller together (snapshot -> step_gameplay -> resolve, §15).
                // The frame delta is clamped and MAX_STEPS caps catch-up per frame
                // (spiral-of-death guard); leftover beyond the cap is dropped.
                self.accumulator += dt.min(MAX_FRAME_TIME);
                let mut steps = 0;
                while self.accumulator >= FIXED_DT && steps < MAX_STEPS {
                    self.schedule.run(&mut self.world);
                    step_player(
                        &mut self.player,
                        wish,
                        self.input.jump,
                        vgo,
                        self.noclip,
                        &self.colliders,
                    );
                    self.input.jump = false; // consumed by this step
                    self.accumulator -= FIXED_DT;
                    steps += 1;
                }
                // How far we are into the next step, in [0,1): the render alpha.
                let alpha = (self.accumulator / FIXED_DT).clamp(0.0, 1.0);

                // Extract: interpolate each entity's sim state (prev -> curr) by
                // alpha, then build its model matrix. This is the sim<->render
                // seam; interpolation lives here per §4. Static level pieces lack
                // Velocity/Spin so `integrate` skips them; prev == curr keeps them
                // fixed through the same interpolation path.
                let fits = &self.fits;
                let mesh_max = fits.len().saturating_sub(1);
                let mut items: Vec<(MeshId, InstanceData)> =
                    Vec::with_capacity((GRID * GRID * GRID) as usize + 8);
                let mut q = self.world.query::<(
                    &Position,
                    &PrevPosition,
                    &Rotation,
                    &PrevRotation,
                    &Scale,
                    &Mesh,
                    &Material,
                )>();
                for (p, pp, r, pr, scale, mesh, material) in q.iter(&self.world) {
                    let id = (mesh.0 .0 as usize).min(mesh_max);
                    let pos = pp.0.lerp(p.0, alpha);
                    let angle = pr.0 + (r.0 - pr.0) * alpha;
                    let model = Mat4::from_translation(pos)
                        * Mat4::from_rotation_y(angle)
                        * Mat4::from_scale(scale.0)
                        * fits[id];
                    items.push((MeshId(id as u32), InstanceData::new(model, material.0)));
                }

                // Camera: interpolate the player's body, offset to eye height.
                let eye = self.player.prev_pos.lerp(self.player.pos, alpha)
                    + Vec3::new(0.0, EYE_HEIGHT, 0.0);
                let size = self.window.as_ref().unwrap().inner_size();
                let aspect = size.width as f32 / size.height.max(1) as f32;
                let view_proj = self.player.view_proj(eye, aspect);
                let inv_view_proj = view_proj.inverse();
                let light_dir = self.light_dir;
                let camera_pos = eye;

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
                    r.draw_frame(
                        |cmd, extent, frame| {
                            // Background first (depth off), then meshes over it.
                            sky.draw(cmd, extent, inv_view_proj, camera_pos, light_dir);
                            m.draw(cmd, extent, frame, view_proj, light_dir, camera_pos, &mut items)
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

/// Spawn one static level entity: rendered through the normal instanced path but
/// carrying no `Velocity`/`Spin`, so `integrate` skips it (it never moves or
/// wraps). `prev == curr` makes it a no-op through the interpolation path.
fn spawn_static(world: &mut World, center: Vec3, size: Vec3, mesh: u32, material: u32) {
    world.spawn((
        Position(center),
        PrevPosition(center),
        Rotation(0.0),
        PrevRotation(0.0),
        Scale(size),
        Mesh(MeshId(mesh)),
        Material(material),
    ));
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
