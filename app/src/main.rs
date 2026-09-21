//! Milestone 2+: a drifting field of ~1000 lit ECS entities. Several meshes
//! share one vertex/index buffer; each entity picks a mesh at random, and
//! extract sorts instances by mesh so each draws as one batched run. Shading
//! reads a **material** per instance (`material_id` → a materials SSBO), whose
//! base color can be a **texture** sampled from a bindless array: loaded glTF
//! meshes use their file's base-color texture/factor, procedural sphere/cube
//! instances use a shared palette.
//!
//! With no arguments this runs the procedural orb demo. Given glTF/GLB paths
//! (`cargo run -- level.glb`) it instead loads them as **scenes**: every node
//! becomes its own entity with that node's world transform and its primitive's
//! own material, nodes sharing a mesh draw as instances, and each gets a trimesh
//! collider so the level is walkable. A directional light with a Cook-Torrance **PBR** BRDF plus
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

use std::collections::HashMap;
use std::time::Instant;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ExecutorKind;
use feather_assets::MeshData;
use feather_gfx::{GpuTimes, Renderer, FRAMES_IN_FLIGHT, SHADOW_CASCADES};
use feather_platform::winit;
use feather_render::{
    CascadeSetup, ClusterView, FxaaPass, GpuLight, InstanceData, MeshId, MeshRenderer, SkyPass,
    TonemapPass, UiPass,
};
use glam::{Mat4, Vec3, Vec4};
use rapier3d::control::{CharacterAutostep, CharacterLength, KinematicCharacterController};
use rapier3d::prelude::{
    BroadPhaseBvh, CCDSolver, ColliderBuilder, ColliderHandle, ColliderSet, ImpulseJointSet,
    IntegrationParameters, IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline, Pose,
    QueryFilter, RigidBodyBuilder, RigidBodyHandle, RigidBodySet, Vector,
};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
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
const SHADOW_DISTANCE: f32 = 60.0;
/// Practical-split weighting (§11): 0 = uniform, 1 = logarithmic. 0.75 keeps
/// near cascades tight without starving the far one.
const SHADOW_LAMBDA: f32 = 0.75;
const SHADOW_BACK: f32 = 40.0;

/// Shadow-quality preset (§13). Shadow cost is dominated by shadow-map texel
/// count, so resolution is the knob. `Off` additionally stops feeding casters,
/// which leaves the map cleared so every fragment compares as lit — no shader
/// branch needed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ShadowQuality {
    Off,
    Low,
    Medium,
    High,
}

impl ShadowQuality {
    /// Dimension of **one cascade**; there are `SHADOW_CASCADES` of them, so the
    /// texel count is 4x this squared. High is 4x2048², exactly the texel count
    /// and the memory of the single 4096² map cascades replaced — the pass is
    /// fill-bound (§11), so this redistributes resolution rather than adding
    /// cost.
    fn dim(self) -> u32 {
        match self {
            // Small but non-zero: the pass still clears, and a 512² clear is noise
            // next to what a full-resolution clear costs.
            Self::Off => 512,
            Self::Low => 512,
            Self::Medium => 1024,
            Self::High => 2048,
        }
    }

    fn casts(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Steps quality *down* and wraps, so repeatedly pressing the key reads as a
    /// ramp (High → Medium → Low → Off → High). Cycling upward from the High
    /// default would drop straight to Off on the first press, which looks like a
    /// flicker rather than a setting.
    fn next(self) -> Self {
        match self {
            Self::High => Self::Medium,
            Self::Medium => Self::Low,
            Self::Low => Self::Off,
            Self::Off => Self::High,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low (4x512)",
            Self::Medium => "medium (4x1024)",
            Self::High => "high (4x2048)",
        }
    }

    /// Menu form: the UI font is A-Z/0-9 only, so no lower case or brackets.
    fn menu_label(self) -> &'static str {
        match self {
            Self::Off => "OFF",
            Self::Low => "LOW",
            Self::Medium => "MEDIUM",
            Self::High => "HIGH",
        }
    }
}

/// Runtime graphics settings (§13). Render configuration, deliberately *not* an
/// ECS resource — §1 scopes the `World` to simulation state.
///
/// Anti-aliasing will join this when there is a second AA mode to choose between:
/// MSAA still needs a resolve attachment before the tonemap pass could sample a
/// multisampled target, and SMAA does not exist yet, so an `aa` field today would
/// be a setting that does nothing.
struct GraphicsSettings {
    shadows: ShadowQuality,
    /// MSAA sample count for the geometry pass (§13), `--msaa N` at startup.
    /// Startup-only rather than live: the sample count is baked into every
    /// geometry pipeline, so changing it means rebuilding them all — the usual
    /// "applies on restart" trade. `gfx` clamps this to what the device supports.
    msaa: u32,
    /// FXAA post-AA (§13). Unlike MSAA this is live-toggleable (`F2`): it bakes
    /// nothing into pipelines, and the LDR intermediate is always allocated.
    fxaa: bool,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            shadows: ShadowQuality::High,
            msaa: 1,
            fxaa: false,
        }
    }
}

/// One screen of the pause menu (§19). Screens form a tree rooted at `Root`;
/// `Menu` walks it with an explicit stack.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MenuScreen {
    /// Pre-game menu, shown when nothing is loaded.
    MainRoot,
    Root,
    Options,
    Graphics,
    Sound,
    Gameplay,
}

impl MenuScreen {
    fn title(self) -> &'static str {
        match self {
            Self::MainRoot => "FEATHER",
            Self::Root => "PAUSED",
            Self::Options => "OPTIONS",
            Self::Graphics => "GRAPHICS",
            Self::Sound => "SOUND",
            Self::Gameplay => "GAMEPLAY",
        }
    }
}

/// What activating a row does. `Inert` is a row that exists to *show* something
/// (or to mark a planned setting) but cannot be changed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MenuAction {
    Resume,
    Enter(MenuScreen),
    Back,
    Quit,
    ToggleFxaa,
    CycleShadows,
    CycleMsaa,
    NewGame,
    ToMainMenu,
    Inert,
}

/// A single drawn row. Labels are built fresh from the live settings each time,
/// so a value shown here cannot go stale — pressing F2 with the graphics screen
/// open updates the row immediately.
///
/// Labels use **A-Z, 0-9 and spaces only**: the 5x7 UI font covers exactly that
/// and renders anything else as a blank (`render/src/ui.rs:68`), so a colon or
/// a bracket would silently turn into whitespace.
struct MenuRow {
    label: String,
    action: MenuAction,
}

impl MenuRow {
    fn new(label: impl Into<String>, action: MenuAction) -> Self {
        Self {
            label: label.into(),
            action,
        }
    }

    /// Inert rows draw dimmer. They stay *selectable* so arrow navigation does
    /// not silently skip the information they carry.
    fn enabled(&self) -> bool {
        !matches!(self.action, MenuAction::Inert)
    }
}

/// What the caller must do after the menu handled an input. Returning this
/// rather than acting directly is what keeps `Menu` free of any dependency on
/// the renderer or the event loop — and therefore unit-testable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MenuOutcome {
    Stay,
    Resume,
    Quit,
    ApplyShadows,
    ApplyFxaa,
    ApplyMsaa,
    StartSession,
    EndSession,
}

/// The rows of one screen, given the settings they display and whether a world
/// is currently loaded.
///
/// `in_session` is not cosmetic: it decides whether MSAA is changeable. The
/// sample count is baked into the mesh and sky pipelines at creation, so it can
/// only move while no session owns them.
fn screen_rows(screen: MenuScreen, s: &GraphicsSettings, in_session: bool) -> Vec<MenuRow> {
    match screen {
        MenuScreen::MainRoot => vec![
            MenuRow::new("NEW GAME", MenuAction::NewGame),
            MenuRow::new("OPTIONS", MenuAction::Enter(MenuScreen::Options)),
            MenuRow::new("QUIT", MenuAction::Quit),
        ],
        MenuScreen::Root => vec![
            MenuRow::new("CONTINUE", MenuAction::Resume),
            MenuRow::new("OPTIONS", MenuAction::Enter(MenuScreen::Options)),
            MenuRow::new("MAIN MENU", MenuAction::ToMainMenu),
            MenuRow::new("EXIT", MenuAction::Quit),
        ],
        MenuScreen::Options => vec![
            MenuRow::new("GRAPHICS", MenuAction::Enter(MenuScreen::Graphics)),
            MenuRow::new("SOUND", MenuAction::Enter(MenuScreen::Sound)),
            MenuRow::new("GAMEPLAY", MenuAction::Enter(MenuScreen::Gameplay)),
            MenuRow::new("BACK", MenuAction::Back),
        ],
        MenuScreen::Graphics => vec![
            MenuRow::new(
                format!("SHADOWS  {}", s.shadows.menu_label()),
                MenuAction::CycleShadows,
            ),
            MenuRow::new(
                format!("FXAA  {}", if s.fxaa { "ON" } else { "OFF" }),
                MenuAction::ToggleFxaa,
            ),
            // Changeable only from the main menu: the mesh and sky pipelines
            // bake the sample count, so it can move only while no session owns
            // them. In game the value is shown but locked — and unlike the old
            // "RESTART" note, MAIN MENU is somewhere you can actually get to.
            if in_session {
                MenuRow::new(format!("MSAA  {}X  MENU ONLY", s.msaa), MenuAction::Inert)
            } else {
                MenuRow::new(format!("MSAA  {}X", s.msaa), MenuAction::CycleMsaa)
            },
            MenuRow::new("BACK", MenuAction::Back),
        ],
        // Placeholders: §20 audio is unstarted, and there are no gameplay
        // settings to bind to yet.
        MenuScreen::Sound => vec![
            MenuRow::new("MASTER VOLUME", MenuAction::Inert),
            MenuRow::new("MUSIC", MenuAction::Inert),
            MenuRow::new("SFX", MenuAction::Inert),
            MenuRow::new("BACK", MenuAction::Back),
        ],
        MenuScreen::Gameplay => vec![
            MenuRow::new("SENSITIVITY", MenuAction::Inert),
            MenuRow::new("FIELD OF VIEW", MenuAction::Inert),
            MenuRow::new("INVERT Y", MenuAction::Inert),
            MenuRow::new("BACK", MenuAction::Back),
        ],
    }
}

/// Pause-menu navigation state. Deliberately knows nothing about the renderer
/// or the window: it mutates `GraphicsSettings` and reports a `MenuOutcome`.
struct Menu {
    screen: MenuScreen,
    index: usize,
    /// `(screen, index)` of each ancestor, so BACK restores the row you
    /// descended from instead of snapping to the top.
    stack: Vec<(MenuScreen, usize)>,
}

impl Menu {
    fn new() -> Self {
        Self {
            screen: MenuScreen::MainRoot,
            index: 0,
            stack: Vec::new(),
        }
    }

    /// Return to a given top level, discarding history. Called when a session
    /// starts (→ `Root`) or ends (→ `MainRoot`), and on unpause, so the menu
    /// always reopens at a sensible place rather than wherever it was left.
    fn reset(&mut self, root: MenuScreen) {
        self.screen = root;
        self.index = 0;
        self.stack.clear();
    }

    fn rows(&self, s: &GraphicsSettings, in_session: bool) -> Vec<MenuRow> {
        screen_rows(self.screen, s, in_session)
    }

    /// Move the selection, wrapping at both ends.
    fn move_by(&mut self, delta: isize, s: &GraphicsSettings, in_session: bool) {
        let n = self.rows(s, in_session).len() as isize;
        if n > 0 {
            self.index = (self.index as isize + delta).rem_euclid(n) as usize;
        }
    }

    /// Point the selection at `index` if it is a real row (used by the mouse).
    fn hover(&mut self, index: usize, s: &GraphicsSettings, in_session: bool) {
        if index < self.rows(s, in_session).len() {
            self.index = index;
        }
    }

    fn descend(&mut self, screen: MenuScreen) {
        self.stack.push((self.screen, self.index));
        self.screen = screen;
        self.index = 0;
    }

    /// Up one level. At the pause root that means resuming; at the *main* root
    /// there is nothing to resume into, so it stays put. This is what makes Esc
    /// walk back out one screen at a time.
    fn back(&mut self) -> MenuOutcome {
        match self.stack.pop() {
            Some((screen, index)) => {
                self.screen = screen;
                self.index = index;
                MenuOutcome::Stay
            }
            None if self.screen == MenuScreen::MainRoot => MenuOutcome::Stay,
            None => MenuOutcome::Resume,
        }
    }

    fn activate(&mut self, s: &mut GraphicsSettings, in_session: bool) -> MenuOutcome {
        let action = match self.rows(s, in_session).get(self.index) {
            Some(row) => row.action,
            None => return MenuOutcome::Stay,
        };
        match action {
            MenuAction::Resume => MenuOutcome::Resume,
            MenuAction::Quit => MenuOutcome::Quit,
            MenuAction::Enter(screen) => {
                self.descend(screen);
                MenuOutcome::Stay
            }
            MenuAction::Back => self.back(),
            MenuAction::ToggleFxaa => {
                s.fxaa = !s.fxaa;
                MenuOutcome::ApplyFxaa
            }
            MenuAction::CycleShadows => {
                s.shadows = s.shadows.next();
                MenuOutcome::ApplyShadows
            }
            MenuAction::CycleMsaa => {
                // 1 -> 2 -> 4 -> 8 -> 1. `Renderer::set_msaa` clamps to what the
                // device actually supports, so an unsupported step lands on the
                // nearest legal count rather than failing.
                s.msaa = match s.msaa {
                    1 => 2,
                    2 => 4,
                    4 => 8,
                    _ => 1,
                };
                MenuOutcome::ApplyMsaa
            }
            MenuAction::NewGame => MenuOutcome::StartSession,
            MenuAction::ToMainMenu => MenuOutcome::EndSession,
            MenuAction::Inert => MenuOutcome::Stay,
        }
    }
}

/// Font pixel size for the menu at a given framebuffer height. One place, so
/// the layout and the renderer cannot disagree about how big the text is.
fn menu_font_px(h: f32) -> f32 {
    (h / 220.0).max(2.0).floor()
}

/// Screen-space rect `(x, y, w, h)` of each row, in **physical** pixels.
///
/// Single source of truth: the redraw handler draws the highlight bar from
/// these and the mouse hit-tests against them, so the visible target and the
/// clickable target can never drift apart. This is the padded bar rather than
/// the tight text box, which also makes a comfortably larger click target than
/// the glyphs alone.
///
/// The block is centred vertically rather than pinned to a fixed fraction,
/// because row counts now vary per screen — that keeps every screen balanced
/// and keeps a four-row screen on-screen at small window sizes.
fn menu_item_rects(w: f32, h: f32, rows: &[MenuRow]) -> Vec<(f32, f32, f32, f32)> {
    let px = menu_font_px(h);
    let line = UiPass::text_height(px) * 2.2;
    let pad = px * 4.0;
    let top = (h - line * rows.len() as f32) * 0.5;
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            let tw = UiPass::text_width(&row.label, px);
            let x = (w - tw) * 0.5;
            let y = top + line * i as f32;
            (
                x - pad,
                y - pad * 0.5,
                tw + pad * 2.0,
                UiPass::text_height(px) + pad,
            )
        })
        .collect()
}

/// Index of the row under `(cx, cy)`, if any. Physical pixels, matching both
/// `Window::inner_size` and winit's `CursorMoved` position.
fn menu_hit(w: f32, h: f32, rows: &[MenuRow], cx: f32, cy: f32) -> Option<usize> {
    menu_item_rects(w, h, rows)
        .iter()
        .position(|&(x, y, rw, rh)| cx >= x && cx < x + rw && cy >= y && cy < y + rh)
}

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
/// World transform for **static** scene geometry loaded from glTF (§18). glTF
/// nodes carry arbitrary rotations, which the demo's
/// `Position`/`Rotation(yaw)`/`Scale` triple cannot express. Static, so there is
/// no prev/curr pair — nothing to interpolate.
#[derive(Component)]
struct Transform(Mat4);
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

/// Mesh registry slots that always exist, before any loaded scene's meshes.
/// `App::new` spawns the orb demo against SPHERE/CUBE, and the level pieces use
/// LEVEL_CUBE, so these indices must match the order they are registered in.
const MESH_SPHERE: u32 = 0;
const MESH_CUBE: u32 = 1;
const MESH_LEVEL_CUBE: u32 = 2;
const MESH_BUILTIN_COUNT: u32 = 3;

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

    /// A fixed triangle-mesh collider for loaded scene geometry. `verts` must
    /// already be in **world** space with the collider left at the identity: glTF
    /// nodes routinely carry non-uniform scale, which a rapier `Pose` cannot
    /// express. Returns `None` (and logs) for degenerate meshes rather than
    /// bringing the level down.
    fn add_static_trimesh(
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

/// Camera projection. Shared by the view matrix and the light clusters (§12),
/// which must describe the same frustum or fragments read the wrong cluster.
const CAMERA_FOV_Y: f32 = 60.0 * std::f32::consts::PI / 180.0;
const CAMERA_NEAR: f32 = 0.1;
const CAMERA_FAR: f32 = 200.0;

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

    /// Full camera basis (forward, right, up) — the frame the view frustum's
    /// corners are built in, for fitting shadow cascades.
    fn camera_basis(&self) -> (Vec3, Vec3, Vec3) {
        let fwd = self.forward();
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        (fwd, right, right.cross(fwd).normalize_or_zero())
    }

    /// World → view (right-handed, looking down -Z).
    fn view(&self, eye: Vec3) -> Mat4 {
        Mat4::look_to_rh(eye, self.forward(), Vec3::Y)
    }

    fn view_proj(&self, eye: Vec3, aspect: f32) -> Mat4 {
        let mut proj = Mat4::perspective_rh(CAMERA_FOV_Y, aspect, CAMERA_NEAR, CAMERA_FAR);
        proj.y_axis.y *= -1.0;
        proj * self.view(eye)
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

/// Frames `--bench` waits at the spawn point before sweeping: lets the player
/// settle onto the ground and the GPU clocks ramp up. 2 s at 60 Hz.
const BENCH_SETTLE: u32 = 120;
/// Frames the 360° sweep takes: 0.5° per frame, 12 s at 60 Hz.
const BENCH_SWEEP: u32 = 720;

/// `--bench`: a scripted, repeatable GPU-timing run (§21), so A/B comparisons
/// don't depend on a hand-driven camera. Skips the menu, starts the session at a
/// fixed window size, settles at the spawn point, then turns a full circle at
/// level pitch recording **raw** per-frame times — the logged EMA hides exactly
/// the view-dependent spread that the light-loop cost is about — prints a
/// summary and exits.
struct Bench {
    frame: u32,
    start_yaw: Option<f32>,
    /// Raw GPU times + visible-light count per sweep frame. The two are
    /// `FRAMES_IN_FLIGHT` frames (1°) apart, which is noise for a distribution.
    samples: Vec<(GpuTimes, usize)>,
    lights: usize,
}

impl Bench {
    fn new() -> Self {
        Self {
            frame: 0,
            start_yaw: None,
            samples: Vec::with_capacity(BENCH_SWEEP as usize),
            lights: 0,
        }
    }

    /// Overrides the player's look for this frame: held at the spawn yaw while
    /// settling, then swept through 360°.
    fn drive(&mut self, look: &mut Look) {
        let start = *self.start_yaw.get_or_insert(look.yaw);
        let t = self.frame.saturating_sub(BENCH_SETTLE) as f32 / BENCH_SWEEP as f32;
        look.yaw = start + std::f32::consts::TAU * t.min(1.0);
        look.pitch = 0.0;
    }

    /// Record this frame's readback; true once the sweep is complete. Timings
    /// read back now belong to the frame `FRAMES_IN_FLIGHT` ago, hence the lag.
    fn record(&mut self, raw: GpuTimes) -> bool {
        let lag = FRAMES_IN_FLIGHT as u32;
        if self.frame >= BENCH_SETTLE + lag {
            self.samples.push((raw, self.lights));
        }
        self.frame += 1;
        self.frame >= BENCH_SETTLE + BENCH_SWEEP + lag
    }

    fn report(&self, width: u32, height: u32) {
        // min / p10 / median / p90 / max, over the sweep.
        fn stats(mut v: Vec<f32>) -> String {
            v.sort_by(f32::total_cmp);
            let at = |q: f32| v[((v.len() - 1) as f32 * q).round() as usize];
            format!(
                "min {:6.2}  p10 {:6.2}  med {:6.2}  p90 {:6.2}  max {:6.2}",
                at(0.0),
                at(0.1),
                at(0.5),
                at(0.9),
                at(1.0)
            )
        }
        let col = |f: fn(&GpuTimes) -> f32| stats(self.samples.iter().map(|(t, _)| f(t)).collect());
        let lights = stats(self.samples.iter().map(|&(_, n)| n as f32).collect());
        eprintln!(
            "[bench] {width}x{height}, {} sweep frames (ms)",
            self.samples.len()
        );
        eprintln!("[bench] shadow  {}", col(|t| t.shadow_ms));
        eprintln!("[bench] cluster {}", col(|t| t.cluster_ms));
        eprintln!("[bench] geo     {}", col(|t| t.geometry_ms));
        eprintln!("[bench] post    {}", col(|t| t.post_ms));
        eprintln!("[bench] frame   {}", col(|t| t.frame_ms));
        eprintln!("[bench] lights  {lights}  (visible, after cull)");
    }
}

struct App {
    // FIELD ORDER IS LOAD-BEARING. Rust drops fields in declaration order, and
    // everything above `renderer` owns GPU resources that must be freed while
    // the device and allocator are still alive — §26's "resources → allocator →
    // device" teardown. `session` therefore comes first: it holds the mesh and
    // sky passes, and freeing those after the device is a use-after-free that
    // only shows up on quit.
    /// The loaded world, or `None` in the main menu. Its presence *is* the app
    /// state: `None` = main menu, `Some` + `paused` = pause menu, `Some` +
    /// `!paused` = playing.
    session: Option<Session>,
    // ---- Engine lifetime: created once, survive every session ----
    tonemap: Option<TonemapPass>,
    fxaa: Option<FxaaPass>,
    ui: Option<UiPass>,
    renderer: Option<Renderer>,
    window: Option<Window>,
    input: Input,
    settings: GraphicsSettings,
    /// Paused by Esc: the fixed step stops, the cursor is released for the menu,
    /// and look/movement input is ignored. Rendering continues so the frozen
    /// scene stays on screen behind the overlay (§14's UI focus flag).
    paused: bool,
    /// Pause-menu navigation state (screen + selection + ancestor stack).
    menu: Menu,
    /// Last known cursor position in physical pixels, for menu hit-testing.
    /// `None` until the pointer first moves — winit reports no position before
    /// that, so there is genuinely nothing to hit-test against.
    cursor: Option<(f32, f32)>,
    light_dir: Vec4,
    exposure: f32,
    /// glTF scenes NEW GAME loads (CLI paths). Empty = the procedural demo.
    /// Kept on `App` rather than `Session` so it survives a teardown.
    scenes: Vec<String>,
    /// `--bench` run state; `None` in normal play.
    bench: Option<Bench>,
    last_frame: Instant,
}

/// A punctual light (§12). Position comes from the entity's `Transform`, so a
/// light can ride on geometry (a lamp that emits) or on a bare marker node.
#[derive(Component, Clone, Copy)]
struct PointLight {
    /// Linear colour.
    color: Vec3,
    intensity: f32,
    /// Distance at which the light reaches exactly zero.
    radius: f32,
    /// Physical size of the emitter, in metres (§12's sphere-light specular).
    /// Zero is a true point, which makes a near-singular highlight on smooth
    /// metal; the default is bulb-sized. Clamped to `[0, radius]`.
    source_radius: f32,
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
fn extract_lights(world: &mut World, frustum: &Frustum) -> Vec<GpuLight> {
    let mut out = Vec::new();
    let mut q = world.query::<(&Transform, &PointLight)>();
    for (t, l) in q.iter(world) {
        let pos = t.0.transform_point3(Vec3::ZERO);
        if !frustum.contains_sphere(pos, l.radius) {
            continue;
        }
        out.push(GpuLight {
            pos_radius: [pos.x, pos.y, pos.z, l.radius],
            radiance_source: {
                let c = l.color * l.intensity;
                [c.x, c.y, c.z, l.source_radius]
            },
        });
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
fn light_attenuation(distance: f32, radius: f32) -> f32 {
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
fn sphere_light_specular(n: Vec3, v: Vec3, delta: Vec3, src: f32, roughness: f32) -> f32 {
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
fn point_light_specular(n: Vec3, v: Vec3, delta: Vec3, roughness: f32) -> f32 {
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
fn ggx_d(ndh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    a2 / (std::f32::consts::PI * d * d)
}

#[cfg(test)]
fn smith_g(ndv: f32, ndl: f32, rough: f32) -> f32 {
    let k = (rough + 1.0).powi(2) / 8.0;
    ndv / (ndv * (1.0 - k) + k) * (ndl / (ndl * (1.0 - k) + k))
}

/// What a prefab spawn function gets: the node's placement plus whatever
/// geometry and parameters it carries.
struct SpawnArgs<'a> {
    transform: Mat4,
    /// `None` for a marker node — geometry-free, there only to place a prefab.
    mesh: Option<MeshId>,
    material: u32,
    /// Local-space geometry, for building a collider.
    mesh_data: Option<&'a MeshData>,
    spec: Option<&'a feather_assets::PrefabSpec>,
}

impl SpawnArgs<'_> {
    /// A boolean param, falling back to `default` when absent or the wrong type.
    fn flag(&self, key: &str, default: bool) -> bool {
        self.spec.and_then(|s| s.bool(key)).unwrap_or(default)
    }
}

/// §18's `HashMap<PrefabId, SpawnFn>`: a new kind of thing is a new function
/// here, not a change to the scene format.
type SpawnFn = fn(&mut World, &SpawnArgs);

/// Static level geometry, with two switches read from `params`:
///
/// - `collide` (default true) — off closes §26's "no per-node opt-out"; a few
///   hundred decorative props otherwise each build a trimesh at load.
/// - `shadow` (default true) — off applies `NoShadowCast`. The demo ground
///   already does this (a flat slab casts nothing useful but rasterizes the
///   whole shadow map); this makes it authorable per node instead of hardcoded.
///
/// One parameterised prefab rather than a `no_collide` / `no_shadow` pair: the
/// switches are independent, so separate ids would need one per combination.
fn spawn_prop(world: &mut World, args: &SpawnArgs) {
    let collide = args.flag("collide", true);
    let shadow = args.flag("shadow", true);
    spawn_scene_node(world, args, collide, !shadow);
}

/// A punctual light (§12), optionally attached to geometry. Params `color`
/// (linear rgb), `intensity` and `radius`, each falling back to a sane default
/// so a bare `{"prefab": "point_light"}` still lights something.
///
/// Also honours `prop`'s `collide` and `shadow` switches with the same defaults:
/// a lamp with geometry is still a physical object, so it collides and casts a
/// sun shadow unless the scene says otherwise. Special-casing it to never cast
/// would make `point_light` the one prefab where `shadow` silently did nothing.
///
/// This is the payoff of §18's registry: a new kind of thing is one more
/// function here, with no change to the scene format — the test scene has been
/// carrying `point_light` nodes since before there was a light system.
fn spawn_point_light(world: &mut World, args: &SpawnArgs) {
    let d = PointLight::default();
    let radius = args.spec.and_then(|s| s.f32("radius")).unwrap_or(d.radius);
    let light = PointLight {
        color: args.spec.and_then(|s| s.vec3("color")).unwrap_or(d.color),
        intensity: args
            .spec
            .and_then(|s| s.f32("intensity"))
            .unwrap_or(d.intensity),
        radius,
        // A source larger than the light's reach is meaningless, and a negative
        // one would invert the representative-point clamp in the shader.
        source_radius: args
            .spec
            .and_then(|s| s.f32("source_radius"))
            .unwrap_or(d.source_radius)
            .clamp(0.0, radius.max(0.0)),
    };
    // Geometry is optional: a marker node lights without being visible, a mesh
    // node is a lamp that both emits and renders.
    match spawn_scene_node(
        world,
        args,
        args.flag("collide", true),
        !args.flag("shadow", true),
    ) {
        Some(e) => {
            world.entity_mut(e).insert(light);
        }
        None => {
            world.spawn((Transform(args.transform), light));
        }
    }
}

/// A node with no prefab at all: geometry that collides and casts, which is what
/// every scene node did before prefabs existed.
fn spawn_static_prop(world: &mut World, args: &SpawnArgs) {
    spawn_scene_node(world, args, true, false);
}

/// Shared body of the geometry prefabs.
fn spawn_scene_node(
    world: &mut World,
    args: &SpawnArgs,
    collide: bool,
    no_shadow: bool,
) -> Option<Entity> {
    let mesh = args.mesh?;
    let collider = match (collide, args.mesh_data) {
        (true, Some(data)) => {
            let verts: Vec<Vector> = data
                .vertices
                .iter()
                .map(|v| to_rapier(args.transform.transform_point3(Vec3::from(v.pos))))
                .collect();
            let tris: Vec<[u32; 3]> = data
                .indices
                .chunks_exact(3)
                .map(|t| [t[0], t[1], t[2]])
                .collect();
            world
                .resource_mut::<Physics>()
                .add_static_trimesh(verts, tris)
        }
        _ => None,
    };
    let mut e = world.spawn((
        Transform(args.transform),
        Mesh(mesh),
        Material(args.material),
    ));
    if let Some(c) = collider {
        e.insert(ColliderRef(c));
    }
    if no_shadow {
        e.insert(NoShadowCast);
    }
    Some(e.id())
}

/// `player_start` places the player and is consumed before the world is built,
/// so it has no spawn function of its own — see `Session::new`.
fn prefab_registry() -> HashMap<&'static str, SpawnFn> {
    let mut r: HashMap<&'static str, SpawnFn> = HashMap::new();
    r.insert("prop", spawn_prop as SpawnFn);
    r.insert("point_light", spawn_point_light as SpawnFn);
    r
}

/// Load the mesh registry and every CLI scene. Runs before the world exists,
/// because `player_start` decides where the player is built.
///
/// Built-ins occupy the fixed MESH_* slots first (the demo sphere/cube, then the
/// unit cube every static level piece is scaled from); each scene's meshes are
/// appended and its node indices rebased onto them.
fn load_scenes(scenes: &[String]) -> (Vec<MeshData>, Vec<feather_assets::SceneNode>) {
    let mut meshes: Vec<MeshData> = vec![
        MeshData::uv_sphere(16, 24, 0.5),
        MeshData::cube(1.0),
        MeshData::cube(1.0),
    ];
    let mut nodes: Vec<feather_assets::SceneNode> = Vec::new();
    for path in scenes {
        match feather_assets::load_gltf_scene(path) {
            Ok(scene) => {
                eprintln!(
                    "loaded {path}: {} meshes, {} nodes",
                    scene.meshes.len(),
                    scene.nodes.len()
                );
                let base = meshes.len();
                meshes.extend(scene.meshes);
                nodes.extend(scene.nodes.into_iter().map(|n| feather_assets::SceneNode {
                    mesh: n.mesh.map(|m| base + m),
                    transform: n.transform,
                    prefab: n.prefab,
                }));
            }
            Err(e) => eprintln!("failed to load {path}: {e} (skipped)"),
        }
    }
    (meshes, nodes)
}

/// Where the player starts: a `player_start` marker's position and yaw, or the
/// hardcoded default when a scene does not place one.
fn player_start(nodes: &[feather_assets::SceneNode]) -> (Vec3, Option<f32>) {
    for n in nodes {
        let Some(spec) = n.prefab.as_ref() else {
            continue;
        };
        if spec.id == "player_start" {
            return (
                n.transform.transform_point3(Vec3::ZERO),
                spec.f32("yaw").map(|d| d.to_radians()),
            );
        }
    }
    (Vec3::new(0.0, GROUND_Y, 8.0), None)
}

/// Per-frame view data the render closures need. Carried as an `Option` so the
/// main menu can skip the shadow and geometry passes entirely.
#[derive(Clone, Copy)]
struct FrameView {
    view_proj: Mat4,
    inv_view_proj: Mat4,
    light_dir: Vec4,
    camera_pos: Vec3,
}

/// Everything with **world lifetime**: the simulation and the GPU resources
/// built from it. Split out from `App` so it can be absent — that is what makes
/// a main menu with nothing loaded representable, and what gives the engine a
/// teardown path it previously did not have.
///
/// Dropping a `Session` frees its GPU resources (`MeshRenderer` and `SkyPass`
/// hold RAII buffers/images), so the device must be idle first — see
/// `App::end_session`.
struct Session {
    world: World,
    schedule: Schedule,
    /// The player entity in `world` — sim state in `Player`, render-rate angles
    /// in `Look` (§1: the World owns simulation state).
    player: Entity,
    mesh: MeshRenderer,
    /// Session-scoped because it is **multisampled**: together with the mesh
    /// pipelines it is the only thing that bakes `Renderer::samples()`, which is
    /// precisely why MSAA can change while no session exists.
    sky: SkyPass,
    /// Per-mesh transform that centers + unit-scales it into the demo grid.
    /// Identity for scene meshes — a level must keep its authored size.
    fits: Vec<Mat4>,
    /// Per-mesh **local** (pre-fit) bounding sphere, for frustum culling (§8).
    mesh_spheres: Vec<(Vec3, f32)>,
    /// Fixed-timestep accumulator: real time not yet consumed by a sim step,
    /// carried across frames. Its fraction of FIXED_DT is the render alpha.
    accumulator: f32,
    noclip: bool,
}

impl Session {
    /// Build a world and the GPU resources that serve it. `scenes` are the CLI
    /// glTF paths; empty means the procedural orb demo, exactly as before.
    fn new(renderer: &Renderer, scenes: &[String]) -> Self {
        // Scenes load *first*, because a `player_start` marker (§18) decides
        // where the player goes and the player is built below.
        let (meshes, scene_nodes) = load_scenes(scenes);
        let (start_pos, start_yaw) = player_start(&scene_nodes);

        let mut world = World::new();
        world.insert_resource(FrameCount::default());
        let mut physics = Physics::new();
        // Feet on the ground. The player is a normal ECS entity: sim state in
        // `Player`, render-rate angles in `Look` (§15).
        let body = Player::new(&mut physics, start_pos);
        world.insert_resource(physics);
        world.insert_resource(InputState::default());
        let mut look = Look::new();
        if let Some(yaw) = start_yaw {
            look.yaw = yaw;
        }
        let player = world.spawn((body, look)).id();

        // The drifting orb demo only runs when no scene was given — a loaded level
        // is what you want to look at, and 1000 orbs would bury it.
        let half = (GRID as f32 - 1.0) / 2.0;
        let orb_count = if scenes.is_empty() {
            GRID * GRID * GRID
        } else {
            0
        };
        for i in 0..orb_count {
            let (x, y, z) = (i % GRID, (i / GRID) % GRID, i / (GRID * GRID));
            let pos = Vec3::new(x as f32 - half, y as f32 - half, z as f32 - half) * 1.6;
            let u = i as u32;
            let vel = Vec3::new(
                rand01(u * 3) - 0.5,
                rand01(u * 3 + 1) - 0.5,
                rand01(u * 3 + 2) - 0.5,
            ) * 1.5;
            let spin = (rand01(u * 7 + 11) - 0.5) * 3.0;
            // Procedural sphere or cube, with a random shared palette material.
            let mesh = if rand01(u * 17 + 5) < 0.5 {
                MESH_SPHERE
            } else {
                MESH_CUBE
            };
            let material = (rand01(u * 23 + 7) * PALETTE as f32) as u32 % PALETTE;
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

        // Built-ins get the demo fit (centre + unit-scale); scene meshes keep their
        // authored transform, so identity.
        let fits: Vec<Mat4> = meshes
            .iter()
            .enumerate()
            .map(|(i, m)| {
                if (i as u32) < MESH_BUILTIN_COUNT {
                    fit_transform(m)
                } else {
                    Mat4::IDENTITY
                }
            })
            .collect();
        let mesh_spheres: Vec<(Vec3, f32)> = meshes.iter().map(local_sphere).collect();

        // Material table: shared palette first (indices 0..PALETTE), then each
        // mesh's own material (index PALETTE + mesh_id) — matches the ids that
        // the world build above assigned to entities.
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
            &mut world,
            Vec3::new(0.0, GROUND_Y - 0.5, 0.0),
            Vec3::new(80.0, 1.0, 80.0),
            MESH_LEVEL_CUBE,
            ground_mat,
        );
        // This ground is a flat slab: it casts nothing useful but would rasterize
        // the whole shadow map. It still *receives* shadows (receiving is sampling
        // the map, not being in it). Relief terrain would drop this marker.
        world.entity_mut(ground).insert(NoShadowCast);
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
            spawn_static(&mut world, center, size, MESH_LEVEL_CUBE, box_mat);
        }
        // Scene geometry from the CLI glTF files (§18): one entity per node,
        // carrying that node's world transform, so nodes sharing a mesh draw as
        // instances. Each also gets a fixed trimesh collider so the level is
        // walkable. Material ids follow the same `PALETTE + mesh index` rule the
        // table below is built with.
        let registry = prefab_registry();
        let mut unknown: Vec<&str> = Vec::new();
        // Counted so the effect of `collide: false` is visible in the log rather
        // than having to be taken on trust.
        let colliders_before = world.resource::<Physics>().colliders.len();
        let mut spawned = 0usize;
        for node in &scene_nodes {
            // `player_start` was consumed before the world was built.
            if node.prefab.as_ref().is_some_and(|s| s.id == "player_start") {
                continue;
            }
            let mesh_idx = node.mesh;
            let args = SpawnArgs {
                transform: node.transform,
                mesh: mesh_idx.map(|i| MeshId(i as u32)),
                material: mesh_idx.map_or(0, |i| PALETTE + i as u32),
                mesh_data: mesh_idx.map(|i| &meshes[i]),
                spec: node.prefab.as_ref(),
            };
            match node.prefab.as_ref() {
                Some(spec) => match registry.get(spec.id.as_str()) {
                    Some(f) => f(&mut world, &args),
                    None => {
                        // Unknown ids are authorable-ahead-of-time, not errors:
                        // fall back to static geometry and say so once.
                        if !unknown.contains(&spec.id.as_str()) {
                            eprintln!("unknown prefab {:?}; spawning as static geometry", spec.id);
                            unknown.push(spec.id.as_str());
                        }
                        spawn_static_prop(&mut world, &args);
                    }
                },
                None => spawn_static_prop(&mut world, &args),
            }
            spawned += 1;
        }
        if spawned > 0 {
            let colliders = world.resource::<Physics>().colliders.len() - colliders_before;
            eprintln!("scene: {spawned} nodes spawned, {colliders} colliders built");
        }

        // One step so the broad-phase BVH the character controller shape-casts
        // against contains the level before the first fixed tick.
        world.resource_mut::<Physics>().step();

        let (mesh, _ids) = MeshRenderer::new(renderer, &meshes, &materials, MAX_INSTANCES);
        let sky = SkyPass::new(renderer);

        Self {
            world,
            schedule,
            player,
            mesh,
            sky,
            fits,
            mesh_spheres,
            accumulator: 0.0,
            noclip: false,
        }
    }
}

impl App {
    /// Starts with **no session**: the app opens on the main menu and only
    /// builds a world when NEW GAME is chosen.
    fn new(scenes: Vec<String>, settings: GraphicsSettings, bench: bool) -> Self {
        let light = Vec3::new(-0.4, -1.0, -0.3).normalize();
        Self {
            session: None,
            tonemap: None,
            fxaa: None,
            ui: None,
            renderer: None,
            window: None,
            input: Input::default(),
            settings,
            paused: false,
            menu: Menu::new(),
            cursor: None,
            light_dir: Vec4::new(light.x, light.y, light.z, 0.0),
            exposure: 1.0,
            scenes,
            bench: bench.then(Bench::new),
            last_frame: Instant::now(),
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

impl App {
    /// Grab and hide the cursor for mouselook, or release it for the menu.
    /// Falls back to `Confined` where `Locked` is unsupported, as at startup.
    fn set_cursor_captured(&self, captured: bool) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if captured {
            let _ = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
        }
        window.set_cursor_visible(!captured);
    }

    /// Is the menu taking input? True while paused, and always in the main menu
    /// (where there is no game to take it instead).
    fn menu_active(&self) -> bool {
        self.paused || self.session.is_none()
    }

    /// Whether MSAA is locked — i.e. whether a session owns the pipelines that
    /// baked the sample count.
    fn in_session(&self) -> bool {
        self.session.is_some()
    }

    /// Enter or leave the pause menu. No-op with no session: the main menu is
    /// not a paused game.
    fn set_paused(&mut self, paused: bool) {
        if self.paused == paused || self.session.is_none() {
            return;
        }
        self.paused = paused;
        self.set_cursor_captured(!paused);
        if !paused {
            // Re-opening should always start at the root, not wherever the
            // player happened to leave the menu.
            self.menu.reset(MenuScreen::Root);
            // Resuming: the wall-clock gap while paused is not simulation time.
            // Without this the accumulator sees the whole pause as one frame
            // delta (clamped by MAX_FRAME_TIME, but still a visible jump).
            self.last_frame = Instant::now();
            self.input.jump = false;
        }
    }

    /// Cycle the shadow-quality preset and apply it live (§13).
    ///
    /// Resizing the shadow map frees an image an in-flight frame could still be
    /// sampling, and invalidates the mesh renderer's descriptor pointing at it —
    /// so both happen with the device idle, and the descriptor is re-pointed
    /// immediately afterwards.
    fn cycle_shadow_quality(&mut self) {
        self.settings.shadows = self.settings.shadows.next();
        self.apply_shadow_quality();
    }

    /// Push `settings.fxaa` into the renderer. Cheap: just a flag, since the
    /// LDR intermediate is always allocated.
    fn apply_fxaa(&mut self) {
        if let Some(r) = self.renderer.as_mut() {
            r.set_fxaa(self.settings.fxaa);
        }
        eprintln!(
            "[quality] fxaa: {}",
            if self.settings.fxaa { "on" } else { "off" }
        );
    }

    /// Act on what the menu decided. Keeps the renderer and event loop out of
    /// `Menu` itself, so its logic stays pure and testable.
    fn handle_menu_outcome(&mut self, outcome: MenuOutcome, event_loop: &ActiveEventLoop) {
        match outcome {
            MenuOutcome::Stay => {}
            MenuOutcome::Resume => self.set_paused(false),
            MenuOutcome::Quit => event_loop.exit(),
            MenuOutcome::ApplyShadows => self.apply_shadow_quality(),
            MenuOutcome::ApplyFxaa => self.apply_fxaa(),
            MenuOutcome::ApplyMsaa => self.apply_msaa(),
            MenuOutcome::StartSession => self.start_session(),
            MenuOutcome::EndSession => self.end_session(),
        }
    }

    /// Apply whatever `settings.shadows` currently is. Separate from cycling so
    /// F1 and the menu row drive the same path and cannot drift.
    fn apply_shadow_quality(&mut self) {
        let dim = self.settings.shadows.dim();
        if let Some(r) = self.renderer.as_mut() {
            r.wait_idle();
            r.set_shadow_dim(dim);
            // Re-point the descriptor only when a session is holding one; with
            // no world loaded a fresh `MeshRenderer` will bind the new map when
            // the next session starts.
            if let Some(s) = self.session.as_mut() {
                s.mesh
                    .set_shadow_map(r.shadow_view(), r.shadow_sampler(), dim);
            }
        }
        eprintln!("[quality] shadows: {}", self.settings.shadows.label());
    }

    /// Build a world from the CLI scenes and drop into it.
    fn start_session(&mut self) {
        let Some(renderer) = self.renderer.as_ref() else {
            return;
        };
        let session = Session::new(renderer, &self.scenes);
        self.session = Some(session);
        self.paused = false;
        self.menu.reset(MenuScreen::Root);
        self.set_cursor_captured(true);
        // The build took real wall-clock time that is not simulation time.
        self.last_frame = Instant::now();
    }

    /// Tear the world down and return to the main menu.
    ///
    /// **Idle first.** `Session` owns GPU buffers and images that a queued frame
    /// may still be reading; dropping them while in flight is a use-after-free
    /// that validation catches. A hitch here is free — nothing is animating.
    fn end_session(&mut self) {
        if let Some(r) = self.renderer.as_ref() {
            r.wait_idle();
        }
        self.session = None;
        self.paused = false;
        self.menu.reset(MenuScreen::MainRoot);
        self.set_cursor_captured(false);
    }

    /// Apply `settings.msaa`. Only reachable with no session loaded, because the
    /// mesh and sky pipelines bake the sample count at creation.
    fn apply_msaa(&mut self) {
        if let Some(r) = self.renderer.as_mut() {
            r.set_msaa(self.settings.msaa);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let mut attrs = feather_platform::window_attributes("feather — first person");
        if self.bench.is_some() {
            // A fixed extent, so bench runs are comparable (the compositor may
            // still override it; the report prints what was actually used).
            attrs = attrs
                .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
                .with_resizable(false);
        }
        let window = event_loop.create_window(attrs).expect("create window");
        // Opens on the main menu, so the cursor starts free rather than grabbed.
        window.set_cursor_visible(true);

        let size = window.inner_size();
        let renderer = Renderer::new(&window, size.width, size.height, self.settings.msaa)
            .expect("create renderer");

        // Engine-lifetime passes only. All three are single-sample by design, so
        // none of them cares about the MSAA setting; the two that do (mesh, sky)
        // belong to a `Session` and are built when one starts.
        let tonemap = TonemapPass::new(&renderer);
        let fxaa = FxaaPass::new(&renderer);
        let ui = UiPass::new(&renderer);

        self.tonemap = Some(tonemap);
        self.fxaa = Some(fxaa);
        self.ui = Some(ui);
        self.renderer = Some(renderer);
        self.window = Some(window);
        if self.bench.is_some() {
            self.start_session();
        }
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
                        // F1 cycles the shadow-quality preset (§13).
                        KeyCode::F1 if pressed => self.cycle_shadow_quality(),
                        // F2 toggles FXAA (§13).
                        KeyCode::F2 if pressed => {
                            self.settings.fxaa = !self.settings.fxaa;
                            self.apply_fxaa();
                        }
                        // V toggles noclip (free flight) for inspecting the scene.
                        KeyCode::KeyV if pressed => {
                            if let Some(s) = self.session.as_mut() {
                                s.noclip = !s.noclip;
                            }
                        }
                        // Exposure control (showcases the HDR/tonemap pipeline).
                        KeyCode::BracketLeft if pressed => {
                            self.exposure = (self.exposure * 0.8).max(0.05);
                        }
                        KeyCode::BracketRight if pressed => {
                            self.exposure = (self.exposure * 1.25).min(16.0);
                        }
                        // Esc walks back out one screen at a time, and only
                        // unpauses from the root — so leaving a submenu does not
                        // dump you straight into the game.
                        KeyCode::Escape if pressed => {
                            if self.session.is_some() && !self.paused {
                                self.set_paused(true);
                            } else {
                                // In a menu: step back one screen. At the main
                                // root that is a no-op — there is nothing to
                                // resume into.
                                let outcome = self.menu.back();
                                self.handle_menu_outcome(outcome, event_loop);
                            }
                        }
                        KeyCode::ArrowUp if pressed && self.menu_active() => {
                            let in_session = self.in_session();
                            self.menu.move_by(-1, &self.settings, in_session);
                        }
                        KeyCode::ArrowDown if pressed && self.menu_active() => {
                            let in_session = self.in_session();
                            self.menu.move_by(1, &self.settings, in_session);
                        }
                        KeyCode::Enter if pressed && self.menu_active() => {
                            let in_session = self.in_session();
                            let outcome = self.menu.activate(&mut self.settings, in_session);
                            self.handle_menu_outcome(outcome, event_loop);
                        }
                        _ => {}
                    }
                }
            }
            // Mouse in the pause menu (§19). Only while paused: mouselook is a
            // DeviceEvent on a separate path, so gameplay is untouched.
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = Some((position.x as f32, position.y as f32));
                if self.menu_active() {
                    if let Some(window) = self.window.as_ref() {
                        let size = window.inner_size();
                        let in_session = self.session.is_some();
                        // Hover drives the *existing* selection rather than a
                        // second highlight state, so keyboard and mouse stay
                        // interchangeable mid-interaction. Off the entries the
                        // selection is left alone, so something is always
                        // selected for Enter.
                        let rows = self.menu.rows(&self.settings, in_session);
                        if let Some(i) = menu_hit(
                            size.width as f32,
                            size.height as f32,
                            &rows,
                            position.x as f32,
                            position.y as f32,
                        ) {
                            self.menu.hover(i, &self.settings, in_session);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if self.menu_active()
                    && button == MouseButton::Left
                    && state == ElementState::Pressed
                {
                    if let (Some(window), Some((cx, cy))) = (self.window.as_ref(), self.cursor) {
                        let size = window.inner_size();
                        // Activate only when actually over an entry: a stray
                        // click on the dimmed backdrop should do nothing.
                        let in_session = self.session.is_some();
                        let rows = self.menu.rows(&self.settings, in_session);
                        if let Some(i) =
                            menu_hit(size.width as f32, size.height as f32, &rows, cx, cy)
                        {
                            self.menu.hover(i, &self.settings, in_session);
                            let outcome = self.menu.activate(&mut self.settings, in_session);
                            self.handle_menu_outcome(outcome, event_loop);
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                // Paused: swallow the accumulated look delta so releasing Esc
                // does not snap the camera by the whole menu's worth of motion.
                if self.paused {
                    self.input.mouse_dx = 0.0;
                    self.input.mouse_dy = 0.0;
                }

                // Simulation, camera and extract all belong to a loaded world.
                // With no session this is skipped entirely and the frame becomes
                // "clear the HDR target, tonemap it, draw the menu over it" —
                // the geometry pass already clears, so the main menu needs no
                // background path of its own.
                let mut frame_view: Option<FrameView> = None;
                if let Some(s) = self.session.as_mut() {
                    // Look updates at render rate for responsive aim (§15).
                    let sens = if self.paused { 0.0 } else { 0.0025 };
                    let look = {
                        let mut look = s.world.get_mut::<Look>(s.player).expect("player has Look");
                        look.yaw += self.input.mouse_dx * sens;
                        look.pitch = (look.pitch - self.input.mouse_dy * sens).clamp(-1.54, 1.54);
                        if let Some(b) = self.bench.as_mut() {
                            b.drive(&mut look);
                        }
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
                        let mut state = s.world.resource_mut::<InputState>();
                        state.wish = wish;
                        state.vertical = vgo;
                        state.noclip = s.noclip;
                        state.jump |= self.input.jump;
                    }
                    self.input.jump = false;

                    // Fixed-timestep sim: consume the accumulator in whole FIXED_DT
                    // steps. The schedule advances gameplay and brackets the rapier step
                    // with the two sync systems (§15). The frame delta is clamped and
                    // MAX_STEPS caps catch-up per frame (spiral-of-death guard);
                    // leftover beyond the cap is dropped.
                    // Paused: no simulation advances, so prev == curr and the frozen
                    // scene keeps rendering behind the menu.
                    s.accumulator += if self.paused {
                        0.0
                    } else {
                        dt.min(MAX_FRAME_TIME)
                    };
                    let mut steps = 0;
                    while s.accumulator >= FIXED_DT && steps < MAX_STEPS {
                        s.schedule.run(&mut s.world);
                        s.accumulator -= FIXED_DT;
                        steps += 1;
                    }
                    // How far we are into the next step, in [0,1): the render alpha.
                    let alpha = (s.accumulator / FIXED_DT).clamp(0.0, 1.0);

                    // Camera + sun matrices first — the extract loop culls against them.
                    // Camera: interpolate the player's body, offset to eye height.
                    // Copy the pair out before the extract query re-borrows the World.
                    let (p_prev, p_pos) = {
                        let p = s.world.get::<Player>(s.player).expect("player body");
                        (p.prev_pos, p.pos)
                    };
                    let eye = p_prev.lerp(p_pos, alpha) + Vec3::new(0.0, EYE_HEIGHT, 0.0);
                    let size = self.window.as_ref().unwrap().inner_size();
                    let aspect = size.width as f32 / size.height.max(1) as f32;
                    let view_proj = look.view_proj(eye, aspect);
                    let cluster_view = ClusterView {
                        view: look.view(eye),
                        fov_y: CAMERA_FOV_Y,
                        aspect,
                        near: CAMERA_NEAR,
                        far: CAMERA_FAR,
                    };
                    let inv_view_proj = view_proj.inverse();
                    let light_dir = self.light_dir;
                    let camera_pos = eye;

                    // Sun shadow matrix (§11): a tight ortho that **follows the player**,
                    // so shadows exist wherever you walk instead of only near the origin.
                    // It is centered on the player's body, not the view direction, so
                    // turning never disturbs the shadow map — only walking moves it, and
                    // the texel snap below keeps that from crawling.
                    let sun_dir = self.light_dir.truncate().normalize_or_zero();
                    // One ortho per cascade, each fitted to the bounding sphere of
                    // its slice of the view frustum and texel-snapped.
                    let dim = self.settings.shadows.dim();
                    let splits = cascade_splits(0.1, SHADOW_DISTANCE, SHADOW_LAMBDA);
                    let (fwd, right_v, up_v) = look.camera_basis();
                    let mut cascades = [CascadeSetup {
                        view_proj: Mat4::IDENTITY,
                        texel_world: 0.0,
                    }; SHADOW_CASCADES];
                    let mut near = 0.1f32;
                    for (i, far) in splits.iter().copied().enumerate() {
                        let (centre, radius) = slice_sphere(
                            eye,
                            fwd,
                            right_v,
                            up_v,
                            60f32.to_radians(),
                            aspect,
                            near,
                            far,
                        );
                        let (view_proj, texel_world) = fit_cascade(centre, radius, sun_dir, dim);
                        cascades[i] = CascadeSetup {
                            view_proj,
                            texel_world,
                        };
                        near = far;
                    }

                    // Per-view frustum culling (§8): bounding sphere vs six planes, once
                    // per view. Camera frustum trims the main pass; each cascade's ortho
                    // trims its own caster set.
                    let camera_frustum = Frustum::from_view_proj(&view_proj);
                    let light_frusta: [Frustum; SHADOW_CASCADES] =
                        std::array::from_fn(|i| Frustum::from_view_proj(&cascades[i].view_proj));
                    let lights = extract_lights(&mut s.world, &camera_frustum);

                    // Extract: interpolate each entity's sim state (prev -> curr) by
                    // alpha, build its model matrix, and route it to the camera-visible
                    // set (main pass) and/or the light-visible set (shadow pass). This is
                    // the sim<->render seam; interpolation lives here per §4. Static level
                    // pieces lack Velocity/Spin so `integrate` skips them; prev == curr.
                    // `Off` simply stops feeding casters: the map is then only cleared,
                    // so every fragment compares against 1.0 and reads as lit.
                    let casts = self.settings.shadows.casts();
                    let fits = &s.fits;
                    let spheres = &s.mesh_spheres;
                    let mesh_max = fits.len().saturating_sub(1);
                    let cap = (GRID * GRID * GRID) as usize + 8;
                    let mut main_items: Vec<(MeshId, InstanceData)> = Vec::with_capacity(cap);
                    // One caster list per cascade: a caster can be in several,
                    // since the cascades overlap in world space.
                    let mut shadow_items: Vec<Vec<(MeshId, InstanceData)>> = (0..SHADOW_CASCADES)
                        .map(|_| Vec::with_capacity(cap))
                        .collect();
                    let mut q = s.world.query::<(
                        &Position,
                        &PrevPosition,
                        &Rotation,
                        &PrevRotation,
                        &Scale,
                        &Mesh,
                        &Material,
                        Option<&NoShadowCast>,
                    )>();
                    for (p, pp, r, pr, scale, mesh, material, no_cast) in q.iter(&s.world) {
                        let id = (mesh.0 .0 as usize).min(mesh_max);
                        let pos = pp.0.lerp(p.0, alpha);
                        let angle = pr.0 + (r.0 - pr.0) * alpha;
                        let model = Mat4::from_translation(pos)
                            * Mat4::from_rotation_y(angle)
                            * Mat4::from_scale(scale.0)
                            * fits[id];
                        let item = (MeshId(id as u32), InstanceData::new(model, material.0));
                        let (c, radius) = world_sphere(&model, spheres[id]);
                        if camera_frustum.contains_sphere(c, radius) {
                            main_items.push(item);
                        }
                        if casts && no_cast.is_none() {
                            for (ci, f) in light_frusta.iter().enumerate() {
                                if f.contains_sphere(c, radius) {
                                    shadow_items[ci].push(item);
                                }
                            }
                        }
                    }

                    // Static scene geometry (§18): the node matrix *is* the transform,
                    // so there is nothing to interpolate. Separate query rather than an
                    // Option<> branch, to keep the two archetypes clean.
                    let mut qs = s
                        .world
                        .query::<(&Transform, &Mesh, &Material, Option<&NoShadowCast>)>();
                    for (t, mesh, material, no_cast) in qs.iter(&s.world) {
                        let id = (mesh.0 .0 as usize).min(mesh_max);
                        let model = t.0 * fits[id];
                        let item = (MeshId(id as u32), InstanceData::new(model, material.0));
                        let (c, radius) = world_sphere(&model, spheres[id]);
                        if camera_frustum.contains_sphere(c, radius) {
                            main_items.push(item);
                        }
                        if casts && no_cast.is_none() {
                            for (ci, f) in light_frusta.iter().enumerate() {
                                if f.contains_sphere(c, radius) {
                                    shadow_items[ci].push(item);
                                }
                            }
                        }
                    }

                    // CPU prep once (sort + stage instances/globals); the shadow
                    // and main passes then replay them. Uploads happen in
                    // draw_shadow, after the frame fence.
                    s.mesh.prepare_frame(
                        &mut main_items,
                        &mut shadow_items,
                        &cascades,
                        &lights,
                        &cluster_view,
                    );
                    if let Some(b) = self.bench.as_mut() {
                        b.lights = lights.len();
                    }
                    frame_view = Some(FrameView {
                        view_proj,
                        inv_view_proj,
                        light_dir,
                        camera_pos,
                    });
                }

                // Overlay geometry is built on the CPU here; the upload and draw
                // happen inside the UI pass below.
                if let Some(ui) = self.ui.as_mut() {
                    let size = self.window.as_ref().unwrap().inner_size();
                    ui.begin(size.width, size.height);
                    if self.paused || self.session.is_none() {
                        let (w, h) = (size.width as f32, size.height as f32);
                        // Dim the frozen scene. Colours are linear: this is drawn
                        // into the _SRGB swapchain, which encodes on store. In the
                        // main menu there is no scene behind it, just the geometry
                        // pass's clear — the same rect darkens it to a backdrop.
                        ui.rect(0.0, 0.0, w, h, [0.0, 0.0, 0.0, 0.55]);

                        let px = menu_font_px(h);
                        let title_px = px * 1.6;
                        // Title names the current screen, so a submenu is
                        // self-identifying.
                        let title = self.menu.screen.title();
                        ui.text(
                            (w - UiPass::text_width(title, title_px)) * 0.5,
                            h * 0.28,
                            title_px,
                            [0.9, 0.9, 0.9, 1.0],
                            title,
                        );

                        // Rows and rects come from the same functions the mouse
                        // hit-tests against, so highlight and click target match.
                        let rows = self.menu.rows(&self.settings, self.session.is_some());
                        let rects = menu_item_rects(w, h, &rows);
                        let pad = px * 4.0;
                        for (i, row) in rows.iter().enumerate() {
                            let selected = i == self.menu.index;
                            let (rx, ry, rw, rh) = rects[i];
                            if selected {
                                // Highlight bar behind the current entry.
                                ui.rect(rx, ry, rw, rh, [0.25, 0.45, 0.85, 0.85]);
                            }
                            // Inert rows read dimmer: they are information or a
                            // planned setting, not something you can change.
                            let color = match (selected, row.enabled()) {
                                (true, true) => [1.0, 1.0, 1.0, 1.0],
                                (true, false) => [0.72, 0.72, 0.72, 1.0],
                                (false, true) => [0.65, 0.65, 0.65, 1.0],
                                (false, false) => [0.40, 0.40, 0.40, 1.0],
                            };
                            // Text sits inset from the bar by the same padding
                            // the rect was grown by.
                            ui.text(rx + pad, ry + pad * 0.5, px, color, &row.label);
                        }
                    }
                }

                let exposure = self.exposure;
                if let (Some(r), Some(tm), Some(fx), Some(ui)) = (
                    self.renderer.as_mut(),
                    self.tonemap.as_mut(),
                    self.fxaa.as_mut(),
                    self.ui.as_ref(),
                ) {
                    // HDR view/sampler are stable except across resize; capture
                    // before the mutable draw_frame borrow, refresh in `update`.
                    // No session, or shadows off: the cascade passes collapse to
                    // one layered clear rather than four empty passes.
                    r.set_shadow_casters(self.session.is_some() && self.settings.shadows.casts());
                    let hdr_view = r.hdr_view();
                    let ldr_view = r.ldr_view();
                    let hdr_sampler = r.hdr_sampler();
                    // Shared immutably by the shadow and geometry closures — the
                    // draw methods take &self, only `prepare_frame` above needed
                    // &mut, and that already ran.
                    let session = self.session.as_ref();
                    r.draw_frame(
                        // Shadow pass: sun depth map (also flushes this frame's
                        // buffers). No session means no casters and no buffers —
                        // the pass still clears, so every fragment reads as lit.
                        |cmd, extent, frame, cascade| {
                            if let Some(s) = session {
                                s.mesh.draw_shadow(cmd, extent, frame, cascade);
                            }
                        },
                        // Light clusters (§12): after draw_shadow has uploaded
                        // this frame's lights + globals, before the main pass.
                        |cmd, frame| {
                            if let Some(s) = session {
                                s.mesh.dispatch_clusters(cmd, frame);
                            }
                        },
                        // Geometry (§10): depth prepass, then the lit opaque pass
                        // (each pixel shaded once), then the sky depth-tested into
                        // whatever background is left. Skipped wholesale in the
                        // main menu, leaving the attachment's clear colour.
                        |cmd, extent, frame| {
                            if let (Some(s), Some(v)) = (session, frame_view) {
                                s.mesh.draw_depth_prepass(
                                    cmd,
                                    extent,
                                    frame,
                                    v.view_proj,
                                    v.light_dir,
                                    v.camera_pos,
                                );
                                s.mesh.draw_main(
                                    cmd,
                                    extent,
                                    frame,
                                    v.view_proj,
                                    v.light_dir,
                                    v.camera_pos,
                                );
                                s.sky
                                    .draw(cmd, extent, v.inv_view_proj, v.camera_pos, v.light_dir);
                            }
                        },
                        |cmd, extent, frame| {
                            tm.update(frame, hdr_view, hdr_sampler);
                            tm.draw(cmd, extent, frame, exposure);
                        },
                        // Only invoked when FXAA is enabled.
                        |cmd, extent, frame| {
                            fx.update(frame, ldr_view, hdr_sampler);
                            fx.draw(cmd, extent, frame);
                        },
                        // Overlay, blended over the finished frame. No-op when the
                        // menu is closed (nothing was built).
                        |cmd, extent, frame| ui.draw(cmd, extent, frame),
                    );
                }

                if let (Some(b), Some(r)) = (self.bench.as_mut(), self.renderer.as_ref()) {
                    if self.session.is_some() && b.record(r.gpu_times_raw()) {
                        let size = self
                            .window
                            .as_ref()
                            .map(|w| w.inner_size())
                            .unwrap_or_default();
                        b.report(size.width, size.height);
                        event_loop.exit();
                    }
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

/// Practical/parallel-split scheme (§11): the far depth of each cascade, blending
/// a logarithmic and a uniform distribution by `lambda`.
///
/// Logarithmic alone starves the far cascade; uniform alone wastes most of the
/// resolution on distance nobody looks at. `lambda` picks between them.
fn cascade_splits(near: f32, far: f32, lambda: f32) -> [f32; SHADOW_CASCADES] {
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
fn slice_sphere(
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
fn fit_cascade(centre: Vec3, radius: f32, sun_dir: Vec3, dim: u32) -> (Mat4, f32) {
    let texel_world = (2.0 * radius) / dim as f32;
    let basis = Mat4::look_at_rh(Vec3::ZERO, sun_dir, Vec3::Y);
    let c = basis.transform_point3(centre);
    let snapped = Vec3::new(
        (c.x / texel_world).round() * texel_world,
        (c.y / texel_world).round() * texel_world,
        c.z,
    );
    let centre = basis.inverse().transform_point3(snapped);
    // Pull the light back past the sphere so casters above it are still inside
    // the depth range. Without pancaking (§11, not yet implemented) a caster
    // further than this toward the light is clipped and stops casting.
    let back = radius + SHADOW_BACK;
    let light_eye = centre - sun_dir * back;
    let view = Mat4::look_at_rh(light_eye, centre, Vec3::Y);
    let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.0, back + radius);
    (proj * view, texel_world)
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

/// A mesh's **local** (pre-fit) bounding sphere `(centre, radius)`. Extract maps
/// it through the model matrix — which already includes the fit — to get the
/// world sphere the frustum test uses (§8).
fn local_sphere(mesh: &MeshData) -> (Vec3, f32) {
    let (min, max) = mesh.bounds();
    let c = (min + max) * 0.5;
    (c, (max - c).length())
}

/// World bounding sphere of `local` under `model`: the centre transforms with it,
/// and the radius scales by the model's largest axis scale — conservative under
/// non-uniform scale, which is what culling wants.
fn world_sphere(model: &Mat4, local: (Vec3, f32)) -> (Vec3, f32) {
    let s = model
        .x_axis
        .truncate()
        .length()
        .max(model.y_axis.truncate().length())
        .max(model.z_axis.truncate().length());
    (model.transform_point3(local.0), local.1 * s)
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
    // CLI paths are glTF *scenes* to load and walk around
    // (`cargo run -- level.glb`); each node becomes its own entity. With no args,
    // the procedural drifting-orb demo runs instead. `--msaa N` (1/2/4/8) picks
    // the geometry-pass sample count, clamped to device support. `--bench` runs
    // the scripted timing sweep (see `Bench`) instead of the menu.
    let mut settings = GraphicsSettings::default();
    let mut scenes: Vec<String> = Vec::new();
    let mut bench = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--msaa" => match args.next().and_then(|v| v.parse::<u32>().ok()) {
                Some(n) => settings.msaa = n,
                None => eprintln!("--msaa needs a sample count (1/2/4/8); ignoring"),
            },
            "--bench" => bench = true,
            _ => scenes.push(a),
        }
    }

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(scenes, settings, bench);
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

    // ---- Pause menu: layout, hit-testing and navigation (§19) ----
    //
    // `Menu` deliberately depends on neither the renderer nor the event loop,
    // and the layout is a pure function of the framebuffer size, so all of this
    // runs without a GPU or a window.

    /// A few sizes worth covering: a typical window, a wide one, and one small
    /// enough that `menu_font_px` clamps to its 2.0 floor.
    const SIZES: [(f32, f32); 3] = [(1280.0, 720.0), (2560.0, 1440.0), (320.0, 200.0)];

    const SCREENS: [MenuScreen; 6] = [
        MenuScreen::MainRoot,
        MenuScreen::Root,
        MenuScreen::Options,
        MenuScreen::Graphics,
        MenuScreen::Sound,
        MenuScreen::Gameplay,
    ];

    #[test]
    fn menu_rect_centres_hit_their_own_row() {
        let s = GraphicsSettings::default();
        for screen in SCREENS {
            let rows = screen_rows(screen, &s, true);
            assert!(!rows.is_empty(), "{screen:?} has no rows");
            for (w, h) in SIZES {
                for (i, &(x, y, rw, rh)) in menu_item_rects(w, h, &rows).iter().enumerate() {
                    let (cx, cy) = (x + rw * 0.5, y + rh * 0.5);
                    assert_eq!(
                        menu_hit(w, h, &rows, cx, cy),
                        Some(i),
                        "{screen:?} row {i} at {w}x{h} did not hit itself"
                    );
                }
            }
        }
    }

    #[test]
    fn menu_rows_are_ordered_disjoint_and_onscreen() {
        let s = GraphicsSettings::default();
        for screen in SCREENS {
            let rows = screen_rows(screen, &s, true);
            for (w, h) in SIZES {
                let rects = menu_item_rects(w, h, &rows);
                for pair in rects.windows(2) {
                    let (_, y0, _, h0) = pair[0];
                    let (_, y1, _, _) = pair[1];
                    assert!(y1 > y0, "{screen:?} rows out of order at {w}x{h}");
                    assert!(y1 >= y0 + h0, "{screen:?} rows overlap at {w}x{h}");
                }
                // Vertical centring must keep even the longest screen on screen.
                for (x, y, rw, rh) in rects {
                    assert!(
                        y >= 0.0 && y + rh <= h,
                        "{screen:?} row off screen at {w}x{h}"
                    );
                    assert!(
                        ((x + rw * 0.5) - w * 0.5).abs() < 0.001,
                        "{screen:?} not centred"
                    );
                }
            }
        }
    }

    #[test]
    fn menu_misses_gaps_and_backdrop() {
        let s = GraphicsSettings::default();
        let rows = screen_rows(MenuScreen::Root, &s, true);
        let (w, h) = (1280.0, 720.0);
        let rects = menu_item_rects(w, h, &rows);
        let gap_y = (rects[0].1 + rects[0].3 + rects[1].1) * 0.5;
        assert_eq!(menu_hit(w, h, &rows, w * 0.5, gap_y), None, "gap hit");
        assert_eq!(menu_hit(w, h, &rows, w * 0.5, 0.0), None, "top hit");
        assert_eq!(menu_hit(w, h, &rows, w * 0.5, h - 1.0), None, "bottom hit");
        assert_eq!(
            menu_hit(w, h, &rows, 0.0, rects[0].1 + 1.0),
            None,
            "left hit"
        );
        assert_eq!(
            menu_hit(w, h, &rows, w - 1.0, rects[0].1 + 1.0),
            None,
            "right hit"
        );
    }

    /// A menu as it exists with a world loaded: the app resets to `Root` when a
    /// session starts, so these tests begin where the pause menu does.
    fn in_game_menu() -> Menu {
        let mut m = Menu::new();
        m.reset(MenuScreen::Root);
        m
    }

    /// Select the row whose action matches, then activate it.
    fn activate(m: &mut Menu, s: &mut GraphicsSettings, want: MenuAction) -> MenuOutcome {
        let i = m
            .rows(s, true)
            .iter()
            .position(|r| r.action == want)
            .unwrap_or_else(|| panic!("no {want:?} row on {:?}", m.screen));
        m.index = i;
        m.activate(s, true)
    }

    #[test]
    fn back_restores_the_row_you_descended_from() {
        let mut s = GraphicsSettings::default();
        let mut m = in_game_menu();
        activate(&mut m, &mut s, MenuAction::Enter(MenuScreen::Options));
        assert_eq!(m.screen, MenuScreen::Options);
        // Descend from GAMEPLAY specifically, not the first row, so a restored
        // index is distinguishable from a reset one.
        activate(&mut m, &mut s, MenuAction::Enter(MenuScreen::Gameplay));
        assert_eq!(m.screen, MenuScreen::Gameplay);
        assert_eq!(m.index, 0, "entering a screen starts at the top");

        assert_eq!(m.back(), MenuOutcome::Stay);
        assert_eq!(m.screen, MenuScreen::Options);
        let gameplay_row = screen_rows(MenuScreen::Options, &s, true)
            .iter()
            .position(|r| r.action == MenuAction::Enter(MenuScreen::Gameplay))
            .unwrap();
        assert_eq!(m.index, gameplay_row, "BACK did not restore the selection");

        assert_eq!(m.back(), MenuOutcome::Stay);
        assert_eq!(m.screen, MenuScreen::Root);
        // Root has no ancestor, so backing out again leaves the menu entirely.
        assert_eq!(m.back(), MenuOutcome::Resume);
    }

    #[test]
    fn graphics_rows_change_the_settings() {
        let mut s = GraphicsSettings::default();
        let mut m = in_game_menu();
        m.screen = MenuScreen::Graphics;

        let before = s.fxaa;
        assert_eq!(
            activate(&mut m, &mut s, MenuAction::ToggleFxaa),
            MenuOutcome::ApplyFxaa
        );
        assert_eq!(s.fxaa, !before, "FXAA row did not toggle the setting");

        // Cycling steps down and wraps: High -> Medium -> Low -> Off -> High.
        s.shadows = ShadowQuality::High;
        for want in [
            ShadowQuality::Medium,
            ShadowQuality::Low,
            ShadowQuality::Off,
            ShadowQuality::High,
        ] {
            assert_eq!(
                activate(&mut m, &mut s, MenuAction::CycleShadows),
                MenuOutcome::ApplyShadows
            );
            assert!(s.shadows == want, "unexpected shadow step");
        }
    }

    #[test]
    fn inert_rows_change_nothing() {
        let mut m = in_game_menu();
        for screen in [
            MenuScreen::Graphics,
            MenuScreen::Sound,
            MenuScreen::Gameplay,
        ] {
            let mut s = GraphicsSettings::default();
            m.screen = screen;
            m.stack.clear();
            let inert: Vec<usize> = screen_rows(screen, &s, true)
                .iter()
                .enumerate()
                .filter(|(_, r)| !r.enabled())
                .map(|(i, _)| i)
                .collect();
            assert!(!inert.is_empty(), "{screen:?} has no inert row to check");
            for i in inert {
                m.index = i;
                assert_eq!(m.activate(&mut s, true), MenuOutcome::Stay);
                // The MSAA row in particular must not quietly mutate anything.
                assert_eq!(s.shadows, ShadowQuality::High);
                assert!(!s.fxaa);
                assert_eq!(s.msaa, 1);
                assert_eq!(m.screen, screen, "inert row navigated away");
            }
        }
    }

    #[test]
    fn selection_wraps_in_both_directions() {
        let s = GraphicsSettings::default();
        let mut m = in_game_menu();
        m.screen = MenuScreen::Graphics;
        let n = m.rows(&s, true).len();
        m.index = 0;
        m.move_by(-1, &s, true);
        assert_eq!(m.index, n - 1, "up from the top did not wrap");
        m.move_by(1, &s, true);
        assert_eq!(m.index, 0, "down from the bottom did not wrap");
    }

    #[test]
    fn unpausing_resets_to_the_root_screen() {
        let mut s = GraphicsSettings::default();
        let mut m = in_game_menu();
        activate(&mut m, &mut s, MenuAction::Enter(MenuScreen::Options));
        activate(&mut m, &mut s, MenuAction::Enter(MenuScreen::Graphics));
        m.reset(MenuScreen::Root);
        assert_eq!(m.screen, MenuScreen::Root);
        assert_eq!(m.index, 0);
        assert!(m.stack.is_empty());
    }

    /// Menu labels must stay inside what the 5x7 UI font can draw (A-Z, 0-9 and
    /// space); anything else renders as a blank, which would silently mangle a
    /// row. Guards against someone adding a colon or brackets later.
    #[test]
    fn menu_labels_are_drawable() {
        let s = GraphicsSettings::default();
        for screen in SCREENS {
            for row in screen_rows(screen, &s, true) {
                for c in row.label.chars() {
                    assert!(
                        c.is_ascii_uppercase() || c.is_ascii_digit() || c == ' ',
                        "{screen:?} label {:?} has undrawable {c:?}",
                        row.label
                    );
                }
            }
        }
    }

    // ---- Session lifetime and the main menu ----

    #[test]
    fn msaa_is_changeable_only_outside_a_session() {
        let s = GraphicsSettings::default();
        let row_action = |in_session: bool| {
            screen_rows(MenuScreen::Graphics, &s, in_session)
                .into_iter()
                .find(|r| r.label.starts_with("MSAA"))
                .expect("graphics has an MSAA row")
                .action
        };
        // In game the pipelines have baked the sample count, so the row must be
        // inert — this is the guard that stops MSAA changing under live pipelines.
        assert_eq!(row_action(true), MenuAction::Inert);
        assert_eq!(row_action(false), MenuAction::CycleMsaa);
    }

    #[test]
    fn msaa_row_cycles_supported_counts() {
        let mut s = GraphicsSettings::default();
        let mut m = Menu::new();
        m.screen = MenuScreen::Graphics;
        assert_eq!(s.msaa, 1);
        for want in [2, 4, 8, 1] {
            let i = m
                .rows(&s, false)
                .iter()
                .position(|r| r.action == MenuAction::CycleMsaa)
                .expect("msaa row");
            m.index = i;
            assert_eq!(m.activate(&mut s, false), MenuOutcome::ApplyMsaa);
            assert_eq!(s.msaa, want);
        }
    }

    #[test]
    fn main_menu_starts_a_session_and_pause_menu_ends_it() {
        let mut s = GraphicsSettings::default();
        let mut m = Menu::new();
        assert_eq!(m.screen, MenuScreen::MainRoot, "app opens on the main menu");

        let i = m
            .rows(&s, false)
            .iter()
            .position(|r| r.action == MenuAction::NewGame)
            .expect("new game row");
        m.index = i;
        assert_eq!(m.activate(&mut s, false), MenuOutcome::StartSession);

        // The app resets to Root when the session starts.
        m.reset(MenuScreen::Root);
        let i = m
            .rows(&s, true)
            .iter()
            .position(|r| r.action == MenuAction::ToMainMenu)
            .expect("main menu row");
        m.index = i;
        assert_eq!(m.activate(&mut s, true), MenuOutcome::EndSession);
    }

    #[test]
    fn esc_at_the_main_root_has_nowhere_to_go() {
        let mut m = Menu::new();
        // No session to resume into, so backing out must stay put rather than
        // reporting Resume and dropping the app into a world that isn't loaded.
        assert_eq!(m.back(), MenuOutcome::Stay);
        assert_eq!(m.screen, MenuScreen::MainRoot);
    }

    #[test]
    fn options_is_reachable_from_both_roots() {
        let mut s = GraphicsSettings::default();
        for (root, in_session) in [(MenuScreen::MainRoot, false), (MenuScreen::Root, true)] {
            let mut m = Menu::new();
            m.reset(root);
            let i = m
                .rows(&s, in_session)
                .iter()
                .position(|r| r.action == MenuAction::Enter(MenuScreen::Options))
                .unwrap_or_else(|| panic!("{root:?} has no OPTIONS row"));
            m.index = i;
            assert_eq!(m.activate(&mut s, in_session), MenuOutcome::Stay);
            assert_eq!(m.screen, MenuScreen::Options);
            // And back returns to the root it came from, not a hardcoded one.
            assert_eq!(m.back(), MenuOutcome::Stay);
            assert_eq!(m.screen, root);
        }
    }

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
    fn slice_sphere_radius_is_rotation_invariant() {
        // §11 fits a sphere rather than a box precisely so that turning cannot
        // change the world size a cascade covers. If this regresses, shadows
        // shimmer while you look around.
        let eye = Vec3::new(3.0, 1.5, -2.0);
        let mut reference = None;
        for step in 0..48 {
            let yaw = step as f32 * std::f32::consts::TAU / 48.0;
            for pitch in [-1.2f32, -0.3, 0.0, 0.5, 1.2] {
                let look = Look { yaw, pitch };
                let (fwd, right, up) = look.camera_basis();
                let (_, radius) = slice_sphere(
                    eye,
                    fwd,
                    right,
                    up,
                    60f32.to_radians(),
                    16.0 / 9.0,
                    4.0,
                    20.0,
                );
                match reference {
                    None => reference = Some(radius),
                    Some(r) => assert!(
                        (radius - r).abs() < 1e-3,
                        "radius moved with orientation: {radius} vs {r}"
                    ),
                }
            }
        }
    }

    #[test]
    fn slice_sphere_contains_its_frustum_corners() {
        let eye = Vec3::new(-1.0, 2.0, 5.0);
        let look = Look {
            yaw: 0.7,
            pitch: -0.2,
        };
        let (fwd, right, up) = look.camera_basis();
        let (fov, aspect, near, far) = (60f32.to_radians(), 1.6, 2.0, 30.0);
        let (centre, radius) = slice_sphere(eye, fwd, right, up, fov, aspect, near, far);
        let tan_v = (fov * 0.5).tan();
        let tan_h = tan_v * aspect;
        for d in [near, far] {
            let c = eye + fwd * d;
            for sx in [-1.0f32, 1.0] {
                for sy in [-1.0f32, 1.0] {
                    let corner = c + right * (tan_h * d) * sx + up * (tan_v * d) * sy;
                    assert!(
                        corner.distance(centre) <= radius + 1e-3,
                        "corner outside the fitted sphere"
                    );
                }
            }
        }
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

    #[test]
    fn every_cascade_sees_the_player_position() {
        // The chain splits -> sphere fit -> ortho must actually cover the view.
        // A point just inside each split must project inside that cascade's box.
        let eye = Vec3::new(0.0, GROUND_Y + EYE_HEIGHT, 8.0);
        let look = Look::new();
        let (fwd, right, up) = look.camera_basis();
        let sun = Vec3::new(-0.4, -1.0, -0.3).normalize();
        let splits = cascade_splits(0.1, SHADOW_DISTANCE, SHADOW_LAMBDA);
        let mut near = 0.1f32;
        for (i, far) in splits.iter().copied().enumerate() {
            let (centre, radius) = slice_sphere(
                eye,
                fwd,
                right,
                up,
                60f32.to_radians(),
                16.0 / 9.0,
                near,
                far,
            );
            let (vp, _) = fit_cascade(centre, radius, sun, 2048);
            // A point in the middle of this slice, on the view axis.
            let probe = eye + fwd * (near + (far - near) * 0.5);
            let clip = vp * probe.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            assert!(
                ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 && (0.0..=1.0).contains(&ndc.z),
                "cascade {i} does not cover the middle of its own slice: {ndc:?}"
            );
            near = far;
        }
    }

    // ---- §18 prefabs ----

    /// A world with just enough in it to spawn scene nodes into.
    fn prefab_world() -> World {
        let mut w = World::new();
        w.insert_resource(Physics::new());
        w
    }

    fn node(prefab: Option<&str>, params: serde_json::Value) -> feather_assets::SceneNode {
        feather_assets::SceneNode {
            mesh: Some(0),
            transform: Mat4::IDENTITY,
            prefab: prefab.map(|id| feather_assets::PrefabSpec {
                id: id.to_string(),
                params,
            }),
        }
    }

    fn spawn_one(w: &mut World, spec: Option<&feather_assets::PrefabSpec>, cube: &MeshData) {
        let args = SpawnArgs {
            transform: Mat4::IDENTITY,
            mesh: Some(MeshId(0)),
            material: 0,
            mesh_data: Some(cube),
            spec,
        };
        match spec.and_then(|s| prefab_registry().get(s.id.as_str()).copied()) {
            Some(f) => f(w, &args),
            None => spawn_static_prop(w, &args),
        }
    }

    #[test]
    fn collide_param_controls_whether_a_collider_is_built() {
        let cube = MeshData::cube(1.0);

        // Default (no prefab): geometry collides, exactly as before prefabs.
        let mut w = prefab_world();
        spawn_one(&mut w, None, &cube);
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 1);
        assert_eq!(w.resource::<Physics>().colliders.len(), 1);

        // collide: false skips the trimesh — the §26 per-node opt-out.
        let n = node(Some("prop"), serde_json::json!({ "collide": false }));
        let mut w = prefab_world();
        spawn_one(&mut w, n.prefab.as_ref(), &cube);
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 0);
        assert_eq!(
            w.resource::<Physics>().colliders.len(),
            0,
            "collider was still built"
        );
        // It must still render.
        assert_eq!(w.query::<&Mesh>().iter(&w).count(), 1);
    }

    #[test]
    fn shadow_param_controls_the_no_cast_marker() {
        let cube = MeshData::cube(1.0);
        let n = node(Some("prop"), serde_json::json!({ "shadow": false }));
        let mut w = prefab_world();
        spawn_one(&mut w, n.prefab.as_ref(), &cube);
        assert_eq!(w.query::<&NoShadowCast>().iter(&w).count(), 1);
        // Independent of collision: this one still collides.
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 1);

        // Absent param defaults to casting.
        let n = node(Some("prop"), serde_json::json!({}));
        let mut w = prefab_world();
        spawn_one(&mut w, n.prefab.as_ref(), &cube);
        assert_eq!(w.query::<&NoShadowCast>().iter(&w).count(), 0);
    }

    #[test]
    fn unknown_prefab_falls_back_to_static_geometry() {
        let cube = MeshData::cube(1.0);
        // `trigger` has no implementation yet (§18 lists it; nothing consumes it).
        // It must still appear as geometry rather than vanishing or aborting the
        // load. This deliberately names a prefab that does not exist — when one
        // is added, point this at another unimplemented id rather than deleting
        // the test, since the fallback is what keeps scenes forward-compatible.
        let n = node(Some("trigger"), serde_json::json!({ "radius": 8.0 }));
        assert!(prefab_registry().get("trigger").is_none());
        let mut w = prefab_world();
        spawn_one(&mut w, n.prefab.as_ref(), &cube);
        assert_eq!(w.query::<&Mesh>().iter(&w).count(), 1);
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 1);
    }

    #[test]
    fn player_start_marker_places_the_player() {
        let marker = feather_assets::SceneNode {
            mesh: None,
            transform: Mat4::from_translation(Vec3::new(4.0, -2.0, 7.0)),
            prefab: Some(feather_assets::PrefabSpec {
                id: "player_start".into(),
                params: serde_json::json!({ "yaw": 180.0 }),
            }),
        };
        let (pos, yaw) = player_start(std::slice::from_ref(&marker));
        assert_eq!(pos, Vec3::new(4.0, -2.0, 7.0));
        assert_eq!(yaw, Some(std::f32::consts::PI));

        // No marker: the hardcoded default, so unmarked scenes and the orb demo
        // behave exactly as before.
        let (pos, yaw) = player_start(&[]);
        assert_eq!(pos, Vec3::new(0.0, GROUND_Y, 8.0));
        assert_eq!(yaw, None);

        // A marker without a yaw keeps the default facing.
        let mut m = marker;
        m.prefab.as_mut().unwrap().params = serde_json::json!({});
        let (_, yaw) = player_start(std::slice::from_ref(&m));
        assert_eq!(yaw, None);
    }

    // ---- §12 punctual lights ----

    #[test]
    fn point_light_prefab_reads_params_with_defaults() {
        let cube = MeshData::cube(1.0);
        let spec = feather_assets::PrefabSpec {
            id: "point_light".into(),
            params: serde_json::json!({
                "color": [1.0, 0.5, 0.25], "intensity": 7.5, "radius": 3.0
            }),
        };
        let mut w = prefab_world();
        spawn_one(&mut w, Some(&spec), &cube);
        let l = *w.query::<&PointLight>().single(&w).unwrap();
        assert_eq!(l.color, Vec3::new(1.0, 0.5, 0.25));
        assert_eq!(l.intensity, 7.5);
        assert_eq!(l.radius, 3.0);
        assert_eq!(l.source_radius, PointLight::default().source_radius);

        // source_radius: read when given, clamped to [0, radius].
        for (given, want) in [(0.4, 0.4), (0.0, 0.0), (-1.0, 0.0), (9.0, 3.0)] {
            let sized = feather_assets::PrefabSpec {
                id: "point_light".into(),
                params: serde_json::json!({ "radius": 3.0, "source_radius": given }),
            };
            let mut w = prefab_world();
            spawn_one(&mut w, Some(&sized), &cube);
            let l = *w.query::<&PointLight>().single(&w).unwrap();
            assert_eq!(l.source_radius, want, "source_radius {given}");
        }

        // The geometry switches behave as they do on `prop`, rather than
        // point_light being the one prefab where `shadow` silently does nothing.
        let dark = feather_assets::PrefabSpec {
            id: "point_light".into(),
            params: serde_json::json!({ "shadow": false, "collide": false }),
        };
        let mut w = prefab_world();
        spawn_one(&mut w, Some(&dark), &cube);
        assert_eq!(w.query::<&NoShadowCast>().iter(&w).count(), 1);
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 0);
        assert_eq!(w.query::<&PointLight>().iter(&w).count(), 1);

        // Defaults match `prop`: a lamp is still a physical object.
        let plain = feather_assets::PrefabSpec {
            id: "point_light".into(),
            params: serde_json::json!({}),
        };
        let mut w = prefab_world();
        spawn_one(&mut w, Some(&plain), &cube);
        assert_eq!(w.query::<&NoShadowCast>().iter(&w).count(), 0);
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 1);

        // A bare prefab still lights something, rather than being a black light.
        let bare = feather_assets::PrefabSpec {
            id: "point_light".into(),
            params: serde_json::json!({}),
        };
        let mut w = prefab_world();
        spawn_one(&mut w, Some(&bare), &cube);
        let l = *w.query::<&PointLight>().single(&w).unwrap();
        let d = PointLight::default();
        assert_eq!(
            (l.color, l.intensity, l.radius),
            (d.color, d.intensity, d.radius)
        );
        assert!(l.intensity > 0.0 && l.radius > 0.0);
    }

    #[test]
    fn light_extract_culls_by_sphere_not_point() {
        // Looking down -Z from the origin, matching Look::new().
        let look = Look::new();
        let eye = Vec3::ZERO;
        let vp = look.view_proj(eye, 16.0 / 9.0);
        let frustum = Frustum::from_view_proj(&vp);

        let mut w = World::new();
        // In view.
        w.spawn((
            Transform(Mat4::from_translation(Vec3::new(0.0, 0.0, -20.0))),
            PointLight {
                radius: 2.0,
                ..Default::default()
            },
        ));
        // Behind the camera, nowhere near.
        w.spawn((
            Transform(Mat4::from_translation(Vec3::new(0.0, 0.0, 60.0))),
            PointLight {
                radius: 2.0,
                ..Default::default()
            },
        ));
        // Centre outside the frustum, but its radius reaches in — the case a
        // point test drops, showing up as lights popping at the screen edge.
        w.spawn((
            Transform(Mat4::from_translation(Vec3::new(40.0, 0.0, -20.0))),
            PointLight {
                radius: 28.0,
                ..Default::default()
            },
        ));

        let lights = extract_lights(&mut w, &frustum);
        assert_eq!(
            lights.len(),
            2,
            "expected the in-view and the overlapping one"
        );
        assert!(
            lights.iter().all(|l| l.pos_radius[2] < 0.0),
            "the light behind the camera should have been culled"
        );
    }

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
