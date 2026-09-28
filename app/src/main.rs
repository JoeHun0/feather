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

mod audio;
mod config;

use audio::{Audio, AudioSettings, StepTracker, Surface};
use config::controls::{Action, Controls};

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ExecutorKind;
use feather_assets::MeshData;
use feather_gfx::{GpuTimes, Renderer, FRAMES_IN_FLIGHT, SHADOW_CASCADES};
use feather_platform::winit;
use feather_render::{
    CascadeSetup, ClusterView, Environment, FrameStats, FxaaPass, GpuLight, InstanceData, MeshId,
    MeshRenderer, SkyPass, TonemapPass, UiPass,
};
use glam::{Mat4, Vec3, Vec4};
use rapier3d::control::{CharacterAutostep, CharacterLength, KinematicCharacterController};
use rapier3d::geometry::ContactManifold;
use rapier3d::parry::bounding_volume::BoundingVolume;
use rapier3d::parry::query::{DefaultQueryDispatcher, PersistentQueryDispatcher, ShapeCastOptions};
use rapier3d::parry::shape::{Ball, Shape};
use rapier3d::prelude::{
    BroadPhaseBvh, CCDSolver, Collider, ColliderBuilder, ColliderHandle, ColliderSet,
    ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet, NarrowPhase,
    PhysicsPipeline, Pose, QueryFilter, QueryPipeline, RigidBodyBuilder, RigidBodyHandle,
    RigidBodySet, Vector,
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
/// How far below a grounded player the surface probe looks (§20). The
/// controller holds the capsule ~2 cm off the ground and counts contacts
/// within ~7 cm as grounded, so 10 cm finds whatever it counted.
const GROUND_PROBE: f32 = 0.1;
/// How close a downward-facing surface must be to count as touching the
/// capsule, for the overhang clip in `player_target` (§15), and so how far
/// short of one a grounded player stops. The controller keeps the capsule
/// ~2 cm off everything.
const OVERHANG_REACH: f32 = 0.05;

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

    /// Settings-file form (`config/graphics.toml`).
    fn config_name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    fn from_config_name(name: &str) -> Option<Self> {
        [Self::Off, Self::Low, Self::Medium, Self::High]
            .into_iter()
            .find(|q| q.config_name() == name)
    }
}

/// Windowed or fullscreen (§13). Fullscreen is **borderless** on the current
/// monitor: exclusive fullscreen (a video-mode change) is ignored by winit on
/// Wayland, and borderless switches instantly with nothing to rebuild.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DisplayMode {
    Windowed,
    Fullscreen,
}

impl DisplayMode {
    fn toggled(self) -> Self {
        match self {
            Self::Windowed => Self::Fullscreen,
            Self::Fullscreen => Self::Windowed,
        }
    }

    /// What winit's `set_fullscreen` / `with_fullscreen` take.
    fn fullscreen(self) -> Option<winit::window::Fullscreen> {
        match self {
            Self::Windowed => None,
            Self::Fullscreen => Some(winit::window::Fullscreen::Borderless(None)),
        }
    }

    /// Settings-file form (`config/graphics.toml`), also used in the log.
    fn config_name(self) -> &'static str {
        match self {
            Self::Windowed => "windowed",
            Self::Fullscreen => "fullscreen",
        }
    }

    fn from_config_name(name: &str) -> Option<Self> {
        [Self::Windowed, Self::Fullscreen]
            .into_iter()
            .find(|d| d.config_name() == name)
    }

    /// Menu form: the UI font is A-Z/0-9 only.
    fn menu_label(self) -> &'static str {
        match self {
            Self::Windowed => "WINDOWED",
            Self::Fullscreen => "FULLSCREEN",
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
    /// Windowed or borderless fullscreen; live (menu, F11), saved.
    display: DisplayMode,
    /// Vertical field of view, whole degrees (30-120); live, saved.
    fov_deg: u32,
    /// MSAA sample count for the geometry pass (§13), `--msaa N` at startup.
    /// Startup-only rather than live: the sample count is baked into every
    /// geometry pipeline, so changing it means rebuilding them all — the usual
    /// "applies on restart" trade. `gfx` clamps this to what the device supports.
    msaa: u32,
    /// FXAA post-AA (§13). Unlike MSAA this is live-toggleable (`F2`): it bakes
    /// nothing into pipelines, and the LDR intermediate is always allocated.
    fxaa: bool,
    /// Use baked assets from `BAKE_DIR` when present (§17): BC7 textures and
    /// meshes with LODs. `--no-bake` forces the raw assets, for A/B comparisons.
    bake: bool,
    /// Pick LODs per view (§17). `--no-lod` pins LOD0 while keeping the baked
    /// vertex order, so an A/B separates the ordering win from the LOD win.
    lod: bool,
}

/// Where `feather-bake` writes, and the runtime looks for, baked assets
/// (`tex/` and `mesh/` below it).
const BAKE_DIR: &str = "scratch/bake";

impl GraphicsSettings {
    /// The vertical FOV in radians, as the camera and everything that must
    /// match it use it.
    fn fov_y(&self) -> f32 {
        (self.fov_deg as f32).to_radians()
    }

    /// The next FIELD OF VIEW preset up, wrapping. A value that isn't a
    /// preset (typed into the file) goes to the next preset above it.
    fn step_fov(&mut self) {
        self.fov_deg = FOV_PRESETS
            .into_iter()
            .find(|&p| p > self.fov_deg)
            .unwrap_or(FOV_PRESETS[0]);
    }
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            shadows: ShadowQuality::High,
            display: DisplayMode::Windowed,
            fov_deg: DEFAULT_FOV_DEG,
            msaa: 1,
            fxaa: false,
            bake: true,
            lod: true,
        }
    }
}

/// What `main` loaded from `config/` (defaults and no files under `--bench`).
struct Configs {
    graphics: Option<config::ConfigFile>,
    controls: Controls,
    controls_file: Option<config::ConfigFile>,
    audio: AudioSettings,
    audio_file: Option<config::ConfigFile>,
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
    /// Key bindings (§14): one row per action, Enter to rebind.
    Controls,
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
            Self::Controls => "CONTROLS",
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
    ToggleDisplay,
    NewGame,
    ToMainMenu,
    /// Next SENSITIVITY preset (§14).
    CycleSensitivity,
    ToggleInvertY,
    /// Next FIELD OF VIEW preset.
    CycleFov,
    /// Next step of one SOUND volume.
    CycleVolume(config::audio::Key),
    /// Wait for a key to bind to this action.
    Rebind(Action),
    /// Default key bindings again.
    ResetKeys,
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
    ApplyDisplay,
    /// The FOV changed: nothing to rebuild (the next frame uses it), just save.
    ApplyFov,
    /// A volume changed: set it on the mixer and save it.
    ApplyAudio(config::audio::Key),
    StartSession,
    EndSession,
    /// Controls changed; save the named part of `controls.toml`.
    SaveControls(ControlsSave),
}

/// Which part of `controls.toml` a menu change touched. Bindings are saved
/// together, because a rebind can take a key from another action too.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ControlsSave {
    Sensitivity,
    InvertY,
    Keys,
}

/// The rows of one screen, given the settings they display, whether a world
/// is currently loaded, and which action (if any) is waiting for a key.
///
/// `in_session` is not cosmetic: it decides whether MSAA is changeable. The
/// sample count is baked into the mesh and sky pipelines at creation, so it can
/// only move while no session owns them.
fn screen_rows(
    screen: MenuScreen,
    s: &GraphicsSettings,
    c: &Controls,
    a: &AudioSettings,
    in_session: bool,
    capturing: Option<Action>,
) -> Vec<MenuRow> {
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
            MenuRow::new("CONTROLS", MenuAction::Enter(MenuScreen::Controls)),
            MenuRow::new("SOUND", MenuAction::Enter(MenuScreen::Sound)),
            MenuRow::new("GAMEPLAY", MenuAction::Enter(MenuScreen::Gameplay)),
            MenuRow::new("BACK", MenuAction::Back),
        ],
        MenuScreen::Graphics => vec![
            MenuRow::new(
                format!("DISPLAY  {}", s.display.menu_label()),
                MenuAction::ToggleDisplay,
            ),
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
        // Volumes in percent, saved to audio.toml (§20).
        MenuScreen::Sound => {
            use config::audio::Key;
            let row = |label: &str, k: Key| {
                MenuRow::new(format!("{label}  {}", k.get(a)), MenuAction::CycleVolume(k))
            };
            vec![
                row("MASTER VOLUME", Key::Master),
                row("SFX", Key::Sfx),
                row("AMBIENCE", Key::Ambience),
                MenuRow::new("BACK", MenuAction::Back),
            ]
        }
        // Sensitivity and invert are saved to controls.toml (§14), the FOV
        // (vertical degrees) to graphics.toml. Sensitivity is a percentage:
        // the font has no decimal point.
        MenuScreen::Gameplay => vec![
            MenuRow::new(
                format!("SENSITIVITY  {}", c.sensitivity_percent()),
                MenuAction::CycleSensitivity,
            ),
            MenuRow::new(
                format!("FIELD OF VIEW  {}", s.fov_deg),
                MenuAction::CycleFov,
            ),
            MenuRow::new(
                format!("INVERT Y  {}", if c.invert_y { "ON" } else { "OFF" }),
                MenuAction::ToggleInvertY,
            ),
            MenuRow::new("BACK", MenuAction::Back),
        ],
        // One row per action; Enter waits for the key to bind.
        MenuScreen::Controls => Action::ALL
            .into_iter()
            .map(|a| {
                let keys = if capturing == Some(a) {
                    "PRESS A KEY".to_string()
                } else {
                    c.keys_label(a)
                };
                MenuRow::new(format!("{}  {keys}", a.menu_label()), MenuAction::Rebind(a))
            })
            .chain([
                MenuRow::new("RESET KEYS", MenuAction::ResetKeys),
                MenuRow::new("BACK", MenuAction::Back),
            ])
            .collect(),
    }
}

/// Pause-menu navigation state. Deliberately knows nothing about the renderer
/// or the window: it mutates `GraphicsSettings` / `Controls` and reports a
/// `MenuOutcome`.
struct Menu {
    screen: MenuScreen,
    index: usize,
    /// `(screen, index)` of each ancestor, so BACK restores the row you
    /// descended from instead of snapping to the top.
    stack: Vec<(MenuScreen, usize)>,
    /// The action waiting for a key on the CONTROLS screen (`capture_key`).
    capturing: Option<Action>,
    /// First visible row when a screen is taller than the window
    /// (`menu_layout`). Kept here so it only moves to follow the selection:
    /// hovering a visible row never scrolls the list under the mouse.
    scroll: usize,
}

impl Menu {
    fn new() -> Self {
        Self {
            screen: MenuScreen::MainRoot,
            index: 0,
            stack: Vec::new(),
            capturing: None,
            scroll: 0,
        }
    }

    /// Return to a given top level, discarding history. Called when a session
    /// starts (→ `Root`) or ends (→ `MainRoot`), and on unpause, so the menu
    /// always reopens at a sensible place rather than wherever it was left.
    fn reset(&mut self, root: MenuScreen) {
        self.screen = root;
        self.index = 0;
        self.stack.clear();
        self.capturing = None;
        self.scroll = 0;
    }

    fn rows(
        &self,
        s: &GraphicsSettings,
        c: &Controls,
        a: &AudioSettings,
        in_session: bool,
    ) -> Vec<MenuRow> {
        screen_rows(self.screen, s, c, a, in_session, self.capturing)
    }

    /// Point the selection at `index`. Leaving the row that is waiting for a
    /// key cancels the wait; staying on it doesn't, so mouse jitter over the
    /// row you just clicked can't.
    fn select(&mut self, index: usize) {
        if index != self.index {
            self.capturing = None;
        }
        self.index = index;
    }

    /// Move the selection, wrapping at both ends.
    fn move_by(
        &mut self,
        delta: isize,
        s: &GraphicsSettings,
        c: &Controls,
        a: &AudioSettings,
        in_session: bool,
    ) {
        let n = self.rows(s, c, a, in_session).len() as isize;
        if n > 0 {
            self.select((self.index as isize + delta).rem_euclid(n) as usize);
        }
    }

    /// Point the selection at `index` if it is a real row (used by the mouse).
    fn hover(
        &mut self,
        index: usize,
        s: &GraphicsSettings,
        c: &Controls,
        a: &AudioSettings,
        in_session: bool,
    ) {
        if index < self.rows(s, c, a, in_session).len() {
            self.select(index);
        }
    }

    fn descend(&mut self, screen: MenuScreen) {
        self.stack.push((self.screen, self.index));
        self.screen = screen;
        self.index = 0;
        self.scroll = 0;
    }

    /// A key pressed while an action waits for one (the app routes presses
    /// here first, and consumes them). Escape cancels. The other menu keys
    /// (Up, Down, Enter) and keys the file can't name keep it waiting, so a
    /// held Enter can't bind itself. Any other key becomes the action's only
    /// binding, taken from whatever action had it.
    fn capture_key(&mut self, c: &mut Controls, code: KeyCode) -> MenuOutcome {
        let Some(action) = self.capturing else {
            return MenuOutcome::Stay;
        };
        if code == KeyCode::Escape {
            self.capturing = None;
            return MenuOutcome::Stay;
        }
        if c.rebind(action, code) {
            self.capturing = None;
            return MenuOutcome::SaveControls(ControlsSave::Keys);
        }
        MenuOutcome::Stay
    }

    /// Up one level. At the pause root that means resuming; at the *main* root
    /// there is nothing to resume into, so it stays put. This is what makes Esc
    /// walk back out one screen at a time.
    fn back(&mut self) -> MenuOutcome {
        self.capturing = None;
        self.scroll = 0;
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

    fn activate(
        &mut self,
        s: &mut GraphicsSettings,
        c: &mut Controls,
        a: &mut AudioSettings,
        in_session: bool,
    ) -> MenuOutcome {
        let action = match self.rows(s, c, a, in_session).get(self.index) {
            Some(row) => row.action,
            None => return MenuOutcome::Stay,
        };
        // Activating anything ends a wait for a key; activating the waiting
        // row again restarts it.
        self.capturing = None;
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
            MenuAction::ToggleDisplay => {
                s.display = s.display.toggled();
                MenuOutcome::ApplyDisplay
            }
            MenuAction::CycleFov => {
                s.step_fov();
                MenuOutcome::ApplyFov
            }
            MenuAction::CycleVolume(k) => {
                k.set(a, audio::step_volume(k.get(a)));
                MenuOutcome::ApplyAudio(k)
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
            MenuAction::CycleSensitivity => {
                c.step_sensitivity();
                MenuOutcome::SaveControls(ControlsSave::Sensitivity)
            }
            MenuAction::ToggleInvertY => {
                c.invert_y = !c.invert_y;
                MenuOutcome::SaveControls(ControlsSave::InvertY)
            }
            MenuAction::Rebind(a) => {
                self.capturing = Some(a);
                MenuOutcome::Stay
            }
            MenuAction::ResetKeys => {
                c.reset_keys();
                MenuOutcome::SaveControls(ControlsSave::Keys)
            }
            MenuAction::Inert => MenuOutcome::Stay,
        }
    }
}

/// Font pixel size for the menu at a given framebuffer height. One place, so
/// the layout and the renderer cannot disagree about how big the text is.
fn menu_font_px(h: f32) -> f32 {
    (h / 220.0).max(2.0).floor()
}

/// Where one screen of the menu goes, in **physical** pixels.
///
/// Single source of truth: the redraw handler draws the title, the rows and
/// the highlight bar from it and the mouse hit-tests against it, so the
/// visible target and the clickable target can never drift apart. Row rects
/// are the padded bar rather than the tight text box, which also makes a
/// comfortably larger click target than the glyphs alone.
struct MenuLayout {
    px: f32,
    title_y: f32,
    /// Index of the first visible row.
    first: usize,
    /// `(x, y, w, h)` of each *visible* row, from `first` on.
    rects: Vec<(f32, f32, f32, f32)>,
    /// Rows hidden above / below the visible window.
    more_above: bool,
    more_below: bool,
}

/// Lay out a screen: the title, then as many rows as fit, the two centred
/// together. A screen taller than the window scrolls: the visible window
/// starts at `scroll`, moved only as far as needed to keep row `index` in it.
/// The caller stores `first` back as the new `scroll`, so the list moves when
/// the selection leaves it and not otherwise.
fn menu_layout(w: f32, h: f32, rows: &[MenuRow], index: usize, scroll: usize) -> MenuLayout {
    let px = menu_font_px(h);
    let text_h = UiPass::text_height(px);
    let line = text_h * 2.2;
    let pad = px * 4.0;
    let title_h = UiPass::text_height(px * 1.6);
    // Room for the rows once the title, a line of gap under it and a margin
    // top and bottom are taken out.
    let avail = h - title_h - line - pad * 2.0;
    let cap = ((avail / line).floor() as usize).max(1);
    let n = rows.len();
    let visible = n.min(cap);
    let first = if n <= cap {
        0
    } else {
        let mut first = scroll.min(n - cap);
        if index < first {
            first = index;
        } else if index >= first + cap {
            first = index + 1 - cap;
        }
        first
    };
    // Centre title + rows as one block.
    let block = title_h + line + line * visible as f32;
    let title_y = ((h - block) * 0.5).max(pad);
    let top = title_y + title_h + line;
    let rects = rows[first..first + visible]
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let tw = UiPass::text_width(&row.label, px);
            let x = (w - tw) * 0.5;
            let y = top + line * i as f32;
            (x - pad, y - pad * 0.5, tw + pad * 2.0, text_h + pad)
        })
        .collect();
    MenuLayout {
        px,
        title_y,
        first,
        rects,
        more_above: first > 0,
        more_below: first + visible < n,
    }
}

/// Index of the row under `(cx, cy)`, if any. Physical pixels, matching both
/// `Window::inner_size` and winit's `CursorMoved` position.
fn menu_hit(layout: &MenuLayout, cx: f32, cy: f32) -> Option<usize> {
    layout
        .rects
        .iter()
        .position(|&(x, y, rw, rh)| cx >= x && cx < x + rw && cy >= y && cy < y + rh)
        .map(|i| layout.first + i)
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

    /// A static convex hull of world-space points (§15 collision proxies).
    /// `None` when there is no hull (too few or collinear points).
    fn add_static_hull(&mut self, points: &[Vector]) -> Option<ColliderHandle> {
        ColliderBuilder::convex_hull(points).map(|b| self.colliders.insert(b))
    }

    /// A fixed axis-aligned box collider (static level geometry).
    fn add_static_box(&mut self, center: Vec3, size: Vec3) -> ColliderHandle {
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
fn tag_surface(c: &mut Collider, surface: Surface) {
    c.user_data = surface.index() as u128;
}

fn collider_surface(c: &Collider) -> Surface {
    Surface::from_index(usize::try_from(c.user_data).unwrap_or(usize::MAX))
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
    /// What the feet were last on (§20): refreshed every grounded tick, kept
    /// while airborne, so the steps after a landing use the new ground.
    surface: Surface,
    /// How many times the controller's slide hit something this tick, a
    /// diagnostic (§15). rapier's slide loop gives up after `SLIDE_PASSES`.
    slide_hits: u32,
    /// Wedged under a shallow overhang (§15): a grounded tick's slide gave
    /// up, so the overhang clip takes in shallow undersides too, for as long
    /// as it keeps changing the motion.
    wedged: bool,
    body: RigidBodyHandle,
    collider: ColliderHandle,
    controller: KinematicCharacterController,
}

/// Camera projection. The vertical FOV is a setting (`GraphicsSettings::fov_y`,
/// default `DEFAULT_FOV_DEG`); the projection, the light clusters (§12), the LOD
/// budget (§17) and the cascade fit (§11) all read it from there each frame,
/// since they must describe the same frustum or fragments read the wrong
/// cluster.
const DEFAULT_FOV_DEG: u32 = 60;
/// What GAMEPLAY > FIELD OF VIEW steps through, in vertical degrees.
const FOV_PRESETS: [u32; 5] = [50, 60, 70, 80, 90];
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

    fn view_proj(&self, eye: Vec3, aspect: f32, fov_y: f32) -> Mat4 {
        let mut proj = Mat4::perspective_rh(fov_y, aspect, CAMERA_NEAR, CAMERA_FAR);
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
                .move_shape(FIXED_DT, &queries, shape, &at, to_rapier(desired), |_| {
                    hits += 1
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

    physics.bodies[p.body]
        .set_next_kinematic_translation(to_rapier(p.pos + Player::CENTER + motion));
}

/// How many passes rapier's slide makes before it gives up (§15).
const SLIDE_PASSES: u32 = 20;

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
fn overhangs_near(
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
const OVERHANG_BENDS: usize = 3;

/// The grounded `motion` of the player whose feet are at `feet`, bent round
/// the overhangs on its way (§15): each leg goes as far as the head may go
/// before it's `OVERHANG_REACH` from the next underside ahead, and the rest
/// is then clipped against that underside, now touching, like the ones
/// touching from the start (`wedged`: see `Player`). The legs are summed
/// into one move for the controller, which still does the real collision;
/// the straight line cuts inside a bend round a curved canopy by only
/// millimetres.
fn around_overhangs(queries: &QueryPipeline, feet: Vec3, motion: Vec3, wedged: bool) -> Vec3 {
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
fn clip_horizontal(mut motion: Vec3, walls: &[(Vec3, f32)]) -> (Vec3, Vec3) {
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
    jump: bool, // latched on the jump action's press edge; consumed by the fixed step
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
    samples: Vec<(GpuTimes, usize, FrameStats)>,
    lights: usize,
    /// This frame's triangles and LOD mix (§17).
    stats: FrameStats,
}

impl Bench {
    fn new() -> Self {
        Self {
            frame: 0,
            start_yaw: None,
            samples: Vec::with_capacity(BENCH_SWEEP as usize),
            lights: 0,
            stats: FrameStats::default(),
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
            self.samples.push((raw, self.lights, self.stats));
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
        let col =
            |f: fn(&GpuTimes) -> f32| stats(self.samples.iter().map(|(t, _, _)| f(t)).collect());
        let lights = stats(self.samples.iter().map(|&(_, n, _)| n as f32).collect());
        let mtris = |f: fn(&FrameStats) -> u64| {
            stats(
                self.samples
                    .iter()
                    .map(|(_, _, s)| f(s) as f32 / 1e6)
                    .collect(),
            )
        };
        // Share of instances at each LOD over the whole sweep, as "0:62% 1:20% ...".
        let mix = |f: fn(&FrameStats) -> [u32; 16]| {
            let mut sum = [0u64; 16];
            for (_, _, s) in &self.samples {
                for (t, n) in sum.iter_mut().zip(f(s)) {
                    *t += n as u64;
                }
            }
            let total = sum.iter().sum::<u64>().max(1) as f64;
            sum.iter()
                .enumerate()
                .filter(|(_, &n)| n > 0)
                .map(|(i, &n)| format!("{i}:{:.0}%", n as f64 * 100.0 / total))
                .collect::<Vec<_>>()
                .join(" ")
        };
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
        eprintln!("[bench] Mtris main   {}", mtris(|s| s.main_tris));
        eprintln!("[bench] Mtris shadow {}", mtris(|s| s.shadow_tris));
        eprintln!("[bench] LOD mix main   {}", mix(|s| s.main_lods));
        eprintln!("[bench] LOD mix shadow {}", mix(|s| s.shadow_lods));
        let count = |f: fn(&FrameStats) -> u32| {
            stats(self.samples.iter().map(|(_, _, s)| f(s) as f32).collect())
        };
        eprintln!("[bench] masked main   {}", count(|s| s.main_masked));
        eprintln!("[bench] masked shadow {}", count(|s| s.shadow_masked));
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
    /// The settings file, for saving changes; `None` under `--bench` or when it
    /// could be neither read nor created.
    config: Option<config::ConfigFile>,
    /// Key bindings and mouse look (`config/controls.toml`, §14).
    controls: config::controls::Controls,
    /// `controls.toml`, for saving menu changes; `None` like `config`.
    controls_file: Option<config::ConfigFile>,
    /// Volumes (`config/audio.toml`, §20) and the file to save them to.
    audio_settings: AudioSettings,
    audio_file: Option<config::ConfigFile>,
    /// The mixer; `None` = silent (no device, or `--bench`).
    audio: Option<Audio<kira::DefaultBackend>>,
    /// Turns the player's motion into footsteps / jump / landing sounds.
    steps: StepTracker,
    /// Keys currently down, so an action bound to several keys stays held
    /// until the last of them is released.
    held_keys: HashSet<KeyCode>,
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
    /// Its bake (§17), whose LODs can stand in for an over-budget mesh.
    baked: Option<&'a feather_assets::bake::BakedMesh>,
    spec: Option<&'a feather_assets::PrefabSpec>,
    /// What the mesh's material sounds like underfoot (§20), for its collider.
    surface: Surface,
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
/// Triangle budget above which a prop collides as a convex hull instead of its
/// exact mesh (§15). Measured on a 34k-triangle Poly Haven lantern: standing on
/// its trimesh cost 27 ms per physics tick in debug (213 ms stepping off the
/// rim) and 2.5 ms in release, and the fixed-step catch-up multiplied that
/// into seconds per frame. Props of a few hundred triangles cost nothing
/// measurable, so they keep exact collision.
const TRIMESH_MAX_TRIS: usize = 2048;
/// Largest geometric error, in world units, of a baked LOD used as a prop's
/// collider in place of its over-budget full mesh (§17). Far below what the
/// player can feel (capsule radius 0.35, autostep 0.4), and far closer than
/// the convex hull it replaces, whose error is the depth of every concavity.
const COLLISION_TOLERANCE: f32 = 0.05;

/// The finest baked LOD usable as a collider: within `TRIMESH_MAX_TRIS` and
/// within `COLLISION_TOLERANCE` once its error is scaled to the node
/// (`world_scale`, its largest axis scale). `None` if none is. LOD errors
/// never decrease and triangle counts never grow along the chain, so the first
/// level under the budget is the only candidate: if it's too coarse, every
/// later one is too.
fn collision_lod(lods: &[feather_assets::bake::Lod], world_scale: f32) -> Option<usize> {
    let i = lods
        .iter()
        .position(|l| l.indices.len() / 3 <= TRIMESH_MAX_TRIS)?;
    (lods[i].error * world_scale <= COLLISION_TOLERANCE).then_some(i)
}

/// A transform's largest axis scale: how much it can stretch an error.
fn max_axis_scale(m: &Mat4) -> f32 {
    m.x_axis
        .truncate()
        .length()
        .max(m.y_axis.truncate().length())
        .max(m.z_axis.truncate().length())
}

/// What `build_collider` actually built, for the load-time `[scene]` line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuiltCollider {
    /// The full mesh (or a baked LOD0, which is the same triangles).
    Mesh,
    /// A coarser baked LOD standing in for an over-budget mesh.
    Lod(usize),
    Hull,
    Box,
}

/// Collider counts for a session, logged once after spawning.
#[derive(Resource, Default, Debug)]
struct ColliderStats {
    mesh: usize,
    lod: usize,
    hull: usize,
    boxes: usize,
    /// Per surface (§20), indexed by `Surface::index`.
    surfaces: [usize; Surface::ALL.len()],
}

impl ColliderStats {
    /// "12 concrete, 400 grass": the surfaces that have colliders.
    fn surfaces_line(&self) -> String {
        let parts: Vec<String> = Surface::ALL
            .into_iter()
            .filter(|s| self.surfaces[s.index()] > 0)
            .map(|s| format!("{} {}", self.surfaces[s.index()], s.name()))
            .collect();
        parts.join(", ")
    }
}

/// How a scene node collides (§15), from the `collider` prefab param.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColliderKind {
    /// Exact mesh within `TRIMESH_MAX_TRIS`, a convex hull above it.
    Auto,
    Mesh,
    Hull,
    /// Oriented bounding box: the hull of the local bounds' eight corners.
    Box,
    None,
}

impl ColliderKind {
    /// `collide: false` wins over `collider`, so older scenes keep meaning.
    fn from_args(args: &SpawnArgs, collide: bool) -> Self {
        if !collide {
            return Self::None;
        }
        match args.spec.and_then(|s| s.str("collider")) {
            None | Some("auto") => Self::Auto,
            Some("mesh") => Self::Mesh,
            Some("hull") => Self::Hull,
            Some("box") => Self::Box,
            Some("none") => Self::None,
            Some(other) => {
                eprintln!("[scene] unknown collider \"{other}\"; using auto");
                Self::Auto
            }
        }
    }
}

/// Build one primitive's static collider in world space, and say what it
/// built. `auto` uses the exact mesh within `TRIMESH_MAX_TRIS`; above it, a
/// baked LOD within `COLLISION_TOLERANCE` if there is one (§17), and a convex
/// hull otherwise. Flat geometry still gets a (zero-thickness) hull, which
/// collides fine (tested). Only points with no hull at all, too few or
/// collinear, fall back to the exact mesh.
fn build_collider(
    physics: &mut Physics,
    data: &MeshData,
    baked: Option<&feather_assets::bake::BakedMesh>,
    transform: Mat4,
    kind: ColliderKind,
) -> Option<(ColliderHandle, BuiltCollider)> {
    let world = |p: Vec3| to_rapier(transform.transform_point3(p));
    let trimesh = |physics: &mut Physics, vertices: &[feather_assets::Vertex], indices: &[u32]| {
        let verts: Vec<Vector> = vertices.iter().map(|v| world(Vec3::from(v.pos))).collect();
        let tris: Vec<[u32; 3]> = indices
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]])
            .collect();
        physics.add_static_trimesh(verts, tris)
    };
    let exact = |physics: &mut Physics| {
        trimesh(physics, &data.vertices, &data.indices).map(|h| (h, BuiltCollider::Mesh))
    };
    let kind = match kind {
        ColliderKind::Auto if data.indices.len() / 3 <= TRIMESH_MAX_TRIS => ColliderKind::Mesh,
        ColliderKind::Auto => {
            let lod = baked
                .and_then(|b| collision_lod(&b.lods, max_axis_scale(&transform)).map(|i| (b, i)));
            if let Some((b, i)) = lod {
                let built = if i == 0 {
                    BuiltCollider::Mesh
                } else {
                    BuiltCollider::Lod(i)
                };
                return trimesh(physics, &b.vertices, &b.lods[i].indices).map(|h| (h, built));
            }
            ColliderKind::Hull
        }
        k => k,
    };
    let (points, built): (Vec<Vector>, _) = match kind {
        ColliderKind::None => return None,
        ColliderKind::Mesh | ColliderKind::Auto => return exact(physics),
        ColliderKind::Hull => (
            data.vertices
                .iter()
                .map(|v| world(Vec3::from(v.pos)))
                .collect(),
            BuiltCollider::Hull,
        ),
        ColliderKind::Box => {
            let (lo, hi) = data.bounds();
            (
                (0..8)
                    .map(|i| {
                        let pick = |bit: usize, a: f32, b: f32| if i & bit == 0 { a } else { b };
                        world(Vec3::new(
                            pick(1, lo.x, hi.x),
                            pick(2, lo.y, hi.y),
                            pick(4, lo.z, hi.z),
                        ))
                    })
                    .collect(),
                BuiltCollider::Box,
            )
        }
    };
    match physics.add_static_hull(&points) {
        Some(h) => Some((h, built)),
        None => {
            eprintln!("[scene] degenerate {kind:?} collider; using the exact mesh");
            exact(physics)
        }
    }
}

fn spawn_scene_node(
    world: &mut World,
    args: &SpawnArgs,
    collide: bool,
    no_shadow: bool,
) -> Option<Entity> {
    let mesh = args.mesh?;
    let kind = ColliderKind::from_args(args, collide);
    let collider = args.mesh_data.and_then(|data| {
        build_collider(
            &mut world.resource_mut::<Physics>(),
            data,
            args.baked,
            args.transform,
            kind,
        )
    });
    if let Some((h, _)) = collider {
        tag_surface(
            &mut world.resource_mut::<Physics>().colliders[h],
            args.surface,
        );
    }
    if let (Some((_, built)), Some(mut stats)) =
        (collider, world.get_resource_mut::<ColliderStats>())
    {
        match built {
            BuiltCollider::Mesh => stats.mesh += 1,
            BuiltCollider::Lod(_) => {
                stats.mesh += 1;
                stats.lod += 1;
            }
            BuiltCollider::Hull => stats.hull += 1,
            BuiltCollider::Box => stats.boxes += 1,
        }
        stats.surfaces[args.surface.index()] += 1;
    }
    let collider = collider.map(|(h, _)| h);
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

/// Each mesh's footstep surface (§20), from its material's `surface` tag,
/// plus the tag names that aren't surfaces (each once, for the caller to
/// report). Resolved per mesh rather than per node, so a typo on a tile used
/// 400 times is one warning. Unknown and untagged both mean concrete.
fn mesh_surfaces(meshes: &[MeshData]) -> (Vec<Surface>, Vec<&str>) {
    let mut unknown: Vec<&str> = Vec::new();
    let surfaces = meshes
        .iter()
        .map(|m| {
            let Some(name) = m.material.surface.as_deref() else {
                return Surface::default();
            };
            Surface::from_name(name).unwrap_or_else(|| {
                if !unknown.contains(&name) {
                    unknown.push(name);
                }
                Surface::default()
            })
        })
        .collect();
    (surfaces, unknown)
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

/// The parameters an `environment` marker (§13, §18) may carry.
const ENVIRONMENT_PARAMS: [&str; 17] = [
    "sun_elevation",
    "sun_azimuth",
    "sun_color",
    "sun_intensity",
    "sky_zenith",
    "sky_horizon",
    "sky_ground",
    "sky_sun_color",
    "sky_intensity",
    "sun_glow",
    "sun_disk",
    "fog_density",
    "fog_height",
    "fog_falloff",
    "fog_color",
    "fog_sun",
    "exposure",
];

/// The level's atmosphere (§13): the first `environment` marker's params over
/// `Environment::default()`, which is also what a level without one gets.
/// Like `player_start`, the marker is read before the world is built. Also
/// returns the params it couldn't use, unknown or unreadable, to warn about.
///
/// The sun is placed by `sun_elevation` (degrees above the horizon) and
/// `sun_azimuth` (degrees clockwise from north, -Z, seen from above: 90 is
/// east, +X); give one and the other keeps the default sun's.
fn environment(nodes: &[feather_assets::SceneNode]) -> (Environment, Vec<String>) {
    let mut env = Environment::default();
    let Some(spec) = nodes
        .iter()
        .filter_map(|n| n.prefab.as_ref())
        .find(|s| s.id == "environment")
    else {
        return (env, Vec::new());
    };
    let mut bad: Vec<String> = spec
        .params
        .as_object()
        .into_iter()
        .flat_map(|o| o.keys())
        .filter(|k| !ENVIRONMENT_PARAMS.contains(&k.as_str()))
        .cloned()
        .collect();
    let has = |key: &str| spec.params.get(key).is_some();
    let mut num = |key: &str, field: &mut f32| {
        if has(key) {
            match spec.f32(key) {
                Some(v) => *field = v,
                None => bad.push(key.to_string()),
            }
        }
    };
    let (mut elevation, mut azimuth) = sun_angles(env.sun_dir);
    num("sun_elevation", &mut elevation);
    num("sun_azimuth", &mut azimuth);
    num("sun_intensity", &mut env.sun_intensity);
    num("sky_intensity", &mut env.sky_intensity);
    num("sun_glow", &mut env.sun_glow);
    num("sun_disk", &mut env.sun_disk);
    num("fog_density", &mut env.fog_density);
    num("fog_height", &mut env.fog_height);
    num("fog_falloff", &mut env.fog_falloff);
    num("fog_sun", &mut env.fog_sun);
    num("exposure", &mut env.exposure);
    if has("sun_elevation") || has("sun_azimuth") {
        env.sun_dir = sun_travel(elevation, azimuth);
    }
    let colours = [
        ("sun_color", &mut env.sun_color),
        ("sky_zenith", &mut env.sky_zenith),
        ("sky_horizon", &mut env.sky_horizon),
        ("sky_ground", &mut env.sky_ground),
        ("sky_sun_color", &mut env.sky_sun_color),
    ];
    for (key, field) in colours {
        if has(key) {
            match spec.vec3(key) {
                Some(v) => *field = v,
                None => bad.push(key.to_string()),
            }
        }
    }
    if has("fog_color") {
        match spec.vec3("fog_color") {
            Some(v) => env.fog_color = Some(v),
            None => bad.push("fog_color".to_string()),
        }
    }
    (env, bad)
}

/// The elevation and azimuth (degrees, as `environment` takes them) of the
/// sun whose light travels along `travel`.
fn sun_angles(travel: Vec3) -> (f32, f32) {
    let to_sun = -travel.normalize();
    let elevation = to_sun.y.clamp(-1.0, 1.0).asin().to_degrees();
    let azimuth = to_sun.x.atan2(-to_sun.z).to_degrees();
    (elevation, azimuth)
}

/// The direction sunlight travels from a sun at `elevation` and `azimuth`
/// (degrees, as `environment` takes them).
fn sun_travel(elevation: f32, azimuth: f32) -> Vec3 {
    let (se, ce) = elevation.to_radians().sin_cos();
    let (sa, ca) = azimuth.to_radians().sin_cos();
    -Vec3::new(ce * sa, se, -ce * ca)
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
    /// The level's atmosphere (§13). Baked into `mesh` and `sky`; the app
    /// takes the sun direction and starting exposure from it.
    environment: Environment,
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
    /// Position and radius of every point light attached to geometry: the
    /// visible lamps (§20 gives each a hum).
    fn world_lamps(&mut self) -> Vec<(Vec3, f32)> {
        let mut q = self
            .world
            .query_filtered::<(&Transform, &PointLight), With<Mesh>>();
        q.iter(&self.world)
            .map(|(t, l)| (t.0.transform_point3(Vec3::ZERO), l.radius))
            .collect()
    }

    /// Build a world and the GPU resources that serve it. `scenes` are the CLI
    /// glTF paths; empty means the procedural orb demo, exactly as before.
    fn new(renderer: &Renderer, scenes: &[String], bake: bool, lod: bool) -> Self {
        let t_start = Instant::now();
        let bake_dir = bake.then(|| std::path::Path::new(BAKE_DIR));
        let b = build_world(scenes, bake_dir);

        let t_renderer = Instant::now();
        let (mut mesh, _ids) = MeshRenderer::new(
            renderer,
            &b.meshes,
            &b.baked,
            &b.materials,
            MAX_INSTANCES,
            bake_dir,
            &b.environment,
        );
        mesh.set_lod_enabled(lod);
        // Where a session's load time goes (§17): parse + images + meshes, then
        // spawning + colliders, then the GPU upload.
        let (t_renderer, t_total) = (t_renderer.elapsed(), t_start.elapsed());
        eprintln!(
            "[load] scenes {:.2} s, world {:.2} s, renderer {:.2} s, total {:.2} s",
            b.t_scenes.as_secs_f32(),
            (t_total - b.t_scenes - t_renderer).as_secs_f32(),
            t_renderer.as_secs_f32(),
            t_total.as_secs_f32(),
        );
        let sky = SkyPass::new(renderer, &b.environment);

        Self {
            world: b.world,
            schedule: b.schedule,
            player: b.player,
            mesh,
            sky,
            environment: b.environment,
            fits: b.fits,
            mesh_spheres: b.mesh_spheres,
            accumulator: 0.0,
            noclip: false,
        }
    }
}

/// A session before it touches the GPU: the scenes it loaded, the simulated
/// world with its level, and the tables the renderer is built from.
/// `Session::new` is this plus the upload, so a test that builds this runs
/// the game's own scene → world path, without a device.
struct WorldBuild {
    world: World,
    schedule: Schedule,
    /// The player entity in `world`.
    player: Entity,
    meshes: Vec<MeshData>,
    baked: Vec<Option<feather_assets::bake::BakedMesh>>,
    materials: Vec<feather_assets::Material>,
    fits: Vec<Mat4>,
    mesh_spheres: Vec<(Vec3, f32)>,
    /// The level's atmosphere (§13), from its `environment` marker.
    environment: Environment,
    /// Time spent loading the scenes and bake, for the `[load]` line.
    t_scenes: std::time::Duration,
}

/// The orb demo's obstacle boxes, as (centre, size), resting on the ground.
/// `tools/gen_testscene.py`'s `LEVEL_BOXES` repeats them: the generators
/// still keep their scenes clear of where they were, so regenerating an old
/// scene gives the same level.
fn demo_boxes() -> [(Vec3, Vec3); 5] {
    [
        (Vec3::new(-3.0, 0.75, 2.0), Vec3::new(1.5, 1.5, 1.5)),
        (Vec3::new(3.5, 1.0, -1.0), Vec3::new(2.0, 2.0, 2.0)),
        (Vec3::new(0.0, 0.5, -4.5), Vec3::new(3.0, 1.0, 1.0)),
        (Vec3::new(-5.0, 1.5, -3.0), Vec3::new(1.0, 3.0, 1.0)),
        (Vec3::new(5.0, 0.5, 4.0), Vec3::new(1.0, 1.0, 4.0)),
    ]
    .map(|(offset, size)| (Vec3::new(offset.x, GROUND_Y + offset.y, offset.z), size))
}

/// The CPU half of `Session::new`: load the CLI `scenes` (empty means the orb
/// demo) and the bake from `bake_dir` (`None` ignores it), then build the
/// world, the player, the level and the schedule.
fn build_world(scenes: &[String], bake_dir: Option<&std::path::Path>) -> WorldBuild {
    // Scenes load *first*, because a `player_start` marker (§18) decides
    // where the player goes and the player is built below.
    let t_start = Instant::now();
    let (meshes, scene_nodes) = load_scenes(scenes);
    // The bake (§17), read once: the renderer draws its LODs and the
    // colliders built below can use them too.
    let baked = bake_dir.map_or_else(Vec::new, |d| {
        feather_assets::bake::load_baked_meshes(d, &meshes)
    });
    let t_scenes = t_start.elapsed();
    let (start_pos, start_yaw) = player_start(&scene_nodes);
    let (environment, bad_params) = environment(&scene_nodes);
    for key in bad_params {
        eprintln!("[scene] environment: can't use param {key:?}; ignored");
    }
    let (surfaces, unknown_surfaces) = mesh_surfaces(&meshes);
    for name in unknown_surfaces {
        eprintln!(
            "[scene] unknown surface {name:?}; using {}",
            Surface::default().name()
        );
    }

    let mut world = World::new();
    world.insert_resource(FrameCount::default());
    world.insert_resource(ColliderStats::default());
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
    let mut materials: Vec<feather_assets::Material> = (0..PALETTE).map(palette_material).collect();
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
    // The demo's obstacle boxes, like its orbs, only when no scene was given:
    // a loaded level is the level, and five brown boxes in its middle aren't
    // part of it.
    if scenes.is_empty() {
        for (center, size) in demo_boxes() {
            spawn_static(&mut world, center, size, MESH_LEVEL_CUBE, box_mat);
        }
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
        // `player_start` and `environment` were consumed before the world
        // was built.
        if node
            .prefab
            .as_ref()
            .is_some_and(|s| s.id == "player_start" || s.id == "environment")
        {
            continue;
        }
        let mesh_idx = node.mesh;
        let args = SpawnArgs {
            transform: node.transform,
            mesh: mesh_idx.map(|i| MeshId(i as u32)),
            material: mesh_idx.map_or(0, |i| PALETTE + i as u32),
            mesh_data: mesh_idx.map(|i| &meshes[i]),
            baked: mesh_idx.and_then(|i| baked.get(i)).and_then(Option::as_ref),
            spec: node.prefab.as_ref(),
            surface: mesh_idx.map_or(Surface::default(), |i| surfaces[i]),
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
        let st = world.resource::<ColliderStats>();
        eprintln!(
            "[scene] colliders: {} mesh ({} from LODs), {} hull, {} box",
            st.mesh, st.lod, st.hull, st.boxes
        );
        eprintln!("[scene] surfaces: {}", st.surfaces_line());
    }

    // One step so the broad-phase BVH the character controller shape-casts
    // against contains the level before the first fixed tick.
    world.resource_mut::<Physics>().step();

    WorldBuild {
        world,
        schedule,
        player,
        meshes,
        baked,
        materials,
        fits,
        mesh_spheres,
        environment,
        t_scenes,
    }
}

impl App {
    /// Starts with **no session**: the app opens on the main menu and only
    /// builds a world when NEW GAME is chosen.
    fn new(
        scenes: Vec<String>,
        settings: GraphicsSettings,
        configs: Configs,
        audio: Option<Audio<kira::DefaultBackend>>,
        bench: bool,
    ) -> Self {
        let Configs {
            graphics: config,
            controls,
            controls_file,
            audio: audio_settings,
            audio_file,
        } = configs;
        let light = Environment::default().sun_dir;
        Self {
            session: None,
            tonemap: None,
            fxaa: None,
            ui: None,
            renderer: None,
            window: None,
            input: Input::default(),
            settings,
            config,
            controls,
            controls_file,
            audio_settings,
            audio_file,
            audio,
            steps: StepTracker::default(),
            held_keys: HashSet::new(),
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
        self.persist(config::graphics::Key::Shadows);
    }

    /// Put the window in `settings.display`. The compositor answers with a
    /// `Resized` event, which recreates the swapchain like any other resize,
    /// so nothing else needs to know.
    fn apply_display(&mut self) {
        if let Some(window) = self.window.as_ref() {
            window.set_fullscreen(self.settings.display.fullscreen());
        }
        eprintln!("[quality] display: {}", self.settings.display.config_name());
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
            MenuOutcome::ApplyShadows => {
                self.apply_shadow_quality();
                self.persist(config::graphics::Key::Shadows);
            }
            MenuOutcome::ApplyFxaa => {
                self.apply_fxaa();
                self.persist(config::graphics::Key::Fxaa);
            }
            MenuOutcome::ApplyMsaa => {
                self.apply_msaa();
                self.persist(config::graphics::Key::Msaa);
            }
            MenuOutcome::ApplyDisplay => {
                self.apply_display();
                self.persist(config::graphics::Key::Display);
            }
            MenuOutcome::ApplyFov => self.persist(config::graphics::Key::Fov),
            MenuOutcome::ApplyAudio(key) => {
                if let Some(a) = self.audio.as_mut() {
                    a.set_volumes(&self.audio_settings);
                }
                if let Some(file) = self.audio_file.as_mut() {
                    if let Err(e) = config::audio::save(file, key, &self.audio_settings) {
                        eprintln!("[config] couldn't save {}: {e}", file.path().display());
                    }
                }
            }
            MenuOutcome::StartSession => self.start_session(),
            MenuOutcome::EndSession => self.end_session(),
            MenuOutcome::SaveControls(part) => {
                // New bindings can change what the keys already down mean.
                if part == ControlsSave::Keys {
                    self.sync_held_actions();
                }
                self.persist_controls(part);
            }
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
        let mut session = Session::new(
            renderer,
            &self.scenes,
            self.settings.bake,
            self.settings.lod,
        );
        // Every visible lamp hums (§20): point lights on geometry. Bare light
        // markers stay silent.
        if let Some(a) = self.audio.as_mut() {
            for (pos, radius) in session.world_lamps() {
                a.add_lamp(pos, radius);
            }
            eprintln!("[audio] {} lamps humming", a.lamp_count());
        }
        self.steps = StepTracker::default();
        // The level's sun and starting exposure (§13); its sky and fog are
        // already baked into the session's pipelines.
        let env = session.environment;
        self.light_dir = env.sun_dir.extend(0.0);
        self.exposure = env.exposure;
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
        if let Some(a) = self.audio.as_mut() {
            a.clear_lamps();
        }
        self.paused = false;
        self.menu.reset(MenuScreen::MainRoot);
        self.set_cursor_captured(false);
    }

    /// A key that isn't one of the fixed menu keys: look it up in the bindings,
    /// refresh the held movement state and run whatever it fired.
    fn bound_key(&mut self, code: KeyCode, pressed: bool, repeat: bool) {
        let fired = self
            .controls
            .key(&mut self.held_keys, code, pressed, repeat);
        self.sync_held_actions();
        for action in fired {
            match action {
                // Latched for the fixed step, which consumes it (§14).
                Action::Jump => self.input.jump = true,
                // Free flight, for inspecting the scene.
                Action::Noclip => {
                    if let Some(s) = self.session.as_mut() {
                        s.noclip = !s.noclip;
                    }
                }
                // Exposure control (showcases the HDR/tonemap pipeline).
                Action::ExposureDown => self.exposure = (self.exposure * 0.8).max(0.05),
                Action::ExposureUp => self.exposure = (self.exposure * 1.25).min(16.0),
                Action::CycleShadows => self.cycle_shadow_quality(),
                Action::ToggleFxaa => {
                    self.settings.fxaa = !self.settings.fxaa;
                    self.apply_fxaa();
                    self.persist(config::graphics::Key::Fxaa);
                }
                Action::ToggleFullscreen => {
                    self.settings.display = self.settings.display.toggled();
                    self.apply_display();
                    self.persist(config::graphics::Key::Display);
                }
                Action::Forward | Action::Back | Action::Left | Action::Right | Action::Down => {}
            }
        }
    }

    /// Movement flags from the keys currently down.
    fn sync_held_actions(&mut self) {
        let (c, held) = (&self.controls, &self.held_keys);
        self.input.forward = c.held(held, Action::Forward);
        self.input.back = c.held(held, Action::Back);
        self.input.left = c.held(held, Action::Left);
        self.input.right = c.held(held, Action::Right);
        self.input.up = c.held(held, Action::Jump);
        self.input.down = c.held(held, Action::Down);
    }

    /// Save `key`'s current value to the settings file. Called only where the
    /// player changed it, so CLI overrides are never written back. A failed
    /// write is logged, never fatal.
    /// Save the part of `controls.toml` a menu change touched, in place. Like
    /// `persist`, a failed write is logged, never fatal, and there's no file
    /// to write under `--bench`.
    fn persist_controls(&mut self, part: ControlsSave) {
        let Some(file) = self.controls_file.as_mut() else {
            return;
        };
        let c = &self.controls;
        let values: Vec<(&str, String)> = match part {
            ControlsSave::Sensitivity => vec![("sensitivity", format!("{:?}", c.sensitivity))],
            ControlsSave::InvertY => vec![("invert_y", c.invert_y.to_string())],
            ControlsSave::Keys => Action::ALL
                .into_iter()
                .map(|a| (a.name(), c.keys_literal(a)))
                .collect(),
        };
        for (key, literal) in values {
            if let Err(e) = file.save_value(key, &literal) {
                eprintln!("[config] couldn't save {}: {e}", file.path().display());
                return;
            }
        }
    }

    fn persist(&mut self, key: config::graphics::Key) {
        if let Some(c) = self.config.as_mut() {
            if let Err(e) = config::graphics::save(c, key, &self.settings) {
                eprintln!("[config] couldn't save {}: {e}", c.path().display());
            }
        }
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
        } else {
            // Open straight into the saved mode rather than flashing a
            // window first. (`--bench` has no config, so it stays windowed.)
            attrs = attrs.with_fullscreen(self.settings.display.fullscreen());
        }
        let window = event_loop.create_window(attrs).expect("create window");
        // Opens on the main menu, so the cursor starts free rather than grabbed.
        window.set_cursor_visible(true);

        let size = window.inner_size();
        let mut renderer = Renderer::new(&window, size.width, size.height, self.settings.msaa)
            .expect("create renderer");
        // The renderer starts at its built-in defaults; the settings may come
        // from the config file. Nothing is in flight yet, but set_shadow_dim's
        // contract asks for an idle device.
        renderer.set_fxaa(self.settings.fxaa);
        renderer.wait_idle();
        renderer.set_shadow_dim(self.settings.shadows.dim());

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
                    // An action on the CONTROLS screen is waiting for a key:
                    // the press goes there and nowhere else, so binding V to
                    // JUMP doesn't also toggle noclip. Repeats are dropped.
                    // Releases still reach the bindings below, harmlessly.
                    if pressed && self.menu_active() && self.menu.capturing.is_some() {
                        if !event.repeat {
                            let outcome = self.menu.capture_key(&mut self.controls, code);
                            self.handle_menu_outcome(outcome, event_loop);
                        }
                        return;
                    }
                    match code {
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
                            self.menu.move_by(
                                -1,
                                &self.settings,
                                &self.controls,
                                &self.audio_settings,
                                in_session,
                            );
                        }
                        KeyCode::ArrowDown if pressed && self.menu_active() => {
                            let in_session = self.in_session();
                            self.menu.move_by(
                                1,
                                &self.settings,
                                &self.controls,
                                &self.audio_settings,
                                in_session,
                            );
                        }
                        KeyCode::Enter if pressed && self.menu_active() => {
                            let in_session = self.in_session();
                            let outcome = self.menu.activate(
                                &mut self.settings,
                                &mut self.controls,
                                &mut self.audio_settings,
                                in_session,
                            );
                            self.handle_menu_outcome(outcome, event_loop);
                        }
                        // Everything else goes through the bindings (§14).
                        _ => self.bound_key(code, pressed, event.repeat),
                    }
                }
            }
            // Keys released while unfocused never report a release, so without
            // this alt-tabbing away while holding W leaves you walking.
            WindowEvent::Focused(false) => {
                self.held_keys.clear();
                self.sync_held_actions();
                self.input.jump = false;
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
                        let rows = self.menu.rows(
                            &self.settings,
                            &self.controls,
                            &self.audio_settings,
                            in_session,
                        );
                        let layout = menu_layout(
                            size.width as f32,
                            size.height as f32,
                            &rows,
                            self.menu.index,
                            self.menu.scroll,
                        );
                        if let Some(i) = menu_hit(&layout, position.x as f32, position.y as f32) {
                            self.menu.hover(
                                i,
                                &self.settings,
                                &self.controls,
                                &self.audio_settings,
                                in_session,
                            );
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
                        let rows = self.menu.rows(
                            &self.settings,
                            &self.controls,
                            &self.audio_settings,
                            in_session,
                        );
                        let layout = menu_layout(
                            size.width as f32,
                            size.height as f32,
                            &rows,
                            self.menu.index,
                            self.menu.scroll,
                        );
                        if let Some(i) = menu_hit(&layout, cx, cy) {
                            self.menu.hover(
                                i,
                                &self.settings,
                                &self.controls,
                                &self.audio_settings,
                                in_session,
                            );
                            let outcome = self.menu.activate(
                                &mut self.settings,
                                &mut self.controls,
                                &mut self.audio_settings,
                                in_session,
                            );
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
                    let sens = if self.paused {
                        0.0
                    } else {
                        self.controls.look_scale()
                    };
                    let pitch_sign = if self.controls.invert_y { -1.0 } else { 1.0 };
                    let look = {
                        let mut look = s.world.get_mut::<Look>(s.player).expect("player has Look");
                        look.yaw += self.input.mouse_dx * sens;
                        look.pitch = (look.pitch - pitch_sign * self.input.mouse_dy * sens)
                            .clamp(-1.54, 1.54);
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
                    // One FOV for the frame: projection, clusters, LOD budget
                    // and cascades all read this.
                    let fov_y = self.settings.fov_y();
                    // Audio on the render clock (§20): the listener is the
                    // camera, and the player's motion makes the SFX. Nothing
                    // new plays while paused.
                    if let Some(a) = self.audio.as_mut() {
                        let (fwd, _, up) = look.camera_basis();
                        a.set_listener(eye, fwd, up);
                        if !self.paused {
                            let p = s.world.get::<Player>(s.player).expect("player body");
                            let events =
                                self.steps
                                    .update(p.pos, p.vel, p.on_ground, s.noclip, p.surface);
                            for e in events {
                                a.play(e);
                            }
                        }
                    }
                    let view_proj = look.view_proj(eye, aspect, fov_y);
                    let cluster_view = ClusterView {
                        view: look.view(eye),
                        fov_y,
                        aspect,
                        near: CAMERA_NEAR,
                        far: CAMERA_FAR,
                        viewport_height: size.height.max(1) as f32,
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
                        let (centre, radius) =
                            slice_sphere(eye, fwd, right_v, up_v, fov_y, aspect, near, far);
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
                    // Casters ignore the cascade's near plane: up-light geometry is
                    // pancaked onto it (§11) rather than dropped.
                    let light_frusta: [Frustum; SHADOW_CASCADES] = std::array::from_fn(|i| {
                        Frustum::from_view_proj(&cascades[i].view_proj).without_near()
                    });
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
                        b.stats = s.mesh.frame_stats();
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

                        // Rows, title and rects come from the same layout the
                        // mouse hit-tests against, so highlight and click target
                        // match. Its `first` becomes the menu's scroll position.
                        let rows = self.menu.rows(
                            &self.settings,
                            &self.controls,
                            &self.audio_settings,
                            self.session.is_some(),
                        );
                        let layout = menu_layout(w, h, &rows, self.menu.index, self.menu.scroll);
                        self.menu.scroll = layout.first;
                        let px = layout.px;
                        let title_px = px * 1.6;
                        // Title names the current screen, so a submenu is
                        // self-identifying.
                        let title = self.menu.screen.title();
                        ui.text(
                            (w - UiPass::text_width(title, title_px)) * 0.5,
                            layout.title_y,
                            title_px,
                            [0.9, 0.9, 0.9, 1.0],
                            title,
                        );

                        let pad = px * 4.0;
                        // Short dim bars where rows are scrolled out of view
                        // (the font has no arrow glyphs).
                        let marker = [0.55, 0.55, 0.55, 0.9];
                        if let (true, Some(&(_, ry, _, _))) =
                            (layout.more_above, layout.rects.first())
                        {
                            ui.rect(w * 0.5 - px * 8.0, ry - px * 2.0, px * 16.0, px, marker);
                        }
                        if let (true, Some(&(_, ry, _, rh))) =
                            (layout.more_below, layout.rects.last())
                        {
                            ui.rect(w * 0.5 - px * 8.0, ry + rh + px, px * 16.0, px, marker);
                        }
                        for (vi, &(rx, ry, rw, rh)) in layout.rects.iter().enumerate() {
                            let i = layout.first + vi;
                            let row = &rows[i];
                            let selected = i == self.menu.index;
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
        ..Default::default()
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

    /// The same frustum with its near plane dropped, for culling **shadow
    /// casters** (§11 pancaking). Geometry nearer the sun than a cascade's near
    /// plane still shadows the cascade: the shadow pipeline's depth clamp
    /// flattens it onto depth 0 instead of clipping it. An ortho's side planes
    /// are parallel to the light, so anything outside them could never shadow
    /// the box and stays culled; only the near plane is at fault.
    fn without_near(mut self) -> Self {
        self.planes[4] = Vec4::new(0.0, 0.0, 0.0, 1.0); // every point passes
        self
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
        ..Default::default()
    }
}

fn main() {
    // CLI paths are glTF *scenes* to load and walk around
    // (`cargo run -- level.glb`); each node becomes its own entity. With no args,
    // the procedural drifting-orb demo runs instead. `--msaa N` (1/2/4/8) picks
    // the geometry-pass sample count, clamped to device support. `--bench` runs
    // the scripted timing sweep (see `Bench`) instead of the menu.
    //
    // Settings: defaults < `config/graphics.toml` < CLI flags. `config/controls.toml`
    // holds the bindings. `--bench` skips both (neither reads nor creates them),
    // so timings never depend on someone's personal settings.
    let mut settings = GraphicsSettings::default();
    let bench_run = std::env::args().skip(1).any(|a| a == "--bench");
    let configs = if bench_run {
        eprintln!("[config] ignored (--bench)");
        Configs {
            graphics: None,
            controls: Controls::default(),
            controls_file: None,
            audio: AudioSettings::default(),
            audio_file: None,
        }
    } else {
        let graphics = config::graphics::load(&mut settings);
        let (controls, controls_file) = config::controls::load();
        let mut audio = AudioSettings::default();
        let audio_file = config::audio::load(&mut audio);
        Configs {
            graphics,
            controls,
            controls_file,
            audio,
            audio_file,
        }
    };
    // Audio never stops a run: no device means silence. `--bench` is silent
    // too, so timings don't include an audio thread.
    let audio = if bench_run {
        None
    } else {
        // Recorded SFX where fetched, synthesised otherwise (§20).
        let sounds = audio::SoundSet::load(std::path::Path::new(audio::SOUNDS_DIR));
        for note in &sounds.notes {
            eprintln!("[audio] {note}");
        }
        let (recorded, synthesised) = (sounds.recorded, sounds.synthesised);
        match Audio::new(
            kira::AudioManagerSettings::default(),
            &configs.audio,
            sounds,
        ) {
            Ok(a) => {
                eprintln!(
                    "[audio] started on the default output device; sounds: {recorded} recorded, {synthesised} synthesised"
                );
                Some(a)
            }
            Err(e) => {
                eprintln!("[audio] unavailable: {e}; running silent");
                None
            }
        }
    };
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
            "--no-bake" => settings.bake = false,
            "--no-lod" => settings.lod = false,
            _ => scenes.push(a),
        }
    }

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    eprintln!(
        "[config] effective: display {}, fov {}, shadows {}, msaa {}, fxaa {}",
        settings.display.config_name(),
        settings.fov_deg,
        settings.shadows.config_name(),
        settings.msaa,
        settings.fxaa
    );
    let mut app = App::new(scenes, settings, configs, audio, bench);
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

    // ---- Pause menu: layout, hit-testing and navigation (§19) ----
    //
    // `Menu` deliberately depends on neither the renderer nor the event loop,
    // and the layout is a pure function of the framebuffer size, so all of this
    // runs without a GPU or a window.

    /// A few sizes worth covering: the default window, a typical one, a wide
    /// one, and one small enough that `menu_font_px` clamps to its 2.0 floor.
    const SIZES: [(f32, f32); 4] = [
        (640.0, 480.0),
        (1280.0, 720.0),
        (2560.0, 1440.0),
        (320.0, 200.0),
    ];

    const SCREENS: [MenuScreen; 7] = [
        MenuScreen::MainRoot,
        MenuScreen::Root,
        MenuScreen::Options,
        MenuScreen::Graphics,
        MenuScreen::Sound,
        MenuScreen::Gameplay,
        MenuScreen::Controls,
    ];

    fn rows_of(screen: MenuScreen) -> Vec<MenuRow> {
        let s = GraphicsSettings::default();
        screen_rows(
            screen,
            &s,
            &Controls::default(),
            &AudioSettings::default(),
            true,
            None,
        )
    }

    /// Every screen, every size, every selected row: the layout keeps what it
    /// shows on screen, ordered, disjoint and centred, and the selected row is
    /// always among the visible ones. CONTROLS (13 rows) has to scroll at the
    /// small sizes; that is what this pins down.
    #[test]
    fn menu_layout_keeps_rows_onscreen_and_the_selection_visible() {
        for screen in SCREENS {
            let rows = rows_of(screen);
            assert!(!rows.is_empty(), "{screen:?} has no rows");
            for (w, h) in SIZES {
                for index in 0..rows.len() {
                    let l = menu_layout(w, h, &rows, index, 0);
                    let visible = l.first..l.first + l.rects.len();
                    assert!(
                        visible.contains(&index),
                        "{screen:?} row {index} hidden at {w}x{h}"
                    );
                    assert_eq!(l.more_above, l.first > 0);
                    assert_eq!(l.more_below, visible.end < rows.len());
                    assert!(l.title_y >= 0.0, "{screen:?} title off screen at {w}x{h}");
                    for pair in l.rects.windows(2) {
                        let (_, y0, _, h0) = pair[0];
                        let (_, y1, _, _) = pair[1];
                        assert!(y1 >= y0 + h0, "{screen:?} rows overlap at {w}x{h}");
                    }
                    let title_bottom = l.title_y + UiPass::text_height(l.px * 1.6);
                    for &(x, y, rw, rh) in &l.rects {
                        assert!(
                            y >= title_bottom,
                            "{screen:?} row under the title at {w}x{h}"
                        );
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
    }

    #[test]
    fn a_tall_screen_scrolls_only_as_far_as_the_selection_needs() {
        let rows = rows_of(MenuScreen::Controls);
        let (w, h) = (320.0, 200.0);
        let cap = menu_layout(w, h, &rows, 0, 0).rects.len();
        assert!(cap < rows.len(), "CONTROLS should scroll at {w}x{h}");
        // Stepping down scrolls one row at a time once past the window...
        let mut scroll = 0;
        for index in 0..rows.len() {
            let l = menu_layout(w, h, &rows, index, scroll);
            assert_eq!(l.first, index.saturating_sub(cap - 1), "row {index}");
            scroll = l.first;
        }
        // ...and selecting any visible row (the mouse) doesn't move it.
        for index in scroll..scroll + cap {
            assert_eq!(menu_layout(w, h, &rows, index, scroll).first, scroll);
        }
        // A stale scroll past the end is clamped.
        let l = menu_layout(w, h, &rows, rows.len() - 1, 999);
        assert_eq!(l.first + l.rects.len(), rows.len());
    }

    #[test]
    fn menu_rect_centres_hit_their_own_row() {
        for screen in SCREENS {
            let rows = rows_of(screen);
            for (w, h) in SIZES {
                // Scrolled to the bottom, so hits must add `first` back.
                let l = menu_layout(w, h, &rows, rows.len() - 1, 0);
                for (vi, &(x, y, rw, rh)) in l.rects.iter().enumerate() {
                    let (cx, cy) = (x + rw * 0.5, y + rh * 0.5);
                    assert_eq!(
                        menu_hit(&l, cx, cy),
                        Some(l.first + vi),
                        "{screen:?} row {vi} at {w}x{h} did not hit itself"
                    );
                }
            }
        }
    }

    #[test]
    fn menu_misses_gaps_and_backdrop() {
        let rows = rows_of(MenuScreen::Root);
        let (w, h) = (1280.0, 720.0);
        let l = menu_layout(w, h, &rows, 0, 0);
        let r = &l.rects;
        let gap_y = (r[0].1 + r[0].3 + r[1].1) * 0.5;
        assert_eq!(menu_hit(&l, w * 0.5, gap_y), None, "gap hit");
        assert_eq!(menu_hit(&l, w * 0.5, 0.0), None, "top hit");
        assert_eq!(menu_hit(&l, w * 0.5, h - 1.0), None, "bottom hit");
        assert_eq!(menu_hit(&l, 0.0, r[0].1 + 1.0), None, "left hit");
        assert_eq!(menu_hit(&l, w - 1.0, r[0].1 + 1.0), None, "right hit");
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
            .rows(s, &Controls::default(), &AudioSettings::default(), true)
            .iter()
            .position(|r| r.action == want)
            .unwrap_or_else(|| panic!("no {want:?} row on {:?}", m.screen));
        m.index = i;
        m.activate(
            s,
            &mut Controls::default(),
            &mut AudioSettings::default(),
            true,
        )
    }

    /// Like `activate`, with the controls the GAMEPLAY / CONTROLS rows change.
    fn activate_c(
        m: &mut Menu,
        s: &mut GraphicsSettings,
        c: &mut Controls,
        want: MenuAction,
    ) -> MenuOutcome {
        let i = m
            .rows(s, c, &AudioSettings::default(), true)
            .iter()
            .position(|r| r.action == want)
            .unwrap_or_else(|| panic!("no {want:?} row on {:?}", m.screen));
        m.select(i);
        m.activate(s, c, &mut AudioSettings::default(), true)
    }

    fn label_of(m: &Menu, s: &GraphicsSettings, c: &Controls, want: MenuAction) -> String {
        m.rows(s, c, &AudioSettings::default(), true)
            .into_iter()
            .find(|r| r.action == want)
            .map(|r| r.label)
            .unwrap_or_else(|| panic!("no {want:?} row"))
    }

    /// OPTIONS > CONTROLS, as a player gets there from the pause menu.
    fn controls_menu() -> (Menu, GraphicsSettings, Controls) {
        let (mut s, mut c) = (GraphicsSettings::default(), Controls::default());
        let mut m = in_game_menu();
        activate_c(
            &mut m,
            &mut s,
            &mut c,
            MenuAction::Enter(MenuScreen::Options),
        );
        activate_c(
            &mut m,
            &mut s,
            &mut c,
            MenuAction::Enter(MenuScreen::Controls),
        );
        (m, s, c)
    }

    #[test]
    fn display_row_toggles_windowed_and_fullscreen() {
        let (mut s, mut c) = (GraphicsSettings::default(), Controls::default());
        let mut m = in_game_menu();
        activate_c(
            &mut m,
            &mut s,
            &mut c,
            MenuAction::Enter(MenuScreen::Options),
        );
        activate_c(
            &mut m,
            &mut s,
            &mut c,
            MenuAction::Enter(MenuScreen::Graphics),
        );
        let label = |m: &Menu, s: &GraphicsSettings| {
            label_of(m, s, &Controls::default(), MenuAction::ToggleDisplay)
        };
        assert_eq!(label(&m, &s), "DISPLAY  WINDOWED");
        assert_eq!(
            activate_c(&mut m, &mut s, &mut c, MenuAction::ToggleDisplay),
            MenuOutcome::ApplyDisplay
        );
        assert_eq!(s.display, DisplayMode::Fullscreen);
        assert!(s.display.fullscreen().is_some());
        assert_eq!(label(&m, &s), "DISPLAY  FULLSCREEN");
        activate_c(&mut m, &mut s, &mut c, MenuAction::ToggleDisplay);
        assert_eq!(s.display, DisplayMode::Windowed);
        assert!(s.display.fullscreen().is_none());
    }

    #[test]
    fn sound_rows_step_volumes_and_save() {
        use config::audio::Key;
        let (mut s, mut c, mut a) = (
            GraphicsSettings::default(),
            Controls::default(),
            AudioSettings::default(),
        );
        let mut m = in_game_menu();
        let mut go = |m: &mut Menu, a: &mut AudioSettings, want: MenuAction| {
            let i = m
                .rows(&s, &c, a, true)
                .iter()
                .position(|r| r.action == want)
                .unwrap_or_else(|| panic!("no {want:?} row"));
            m.select(i);
            m.activate(&mut s, &mut c, a, true)
        };
        go(&mut m, &mut a, MenuAction::Enter(MenuScreen::Options));
        go(&mut m, &mut a, MenuAction::Enter(MenuScreen::Sound));
        let labels: Vec<String> = m
            .rows(&GraphicsSettings::default(), &Controls::default(), &a, true)
            .into_iter()
            .map(|r| r.label)
            .collect();
        assert_eq!(
            labels,
            ["MASTER VOLUME  80", "SFX  100", "AMBIENCE  100", "BACK"]
        );
        // Master starts between steps (80): up to 100, then wraps to 0.
        let mut seen = vec![];
        for _ in 0..3 {
            assert_eq!(
                go(&mut m, &mut a, MenuAction::CycleVolume(Key::Master)),
                MenuOutcome::ApplyAudio(Key::Master)
            );
            seen.push(a.master);
        }
        assert_eq!(seen, [100, 0, 25]);
        go(&mut m, &mut a, MenuAction::CycleVolume(Key::Sfx));
        assert_eq!((a.sfx, a.ambience), (0, 100), "only SFX moved");
    }

    #[test]
    fn gameplay_rows_change_and_save_the_controls() {
        let (mut s, mut c) = (GraphicsSettings::default(), Controls::default());
        let mut m = in_game_menu();
        activate_c(
            &mut m,
            &mut s,
            &mut c,
            MenuAction::Enter(MenuScreen::Options),
        );
        activate_c(
            &mut m,
            &mut s,
            &mut c,
            MenuAction::Enter(MenuScreen::Gameplay),
        );
        assert_eq!(
            label_of(&m, &s, &c, MenuAction::CycleSensitivity),
            "SENSITIVITY  100"
        );
        assert_eq!(
            activate_c(&mut m, &mut s, &mut c, MenuAction::CycleSensitivity),
            MenuOutcome::SaveControls(ControlsSave::Sensitivity)
        );
        assert_eq!(c.sensitivity, 1.25);
        assert_eq!(
            label_of(&m, &s, &c, MenuAction::CycleSensitivity),
            "SENSITIVITY  125"
        );
        assert_eq!(
            activate_c(&mut m, &mut s, &mut c, MenuAction::ToggleInvertY),
            MenuOutcome::SaveControls(ControlsSave::InvertY)
        );
        assert!(c.invert_y);
        assert_eq!(
            label_of(&m, &s, &c, MenuAction::ToggleInvertY),
            "INVERT Y  ON"
        );
    }

    #[test]
    fn rebinding_waits_for_a_key_and_takes_it_from_others() {
        let (mut m, s, mut c) = controls_menu();
        let jump = MenuAction::Rebind(Action::Jump);
        assert_eq!(label_of(&m, &s, &c, jump), "JUMP  SPACE");
        let mut s2 = GraphicsSettings::default();
        assert_eq!(activate_c(&mut m, &mut s2, &mut c, jump), MenuOutcome::Stay);
        assert_eq!(m.capturing, Some(Action::Jump));
        assert_eq!(label_of(&m, &s, &c, jump), "JUMP  PRESS A KEY");

        // Menu keys and keys the file can't name keep it waiting.
        for key in [
            KeyCode::Enter,
            KeyCode::ArrowUp,
            KeyCode::ArrowDown,
            KeyCode::PrintScreen,
        ] {
            assert_eq!(m.capture_key(&mut c, key), MenuOutcome::Stay, "{key:?}");
            assert_eq!(m.capturing, Some(Action::Jump), "{key:?} ended the wait");
        }
        assert_eq!(c, Controls::default(), "nothing bound yet");

        // V was NOCLIP's: JUMP gets it, NOCLIP is left unbound.
        assert_eq!(
            m.capture_key(&mut c, KeyCode::KeyV),
            MenuOutcome::SaveControls(ControlsSave::Keys)
        );
        assert_eq!(m.capturing, None);
        assert_eq!(label_of(&m, &s, &c, jump), "JUMP  V");
        assert_eq!(
            label_of(&m, &s, &c, MenuAction::Rebind(Action::Noclip)),
            "NOCLIP  NONE"
        );
        // Not waiting any more: further keys do nothing.
        assert_eq!(m.capture_key(&mut c, KeyCode::KeyK), MenuOutcome::Stay);
        assert_eq!(label_of(&m, &s, &c, jump), "JUMP  V");
    }

    #[test]
    fn escape_moving_or_back_cancel_the_wait() {
        let (mut m, mut s, mut c) = controls_menu();
        let jump = MenuAction::Rebind(Action::Jump);

        activate_c(&mut m, &mut s, &mut c, jump);
        assert_eq!(m.capture_key(&mut c, KeyCode::Escape), MenuOutcome::Stay);
        assert_eq!((m.capturing, m.screen), (None, MenuScreen::Controls));
        assert_eq!(c, Controls::default());

        // Hovering the waiting row (mouse jitter) keeps waiting; another row
        // doesn't.
        activate_c(&mut m, &mut s, &mut c, jump);
        let i = m.index;
        m.hover(i, &s, &c, &AudioSettings::default(), true);
        assert_eq!(m.capturing, Some(Action::Jump));
        m.move_by(1, &s, &c, &AudioSettings::default(), true);
        assert_eq!(m.capturing, None);

        activate_c(&mut m, &mut s, &mut c, jump);
        m.back();
        assert_eq!((m.capturing, m.screen), (None, MenuScreen::Options));
    }

    #[test]
    fn reset_keys_restores_the_defaults() {
        let (mut m, mut s, mut c) = controls_menu();
        c.rebind(Action::Forward, KeyCode::KeyV);
        c.sensitivity = 2.0;
        assert_eq!(
            activate_c(&mut m, &mut s, &mut c, MenuAction::ResetKeys),
            MenuOutcome::SaveControls(ControlsSave::Keys)
        );
        assert_eq!(c.keys_label(Action::Forward), "W");
        assert_eq!(c.keys_label(Action::Noclip), "V");
        assert_eq!(c.sensitivity, 2.0, "mouse look isn't a key binding");
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
        let gameplay_row = screen_rows(
            MenuScreen::Options,
            &s,
            &Controls::default(),
            &AudioSettings::default(),
            true,
            None,
        )
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
        // GAMEPLAY and SOUND had placeholders until FIELD OF VIEW and audio
        // went live; only GRAPHICS (MSAA, locked in game) still has one.
        for screen in [MenuScreen::Graphics] {
            let mut s = GraphicsSettings::default();
            m.screen = screen;
            m.stack.clear();
            let inert: Vec<usize> = screen_rows(
                screen,
                &s,
                &Controls::default(),
                &AudioSettings::default(),
                true,
                None,
            )
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.enabled())
            .map(|(i, _)| i)
            .collect();
            assert!(!inert.is_empty(), "{screen:?} has no inert row to check");
            for i in inert {
                m.index = i;
                assert_eq!(
                    m.activate(
                        &mut s,
                        &mut Controls::default(),
                        &mut AudioSettings::default(),
                        true
                    ),
                    MenuOutcome::Stay
                );
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
        let n = m
            .rows(&s, &Controls::default(), &AudioSettings::default(), true)
            .len();
        m.index = 0;
        m.move_by(
            -1,
            &s,
            &Controls::default(),
            &AudioSettings::default(),
            true,
        );
        assert_eq!(m.index, n - 1, "up from the top did not wrap");
        m.move_by(1, &s, &Controls::default(), &AudioSettings::default(), true);
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
            for row in screen_rows(
                screen,
                &s,
                &Controls::default(),
                &AudioSettings::default(),
                true,
                None,
            ) {
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
            screen_rows(
                MenuScreen::Graphics,
                &s,
                &Controls::default(),
                &AudioSettings::default(),
                in_session,
                None,
            )
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
                .rows(&s, &Controls::default(), &AudioSettings::default(), false)
                .iter()
                .position(|r| r.action == MenuAction::CycleMsaa)
                .expect("msaa row");
            m.index = i;
            assert_eq!(
                m.activate(
                    &mut s,
                    &mut Controls::default(),
                    &mut AudioSettings::default(),
                    false
                ),
                MenuOutcome::ApplyMsaa
            );
            assert_eq!(s.msaa, want);
        }
    }

    #[test]
    fn main_menu_starts_a_session_and_pause_menu_ends_it() {
        let mut s = GraphicsSettings::default();
        let mut m = Menu::new();
        assert_eq!(m.screen, MenuScreen::MainRoot, "app opens on the main menu");

        let i = m
            .rows(&s, &Controls::default(), &AudioSettings::default(), false)
            .iter()
            .position(|r| r.action == MenuAction::NewGame)
            .expect("new game row");
        m.index = i;
        assert_eq!(
            m.activate(
                &mut s,
                &mut Controls::default(),
                &mut AudioSettings::default(),
                false
            ),
            MenuOutcome::StartSession
        );

        // The app resets to Root when the session starts.
        m.reset(MenuScreen::Root);
        let i = m
            .rows(&s, &Controls::default(), &AudioSettings::default(), true)
            .iter()
            .position(|r| r.action == MenuAction::ToMainMenu)
            .expect("main menu row");
        m.index = i;
        assert_eq!(
            m.activate(
                &mut s,
                &mut Controls::default(),
                &mut AudioSettings::default(),
                true
            ),
            MenuOutcome::EndSession
        );
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
                .rows(
                    &s,
                    &Controls::default(),
                    &AudioSettings::default(),
                    in_session,
                )
                .iter()
                .position(|r| r.action == MenuAction::Enter(MenuScreen::Options))
                .unwrap_or_else(|| panic!("{root:?} has no OPTIONS row"));
            m.index = i;
            assert_eq!(
                m.activate(
                    &mut s,
                    &mut Controls::default(),
                    &mut AudioSettings::default(),
                    in_session
                ),
                MenuOutcome::Stay
            );
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
        // Every FOV the menu offers, and both ends of the file's range.
        for deg in [30.0f32, 50.0, 60.0, 90.0, 120.0] {
            let (fov, aspect, near, far) = (deg.to_radians(), 1.6, 2.0, 30.0);
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
                            "corner outside the fitted sphere at {deg} degrees"
                        );
                    }
                }
            }
        }
    }

    /// The projection really uses the FOV it's given: a point 40 degrees above
    /// the view axis is inside a 90-degree frustum and outside a 60-degree one,
    /// and the matrix matches glam's own perspective (Y flipped for Vulkan).
    #[test]
    fn view_proj_uses_the_fov_setting() {
        let look = Look::new();
        let eye = Vec3::ZERO;
        let point = look.forward() * 10.0 + Vec3::Y * 10.0 * 40f32.to_radians().tan();
        let visible = |deg: f32| {
            Frustum::from_view_proj(&look.view_proj(eye, 1.0, deg.to_radians()))
                .contains_sphere(point, 0.0)
        };
        assert!(visible(90.0));
        assert!(!visible(60.0));
        let fov = 75f32.to_radians();
        let mut want = Mat4::perspective_rh(fov, 1.5, CAMERA_NEAR, CAMERA_FAR);
        want.y_axis.y *= -1.0;
        let got = look.view_proj(eye, 1.5, fov);
        assert!(got.abs_diff_eq(want * look.view(eye), 1e-6));
    }

    #[test]
    fn fov_row_steps_through_presets_and_saves() {
        let (mut s, mut c) = (GraphicsSettings::default(), Controls::default());
        let mut m = in_game_menu();
        activate_c(
            &mut m,
            &mut s,
            &mut c,
            MenuAction::Enter(MenuScreen::Options),
        );
        activate_c(
            &mut m,
            &mut s,
            &mut c,
            MenuAction::Enter(MenuScreen::Gameplay),
        );
        let label = |m: &Menu, s: &GraphicsSettings| {
            label_of(m, s, &Controls::default(), MenuAction::CycleFov)
        };
        assert_eq!(label(&m, &s), "FIELD OF VIEW  60");
        let mut seen = vec![s.fov_deg];
        for _ in 0..FOV_PRESETS.len() {
            assert_eq!(
                activate_c(&mut m, &mut s, &mut c, MenuAction::CycleFov),
                MenuOutcome::ApplyFov
            );
            seen.push(s.fov_deg);
        }
        assert_eq!(seen, [60, 70, 80, 90, 50, 60]);
        assert_eq!(label(&m, &s), "FIELD OF VIEW  60");
        // A hand-typed value steps to the next preset above it.
        s.fov_deg = 73;
        s.step_fov();
        assert_eq!(s.fov_deg, 80);
        s.fov_deg = 120;
        s.step_fov();
        assert_eq!(s.fov_deg, 50);
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
            baked: None,
            spec,
            surface: mesh_surfaces(std::slice::from_ref(cube)).0[0],
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

    /// A level without an `environment` marker keeps the look every level had
    /// before levels could choose one, and so does one whose marker is empty.
    #[test]
    fn a_level_without_an_environment_keeps_the_default_look() {
        let prop = || node(Some("prop"), serde_json::json!({}));
        assert_eq!(environment(&[prop()]), (Environment::default(), Vec::new()));
        let empty = node(Some("environment"), serde_json::json!({}));
        assert_eq!(
            environment(&[prop(), empty]),
            (Environment::default(), Vec::new())
        );
    }

    #[test]
    fn an_environment_marker_sets_the_atmosphere() {
        let marker = node(
            Some("environment"),
            serde_json::json!({
                "sun_elevation": 90.0, "sun_azimuth": 0.0,
                "sun_color": [1.0, 0.5, 0.25], "sun_intensity": 3.0,
                "sky_zenith": [0.1, 0.2, 0.3], "sky_horizon": [0.4, 0.5, 0.6],
                "sky_ground": [0.7, 0.8, 0.9], "sky_sun_color": [0.9, 0.9, 0.8],
                "sky_intensity": 1.5, "sun_glow": 0.2, "sun_disk": 0.0,
                "fog_density": 0.02, "fog_height": -9.0, "fog_falloff": 0.08,
                "fog_color": [0.5, 0.52, 0.48], "fog_sun": 0.3, "exposure": 1.2,
            }),
        );
        let (env, bad) = environment(&[marker]);
        assert!(bad.is_empty(), "{bad:?}");
        assert!(
            (env.sun_dir - Vec3::NEG_Y).length() < 1e-6,
            "{:?}",
            env.sun_dir
        );
        let want = Environment {
            sun_dir: env.sun_dir,
            sun_color: Vec3::new(1.0, 0.5, 0.25),
            sun_intensity: 3.0,
            sky_zenith: Vec3::new(0.1, 0.2, 0.3),
            sky_horizon: Vec3::new(0.4, 0.5, 0.6),
            sky_ground: Vec3::new(0.7, 0.8, 0.9),
            sky_sun_color: Vec3::new(0.9, 0.9, 0.8),
            sky_intensity: 1.5,
            sun_glow: 0.2,
            sun_disk: 0.0,
            fog_density: 0.02,
            fog_height: -9.0,
            fog_falloff: 0.08,
            fog_color: Some(Vec3::new(0.5, 0.52, 0.48)),
            fog_sun: 0.3,
            exposure: 1.2,
        };
        assert_eq!(env, want);
    }

    /// Params left out keep their defaults; ones it doesn't know, or can't
    /// read, are reported (for a warning) and change nothing.
    #[test]
    fn a_partial_environment_keeps_the_rest() {
        let marker = node(
            Some("environment"),
            serde_json::json!({
                "fog_density": 0.05, "fog": 1.0, "sky_zenith": "grey", "fog_color": 0.5,
            }),
        );
        let (env, mut bad) = environment(&[marker]);
        bad.sort();
        assert_eq!(bad, ["fog", "fog_color", "sky_zenith"]);
        let want = Environment {
            fog_density: 0.05,
            ..Default::default()
        };
        assert_eq!(env, want);
    }

    /// Elevation is above the horizon and azimuth clockwise from north (-Z),
    /// seen from above; the direction is the way the light *travels*, away
    /// from the sun. Giving one angle keeps the default sun's other one.
    #[test]
    fn the_sun_is_placed_by_elevation_and_azimuth() {
        let close = |a: Vec3, b: Vec3| (a - b).length() < 1e-5;
        assert!(close(sun_travel(90.0, 123.0), Vec3::NEG_Y));
        assert!(close(sun_travel(0.0, 0.0), Vec3::Z)); // sun in the north
        assert!(close(sun_travel(0.0, 90.0), Vec3::NEG_X)); // sun in the east
        let d = Environment::default().sun_dir;
        let (el, az) = sun_angles(d);
        assert!(close(sun_travel(el, az), d));
        let marker = node(
            Some("environment"),
            serde_json::json!({ "sun_elevation": 10.0 }),
        );
        let (env, _) = environment(&[marker]);
        let (el2, az2) = sun_angles(env.sun_dir);
        assert!(
            (el2 - 10.0).abs() < 1e-3 && (az2 - az).abs() < 1e-3,
            "{el2} {az2}"
        );
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

    #[test]
    fn a_material_surface_tags_the_collider() {
        use Surface::{Concrete, Grass, Snow};
        let cube = |surface: Option<&str>| {
            let mut m = MeshData::cube(1.0);
            m.material.surface = surface.map(str::to_string);
            m
        };
        let meshes = [
            cube(Some("grass")),
            cube(Some("Snow")),
            cube(None),
            cube(Some("lava")),
            cube(Some("lava")),
        ];
        let (surfaces, unknown) = mesh_surfaces(&meshes);
        assert_eq!(surfaces, [Grass, Snow, Concrete, Concrete, Concrete]);
        assert_eq!(unknown, ["lava"], "each unknown name is reported once");

        // Spawned as a scene node, the collider carries it.
        for (mesh, want) in meshes.iter().zip(surfaces) {
            let mut w = prefab_world();
            spawn_one(&mut w, None, mesh);
            let c = w
                .query::<&ColliderRef>()
                .iter(&w)
                .next()
                .expect("collider")
                .0;
            let got = collider_surface(&w.resource::<Physics>().colliders[c]);
            assert_eq!(got, want);
        }

        let mut stats = ColliderStats::default();
        stats.surfaces[Concrete.index()] = 2;
        stats.surfaces[Snow.index()] = 1;
        assert_eq!(stats.surfaces_line(), "2 concrete, 1 snow");
    }

    // ---- end to end: a scene file through the game's own world build ----

    /// Where the end-to-end level's pads go: a row along +X at this z, clear
    /// of the built-in boxes, pad `i` centred at `pad_x(i)`, 4 m long with
    /// 1 m of ground between pads. The player starts at `PAD_START`.
    const PAD_Z: f32 = 9.75;
    const PAD_START: Vec3 = Vec3::new(1.0, GROUND_Y, PAD_Z);

    fn pad_x(i: usize) -> f32 {
        5.5 + 5.0 * i as f32
    }

    /// A glTF level: one flat pad (4 x 0.05 x 3.5 m) per `(material name,
    /// extras.surface)`, placed by `pad_x`, and a `player_start` marker at
    /// `PAD_START` facing +X. Written as a real .gltf + .bin in `dir`. The
    /// pads share the unit cube's geometry, scaled by their nodes, but each
    /// has its own mesh, since the app derives one material per mesh.
    fn write_pad_level(dir: &std::path::Path, pads: &[(&str, Option<&str>)]) -> String {
        use serde_json::json;
        let cube = MeshData::cube(1.0);
        let mut bin: Vec<u8> = cube
            .vertices
            .iter()
            .flat_map(|v| v.pos)
            .flat_map(f32::to_le_bytes)
            .collect();
        let positions = bin.len();
        bin.extend(cube.indices.iter().flat_map(|i| i.to_le_bytes()));
        std::fs::write(dir.join("pads.bin"), &bin).unwrap();

        let mut nodes: Vec<serde_json::Value> = (0..pads.len())
            .map(|i| {
                json!({
                    "mesh": i,
                    "translation": [pad_x(i), GROUND_Y + 0.025, PAD_Z],
                    "scale": [4.0, 0.05, 3.5],
                })
            })
            .collect();
        nodes.push(json!({
            "translation": PAD_START.to_array(),
            "extras": { "prefab": "player_start", "params": { "yaw": 0.0 } },
        }));
        let meshes: Vec<serde_json::Value> = (0..pads.len())
            .map(|i| json!({ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1, "material": i }] }))
            .collect();
        let materials: Vec<serde_json::Value> = pads
            .iter()
            .map(|&(name, surface)| match surface {
                Some(s) => json!({ "name": name, "extras": { "surface": s } }),
                None => json!({ "name": name }),
            })
            .collect();
        let doc = json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": (0..nodes.len()).collect::<Vec<_>>() }],
            "nodes": nodes,
            "meshes": meshes,
            "materials": materials,
            "accessors": [
                {
                    "bufferView": 0, "componentType": 5126, "type": "VEC3",
                    "count": cube.vertices.len(),
                    "min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5],
                },
                {
                    "bufferView": 1, "componentType": 5125, "type": "SCALAR",
                    "count": cube.indices.len(),
                },
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": positions },
                { "buffer": 0, "byteOffset": positions, "byteLength": bin.len() - positions },
            ],
            "buffers": [{ "byteLength": bin.len(), "uri": "pads.bin" }],
        });
        let path = dir.join("pads.gltf");
        std::fs::write(&path, doc.to_string()).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// Build the level at `path` the way `Session::new` does, then walk +X
    /// along the pads on the game's real schedule, feeding the player to a
    /// `StepTracker` once per tick as the app does once per frame. Returns
    /// each step's x and surface, and the load's colliders per surface.
    fn walk_pad_level(path: String) -> (Vec<(f32, Surface)>, String) {
        let mut b = build_world(&[path], None);
        let start = b.world.get::<Player>(b.player).expect("player").pos;
        assert_eq!(start, PAD_START, "the player starts at the scene's marker");
        let surfaces = b.world.resource::<ColliderStats>().surfaces_line();
        b.world.resource_mut::<InputState>().wish = Vec3::X;
        let mut tracker = StepTracker::default();
        let mut steps = Vec::new();
        for _ in 0..600 {
            b.schedule.run(&mut b.world);
            let p = b.world.get::<Player>(b.player).expect("player");
            for e in tracker.update(p.pos, p.vel, p.on_ground, false, p.surface) {
                if let crate::audio::SoundEvent::Step(s) = e {
                    steps.push((p.pos.x, s));
                }
            }
            // The last pad ends at 32.5 and the ground at 40.
            if p.pos.x > 34.0 {
                return (steps, surfaces);
            }
        }
        panic!("never got past the pads: {steps:?}");
    }

    /// Every step well inside pad `i` must sound like `want[i]`, every step
    /// well clear of the pads like the concrete ground, and every pad must get
    /// a step, so an empty walk can't pass. Within 0.4 m of a pad's edge (the
    /// capsule's radius and a bit) either answer is right, so those steps
    /// aren't judged.
    fn check_steps(steps: &[(f32, Surface)], want: &[Surface]) {
        const EDGE: f32 = 0.4;
        let mut inside = vec![0; want.len()];
        for &(x, got) in steps {
            match (0..want.len()).find(|&i| (x - pad_x(i)).abs() < 2.0 + EDGE) {
                Some(i) if (x - pad_x(i)).abs() <= 2.0 - EDGE => {
                    assert_eq!(got, want[i], "step at x {x:.2} on pad {i}: {steps:?}");
                    inside[i] += 1;
                }
                Some(_) => {}
                None => assert_eq!(got, Surface::Concrete, "step at x {x:.2}: {steps:?}"),
            }
        }
        assert!(
            inside.iter().all(|&n| n > 0),
            "steps inside each pad: {inside:?}, {steps:?}"
        );
    }

    /// The whole chain on a real file: glTF material extras → the loader →
    /// per-mesh surfaces → spawned colliders → the controller's probe →
    /// `StepTracker`, all through `build_world` and the real schedule. (From
    /// the step event on, `each_surface_plays_its_own_steps` has it.)
    #[test]
    fn a_scene_steps_like_its_materials() {
        use Surface::{Carpet, Concrete, Grass, Snow, Wood};
        let pads = [
            ("turf", Some("grass"), Grass),
            ("planks", Some("wood"), Wood),
            ("rug", Some("Carpet"), Carpet), // any case
            ("drift", Some("snow"), Snow),
            ("magma", Some("lava"), Concrete), // not a surface
            ("plain", None, Concrete),         // untagged
        ];
        let dir = crate::config::test_dir("e2e-tagged");
        std::fs::create_dir_all(&dir).unwrap();
        let tags: Vec<(&str, Option<&str>)> = pads.iter().map(|&(n, t, _)| (n, t)).collect();
        let (steps, surfaces) = walk_pad_level(write_pad_level(&dir, &tags));
        assert_eq!(surfaces, "2 concrete, 1 grass, 1 wood, 1 carpet, 1 snow");
        check_steps(&steps, &pads.map(|p| p.2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A level's `environment` marker reaches the session through the game's
    /// own build, which the renderer then bakes in (§13); without it, the
    /// same level gets the default look.
    #[test]
    fn a_levels_environment_reaches_the_world_build() {
        let dir = crate::config::test_dir("e2e-environment");
        std::fs::create_dir_all(&dir).unwrap();
        let plain = write_pad_level(&dir, &[("plain", None)]);
        let b = build_world(std::slice::from_ref(&plain), None);
        assert_eq!(b.environment, Environment::default());

        let mut doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&plain).unwrap()).unwrap();
        let nodes = doc["nodes"].as_array_mut().unwrap();
        nodes.push(serde_json::json!({
            "extras": { "prefab": "environment", "params": { "fog_density": 0.03, "sun_disk": 0.0 } },
        }));
        let marker = nodes.len() - 1;
        doc["scenes"][0]["nodes"]
            .as_array_mut()
            .unwrap()
            .push(marker.into());
        std::fs::write(&plain, doc.to_string()).unwrap();
        let b = build_world(&[plain], None);
        let want = Environment {
            fog_density: 0.03,
            sun_disk: 0.0,
            ..Default::default()
        };
        assert_eq!(b.environment, want);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The orb demo's obstacle boxes stand in the demo, and nowhere in a
    /// loaded level: probing each box's centre finds a collider in the first
    /// and nothing in the second.
    #[test]
    fn a_loaded_level_has_no_demo_boxes() {
        let has_box = |b: &WorldBuild| {
            let ph = b.world.resource::<Physics>();
            let queries = ph.broad_phase.as_query_pipeline(
                ph.narrow_phase.query_dispatcher(),
                &ph.bodies,
                &ph.colliders,
                QueryFilter::default(),
            );
            demo_boxes()
                .map(|(centre, _)| queries.intersect_point(to_rapier(centre)).next().is_some())
        };
        assert_eq!(has_box(&build_world(&[], None)), [true; 5]);
        let dir = crate::config::test_dir("e2e-no-demo-boxes");
        std::fs::create_dir_all(&dir).unwrap();
        let level = write_pad_level(&dir, &[("plain", None)]);
        assert_eq!(has_box(&build_world(&[level], None)), [false; 5]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The same level with no tags: nothing but concrete, so what the tagged
    /// walk heard came from the tags.
    #[test]
    fn an_untagged_scene_steps_on_concrete() {
        let dir = crate::config::test_dir("e2e-untagged");
        std::fs::create_dir_all(&dir).unwrap();
        let pads = ["turf", "planks", "rug", "drift", "magma", "plain"].map(|n| (n, None));
        let (steps, surfaces) = walk_pad_level(write_pad_level(&dir, &pads));
        assert_eq!(surfaces, "6 concrete");
        check_steps(&steps, &[Surface::Concrete; 6]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- §12 punctual lights ----

    /// The shape rapier built for the single collider-carrying prop in `w`.
    fn collider_shape(w: &mut World) -> Option<rapier3d::parry::shape::ShapeType> {
        let c = w.query::<&ColliderRef>().iter(w).next()?.0;
        Some(
            w.resource::<Physics>()
                .colliders
                .get(c)?
                .shape()
                .shape_type(),
        )
    }

    #[test]
    fn collider_param_selects_the_proxy() {
        use rapier3d::parry::shape::ShapeType::{ConvexPolyhedron, TriMesh};
        let cube = MeshData::cube(1.0); // 12 triangles
        let dense = MeshData::uv_sphere(64, 64, 1.0); // well past TRIMESH_MAX_TRIS
        assert!(dense.indices.len() / 3 > TRIMESH_MAX_TRIS);
        let cases: [(&MeshData, serde_json::Value, Option<_>); 8] = [
            // Auto: exact under the budget, a hull over it.
            (&cube, serde_json::json!({}), Some(TriMesh)),
            (&dense, serde_json::json!({}), Some(ConvexPolyhedron)),
            // Explicit choices override the budget either way.
            (
                &cube,
                serde_json::json!({ "collider": "hull" }),
                Some(ConvexPolyhedron),
            ),
            (
                &dense,
                serde_json::json!({ "collider": "mesh" }),
                Some(TriMesh),
            ),
            (
                &dense,
                serde_json::json!({ "collider": "box" }),
                Some(ConvexPolyhedron),
            ),
            (&dense, serde_json::json!({ "collider": "none" }), None),
            // `collide: false` still wins, so older scenes keep their meaning.
            (
                &cube,
                serde_json::json!({ "collide": false, "collider": "mesh" }),
                None,
            ),
            // An unknown value warns and falls back to auto.
            (
                &cube,
                serde_json::json!({ "collider": "bogus" }),
                Some(TriMesh),
            ),
        ];
        for (i, (mesh, params, want)) in cases.into_iter().enumerate() {
            let spec = feather_assets::PrefabSpec {
                id: "prop".into(),
                params,
            };
            let mut w = prefab_world();
            spawn_one(&mut w, Some(&spec), mesh);
            assert_eq!(collider_shape(&mut w), want, "case {i}");
        }
    }

    #[test]
    fn a_flat_hull_still_holds_the_player() {
        // Coplanar points: parry still builds a (zero-thickness) hull rather
        // than failing. What matters is that it collides, so check that
        // directly: a floor tile asked for a hull, raised 1 m, must hold the
        // player instead of letting them fall through to the ground.
        let v = |x: f32, z: f32| feather_assets::Vertex {
            pos: [x, 0.0, z],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
        };
        let quad = MeshData {
            vertices: vec![v(-2.0, -2.0), v(2.0, -2.0), v(2.0, 2.0), v(-2.0, 2.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            material: Default::default(),
        };
        let at = Mat4::from_translation(Vec3::new(0.0, GROUND_Y + 1.0, 0.0));
        let (mut physics, _) = setup(&[], Vec3::new(20.0, GROUND_Y, 20.0));
        build_collider(&mut physics, &quad, None, at, ColliderKind::Hull).expect("collider");
        let mut p = Player::new(&mut physics, Vec3::new(0.0, GROUND_Y + 2.0, 0.0));
        physics.step();
        for _ in 0..90 {
            step(&mut p, &mut physics, Vec3::ZERO, false, 0.0, false);
        }
        assert!(p.on_ground, "should stand on the tile");
        assert!(
            (p.pos.y - (GROUND_Y + 1.0)).abs() < 0.05,
            "stood at {}",
            p.pos.y
        );
    }

    #[test]
    fn a_dense_prop_is_walkable_as_a_hull() {
        // The lantern case, synthetic: a 30 cm ball of ~32k triangles on the
        // ground. Auto makes it a hull; the player can stand on it and walk off.
        let ball = MeshData::uv_sphere(128, 128, 0.3);
        assert!(ball.indices.len() / 3 > TRIMESH_MAX_TRIS);
        let at = Mat4::from_translation(Vec3::new(0.0, GROUND_Y + 0.3, 0.0));
        let (mut physics, _) = setup(&[], Vec3::new(20.0, GROUND_Y, 20.0));
        let (c, built) =
            build_collider(&mut physics, &ball, None, at, ColliderKind::Auto).expect("collider");
        assert_eq!(built, BuiltCollider::Hull, "no bake: over budget is a hull");
        assert_eq!(
            physics.colliders.get(c).unwrap().shape().shape_type(),
            rapier3d::parry::shape::ShapeType::ConvexPolyhedron
        );
        let mut p = Player::new(&mut physics, Vec3::new(0.0, GROUND_Y + 1.0, 0.0));
        physics.step();
        for _ in 0..60 {
            step(&mut p, &mut physics, Vec3::ZERO, false, 0.0, false);
        }
        assert!(p.on_ground, "should stand on the ball");
        assert!(
            p.pos.y > GROUND_Y + 0.4,
            "stood at {}, not on top of the ball",
            p.pos.y
        );
        for _ in 0..90 {
            step(&mut p, &mut physics, Vec3::X, false, 0.0, false);
        }
        assert!(
            p.on_ground && (p.pos.y - GROUND_Y).abs() < 0.05,
            "landed at {}",
            p.pos.y
        );
    }

    fn lod(error: f32, tris: usize) -> feather_assets::bake::Lod {
        feather_assets::bake::Lod {
            error,
            indices: vec![0; tris * 3],
        }
    }

    #[test]
    fn collision_lod_takes_the_finest_level_within_budget_and_tolerance() {
        let chain = [
            lod(0.0, 8000),
            lod(0.01, 4000),
            lod(0.03, 2000),
            lod(0.2, 1000),
        ];
        // LOD1 is too big; LOD2 is the first under the budget, 3 cm off.
        assert_eq!(collision_lod(&chain, 1.0), Some(2));
        // At 2x its error is 6 cm, over tolerance, and so is every later level.
        assert_eq!(collision_lod(&chain, 2.0), None);
        // A small mesh qualifies as itself.
        assert_eq!(
            collision_lod(&[lod(0.0, 500), lod(0.02, 250)], 10.0),
            Some(0)
        );
        // Nothing under the budget at all.
        assert_eq!(
            collision_lod(&[lod(0.0, 9000), lod(0.001, 5000)], 1.0),
            None
        );
        assert_eq!(collision_lod(&[], 1.0), None);
    }

    /// A dense dish, 6 m wide and 1 m deep: a concave prop, over the triangle
    /// budget. Returns the fine mesh and a bake whose LOD1 is every 4th grid
    /// line (the same vertex array, as a real bake shares it).
    fn dish() -> (MeshData, feather_assets::bake::BakedMesh) {
        const N: usize = 64;
        let (r, depth) = (3.0f32, 1.0f32);
        let mut m = MeshData::cube(1.0);
        m.vertices.clear();
        m.indices.clear();
        for i in 0..=N {
            for j in 0..=N {
                let x = -r + 2.0 * r * j as f32 / N as f32;
                let z = -r + 2.0 * r * i as f32 / N as f32;
                let y = depth * ((x * x + z * z) / (r * r)).min(1.0);
                m.vertices.push(feather_assets::Vertex {
                    pos: [x, y, z],
                    normal: [0.0, 1.0, 0.0],
                    uv: [0.0, 0.0],
                });
            }
        }
        // Counter-clockwise from above, so every triangle faces up.
        let grid = |step: usize| {
            let mut idx = Vec::new();
            for i in (0..N).step_by(step) {
                for j in (0..N).step_by(step) {
                    let v = |i: usize, j: usize| (i * (N + 1) + j) as u32;
                    let (a, b, c, d) = (
                        v(i, j),
                        v(i + step, j),
                        v(i, j + step),
                        v(i + step, j + step),
                    );
                    idx.extend_from_slice(&[a, b, c, c, b, d]);
                }
            }
            idx
        };
        m.indices = grid(1);
        let baked = feather_assets::bake::BakedMesh {
            vertices: m.vertices.clone(),
            lods: vec![
                feather_assets::bake::Lod {
                    error: 0.0,
                    indices: m.indices.clone(),
                },
                feather_assets::bake::Lod {
                    error: 0.01,
                    indices: grid(4),
                },
            ],
        };
        (m, baked)
    }

    /// Drop the player into the dish; where do its feet come to rest?
    fn rest_height_in_dish(
        baked: Option<&feather_assets::bake::BakedMesh>,
    ) -> (f32, BuiltCollider) {
        let (mesh, bake) = dish();
        assert!(mesh.indices.len() / 3 > TRIMESH_MAX_TRIS);
        let baked = baked.map(|_| &bake);
        let at = Mat4::from_translation(Vec3::new(0.0, GROUND_Y + 0.2, 0.0));
        let (mut physics, _) = setup(&[], Vec3::new(20.0, GROUND_Y, 20.0));
        let (_, built) =
            build_collider(&mut physics, &mesh, baked, at, ColliderKind::Auto).expect("collider");
        let mut p = Player::new(&mut physics, Vec3::new(0.0, GROUND_Y + 2.5, 0.0));
        physics.step();
        for _ in 0..90 {
            step(&mut p, &mut physics, Vec3::ZERO, false, 0.0, false);
        }
        assert!(p.on_ground, "never landed ({built:?})");
        (p.pos.y - GROUND_Y, built)
    }

    /// The point of LOD collision: a hull fills the dish up to its rim, a LOD
    /// trimesh lets you stand in it.
    #[test]
    fn a_baked_lod_keeps_a_concave_prop_concave() {
        let (y, built) = rest_height_in_dish(Some(&dish().1));
        println!("dish, LOD1 trimesh: feet at {y:.3} above ground");
        assert_eq!(built, BuiltCollider::Lod(1));
        assert!(
            y < 0.5,
            "stood at {y}, not down in the dish (bottom at 0.2)"
        );
        let (y, built) = rest_height_in_dish(None);
        println!("dish, hull: feet at {y:.3} above ground");
        assert_eq!(built, BuiltCollider::Hull);
        assert!(
            y > 1.1,
            "stood at {y}, but the hull's lid is at the rim (1.2)"
        );
    }

    #[test]
    fn explicit_mesh_collider_stays_exact() {
        let (mesh, bake) = dish();
        let (mut physics, _) = setup(&[], Vec3::new(20.0, GROUND_Y, 20.0));
        let (_, built) = build_collider(
            &mut physics,
            &mesh,
            Some(&bake),
            Mat4::IDENTITY,
            ColliderKind::Mesh,
        )
        .expect("collider");
        assert_eq!(built, BuiltCollider::Mesh);
    }

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
        let vp = look.view_proj(eye, 16.0 / 9.0, GraphicsSettings::default().fov_y());
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

    /// The test level's tower (tools/gen_testscene.py): a 70 m pillar at
    /// (30, 30), and the point on the ground where its top's shadow lands.
    fn tower_top_and_its_shadow() -> (Vec3, Vec3, Vec3) {
        let sun = Vec3::new(-0.4, -1.0, -0.3).normalize(); // as App::new
        let top = Vec3::new(30.0, GROUND_Y + 69.5, 30.0);
        let tip = top + sun * ((GROUND_Y - top.y) / sun.y);
        (sun, top, tip)
    }

    #[test]
    fn tall_casters_beyond_the_near_plane_still_cast() {
        // The real cascade setup, with the player standing on the tower's
        // shadow tip and looking at the tower.
        let (sun, top, tip) = tower_top_and_its_shadow();
        let eye = Vec3::new(tip.x, GROUND_Y + EYE_HEIGHT, tip.z);
        let mut look = Look::new();
        let to_tower = (top - eye).with_y(0.0).normalize();
        look.yaw = to_tower.z.atan2(to_tower.x);
        let (fwd, right, up) = look.camera_basis();
        let splits = cascade_splits(0.1, SHADOW_DISTANCE, SHADOW_LAMBDA);
        let mut near = 0.1;
        let mut covering = 0;
        let mut clipped = Vec::new();
        for (i, far) in splits.into_iter().enumerate() {
            let fov = GraphicsSettings::default().fov_y();
            let (c, r) = slice_sphere(eye, fwd, right, up, fov, 16.0 / 9.0, near, far);
            near = far;
            let (vp, _) = fit_cascade(c, r, sun, 2048);
            let full = Frustum::from_view_proj(&vp);
            if !full.contains_sphere(tip, 0.0) {
                continue; // the shadow lands outside this cascade's box
            }
            covering += 1;
            let caster_z = vp.project_point3(top).z;
            let receiver_z = vp.project_point3(tip).z;
            assert!(
                receiver_z > 0.0,
                "cascade {i}: receiver at depth {receiver_z}"
            );
            let kept = Frustum::from_view_proj(&vp)
                .without_near()
                .contains_sphere(top, 0.5);
            assert!(kept, "cascade {i}: caster culling must keep the top");
            if caster_z < 0.0 {
                // Up-light of the near plane: before pancaking it was culled and
                // cast nothing here; now it is kept, and the depth clamp puts it
                // at 0, in front of the receiver.
                assert!(
                    !full.contains_sphere(top, 0.5),
                    "cascade {i}: top not culled"
                );
                clipped.push(i);
            } else {
                // A cascade big enough to reach it was never affected.
                assert!(full.contains_sphere(top, 0.5), "cascade {i}");
            }
        }
        // The bug bites where it is most visible: the tight cascades around
        // the player, so the shadow tip vanishes as you walk onto it.
        assert!(clipped.contains(&0), "clipped cascades: {clipped:?}");
        assert!(covering > 0, "no cascade covers the shadow tip");
    }

    #[test]
    fn without_near_only_drops_the_near_plane() {
        let (sun, _, tip) = tower_top_and_its_shadow();
        let (vp, _) = fit_cascade(tip, 5.0, sun, 2048);
        let full = Frustum::from_view_proj(&vp);
        let casters = Frustum::from_view_proj(&vp).without_near();
        for i in 0..2000u32 {
            let p = tip
                + Vec3::new(
                    rand01(i * 5) - 0.5,
                    rand01(i * 5 + 1) - 0.5,
                    rand01(i * 5 + 2) - 0.5,
                ) * 200.0;
            let r = rand01(i * 5 + 3) * 3.0;
            let n = full.planes[4];
            let in_front_of_near = n.truncate().dot(p) + n.w >= -r;
            if in_front_of_near {
                assert_eq!(full.contains_sphere(p, r), casters.contains_sphere(p, r));
            } else {
                assert!(!full.contains_sphere(p, r));
            }
        }
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
