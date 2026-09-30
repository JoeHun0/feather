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

use audio::{Audio, AudioSettings, StepTracker};
use config::controls::{Action, Controls};

use std::collections::HashSet;
use std::time::Instant;

use bevy_ecs::prelude::*;
use feather_game::components::{
    Hanging, Material, Mesh, NoShadowCast, Position, PrevPosition, PrevRotation, Rotation, Scale,
    Transform, FIXED_DT, GRID, MESH_ROPE, PALETTE,
};
use feather_game::controller::{
    InputState, Look, Player, CAMERA_FAR, CAMERA_NEAR, DEFAULT_FOV_DEG, EYE_HEIGHT, FOV_PRESETS,
};
use feather_game::level::build_world;
use feather_game::lights::{extract_lights, PointLight};
use feather_game::{rope, weather};
use feather_gfx::{GpuTimes, Renderer, FRAMES_IN_FLIGHT, SHADOW_CASCADES};
use feather_platform::winit;
use feather_render::{
    cascade_splits, fit_cascade, slice_sphere, world_sphere, AoPass, AoProjection, BloomPass,
    CascadeSetup, ClusterView, Environment, ExposureParams, ExposurePass, FrameStats, Frustum,
    FxaaPass, InstanceData, MeshId, MeshRenderer, SkyPass, TaaPass, TaaPush, TonemapPass, UiPass,
    BLOOM_STRENGTH, SHADOW_DISTANCE, SHADOW_LAMBDA,
};
use glam::{Mat4, Vec3, Vec4};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const MAX_INSTANCES: u32 = 8192;

// Fixed-timestep sim (§4). The schedule advances in whole `FIXED_DT` steps from
// an accumulator; render interpolates the remainder. `MAX_FRAME_TIME` clamps a
// single frame's real delta and `MAX_STEPS` caps steps per frame — together the
// spiral-of-death guard when the app stalls (debugger break, lost focus).
const MAX_FRAME_TIME: f32 = 0.25;
const MAX_STEPS: u32 = 8;

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
    /// TAA (§13). Live-toggleable (`F3`) like FXAA: its history is always
    /// allocated; off, the camera isn't jittered and the resolve doesn't run.
    taa: bool,
    /// Bloom (§13). Live-toggleable like FXAA: its chain is always allocated
    /// and it bakes nothing in; off, the passes simply don't run.
    bloom: bool,
    /// Auto-exposure (§13). Live-toggleable; off, the exposure is fixed (the
    /// level's `exposure`, adjusted by `[`/`]`).
    auto_exposure: bool,
    /// GTAO (§13). Live-toggleable: its targets are always allocated; off,
    /// its passes don't run and the main pass skips its result.
    ambient_occlusion: bool,
    /// Use baked assets from `BAKE_DIR` when present (§17): BC7 textures and
    /// meshes with LODs. `--no-bake` forces the raw assets, for A/B comparisons.
    bake: bool,
    /// Pick LODs per view (§17). `--no-lod` pins LOD0 while keeping the baked
    /// vertex order, so an A/B separates the ordering win from the LOD win.
    lod: bool,
    /// The session's weather (§13): the clock, the chosen weather and OPTIONS
    /// > WEATHER's fog overrides. Never saved, and reset by each new game.
    weather: weather::SessionWeather,
    /// Occlude the ambient light by the level's baked sky visibility (§13),
    /// when it has one. `--no-sky-occlusion` turns it off for A/B runs; not a
    /// menu option, since it's part of the look rather than a cost to trade.
    sky_occlusion: bool,
}

/// Where `feather-bake` writes, and the runtime looks for, baked assets
/// (`tex/`, `mesh/` and `sky/` below it).
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
            taa: true,
            bloom: true,
            auto_exposure: true,
            ambient_occlusion: true,
            bake: true,
            lod: true,
            sky_occlusion: true,
            weather: weather::SessionWeather::default(),
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
    /// The session's weather (§13), in game only: a temporary stand-in until
    /// the weather engine.
    Weather,
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
            Self::Weather => "WEATHER",
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
    ToggleTaa,
    ToggleBloom,
    ToggleAutoExposure,
    ToggleAmbientOcclusion,
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
    /// Next WEATHER choice or preset (§13): the weather, the time (+1 h),
    /// the clock's speed, the fog's density and height.
    CycleWeather,
    StepTime,
    CycleSpeed,
    CycleFogDensity,
    CycleFogHeight,
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
    ApplyTaa,
    ApplyBloom,
    ApplyAutoExposure,
    ApplyAmbientOcclusion,
    ApplyMsaa,
    ApplyDisplay,
    /// The FOV changed: nothing to rebuild (the next frame uses it), just save.
    ApplyFov,
    /// The weather changed: the next frame uses it; never saved.
    ApplyWeather,
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
        ]
        .into_iter()
        // The weather is the session's, reset by each new game: from the
        // main menu it would change nothing.
        .chain(in_session.then(|| MenuRow::new("WEATHER", MenuAction::Enter(MenuScreen::Weather))))
        .chain([MenuRow::new("BACK", MenuAction::Back)])
        .collect(),
        MenuScreen::Weather => {
            let w = &s.weather;
            // Under LEVEL a clock moves the level's sun and nothing else.
            let sun_only = if w.choice == 0 && w.clock.is_some() {
                "  SUN ONLY"
            } else {
                ""
            };
            vec![
                MenuRow::new(
                    format!(
                        "WEATHER  {}{}",
                        weather::choice_name(w.choice),
                        if w.transition().is_some() {
                            "  ARRIVING"
                        } else {
                            ""
                        }
                    ),
                    MenuAction::CycleWeather,
                ),
                MenuRow::new(
                    format!("TIME  {}{sun_only}", w.time_label()),
                    MenuAction::StepTime,
                ),
                MenuRow::new(
                    format!("SPEED  {}", w.speed_label()),
                    MenuAction::CycleSpeed,
                ),
                MenuRow::new(
                    format!("FOG DENSITY  {}", weather::FOG_DENSITIES[w.fog_density].0),
                    MenuAction::CycleFogDensity,
                ),
                MenuRow::new(
                    format!("FOG HEIGHT  {}", weather::FOG_HEIGHTS[w.fog_height].0),
                    MenuAction::CycleFogHeight,
                ),
                MenuRow::new("BACK", MenuAction::Back),
            ]
        }
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
            MenuRow::new(
                format!("TAA  {}", if s.taa { "ON" } else { "OFF" }),
                MenuAction::ToggleTaa,
            ),
            MenuRow::new(
                format!("BLOOM  {}", if s.bloom { "ON" } else { "OFF" }),
                MenuAction::ToggleBloom,
            ),
            MenuRow::new(
                format!(
                    "AUTO EXPOSURE  {}",
                    if s.auto_exposure { "ON" } else { "OFF" }
                ),
                MenuAction::ToggleAutoExposure,
            ),
            MenuRow::new(
                format!(
                    "AMBIENT OCCLUSION  {}",
                    if s.ambient_occlusion { "ON" } else { "OFF" }
                ),
                MenuAction::ToggleAmbientOcclusion,
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
            MenuAction::ToggleTaa => {
                s.taa = !s.taa;
                MenuOutcome::ApplyTaa
            }
            MenuAction::ToggleBloom => {
                s.bloom = !s.bloom;
                MenuOutcome::ApplyBloom
            }
            MenuAction::ToggleAutoExposure => {
                s.auto_exposure = !s.auto_exposure;
                MenuOutcome::ApplyAutoExposure
            }
            MenuAction::ToggleAmbientOcclusion => {
                s.ambient_occlusion = !s.ambient_occlusion;
                MenuOutcome::ApplyAmbientOcclusion
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
            MenuAction::CycleWeather => {
                s.weather.cycle_weather();
                MenuOutcome::ApplyWeather
            }
            MenuAction::StepTime => {
                s.weather.step_time();
                MenuOutcome::ApplyWeather
            }
            MenuAction::CycleSpeed => {
                s.weather.cycle_speed();
                MenuOutcome::ApplyWeather
            }
            MenuAction::CycleFogDensity => {
                let w = &mut s.weather;
                w.fog_density = (w.fog_density + 1) % weather::FOG_DENSITIES.len();
                MenuOutcome::ApplyWeather
            }
            MenuAction::CycleFogHeight => {
                let w = &mut s.weather;
                w.fog_height = (w.fog_height + 1) % weather::FOG_HEIGHTS.len();
                MenuOutcome::ApplyWeather
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
    /// The auto-exposure the GPU metered (§13), per sampled frame; empty
    /// with it off.
    exposures: Vec<f32>,
    /// The latest metered exposure, sampled with the next `record`.
    exposure: Option<f32>,
}

impl Bench {
    fn new() -> Self {
        Self {
            frame: 0,
            start_yaw: None,
            samples: Vec::with_capacity(BENCH_SWEEP as usize),
            lights: 0,
            stats: FrameStats::default(),
            exposures: Vec::new(),
            exposure: None,
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
            self.exposures.extend(self.exposure.take());
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
        eprintln!("[bench] ao      {}", col(|t| t.ao_ms));
        eprintln!("[bench] taa     {}", col(|t| t.taa_ms));
        eprintln!("[bench] bloom   {}", col(|t| t.bloom_ms));
        eprintln!("[bench] expo    {}", col(|t| t.exposure_ms));
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
        if !self.exposures.is_empty() {
            eprintln!(
                "[bench] exposure {}  (auto, metered)",
                stats(self.exposures.clone())
            );
        }
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
    bloom: Option<BloomPass>,
    exposure_pass: Option<ExposurePass>,
    ao_pass: Option<AoPass>,
    fxaa: Option<FxaaPass>,
    taa: Option<TaaPass>,
    /// TAA's frame count (the jitter phase) and last frame's unjittered
    /// view-projection, for the reprojection (§13).
    taa_frame: u64,
    prev_view_proj: Option<Mat4>,
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
    /// Auto-exposure snaps to the scene next frame (a new level, or turned
    /// back on) instead of adapting from where it was.
    exposure_reset: bool,
    /// The level's auto-exposure range (§13).
    exposure_range: (f32, f32),
    /// glTF scenes NEW GAME loads (CLI paths). Empty = the procedural demo.
    /// Kept on `App` rather than `Session` so it survives a teardown.
    scenes: Vec<String>,
    /// `--bench` run state; `None` in normal play.
    bench: Option<Bench>,
    last_frame: Instant,
}

/// Per-frame view data the render closures need. Carried as an `Option` so the
/// main menu can skip the shadow and geometry passes entirely.
#[derive(Clone, Copy)]
struct FrameView {
    /// What the camera passes draw with: jittered under TAA (§13).
    view_proj: Mat4,
    inv_view_proj: Mat4,
    /// TAA's reprojection: this frame's unjittered clip space to last frame's.
    reproject: Mat4,
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
    /// The level's atmosphere (§13). Each frame the weather makes this
    /// frame's from it and hands `mesh` (and so `sky`) its `Atmosphere`.
    environment: Environment,
    /// The level's weather and sun's path (§13): where a new game's clock and
    /// weather start.
    level_weather: weather::LevelWeather,
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
    fn new(renderer: &Renderer, scenes: &[String], bake: bool, lod: bool, sky: bool) -> Self {
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
            b.sky.as_ref().filter(|_| sky),
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
        let sky = SkyPass::new(renderer, mesh.set_layout());

        Self {
            world: b.world,
            schedule: b.schedule,
            player: b.player,
            mesh,
            sky,
            environment: b.environment,
            level_weather: b.level_weather,
            fits: b.fits,
            mesh_spheres: b.mesh_spheres,
            accumulator: 0.0,
            noclip: false,
        }
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
            taa: None,
            taa_frame: 0,
            prev_view_proj: None,
            bloom: None,
            exposure_pass: None,
            ao_pass: None,
            exposure_reset: true,
            exposure_range: (0.125, 8.0),
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

    /// Nothing to push: each frame turns TAA on in the renderer while
    /// there's a session and the setting is on.
    fn log_taa(&self) {
        eprintln!(
            "[quality] taa: {}",
            if self.settings.taa { "on" } else { "off" }
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
            MenuOutcome::ApplyTaa => {
                self.log_taa();
                self.persist(config::graphics::Key::Taa);
            }
            MenuOutcome::ApplyWeather => {
                let w = self.settings.weather;
                eprintln!(
                    "[quality] weather: {}, time {}, speed {}, fog density {}, fog height {}",
                    weather::choice_name(w.choice),
                    w.time_label(),
                    w.speed_label(),
                    weather::FOG_DENSITIES[w.fog_density].0,
                    weather::FOG_HEIGHTS[w.fog_height].0
                );
            }
            MenuOutcome::ApplyBloom => {
                // Nothing to push: the frame reads the setting as it draws.
                eprintln!(
                    "[quality] bloom: {}",
                    if self.settings.bloom { "on" } else { "off" }
                );
                self.persist(config::graphics::Key::Bloom);
            }
            MenuOutcome::ApplyAutoExposure => {
                eprintln!(
                    "[quality] auto exposure: {}",
                    if self.settings.auto_exposure {
                        "on"
                    } else {
                        "off"
                    }
                );
                // Back on, snap to the scene rather than fade from stale state.
                self.exposure_reset = true;
                self.persist(config::graphics::Key::AutoExposure);
            }
            MenuOutcome::ApplyAmbientOcclusion => {
                eprintln!(
                    "[quality] ambient occlusion: {}",
                    if self.settings.ambient_occlusion {
                        "on"
                    } else {
                        "off"
                    }
                );
                self.persist(config::graphics::Key::AmbientOcclusion);
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
            self.settings.sky_occlusion,
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
        // The level's sun and starting exposure (§13); its sky and fog go to
        // the GPU each frame.
        let env = session.environment;
        self.light_dir = env.sun_dir.extend(0.0);
        // A new game has the level's own weather.
        self.settings.weather = weather::SessionWeather::new(&session.level_weather, env.sun_dir);
        self.exposure = env.exposure;
        self.exposure_range = (env.exposure_min, env.exposure_max);
        self.exposure_reset = true;
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
                Action::ToggleTaa => {
                    self.settings.taa = !self.settings.taa;
                    self.log_taa();
                    self.persist(config::graphics::Key::Taa);
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
        let bloom = BloomPass::new(&renderer);
        let exposure_pass = ExposurePass::new(&renderer);
        let ao_pass = AoPass::new(&renderer);
        let fxaa = FxaaPass::new(&renderer);
        let taa = TaaPass::new(&renderer);
        let ui = UiPass::new(&renderer);

        self.tonemap = Some(tonemap);
        self.bloom = Some(bloom);
        self.exposure_pass = Some(exposure_pass);
        self.ao_pass = Some(ao_pass);
        self.fxaa = Some(fxaa);
        self.taa = Some(taa);
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
                    // TAA (§13): the prepass, main pass and sky draw with a
                    // sub-pixel jitter; culling, LOD and the clusters keep the
                    // true camera, which the reprojection uses too.
                    let draw_view_proj = if self.settings.taa {
                        let px =
                            glam::Vec2::new(size.width.max(1) as f32, size.height.max(1) as f32);
                        feather_render::jittered(
                            view_proj,
                            feather_render::jitter(self.taa_frame),
                            px,
                        )
                    } else {
                        view_proj
                    };
                    let reproject = feather_render::reprojection(
                        self.prev_view_proj.unwrap_or(view_proj),
                        view_proj,
                    );
                    self.prev_view_proj = Some(view_proj);
                    self.taa_frame += 1;
                    let cluster_view = ClusterView {
                        view: look.view(eye),
                        fov_y,
                        aspect,
                        near: CAMERA_NEAR,
                        far: CAMERA_FAR,
                        viewport_height: size.height.max(1) as f32,
                    };
                    let inv_view_proj = draw_view_proj.inverse();
                    // The weather (§13): the clock moves with the simulation
                    // (not while paused, nor in a bench, which must repeat),
                    // and this frame's atmosphere and light follow.
                    if self.bench.is_none() {
                        self.settings.weather.advance(steps as f32 * FIXED_DT);
                    }
                    let (env, light) = self
                        .settings
                        .weather
                        .frame(&s.environment, &s.level_weather.path);
                    self.light_dir = light.extend(0.0);
                    self.exposure_range = (env.exposure_min, env.exposure_max);
                    s.mesh.set_atmosphere(env.atmosphere());
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
                    let lights = extract_lights(&mut s.world, &camera_frustum, alpha);

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

                    // Ropes (§15): each segment a stretched unit cylinder, then
                    // the item, all interpolated like the moving entities.
                    let rope_fit = fits[MESH_ROPE as usize];
                    let rope_mat = PALETTE + MESH_ROPE;
                    let mut qh = s.world.query::<&Hanging>();
                    for h in qh.iter(&s.world) {
                        let points = h.rope.interpolated(alpha);
                        let radius = h.rope.params.radius;
                        let segments = points.windows(2).map(|w| {
                            let model = rope::segment_matrix(w[0], w[1], radius) * rope_fit;
                            (MESH_ROPE as usize, model, rope_mat)
                        });
                        let item = h.item.map(|(mesh, material, local)| {
                            let id = (mesh.0 as usize).min(mesh_max);
                            (id, rope::item_matrix(&points, local) * fits[id], material)
                        });
                        for (id, model, material) in segments.chain(item) {
                            let item = (MeshId(id as u32), InstanceData::new(model, material));
                            let (c, radius) = world_sphere(&model, spheres[id]);
                            if camera_frustum.contains_sphere(c, radius) {
                                main_items.push(item);
                            }
                            if casts && h.casts {
                                for (ci, f) in light_frusta.iter().enumerate() {
                                    if f.contains_sphere(c, radius) {
                                        shadow_items[ci].push(item);
                                    }
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
                        view_proj: draw_view_proj,
                        inv_view_proj,
                        reproject,
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
                // Bloom and auto-exposure (§13) only over a world, and only
                // when they're on.
                let bloom_on = self.settings.bloom && self.session.is_some();
                let auto_on = self.settings.auto_exposure && self.session.is_some();
                // GTAO (§13) likewise, with the camera it reconstructs from.
                let ao_on = self.settings.ambient_occlusion && self.session.is_some();
                let fov_y = self.settings.fov_y();
                let exposure_params = ExposureParams {
                    dt: dt.min(0.1),
                    min: self.exposure_range.0,
                    max: self.exposure_range.1,
                    reset: self.exposure_reset,
                };
                if auto_on {
                    self.exposure_reset = false;
                }
                let mut metered = None;
                if let (
                    Some(r),
                    Some(tm),
                    Some(bp),
                    Some(ep),
                    Some(ap),
                    Some(fx),
                    Some(tp),
                    Some(ui),
                ) = (
                    self.renderer.as_mut(),
                    self.tonemap.as_mut(),
                    self.bloom.as_mut(),
                    self.exposure_pass.as_mut(),
                    self.ao_pass.as_mut(),
                    self.fxaa.as_mut(),
                    self.taa.as_mut(),
                    self.ui.as_ref(),
                ) {
                    let exposure_state = ep.state_buffer();
                    // HDR view/sampler are stable except across resize; capture
                    // before the mutable draw_frame borrow, refresh in `update`.
                    // No session, or shadows off: the cascade passes collapse to
                    // one layered clear rather than four empty passes.
                    r.set_shadow_casters(self.session.is_some() && self.settings.shadows.casts());
                    // TAA runs over a level only; the menu's frame is a
                    // clear. Turning it on starts a new history, so a new
                    // session begins without the last one's.
                    r.set_taa(self.settings.taa && self.session.is_some());
                    let taa_views = r.taa_views();
                    let scene_hdr_view = r.scene_hdr_view();
                    let taa_push = TaaPush {
                        reproject: frame_view.map_or(Mat4::IDENTITY, |v| v.reproject),
                        history_valid: r.taa_history_valid(),
                        exposure,
                        auto_exposure: auto_on,
                    };
                    let hdr_view = r.hdr_view();
                    let ldr_view = r.ldr_view();
                    let hdr_sampler = r.hdr_sampler();
                    let (bloom_views, bloom_extents) = {
                        let (v, e) = r.bloom_mips();
                        (v.to_vec(), e.to_vec())
                    };
                    // GTAO's inputs and output move when the targets are
                    // recreated; the generation says when.
                    let depth_view = r.depth_sample_view();
                    let ao_views = r.ao_views();
                    let ao_depth_views = r.ao_depth_views();
                    let nearest = r.nearest_sampler();
                    let generation = r.targets_generation();
                    if let Some(s) = self.session.as_mut() {
                        s.mesh.set_ao(
                            ao_on,
                            (ao_views[1], ao_depth_views.0),
                            nearest,
                            r.extent(),
                            generation,
                        );
                    }
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
                        // Depth prepass (§10), on its own so GTAO can read it.
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
                            }
                        },
                        // GTAO (§13), from the prepass's depth.
                        |cmd, extent, frame| {
                            if ao_on {
                                ap.update(
                                    frame,
                                    generation,
                                    depth_view,
                                    ao_views,
                                    ao_depth_views,
                                    nearest,
                                );
                                let proj = AoProjection {
                                    near: CAMERA_NEAR,
                                    far: CAMERA_FAR,
                                    fov_y,
                                    aspect: extent.width as f32 / extent.height.max(1) as f32,
                                };
                                ap.dispatch(cmd, frame, extent, proj);
                            }
                        },
                        // Geometry: the lit opaque pass over the prepass's depth
                        // (each pixel shaded once), then the sky depth-tested into
                        // whatever background is left. Skipped wholesale in the
                        // main menu, leaving the attachment's clear colour.
                        |cmd, extent, frame| {
                            if let (Some(s), Some(v)) = (session, frame_view) {
                                s.mesh.draw_main(
                                    cmd,
                                    extent,
                                    frame,
                                    v.view_proj,
                                    v.light_dir,
                                    v.camera_pos,
                                );
                                s.sky.draw(
                                    cmd,
                                    extent,
                                    s.mesh.frame_set(frame),
                                    v.inv_view_proj,
                                    v.camera_pos,
                                    v.light_dir,
                                );
                            }
                        },
                        // TAA (§13): only invoked while it's on.
                        |cmd, extent, frame| {
                            tp.update(
                                frame,
                                scene_hdr_view,
                                depth_view,
                                taa_views.0,
                                taa_views.1,
                                exposure_state,
                                hdr_sampler,
                                nearest,
                            );
                            tp.dispatch(cmd, frame, extent, &taa_push);
                        },
                        // Bloom (§13): down the chain and back up, before the
                        // tonemap mixes it in.
                        |cmd, extent, frame| {
                            if bloom_on {
                                bp.update(frame, hdr_view, &bloom_views, hdr_sampler);
                                bp.dispatch(cmd, frame, extent, &bloom_extents);
                            }
                        },
                        // Auto-exposure (§13). First the result this frame
                        // slot's last run left (its fence has been waited
                        // on), for the bench; then this frame's metering.
                        |cmd, extent, frame| {
                            if auto_on {
                                metered = Some(ep.last(frame));
                                ep.update(frame, hdr_view, hdr_sampler);
                                ep.dispatch(cmd, frame, extent, exposure_params);
                            }
                        },
                        |cmd, extent, frame| {
                            tm.update(frame, hdr_view, bloom_views[0], exposure_state, hdr_sampler);
                            let strength = if bloom_on { BLOOM_STRENGTH } else { 0.0 };
                            tm.draw(
                                cmd,
                                extent,
                                frame,
                                exposure,
                                strength,
                                bloom_views.len(),
                                auto_on,
                            );
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

                if let (Some(b), Some(m)) = (self.bench.as_mut(), metered) {
                    b.exposure = Some(m.exposure);
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

/// The command line: flags override `settings` for this run (never saved);
/// anything else is a scene to load. Returns the scenes and whether `--bench`
/// was given.
fn parse_args(
    args: impl IntoIterator<Item = String>,
    settings: &mut GraphicsSettings,
) -> (Vec<String>, bool) {
    let mut scenes: Vec<String> = Vec::new();
    let mut bench = false;
    let mut args = args.into_iter();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--msaa" => match args.next().and_then(|v| v.parse::<u32>().ok()) {
                Some(n) => settings.msaa = n,
                None => eprintln!("--msaa needs a sample count (1/2/4/8); ignoring"),
            },
            "--bench" => bench = true,
            "--no-bake" => settings.bake = false,
            "--no-lod" => settings.lod = false,
            // For A/B timing: --bench ignores config/, so this is how a
            // bench run turns these off.
            "--no-bloom" => settings.bloom = false,
            "--no-taa" => settings.taa = false,
            "--no-auto-exposure" => settings.auto_exposure = false,
            "--no-ao" => settings.ambient_occlusion = false,
            "--no-sky-occlusion" => settings.sky_occlusion = false,
            _ => scenes.push(a),
        }
    }
    (scenes, bench)
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
    let (scenes, bench) = parse_args(std::env::args().skip(1), &mut settings);

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    eprintln!(
        "[config] effective: display {}, fov {}, shadows {}, msaa {}, fxaa {}, taa {}, bloom {}, auto exposure {}, ambient occlusion {}",
        settings.display.config_name(),
        settings.fov_deg,
        settings.shadows.config_name(),
        settings.msaa,
        settings.fxaa,
        settings.taa,
        settings.bloom,
        settings.auto_exposure,
        settings.ambient_occlusion
    );
    let mut app = App::new(scenes, settings, configs, audio, bench);
    event_loop.run_app(&mut app).expect("run app");
}

#[cfg(test)]
mod tests {
    use super::*;
    use feather_assets::MeshData;
    use feather_game::components::rand01;
    use feather_game::controller::GROUND_Y;
    use feather_game::level::{demo_boxes, WorldBuild};
    use feather_game::physics::{to_rapier, Physics};
    use feather_game::prefab::ColliderStats;
    use feather_game::Surface;
    use rapier3d::prelude::QueryFilter;

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

    const SCREENS: [MenuScreen; 8] = [
        MenuScreen::MainRoot,
        MenuScreen::Root,
        MenuScreen::Options,
        MenuScreen::Graphics,
        MenuScreen::Sound,
        MenuScreen::Gameplay,
        MenuScreen::Controls,
        MenuScreen::Weather,
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

    /// WEATHER is under OPTIONS in game only: a new game resets it.
    #[test]
    fn weather_is_in_the_pause_menus_options_only() {
        let s = GraphicsSettings::default();
        let options = |in_session| {
            screen_rows(
                MenuScreen::Options,
                &s,
                &Controls::default(),
                &AudioSettings::default(),
                in_session,
                None,
            )
        };
        let weather = MenuAction::Enter(MenuScreen::Weather);
        assert!(options(true).iter().any(|r| r.action == weather));
        assert!(!options(false).iter().any(|r| r.action == weather));
        assert_eq!(options(true).last().unwrap().action, MenuAction::Back);
    }

    /// Each WEATHER row steps through its choices and wraps, and says so. A
    /// clock under LEVEL moves the sun only, and the row says that too.
    #[test]
    fn weather_rows_cycle_their_presets() {
        let mut s = GraphicsSettings::default();
        s.weather = weather::SessionWeather::new(&weather::LevelWeather::default(), Vec3::NEG_Y);
        let mut m = in_game_menu();
        m.screen = MenuScreen::Weather;
        let label = |m: &Menu, s: &GraphicsSettings, a| label_of(m, s, &Controls::default(), a);
        assert_eq!(label(&m, &s, MenuAction::CycleWeather), "WEATHER  LEVEL");
        assert_eq!(label(&m, &s, MenuAction::StepTime), "TIME  LEVEL");
        assert_eq!(label(&m, &s, MenuAction::CycleSpeed), "SPEED  10X");
        for (action, len) in [
            (MenuAction::CycleWeather, weather::WEATHERS.len() + 1),
            (MenuAction::CycleSpeed, weather::SPEEDS.len()),
            (MenuAction::CycleFogDensity, weather::FOG_DENSITIES.len()),
            (MenuAction::CycleFogHeight, weather::FOG_HEIGHTS.len()),
        ] {
            let first = label(&m, &s, action);
            let mut seen = Vec::new();
            for _ in 0..len {
                assert_eq!(activate(&mut m, &mut s, action), MenuOutcome::ApplyWeather);
                seen.push(label(&m, &s, action));
            }
            assert_eq!(
                seen.last(),
                Some(&first),
                "{action:?} didn't wrap: {seen:?}"
            );
            seen.sort();
            seen.dedup();
            assert_eq!(seen.len(), len, "{action:?} repeated a choice");
        }
        // A weather under a sun straight up starts the clock at noon; TIME
        // steps it an hour a press, and back under LEVEL it moves the sun only.
        activate(&mut m, &mut s, MenuAction::CycleWeather);
        assert_eq!(label(&m, &s, MenuAction::CycleWeather), "WEATHER  CLEAR");
        assert_eq!(label(&m, &s, MenuAction::StepTime), "TIME  12 00");
        activate(&mut m, &mut s, MenuAction::StepTime);
        assert_eq!(label(&m, &s, MenuAction::StepTime), "TIME  13 00");
        for _ in 0..weather::WEATHERS.len() {
            activate(&mut m, &mut s, MenuAction::CycleWeather);
        }
        assert_eq!(label(&m, &s, MenuAction::CycleWeather), "WEATHER  LEVEL");
        assert_eq!(label(&m, &s, MenuAction::StepTime), "TIME  13 00  SUN ONLY");
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

    /// Flags override the settings for one run; everything else is a scene.
    #[test]
    fn command_line_flags_and_scenes() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut s = GraphicsSettings::default();
        assert!(s.bloom && s.bake && s.lod && s.sky_occlusion && s.ambient_occlusion);
        let (scenes, bench) = parse_args(
            args(&[
                "--bench",
                "a.glb",
                "--no-bloom",
                "--msaa",
                "4",
                "--no-lod",
                "b.gltf",
                "--no-auto-exposure",
                "--no-sky-occlusion",
                "--no-ao",
                "--no-taa",
            ]),
            &mut s,
        );
        assert_eq!(scenes, ["a.glb", "b.gltf"]);
        assert!(bench);
        assert_eq!((s.bloom, s.msaa, s.lod, s.bake), (false, 4, false, true));
        assert!(!s.auto_exposure && !s.sky_occlusion && !s.ambient_occlusion && !s.taa);
        // A bad sample count is ignored, not taken as a scene.
        let mut s = GraphicsSettings::default();
        let (scenes, bench) = parse_args(args(&["--msaa", "x", "--no-bake"]), &mut s);
        assert!(scenes.is_empty() && !bench);
        assert_eq!((s.msaa, s.bake, s.bloom, s.taa), (1, false, true, true));
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

        // TAA is on out of the box (§13), and its row turns it off.
        assert!(s.taa);
        assert_eq!(
            activate(&mut m, &mut s, MenuAction::ToggleTaa),
            MenuOutcome::ApplyTaa
        );
        assert!(!s.taa, "TAA row did not toggle the setting");
        assert_eq!(
            label_of(&m, &s, &Controls::default(), MenuAction::ToggleTaa),
            "TAA  OFF"
        );

        let before = s.bloom;
        assert_eq!(
            activate(&mut m, &mut s, MenuAction::ToggleBloom),
            MenuOutcome::ApplyBloom
        );
        assert_eq!(s.bloom, !before, "BLOOM row did not toggle the setting");
        assert!(
            label_of(&m, &s, &Controls::default(), MenuAction::ToggleBloom).starts_with("BLOOM")
        );

        let before = s.auto_exposure;
        assert_eq!(
            activate(&mut m, &mut s, MenuAction::ToggleAutoExposure),
            MenuOutcome::ApplyAutoExposure
        );
        assert_eq!(
            s.auto_exposure, !before,
            "AUTO EXPOSURE row did not toggle it"
        );

        let before = s.ambient_occlusion;
        assert_eq!(
            activate(&mut m, &mut s, MenuAction::ToggleAmbientOcclusion),
            MenuOutcome::ApplyAmbientOcclusion
        );
        assert_eq!(
            s.ambient_occlusion, !before,
            "AMBIENT OCCLUSION row did not toggle it"
        );

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

    // ---- Shadow cascades (§11) with the camera: render's cascade maths fed
    // by `Look`; the pure ones are in render/src/cascades.rs ----

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

    /// A level's baked sky visibility (§13) reaches the session: found by the
    /// scene's own key under the bake root, and only there. A volume baked
    /// for another version of the level (a moved pad) isn't used, and nor is
    /// anything with the bake off.
    #[test]
    fn a_levels_sky_visibility_reaches_the_world_build() {
        use feather_assets::bake::{baked_sky_path, sky_key, SkyVis, SkyVolume, SKY_DIR};
        let dir = crate::config::test_dir("e2e-sky");
        let bake = dir.join("bake");
        std::fs::create_dir_all(bake.join(SKY_DIR)).unwrap();
        let level = write_pad_level(&dir, &[("plain", None)]);
        let volume = SkyVolume {
            origin: Vec3::new(-1.0, -2.0, -3.0),
            cell: 0.5,
            dims: [2, 1, 1],
            texels: vec![[SkyVis::OPEN.encode(), [0; 4]].concat().try_into().unwrap(); 2],
        };
        let key = sky_key(&feather_assets::load_gltf_scene(&level).unwrap());
        assert!(build_world(std::slice::from_ref(&level), Some(&bake))
            .sky
            .is_none());
        volume
            .write(&baked_sky_path(&bake.join(SKY_DIR), key))
            .unwrap();
        let b = build_world(std::slice::from_ref(&level), Some(&bake));
        assert_eq!(b.sky, Some(volume));
        assert!(build_world(std::slice::from_ref(&level), None)
            .sky
            .is_none());
        // With several scenes, the first one's volume is the session's.
        let other_dir = dir.join("other");
        std::fs::create_dir_all(&other_dir).unwrap();
        let other = write_pad_level(&other_dir, &[("plain", None), ("more", None)]);
        let two = |a: &String, b: &String| build_world(&[a.clone(), b.clone()], Some(&bake)).sky;
        assert!(two(&level, &other).is_some());
        assert!(two(&other, &level).is_none());

        // Move the pad: a different level, so the old volume isn't its.
        let mut doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&level).unwrap()).unwrap();
        doc["nodes"][0]["translation"][0] = 99.0.into();
        std::fs::write(&level, doc.to_string()).unwrap();
        assert!(build_world(&[level], Some(&bake)).sky.is_none());
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

    /// The light a lantern carries is where the lantern is drawn: at rest,
    /// the node's transform applied to `at`; swung, carried with the item.
    #[test]
    fn a_hanging_light_rides_its_item() {
        let look = Look::new();
        let vp = look.view_proj(Vec3::ZERO, 16.0 / 9.0, GraphicsSettings::default().fov_y());
        let frustum = Frustum::from_view_proj(&vp);
        let at = Vec3::new(0.0, -0.17, 0.0);
        let local = Mat4::from_scale(Vec3::splat(1.3));
        let node = Vec3::new(1.0, 2.0, -10.0);
        let hang = |lit: bool| Hanging {
            rope: rope::Rope::new(node + Vec3::Y * 1.5, rope::RopeParams::default()),
            item: Some((MeshId(0), 0, local)),
            casts: true,
            light: lit.then_some((PointLight::default(), at)),
        };
        let mut w = World::new();
        w.spawn(hang(true));
        let lights = extract_lights(&mut w, &frustum, 1.0);
        assert_eq!(lights.len(), 1);
        let want = node + local.transform_vector3(at);
        let got = Vec3::from_slice(&lights[0].pos_radius[..3]);
        assert!((got - want).length() < 1e-5, "{got} vs {want}");
        // Swing it: the light follows the item's matrix, not the node.
        let mut w = World::new();
        let mut h = hang(true);
        for p in h.rope.points.iter_mut().skip(1) {
            *p += Vec3::new(0.4, 0.0, 0.0);
        }
        let item = rope::item_matrix(&h.rope.points, local);
        w.spawn(h);
        let got = Vec3::from_slice(&extract_lights(&mut w, &frustum, 1.0)[0].pos_radius[..3]);
        assert!((got - item.transform_point3(at)).length() < 1e-5);
        assert!(
            (got - want).x > 0.3,
            "it didn't move with the lantern: {got}"
        );
        // Control: unlit, no light.
        let mut w = World::new();
        w.spawn(hang(false));
        assert!(extract_lights(&mut w, &frustum, 1.0).is_empty());
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

        let lights = extract_lights(&mut w, &frustum, 1.0);
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
}
