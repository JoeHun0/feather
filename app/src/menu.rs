//! The pause and main menus (§19): screens, rows and what activating one
//! does, and their layout and hit-testing. No renderer and no event loop:
//! the app draws `Menu`'s rows and feeds it keys and clicks.

use crate::audio::AudioSettings;
use crate::config::controls::{Action, Controls};
use crate::{audio, config, GraphicsSettings};
use feather_game::weather;
use feather_platform::winit::keyboard::KeyCode;
use feather_render::Font;

/// One screen of the pause menu (§19). Screens form a tree rooted at `Root`;
/// `Menu` walks it with an explicit stack.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MenuScreen {
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
    pub fn title(self) -> &'static str {
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
pub enum MenuAction {
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
pub struct MenuRow {
    pub label: String,
    pub action: MenuAction,
}

impl MenuRow {
    pub fn new(label: impl Into<String>, action: MenuAction) -> Self {
        Self {
            label: label.into(),
            action,
        }
    }

    /// Inert rows draw dimmer. They stay *selectable* so arrow navigation does
    /// not silently skip the information they carry.
    pub fn enabled(&self) -> bool {
        !matches!(self.action, MenuAction::Inert)
    }
}

/// What the caller must do after the menu handled an input. Returning this
/// rather than acting directly is what keeps `Menu` free of any dependency on
/// the renderer or the event loop — and therefore unit-testable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MenuOutcome {
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
pub enum ControlsSave {
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
pub fn screen_rows(
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
pub struct Menu {
    pub screen: MenuScreen,
    pub index: usize,
    /// `(screen, index)` of each ancestor, so BACK restores the row you
    /// descended from instead of snapping to the top.
    pub stack: Vec<(MenuScreen, usize)>,
    /// The action waiting for a key on the CONTROLS screen (`capture_key`).
    pub capturing: Option<Action>,
    /// First visible row when a screen is taller than the window
    /// (`menu_layout`). Kept here so it only moves to follow the selection:
    /// hovering a visible row never scrolls the list under the mouse.
    pub scroll: usize,
}

impl Menu {
    pub fn new() -> Self {
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
    pub fn reset(&mut self, root: MenuScreen) {
        self.screen = root;
        self.index = 0;
        self.stack.clear();
        self.capturing = None;
        self.scroll = 0;
    }

    pub fn rows(
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
    pub fn select(&mut self, index: usize) {
        if index != self.index {
            self.capturing = None;
        }
        self.index = index;
    }

    /// Move the selection, wrapping at both ends.
    pub fn move_by(
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
    pub fn hover(
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

    pub fn descend(&mut self, screen: MenuScreen) {
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
    pub fn capture_key(&mut self, c: &mut Controls, code: KeyCode) -> MenuOutcome {
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
    pub fn back(&mut self) -> MenuOutcome {
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

    pub fn activate(
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

/// Font size (em, in screen px) for the menu at a given framebuffer height.
/// One place, so the layout and the renderer cannot disagree about how big
/// the text is.
pub fn menu_font_size(h: f32) -> f32 {
    (h / 36.0).max(10.0)
}

/// Where one screen of the menu goes, in **physical** pixels.
///
/// Single source of truth: the redraw handler draws the title, the rows and
/// the highlight bar from it and the mouse hit-tests against it, so the
/// visible target and the clickable target can never drift apart. Row rects
/// are the padded bar rather than the tight text box, which also makes a
/// comfortably larger click target than the glyphs alone.
pub struct MenuLayout {
    /// Font size (em, screen px) the rows are laid out at.
    pub size: f32,
    pub title_y: f32,
    /// Index of the first visible row.
    pub first: usize,
    /// `(x, y, w, h)` of each *visible* row, from `first` on.
    pub rects: Vec<(f32, f32, f32, f32)>,
    /// Rows hidden above / below the visible window.
    pub more_above: bool,
    pub more_below: bool,
}

/// Lay out a screen: the title, then as many rows as fit, the two centred
/// together. A screen taller than the window scrolls: the visible window
/// starts at `scroll`, moved only as far as needed to keep row `index` in it.
/// The caller stores `first` back as the new `scroll`, so the list moves when
/// the selection leaves it and not otherwise.
pub fn menu_layout(
    font: &Font,
    w: f32,
    h: f32,
    rows: &[MenuRow],
    index: usize,
    scroll: usize,
) -> MenuLayout {
    let size = menu_font_size(h);
    let text_h = font.height(size);
    let line = text_h * 2.2;
    let pad = size * 0.55;
    let title_h = font.height(size * 1.6);
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
            let tw = font.width(&row.label, size);
            let x = (w - tw) * 0.5;
            let y = top + line * i as f32;
            (x - pad, y - pad * 0.5, tw + pad * 2.0, text_h + pad)
        })
        .collect();
    MenuLayout {
        size,
        title_y,
        first,
        rects,
        more_above: first > 0,
        more_below: first + visible < n,
    }
}

/// Index of the row under `(cx, cy)`, if any. Physical pixels, matching both
/// `Window::inner_size` and winit's `CursorMoved` position.
pub fn menu_hit(layout: &MenuLayout, cx: f32, cy: f32) -> Option<usize> {
    layout
        .rects
        .iter()
        .position(|&(x, y, rw, rh)| cx >= x && cx < x + rw && cy >= y && cy < y + rh)
        .map(|i| layout.first + i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DisplayMode, ShadowQuality};
    use feather_game::controller::FOV_PRESETS;
    use glam::Vec3;

    // ---- Pause menu: layout, hit-testing and navigation (§19) ----
    //
    // `Menu` deliberately depends on neither the renderer nor the event loop,
    // and the layout is a pure function of the framebuffer size, so all of this
    // runs without a GPU or a window.

    /// A few sizes worth covering: the default window, a typical one, a wide
    /// one, and one small enough that `menu_font_size` clamps to its floor.
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

    /// The same font the renderer draws with (menu_layout sizes from it).
    fn font() -> Font {
        Font::rasterize(include_bytes!("../../render/fonts/DejaVuSansMono.ttf"), 18)
            .expect("the bundled font rasterizes")
    }

    /// Every screen, every size, every selected row: the layout keeps what it
    /// shows on screen, ordered, disjoint and centred, and the selected row is
    /// always among the visible ones. CONTROLS (13 rows) has to scroll at the
    /// small sizes; that is what this pins down.
    #[test]
    fn menu_layout_keeps_rows_onscreen_and_the_selection_visible() {
        for screen in SCREENS {
            let rows = rows_of(screen);
            let font = font();
            assert!(!rows.is_empty(), "{screen:?} has no rows");
            for (w, h) in SIZES {
                for index in 0..rows.len() {
                    let l = menu_layout(&font, w, h, &rows, index, 0);
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
                    let title_bottom = l.title_y + font.height(l.size * 1.6);
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
        let font = font();
        let cap = menu_layout(&font, w, h, &rows, 0, 0).rects.len();
        assert!(cap < rows.len(), "CONTROLS should scroll at {w}x{h}");
        // Stepping down scrolls one row at a time once past the window...
        let mut scroll = 0;
        for index in 0..rows.len() {
            let l = menu_layout(&font, w, h, &rows, index, scroll);
            assert_eq!(l.first, index.saturating_sub(cap - 1), "row {index}");
            scroll = l.first;
        }
        // ...and selecting any visible row (the mouse) doesn't move it.
        for index in scroll..scroll + cap {
            assert_eq!(menu_layout(&font, w, h, &rows, index, scroll).first, scroll);
        }
        // A stale scroll past the end is clamped.
        let l = menu_layout(&font, w, h, &rows, rows.len() - 1, 999);
        assert_eq!(l.first + l.rects.len(), rows.len());
    }

    #[test]
    fn menu_rect_centres_hit_their_own_row() {
        for screen in SCREENS {
            let rows = rows_of(screen);
            let font = font();
            for (w, h) in SIZES {
                // Scrolled to the bottom, so hits must add `first` back.
                let l = menu_layout(&font, w, h, &rows, rows.len() - 1, 0);
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
        let l = menu_layout(&font(), w, h, &rows, 0, 0);
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
            (MenuAction::CycleWeather, weather::tables().len() + 1),
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
        for _ in 0..weather::tables().len() {
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

    /// Menu labels must stay inside what the UI font can draw (printable
    /// ASCII); anything else renders as nothing, which would silently mangle
    /// a row. Guards against someone adding an em dash or «» later.
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
                        c.is_ascii_graphic() || c == ' ',
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
}
