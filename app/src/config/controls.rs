//! Key bindings and mouse look (§14): `config/controls.toml`.
//!
//! The bindings table §14 plans: physical keys → abstract actions, so gameplay
//! code never names a key. Created with defaults, hand-editable, and saved one
//! value at a time by the OPTIONS > CONTROLS and GAMEPLAY menus (§19), the same
//! in-place way as `graphics.toml`.
//!
//! The menu keys (Escape, Up, Down, Enter) are not bindable, so neither a bad
//! file nor a slip in the rebinding screen can lock the player out of the menu.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use feather_platform::winit::keyboard::KeyCode;

use super::{entries, open_or_create, string_list, warning, ConfigFile};

/// Relative to the working directory, like `graphics::CONFIG_PATH`.
pub const CONTROLS_PATH: &str = "config/controls.toml";

/// Look speed at `sensitivity = 1.0`, in radians per raw mouse count (the
/// value that was hard-coded before this file existed).
pub const BASE_SENSITIVITY: f32 = 0.0025;
const MAX_SENSITIVITY: f32 = 20.0;
/// What the GAMEPLAY menu's SENSITIVITY row steps through, in percent (the
/// font has no decimal point, so the row shows a percentage).
const SENSITIVITY_PRESETS: [u32; 9] = [25, 50, 75, 100, 125, 150, 200, 300, 400];

/// Something a key can be bound to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Action {
    Forward,
    Back,
    Left,
    Right,
    /// Jump on the press; also "fly up" while held in noclip.
    Jump,
    /// Fly down in noclip.
    Down,
    Noclip,
    ExposureDown,
    ExposureUp,
    CycleShadows,
    ToggleFxaa,
    ToggleFullscreen,
}

impl Action {
    /// File order.
    pub const ALL: [Self; 12] = [
        Self::Forward,
        Self::Back,
        Self::Left,
        Self::Right,
        Self::Jump,
        Self::Down,
        Self::Noclip,
        Self::ExposureDown,
        Self::ExposureUp,
        Self::CycleShadows,
        Self::ToggleFxaa,
        Self::ToggleFullscreen,
    ];

    /// The key it's saved under in `controls.toml`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Back => "back",
            Self::Left => "left",
            Self::Right => "right",
            Self::Jump => "jump",
            Self::Down => "down",
            Self::Noclip => "noclip",
            Self::ExposureDown => "exposure_down",
            Self::ExposureUp => "exposure_up",
            Self::CycleShadows => "cycle_shadows",
            Self::ToggleFxaa => "toggle_fxaa",
            Self::ToggleFullscreen => "toggle_fullscreen",
        }
    }

    fn default_key(self) -> KeyCode {
        match self {
            Self::Forward => KeyCode::KeyW,
            Self::Back => KeyCode::KeyS,
            Self::Left => KeyCode::KeyA,
            Self::Right => KeyCode::KeyD,
            Self::Jump => KeyCode::Space,
            Self::Down => KeyCode::ControlLeft,
            Self::Noclip => KeyCode::KeyV,
            Self::ExposureDown => KeyCode::BracketLeft,
            Self::ExposureUp => KeyCode::BracketRight,
            Self::CycleShadows => KeyCode::F1,
            Self::ToggleFxaa => KeyCode::F2,
            Self::ToggleFullscreen => KeyCode::F11,
        }
    }

    /// What the template says about it.
    fn describe(self) -> &'static str {
        match self {
            Self::Forward => "move forward",
            Self::Back => "move back",
            Self::Left => "strafe left",
            Self::Right => "strafe right",
            Self::Jump => "jump; fly up in noclip",
            Self::Down => "fly down in noclip",
            Self::Noclip => "noclip (free flight) on/off",
            Self::ExposureDown => "exposure down (repeats while held)",
            Self::ExposureUp => "exposure up (repeats while held)",
            Self::CycleShadows => "cycle shadow quality",
            Self::ToggleFxaa => "FXAA on/off",
            Self::ToggleFullscreen => "windowed / fullscreen",
        }
    }

    /// Menu form: upper case, underscores as spaces (the 5x7 font has A-Z,
    /// 0-9 and space only).
    pub fn menu_label(self) -> String {
        self.name().to_ascii_uppercase().replace('_', " ")
    }

    /// Fires again on key auto-repeat. Everything else fires once, when the
    /// action becomes active: holding V must not flicker noclip.
    fn repeats(self) -> bool {
        matches!(self, Self::ExposureDown | Self::ExposureUp)
    }
}

/// Owned by the menu (§19), never bindable.
const RESERVED: [KeyCode; 4] = [
    KeyCode::Escape,
    KeyCode::ArrowUp,
    KeyCode::ArrowDown,
    KeyCode::Enter,
];

/// Physical keys (layout-independent, §14), named by their US-QWERTY position.
/// Punctuation is spelled out, which keeps `"\"` and `"'"` out of the file.
/// The reserved keys are named too, so binding one gets a precise warning.
const KEY_NAMES: &[(&str, KeyCode)] = &[
    ("A", KeyCode::KeyA),
    ("B", KeyCode::KeyB),
    ("C", KeyCode::KeyC),
    ("D", KeyCode::KeyD),
    ("E", KeyCode::KeyE),
    ("F", KeyCode::KeyF),
    ("G", KeyCode::KeyG),
    ("H", KeyCode::KeyH),
    ("I", KeyCode::KeyI),
    ("J", KeyCode::KeyJ),
    ("K", KeyCode::KeyK),
    ("L", KeyCode::KeyL),
    ("M", KeyCode::KeyM),
    ("N", KeyCode::KeyN),
    ("O", KeyCode::KeyO),
    ("P", KeyCode::KeyP),
    ("Q", KeyCode::KeyQ),
    ("R", KeyCode::KeyR),
    ("S", KeyCode::KeyS),
    ("T", KeyCode::KeyT),
    ("U", KeyCode::KeyU),
    ("V", KeyCode::KeyV),
    ("W", KeyCode::KeyW),
    ("X", KeyCode::KeyX),
    ("Y", KeyCode::KeyY),
    ("Z", KeyCode::KeyZ),
    ("0", KeyCode::Digit0),
    ("1", KeyCode::Digit1),
    ("2", KeyCode::Digit2),
    ("3", KeyCode::Digit3),
    ("4", KeyCode::Digit4),
    ("5", KeyCode::Digit5),
    ("6", KeyCode::Digit6),
    ("7", KeyCode::Digit7),
    ("8", KeyCode::Digit8),
    ("9", KeyCode::Digit9),
    ("F1", KeyCode::F1),
    ("F2", KeyCode::F2),
    ("F3", KeyCode::F3),
    ("F4", KeyCode::F4),
    ("F5", KeyCode::F5),
    ("F6", KeyCode::F6),
    ("F7", KeyCode::F7),
    ("F8", KeyCode::F8),
    ("F9", KeyCode::F9),
    ("F10", KeyCode::F10),
    ("F11", KeyCode::F11),
    ("F12", KeyCode::F12),
    ("Space", KeyCode::Space),
    ("Tab", KeyCode::Tab),
    ("Backspace", KeyCode::Backspace),
    ("CapsLock", KeyCode::CapsLock),
    ("LeftShift", KeyCode::ShiftLeft),
    ("RightShift", KeyCode::ShiftRight),
    ("LeftCtrl", KeyCode::ControlLeft),
    ("RightCtrl", KeyCode::ControlRight),
    ("LeftAlt", KeyCode::AltLeft),
    ("RightAlt", KeyCode::AltRight),
    ("Left", KeyCode::ArrowLeft),
    ("Right", KeyCode::ArrowRight),
    ("Insert", KeyCode::Insert),
    ("Delete", KeyCode::Delete),
    ("Home", KeyCode::Home),
    ("End", KeyCode::End),
    ("PageUp", KeyCode::PageUp),
    ("PageDown", KeyCode::PageDown),
    ("LeftBracket", KeyCode::BracketLeft),
    ("RightBracket", KeyCode::BracketRight),
    ("Semicolon", KeyCode::Semicolon),
    ("Quote", KeyCode::Quote),
    ("Comma", KeyCode::Comma),
    ("Period", KeyCode::Period),
    ("Slash", KeyCode::Slash),
    ("Backslash", KeyCode::Backslash),
    ("Minus", KeyCode::Minus),
    ("Equal", KeyCode::Equal),
    ("Backquote", KeyCode::Backquote),
    ("Numpad0", KeyCode::Numpad0),
    ("Numpad1", KeyCode::Numpad1),
    ("Numpad2", KeyCode::Numpad2),
    ("Numpad3", KeyCode::Numpad3),
    ("Numpad4", KeyCode::Numpad4),
    ("Numpad5", KeyCode::Numpad5),
    ("Numpad6", KeyCode::Numpad6),
    ("Numpad7", KeyCode::Numpad7),
    ("Numpad8", KeyCode::Numpad8),
    ("Numpad9", KeyCode::Numpad9),
    ("Escape", KeyCode::Escape),
    ("Up", KeyCode::ArrowUp),
    ("Down", KeyCode::ArrowDown),
    ("Enter", KeyCode::Enter),
];

fn key_name(code: KeyCode) -> &'static str {
    KEY_NAMES
        .iter()
        .find(|(_, c)| *c == code)
        .map_or("?", |(n, _)| n)
}

/// Whether the menu owns `code`, so it can never be bound.
pub fn is_reserved(code: KeyCode) -> bool {
    RESERVED.contains(&code)
}

fn key_from_name(name: &str) -> Option<KeyCode> {
    KEY_NAMES
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, c)| *c)
}

/// Bindings plus mouse look.
#[derive(Clone, Debug, PartialEq)]
pub struct Controls {
    keys: HashMap<Action, Vec<KeyCode>>,
    /// Multiplier on `BASE_SENSITIVITY`.
    pub sensitivity: f32,
    pub invert_y: bool,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            keys: Action::ALL
                .iter()
                .map(|&a| (a, vec![a.default_key()]))
                .collect(),
            sensitivity: 1.0,
            invert_y: false,
        }
    }
}

impl Controls {
    /// Bind `action` to exactly `key`, taking `key` away from any other
    /// action (which may be left unbound). Refuses, returning `false`, a
    /// reserved menu key or one with no name, which the file couldn't store.
    pub fn rebind(&mut self, action: Action, key: KeyCode) -> bool {
        if is_reserved(key) || key_name(key) == "?" {
            return false;
        }
        for keys in self.keys.values_mut() {
            keys.retain(|&k| k != key);
        }
        self.keys.insert(action, vec![key]);
        true
    }

    /// Default bindings again; sensitivity and invert are kept.
    pub fn reset_keys(&mut self) {
        self.keys = Controls::default().keys;
    }

    /// The menu's view of an action's keys: `W`, `W I`, or `NONE`.
    pub fn keys_label(&self, a: Action) -> String {
        let keys = self.keys(a);
        if keys.is_empty() {
            return "NONE".into();
        }
        keys.iter()
            .map(|&k| key_name(k).to_ascii_uppercase())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The TOML value `controls.toml` stores for an action: `["W", "I"]`.
    pub fn keys_literal(&self, a: Action) -> String {
        let names: Vec<String> = self
            .keys(a)
            .iter()
            .map(|&k| format!("\"{}\"", key_name(k)))
            .collect();
        format!("[{}]", names.join(", "))
    }

    /// Sensitivity as the menu shows it, in percent.
    pub fn sensitivity_percent(&self) -> u32 {
        (self.sensitivity * 100.0).round() as u32
    }

    /// The next SENSITIVITY preset up, wrapping. A value that isn't a preset
    /// (typed into the file) goes to the next preset above it.
    pub fn step_sensitivity(&mut self) {
        let now = self.sensitivity_percent();
        let next = SENSITIVITY_PRESETS
            .into_iter()
            .find(|&p| p > now)
            .unwrap_or(SENSITIVITY_PRESETS[0]);
        self.sensitivity = next as f32 / 100.0;
    }

    /// Radians of look per raw mouse count.
    pub fn look_scale(&self) -> f32 {
        BASE_SENSITIVITY * self.sensitivity
    }

    fn keys(&self, a: Action) -> &[KeyCode] {
        self.keys.get(&a).map_or(&[], Vec::as_slice)
    }

    /// Whether any key bound to `a` is down.
    pub fn held(&self, held: &HashSet<KeyCode>, a: Action) -> bool {
        self.keys(a).iter().any(|k| held.contains(k))
    }

    /// Record one key event in `held` and return the actions it fires: an
    /// action fires when it becomes active (its first bound key goes down), and
    /// repeating actions also fire on auto-repeat. Held-state actions
    /// (movement) are read afterwards with `held`.
    pub fn key(
        &self,
        held: &mut HashSet<KeyCode>,
        code: KeyCode,
        pressed: bool,
        repeat: bool,
    ) -> Vec<Action> {
        let bound: Vec<Action> = Action::ALL
            .into_iter()
            .filter(|&a| self.keys(a).contains(&code))
            .collect();
        let was: Vec<bool> = bound.iter().map(|&a| self.held(held, a)).collect();
        if pressed {
            held.insert(code);
        } else {
            held.remove(&code);
        }
        bound
            .into_iter()
            .zip(was)
            .filter(|&(a, was)| {
                pressed && ((!was && self.held(held, a)) || (repeat && a.repeats()))
            })
            .map(|(a, _)| a)
            .collect()
    }
}

/// The file written when none exists, generated from `Controls::default()` so
/// it cannot drift from the code.
pub fn default_text() -> String {
    let d = Controls::default();
    let mut t = String::from(
        "# Feather controls. Written with defaults when missing; the OPTIONS >
# CONTROLS and GAMEPLAY menus save changes here. Delete it to reset. Graphics
# settings are in graphics.toml.
#
# action = [\"Key\", ...]: one or more keys, or [] to unbind. Keys are physical
# positions named as on a US QWERTY keyboard, whatever your layout:
#   A-Z  0-9  F1-F12  Space Tab Backspace CapsLock  LeftShift RightShift
#   LeftCtrl RightCtrl LeftAlt RightAlt  Left Right (arrows)  Insert Delete
#   Home End PageUp PageDown  Numpad0-Numpad9  LeftBracket RightBracket
#   Semicolon Quote Comma Period Slash Backslash Minus Equal Backquote
# Escape, Up, Down and Enter belong to the menu and can't be bound.

",
    );
    for a in Action::ALL {
        let keys: Vec<String> = d
            .keys(a)
            .iter()
            .map(|&k| format!("\"{}\"", key_name(k)))
            .collect();
        let line = format!("{} = [{}]", a.name(), keys.join(", "));
        t.push_str(&format!("{line:<32}# {}\n", a.describe()));
    }
    t.push_str(&format!(
        "
# Mouse look speed, a multiplier: 1.0 is the default, 2.0 twice as fast.
sensitivity = {:?}
# true: moving the mouse forward looks down.
invert_y = {}
",
        d.sensitivity, d.invert_y
    ));
    t
}

/// `Controls::default()` with every valid line of `text` applied, and one
/// warning per problem (lines ignored, or keys bound to several actions).
pub fn parse(text: &str) -> (Controls, Vec<String>) {
    let mut c = Controls::default();
    let (lines, mut warnings) = entries(text);
    for e in lines {
        let mut warn = |msg: String| warnings.push(warning(e.line, msg));
        match e.key {
            "sensitivity" => match e.value.parse::<f32>() {
                Ok(s) if s > 0.0 && s <= MAX_SENSITIVITY => c.sensitivity = s,
                _ => warn(format!(
                    "sensitivity must be a number above 0 and at most {MAX_SENSITIVITY}, got {}",
                    e.value
                )),
            },
            "invert_y" => match e.value {
                "true" => c.invert_y = true,
                "false" => c.invert_y = false,
                v => warn(format!("invert_y must be true or false, got {v}")),
            },
            name => {
                let Some(action) = Action::ALL.into_iter().find(|a| a.name() == name) else {
                    warn(format!("unknown setting `{name}`"));
                    continue;
                };
                let Some(names) = string_list(e.value) else {
                    warn(format!(
                        "{name} must be a list of quoted key names like [\"W\"], got {}",
                        e.value
                    ));
                    continue;
                };
                // All or nothing: a half-applied line is harder to notice.
                let mut keys = Vec::new();
                let mut ok = true;
                for n in names {
                    match key_from_name(n) {
                        Some(k) if RESERVED.contains(&k) => {
                            warn(format!("\"{n}\" is reserved for the menu"));
                            ok = false;
                            break;
                        }
                        Some(k) => {
                            if !keys.contains(&k) {
                                keys.push(k);
                            }
                        }
                        None => {
                            warn(format!("unknown key name \"{n}\""));
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    c.keys.insert(action, keys);
                }
            }
        }
    }
    // Checked on the result, so a default binding that collides with a new
    // one is caught too.
    for (i, &a) in Action::ALL.iter().enumerate() {
        for &b in &Action::ALL[i + 1..] {
            for k in c.keys(a).iter().filter(|k| c.keys(b).contains(k)) {
                warnings.push(format!(
                    "\"{}\" is bound to both {} and {}; it will do both",
                    key_name(*k),
                    a.name(),
                    b.name()
                ));
            }
        }
    }
    (c, warnings)
}

/// Startup entry point: read (or create) the file and log what happened.
/// Returns the file for saving, or `None` if it can be neither read nor
/// created; the controls then fall back to the defaults and nothing is saved.
pub fn load() -> (Controls, Option<ConfigFile>) {
    let path = Path::new(CONTROLS_PATH);
    match open_or_create(path, default_text) {
        Ok((text, created)) => {
            let (c, warnings) = parse(&text);
            for w in &warnings {
                eprintln!("[config] {}: {w}", path.display());
            }
            if created {
                eprintln!("[config] wrote defaults to {}", path.display());
            } else {
                eprintln!("[config] loaded {}", path.display());
            }
            (c, Some(ConfigFile::new(path, text)))
        }
        Err(e) => {
            eprintln!(
                "[config] can't use {}: {e}; using default controls, not saving",
                path.display()
            );
            (Controls::default(), None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(text: &str) -> (Controls, Vec<String>) {
        parse(text)
    }

    #[test]
    fn default_text_parses_back_to_defaults() {
        let (c, w) = parsed(&default_text());
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(c, Controls::default());
    }

    #[test]
    fn key_names_round_trip_and_cover_the_defaults() {
        for (name, code) in KEY_NAMES {
            assert_eq!(key_from_name(name), Some(*code), "{name}");
            assert_eq!(key_name(*code), *name, "{code:?} has two names");
            assert_eq!(key_from_name(&name.to_ascii_lowercase()), Some(*code));
        }
        for a in Action::ALL {
            assert_ne!(
                key_name(a.default_key()),
                "?",
                "{a:?}'s default key has no name"
            );
        }
    }

    #[test]
    fn bindings_apply() {
        let (c, w) = parsed("forward = [\"W\", \"up\" ]\n");
        // "up" is reserved: the whole line is ignored.
        assert_eq!(w.len(), 1, "{w:?}");
        assert_eq!(c.keys(Action::Forward), &[KeyCode::KeyW]);

        let (c, w) = parsed(
            "forward = [\"W\", \"i\"]\njump = \"J\"\nnoclip = []\nsensitivity = 2.5\ninvert_y = true\n",
        );
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(c.keys(Action::Forward), &[KeyCode::KeyW, KeyCode::KeyI]);
        assert_eq!(c.keys(Action::Jump), &[KeyCode::KeyJ]);
        assert!(c.keys(Action::Noclip).is_empty());
        assert_eq!((c.sensitivity, c.invert_y), (2.5, true));
        // Untouched actions keep their defaults.
        assert_eq!(c.keys(Action::Back), &[KeyCode::KeyS]);
    }

    /// Each bad line must warn *and* leave everything at the defaults.
    #[test]
    fn bad_lines_warn_and_keep_defaults() {
        for bad in [
            "fly = [\"F\"]",
            "forward = [\"Wq\"]",
            "forward = [\"W\", \"Nope\"]",
            "forward = [\"Escape\"]",
            "forward = [\"Enter\"]",
            "forward = W",
            "sensitivity = 0",
            "sensitivity = -1",
            "sensitivity = 21",
            "sensitivity = abc",
            "sensitivity = NaN",
            "invert_y = 2",
        ] {
            let (c, w) = parsed(bad);
            assert_eq!(w.len(), 1, "`{bad}` should warn once, got {w:?}");
            assert_eq!(c, Controls::default(), "`{bad}` changed something");
        }
    }

    #[test]
    fn a_key_on_two_actions_warns_and_binds_both() {
        // J on jump and noclip; and V (noclip's default) reused for forward.
        let (c, w) = parsed("jump = [\"J\"]\nnoclip = [\"J\"]\nforward = [\"V\"]\n");
        assert_eq!(w.len(), 1, "{w:?}");
        let (c2, w2) = parsed("forward = [\"V\"]\n");
        assert_eq!(
            w2.len(),
            1,
            "a clash with a default binding must warn: {w2:?}"
        );
        assert_eq!(c2.keys(Action::Noclip), &[KeyCode::KeyV]);
        let mut held = HashSet::new();
        let fired = c.key(&mut held, KeyCode::KeyJ, true, false);
        assert_eq!(fired, vec![Action::Jump, Action::Noclip]);
    }

    #[test]
    fn two_keys_on_one_action_hold_until_both_released() {
        let (c, _) = parsed("forward = [\"W\", \"I\"]");
        let mut held = HashSet::new();
        c.key(&mut held, KeyCode::KeyW, true, false);
        c.key(&mut held, KeyCode::KeyI, true, false);
        c.key(&mut held, KeyCode::KeyW, false, false);
        assert!(c.held(&held, Action::Forward), "releasing W cancelled I");
        c.key(&mut held, KeyCode::KeyI, false, false);
        assert!(!c.held(&held, Action::Forward));
    }

    #[test]
    fn fullscreen_toggle_is_f11_and_fires_once() {
        let c = Controls::default();
        let mut held = HashSet::new();
        assert_eq!(
            c.key(&mut held, KeyCode::F11, true, false),
            vec![Action::ToggleFullscreen]
        );
        assert!(c.key(&mut held, KeyCode::F11, true, true).is_empty());
        // Older files without the line still get it.
        let (c, w) = parsed("forward = [\"W\"]");
        assert!(w.is_empty());
        assert_eq!(c.keys(Action::ToggleFullscreen), &[KeyCode::F11]);
    }

    #[test]
    fn toggles_ignore_auto_repeat_but_exposure_repeats() {
        let c = Controls::default();
        let mut held = HashSet::new();
        assert_eq!(
            c.key(&mut held, KeyCode::KeyV, true, false),
            vec![Action::Noclip]
        );
        assert!(c.key(&mut held, KeyCode::KeyV, true, true).is_empty());
        assert!(c.key(&mut held, KeyCode::KeyV, false, false).is_empty());
        let up = KeyCode::BracketRight;
        assert_eq!(c.key(&mut held, up, true, false), vec![Action::ExposureUp]);
        assert_eq!(c.key(&mut held, up, true, true), vec![Action::ExposureUp]);
    }

    #[test]
    fn jump_fires_once_per_activation() {
        let (c, _) = parsed("jump = [\"J\", \"Space\"]");
        let mut held = HashSet::new();
        assert_eq!(
            c.key(&mut held, KeyCode::KeyJ, true, false),
            vec![Action::Jump]
        );
        // A second jump key while the first is down is not a new jump.
        assert!(c.key(&mut held, KeyCode::Space, true, false).is_empty());
        assert!(c.held(&held, Action::Jump));
    }

    #[test]
    fn rebind_takes_the_key_from_other_actions() {
        let mut c = Controls::default();
        assert!(c.rebind(Action::Forward, KeyCode::KeyV));
        assert_eq!(c.keys(Action::Forward), &[KeyCode::KeyV]);
        assert!(c.keys(Action::Noclip).is_empty(), "V was NOCLIP's");
        assert_eq!(c.keys_label(Action::Noclip), "NONE");
        assert_eq!(c.keys_label(Action::Forward), "V");
        // Reserved and unnameable keys are refused, and nothing changes.
        let before = c.clone();
        assert!(!c.rebind(Action::Jump, KeyCode::Enter));
        assert!(!c.rebind(Action::Jump, KeyCode::PrintScreen));
        assert_eq!(c, before);
        // A multi-key action becomes the one key.
        let (mut c, _) = parsed("forward = [\"W\", \"I\"]");
        assert_eq!(c.keys_label(Action::Forward), "W I");
        c.rebind(Action::Forward, KeyCode::KeyK);
        assert_eq!(c.keys_literal(Action::Forward), "[\"K\"]");
    }

    #[test]
    fn reset_keys_keeps_mouse_look() {
        let (mut c, _) = parsed("jump = [\"J\"]\nsensitivity = 2.5\ninvert_y = true");
        c.reset_keys();
        assert_eq!(c.keys(Action::Jump), &[KeyCode::Space]);
        assert_eq!((c.sensitivity, c.invert_y), (2.5, true));
    }

    #[test]
    fn sensitivity_steps_through_presets_and_wraps() {
        let mut c = Controls::default();
        let mut seen = vec![c.sensitivity_percent()];
        for _ in 0..SENSITIVITY_PRESETS.len() {
            c.step_sensitivity();
            seen.push(c.sensitivity_percent());
        }
        assert_eq!(seen, [100, 125, 150, 200, 300, 400, 25, 50, 75, 100]);
        // A hand-typed value goes to the next preset above it.
        let (mut c, _) = parsed("sensitivity = 1.37");
        c.step_sensitivity();
        assert_eq!(c.sensitivity_percent(), 150);
    }

    #[test]
    fn saved_literals_parse_back() {
        let mut c = Controls::default();
        c.rebind(Action::ExposureDown, KeyCode::Numpad4);
        c.step_sensitivity();
        let text = format!(
            "exposure_down = {}\nsensitivity = {:?}\n",
            c.keys_literal(Action::ExposureDown),
            c.sensitivity
        );
        let (back, w) = parsed(&text);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(back.keys(Action::ExposureDown), &[KeyCode::Numpad4]);
        assert_eq!(back.sensitivity, c.sensitivity);
    }

    #[test]
    fn menu_labels_are_drawable() {
        let drawable = |s: &str| {
            s.chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == ' ')
        };
        for a in Action::ALL {
            assert!(drawable(&a.menu_label()), "{:?}", a.menu_label());
        }
        let mut c = Controls::default();
        for (_, code) in KEY_NAMES.iter().filter(|(_, k)| !is_reserved(*k)) {
            c.rebind(Action::Jump, *code);
            let label = c.keys_label(Action::Jump);
            assert!(drawable(&label), "{label:?}");
        }
    }

    /// What the menu's save does to a real file: every action line rewritten
    /// in place, comments untouched, and the file parses back to the same
    /// controls.
    #[test]
    fn saving_bindings_keeps_the_file_readable_and_commented() {
        let dir = crate::config::test_dir("controls-save");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("controls.toml");
        std::fs::write(&path, default_text()).unwrap();
        let mut file = ConfigFile::new(&path, default_text());
        let mut c = Controls::default();
        c.rebind(Action::Forward, KeyCode::KeyV); // leaves NOCLIP unbound
        c.rebind(Action::ExposureUp, KeyCode::Numpad8);
        c.step_sensitivity();
        for a in Action::ALL {
            file.save_value(a.name(), &c.keys_literal(a)).unwrap();
        }
        file.save_value("sensitivity", &format!("{:?}", c.sensitivity))
            .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let (back, w) = parse(&text);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(back, c);
        let comments = |t: &str| t.lines().filter(|l| l.contains('#')).count();
        assert_eq!(comments(&text), comments(&default_text()));
        assert!(text.contains("noclip = []"), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rebinding_moves_the_action() {
        let (c, _) = parsed("jump = [\"J\"]");
        let mut held = HashSet::new();
        assert!(c.key(&mut held, KeyCode::Space, true, false).is_empty());
        assert_eq!(
            c.key(&mut held, KeyCode::KeyJ, true, false),
            vec![Action::Jump]
        );
    }
}
