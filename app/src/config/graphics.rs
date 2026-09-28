//! Persistent graphics settings (§13): `config/graphics.toml`.
//!
//! Precedence is defaults < file < CLI. Only a key the player just changed (menu,
//! F1, F2) is ever written back, so a `--msaa 4` override never leaks into the
//! file. Saving edits the matching line's value in place and leaves every other
//! byte alone, so comments and lines the parser didn't understand survive.

use std::fs;
use std::io;
use std::path::Path;

use super::{entries, open_or_create, string, warning, ConfigFile};
use crate::{DisplayMode, GraphicsSettings, ShadowQuality};

/// Relative to the working directory, like `BAKE_DIR`: fine while the game is
/// run via cargo from the repo root.
pub const CONFIG_PATH: &str = "config/graphics.toml";
/// Where this file lived before controls got a file of their own.
const OLD_PATH: &str = "config/settings.toml";

/// A persisted setting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Key {
    Display,
    Fov,
    Shadows,
    Msaa,
    Fxaa,
    Bloom,
    AutoExposure,
    AmbientOcclusion,
}

impl Key {
    fn name(self) -> &'static str {
        match self {
            Self::Display => "display",
            Self::Fov => "fov",
            Self::Shadows => "shadows",
            Self::Msaa => "msaa",
            Self::Fxaa => "fxaa",
            Self::Bloom => "bloom",
            Self::AutoExposure => "auto_exposure",
            Self::AmbientOcclusion => "ambient_occlusion",
        }
    }

    /// The TOML literal for this key's current value in `s`.
    fn value(self, s: &GraphicsSettings) -> String {
        match self {
            Self::Display => format!("\"{}\"", s.display.config_name()),
            Self::Fov => s.fov_deg.to_string(),
            Self::Shadows => format!("\"{}\"", s.shadows.config_name()),
            Self::Msaa => s.msaa.to_string(),
            Self::Fxaa => s.fxaa.to_string(),
            Self::Bloom => s.bloom.to_string(),
            Self::AutoExposure => s.auto_exposure.to_string(),
            Self::AmbientOcclusion => s.ambient_occlusion.to_string(),
        }
    }
}

/// The file written when none exists. Values come from
/// `GraphicsSettings::default()`, so the template cannot drift from the code.
pub fn default_text() -> String {
    let d = GraphicsSettings::default();
    format!(
        "# Feather graphics settings. Written with defaults when missing; the menu,
# F1 and F2 save changes here. Delete this file to reset. Command-line flags
# (e.g. --msaa 4) override it for one run and are never saved. Key bindings
# are in controls.toml.

# Window: \"windowed\" or \"fullscreen\" (borderless, on the current monitor).
# F11 toggles it.
display = {}

# Vertical field of view in whole degrees, 30 to 120 (60 is about 91
# horizontal on a 16:9 screen). GAMEPLAY > FIELD OF VIEW steps 50 to 90.
fov = {}

# Shadow quality: \"high\", \"medium\", \"low\" or \"off\". F1 cycles it in game.
shadows = {}

# MSAA sample count: 1, 2, 4 or 8, clamped to what the GPU supports. It is baked
# into the render pipelines, so the menu can change it only from the main menu.
msaa = {}

# FXAA post-process anti-aliasing: true or false. F2 toggles it in game.
fxaa = {}

# Bloom: bright light glows softly into its surroundings. true or false.
bloom = {}

# Auto-exposure: the image brightens in the dark and darkens in bright light,
# as eyes adapt. true or false; off, [ and ] set a fixed exposure.
auto_exposure = {}

# Ambient occlusion (GTAO): contact shadows in corners and under objects.
# true or false.
ambient_occlusion = {}
",
        Key::Display.value(&d),
        Key::Fov.value(&d),
        Key::Shadows.value(&d),
        Key::Msaa.value(&d),
        Key::Fxaa.value(&d),
        Key::Bloom.value(&d),
        Key::AutoExposure.value(&d),
        Key::AmbientOcclusion.value(&d),
    )
}

/// Apply every valid line of `text` to `s`. Returns one warning per line that
/// was ignored; the keys they named keep whatever `s` already held.
pub fn parse(text: &str, s: &mut GraphicsSettings) -> Vec<String> {
    let (lines, mut warnings) = entries(text);
    for e in lines {
        let (k, v) = (e.key, e.value);
        let mut warn = |msg: String| warnings.push(warning(e.line, msg));
        match k {
            "display" => match string(v).and_then(DisplayMode::from_config_name) {
                Some(d) => s.display = d,
                None => warn(format!(
                    "display must be \"windowed\" or \"fullscreen\", got {v}"
                )),
            },
            "fov" => match v.parse::<u32>() {
                Ok(n @ 30..=120) => s.fov_deg = n,
                _ => warn(format!("fov must be whole degrees from 30 to 120, got {v}")),
            },
            "shadows" => {
                let q = string(v).and_then(ShadowQuality::from_config_name);
                match q {
                    Some(q) => s.shadows = q,
                    None => warn(format!(
                        "shadows must be \"high\", \"medium\", \"low\" or \"off\", got {v}"
                    )),
                }
            }
            "msaa" => match v.parse::<u32>() {
                Ok(n @ (1 | 2 | 4 | 8)) => s.msaa = n,
                _ => warn(format!("msaa must be 1, 2, 4 or 8, got {v}")),
            },
            "fxaa" => match v {
                "true" => s.fxaa = true,
                "false" => s.fxaa = false,
                _ => warn(format!("fxaa must be true or false, got {v}")),
            },
            "bloom" => match v {
                "true" => s.bloom = true,
                "false" => s.bloom = false,
                _ => warn(format!("bloom must be true or false, got {v}")),
            },
            "auto_exposure" => match v {
                "true" => s.auto_exposure = true,
                "false" => s.auto_exposure = false,
                _ => warn(format!("auto_exposure must be true or false, got {v}")),
            },
            "ambient_occlusion" => match v {
                "true" => s.ambient_occlusion = true,
                "false" => s.ambient_occlusion = false,
                _ => warn(format!("ambient_occlusion must be true or false, got {v}")),
            },
            _ => warn(format!("unknown key `{k}`")),
        }
    }
    warnings
}

/// How `open` found the file.
#[derive(Debug, PartialEq)]
pub enum Opened {
    /// It existed; these lines were ignored.
    Existing(Vec<String>),
    /// It didn't; a default one was written.
    Created,
}

/// Read `path` into `s`, or write a default file there if it doesn't exist.
/// An error means it could neither be read nor created; the caller then runs
/// on defaults and saves nothing (and never overwrites the file).
pub fn open(path: &Path, s: &mut GraphicsSettings) -> io::Result<(ConfigFile, Opened)> {
    let (text, created) = open_or_create(path, default_text)?;
    // Parsed even when just created, so the file is the single source of the
    // values from the first run on.
    let warnings = parse(&text, s);
    let file = ConfigFile::new(path, text);
    Ok((
        file,
        if created {
            Opened::Created
        } else {
            Opened::Existing(warnings)
        },
    ))
}

/// Write `key`'s current value from `s` (see `ConfigFile::save_value`).
pub fn save(file: &mut ConfigFile, key: Key, s: &GraphicsSettings) -> io::Result<()> {
    file.save_value(key.name(), &key.value(s))
}

/// Startup entry point: open the file, log what happened, and return it for
/// saving (`None` if it can be neither read nor created).
pub fn load(s: &mut GraphicsSettings) -> Option<ConfigFile> {
    let path = Path::new(CONFIG_PATH);
    match migrate(Path::new(OLD_PATH), path) {
        Ok(true) => eprintln!("[config] renamed {OLD_PATH} to {CONFIG_PATH}"),
        Ok(false) => {}
        Err(e) => eprintln!("[config] couldn't rename {OLD_PATH} to {CONFIG_PATH}: {e}"),
    }
    match open(path, s) {
        Ok((file, Opened::Created)) => {
            eprintln!("[config] wrote defaults to {}", path.display());
            Some(file)
        }
        Ok((file, Opened::Existing(warnings))) => {
            for w in &warnings {
                eprintln!("[config] {}: {w}", path.display());
            }
            eprintln!("[config] loaded {}", path.display());
            Some(file)
        }
        Err(e) => {
            eprintln!(
                "[config] can't use {}: {e}; running on defaults, not saving",
                path.display()
            );
            None
        }
    }
}

/// Move the pre-split `settings.toml` to `graphics.toml`, once: only when the
/// old file exists and the new one doesn't, so it never overwrites anything.
fn migrate(old: &Path, new: &Path) -> io::Result<bool> {
    if old.is_file() && !new.exists() {
        fs::rename(old, new)?;
        return Ok(true);
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(text: &str) -> (GraphicsSettings, Vec<String>) {
        let mut s = GraphicsSettings::default();
        let w = parse(text, &mut s);
        (s, w)
    }

    /// Settings fields compared as a tuple (the struct has no PartialEq).
    fn key(s: &GraphicsSettings) -> String {
        // Debug text of every field compared (the struct has no PartialEq).
        format!(
            "{:?}",
            (
                s.display,
                s.fov_deg,
                s.shadows,
                s.msaa,
                s.fxaa,
                s.bloom,
                s.auto_exposure,
                s.ambient_occlusion,
                s.bake,
                s.lod,
            )
        )
    }

    #[test]
    fn default_text_parses_back_to_defaults() {
        let (s, w) = parsed(&default_text());
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(key(&s), key(&GraphicsSettings::default()));
    }

    #[test]
    fn values_apply() {
        let (s, w) =
            parsed("shadows = \"low\"  # cheap\nmsaa = 4\n  fxaa=true\ndisplay = \"fullscreen\"\nbloom = false\nauto_exposure = false\nambient_occlusion = false\n");
        assert!(w.is_empty(), "{w:?}");
        assert_eq!((s.shadows, s.msaa, s.fxaa), (ShadowQuality::Low, 4, true));
        assert!(!s.bloom && !s.auto_exposure && !s.ambient_occlusion);
        assert_eq!(s.display, DisplayMode::Fullscreen);
        let (s, w) = parsed("fov = 30\n");
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(s.fov_deg, 30);
        assert_eq!(parsed("fov = 120").0.fov_deg, 120);
    }

    /// Each bad line must warn *and* leave its key at the default.
    #[test]
    fn bad_lines_warn_and_keep_defaults() {
        let d = GraphicsSettings::default();
        for bad in [
            "msaa = 3",
            "msaa = four",
            "fxaa = maybe",
            "bloom = 1",
            "auto_exposure = yes",
            "ambient_occlusion = on",
            "shadows = \"ultra\"",
            "shadows = low",
            "display = \"borderless\"",
            "display = fullscreen",
            "fov = 29",
            "fov = 121",
            "fov = 75.5",
            "[graphics]",
            "vsync = true",
            "just words",
        ] {
            let (s, w) = parsed(bad);
            assert_eq!(w.len(), 1, "`{bad}` should warn once, got {w:?}");
            assert_eq!(key(&s), key(&d), "`{bad}` changed a setting");
        }
    }

    use crate::config::test_dir;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        test_dir(name).join("sub").join("graphics.toml")
    }

    #[test]
    fn open_creates_once_then_reads_and_saves() {
        let path = temp_path("create");
        let mut s = GraphicsSettings::default();
        let (mut file, how) = open(&path, &mut s).unwrap();
        assert_eq!(how, Opened::Created);
        assert_eq!(fs::read_to_string(&path).unwrap(), default_text());

        // Hand edit, then a save of a *different* key keeps it.
        let edited = default_text().replace("msaa = 1", "msaa = 4");
        fs::write(&path, &edited).unwrap();
        s.fxaa = true;
        save(&mut file, Key::Fxaa, &s).unwrap();
        let on_disk = fs::read_to_string(&path).unwrap();
        assert!(on_disk.contains("msaa = 4") && on_disk.contains("fxaa = true"));

        // Reopening reads, never rewrites.
        let mut s2 = GraphicsSettings::default();
        let (_, how) = open(&path, &mut s2).unwrap();
        assert_eq!(how, Opened::Existing(vec![]));
        assert_eq!((s2.msaa, s2.fxaa), (4, true));
        assert_eq!(fs::read_to_string(&path).unwrap(), on_disk);
        let _ = fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn settings_toml_migrates_once() {
        let dir = test_dir("migrate");
        fs::create_dir_all(&dir).unwrap();
        let (old, new) = (dir.join("settings.toml"), dir.join("graphics.toml"));
        let text = default_text().replace("msaa = 1", "msaa = 8");
        fs::write(&old, &text).unwrap();
        assert!(migrate(&old, &new).unwrap());
        assert!(!old.exists());
        assert_eq!(fs::read_to_string(&new).unwrap(), text);
        // A second run, or a stray old file next to a new one, changes nothing.
        assert!(!migrate(&old, &new).unwrap());
        fs::write(&old, "msaa = 2\n").unwrap();
        assert!(!migrate(&old, &new).unwrap());
        assert_eq!(fs::read_to_string(&new).unwrap(), text);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Unreadable (here: not UTF-8) must be an error, and the file must stay
    /// exactly as it was.
    #[test]
    fn unreadable_file_is_left_alone() {
        let path = temp_path("unreadable");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let junk = [0xffu8, 0xfe, b'x'];
        fs::write(&path, junk).unwrap();
        let mut s = GraphicsSettings::default();
        assert!(open(&path, &mut s).is_err());
        assert_eq!(fs::read(&path).unwrap(), junk);
        let _ = fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }
}
