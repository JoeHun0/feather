//! Persistent graphics settings (§13): `config/settings.toml`, gitignored.
//!
//! A flat `key = value` subset of TOML, parsed by hand: three keys don't justify
//! a dependency, and a crate becomes worth it only when the file needs tables
//! (key bindings). Two rules shape everything here:
//!
//! - **Never fatal.** A missing file is created with defaults; a bad line is a
//!   warning and leaves that key at its default.
//! - **Never destructive.** Saving edits the matching line's value in place and
//!   leaves every other byte alone, so comments and lines the parser didn't
//!   understand survive. An unreadable file is left untouched, not replaced.
//!
//! Precedence is defaults < file < CLI. Only a key the player just changed is
//! ever written back, so a `--msaa 4` override never leaks into the file.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::{GraphicsSettings, ShadowQuality};

/// Relative to the working directory, like `BAKE_DIR`: fine while the game is
/// run via cargo from the repo root.
pub const CONFIG_PATH: &str = "config/settings.toml";

/// A persisted setting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Key {
    Shadows,
    Msaa,
    Fxaa,
}

impl Key {
    fn name(self) -> &'static str {
        match self {
            Self::Shadows => "shadows",
            Self::Msaa => "msaa",
            Self::Fxaa => "fxaa",
        }
    }

    /// The TOML literal for this key's current value in `s`.
    fn value(self, s: &GraphicsSettings) -> String {
        match self {
            Self::Shadows => format!("\"{}\"", s.shadows.config_name()),
            Self::Msaa => s.msaa.to_string(),
            Self::Fxaa => s.fxaa.to_string(),
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
# (e.g. --msaa 4) override it for one run and are never saved.

# Shadow quality: \"high\", \"medium\", \"low\" or \"off\". F1 cycles it in game.
shadows = {}

# MSAA sample count: 1, 2, 4 or 8, clamped to what the GPU supports. It is baked
# into the render pipelines, so the menu can change it only from the main menu.
msaa = {}

# FXAA post-process anti-aliasing: true or false. F2 toggles it in game.
fxaa = {}
",
        Key::Shadows.value(&d),
        Key::Msaa.value(&d),
        Key::Fxaa.value(&d),
    )
}

/// Byte index where a `#` comment starts (outside a quoted string), if any.
fn comment_start(line: &str) -> Option<usize> {
    let mut quoted = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '#' if !quoted => return Some(i),
            _ => {}
        }
    }
    None
}

/// The `key = value` part of a line: comment stripped, trimmed.
fn code(line: &str) -> &str {
    line[..comment_start(line).unwrap_or(line.len())].trim()
}

/// Apply every valid line of `text` to `s`. Returns one warning per line that
/// was ignored; the keys they named keep whatever `s` already held.
pub fn parse(text: &str, s: &mut GraphicsSettings) -> Vec<String> {
    let mut warnings = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = code(raw);
        if line.is_empty() {
            continue;
        }
        let mut warn = |msg: String| warnings.push(format!("line {}: {msg}; ignored", i + 1));
        if line.starts_with('[') {
            warn(format!("tables are not supported (`{line}`)"));
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            warn(format!("expected `key = value`, got `{line}`"));
            continue;
        };
        let (k, v) = (k.trim(), v.trim());
        match k {
            "shadows" => {
                let q = v
                    .strip_prefix('"')
                    .and_then(|v| v.strip_suffix('"'))
                    .and_then(ShadowQuality::from_config_name);
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
            _ => warn(format!("unknown key `{k}`")),
        }
    }
    warnings
}

/// `text` with `key`'s value replaced by `value` on every line that sets it,
/// every other byte (spacing, comments, other lines) unchanged. Appends
/// `key = value` when no line sets it.
pub fn set_value(text: &str, key: &str, value: &str) -> String {
    let mut out = String::with_capacity(text.len() + key.len() + value.len() + 4);
    let mut found = false;
    for raw in text.split_inclusive('\n') {
        let body = raw.trim_end_matches(['\n', '\r']);
        let code_end = comment_start(body).unwrap_or(body.len());
        let eq = body[..code_end].find('=');
        match eq {
            Some(eq) if body[..eq].trim() == key => {
                found = true;
                let region = &body[eq + 1..code_end];
                let lead = region.len() - region.trim_start().len();
                let trail = region.len() - region.trim_end().len();
                out.push_str(&body[..eq + 1 + lead]);
                out.push_str(value);
                out.push_str(&raw[code_end - trail..]);
            }
            _ => out.push_str(raw),
        }
    }
    if !found {
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&format!("{key} = {value}\n"));
    }
    out
}

/// Write via a temp file + rename, so a crash mid-write never leaves a
/// truncated settings file (the same pattern as the texture bake, §17).
fn write_atomic(path: &Path, text: &str) -> io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("toml.tmp");
    fs::write(&tmp, text)?;
    fs::rename(&tmp, path)
}

/// How `ConfigFile::open` found the file.
#[derive(Debug, PartialEq)]
pub enum Opened {
    /// It existed; these lines were ignored.
    Existing(Vec<String>),
    /// It didn't; a default one was written.
    Created,
}

/// The settings file, kept open for saving.
pub struct ConfigFile {
    path: PathBuf,
    /// Last text read or written, the base for a save if the file has since
    /// become unreadable.
    text: String,
}

impl ConfigFile {
    /// Read `path` into `s`, or write a default file there if it doesn't exist.
    /// An error means it could neither be read nor created; the caller then
    /// runs on defaults and saves nothing (and never overwrites the file).
    pub fn open(path: &Path, s: &mut GraphicsSettings) -> io::Result<(Self, Opened)> {
        match fs::read_to_string(path) {
            Ok(text) => {
                let warnings = parse(&text, s);
                let file = Self {
                    path: path.to_owned(),
                    text,
                };
                Ok((file, Opened::Existing(warnings)))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                let text = default_text();
                write_atomic(path, &text)?;
                // Parse what was written, so the file is the single source of
                // the values even on first run.
                parse(&text, s);
                let file = Self {
                    path: path.to_owned(),
                    text,
                };
                Ok((file, Opened::Created))
            }
            Err(e) => Err(e),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write `key`'s value from `s`. Edits the file as it is on disk now, so a
    /// hand edit made while the game runs is kept (except for this one key).
    pub fn save(&mut self, key: Key, s: &GraphicsSettings) -> io::Result<()> {
        let base = fs::read_to_string(&self.path).unwrap_or_else(|_| self.text.clone());
        let text = set_value(&base, key.name(), &key.value(s));
        if text != base {
            write_atomic(&self.path, &text)?;
        }
        self.text = text;
        Ok(())
    }
}

/// Startup entry point: open the file, log what happened, and return it for
/// saving (`None` if it can be neither read nor created).
pub fn load(s: &mut GraphicsSettings) -> Option<ConfigFile> {
    let path = Path::new(CONFIG_PATH);
    match ConfigFile::open(path, s) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(text: &str) -> (GraphicsSettings, Vec<String>) {
        let mut s = GraphicsSettings::default();
        let w = parse(text, &mut s);
        (s, w)
    }

    /// Settings fields compared as a tuple (the struct has no PartialEq).
    fn key(s: &GraphicsSettings) -> (ShadowQuality, u32, bool, bool) {
        (s.shadows, s.msaa, s.fxaa, s.bake)
    }

    #[test]
    fn default_text_parses_back_to_defaults() {
        let (s, w) = parsed(&default_text());
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(key(&s), key(&GraphicsSettings::default()));
    }

    #[test]
    fn values_apply() {
        let (s, w) = parsed("shadows = \"low\"  # cheap\nmsaa = 4\n  fxaa=true\n");
        assert!(w.is_empty(), "{w:?}");
        assert_eq!((s.shadows, s.msaa, s.fxaa), (ShadowQuality::Low, 4, true));
    }

    /// Each bad line must warn *and* leave its key at the default.
    #[test]
    fn bad_lines_warn_and_keep_defaults() {
        let d = GraphicsSettings::default();
        for bad in [
            "msaa = 3",
            "msaa = four",
            "fxaa = maybe",
            "shadows = \"ultra\"",
            "shadows = low",
            "[graphics]",
            "vsync = true",
            "just words",
        ] {
            let (s, w) = parsed(bad);
            assert_eq!(w.len(), 1, "`{bad}` should warn once, got {w:?}");
            assert_eq!(key(&s), key(&d), "`{bad}` changed a setting");
        }
    }

    #[test]
    fn hash_inside_quotes_is_not_a_comment() {
        assert_eq!(comment_start("a = \"x#y\" # c"), Some(10));
        assert_eq!(code("  # only a comment"), "");
    }

    #[test]
    fn set_value_edits_only_the_value() {
        let text = "# top\nshadows = \"high\"   # keep me\r\nmsaa=1\nweird line\n";
        let out = set_value(text, "shadows", "\"off\"");
        assert_eq!(
            out,
            "# top\nshadows = \"off\"   # keep me\r\nmsaa=1\nweird line\n"
        );
        let out = set_value(&out, "msaa", "8");
        assert_eq!(
            out,
            "# top\nshadows = \"off\"   # keep me\r\nmsaa=8\nweird line\n"
        );
        // A key named inside a comment is not a setting.
        assert_eq!(
            set_value("# fxaa = true\n", "fxaa", "false"),
            "# fxaa = true\nfxaa = false\n"
        );
    }

    #[test]
    fn set_value_appends_a_missing_key() {
        assert_eq!(
            set_value("msaa = 2", "fxaa", "true"),
            "msaa = 2\nfxaa = true\n"
        );
        assert_eq!(set_value("", "fxaa", "true"), "fxaa = true\n");
    }

    /// A fresh directory per test under the system temp dir.
    fn temp_path(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("feather-config-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir.join("sub").join("settings.toml")
    }

    #[test]
    fn open_creates_once_then_reads_and_saves() {
        let path = temp_path("create");
        let mut s = GraphicsSettings::default();
        let (mut file, how) = ConfigFile::open(&path, &mut s).unwrap();
        assert_eq!(how, Opened::Created);
        assert_eq!(fs::read_to_string(&path).unwrap(), default_text());

        // Hand edit, then a save of a *different* key keeps it.
        let edited = default_text().replace("msaa = 1", "msaa = 4");
        fs::write(&path, &edited).unwrap();
        s.fxaa = true;
        file.save(Key::Fxaa, &s).unwrap();
        let on_disk = fs::read_to_string(&path).unwrap();
        assert!(on_disk.contains("msaa = 4") && on_disk.contains("fxaa = true"));

        // Reopening reads, never rewrites.
        let mut s2 = GraphicsSettings::default();
        let (_, how) = ConfigFile::open(&path, &mut s2).unwrap();
        assert_eq!(how, Opened::Existing(vec![]));
        assert_eq!((s2.msaa, s2.fxaa), (4, true));
        assert_eq!(fs::read_to_string(&path).unwrap(), on_disk);
        let _ = fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
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
        assert!(ConfigFile::open(&path, &mut s).is_err());
        assert_eq!(fs::read(&path).unwrap(), junk);
        let _ = fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }
}
