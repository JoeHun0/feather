//! Persistent audio volumes (§20): `config/audio.toml`, saved one value at a
//! time by the SOUND menu, in place, like `graphics.toml`.

use std::path::Path;

use super::{entries, open_or_create, warning, ConfigFile};
use crate::audio::AudioSettings;

/// Relative to the working directory, like `graphics::CONFIG_PATH`.
pub const AUDIO_PATH: &str = "config/audio.toml";

/// A persisted volume.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Key {
    Master,
    Sfx,
    Ambience,
}

impl Key {
    pub const ALL: [Self; 3] = [Self::Master, Self::Sfx, Self::Ambience];

    pub fn name(self) -> &'static str {
        match self {
            Self::Master => "master",
            Self::Sfx => "sfx",
            Self::Ambience => "ambience",
        }
    }

    pub fn get(self, a: &AudioSettings) -> u32 {
        match self {
            Self::Master => a.master,
            Self::Sfx => a.sfx,
            Self::Ambience => a.ambience,
        }
    }

    pub fn set(self, a: &mut AudioSettings, v: u32) {
        match self {
            Self::Master => a.master = v,
            Self::Sfx => a.sfx = v,
            Self::Ambience => a.ambience = v,
        }
    }
}

/// The file written when none exists, from `AudioSettings::default()`.
pub fn default_text() -> String {
    let d = AudioSettings::default();
    format!(
        "# Feather audio volumes, in percent (0 to 100). Written with defaults when
# missing; the OPTIONS > SOUND menu saves changes here. Delete it to reset.

# Everything.
master = {}
# Footsteps, jumps, landings.
sfx = {}
# Lamp hums and other ambient sound.
ambience = {}
",
        d.master, d.sfx, d.ambience
    )
}

/// Apply every valid line of `text` to `a`; one warning per ignored line.
pub fn parse(text: &str, a: &mut AudioSettings) -> Vec<String> {
    let (lines, mut warnings) = entries(text);
    for e in lines {
        let Some(key) = Key::ALL.into_iter().find(|k| k.name() == e.key) else {
            warnings.push(warning(e.line, format!("unknown key `{}`", e.key)));
            continue;
        };
        match e.value.parse::<u32>() {
            Ok(v @ 0..=100) => key.set(a, v),
            _ => warnings.push(warning(
                e.line,
                format!(
                    "{} must be a whole percent from 0 to 100, got {}",
                    e.key, e.value
                ),
            )),
        }
    }
    warnings
}

/// Save `key`'s current value (see `ConfigFile::save_value`).
pub fn save(file: &mut ConfigFile, key: Key, a: &AudioSettings) -> std::io::Result<()> {
    file.save_value(key.name(), &key.get(a).to_string())
}

/// Startup entry point: read (or create) the file into `a`, log what
/// happened, and return it for saving (`None`: defaults, nothing saved).
pub fn load(a: &mut AudioSettings) -> Option<ConfigFile> {
    let path = Path::new(AUDIO_PATH);
    match open_or_create(path, default_text) {
        Ok((text, created)) => {
            let warnings = parse(&text, a);
            for w in &warnings {
                eprintln!("[config] {}: {w}", path.display());
            }
            if created {
                eprintln!("[config] wrote defaults to {}", path.display());
            } else {
                eprintln!("[config] loaded {}", path.display());
            }
            Some(ConfigFile::new(path, text))
        }
        Err(e) => {
            eprintln!(
                "[config] can't use {}: {e}; default volumes, not saving",
                path.display()
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_text_parses_back_to_defaults() {
        let mut a = AudioSettings::default();
        let w = parse(&default_text(), &mut a);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(a, AudioSettings::default());
    }

    #[test]
    fn values_apply_and_bad_lines_keep_defaults() {
        let mut a = AudioSettings::default();
        assert!(parse("master = 50\nsfx=0\nambience = 100", &mut a).is_empty());
        assert_eq!((a.master, a.sfx, a.ambience), (50, 0, 100));
        for bad in [
            "master = 101",
            "sfx = -5",
            "ambience = loud",
            "music = 50",
            "[x]",
        ] {
            let mut a = AudioSettings::default();
            let w = parse(bad, &mut a);
            assert_eq!(w.len(), 1, "`{bad}`: {w:?}");
            assert_eq!(a, AudioSettings::default(), "`{bad}` changed something");
        }
    }

    #[test]
    fn saving_edits_one_value_in_place() {
        let dir = crate::config::test_dir("audio-save");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("audio.toml");
        std::fs::write(&path, default_text()).unwrap();
        let mut file = ConfigFile::new(&path, default_text());
        let mut a = AudioSettings::default();
        a.sfx = 25;
        save(&mut file, Key::Sfx, &a).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, default_text().replace("sfx = 100", "sfx = 25"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
