//! The developer console (§19): a Quake-style overlay on the UiPass — press
//! ` in game, type commands, watch the world answer. The world keeps
//! simulating behind it; opening it is not a pause. Like `Menu`, the state
//! and the parsing live here, event- and renderer-free, so they're unit
//! tested without a GPU; `main` routes keys and applies the commands.

use feather_game::weather;
use feather_render::UiPass;

/// How many log lines the scrollback keeps.
const SCROLLBACK: usize = 64;

/// The console's state: the typing line, the command history, the log.
#[derive(Default)]
pub struct Console {
    pub open: bool,
    line: String,
    history: Vec<String>,
    /// Where the history walk stands; `history.len()` means "the line being
    /// typed", which the walk restores on the way back down.
    at: usize,
    partial: String,
    scrollback: Vec<String>,
}

/// One parsed command. `parse` is pure; `main` applies it.
#[derive(Debug, PartialEq)]
pub enum Command {
    /// Set the weather by choice (0 = LEVEL).
    Weather(usize),
    /// Set the clock's hour, or `None` for the level's own sun.
    Time(Option<f32>),
    /// Game seconds per real second.
    Speed(f32),
    /// Raw fog overrides over the presets; `None` clears.
    FogDensity(Option<f32>),
    FogHeight(Option<f32>),
    Exposure(f32),
    Fov(u32),
    Noclip,
    Teleport(f32, f32, f32),
    Clear,
    Help,
}

/// What `help` prints: one line per command, in the font's charset.
pub const HELP: &[&str] = &[
    "WEATHER LEVEL CLEAR OVERCAST FOGGY",
    "TIME 0-24 OR LEVEL",
    "SPEED GAME SECONDS PER REAL SECOND 0 OR MORE",
    "FOG DENSITY 0 OR MORE OR OFF",
    "FOG HEIGHT 0 OR MORE OR OFF",
    "EXPOSURE GREATER THAN 0",
    "FOV 30-120",
    "NOCLIP",
    "TELEPORT X Y Z",
    "CLEAR HELP",
];

impl Console {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn toggle(&mut self) {
        self.open = !self.open;
    }

    /// Type text: letters, digits, space, `.` and `-` (what the font can
    /// show); accepted case-insensitively, kept and shown uppercase.
    pub fn text(&mut self, s: &str) {
        for c in s.chars() {
            if c.is_ascii_alphanumeric() || matches!(c, ' ' | '.' | '-') {
                self.line.push(c.to_ascii_uppercase());
            }
        }
    }

    pub fn backspace(&mut self) {
        self.line.pop();
    }

    /// Submit the line: it enters the history (consecutive duplicates
    /// collapse) and is returned for the caller to run. An empty line is
    /// nothing.
    pub fn enter(&mut self) -> Option<String> {
        if self.line.is_empty() {
            return None;
        }
        let line = std::mem::take(&mut self.line);
        if self.history.last() != Some(&line) {
            self.history.push(line.clone());
        }
        self.at = self.history.len();
        self.partial.clear();
        Some(line)
    }

    /// Up: the older command.
    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        if self.at == self.history.len() {
            self.partial = self.line.clone();
            self.at -= 1;
        } else if self.at > 0 {
            self.at -= 1;
        }
        self.line = self.history[self.at].clone();
    }

    /// Down: the newer command, then the line that was being typed.
    pub fn history_next(&mut self) {
        if self.at >= self.history.len() {
            return;
        }
        self.at += 1;
        self.line = if self.at == self.history.len() {
            std::mem::take(&mut self.partial)
        } else {
            self.history[self.at].clone()
        };
    }

    pub fn clear(&mut self) {
        self.scrollback.clear();
    }

    /// A line into the log, oldest dropped past the cap.
    pub fn push(&mut self, line: impl Into<String>) {
        self.scrollback.push(line.into());
        if self.scrollback.len() > SCROLLBACK {
            self.scrollback.remove(0);
        }
    }

    /// The panel: a dim backdrop over the top of the screen, the log's last
    /// lines, and the prompt with a blinking block cursor.
    pub fn draw(&self, ui: &mut UiPass, w: f32, h: f32, blink: bool) {
        let size = (h / 50.0).max(9.0);
        let margin = size * 0.6;
        let text_h = ui.text_height(size);
        let pitch = text_h + size * 0.4;
        let panel = h * 0.4;
        ui.rect(0.0, 0.0, w, panel, [0.02, 0.02, 0.02, 0.75]);
        let text = [0.8, 0.85, 0.8, 0.95];
        // The log, newest just above the prompt, upwards while there's room.
        let mut y = panel - pitch - margin;
        for line in self.scrollback.iter().rev() {
            if y < margin {
                break;
            }
            ui.text(margin, y, size, text, line);
            y -= pitch;
        }
        // The prompt: a solid marker, then the line, then the cursor.
        let base = panel - margin - text_h;
        let marker = size * 0.55;
        ui.rect(
            margin,
            base + size * 0.3,
            marker,
            marker,
            [0.55, 0.8, 0.35, 0.9],
        );
        let text_x = margin + marker + size * 0.5;
        ui.text(text_x, base, size, text, &self.line);
        if blink {
            let cx = text_x + ui.text_width(&self.line, size) + size * 0.3;
            ui.rect(cx, base, size * 0.4, text_h, text);
        }
    }
}

fn f32_of(s: Option<&&str>, what: &str) -> Result<f32, String> {
    s.ok_or_else(|| format!("USAGE: {what}"))?
        .parse()
        .map_err(|_| format!("USAGE: {what}"))
}

/// Parse one submitted line. Unknown commands and bad arguments come back as
/// the usage they'd need.
pub fn parse(line: &str) -> Result<Command, String> {
    let mut words = line.split_whitespace();
    let cmd = words.next().unwrap_or("").to_ascii_lowercase();
    let rest: Vec<&str> = words.collect();
    let usage = |u: &str| Err(format!("USAGE: {u}"));
    match cmd.as_str() {
        "" => usage("HELP FOR THE LIST"),
        "help" => Ok(Command::Help),
        "clear" => Ok(Command::Clear),
        "weather" => {
            let name = rest
                .first()
                .ok_or("USAGE: WEATHER LEVEL CLEAR OVERCAST FOGGY")?;
            weather::choice_named(name)
                .map(Command::Weather)
                .ok_or_else(|| format!("UNKNOWN WEATHER {}", name.to_ascii_uppercase()))
        }
        "time" => match rest.first().map(|s| s.to_ascii_lowercase()) {
            None => usage("TIME 0-24, OR LEVEL"),
            Some(s) if s == "level" => Ok(Command::Time(None)),
            Some(s) => match s.parse::<f32>() {
                Ok(h) if (0.0..=24.0).contains(&h) => Ok(Command::Time(Some(h))),
                _ => usage("TIME 0-24, OR LEVEL"),
            },
        },
        "speed" => {
            let v = f32_of(rest.first(), "SPEED GAME SECONDS PER REAL SECOND")?;
            (v >= 0.0)
                .then_some(Command::Speed(v))
                .ok_or_else(|| "USAGE: SPEED 0 OR MORE".to_string())
        }
        "fog_density" | "fogdensity" => match rest.first().map(|s| s.to_ascii_lowercase()) {
            None => usage("FOG DENSITY 0 OR MORE, OR OFF"),
            Some(s) if s == "off" => Ok(Command::FogDensity(None)),
            Some(s) => match s.parse::<f32>() {
                Ok(v) if v >= 0.0 => Ok(Command::FogDensity(Some(v))),
                _ => usage("FOG DENSITY 0 OR MORE, OR OFF"),
            },
        },
        "fog_height" | "fogheight" => match rest.first().map(|s| s.to_ascii_lowercase()) {
            None => usage("FOG HEIGHT 0 OR MORE, OR OFF"),
            Some(s) if s == "off" => Ok(Command::FogHeight(None)),
            Some(s) => match s.parse::<f32>() {
                Ok(v) if v >= 0.0 => Ok(Command::FogHeight(Some(v))),
                _ => usage("FOG HEIGHT 0 OR MORE, OR OFF"),
            },
        },
        "exposure" => {
            let v = f32_of(rest.first(), "EXPOSURE GREATER THAN 0")?;
            (v > 0.0)
                .then_some(Command::Exposure(v))
                .ok_or_else(|| "USAGE: EXPOSURE GREATER THAN 0".to_string())
        }
        "fov" => match rest.first().and_then(|s| s.parse::<u32>().ok()) {
            Some(v @ 30..=120) => Ok(Command::Fov(v)),
            _ => usage("FOV 30-120"),
        },
        "noclip" => Ok(Command::Noclip),
        "teleport" => {
            if rest.len() != 3 {
                return usage("TELEPORT X Y Z");
            }
            let n = |s: &str| {
                s.parse::<f32>()
                    .map_err(|_| "USAGE: TELEPORT X Y Z".to_string())
            };
            Ok(Command::Teleport(n(rest[0])?, n(rest[1])?, n(rest[2])?))
        }
        other => Err(format!(
            "UNKNOWN COMMAND {} — HELP FOR THE LIST",
            other.to_ascii_uppercase()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_command() {
        assert_eq!(parse("help").unwrap(), Command::Help);
        assert_eq!(parse("clear").unwrap(), Command::Clear);
        assert_eq!(parse("weather foggy").unwrap(), Command::Weather(3));
        assert_eq!(parse("WEATHER LEVEL").unwrap(), Command::Weather(0));
        assert_eq!(parse("time 18.5").unwrap(), Command::Time(Some(18.5)));
        assert_eq!(parse("time level").unwrap(), Command::Time(None));
        assert_eq!(parse("speed 0").unwrap(), Command::Speed(0.0));
        assert_eq!(parse("speed 600").unwrap(), Command::Speed(600.0));
        assert_eq!(
            parse("fog_density 0.2").unwrap(),
            Command::FogDensity(Some(0.2))
        );
        assert_eq!(parse("fogdensity off").unwrap(), Command::FogDensity(None));
        assert_eq!(
            parse("fog_height 0.08").unwrap(),
            Command::FogHeight(Some(0.08))
        );
        assert_eq!(parse("fogheight off").unwrap(), Command::FogHeight(None));
        assert_eq!(parse("exposure 1.5").unwrap(), Command::Exposure(1.5));
        assert_eq!(parse("fov 90").unwrap(), Command::Fov(90));
        assert_eq!(parse("noclip").unwrap(), Command::Noclip);
        assert_eq!(
            parse("teleport 0 -7.36 37").unwrap(),
            Command::Teleport(0.0, -7.36, 37.0)
        );
    }

    #[test]
    fn rejects_bad_commands_with_usage() {
        for (line, needle) in [
            ("fly", "UNKNOWN COMMAND FLY"),
            ("weather mist", "UNKNOWN WEATHER MIST"),
            ("weather", "USAGE"),
            ("time 25", "USAGE"),
            ("time noon", "USAGE"),
            ("speed -1", "USAGE"),
            ("fog_density", "USAGE"),
            ("exposure 0", "USAGE"),
            ("fov 20", "USAGE"),
            ("teleport 1 2", "USAGE"),
            ("teleport 1 2 z", "USAGE"),
        ] {
            let err = parse(line).unwrap_err();
            assert!(err.contains(needle), "{line:?} → {err:?}, want {needle:?}");
        }
    }

    #[test]
    fn typing_appends_whats_printable() {
        let mut c = Console::new();
        c.text("weather Fog.");
        assert_eq!(c.line, "WEATHER FOG.");
        c.text("1-2+3,4`x"); // '+' and '`' drop, the rest land
        assert_eq!(c.line, "WEATHER FOG.1-234X");
        c.backspace();
        assert_eq!(c.line, "WEATHER FOG.1-234");
        for _ in 0..99 {
            c.backspace();
        }
        assert!(c.line.is_empty(), "backspace at 0 is harmless");
    }

    #[test]
    fn enter_submits_and_history_dedupes() {
        let mut c = Console::new();
        assert_eq!(c.enter(), None, "an empty line is nothing");
        c.text("help");
        assert_eq!(c.enter().as_deref(), Some("HELP"));
        c.text("help");
        assert_eq!(c.enter().as_deref(), Some("HELP"));
        assert_eq!(c.history.len(), 1, "consecutive duplicates collapse");
        c.text("clear");
        assert_eq!(c.enter().as_deref(), Some("CLEAR"));
        assert_eq!(c.history, ["HELP", "CLEAR"]);
        assert!(c.line.is_empty());
    }

    #[test]
    fn history_walks_oldest_and_back() {
        let mut c = Console::new();
        c.text("a");
        c.enter();
        c.text("b");
        c.enter();
        c.text("partial");
        c.history_prev();
        assert_eq!(c.line, "B");
        c.history_prev();
        assert_eq!(c.line, "A", "up walks to the oldest");
        c.history_prev();
        assert_eq!(c.line, "A", "and stays at the oldest");
        c.history_next();
        assert_eq!(c.line, "B");
        c.history_next();
        assert_eq!(c.line, "PARTIAL", "down restores the line being typed");
        c.history_next();
        assert_eq!(c.line, "PARTIAL", "and stays at the newest");
    }

    #[test]
    fn the_log_keeps_its_cap() {
        let mut c = Console::new();
        for i in 0..100 {
            c.push(format!("LINE {i}"));
        }
        assert_eq!(c.scrollback.len(), 64);
        assert_eq!(c.scrollback.first().unwrap(), "LINE 36");
        c.clear();
        assert!(c.scrollback.is_empty());
    }
}
