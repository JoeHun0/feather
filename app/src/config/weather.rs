//! The weather key tables as data (`config/weather.json`, §13's keys as
//! data): parsed here, installed into the game, and hot-reloaded when the
//! file changes on disk. A bad file never applies — the running look is
//! untouched and every problem is named, the config idiom.

use feather_game::weather::{self, Key, WeatherData};
use glam::Vec3;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub const CONFIG_PATH: &str = "config/weather.json";

/// Parse the file's text into owned weather tables. Every problem is
/// collected, not just the first: a tuning session wants the whole list.
pub fn parse(text: &str) -> Result<Vec<WeatherData>, Vec<String>> {
    let doc: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => return Err(vec![format!("not JSON ({e})")]),
    };
    let Some(list) = doc["weathers"].as_array() else {
        return Err(vec!["no \"weathers\" array".to_string()]);
    };
    let mut problems = Vec::new();
    if list.is_empty() {
        problems.push("no weathers".to_string());
    }
    let mut out = Vec::new();
    for (i, w) in list.iter().enumerate() {
        let name = w["name"].as_str().unwrap_or("").to_string();
        let tag = if name.is_empty() {
            format!("weathers[{i}]")
        } else {
            name.clone()
        };
        let mut keys = Vec::new();
        match w["keys"].as_array() {
            None => problems.push(format!("{tag}: no \"keys\" array")),
            Some(key_list) => {
                for k in key_list {
                    match key(k) {
                        Ok(pair) => keys.push(pair),
                        Err(fields) => {
                            for p in fields {
                                problems.push(format!("{tag}: {p}"));
                            }
                        }
                    }
                }
            }
        }
        out.push(WeatherData { name, keys });
    }
    // The same rules the well-formedness test guards: what the test
    // enforces on the defaults, a file must pass.
    problems.extend(weather::problems(&out));
    if problems.is_empty() {
        Ok(out)
    } else {
        Err(problems)
    }
}

/// One key object: its hour and every field, all mandatory — the file has
/// no inheritance, so nothing is left out. Unknown fields are ignored, so a
/// file written for a newer engine still loads. Every bad field is named.
fn key(v: &serde_json::Value) -> Result<(f32, Key), Vec<String>> {
    let mut bad = Vec::new();
    let key = Key {
        sky_zenith: vec3(v, "sky_zenith", &mut bad).unwrap_or(Vec3::ONE),
        sky_horizon: vec3(v, "sky_horizon", &mut bad).unwrap_or(Vec3::ONE),
        sky_ground: vec3(v, "sky_ground", &mut bad).unwrap_or(Vec3::ONE),
        sky_sun_color: vec3(v, "sky_sun_color", &mut bad).unwrap_or(Vec3::ONE),
        sky_intensity: num(v, "sky_intensity", &mut bad).unwrap_or(1.0),
        sun_glow: num(v, "sun_glow", &mut bad).unwrap_or(0.0),
        sun_disk: num(v, "sun_disk", &mut bad).unwrap_or(0.0),
        sun_color: vec3(v, "sun_color", &mut bad).unwrap_or(Vec3::ONE),
        sun_intensity: num(v, "sun_intensity", &mut bad).unwrap_or(1.0),
        moon_color: vec3(v, "moon_color", &mut bad).unwrap_or(Vec3::ONE),
        moon_intensity: num(v, "moon_intensity", &mut bad).unwrap_or(1.0),
        fog_density: num(v, "fog_density", &mut bad).unwrap_or(0.0),
        fog_falloff: num(v, "fog_falloff", &mut bad).unwrap_or(0.0),
        fog_color: vec3(v, "fog_color", &mut bad).unwrap_or(Vec3::ONE),
        fog_sun: num(v, "fog_sun", &mut bad).unwrap_or(0.0),
        exposure_min: num(v, "exposure_min", &mut bad).unwrap_or(0.0),
        exposure_max: num(v, "exposure_max", &mut bad).unwrap_or(1.0),
    };
    let hour = num(v, "hour", &mut bad).unwrap_or(0.0);
    if bad.is_empty() {
        Ok((hour, key))
    } else {
        Err(bad)
    }
}

fn num(v: &serde_json::Value, k: &str, bad: &mut Vec<String>) -> Option<f32> {
    match v[k].as_f64() {
        Some(x) => Some(x as f32),
        None => {
            bad.push(format!("{k} not a number"));
            None
        }
    }
}

fn vec3(v: &serde_json::Value, k: &str, bad: &mut Vec<String>) -> Option<Vec3> {
    let problem = |bad: &mut Vec<String>| bad.push(format!("{k} not [r, g, b]"));
    let Some(a) = v[k].as_array() else {
        problem(bad);
        return None;
    };
    if a.len() != 3 {
        problem(bad);
        return None;
    }
    let mut c = [0.0f32; 3];
    for (i, s) in c.iter_mut().enumerate() {
        match a[i].as_f64() {
            Some(x) => *s = x as f32,
            None => {
                problem(bad);
                return None;
            }
        }
    }
    Some(Vec3::from_array(c))
}

/// The shipped file, watched: [`load`](Self::load) installs it, [`poll`](Self::poll)
/// re-parses and installs whenever the file's mtime moves.
pub struct WeatherFile {
    path: PathBuf,
    mtime: Option<SystemTime>,
}

impl WeatherFile {
    /// Open, parse and install the file. `None` — with the problems logged —
    /// when it's missing or invalid: a run never depends on it, and the
    /// compiled-in defaults carry on.
    pub fn load(path: &Path) -> Option<Self> {
        let mut f = Self {
            path: path.into(),
            mtime: None,
        };
        match f.read() {
            Ok(()) => {
                eprintln!("[weather] keys from {}", path.display());
                Some(f)
            }
            Err(problems) => {
                eprintln!(
                    "[weather] keeping the built-in keys, {} can't be used: {problems:?}",
                    path.display()
                );
                None
            }
        }
    }

    /// `Some` when the file changed since the last read: `Ok(())` if the new
    /// tables installed, `Err` with every problem when they didn't (the
    /// current tables stay). `None` when nothing moved.
    pub fn poll(&mut self) -> Option<Result<(), Vec<String>>> {
        if self.modified() == self.mtime {
            return None;
        }
        Some(self.read())
    }

    fn modified(&self) -> Option<SystemTime> {
        std::fs::metadata(&self.path)
            .ok()
            .and_then(|m| m.modified().ok())
    }

    fn read(&mut self) -> Result<(), Vec<String>> {
        let text =
            std::fs::read_to_string(&self.path).map_err(|e| vec![format!("can't read ({e})")])?;
        let data = parse(&text)?;
        weather::install(data);
        self.mtime = self.modified();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHIPPED: &str = include_str!("../../../config/weather.json");

    /// The file in the repo is exactly the compiled-in defaults: someone
    /// tuning starts from the look they already have.
    #[test]
    fn the_shipped_file_is_the_defaults() {
        let data = parse(SHIPPED).expect("the shipped file parses");
        assert_eq!(data.len(), weather::WEATHERS.len());
        for (parsed, default) in data.iter().zip(&weather::WEATHERS) {
            assert_eq!(parsed.name, default.name);
            assert_eq!(
                parsed.keys, default.keys,
                "{}'s keys differ from the compiled-in defaults",
                parsed.name
            );
        }
    }

    /// A bad file is rejected with every problem named, and the parser
    /// never invents data.
    #[test]
    fn bad_files_name_every_problem() {
        let not_json = parse("weather :)").unwrap_err();
        assert!(not_json[0].contains("not JSON"), "{not_json:?}");

        let no_array = parse("{}").unwrap_err();
        assert!(
            no_array.iter().any(|p| p.contains("no \"weathers\" array")),
            "{no_array:?}"
        );

        let bad_keys =
            parse(r#"{ "weathers": [ { "name": "MIST", "keys": [ { "hour": 0.0 } ] } ] }"#)
                .unwrap_err();
        let joined = bad_keys.join("\n");
        for field in ["sky_zenith", "fog_density", "exposure_max"] {
            assert!(joined.contains(field), "{field} not named in:\n{joined}");
        }

        // 17 fields but unsorted hours: the game's own rules bite too.
        let mut value: serde_json::Value = serde_json::from_str(SHIPPED).unwrap();
        value["weathers"][0]["keys"]
            .as_array_mut()
            .unwrap()
            .swap(0, 1);
        let unsorted = parse(&value.to_string()).unwrap_err();
        assert!(
            unsorted.iter().any(|p| p.contains("unsorted")),
            "{unsorted:?}"
        );
        assert!(parse(&value.to_string()).is_err());
    }

    /// Load installs the shipped tables; poll is quiet until the file
    /// changes; a valid change applies; an invalid one leaves the current
    /// tables alone. The changed table keeps the default names and count,
    /// so this can't disturb the other tests sharing the process.
    #[test]
    fn reloads_apply_only_valid_changes() {
        let dir = crate::config::test_dir("weather-reload");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weather.json");
        std::fs::write(&path, SHIPPED).unwrap();
        let mut file = WeatherFile::load(&path).expect("the shipped file loads");
        assert!(file.poll().is_none(), "unchanged file is quiet");

        let nudged = {
            let mut value: serde_json::Value = serde_json::from_str(SHIPPED).unwrap();
            let key = &mut value["weathers"][0]["keys"][0];
            key["fog_density"] = (key["fog_density"].as_f64().unwrap() * 2.0).into();
            value.to_string()
        };
        std::fs::write(&path, &nudged).unwrap();
        file.poll()
            .expect("the change is seen")
            .expect("and applies");
        let live = weather::tables()[0].keys[0].1.fog_density;
        let want = weather::WEATHERS[0].keys[0].1.fog_density * 2.0;
        assert_eq!(live, want, "the new tables are live");

        std::fs::write(&path, "{ not json").unwrap();
        let err = file
            .poll()
            .expect("the change is seen")
            .expect_err("and is rejected");
        assert!(err[0].contains("not JSON"));
        assert_eq!(
            weather::tables()[0].keys[0].1.fog_density,
            want,
            "a bad file leaves the current tables alone"
        );

        // Restore the file and the global, leaving no trace.
        std::fs::write(&path, SHIPPED).unwrap();
        file.poll().unwrap().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
