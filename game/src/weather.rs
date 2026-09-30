//! The weather and the time of day (§13's weather engine), built as X-Ray
//! builds them: a game clock, the sun's real path across the sky with a full
//! moon opposite it, and weathers as hourly keyframes of the whole atmosphere,
//! interpolated between keys and blended from one weather to the next.
//!
//! Everything here is plain data and maths, testable without a GPU (§1). Each
//! frame, `SessionWeather::frame` turns the level's `Environment`, the clock
//! and the chosen weather into this frame's `Environment` and the direction
//! the directional light travels; the renderer takes the rest from there
//! (`Environment::atmosphere`).
//!
//! **Keys are authored against a reference day,** an equinox: sunrise at 6
//! and sunset at 18. The clock is warped onto it (`reference_hour`), so a key
//! written for dawn lands on the real sunrise whatever the latitude and the
//! day of the year.

use feather_render::Environment;
use glam::Vec3;

/// Where the level is on Earth and when in the year (`environment`'s
/// `latitude` and `day_of_year`): what the sun's path depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolarPath {
    /// Degrees north (negative: south).
    pub latitude: f32,
    /// 1 = 1 January.
    pub day_of_year: f32,
}

impl Default for SolarPath {
    /// Near Chernobyl (51.3° N) at the end of April: sunrise 04:44 and
    /// sunset 19:16 in local solar time, the sun 53.3° up at noon.
    fn default() -> Self {
        Self {
            latitude: 51.3,
            day_of_year: 120.0,
        }
    }
}

impl SolarPath {
    /// The sun's declination (radians): Cooper's approximation.
    fn declination(&self) -> f64 {
        let n = self.day_of_year as f64;
        23.44f64.to_radians() * (std::f64::consts::TAU * (284.0 + n) / 365.0).sin()
    }

    /// The unit direction towards the sun at `hours` of local solar time: the
    /// engine's east is +X and its north −Z, as for `sun_azimuth`.
    pub fn sun_at(&self, hours: f64) -> Vec3 {
        let (lat, dec) = ((self.latitude as f64).to_radians(), self.declination());
        let ha = (15.0 * (hours - 12.0)).to_radians();
        let up = lat.sin() * dec.sin() + lat.cos() * dec.cos() * ha.cos();
        let east = -dec.cos() * ha.sin();
        let north = lat.cos() * dec.sin() - lat.sin() * dec.cos() * ha.cos();
        Vec3::new(east as f32, up as f32, -north as f32).normalize()
    }

    /// Sunrise and sunset (hours of solar time), or `None` where the sun
    /// doesn't set or doesn't rise that day.
    pub fn day(&self) -> Option<(f64, f64)> {
        let (lat, dec) = ((self.latitude as f64).to_radians(), self.declination());
        let cos_h = -lat.tan() * dec.tan();
        if !(-1.0..=1.0).contains(&cos_h) {
            return None;
        }
        let h = cos_h.acos().to_degrees() / 15.0;
        Some((12.0 - h, 12.0 + h))
    }

    /// The hour at which the sun stands nearest `to_sun`, to the minute: where
    /// the clock starts when a level's own sun meets a weather.
    pub fn hour_nearest(&self, to_sun: Vec3) -> f64 {
        let to_sun = to_sun.normalize();
        (0..24 * 60)
            .map(|m| m as f64 / 60.0)
            .max_by(|&a, &b| {
                let (da, db) = (self.sun_at(a).dot(to_sun), self.sun_at(b).dot(to_sun));
                da.total_cmp(&db)
            })
            .unwrap_or(12.0)
    }
}

/// The reference-day hour (sunrise 6, sunset 18) that `hours` of a day with
/// sunrise and sunset `day` corresponds to: piecewise linear through
/// midnight, sunrise, noon and sunset, so authored dawns and dusks land on
/// the real ones. A day with no sunrise or sunset isn't warped.
pub fn reference_hour(hours: f64, day: Option<(f64, f64)>) -> f32 {
    let h = hours.rem_euclid(24.0);
    let Some((rise, set)) = day else {
        return h as f32;
    };
    let span = |h: f64, a: f64, b: f64, ra: f64, rb: f64| ra + (h - a) / (b - a) * (rb - ra);
    (if h < rise {
        span(h, 0.0, rise, 0.0, 6.0)
    } else if h < 12.0 {
        span(h, rise, 12.0, 6.0, 12.0)
    } else if h < set {
        span(h, 12.0, set, 12.0, 18.0)
    } else {
        span(h, set, 24.0, 18.0, 24.0)
    }) as f32
}

/// The directional light at `hours`: the direction it travels and how much
/// of it reaches the ground (0–1), and whether it's the moon. It's the sun
/// while the sun is up and the full moon, opposite it, while it's down. Each
/// fades in over its first 3° above the horizon, so the light is 0 where it
/// switches from one to the other, and its shadows can't be seen to jump.
pub fn directional_light(path: &SolarPath, hours: f64) -> (Vec3, f32, bool) {
    let sun = path.sun_at(hours);
    let (body, moon) = if sun.y >= 0.0 {
        (sun, false)
    } else {
        (-sun, true)
    };
    let elevation = body.y.clamp(-1.0, 1.0).asin().to_degrees();
    let t = (elevation / 3.0).clamp(0.0, 1.0);
    (-body, t * t * (3.0 - 2.0 * t), moon)
}

/// One keyframe of a weather: the whole atmosphere at one hour of the
/// reference day. Linear HDR, as `Environment` takes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key {
    pub sky_zenith: Vec3,
    pub sky_horizon: Vec3,
    pub sky_ground: Vec3,
    /// The tint of the sun's glow and of the disk (the sun's, or the moon's
    /// at night).
    pub sky_sun_color: Vec3,
    pub sky_intensity: f32,
    pub sun_glow: f32,
    /// The disk of the body that's the light: the sun, or the moon.
    pub sun_disk: f32,
    /// The sunlight, used while the sun is up.
    pub sun_color: Vec3,
    pub sun_intensity: f32,
    /// The moonlight, used while the sun is down.
    pub moon_color: Vec3,
    pub moon_intensity: f32,
    pub fog_density: f32,
    pub fog_falloff: f32,
    pub fog_color: Vec3,
    pub fog_sun: f32,
    /// Auto-exposure's range (§13): night keys widen it.
    pub exposure_min: f32,
    pub exposure_max: f32,
}

/// `a` to `b` by `t`, in log space where both are positive: radiometric
/// values span decades between night and day, and a linear blend would
/// brighten dusk far too early.
fn geo(a: f32, b: f32, t: f32) -> f32 {
    if a == b {
        // A held value stays exact (ln and exp would round it).
        a
    } else if a > 0.0 && b > 0.0 {
        (a.ln() + (b.ln() - a.ln()) * t).exp()
    } else {
        a + (b - a) * t
    }
}

fn geo3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    Vec3::new(geo(a.x, b.x, t), geo(a.y, b.y, t), geo(a.z, b.z, t))
}

fn lin(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

impl Key {
    /// `a` to `b` by `t` (0–1). Radiometric values (colours, intensities, the
    /// fog's density, the exposure range) blend in log space, the shape
    /// parameters (glow, disk, falloff, fog_sun) linearly. The ends are
    /// exact.
    pub fn lerp(a: &Key, b: &Key, t: f32) -> Key {
        if t <= 0.0 {
            return *a;
        }
        if t >= 1.0 {
            return *b;
        }
        Key {
            sky_zenith: geo3(a.sky_zenith, b.sky_zenith, t),
            sky_horizon: geo3(a.sky_horizon, b.sky_horizon, t),
            sky_ground: geo3(a.sky_ground, b.sky_ground, t),
            sky_sun_color: geo3(a.sky_sun_color, b.sky_sun_color, t),
            sky_intensity: geo(a.sky_intensity, b.sky_intensity, t),
            sun_glow: lin(a.sun_glow, b.sun_glow, t),
            sun_disk: lin(a.sun_disk, b.sun_disk, t),
            sun_color: geo3(a.sun_color, b.sun_color, t),
            sun_intensity: geo(a.sun_intensity, b.sun_intensity, t),
            moon_color: geo3(a.moon_color, b.moon_color, t),
            moon_intensity: geo(a.moon_intensity, b.moon_intensity, t),
            fog_density: geo(a.fog_density, b.fog_density, t),
            fog_falloff: lin(a.fog_falloff, b.fog_falloff, t),
            fog_color: geo3(a.fog_color, b.fog_color, t),
            fog_sun: lin(a.fog_sun, b.fog_sun, t),
            exposure_min: geo(a.exposure_min, b.exposure_min, t),
            exposure_max: geo(a.exposure_max, b.exposure_max, t),
        }
    }

    /// This key as the frame's `Environment`: the sky glowing towards the sun
    /// at `to_sun`, and the directional light the sun or the moon as
    /// `directional_light` says, scaled by its `fade`. The fog's height and
    /// the exposure compensation stay the level's: a world height and a
    /// player adjustment.
    fn environment(&self, level: &Environment, to_sun: Vec3, fade: f32, moon: bool) -> Environment {
        let (color, intensity) = if moon {
            (self.moon_color, self.moon_intensity)
        } else {
            (self.sun_color, self.sun_intensity)
        };
        Environment {
            sun_dir: -to_sun,
            sun_color: color,
            sun_intensity: intensity * fade,
            sky_zenith: self.sky_zenith,
            sky_horizon: self.sky_horizon,
            sky_ground: self.sky_ground,
            sky_sun_color: self.sky_sun_color,
            sky_intensity: self.sky_intensity,
            sun_glow: self.sun_glow,
            sun_disk: self.sun_disk,
            fog_density: self.fog_density,
            fog_height: level.fog_height,
            fog_falloff: self.fog_falloff,
            fog_color: Some(self.fog_color),
            fog_sun: self.fog_sun,
            exposure: level.exposure,
            exposure_min: self.exposure_min,
            exposure_max: self.exposure_max,
        }
    }
}

/// A weather: its keys, by reference-day hour, sorted, within [0, 24). The
/// last blends into the first across midnight.
#[derive(Debug)]
pub struct Weather {
    /// As the WEATHER menu shows it (A–Z only).
    pub name: &'static str,
    pub keys: &'static [(f32, Key)],
}

impl Weather {
    /// The atmosphere at reference hour `hour`, between the keys round it.
    pub fn sample(&self, hour: f32) -> Key {
        let h = hour.rem_euclid(24.0);
        let keys = self.keys;
        let next = keys.iter().position(|&(k, _)| k > h).unwrap_or(keys.len());
        let (a, b) = if next == 0 || next == keys.len() {
            // Across midnight: from the last key to the first, a day later.
            (keys.len() - 1, 0)
        } else {
            (next - 1, next)
        };
        let (ha, hb) = (keys[a].0, keys[b].0);
        let span = (hb - ha).rem_euclid(24.0);
        let into = (h - ha).rem_euclid(24.0);
        let t = if span > 0.0 { into / span } else { 0.0 };
        Key::lerp(&keys[a].1, &keys[b].1, t)
    }
}

const fn rgb(r: f32, g: f32, b: f32) -> Vec3 {
    Vec3::new(r, g, b)
}

// ---- CLEAR: a cloudless spring day; the default look at noon. ----

const CLEAR_DAY: Key = Key {
    sky_zenith: rgb(0.10, 0.22, 0.55),
    sky_horizon: rgb(0.55, 0.65, 0.85),
    sky_ground: rgb(0.17, 0.18, 0.19),
    sky_sun_color: rgb(1.0, 0.95, 0.85),
    sky_intensity: 1.0,
    sun_glow: 0.6,
    sun_disk: 60.0,
    sun_color: rgb(1.0, 0.95, 0.86),
    sun_intensity: 8.0,
    moon_color: rgb(0.55, 0.65, 1.0),
    moon_intensity: 0.05,
    fog_density: 0.004,
    fog_falloff: 0.04,
    fog_color: rgb(0.50, 0.58, 0.72),
    fog_sun: 0.2,
    exposure_min: 0.125,
    exposure_max: 8.0,
};
const CLEAR_NIGHT: Key = Key {
    sky_zenith: rgb(0.003, 0.0045, 0.010),
    sky_horizon: rgb(0.006, 0.0075, 0.012),
    sky_ground: rgb(0.0025, 0.0025, 0.003),
    sky_sun_color: rgb(0.75, 0.82, 1.0),
    sun_glow: 0.004,
    // The moon: the sun's disk, so kept faint, or night's exposure blows it
    // up into a floodlight.
    sun_disk: 0.15,
    sun_color: rgb(1.0, 0.25, 0.05),
    sun_intensity: 0.5,
    fog_color: rgb(0.004, 0.005, 0.008),
    fog_sun: 0.0,
    exposure_min: 0.25,
    exposure_max: 16.0,
    ..CLEAR_DAY
};
const CLEAR_KEYS: [(f32, Key); 17] = [
    (0.0, CLEAR_NIGHT),
    (4.0, CLEAR_NIGHT),
    (
        5.0,
        Key {
            sky_zenith: rgb(0.008, 0.013, 0.030),
            sky_horizon: rgb(0.030, 0.026, 0.032),
            sky_ground: rgb(0.005, 0.005, 0.006),
            sky_sun_color: rgb(1.0, 0.55, 0.35),
            sun_glow: 0.02,
            fog_color: rgb(0.015, 0.016, 0.022),
            exposure_max: 8.0,
            ..CLEAR_NIGHT
        },
    ),
    (
        5.5,
        Key {
            sky_zenith: rgb(0.025, 0.045, 0.095),
            sky_horizon: rgb(0.13, 0.10, 0.09),
            sky_ground: rgb(0.015, 0.015, 0.016),
            sky_sun_color: rgb(1.0, 0.5, 0.25),
            sun_glow: 0.15,
            fog_color: rgb(0.08, 0.07, 0.07),
            fog_sun: 0.3,
            // Twilight stays dusky: auto-exposure may not lift it to day.
            exposure_max: 4.0,
            ..CLEAR_NIGHT
        },
    ),
    (
        6.0,
        Key {
            sky_zenith: rgb(0.06, 0.10, 0.21),
            sky_horizon: rgb(0.40, 0.29, 0.21),
            sky_ground: rgb(0.05, 0.045, 0.04),
            sky_sun_color: rgb(1.0, 0.52, 0.24),
            sun_glow: 0.9,
            sun_disk: 40.0,
            sun_color: rgb(1.0, 0.30, 0.06),
            sun_intensity: 1.0,
            fog_color: rgb(0.36, 0.28, 0.24),
            fog_sun: 0.6,
            ..CLEAR_DAY
        },
    ),
    (
        6.5,
        Key {
            sky_zenith: rgb(0.08, 0.16, 0.36),
            sky_horizon: rgb(0.50, 0.45, 0.42),
            sky_ground: rgb(0.10, 0.095, 0.09),
            sky_sun_color: rgb(1.0, 0.66, 0.40),
            sun_glow: 0.8,
            sun_disk: 55.0,
            sun_color: rgb(1.0, 0.55, 0.20),
            sun_intensity: 2.0,
            fog_color: rgb(0.45, 0.42, 0.42),
            fog_sun: 0.5,
            ..CLEAR_DAY
        },
    ),
    (
        7.5,
        Key {
            sky_zenith: rgb(0.095, 0.20, 0.48),
            sky_horizon: rgb(0.53, 0.60, 0.74),
            sky_ground: rgb(0.15, 0.155, 0.16),
            sky_sun_color: rgb(1.0, 0.85, 0.66),
            sun_glow: 0.65,
            sun_color: rgb(1.0, 0.78, 0.52),
            sun_intensity: 4.8,
            fog_sun: 0.3,
            ..CLEAR_DAY
        },
    ),
    (
        9.0,
        Key {
            sky_zenith: rgb(0.10, 0.215, 0.53),
            sky_horizon: rgb(0.55, 0.64, 0.83),
            sky_ground: rgb(0.165, 0.175, 0.185),
            sky_sun_color: rgb(1.0, 0.92, 0.80),
            sun_color: rgb(1.0, 0.90, 0.74),
            sun_intensity: 7.0,
            ..CLEAR_DAY
        },
    ),
    (12.0, CLEAR_DAY),
    (
        15.0,
        Key {
            sky_zenith: rgb(0.10, 0.215, 0.53),
            sky_horizon: rgb(0.55, 0.64, 0.83),
            sky_ground: rgb(0.165, 0.175, 0.185),
            sky_sun_color: rgb(1.0, 0.92, 0.80),
            sun_color: rgb(1.0, 0.89, 0.72),
            sun_intensity: 7.0,
            ..CLEAR_DAY
        },
    ),
    (
        16.5,
        Key {
            sky_zenith: rgb(0.095, 0.20, 0.47),
            sky_horizon: rgb(0.55, 0.58, 0.68),
            sky_ground: rgb(0.15, 0.15, 0.15),
            sky_sun_color: rgb(1.0, 0.82, 0.60),
            sun_glow: 0.7,
            sun_color: rgb(1.0, 0.75, 0.48),
            sun_intensity: 4.6,
            fog_sun: 0.35,
            ..CLEAR_DAY
        },
    ),
    (
        17.5,
        Key {
            sky_zenith: rgb(0.08, 0.15, 0.34),
            sky_horizon: rgb(0.55, 0.42, 0.36),
            sky_ground: rgb(0.10, 0.09, 0.08),
            sky_sun_color: rgb(1.0, 0.60, 0.32),
            sun_glow: 0.9,
            sun_disk: 55.0,
            sun_color: rgb(1.0, 0.50, 0.16),
            sun_intensity: 2.0,
            fog_color: rgb(0.48, 0.40, 0.36),
            fog_sun: 0.6,
            ..CLEAR_DAY
        },
    ),
    (
        18.0,
        Key {
            sky_zenith: rgb(0.055, 0.09, 0.19),
            sky_horizon: rgb(0.45, 0.26, 0.17),
            sky_ground: rgb(0.05, 0.042, 0.036),
            sky_sun_color: rgb(1.0, 0.45, 0.18),
            sun_glow: 1.1,
            sun_disk: 40.0,
            sun_color: rgb(1.0, 0.28, 0.05),
            sun_intensity: 1.0,
            fog_color: rgb(0.38, 0.26, 0.20),
            fog_sun: 0.7,
            ..CLEAR_DAY
        },
    ),
    (
        18.5,
        Key {
            sky_zenith: rgb(0.022, 0.04, 0.09),
            sky_horizon: rgb(0.15, 0.09, 0.08),
            sky_ground: rgb(0.014, 0.013, 0.014),
            sky_sun_color: rgb(1.0, 0.42, 0.20),
            sun_glow: 0.18,
            fog_color: rgb(0.09, 0.06, 0.06),
            fog_sun: 0.3,
            // Twilight stays dusky: auto-exposure may not lift it to day.
            exposure_max: 4.0,
            ..CLEAR_NIGHT
        },
    ),
    (
        19.0,
        Key {
            sky_zenith: rgb(0.008, 0.012, 0.028),
            sky_horizon: rgb(0.03, 0.022, 0.028),
            sky_ground: rgb(0.005, 0.005, 0.006),
            sky_sun_color: rgb(0.9, 0.6, 0.5),
            sun_glow: 0.02,
            fog_color: rgb(0.015, 0.014, 0.02),
            fog_sun: 0.05,
            exposure_max: 8.0,
            ..CLEAR_NIGHT
        },
    ),
    (20.0, CLEAR_NIGHT),
    (22.0, CLEAR_NIGHT),
];

// ---- OVERCAST: the zone's grey-green haze; its 15:00 key is the zone's
// own `environment`, the look it had before there was a clock. ----

const OVERCAST_DAY: Key = Key {
    sky_zenith: rgb(0.36, 0.39, 0.40),
    sky_horizon: rgb(0.58, 0.60, 0.56),
    sky_ground: rgb(0.16, 0.15, 0.12),
    sky_sun_color: rgb(1.0, 0.95, 0.85),
    sky_intensity: 1.0,
    sun_glow: 0.25,
    sun_disk: 0.0,
    sun_color: rgb(1.0, 0.92, 0.80),
    sun_intensity: 2.5,
    moon_color: rgb(0.6, 0.66, 0.8),
    moon_intensity: 0.006,
    fog_density: 0.035,
    fog_falloff: 0.08,
    fog_color: rgb(0.50, 0.52, 0.48),
    fog_sun: 0.3,
    exposure_min: 0.125,
    exposure_max: 8.0,
};
const OVERCAST_NIGHT: Key = Key {
    sky_zenith: rgb(0.004, 0.0045, 0.005),
    sky_horizon: rgb(0.006, 0.0065, 0.0065),
    sky_ground: rgb(0.002, 0.002, 0.002),
    sun_glow: 0.0,
    sun_color: rgb(1.0, 0.4, 0.2),
    sun_intensity: 0.3,
    fog_color: rgb(0.005, 0.0055, 0.0055),
    fog_sun: 0.0,
    exposure_min: 0.25,
    exposure_max: 16.0,
    ..OVERCAST_DAY
};
const OVERCAST_KEYS: [(f32, Key); 12] = [
    (0.0, OVERCAST_NIGHT),
    (4.5, OVERCAST_NIGHT),
    (
        5.5,
        Key {
            sky_zenith: rgb(0.05, 0.055, 0.065),
            sky_horizon: rgb(0.08, 0.08, 0.08),
            sky_ground: rgb(0.02, 0.02, 0.018),
            sky_sun_color: rgb(1.0, 0.75, 0.6),
            sun_glow: 0.05,
            fog_color: rgb(0.07, 0.07, 0.07),
            // Twilight stays dusky: auto-exposure may not lift it to day.
            exposure_max: 4.0,
            ..OVERCAST_NIGHT
        },
    ),
    (
        6.0,
        Key {
            sky_zenith: rgb(0.14, 0.15, 0.16),
            sky_horizon: rgb(0.24, 0.23, 0.21),
            sky_ground: rgb(0.06, 0.055, 0.045),
            sky_sun_color: rgb(1.0, 0.8, 0.6),
            sun_glow: 0.15,
            sun_color: rgb(1.0, 0.6, 0.35),
            sun_intensity: 0.6,
            fog_density: 0.05,
            fog_color: rgb(0.21, 0.21, 0.19),
            ..OVERCAST_DAY
        },
    ),
    (
        7.5,
        Key {
            sky_zenith: rgb(0.28, 0.30, 0.32),
            sky_horizon: rgb(0.46, 0.47, 0.44),
            sky_ground: rgb(0.12, 0.115, 0.095),
            sun_glow: 0.2,
            sun_color: rgb(1.0, 0.82, 0.64),
            sun_intensity: 1.6,
            fog_density: 0.045,
            fog_color: rgb(0.40, 0.42, 0.39),
            ..OVERCAST_DAY
        },
    ),
    (
        11.0,
        Key {
            sky_zenith: rgb(0.38, 0.41, 0.43),
            sky_horizon: rgb(0.60, 0.62, 0.58),
            sky_ground: rgb(0.17, 0.16, 0.13),
            sun_color: rgb(1.0, 0.94, 0.84),
            sun_intensity: 2.8,
            fog_color: rgb(0.52, 0.54, 0.50),
            ..OVERCAST_DAY
        },
    ),
    (15.0, OVERCAST_DAY),
    (
        17.0,
        Key {
            sky_zenith: rgb(0.26, 0.27, 0.28),
            sky_horizon: rgb(0.44, 0.43, 0.39),
            sky_ground: rgb(0.11, 0.10, 0.08),
            sky_sun_color: rgb(1.0, 0.85, 0.7),
            sun_color: rgb(1.0, 0.80, 0.60),
            sun_intensity: 1.5,
            fog_color: rgb(0.38, 0.38, 0.34),
            ..OVERCAST_DAY
        },
    ),
    (
        18.0,
        Key {
            sky_zenith: rgb(0.12, 0.12, 0.13),
            sky_horizon: rgb(0.22, 0.19, 0.16),
            sky_ground: rgb(0.05, 0.045, 0.035),
            sky_sun_color: rgb(1.0, 0.7, 0.5),
            sun_glow: 0.15,
            sun_color: rgb(1.0, 0.55, 0.3),
            sun_intensity: 0.5,
            fog_color: rgb(0.19, 0.17, 0.15),
            ..OVERCAST_DAY
        },
    ),
    (
        18.7,
        Key {
            sky_zenith: rgb(0.035, 0.037, 0.042),
            sky_horizon: rgb(0.06, 0.055, 0.052),
            sky_ground: rgb(0.012, 0.012, 0.011),
            sky_sun_color: rgb(1.0, 0.7, 0.5),
            sun_glow: 0.03,
            fog_color: rgb(0.05, 0.047, 0.045),
            // Twilight stays dusky: auto-exposure may not lift it to day.
            exposure_max: 4.0,
            ..OVERCAST_NIGHT
        },
    ),
    (19.5, OVERCAST_NIGHT),
    (22.0, OVERCAST_NIGHT),
];

// ---- FOGGY: a white morning mist that thins through the day. ----

const FOGGY_DAY: Key = Key {
    sky_zenith: rgb(0.42, 0.45, 0.48),
    sky_horizon: rgb(0.62, 0.64, 0.66),
    sky_ground: rgb(0.18, 0.18, 0.17),
    sky_sun_color: rgb(1.0, 0.96, 0.9),
    sky_intensity: 1.0,
    sun_glow: 0.5,
    sun_disk: 3.0,
    sun_color: rgb(1.0, 0.93, 0.82),
    sun_intensity: 3.0,
    moon_color: rgb(0.6, 0.66, 0.8),
    moon_intensity: 0.01,
    fog_density: 0.09,
    fog_falloff: 0.12,
    fog_color: rgb(0.60, 0.62, 0.64),
    fog_sun: 0.35,
    exposure_min: 0.125,
    exposure_max: 8.0,
};
const FOGGY_NIGHT: Key = Key {
    sky_zenith: rgb(0.004, 0.0048, 0.006),
    sky_horizon: rgb(0.007, 0.0075, 0.008),
    sky_ground: rgb(0.002, 0.002, 0.0022),
    sky_sun_color: rgb(0.75, 0.8, 0.95),
    sun_glow: 0.0,
    sun_disk: 0.05,
    sun_color: rgb(1.0, 0.4, 0.2),
    sun_intensity: 0.3,
    fog_color: rgb(0.007, 0.0075, 0.008),
    fog_sun: 0.0,
    exposure_min: 0.25,
    exposure_max: 16.0,
    ..FOGGY_DAY
};
const FOGGY_KEYS: [(f32, Key); 12] = [
    (0.0, FOGGY_NIGHT),
    (4.5, FOGGY_NIGHT),
    (
        5.5,
        Key {
            sky_zenith: rgb(0.05, 0.055, 0.065),
            sky_horizon: rgb(0.09, 0.09, 0.095),
            sky_ground: rgb(0.02, 0.02, 0.02),
            sky_sun_color: rgb(1.0, 0.7, 0.5),
            sun_glow: 0.05,
            fog_density: 0.12,
            fog_falloff: 0.10,
            fog_color: rgb(0.085, 0.085, 0.09),
            // Twilight stays dusky: auto-exposure may not lift it to day.
            exposure_max: 4.0,
            ..FOGGY_NIGHT
        },
    ),
    (
        6.0,
        Key {
            sky_zenith: rgb(0.15, 0.155, 0.17),
            sky_horizon: rgb(0.28, 0.26, 0.25),
            sky_ground: rgb(0.06, 0.058, 0.055),
            sky_sun_color: rgb(1.0, 0.7, 0.45),
            sun_glow: 0.4,
            sun_color: rgb(1.0, 0.5, 0.25),
            sun_intensity: 0.6,
            fog_density: 0.14,
            fog_falloff: 0.10,
            fog_color: rgb(0.27, 0.25, 0.24),
            fog_sun: 0.6,
            ..FOGGY_DAY
        },
    ),
    (
        7.5,
        Key {
            sky_zenith: rgb(0.32, 0.34, 0.37),
            sky_horizon: rgb(0.52, 0.52, 0.52),
            sky_ground: rgb(0.14, 0.14, 0.13),
            sky_sun_color: rgb(1.0, 0.85, 0.7),
            sun_color: rgb(1.0, 0.78, 0.55),
            sun_intensity: 1.6,
            fog_density: 0.12,
            fog_color: rgb(0.52, 0.52, 0.52),
            fog_sun: 0.5,
            ..FOGGY_DAY
        },
    ),
    (10.0, FOGGY_DAY),
    (
        14.0,
        Key {
            fog_density: 0.06,
            ..FOGGY_DAY
        },
    ),
    (
        17.0,
        Key {
            sky_zenith: rgb(0.28, 0.29, 0.30),
            sky_horizon: rgb(0.46, 0.44, 0.41),
            sky_ground: rgb(0.12, 0.115, 0.10),
            sky_sun_color: rgb(1.0, 0.82, 0.62),
            sun_color: rgb(1.0, 0.75, 0.5),
            sun_intensity: 1.6,
            fog_density: 0.07,
            fog_color: rgb(0.44, 0.42, 0.40),
            fog_sun: 0.5,
            ..FOGGY_DAY
        },
    ),
    (
        18.0,
        Key {
            sky_zenith: rgb(0.12, 0.12, 0.13),
            sky_horizon: rgb(0.24, 0.20, 0.17),
            sky_ground: rgb(0.05, 0.045, 0.04),
            sky_sun_color: rgb(1.0, 0.6, 0.35),
            sun_glow: 0.4,
            sun_color: rgb(1.0, 0.45, 0.2),
            sun_intensity: 0.5,
            fog_color: rgb(0.22, 0.19, 0.17),
            fog_sun: 0.6,
            ..FOGGY_DAY
        },
    ),
    (
        18.7,
        Key {
            sky_zenith: rgb(0.035, 0.037, 0.042),
            sky_horizon: rgb(0.065, 0.06, 0.058),
            sky_ground: rgb(0.012, 0.012, 0.012),
            sky_sun_color: rgb(1.0, 0.6, 0.4),
            sun_glow: 0.05,
            fog_density: 0.10,
            fog_color: rgb(0.06, 0.055, 0.055),
            // Twilight stays dusky: auto-exposure may not lift it to day.
            exposure_max: 4.0,
            ..FOGGY_NIGHT
        },
    ),
    (19.5, FOGGY_NIGHT),
    (22.0, FOGGY_NIGHT),
];

/// The weathers a level or the WEATHER menu can choose. WEATHER's first
/// choice, LEVEL, is the level's own `environment` instead.
pub const WEATHERS: [Weather; 3] = [
    Weather {
        name: "CLEAR",
        keys: &CLEAR_KEYS,
    },
    Weather {
        name: "OVERCAST",
        keys: &OVERCAST_KEYS,
    },
    Weather {
        name: "FOGGY",
        keys: &FOGGY_KEYS,
    },
];

/// The weather `name` names (any case), as a WEATHER choice: 0 for
/// `level`, i for `WEATHERS[i - 1]`.
pub fn choice_named(name: &str) -> Option<usize> {
    if name.eq_ignore_ascii_case("level") {
        return Some(0);
    }
    WEATHERS
        .iter()
        .position(|w| w.name.eq_ignore_ascii_case(name))
        .map(|i| i + 1)
}

/// A WEATHER choice's name.
pub fn choice_name(choice: usize) -> &'static str {
    match choice {
        0 => "LEVEL",
        i => WEATHERS[i - 1].name,
    }
}

/// FOG DENSITY's presets, over the weather's: per metre at the fog's height
/// (the zone's own is 0.035). `None` is the weather's.
pub const FOG_DENSITIES: [(&str, Option<f32>); 7] = [
    ("WEATHER", None),
    ("OFF", Some(0.0)),
    ("THIN", Some(0.008)),
    ("LIGHT", Some(0.018)),
    ("MEDIUM", Some(0.035)),
    ("THICK", Some(0.06)),
    ("HEAVY", Some(0.1)),
];

/// FOG HEIGHT's presets, over the weather's: the falloff, per metre (the
/// density falls by e every 1/falloff metres up; 0 is even, the same at every
/// height; the zone's own is 0.08). `None` is the weather's.
pub const FOG_HEIGHTS: [(&str, Option<f32>); 6] = [
    ("WEATHER", None),
    ("EVEN", Some(0.0)),
    ("TALL", Some(0.03)),
    ("MEDIUM", Some(0.08)),
    ("LOW", Some(0.2)),
    ("GROUND", Some(0.5)),
];

/// SPEED's presets: game seconds per real second.
pub const SPEEDS: [(&str, f32); 5] = [
    ("PAUSED", 0.0),
    ("1X", 1.0),
    ("10X", 10.0),
    ("60X", 60.0),
    ("600X", 600.0),
];

/// How long a change of weather takes, in game hours.
pub const TRANSITION_HOURS: f32 = 0.5;
/// A transition's pace never drops below this many game seconds a second, so
/// a picked weather still arrives with the clock PAUSED.
const TRANSITION_MIN_SPEED: f32 = 10.0;

/// What a level's `environment` marker says about its weather.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelWeather {
    pub path: SolarPath,
    /// The WEATHER choice a new game starts with (`weather`).
    pub choice: usize,
    /// The hour a new game starts at (`time`). Without one, LEVEL keeps the
    /// level's own sun and a weather starts where the sun is nearest it.
    pub time: Option<f32>,
    /// The clock's speed (`time_speed`), game seconds per real second.
    pub speed: f32,
}

impl Default for LevelWeather {
    fn default() -> Self {
        Self {
            path: SolarPath::default(),
            choice: 0,
            time: None,
            speed: 10.0,
        }
    }
}

/// A change of weather under way: from what was on screen when it was
/// picked, towards the new weather's keys.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Transition {
    from: Key,
    /// Game hours since it began.
    elapsed: f32,
}

/// The session's weather (§13): the clock, the chosen weather and any change
/// under way, and the WEATHER menu's fog overrides. Never saved: a new game
/// starts from the level's (`new`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SessionWeather {
    /// 0 = LEVEL, i = `WEATHERS[i - 1]`.
    pub choice: usize,
    /// Indices into `FOG_DENSITIES` and `FOG_HEIGHTS`.
    pub fog_density: usize,
    pub fog_height: usize,
    /// Hours of local solar time, [0, 24), or `None` for the level's own sun,
    /// fixed: only LEVEL can run without a clock.
    pub clock: Option<f64>,
    /// Game seconds per real second.
    pub speed: f32,
    transition: Option<Transition>,
    /// What the last frame showed, a weather's key: where the next change of
    /// weather starts from.
    last: Option<Key>,
    /// The hour at which the sun stands nearest the level's own, where a
    /// clock starts from.
    level_hour: f64,
}

impl SessionWeather {
    /// A new game's weather: the level's choice and clock, with the level's
    /// own sun at `level_sun` (the direction its light travels).
    pub fn new(level: &LevelWeather, level_sun: Vec3) -> Self {
        let level_hour = level.path.hour_nearest(-level_sun);
        let clock = match (level.time, level.choice) {
            (Some(t), _) => Some(t as f64),
            (None, 0) => None,
            (None, _) => Some(level_hour),
        };
        Self {
            choice: level.choice,
            clock,
            speed: level.speed,
            level_hour,
            ..Default::default()
        }
    }

    /// Advance `dt` real seconds: the clock at its speed, and any change of
    /// weather at least at `TRANSITION_MIN_SPEED`.
    pub fn advance(&mut self, dt: f32) {
        if let Some(h) = self.clock.as_mut() {
            *h = (*h + (dt * self.speed) as f64 / 3600.0).rem_euclid(24.0);
        }
        if let Some(t) = self.transition.as_mut() {
            t.elapsed += dt * self.speed.max(TRANSITION_MIN_SPEED) / 3600.0;
            if t.elapsed >= TRANSITION_HOURS {
                self.transition = None;
            }
        }
    }

    /// WEATHER's next choice. From one weather to another it's a change over
    /// `TRANSITION_HOURS`, from what's on screen; to or from LEVEL, whose
    /// palette isn't a weather's, it's at once. A weather needs a clock: one
    /// starts where the sun is nearest the level's.
    pub fn cycle_weather(&mut self) {
        let next = (self.choice + 1) % (WEATHERS.len() + 1);
        self.transition = match (self.choice, next, self.last) {
            (1.., 1.., Some(from)) => Some(Transition { from, elapsed: 0.0 }),
            _ => None,
        };
        if next == 0 {
            self.last = None;
        }
        self.choice = next;
        if next != 0 && self.clock.is_none() {
            self.clock = Some(self.level_hour);
        }
    }

    /// TIME's next step: the next whole hour, from the level's sun's if
    /// there's no clock yet.
    pub fn step_time(&mut self) {
        let h = self.clock.unwrap_or(self.level_hour);
        self.clock = Some((h.floor() + 1.0).rem_euclid(24.0));
    }

    /// SPEED's next preset, after the current speed.
    pub fn cycle_speed(&mut self) {
        let next = SPEEDS.iter().position(|&(_, v)| v > self.speed);
        self.speed = SPEEDS[next.unwrap_or(0)].1;
    }

    /// Whether a change of weather is under way, and how far (0–1).
    pub fn transition(&self) -> Option<f32> {
        self.transition.map(|t| t.elapsed / TRANSITION_HOURS)
    }

    /// This frame's atmosphere and the direction the directional light
    /// travels, from the level's `environment` and the path of its sun.
    /// Remembers what it showed, for the next change of weather.
    pub fn frame(&mut self, level: &Environment, path: &SolarPath) -> (Environment, Vec3) {
        let (mut env, light) = match (self.choice, self.clock) {
            // The level's own look, exactly.
            (0, None) => (*level, level.sun_dir),
            // The level's palette under a moving sun (SUN ONLY): the sun sets
            // and there's no moon, so its light is 0 by night.
            (0, Some(h)) => {
                let (light, fade, moon) = directional_light(path, h);
                let env = Environment {
                    sun_dir: -path.sun_at(h),
                    sun_intensity: if moon {
                        0.0
                    } else {
                        level.sun_intensity * fade
                    },
                    ..*level
                };
                (env, light)
            }
            (i, clock) => {
                let h = clock.unwrap_or(self.level_hour);
                let now = WEATHERS[i - 1].sample(reference_hour(h, path.day()));
                let key = match self.transition {
                    Some(t) => {
                        let s = (t.elapsed / TRANSITION_HOURS).clamp(0.0, 1.0);
                        Key::lerp(&t.from, &now, s * s * (3.0 - 2.0 * s))
                    }
                    None => now,
                };
                self.last = Some(key);
                let (light, fade, moon) = directional_light(path, h);
                (key.environment(level, path.sun_at(h), fade, moon), light)
            }
        };
        if let Some(d) = FOG_DENSITIES[self.fog_density].1 {
            env.fog_density = d;
        }
        if let Some(f) = FOG_HEIGHTS[self.fog_height].1 {
            env.fog_falloff = f;
        }
        (env, light)
    }

    /// TIME's label: `HH MM` (the menu's font has no colon), or LEVEL for the
    /// level's own sun.
    pub fn time_label(&self) -> String {
        match self.clock {
            None => "LEVEL".to_string(),
            Some(h) => {
                let m = (h * 60.0).floor() as u32 % (24 * 60);
                format!("{:02} {:02}", m / 60, m % 60)
            }
        }
    }

    /// SPEED's label: its preset's, or the multiple a level asked for.
    pub fn speed_label(&self) -> String {
        match SPEEDS.iter().find(|&&(_, v)| v == self.speed) {
            Some((name, _)) => name.to_string(),
            None => format!("{}X", self.speed.round() as u32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elevation(v: Vec3) -> f32 {
        v.y.clamp(-1.0, 1.0).asin().to_degrees()
    }

    /// The sun's path: at noon due south at `90 − latitude + declination`,
    /// in the east in the morning and the west in the afternoon; an equinox
    /// rises at 6 and sets at 18, and a sunrise and sunset are where the sun
    /// crosses the horizon.
    #[test]
    fn the_sun_follows_its_path() {
        let path = SolarPath::default();
        let dec = path.declination().to_degrees() as f32;
        let noon = path.sun_at(12.0);
        assert!(
            (elevation(noon) - (90.0 - 51.3 + dec)).abs() < 1e-3,
            "{noon}"
        );
        assert!(
            noon.x.abs() < 1e-5 && noon.z > 0.0,
            "noon is due south: {noon}"
        );
        assert!(
            (elevation(noon) - 53.3).abs() < 0.05,
            "end of April: {noon}"
        );
        assert!(path.sun_at(9.0).x > 0.3 && path.sun_at(15.0).x < -0.3);
        let equinox = SolarPath {
            day_of_year: 81.0,
            ..path
        };
        let (rise, set) = equinox.day().unwrap();
        assert!(
            (rise - 6.0).abs() < 1e-3 && (set - 18.0).abs() < 1e-3,
            "{rise} {set}"
        );
        let (rise, set) = path.day().unwrap();
        assert!(
            (4.7..4.8).contains(&rise) && (19.2..19.3).contains(&set),
            "{rise} {set}"
        );
        for h in [rise, set] {
            assert!(path.sun_at(h).y.abs() < 1e-5, "{h}: {}", path.sun_at(h));
        }
        assert!(path.sun_at(rise + 0.1).y > 0.0 && path.sun_at(rise - 0.1).y < 0.0);
        // No sunset in the Arctic summer, no sunrise in its winter.
        let arctic = |day_of_year| SolarPath {
            latitude: 80.0,
            day_of_year,
        };
        assert_eq!(arctic(172.0).day(), None);
        assert_eq!(arctic(355.0).day(), None);
        // A level's own sun is found again on the path.
        for h in [6.5, 12.0, 15.25] {
            assert!((path.hour_nearest(path.sun_at(h)) - h).abs() < 1.0 / 60.0 + 1e-9);
        }
    }

    /// The reference day: sunrise maps to 6 and sunset to 18, midnight and
    /// noon stay, the map only ever climbs, and an equinox isn't warped.
    #[test]
    fn the_clock_is_warped_onto_the_reference_day() {
        let day = Some((5.0, 19.0));
        for (h, want) in [
            (0.0, 0.0),
            (5.0, 6.0),
            (12.0, 12.0),
            (19.0, 18.0),
            (2.5, 3.0),
        ] {
            assert!((reference_hour(h, day) - want).abs() < 1e-5, "{h}");
        }
        let mut last = -1.0;
        for m in 0..24 * 60 {
            let r = reference_hour(m as f64 / 60.0, day);
            assert!(r > last, "not climbing at minute {m}");
            last = r;
        }
        assert!(last < 24.0);
        for m in (0..24 * 60).step_by(7) {
            let h = m as f64 / 60.0;
            assert!((reference_hour(h, Some((6.0, 18.0))) as f64 - h).abs() < 1e-5);
            assert_eq!(reference_hour(h, None), h as f32);
        }
    }

    const A: Key = CLEAR_DAY;
    const B: Key = CLEAR_NIGHT;

    /// Blending keys: exact at the ends; radiometric values geometric (the
    /// halfway colour is the two's geometric mean), shapes linear.
    #[test]
    fn keys_blend_in_log_space() {
        assert_eq!(Key::lerp(&A, &B, 0.0), A);
        assert_eq!(Key::lerp(&A, &B, 1.0), B);
        let half = Key::lerp(&A, &B, 0.5);
        let mean = |a: f32, b: f32| (a * b).sqrt();
        let close = |x: f32, y: f32| (x - y).abs() <= 1e-5 * y.abs().max(1e-3);
        assert!(close(
            half.sky_zenith.x,
            mean(A.sky_zenith.x, B.sky_zenith.x)
        ));
        assert!(close(
            half.sun_intensity,
            mean(A.sun_intensity, B.sun_intensity)
        ));
        assert!(close(
            half.exposure_max,
            mean(A.exposure_max, B.exposure_max)
        ));
        assert!(close(half.sun_glow, (A.sun_glow + B.sun_glow) / 2.0));
        assert!(close(half.sun_disk, (A.sun_disk + B.sun_disk) / 2.0));
        // A zero blends linearly (OVERCAST's disk is 0).
        let zero = Key { sun_disk: 0.0, ..A };
        assert!(close(
            Key::lerp(&zero, &A, 0.25).sun_disk,
            A.sun_disk * 0.25
        ));
    }

    /// Sampling a weather: its keys exactly at their hours, a blend between
    /// them, and across midnight from the last key to the first.
    #[test]
    fn a_weather_samples_between_its_keys() {
        for w in &WEATHERS {
            for &(h, k) in w.keys {
                assert_eq!(w.sample(h), k, "{} at {h}", w.name);
            }
        }
        let keys: &'static [(f32, Key)] = &[(2.0, A), (20.0, B)];
        let w = Weather { name: "TEST", keys };
        assert_eq!(w.sample(11.0), Key::lerp(&A, &B, 0.5));
        assert_eq!(w.sample(23.0), Key::lerp(&B, &A, 0.5));
        assert_eq!(w.sample(1.0), Key::lerp(&B, &A, 5.0 / 6.0));
        assert_eq!(w.sample(24.0), w.sample(0.0));
    }

    /// Every weather's keys are sorted within a day, and every value is one
    /// the shaders and the exposure can use. OVERCAST's 15:00 key is the
    /// zone's own `environment` (gen_zonescene.py), the look it had before.
    #[test]
    fn the_weathers_are_well_formed() {
        for w in &WEATHERS {
            assert!(w.name.bytes().all(|b| b.is_ascii_uppercase()), "{}", w.name);
            assert_eq!(choice_named(&w.name.to_lowercase()), Some(choice_of(w)));
            assert!(
                w.keys.windows(2).all(|p| p[0].0 < p[1].0),
                "{} unsorted",
                w.name
            );
            assert!(w.keys.iter().all(|&(h, _)| (0.0..24.0).contains(&h)));
            for &(h, k) in w.keys {
                let colours = [
                    k.sky_zenith,
                    k.sky_horizon,
                    k.sky_ground,
                    k.sky_sun_color,
                    k.sun_color,
                    k.moon_color,
                    k.fog_color,
                ];
                for c in colours {
                    assert!(
                        c.is_finite() && c.min_element() > 0.0,
                        "{} {h}: {c}",
                        w.name
                    );
                }
                for v in [k.sky_intensity, k.sun_intensity, k.moon_intensity] {
                    assert!(v.is_finite() && v > 0.0, "{} {h}", w.name);
                }
                for v in [
                    k.sun_glow,
                    k.sun_disk,
                    k.fog_density,
                    k.fog_falloff,
                    k.fog_sun,
                ] {
                    assert!(v.is_finite() && v >= 0.0, "{} {h}", w.name);
                }
                assert!(0.0 < k.exposure_min && k.exposure_min < k.exposure_max);
            }
        }
        let zone = include_str!("../../tools/gen_zonescene.py");
        let zone = &zone[zone.find("ENVIRONMENT = {").unwrap()..];
        let zone = &zone[..zone.find("\n}").unwrap()];
        // `"key": 1.0,` or `"key": [r, g, b],` in the Python dict.
        let value = |key: &str| -> Vec<f32> {
            let at = zone.find(&format!("\"{key}\":")).unwrap() + key.len() + 3;
            let v = zone[at..].trim_start();
            let v = match v.strip_prefix('[') {
                Some(list) => &list[..list.find(']').unwrap()],
                None => &v[..v.find([',', '\n']).unwrap()],
            };
            v.split(',').map(|x| x.trim().parse().unwrap()).collect()
        };
        let v3 = |key| Vec3::from_slice(&value(key));
        let k = WEATHERS[choice_named("overcast").unwrap() - 1].sample(15.0);
        assert_eq!(k.sky_zenith, v3("sky_zenith"));
        assert_eq!(k.sky_horizon, v3("sky_horizon"));
        assert_eq!(k.sky_ground, v3("sky_ground"));
        assert_eq!(k.sky_sun_color, v3("sky_sun_color"));
        assert_eq!(k.sun_color, v3("sun_color"));
        assert_eq!(k.fog_color, v3("fog_color"));
        for (key, got) in [
            ("sun_intensity", k.sun_intensity),
            ("sky_intensity", k.sky_intensity),
            ("sun_glow", k.sun_glow),
            ("sun_disk", k.sun_disk),
            ("fog_density", k.fog_density),
            ("fog_falloff", k.fog_falloff),
            ("fog_sun", k.fog_sun),
        ] {
            assert_eq!(value(key), [got], "{key}");
        }
    }

    fn choice_of(w: &Weather) -> usize {
        WEATHERS.iter().position(|x| x.name == w.name).unwrap() + 1
    }

    /// The directional light is the sun by day and the full moon opposite
    /// it by night; its strength is continuous through sunrise and sunset,
    /// and 0 where it switches.
    #[test]
    fn the_light_is_the_sun_by_day_and_the_moon_by_night() {
        let path = SolarPath::default();
        let (light, fade, moon) = directional_light(&path, 12.0);
        assert!(!moon && fade == 1.0 && light == -path.sun_at(12.0));
        let (light, fade, moon) = directional_light(&path, 0.0);
        assert!(moon && fade == 1.0 && light == path.sun_at(0.0) && light.y < 0.0);
        let (rise, set) = path.day().unwrap();
        for edge in [rise, set] {
            let mut last = None;
            for s in -600..=600 {
                let h = edge + s as f64 / 3600.0; // a second at a time
                let (_, fade, _) = directional_light(&path, h);
                if let Some(prev) = last {
                    let d: f32 = fade - prev;
                    assert!(d.abs() < 2e-3, "a jump of {d} at {h}");
                }
                last = Some(fade);
            }
            assert!(directional_light(&path, edge).1 < 1e-3);
        }
    }

    /// The clock runs at its speed and wraps the day; paused, it stays, and
    /// LEVEL without a clock has none to run.
    #[test]
    fn the_clock_runs_at_its_speed() {
        let mut w = SessionWeather {
            clock: Some(23.5),
            speed: 60.0,
            ..Default::default()
        };
        w.advance(30.0); // 30 s at 60x: half an hour
        assert!(w.clock.unwrap().abs() < 1e-9, "{:?}", w.clock);
        w.advance(90.0);
        assert!((w.clock.unwrap() - 1.5).abs() < 1e-9);
        w.speed = 0.0;
        w.advance(1000.0);
        assert!((w.clock.unwrap() - 1.5).abs() < 1e-9);
        let mut level = SessionWeather::new(&LevelWeather::default(), Vec3::NEG_Y);
        level.advance(1000.0);
        assert_eq!(level.clock, None);
        assert_eq!(level.time_label(), "LEVEL");
    }

    /// LEVEL without a clock is the level's own look, exactly, and a new
    /// game starts where the level says.
    #[test]
    fn level_is_the_levels_own_look() {
        let level = Environment {
            fog_density: 0.035,
            fog_height: -9.0,
            fog_falloff: 0.08,
            fog_color: Some(Vec3::splat(0.5)),
            ..Default::default()
        };
        let path = SolarPath::default();
        let mut w = SessionWeather::new(&LevelWeather::default(), level.sun_dir);
        assert_eq!(w.frame(&level, &path), (level, level.sun_dir));
        // The fog's overrides go over it.
        w.fog_density = FOG_DENSITIES.iter().position(|p| p.0 == "HEAVY").unwrap();
        w.fog_height = FOG_HEIGHTS.iter().position(|p| p.0 == "GROUND").unwrap();
        let (env, _) = w.frame(&level, &path);
        assert_eq!(
            (env.fog_density, env.fog_height, env.fog_falloff),
            (0.1, -9.0, 0.5)
        );
        // A level's time starts the clock; a weather without one starts where
        // the level's sun is.
        let at = |choice, time| {
            let lw = LevelWeather {
                choice,
                time,
                ..Default::default()
            };
            SessionWeather::new(&lw, -path.sun_at(15.25)).clock
        };
        assert_eq!(at(0, Some(6.5)), Some(6.5));
        assert_eq!(at(0, None), None);
        assert!((at(1, None).unwrap() - 15.25).abs() < 1.0 / 60.0 + 1e-9);
    }

    /// LEVEL with a clock (SUN ONLY): the level's own palette under its sun
    /// on the path, faded in over the horizon; by night there's no moon, so
    /// the light is 0 while the sky still glows towards the sun.
    #[test]
    fn level_with_a_clock_moves_the_sun_only() {
        let level = Environment {
            sky_zenith: Vec3::new(0.3, 0.2, 0.1),
            sun_intensity: 5.0,
            fog_density: 0.035,
            ..Default::default()
        };
        let path = SolarPath::default();
        let mut w = SessionWeather {
            clock: Some(12.0),
            ..Default::default()
        };
        let (env, light) = w.frame(&level, &path);
        assert_eq!(light, -path.sun_at(12.0), "the sun is on its path");
        assert_eq!(env.sun_dir, light);
        assert_eq!(
            Environment {
                sun_dir: level.sun_dir,
                ..env
            },
            level,
            "everything but the sun is the level's"
        );
        // Sunrise: the sun fades in, up to the level's own strength.
        let (rise, _) = path.day().unwrap();
        w.clock = Some(rise);
        assert!(w.frame(&level, &path).0.sun_intensity < 1e-3);
        w.clock = Some(rise + 1.0);
        assert_eq!(w.frame(&level, &path).0.sun_intensity, 5.0);
        // Night: no moon. The light points the moon's way at 0 strength, and
        // the sky glows towards the sun, below the horizon.
        w.clock = Some(1.0);
        let (env, light) = w.frame(&level, &path);
        assert_eq!(env.sun_intensity, 0.0, "a moon under LEVEL");
        assert!(
            light.y < 0.0 && env.sun_dir.y > 0.0,
            "{light} {}",
            env.sun_dir
        );
        assert_eq!(env.sky_zenith, level.sky_zenith);
    }

    /// A weather's frame: the key at the warped hour, the fog's height the
    /// level's, the light the sun's, faded, or the moon's.
    #[test]
    fn a_weather_makes_the_frame() {
        let level = Environment {
            fog_height: -9.0,
            exposure: 1.5,
            ..Default::default()
        };
        let path = SolarPath::default();
        let mut w = SessionWeather {
            choice: 1,
            clock: Some(12.0),
            ..Default::default()
        };
        let (env, light) = w.frame(&level, &path);
        assert_eq!(env.sky_zenith, CLEAR_DAY.sky_zenith);
        assert_eq!((env.fog_height, env.exposure), (-9.0, 1.5));
        assert_eq!(light, -path.sun_at(12.0));
        assert_eq!(env.sun_dir, light, "by day the glow is the light's");
        assert_eq!(env.sun_intensity, CLEAR_DAY.sun_intensity);
        w.clock = Some(1.0);
        let (env, light) = w.frame(&level, &path);
        assert_eq!(env.sun_color, CLEAR_NIGHT.moon_color);
        assert_eq!(env.sun_intensity, CLEAR_NIGHT.moon_intensity);
        assert!(
            light.y < 0.0 && env.sun_dir.y > 0.0,
            "the moon lights, the sun glows"
        );
    }

    /// A change of weather starts from what was on screen, eases to the new
    /// weather over TRANSITION_HOURS, and a new pick mid-way starts from
    /// where the last one had got to: no frame jumps.
    #[test]
    fn a_change_of_weather_is_continuous() {
        let level = Environment::default();
        let path = SolarPath::default();
        let mut w = SessionWeather {
            choice: 1,
            clock: Some(12.0),
            ..Default::default() // paused: the keys stay put
        };
        let (before, _) = w.frame(&level, &path);
        w.cycle_weather();
        assert_eq!(w.choice, 2);
        assert_eq!(w.transition(), Some(0.0));
        let (start, _) = w.frame(&level, &path);
        assert_eq!(start, before, "the change starts from what was shown");
        let target = WEATHERS[1].sample(reference_hour(12.0, path.day()));
        let mut last = start.sky_zenith.x;
        // Paused, a change still runs at TRANSITION_MIN_SPEED: 180 s.
        for _ in 0..8 {
            w.advance(10.0);
            let (env, _) = w.frame(&level, &path);
            assert!(env.sky_zenith.x > last, "not easing towards OVERCAST");
            last = env.sky_zenith.x;
        }
        let (mid, _) = w.frame(&level, &path);
        w.cycle_weather(); // FOGGY, from wherever OVERCAST had got to
        let (restart, _) = w.frame(&level, &path);
        assert_eq!(restart, mid);
        w.advance(TRANSITION_HOURS * 3600.0 / TRANSITION_MIN_SPEED);
        assert_eq!(w.transition(), None);
        let (end, _) = w.frame(&level, &path);
        let foggy = WEATHERS[2].sample(reference_hour(12.0, path.day()));
        assert_eq!(end.sky_zenith, foggy.sky_zenith);
        assert_ne!(target.sky_zenith, foggy.sky_zenith);
        // To and from LEVEL is at once.
        w.cycle_weather();
        assert_eq!((w.choice, w.transition()), (0, None));
        w.cycle_weather();
        assert_eq!((w.choice, w.transition()), (1, None));
    }

    /// SPEED cycles its presets; a level's own speed moves on to the next
    /// preset above it.
    #[test]
    fn speed_cycles_its_presets() {
        let mut w = SessionWeather {
            speed: 5.0,
            ..Default::default()
        };
        assert_eq!(w.speed_label(), "5X");
        w.cycle_speed();
        assert_eq!(w.speed_label(), "10X");
        w.speed = 600.0;
        w.cycle_speed();
        assert_eq!(w.speed_label(), "PAUSED");
    }
}
