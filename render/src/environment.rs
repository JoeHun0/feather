//! A level's atmosphere (§13): the sun, the analytic sky, fog and exposure.
//!
//! All of it travels per frame, so the weather and the time of day can change
//! any of it (§13's weather engine):
//! - the sky palette, the sun's colour and the fog as an `Atmosphere`, the
//!   tail of the globals UBO that `mesh.frag` and `sky.frag` both read;
//! - the sun's *direction* and the exposure as push constants.
//!
//! A level's `environment` marker fills an `Environment`, and the app turns it
//! into those each frame.

use glam::Vec3;

/// See the module docs. `Default` is exactly the look every scene had before
/// scenes could choose one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Environment {
    /// The direction sunlight travels (world space, unit length).
    pub sun_dir: Vec3,
    /// Colour of the direct sunlight; its radiance is this × `sun_intensity`.
    pub sun_color: Vec3,
    pub sun_intensity: f32,
    /// Analytic sky, straight up, at the horizon and below it. Also what the
    /// ambient (IBL) light and the fog are made of.
    pub sky_zenith: Vec3,
    pub sky_horizon: Vec3,
    pub sky_ground: Vec3,
    /// Tint of the sun's glow and disk in the sky.
    pub sky_sun_color: Vec3,
    /// Scales the whole sky, and so the ambient light.
    pub sky_intensity: f32,
    /// Strength of the soft haze round the sun (sky and reflections).
    pub sun_glow: f32,
    /// Strength of the sharp sun disk in the visible sky; 0 hides the sun,
    /// as behind cloud.
    pub sun_disk: f32,
    /// Fog density, per metre, at `fog_height`.
    pub fog_density: f32,
    /// The world height `fog_density` is given at.
    pub fog_height: f32,
    /// How fast the fog thins with height, per metre: the density falls by
    /// e every `1 / fog_falloff` metres. 0 is uniform fog, the old model,
    /// which leaves the sky background unfogged.
    pub fog_falloff: f32,
    /// The fog's own colour, or `None` to take the sky's in each direction.
    pub fog_color: Option<Vec3>,
    /// Strength of the glow towards the sun in the fog (tinted by
    /// `sky_sun_color`).
    pub fog_sun: f32,
    /// Exposure: compensation on the metered value with auto-exposure (1 =
    /// as metered), or the exposure itself without it. `[`/`]` adjust it.
    pub exposure: f32,
    /// The range auto-exposure may choose from (§13).
    pub exposure_min: f32,
    pub exposure_max: f32,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            sun_dir: Vec3::new(-0.4, -1.0, -0.3).normalize(),
            sun_color: Vec3::ONE,
            sun_intensity: 8.0,
            sky_zenith: Vec3::new(0.10, 0.22, 0.55),
            sky_horizon: Vec3::new(0.55, 0.65, 0.85),
            sky_ground: Vec3::new(0.17, 0.18, 0.19),
            sky_sun_color: Vec3::new(1.0, 0.95, 0.85),
            sky_intensity: 1.0,
            sun_glow: 0.6,
            sun_disk: 60.0,
            fog_density: 0.010,
            fog_height: 0.0,
            fog_falloff: 0.0,
            fog_color: None,
            fog_sun: 0.0,
            exposure: 1.0,
            exposure_min: 0.125,
            exposure_max: 8.0,
        }
    }
}

/// The atmosphere as the shaders read it each frame (§13): the tail of
/// mesh.frag's `Globals`, whose block sky.frag declares too (tests check
/// both, field for field). std140: every field a vec4.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Atmosphere {
    /// rgb = the directional light's colour × intensity.
    pub sun_radiance: [f32; 4],
    /// xyz = the unit direction towards the sun, whose glow the sky shows.
    /// The directional light may be the moon (§13's weather).
    pub sun_dir: [f32; 4],
    /// rgb, w = the sky's intensity.
    pub sky_zenith: [f32; 4],
    /// rgb, w = the sun's glow.
    pub sky_horizon: [f32; 4],
    /// rgb, w = the sun's disk.
    pub sky_ground: [f32; 4],
    /// rgb = the glow's and disk's tint, w = the fog's glow towards the sun.
    pub sky_sun: [f32; 4],
    /// rgb = the fog's own colour, w = 1 to take the sky's instead.
    pub fog_color: [f32; 4],
    /// x = density at y = height, z = falloff.
    pub fog: [f32; 4],
}

impl Environment {
    /// What the shaders read of this environment, each frame. The exposure
    /// travels separately, as does the directional light's direction: the
    /// sun's by day, but the moon's at night, while the sky still glows
    /// towards the sun (`sun_dir`).
    pub fn atmosphere(&self) -> Atmosphere {
        let v = |c: Vec3, w: f32| c.extend(w).to_array();
        Atmosphere {
            sun_radiance: v(self.sun_color * self.sun_intensity, 0.0),
            sun_dir: v(-self.sun_dir, 0.0),
            sky_zenith: v(self.sky_zenith, self.sky_intensity),
            sky_horizon: v(self.sky_horizon, self.sun_glow),
            sky_ground: v(self.sky_ground, self.sun_disk),
            sky_sun: v(self.sky_sun_color, self.fog_sun),
            // Without a colour of its own the fog takes the sky's, and rgb is
            // unused.
            fog_color: match self.fog_color {
                Some(c) => v(c, 0.0),
                None => v(Vec3::splat(0.5), 1.0),
            },
            fog: [self.fog_density, self.fog_height, self.fog_falloff, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How many specialization constants a SPIR-V module declares: its
    /// `OpDecorate … SpecId` instructions.
    fn spec_constant_count(spv: &[u8]) -> usize {
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(words[0], 0x0723_0203, "not SPIR-V");
        let mut count = 0;
        let mut i = 5; // past the header
        while i < words.len() {
            let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
            // OpDecorate (71) target SpecId (1) literal.
            if op == 71 && words[i + 2] == 1 {
                count += 1;
            }
            i += len.max(1);
        }
        count
    }

    /// Nothing of the atmosphere is baked into a pipeline any more: all of it
    /// travels per frame, so the weather can change it (§13). A constant
    /// left behind would compile, and then ignore the weather.
    #[test]
    fn the_shaders_bake_nothing() {
        for (shader, spv) in [
            (
                "mesh.frag",
                &include_bytes!(concat!(env!("OUT_DIR"), "/mesh.frag.spv"))[..],
            ),
            (
                "sky.frag",
                &include_bytes!(concat!(env!("OUT_DIR"), "/sky.frag.spv"))[..],
            ),
        ] {
            assert_eq!(spec_constant_count(spv), 0, "{shader}");
        }
    }

    /// `Environment::default()` reaches the shaders as exactly the values
    /// their specialization constants used to default to, each in its slot:
    /// the look every level without an `environment` marker had.
    #[test]
    fn the_default_atmosphere_is_the_old_look() {
        let a = Environment::default().atmosphere();
        assert_eq!(a.sun_radiance[..3], [8.0, 8.0, 8.0]);
        // The glow's direction was `normalize(-light_dir)`, the sun's.
        let to_sun = -Environment::default().sun_dir;
        assert_eq!(a.sun_dir[..3], to_sun.to_array());
        assert_eq!(a.sky_zenith, [0.10, 0.22, 0.55, 1.0]); // w: intensity
        assert_eq!(a.sky_horizon, [0.55, 0.65, 0.85, 0.6]); // w: glow
        assert_eq!(a.sky_ground, [0.17, 0.18, 0.19, 60.0]); // w: disk
        assert_eq!(a.sky_sun, [1.0, 0.95, 0.85, 0.0]); // w: fog_sun
        assert_eq!(a.fog_color, [0.5, 0.5, 0.5, 1.0]); // w: the sky's colour
        assert_eq!(a.fog[..3], [0.010, 0.0, 0.0]);
    }

    /// Every part of an environment the shaders use reaches them through
    /// `atmosphere()`; the exposure travels apart.
    #[test]
    fn the_atmosphere_carries_every_field() {
        let base = Environment::default();
        let with = |change: fn(&mut Environment)| {
            let mut env = base;
            change(&mut env);
            env
        };
        let changed: [(&str, Environment); 15] = [
            ("sun_dir", with(|e| e.sun_dir = Vec3::NEG_Y)),
            ("sun_color", with(|e| e.sun_color = Vec3::X)),
            ("sun_intensity", with(|e| e.sun_intensity = 2.0)),
            ("sky_zenith", with(|e| e.sky_zenith = Vec3::X)),
            ("sky_horizon", with(|e| e.sky_horizon = Vec3::X)),
            ("sky_ground", with(|e| e.sky_ground = Vec3::X)),
            ("sky_sun_color", with(|e| e.sky_sun_color = Vec3::X)),
            ("sky_intensity", with(|e| e.sky_intensity = 2.0)),
            ("sun_glow", with(|e| e.sun_glow = 0.1)),
            ("sun_disk", with(|e| e.sun_disk = 0.0)),
            ("fog_density", with(|e| e.fog_density = 0.5)),
            ("fog_height", with(|e| e.fog_height = 3.0)),
            ("fog_falloff", with(|e| e.fog_falloff = 0.2)),
            ("fog_color", with(|e| e.fog_color = Some(Vec3::X))),
            ("fog_sun", with(|e| e.fog_sun = 0.5)),
        ];
        for (field, env) in changed {
            assert_ne!(env.atmosphere(), base.atmosphere(), "{field}");
        }
        let apart = Environment {
            exposure: 2.0,
            exposure_min: 1.0,
            exposure_max: 4.0,
            ..base
        };
        assert_eq!(apart.atmosphere(), base.atmosphere());
    }

    /// The shaders' `fog_optical_depth`, transcribed: the reference the
    /// derivation is tested on (the GLSL can't run on the CPU).
    fn fog_optical_depth(env: &Environment, eye: Vec3, dir: Vec3, dist: f32) -> f32 {
        if env.fog_falloff == 0.0 {
            return if dist < 0.0 {
                1e9
            } else {
                env.fog_density * dist
            };
        }
        let base = env.fog_density * (-env.fog_falloff * (eye.y - env.fog_height)).exp();
        let kv = env.fog_falloff * dir.y;
        if dist < 0.0 {
            return if kv > 1e-6 { base / kv } else { 1e9 };
        }
        let x = kv * dist;
        if x.abs() < 0.01 {
            return base * dist * (1.0 - x * (0.5 - x / 6.0));
        }
        base * (1.0 - (-x.max(-80.0)).exp()) / kv
    }

    /// The same optical depth by brute force: Simpson's rule over the
    /// density along the ray, in f64.
    fn integrated(env: &Environment, eye: Vec3, dir: Vec3, dist: f64) -> f64 {
        let rho = |t: f64| {
            let y = eye.y as f64 + t * dir.y as f64;
            env.fog_density as f64 * (-(env.fog_falloff as f64) * (y - env.fog_height as f64)).exp()
        };
        let n = 20_000;
        let h = dist / n as f64;
        let mut sum = rho(0.0) + rho(dist);
        for i in 1..n {
            sum += rho(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * h / 3.0
    }

    /// The closed form of the height fog integral against numeric
    /// integration: climbing, descending and level rays, a thin and a thick
    /// layer, eyes above and inside it.
    #[test]
    fn height_fog_is_the_integral_of_its_density() {
        for falloff in [0.0, 0.02, 0.08, 0.5] {
            let env = Environment {
                fog_density: 0.035,
                fog_height: -9.0,
                fog_falloff: falloff,
                ..Default::default()
            };
            for eye_y in [-7.2, 0.0, 12.0] {
                for dir_y in [-0.9f32, -0.3, -0.01, 0.0, 0.01, 0.3, 0.9] {
                    let dir = Vec3::new((1.0 - dir_y * dir_y).sqrt(), dir_y, 0.0);
                    let eye = Vec3::new(0.0, eye_y, 0.0);
                    for dist in [0.5f32, 10.0, 80.0, 200.0] {
                        let got = fog_optical_depth(&env, eye, dir, dist) as f64;
                        let want = integrated(&env, eye, dir, dist as f64);
                        if want > 50.0 {
                            // Total fog either way; the shader clamps the
                            // exponent there rather than overflow.
                            assert!(got > 50.0, "k {falloff} dir.y {dir_y} dist {dist}: {got}");
                            continue;
                        }
                        assert!(
                            (got - want).abs() <= 1e-4 * want.max(1e-3),
                            "k {falloff} eye {eye_y} dir.y {dir_y} dist {dist}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }

    /// Uniform fog (no falloff) is exactly the old `density · distance`,
    /// which is why levels without a height layer look as they did. And to
    /// infinity, a climbing ray sees ρ(eye) / (k · dir.y), a level or
    /// descending one sees total fog.
    #[test]
    fn uniform_fog_and_the_sky_limit() {
        let old = Environment::default();
        let (eye, dir) = (Vec3::new(3.0, -7.2, 1.0), Vec3::new(0.6, -0.8, 0.0));
        for dist in [1.0, 37.5, 200.0] {
            assert_eq!(
                fog_optical_depth(&old, eye, dir, dist),
                old.fog_density * dist
            );
        }
        let env = Environment {
            fog_density: 0.035,
            fog_height: -9.0,
            fog_falloff: 0.08,
            ..Default::default()
        };
        let up = Vec3::new(0.0, 1.0, 0.0);
        let rho = 0.035 * (-0.08f32 * (-7.2 + 9.0)).exp();
        let sky = fog_optical_depth(&env, eye, up, -1.0);
        assert!((sky - rho / 0.08).abs() < 1e-5, "{sky}");
        // It is the long-distance limit of the finite form.
        let far = fog_optical_depth(&env, eye, up, 1e4);
        assert!((far - sky).abs() < 1e-4, "{far} vs {sky}");
        assert!(fog_optical_depth(&env, eye, Vec3::X, -1.0) >= 1e9);
        assert!(fog_optical_depth(&env, eye, -up, -1.0) >= 1e9);
    }

    /// `sky()`, `fog_optical_depth()` and `fog_color()` are written out in
    /// both shaders (no includes). The sky and the fog on geometry must be
    /// the same functions, or the horizon shows a seam; so their texts must
    /// match exactly.
    #[test]
    fn the_shaders_share_sky_and_fog_functions() {
        let body = |src: &str, sig: &str| -> String {
            let start = src.find(sig).unwrap_or_else(|| panic!("no `{sig}`"));
            let len = src[start..].find("\n}\n").expect("function end") + 3;
            src[start..start + len].to_string()
        };
        let mesh = include_str!("../shaders/mesh.frag");
        let sky = include_str!("../shaders/sky.frag");
        for sig in [
            "vec3 sky(vec3 d)",
            "float fog_optical_depth(",
            "vec3 fog_color(vec3 dir)",
        ] {
            assert_eq!(body(mesh, sig), body(sky, sig), "`{sig}` differs");
        }
    }

    /// The block both shaders read the atmosphere from: mesh.frag's
    /// `Globals`, which sky.frag declares text for text (with the same
    /// cascade count, which sizes it). Its tail is `Atmosphere`, field for
    /// field, and both shaders name its parts with the same `#define`s, so
    /// the shared functions above read the same values in each.
    #[test]
    fn the_shaders_read_the_atmosphere_from_the_globals() {
        let mesh = include_str!("../shaders/mesh.frag");
        let sky = include_str!("../shaders/sky.frag");
        let block = |src: &str| -> String {
            let start = src
                .find("layout(set = 0, binding = 4) uniform Globals {")
                .expect("Globals");
            let len = src[start..].find("} g;").expect("end of Globals") + 4;
            src[start..start + len].to_string()
        };
        assert_eq!(block(mesh), block(sky), "the Globals blocks differ");
        let cascades = format!(
            "const int SHADOW_CASCADES = {};",
            feather_gfx::SHADOW_CASCADES
        );
        for (name, src) in [("mesh.frag", mesh), ("sky.frag", sky)] {
            assert!(src.contains(&cascades), "{name} lacks `{cascades}`");
        }

        let fields: Vec<String> = block(mesh)
            .lines()
            .filter_map(|l| {
                let l = l.trim().strip_prefix("vec4 ")?;
                Some(l[..l.find(';')?].to_string())
            })
            .collect();
        let atmosphere = [
            "sun_radiance",
            "sun_dir",
            "sky_zenith",
            "sky_horizon",
            "sky_ground",
            "sky_sun",
            "fog_color",
            "fog",
        ];
        assert_eq!(fields[fields.len() - atmosphere.len()..], atmosphere);
        assert_eq!(
            std::mem::size_of::<Atmosphere>(),
            atmosphere.len() * 16,
            "all vec4"
        );

        let defines = |src: &str| -> Vec<String> {
            let mut d: Vec<String> = src
                .lines()
                .filter(|l| l.starts_with("#define "))
                .map(String::from)
                .collect();
            d.sort();
            d
        };
        assert_eq!(defines(mesh), defines(sky), "the #defines differ");
        for def in [
            "#define SUN_RADIANCE g.sun_radiance.rgb",
            "#define SUN_DIR g.sun_dir.xyz",
            "#define SKY_ZENITH g.sky_zenith.rgb",
            "#define SKY_INTENSITY g.sky_zenith.w",
            "#define SKY_HORIZON g.sky_horizon.rgb",
            "#define SUN_GLOW g.sky_horizon.w",
            "#define SKY_GROUND g.sky_ground.rgb",
            "#define SUN_DISK g.sky_ground.w",
            "#define SUN_COLOR g.sky_sun.rgb",
            "#define FOG_SUN g.sky_sun.w",
            "#define FOG_COLOR g.fog_color.rgb",
            "#define FOG_SKY g.fog_color.w",
            "#define FOG_DENSITY g.fog.x",
            "#define FOG_HEIGHT g.fog.y",
            "#define FOG_FALLOFF g.fog.z",
        ] {
            assert!(mesh.contains(def), "mesh.frag lacks `{def}`");
        }
    }

    /// sky.frag's push block is the size the pass pushes.
    #[test]
    fn the_sky_pushes_its_block() {
        let sky = include_str!("../shaders/sky.frag");
        let push = &sky[sky.find("uniform Push {").unwrap()..];
        let push = &push[..push.find("} pc;").unwrap()];
        let bytes: usize = push
            .lines()
            .map(|l| match l.trim().split(' ').next() {
                Some("mat4") => 64,
                Some("vec4") => 16,
                _ => 0,
            })
            .sum();
        assert_eq!(bytes as u32, crate::sky::SKY_PUSH_SIZE);
    }
}
