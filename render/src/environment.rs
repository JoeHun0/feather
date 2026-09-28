//! A level's atmosphere (§13): the sun, the analytic sky, fog and exposure.
//!
//! The sky palette, sun and fog reach `mesh.frag` and `sky.frag` as
//! **specialization constants**, baked into both pipelines when a session
//! builds them. A level's atmosphere is fixed for its session, so this costs
//! nothing per frame and needs no descriptor or push-constant space, and one
//! struct feeds both shaders, which used to repeat the same constants by hand.
//! Changing it mid-session (weather, time of day) would need the globals UBO
//! instead (§25).
//!
//! The sun's *direction* and the exposure are not baked: they already travel
//! per frame (push constants), so the app takes them from here at session
//! start.

use ash::vk;
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

/// The specialization constants, in `constant_id` order: the name each
/// shader declares at that id. A shader need not declare them all (the sky
/// has no fog, the mesh no disk); Vulkan ignores map entries a shader lacks.
const SPEC_NAMES: [&str; 26] = [
    "SUN_RADIANCE_R",
    "SUN_RADIANCE_G",
    "SUN_RADIANCE_B",
    "SKY_ZENITH_R",
    "SKY_ZENITH_G",
    "SKY_ZENITH_B",
    "SKY_HORIZON_R",
    "SKY_HORIZON_G",
    "SKY_HORIZON_B",
    "SKY_GROUND_R",
    "SKY_GROUND_G",
    "SKY_GROUND_B",
    "SKY_SUN_R",
    "SKY_SUN_G",
    "SKY_SUN_B",
    "SKY_INTENSITY",
    "SUN_GLOW",
    "SUN_DISK",
    "FOG_DENSITY",
    "FOG_HEIGHT",
    "FOG_FALLOFF",
    "FOG_R",
    "FOG_G",
    "FOG_B",
    "FOG_SKY",
    "FOG_SUN",
];
const SPEC_COUNT: usize = SPEC_NAMES.len();

impl Environment {
    /// This environment's specialization constants, by `constant_id`.
    fn spec_values(&self) -> [f32; SPEC_COUNT] {
        let sun = self.sun_color * self.sun_intensity;
        let [zr, zg, zb] = self.sky_zenith.to_array();
        let [hr, hg, hb] = self.sky_horizon.to_array();
        let [gr, gg, gb] = self.sky_ground.to_array();
        let [sr, sg, sb] = self.sky_sun_color.to_array();
        // FOG_SKY = 1 takes the sky's colour; FOG_R/G/B are then unused.
        let fog = self.fog_color.unwrap_or(Vec3::splat(0.5));
        [
            sun.x,
            sun.y,
            sun.z,
            zr,
            zg,
            zb,
            hr,
            hg,
            hb,
            gr,
            gg,
            gb,
            sr,
            sg,
            sb,
            self.sky_intensity,
            self.sun_glow,
            self.sun_disk,
            self.fog_density,
            self.fog_height,
            self.fog_falloff,
            fog.x,
            fog.y,
            fog.z,
            if self.fog_color.is_some() { 0.0 } else { 1.0 },
            self.fog_sun,
        ]
    }
}

/// `mesh.frag`'s `MASKED` constant (§5): the masked-material variant of the
/// main pipeline. Outside the environment's id range on purpose.
pub(crate) const MASKED_ID: u32 = 100;

/// An environment's specialization data (and optionally `MASKED`), owned so
/// that a `vk::SpecializationInfo` borrowing it lives as long as pipeline
/// creation. Every value is 4 bytes: an f32, or a VkBool32.
pub(crate) struct Specialization {
    entries: Vec<vk::SpecializationMapEntry>,
    data: Vec<u32>,
}

impl Specialization {
    pub(crate) fn new(env: &Environment) -> Self {
        let mut spec = Self {
            entries: Vec::new(),
            data: Vec::new(),
        };
        for (id, v) in env.spec_values().into_iter().enumerate() {
            spec.push(id as u32, v.to_bits());
        }
        spec
    }

    /// The same, with `MASKED` set: the main pipeline for masked materials.
    pub(crate) fn masked(mut self) -> Self {
        self.push(MASKED_ID, vk::TRUE);
        self
    }

    fn push(&mut self, constant_id: u32, value: u32) {
        self.entries.push(vk::SpecializationMapEntry {
            constant_id,
            offset: (self.data.len() * 4) as u32,
            size: 4,
        });
        self.data.push(value);
    }

    pub(crate) fn info(&self) -> vk::SpecializationInfo<'_> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().cast::<u8>(),
                std::mem::size_of_val(self.data.as_slice()),
            )
        };
        vk::SpecializationInfo::default()
            .map_entries(&self.entries)
            .data(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The `(name, default)` of every specialization constant a SPIR-V module
    /// declares, by `SpecId`. Names come from `OpName`, so need a debug build
    /// of the shaders (tests are one); a release build yields `None` names.
    fn spec_constants(spv: &[u8]) -> HashMap<u32, (Option<String>, f32)> {
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(words[0], 0x0723_0203, "not SPIR-V");
        let (mut names, mut spec_ids, mut values) =
            (HashMap::new(), HashMap::new(), HashMap::new());
        let mut i = 5; // past the header
        while i < words.len() {
            let (count, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
            let args = &words[i + 1..i + count];
            match op {
                // OpName: target id, then a nul-terminated UTF-8 string.
                5 => {
                    let bytes: Vec<u8> = args[1..].iter().flat_map(|w| w.to_le_bytes()).collect();
                    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                    names.insert(args[0], String::from_utf8_lossy(&bytes[..end]).into_owned());
                }
                // OpDecorate target SpecId(1) literal.
                71 if args[1] == 1 => {
                    spec_ids.insert(args[0], args[2]);
                }
                // OpSpecConstant result-type result-id value (a 32-bit float here).
                50 => {
                    values.insert(args[1], f32::from_bits(args[2]));
                }
                // OpSpecConstantTrue / OpSpecConstantFalse: bools, as 1 / 0.
                48 | 49 => {
                    values.insert(args[1], if op == 48 { 1.0 } else { 0.0 });
                }
                _ => {}
            }
            i += count.max(1);
        }
        spec_ids
            .iter()
            .map(|(target, &id)| (id, (names.get(target).cloned(), values[target])))
            .collect()
    }

    /// Every specialization constant the two shaders declare must be in the
    /// map at the id the Rust side fills, under the name it expects, and with
    /// a GLSL default equal to `Environment::default()`'s value. A mismatch
    /// would compile and silently shade with the wrong number; this catches
    /// it without a device. Between them the shaders must use every entry.
    #[test]
    fn shaders_declare_the_environment_constants() {
        let defaults = Environment::default().spec_values();
        let mut used = [false; SPEC_COUNT];
        let shaders: [(&str, &[u8]); 2] = [
            (
                "mesh.frag",
                include_bytes!(concat!(env!("OUT_DIR"), "/mesh.frag.spv")),
            ),
            (
                "sky.frag",
                include_bytes!(concat!(env!("OUT_DIR"), "/sky.frag.spv")),
            ),
        ];
        for (shader, spv) in shaders {
            for (id, (name, value)) in spec_constants(spv) {
                if id == MASKED_ID {
                    continue; // not the environment's; checked below
                }
                let i = id as usize;
                assert!(i < SPEC_COUNT, "{shader}: constant_id {id} isn't mapped");
                if let Some(name) = name {
                    assert_eq!(name, SPEC_NAMES[i], "{shader}: constant_id {id}");
                }
                assert!(
                    (value - defaults[i]).abs() <= 1e-6 * defaults[i].abs().max(1.0),
                    "{shader}: {} defaults to {value} in GLSL, {} in Rust",
                    SPEC_NAMES[i],
                    defaults[i]
                );
                used[i] = true;
            }
        }
        for (i, u) in used.iter().enumerate() {
            assert!(u, "no shader declares {}", SPEC_NAMES[i]);
        }
    }

    /// `mesh.frag` declares `MASKED` at the id `Specialization::masked`
    /// sets, defaulting to false: the opaque pipeline, which sets nothing
    /// there, must not cut anything out.
    #[test]
    fn mesh_frag_declares_masked() {
        let spv = include_bytes!(concat!(env!("OUT_DIR"), "/mesh.frag.spv"));
        let (name, value) = spec_constants(spv)
            .remove(&MASKED_ID)
            .expect("mesh.frag declares no constant at MASKED_ID");
        if let Some(name) = name {
            assert_eq!(name, "MASKED");
        }
        assert_eq!(value, 0.0, "MASKED must default to false");
        let spec = Specialization::new(&Environment::default()).masked();
        let entry = spec.entries.last().unwrap();
        assert_eq!(
            (entry.constant_id, spec.data.last()),
            (MASKED_ID, Some(&vk::TRUE))
        );
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

    #[test]
    fn the_map_lays_the_values_out_in_order() {
        let env = Environment {
            fog_density: 0.5,
            ..Default::default()
        };
        let spec = Specialization::new(&env);
        let info = spec.info();
        assert_eq!(info.map_entry_count as usize, SPEC_COUNT);
        assert_eq!(info.data_size, SPEC_COUNT * 4);
        let last = spec.entries[SPEC_COUNT - 1];
        assert_eq!(last.constant_id, 25);
        assert_eq!(f32::from_bits(spec.data[18]), 0.5);
    }
}
