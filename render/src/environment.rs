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
    /// Exponential distance fog towards the sky colour, per metre.
    pub fog_density: f32,
    /// Starting exposure for the tonemap.
    pub exposure: f32,
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
            exposure: 1.0,
        }
    }
}

/// The specialization constants, in `constant_id` order: the name each
/// shader declares at that id. A shader need not declare them all (the sky
/// has no fog, the mesh no disk); Vulkan ignores map entries a shader lacks.
const SPEC_NAMES: [&str; 19] = [
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
        ]
    }
}

/// An environment's specialization data, owned so that a
/// `vk::SpecializationInfo` borrowing it lives as long as pipeline creation.
pub(crate) struct Specialization {
    entries: [vk::SpecializationMapEntry; SPEC_COUNT],
    data: [f32; SPEC_COUNT],
}

impl Specialization {
    pub(crate) fn new(env: &Environment) -> Self {
        let size = std::mem::size_of::<f32>();
        Self {
            entries: std::array::from_fn(|i| vk::SpecializationMapEntry {
                constant_id: i as u32,
                offset: (i * size) as u32,
                size,
            }),
            data: env.spec_values(),
        }
    }

    pub(crate) fn info(&self) -> vk::SpecializationInfo<'_> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&self.data),
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
        assert_eq!(last.constant_id, 18);
        assert_eq!(spec.data[last.offset as usize / 4], 0.5);
    }
}
