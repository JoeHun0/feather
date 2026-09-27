//! Audio (§20): a kira mixer with SFX and AMBIENCE buses under the master
//! track, a listener that follows the camera, and sounds synthesised at
//! startup (no sound files yet).
//!
//! - **SFX** is the player's own: footsteps, jump, landing. Not spatial.
//!   Recorded (Kenney Impact Sounds, CC0, via `tools/fetch_assets.py`) when
//!   the pack is fetched, synthesised otherwise: see `SoundSet`.
//! - **AMBIENCE** holds one spatial sub-track per visible lamp, looping a hum
//!   that pans with the listener and fades out to the light's radius.
//!
//! `Audio` is generic over kira's `Backend`, so the tests run the real mixer
//! through a capture backend and check what it actually produced.

use std::path::Path;
use std::sync::Arc;

use glam::{Quat, Vec3};
use kira::backend::Backend;
use kira::listener::ListenerHandle;
use kira::sound::static_sound::StaticSoundData;
use kira::sound::static_sound::StaticSoundSettings;
use kira::track::{
    MainTrackBuilder, SpatialTrackBuilder, SpatialTrackDistances, SpatialTrackHandle, TrackBuilder,
    TrackHandle,
};
use kira::{AudioManager, AudioManagerSettings, Decibels, Frame, Tween};

/// Sample rate the sounds are synthesised at; kira resamples to the device.
const SAMPLE_RATE: u32 = 48_000;

/// Volumes in percent (`config/audio.toml`, the SOUND menu).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioSettings {
    pub master: u32,
    pub sfx: u32,
    pub ambience: u32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            master: 80,
            sfx: 100,
            ambience: 100,
        }
    }
}

/// What the SOUND menu's volume rows step through, in percent.
pub const VOLUME_STEPS: [u32; 5] = [0, 25, 50, 75, 100];

/// The next volume step up, wrapping; a value between steps goes to the next
/// step above it.
pub fn step_volume(v: u32) -> u32 {
    VOLUME_STEPS
        .into_iter()
        .find(|&s| s > v)
        .unwrap_or(VOLUME_STEPS[0])
}

/// A percentage as kira gain: 100% is 0 dB, 0% is silence.
pub fn percent_to_db(p: u32) -> Decibels {
    if p == 0 {
        Decibels::SILENCE
    } else {
        Decibels(20.0 * (p.min(100) as f32 / 100.0).log10())
    }
}

/// A sound the game asks for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SoundEvent {
    Step,
    Jump,
    /// Touching down after falling at this speed (m/s).
    Land(f32),
}

// ---- synthesis ----

/// Deterministic white noise in [-1, 1] (xorshift), so a sound is the same on
/// every run.
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn frames(samples: Vec<f32>) -> Arc<[Frame]> {
    samples.into_iter().map(Frame::from_mono).collect()
}

fn seconds(s: f32) -> usize {
    (s * SAMPLE_RATE as f32) as usize
}

/// A footstep: ~70 ms of low-passed noise with a 2 ms attack and a fast
/// exponential decay. `seed` varies the grain, so steps don't sound identical.
pub fn footstep(seed: u32) -> Vec<f32> {
    let mut noise = Noise(0x9e37_79b9 ^ seed.wrapping_mul(2_654_435_761).max(1));
    let mut lp = 0.0f32;
    (0..seconds(0.07))
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            lp += 0.15 * (noise.next() - lp);
            let env = (t / 0.002).min(1.0) * (-t / 0.016).exp();
            lp * env * 1.6
        })
        .collect()
}

/// A jump: a short, brighter burst (noise minus its low-passed part).
pub fn jump() -> Vec<f32> {
    let mut noise = Noise(0x1234_5678);
    let mut lp = 0.0f32;
    (0..seconds(0.1))
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            let n = noise.next();
            lp += 0.2 * (n - lp);
            let env = (t / 0.004).min(1.0) * (-t / 0.03).exp();
            (n - lp) * env * 0.35
        })
        .collect()
}

/// A landing: an ~80 Hz thump with a short click on top.
pub fn land() -> Vec<f32> {
    let mut noise = Noise(0x0bad_cafe);
    (0..seconds(0.18))
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            let thump = (std::f32::consts::TAU * 80.0 * t).sin() * (-t / 0.05).exp() * 0.8;
            let click = noise.next() * (-t / 0.004).exp() * 0.3;
            thump + click
        })
        .collect()
}

/// A lamp's electrical hum: one second of 60 Hz plus harmonics. Every partial
/// fits a whole number of periods in the second, so the loop has no seam.
pub fn hum() -> Vec<f32> {
    let tau = std::f32::consts::TAU;
    (0..SAMPLE_RATE as usize)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            ((tau * 60.0 * t).sin() * 0.5
                + (tau * 120.0 * t).sin() * 0.25
                + (tau * 180.0 * t).sin() * 0.1)
                * 0.3
        })
        .collect()
}

/// Fade the last 5 ms of a one-shot to zero, so a sound cut off while still
/// ringing (the landing's thump) ends without a click.
fn fade_out(mut samples: Vec<f32>) -> Vec<f32> {
    let n = seconds(0.005).min(samples.len());
    let len = samples.len();
    for (i, v) in samples[len - n..].iter_mut().enumerate() {
        *v *= 1.0 - (i + 1) as f32 / n as f32;
    }
    samples
}

/// A sound played once: faded out at the end.
fn one_shot(samples: Vec<f32>) -> StaticSoundData {
    sound(fade_out(samples))
}

/// A sound as-is (the hum loops, so it must not fade).
fn sound(samples: Vec<f32>) -> StaticSoundData {
    StaticSoundData {
        sample_rate: SAMPLE_RATE,
        frames: frames(samples),
        settings: StaticSoundSettings::default(),
        slice: None,
    }
}

// ---- the sound set: recorded where fetched, synthesised otherwise ----

/// Where `tools/fetch_assets.py kenney_impact_sounds` puts the recorded SFX
/// (cwd-relative, like the bake dir).
pub const SOUNDS_DIR: &str = "scratch/assets/kenney_impact_sounds";

/// Peak level every clip is normalised to, per event, so a recorded sound and
/// the synthesised one it replaces play at the same level whichever loaded.
const STEP_PEAK: f32 = 0.5;
const JUMP_PEAK: f32 = 0.35;
const LAND_PEAK: f32 = 0.8;

/// A sound plus the gain (dB) that brings its peak to its event's target.
#[derive(Clone)]
struct Clip {
    data: StaticSoundData,
    gain: f32,
}

impl Clip {
    fn new(data: StaticSoundData, target_peak: f32) -> Self {
        let peak = data
            .frames
            .iter()
            .fold(0.0f32, |m, f| m.max(f.left.abs()).max(f.right.abs()));
        let gain = if peak > 1e-6 {
            20.0 * (target_peak / peak).log10()
        } else {
            0.0
        };
        Self { data, gain }
    }

    /// The sound to play, `extra_db` louder than its normalised level.
    fn play(&self, extra_db: f32) -> StaticSoundData {
        self.data.volume(Decibels(self.gain + extra_db))
    }
}

/// The SFX clips, five variants per event (cycled, so repeats don't sound
/// identical).
pub struct SoundSet {
    steps: Vec<Clip>,
    jumps: Vec<Clip>,
    lands: Vec<Clip>,
    pub recorded: usize,
    pub synthesised: usize,
    /// Why anything fell back, for the log.
    pub notes: Vec<String>,
}

impl SoundSet {
    /// Everything synthesised: no files needed (the mixer tests).
    #[cfg(test)]
    pub fn synthesised() -> Self {
        Self::load_with(None)
    }

    /// Recorded clips from `dir`; each file that's missing or doesn't decode
    /// falls back to its synthesised sound, so audio never needs the fetch.
    pub fn load(dir: &Path) -> Self {
        Self::load_with(Some(dir))
    }

    fn load_with(dir: Option<&Path>) -> Self {
        let mut set = Self {
            steps: Vec::new(),
            jumps: Vec::new(),
            lands: Vec::new(),
            recorded: 0,
            synthesised: 0,
            notes: Vec::new(),
        };
        let dir = match dir {
            Some(d) if d.is_dir() => Some(d),
            Some(d) => {
                set.notes.push(format!(
                    "{} not found (`python3 tools/fetch_assets.py kenney_impact_sounds`); using synthesised sounds",
                    d.display()
                ));
                None
            }
            None => None,
        };
        for i in 0..5u32 {
            let step = set.clip(
                dir,
                &format!("footstep_concrete_{i:03}.ogg"),
                STEP_PEAK,
                || footstep(i),
            );
            set.steps.push(step);
            let jump = set.clip(
                dir,
                &format!("impactSoft_medium_{i:03}.ogg"),
                JUMP_PEAK,
                jump,
            );
            set.jumps.push(jump);
            let land = set.clip(
                dir,
                &format!("impactSoft_heavy_{i:03}.ogg"),
                LAND_PEAK,
                land,
            );
            set.lands.push(land);
        }
        set
    }

    fn clip(
        &mut self,
        dir: Option<&Path>,
        file: &str,
        peak: f32,
        synth: impl Fn() -> Vec<f32>,
    ) -> Clip {
        if let Some(dir) = dir {
            let path = dir.join(file);
            match StaticSoundData::from_file(&path) {
                Ok(data) => {
                    self.recorded += 1;
                    return Clip::new(data, peak);
                }
                Err(e) => self
                    .notes
                    .push(format!("{file}: {e}; using a synthesised sound")),
            }
        }
        self.synthesised += 1;
        Clip::new(one_shot(synth()), peak)
    }
}

// ---- events from player motion ----

/// Walking distance between footsteps (m).
const STRIDE: f32 = 1.6;
/// Rising faster than this when leaving the ground is a jump; slower is
/// walking off an edge, which makes no sound.
const JUMP_SPEED: f32 = 1.0;
/// Landing after falling faster than this makes a thump; a step down doesn't.
const LAND_SPEED: f32 = 3.0;
/// A position change bigger than this in one frame is a teleport (spawn,
/// respawn), not walking.
const TELEPORT: f32 = 2.0;

/// Turns the player's motion, sampled once per rendered frame (§20: the
/// render clock), into sound events.
#[derive(Default)]
pub struct StepTracker {
    last: Option<(Vec3, bool)>,
    walked: f32,
    /// Fastest fall since leaving the ground (m/s, positive).
    fall: f32,
}

impl StepTracker {
    pub fn update(
        &mut self,
        pos: Vec3,
        vel: Vec3,
        on_ground: bool,
        noclip: bool,
    ) -> Vec<SoundEvent> {
        let mut out = Vec::new();
        let Some((last_pos, was_ground)) = self.last.replace((pos, on_ground)) else {
            return out;
        };
        if noclip {
            self.walked = 0.0;
            self.fall = 0.0;
            return out;
        }
        let moved = (pos - last_pos) * Vec3::new(1.0, 0.0, 1.0);
        let dist = moved.length();
        if dist > TELEPORT {
            return out;
        }
        if !on_ground {
            self.fall = self.fall.max(-vel.y);
        }
        match (was_ground, on_ground) {
            (true, false) if vel.y > JUMP_SPEED => out.push(SoundEvent::Jump),
            (false, true) => {
                if self.fall > LAND_SPEED {
                    out.push(SoundEvent::Land(self.fall));
                }
                self.fall = 0.0;
                // The next step comes half a stride after landing.
                self.walked = STRIDE * 0.5;
            }
            (true, true) => {
                self.walked += dist;
                if self.walked >= STRIDE {
                    self.walked -= STRIDE;
                    out.push(SoundEvent::Step);
                }
            }
            _ => {}
        }
        out
    }
}

// ---- the mixer ----

/// The mixer and the sounds, alive for the whole run.
pub struct Audio<B: Backend> {
    manager: AudioManager<B>,
    listener: ListenerHandle,
    sfx: TrackHandle,
    ambience: TrackHandle,
    sounds: SoundSet,
    hum: StaticSoundData,
    /// Next variant per event (steps, jumps, landings).
    next: [usize; 3],
    /// One spatial track per lamp; dropping a handle removes its track.
    lamps: Vec<SpatialTrackHandle>,
}

fn mint3(v: Vec3) -> mint::Vector3<f32> {
    mint::Vector3 {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}

fn mint_quat(q: Quat) -> mint::Quaternion<f32> {
    mint::Quaternion {
        v: mint3(Vec3::new(q.x, q.y, q.z)),
        s: q.w,
    }
}

impl<B: Backend> Audio<B> {
    /// Start the mixer. Any failure (no device, no ALSA) is returned as text
    /// for the caller to log; the game then runs silent.
    pub fn new(
        mut settings: AudioManagerSettings<B>,
        volumes: &AudioSettings,
        sounds: SoundSet,
    ) -> Result<Self, String>
    where
        B::Error: std::fmt::Debug,
    {
        // Volumes are built in, not set afterwards: a set_volume tweens from
        // 0 dB, so the first sounds would play too loud for its duration.
        settings.main_track_builder = MainTrackBuilder::new().volume(percent_to_db(volumes.master));
        let mut manager = AudioManager::<B>::new(settings).map_err(|e| format!("{e:?}"))?;
        let listener = manager
            .add_listener(mint3(Vec3::ZERO), mint_quat(Quat::IDENTITY))
            .map_err(|e| format!("{e:?}"))?;
        let sfx = manager
            .add_sub_track(TrackBuilder::new().volume(percent_to_db(volumes.sfx)))
            .map_err(|e| format!("{e:?}"))?;
        let ambience = manager
            .add_sub_track(TrackBuilder::new().volume(percent_to_db(volumes.ambience)))
            .map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            manager,
            listener,
            sfx,
            ambience,
            sounds,
            hum: sound(hum()).loop_region(..),
            next: [0; 3],
            lamps: Vec::new(),
        })
    }

    /// Live volume change (the SOUND menu). The default 10 ms tween avoids
    /// zipper clicks.
    pub fn set_volumes(&mut self, v: &AudioSettings) {
        let now = Tween::default();
        self.manager
            .main_track()
            .set_volume(percent_to_db(v.master), now);
        self.sfx.set_volume(percent_to_db(v.sfx), now);
        self.ambience.set_volume(percent_to_db(v.ambience), now);
    }

    /// The listener is the camera: `forward` and `up` from its basis.
    pub fn set_listener(&mut self, eye: Vec3, forward: Vec3, up: Vec3) {
        // kira's listener hears along -Z with +X as its right ear, like a
        // camera looking down -Z.
        let right = forward.cross(up).normalize_or_zero();
        let q = Quat::from_mat3(&glam::Mat3::from_cols(right, up, -forward));
        self.listener.set_position(mint3(eye), Tween::default());
        self.listener
            .set_orientation(mint_quat(q), Tween::default());
    }

    pub fn play(&mut self, event: SoundEvent) {
        let (slot, clips, extra_db) = match event {
            SoundEvent::Step => (0, &self.sounds.steps, 0.0),
            SoundEvent::Jump => (1, &self.sounds.jumps, 0.0),
            SoundEvent::Land(speed) => {
                // Louder the harder the landing, from a third up to full.
                let loud = ((speed - LAND_SPEED) / 7.0).clamp(0.3, 1.0);
                (2, &self.sounds.lands, 20.0 * loud.log10())
            }
        };
        self.next[slot] = (self.next[slot] + 1) % clips.len();
        let data = clips[self.next[slot]].play(extra_db);
        // A full sound queue drops one sound, never the game.
        let _ = self.sfx.play(data);
    }

    /// A humming lamp at `pos`, audible out to `radius` metres.
    pub fn add_lamp(&mut self, pos: Vec3, radius: f32) {
        let builder = SpatialTrackBuilder::new().distances(SpatialTrackDistances {
            min_distance: 1.0,
            max_distance: radius.max(2.0),
        });
        let Ok(mut track) =
            self.ambience
                .add_spatial_sub_track(self.listener.id(), mint3(pos), builder)
        else {
            return;
        };
        if track.play(self.hum.clone()).is_ok() {
            self.lamps.push(track);
        }
    }

    /// Silence every lamp (the session they belonged to ended).
    pub fn clear_lamps(&mut self) {
        self.lamps.clear();
    }

    pub fn lamp_count(&self) -> usize {
        self.lamps.len()
    }

    #[cfg(test)]
    fn backend(&mut self) -> &mut B {
        self.manager.backend_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kira::backend::Renderer;

    /// A backend that renders on demand into a buffer the test can read: the
    /// real mixer, no sound card.
    struct Capture {
        renderer: Option<Renderer>,
    }

    impl Backend for Capture {
        type Settings = ();
        type Error = ();

        fn setup(_: (), _buffer: usize) -> Result<(Self, u32), ()> {
            Ok((Self { renderer: None }, SAMPLE_RATE))
        }

        fn start(&mut self, renderer: Renderer) -> Result<(), ()> {
            self.renderer = Some(renderer);
            Ok(())
        }
    }

    impl Capture {
        /// Mix `n` frames. Commands sent since the last call (play, volume,
        /// positions) take effect first.
        fn render(&mut self, n: usize) -> Vec<Frame> {
            let r = self.renderer.as_mut().expect("started");
            r.on_start_processing();
            let mut out = vec![0.0f32; n * 2];
            for chunk in out.chunks_mut(256) {
                r.process(chunk, 2);
            }
            out.chunks_exact(2)
                .map(|c| Frame {
                    left: c[0],
                    right: c[1],
                })
                .collect()
        }
    }

    fn audio(v: AudioSettings) -> Audio<Capture> {
        Audio::new(AudioManagerSettings::default(), &v, SoundSet::synthesised())
            .expect("capture backend starts")
    }

    #[test]
    fn a_missing_pack_falls_back_to_synthesised_sounds() {
        let set = SoundSet::load(Path::new("/nonexistent/feather-sounds"));
        assert_eq!((set.recorded, set.synthesised), (0, 15));
        assert_eq!(set.notes.len(), 1, "one note, not fifteen: {:?}", set.notes);
        assert!(set.notes[0].contains("fetch_assets.py kenney_impact_sounds"));
    }

    /// A file that's there but doesn't decode falls back for that sound only,
    /// and says so; the rest (missing here) fall back too, each with a note.
    #[test]
    fn a_broken_file_falls_back_for_that_sound_only() {
        let dir = crate::config::test_dir("sounds-broken");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("footstep_concrete_000.ogg"), b"not an ogg").unwrap();
        let set = SoundSet::load(&dir);
        assert_eq!((set.recorded, set.synthesised), (0, 15));
        assert_eq!(set.notes.len(), 15, "{:?}", set.notes);
        assert!(
            set.notes[0].starts_with("footstep_concrete_000.ogg:"),
            "{}",
            set.notes[0]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clips_are_normalised_to_their_events_peak() {
        let set = SoundSet::synthesised();
        for (clips, target) in [
            (&set.steps, STEP_PEAK),
            (&set.jumps, JUMP_PEAK),
            (&set.lands, LAND_PEAK),
        ] {
            for c in clips {
                let peak = c
                    .data
                    .frames
                    .iter()
                    .fold(0.0f32, |m, f| m.max(f.left.abs()));
                let got = peak * 10f32.powf(c.gain / 20.0);
                assert!((got - target).abs() < 1e-4, "peak {got}, want {target}");
            }
        }
    }

    fn peak(frames: &[Frame]) -> (f32, f32) {
        frames.iter().fold((0.0, 0.0), |(l, r), f| {
            (l.max(f.left.abs()), r.max(f.right.abs()))
        })
    }

    #[test]
    fn synthesised_sounds_are_sane() {
        for (name, s, secs) in [
            ("footstep", fade_out(footstep(0)), 0.07),
            ("jump", fade_out(jump()), 0.1),
            ("land", fade_out(land()), 0.18),
        ] {
            assert_eq!(s.len(), seconds(secs), "{name}");
            assert!(s.iter().all(|v| v.is_finite() && v.abs() <= 1.0), "{name}");
            assert!(s.iter().any(|v| v.abs() > 0.05), "{name} is silent");
            let tail = s[s.len() - 50..].iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(tail < 0.02, "{name} ends with a click ({tail})");
            assert_eq!(*s.last().unwrap(), 0.0, "{name} doesn't end at zero");
        }
        assert_ne!(footstep(0), footstep(1), "step variants differ");
        assert_eq!(footstep(3), footstep(3), "deterministic");
        let h = hum();
        assert!(h.iter().all(|v| v.abs() <= 1.0));
        // Seamless loop: the sample after the last is the first.
        let step = (h[1] - h[0]).abs();
        assert!(
            (h[0] - h[h.len() - 1]).abs() <= step * 1.01,
            "hum loop has a seam"
        );
    }

    #[test]
    fn volume_percent_maps_to_decibels() {
        assert_eq!(percent_to_db(100), Decibels::IDENTITY);
        assert_eq!(percent_to_db(0), Decibels::SILENCE);
        assert!((percent_to_db(50).0 + 6.02).abs() < 0.01);
        assert_eq!(step_volume(0), 25);
        assert_eq!(step_volume(100), 0);
        assert_eq!(step_volume(80), 100, "off-step values go up to the next");
    }

    #[test]
    fn steps_follow_the_stride_and_ignore_noclip() {
        let mut t = StepTracker::default();
        let mut steps = 0;
        for i in 0..=100 {
            let pos = Vec3::new(i as f32 * 0.1, 0.0, 0.0); // 10 m on the ground
            steps += t.update(pos, Vec3::ZERO, true, false).len();
        }
        assert_eq!(steps, (10.0 / STRIDE) as usize);
        let mut t = StepTracker::default();
        for i in 0..=100 {
            let pos = Vec3::new(i as f32 * 0.1, 0.0, 0.0);
            assert!(t.update(pos, Vec3::ZERO, true, true).is_empty());
        }
    }

    #[test]
    fn jumps_and_landings_but_not_ledges_or_steps_down() {
        let up = Vec3::Y * 5.0;
        let mut t = StepTracker::default();
        t.update(Vec3::ZERO, Vec3::ZERO, true, false);
        assert_eq!(
            t.update(Vec3::Y * 0.1, up, false, false),
            [SoundEvent::Jump]
        );
        t.update(Vec3::Y, -Vec3::Y * 5.0, false, false);
        assert_eq!(
            t.update(Vec3::ZERO, Vec3::ZERO, true, false),
            [SoundEvent::Land(5.0)]
        );
        // Walking off a ledge: no jump; a small drop: no landing either.
        let mut t = StepTracker::default();
        t.update(Vec3::ZERO, Vec3::ZERO, true, false);
        assert!(t.update(Vec3::X * 0.1, Vec3::ZERO, false, false).is_empty());
        t.update(Vec3::X * 0.2, -Vec3::Y * 2.0, false, false);
        assert!(t.update(Vec3::X * 0.3, Vec3::ZERO, true, false).is_empty());
        // A teleport isn't walking.
        let mut t = StepTracker::default();
        t.update(Vec3::ZERO, Vec3::ZERO, true, false);
        assert!(t.update(Vec3::X * 50.0, Vec3::ZERO, true, false).is_empty());
    }

    #[test]
    fn the_mixer_plays_and_scales_by_volume() {
        let full = |master: u32, sfx: u32| {
            let mut a = audio(AudioSettings {
                master,
                sfx,
                ambience: 100,
            });
            a.play(SoundEvent::Land(10.0));
            peak(&a.backend().render(seconds(0.2))).0
        };
        let loud = full(100, 100);
        assert!(
            loud > 0.3,
            "a landing at full volume should be loud ({loud})"
        );
        let half = full(50, 100);
        assert!(
            (half / loud - 0.5).abs() < 0.05,
            "master 50% gave {:.3} of full",
            half / loud
        );
        let half = full(100, 50);
        assert!(
            (half / loud - 0.5).abs() < 0.05,
            "sfx 50% gave {:.3} of full",
            half / loud
        );
        assert_eq!(full(0, 100), 0.0);
        assert_eq!(full(100, 0), 0.0);
    }

    #[test]
    fn a_lamp_pans_and_fades_with_distance() {
        let lamp_at = |pos: Vec3, radius: f32| {
            let mut a = audio(AudioSettings {
                master: 100,
                sfx: 100,
                ambience: 100,
            });
            // Listener at the origin looking down -Z, so +X is its right.
            a.set_listener(Vec3::ZERO, -Vec3::Z, Vec3::Y);
            a.add_lamp(pos, radius);
            assert_eq!(a.lamp_count(), 1);
            let b = a.backend();
            b.render(256); // let the commands land
            peak(&b.render(seconds(0.1)))
        };
        let (l, r) = lamp_at(Vec3::new(3.0, 0.0, 0.0), 20.0);
        assert!(r > l * 2.0, "lamp on the right: left {l:.3}, right {r:.3}");
        let (l, r) = lamp_at(Vec3::new(-3.0, 0.0, 0.0), 20.0);
        assert!(l > r * 2.0, "lamp on the left: left {l:.3}, right {r:.3}");
        let (l, r) = lamp_at(Vec3::new(0.0, 0.0, -30.0), 20.0);
        assert!(l < 1e-3 && r < 1e-3, "past its radius: {l}, {r}");
    }
}
