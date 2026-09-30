//! Ropes (§15): a chain of points hanging from a fixed anchor, bending under
//! gravity and the wind, with an item (a lantern) on the end.
//!
//! Each fixed step is position-based (Verlet): every free point moves on by
//! its last step's motion, plus gravity and the drag of the wind relative to
//! it, and then the segments are pulled back to their length, point pair by
//! point pair, in proportion to how light each end is. The anchor never
//! moves, and the item is far heavier than the rope, so the rope drapes from
//! the anchor to the item and bends wherever the wind catches it, rather than
//! swinging as one rigid rod.
//!
//! Drawing needs no renderer support: each segment is an instance of a unit
//! cylinder stretched from one point to the next (`segment_matrix`), and the
//! item an instance turned to hang along the last segment (`item_matrix`).

use glam::{Mat4, Quat, Vec3};

/// Gravity, m/s².
const GRAVITY: f32 = 9.81;
/// Each fixed step runs as this many sub-steps of this many constraint
/// passes. Sub-stepping holds the length far better than passes alone (a
/// 12-segment rope with a lantern, gusting at 6 m/s: worst segment 0.4% long
/// at 8 × 8, 1.0% at 1 × 96), and a test keeps it within 1%.
pub const ROPE_SUBSTEPS: usize = 8;
pub const ROPE_ITERATIONS: usize = 8;
/// The drag of the air, per second, on a rope point and on the item: how fast
/// each would match the wind's speed. It's also what damps the swing.
const ROPE_DRAG: f32 = 3.0;
const ITEM_DRAG: f32 = 0.4;
/// How much heavier the item is than a rope point.
const ITEM_MASS_RATIO: f32 = 40.0;

/// How a step is solved: the real constants, which tests vary.
#[derive(Clone, Copy)]
struct Solver {
    gravity: f32,
    iterations: usize,
    attach: bool,
}

impl Default for Solver {
    fn default() -> Self {
        Self {
            gravity: GRAVITY,
            iterations: ROPE_ITERATIONS,
            attach: true,
        }
    }
}

/// A level's wind (§13's `environment`): a steady part along `dir`, with slow
/// gusts that vary from place to place.
#[derive(Clone, Copy, Debug)]
pub struct Wind {
    /// Horizontal, unit length (or zero for still air).
    pub dir: Vec3,
    /// Mean speed, m/s.
    pub speed: f32,
}

impl Default for Wind {
    fn default() -> Self {
        Self {
            dir: Vec3::X,
            speed: 2.0,
        }
    }
}

impl Wind {
    #[cfg(test)]
    pub const STILL: Wind = Wind {
        dir: Vec3::ZERO,
        speed: 0.0,
    };

    /// The air's velocity at `p`, `t` seconds in: the mean along `dir`, gusting
    /// by up to about ±50% and a quarter of it across. A gust's phase shifts
    /// with position, so neighbours sway alike but not in lockstep.
    pub fn at(&self, p: Vec3, t: f32) -> Vec3 {
        let tau = std::f32::consts::TAU;
        let phase = p.x * 0.13 + p.z * 0.07;
        let along = 1.0
            + 0.35 * (tau * 0.37 * t + phase).sin()
            + 0.15 * (tau * 0.93 * t + 2.0 * phase).sin();
        let across = 0.25 * (tau * 0.21 * t + 3.0 * phase).sin();
        let side = Vec3::new(-self.dir.z, 0.0, self.dir.x);
        (self.dir * along + side * across) * self.speed
    }
}

/// How a rope is made.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeParams {
    /// From the anchor to the item, metres.
    pub length: f32,
    pub segments: usize,
    pub radius: f32,
    /// How strongly it feels the wind (lower indoors).
    pub wind: f32,
}

impl Default for RopeParams {
    fn default() -> Self {
        Self {
            length: 1.5,
            segments: 12,
            radius: 0.008,
            wind: 1.0,
        }
    }
}

/// One rope's state. `points[0]` is the anchor; the last is where the item
/// hangs.
#[derive(Clone, Debug, PartialEq)]
pub struct Rope {
    pub points: Vec<Vec3>,
    /// Where each point was a step ago (Verlet's velocity).
    last: Vec<Vec3>,
    /// The points at the start of the latest step, which the renderer
    /// interpolates from.
    pub prev: Vec<Vec3>,
    inv_mass: Vec<f32>,
    segment: f32,
    pub params: RopeParams,
}

impl Rope {
    /// Hanging straight down from `anchor`, at rest.
    pub fn new(anchor: Vec3, params: RopeParams) -> Self {
        let n = params.segments.max(1);
        let segment = params.length / n as f32;
        let points: Vec<Vec3> = (0..=n)
            .map(|i| anchor - Vec3::Y * segment * i as f32)
            .collect();
        let inv_mass = (0..=n)
            .map(|i| match i {
                0 => 0.0,
                i if i == n => 1.0 / ITEM_MASS_RATIO,
                _ => 1.0,
            })
            .collect();
        Self {
            last: points.clone(),
            prev: points.clone(),
            points,
            inv_mass,
            segment,
            params: RopeParams {
                segments: n,
                ..params
            },
        }
    }

    /// Where it hangs from (tests read it; the app's spawn test is one).
    pub fn anchor(&self) -> Vec3 {
        self.points[0]
    }

    /// Its free end, where the item hangs.
    pub fn end(&self) -> Vec3 {
        *self.points.last().unwrap()
    }

    /// Advance by `dt` at sim time `t` in `wind`, in `ROPE_SUBSTEPS` steps.
    /// `prev` is the state before all of them, for the renderer.
    pub fn step(&mut self, dt: f32, t: f32, wind: &Wind) {
        let start = self.points.clone();
        let h = dt / ROPE_SUBSTEPS as f32;
        for k in 0..ROPE_SUBSTEPS {
            self.step_with(h, t + k as f32 * h, wind, Solver::default());
        }
        self.prev = start;
    }

    fn step_with(&mut self, dt: f32, t: f32, wind: &Wind, solver: Solver) {
        let Solver {
            gravity,
            iterations,
            attach,
        } = solver;
        self.prev.clone_from(&self.points);
        let n = self.points.len() - 1;
        for i in 1..=n {
            let p = self.points[i];
            let velocity = (p - self.last[i]) / dt;
            let drag = if i == n { ITEM_DRAG } else { ROPE_DRAG };
            let air = wind.at(p, t) * self.params.wind;
            let accel = Vec3::NEG_Y * gravity + (air - velocity) * drag;
            self.last[i] = p;
            self.points[i] = p + velocity * dt + accel * dt * dt;
        }
        for pass in 0..iterations {
            // Alternate the sweep's direction: upwards carries the heavy
            // item's pull to the anchor in one pass rather than n.
            for k in 0..n {
                let i = if pass % 2 == 0 { k } else { n - 1 - k };
                let (a, b) = (self.points[i], self.points[i + 1]);
                let (wa, wb) = (self.inv_mass[i], self.inv_mass[i + 1]);
                let d = b - a;
                let len = d.length();
                if len < 1e-9 || wa + wb == 0.0 {
                    continue;
                }
                let fix = d * ((len - self.segment) / (len * (wa + wb)));
                self.points[i] += fix * wa;
                self.points[i + 1] -= fix * wb;
            }
            // Long-range attachment (Kim et al. 2012): no point further from
            // the anchor than the rope above it is long. The pair passes alone
            // converge slowly under a heavy item, and the rope would stretch.
            let anchor = self.points[0];
            for i in (1..=n).filter(|_| attach) {
                let d = self.points[i] - anchor;
                let reach = self.segment * i as f32;
                let len = d.length();
                if len > reach {
                    self.points[i] = anchor + d * (reach / len);
                }
            }
        }
    }

    /// The points `alpha` of the way from the latest step's start to its end.
    pub fn interpolated(&self, alpha: f32) -> Vec<Vec3> {
        self.prev
            .iter()
            .zip(&self.points)
            .map(|(a, b)| a.lerp(*b, alpha))
            .collect()
    }
}

/// The unit cylinder (diameter 1, height 1 along Y, centred) stretched from
/// `a` to `b` at `radius`. It runs `radius` past each end, so neighbours
/// overlap and a bend shows no gap.
pub fn segment_matrix(a: Vec3, b: Vec3, radius: f32) -> Mat4 {
    let d = b - a;
    let len = d.length();
    let dir = if len > 1e-9 { d / len } else { Vec3::Y };
    Mat4::from_scale_rotation_translation(
        Vec3::new(2.0 * radius, len + 2.0 * radius, 2.0 * radius),
        Quat::from_rotation_arc(Vec3::Y, dir),
        (a + b) * 0.5,
    )
}

/// The item's model matrix: `local` (its authored rotation and scale) turned
/// so its +Y runs up the rope's last segment, at the rope's end.
pub fn item_matrix(points: &[Vec3], local: Mat4) -> Mat4 {
    let n = points.len() - 1;
    let up = (points[n - 1] - points[n]).normalize_or(Vec3::Y);
    Mat4::from_translation(points[n])
        * Mat4::from_quat(Quat::from_rotation_arc(Vec3::Y, up))
        * local
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    fn run(rope: &mut Rope, wind: &Wind, seconds: f32, mut each: impl FnMut(&Rope, f32)) {
        let steps = (seconds / DT).round() as usize;
        for s in 0..steps {
            let t = s as f32 * DT;
            rope.step(DT, t, wind);
            each(rope, t);
        }
    }

    fn worst_stretch(rope: &Rope) -> f32 {
        rope.points
            .windows(2)
            .map(|w| ((w[1] - w[0]).length() / rope.segment - 1.0).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn a_still_rope_hangs_straight_at_its_length() {
        let anchor = Vec3::new(1.0, 5.0, -2.0);
        let mut rope = Rope::new(anchor, RopeParams::default());
        run(&mut rope, &Wind::STILL, 5.0, |_, _| {});
        let want = anchor - Vec3::Y * 1.5;
        assert!(
            (rope.end() - want).length() < 0.015,
            "end at {}",
            rope.end()
        );
        for p in &rope.points {
            assert!((p.x - anchor.x).abs() < 1e-4 && (p.z - anchor.z).abs() < 1e-4);
        }
        assert!(
            worst_stretch(&rope) < 0.01,
            "stretch {}",
            worst_stretch(&rope)
        );
        assert_eq!(rope.anchor(), anchor, "the anchor moved");
    }

    #[test]
    fn too_few_passes_stretch_it() {
        // Control for the length: a single pass and no attachment let the
        // lantern pull the rope long.
        let mut rope = Rope::new(Vec3::ZERO, RopeParams::default());
        let solver = Solver {
            iterations: 1,
            attach: false,
            ..Solver::default()
        };
        for s in 0..300 {
            rope.step_with(DT, s as f32 * DT, &Wind::STILL, solver);
        }
        assert!(
            worst_stretch(&rope) > 0.01,
            "stretch {}",
            worst_stretch(&rope)
        );
    }

    /// Displaced sideways and let go in still air: the period of its swing
    /// and how its amplitude goes, from the end's x crossings.
    fn swing(gravity: f32, seconds: f32) -> (f32, Vec<f32>) {
        let params = RopeParams::default();
        let mut rope = Rope::new(Vec3::ZERO, params);
        // Tilt the whole rope 10 degrees about the anchor.
        let q = Quat::from_rotation_z(10f32.to_radians());
        for (p, l) in rope.points.iter_mut().zip(rope.last.iter_mut()) {
            *p = q * *p;
            *l = *p;
        }
        let (mut crossings, mut peaks) = (Vec::new(), Vec::new());
        let (mut last_x, mut peak) = (rope.end().x, 0.0f32);
        for s in 0..(seconds / DT) as usize {
            let t = s as f32 * DT;
            let solver = Solver {
                gravity,
                ..Solver::default()
            };
            let h = DT / ROPE_SUBSTEPS as f32;
            for k in 0..ROPE_SUBSTEPS {
                rope.step_with(h, t + k as f32 * h, &Wind::STILL, solver);
            }
            let x = rope.end().x;
            peak = peak.max(x.abs());
            if last_x < 0.0 && x >= 0.0 {
                crossings.push(t);
                peaks.push(peak);
                peak = 0.0;
            }
            last_x = x;
        }
        let period = (crossings[5] - crossings[1]) / 4.0;
        (period, peaks)
    }

    #[test]
    fn it_swings_like_a_pendulum_and_settles() {
        let (period, peaks) = swing(GRAVITY, 600.0);
        let want = std::f32::consts::TAU * (1.5 / GRAVITY).sqrt();
        assert!(
            (period / want - 1.0).abs() < 0.1,
            "period {period} s, a pendulum's {want}"
        );
        // It dies down, and never grows.
        assert!(
            peaks[6] < 0.5 * peaks[0],
            "amplitude {} -> {}",
            peaks[0],
            peaks[6]
        );
        for w in peaks.windows(2) {
            assert!(w[1] <= w[0] + 1e-5, "amplitude grew {} -> {}", w[0], w[1]);
        }
        // Control: gravity is what sets the period.
        let (fast, _) = swing(4.0 * GRAVITY, 60.0);
        assert!(
            (fast / period - 0.5).abs() < 0.06,
            "4g period {fast} vs {period}"
        );
    }

    #[test]
    fn wind_blows_it_downwind_and_it_holds_its_length() {
        let lean = |speed: f32| {
            let wind = Wind {
                dir: Vec3::new(0.6, 0.0, 0.8),
                speed,
            };
            let mut rope = Rope::new(Vec3::ZERO, RopeParams::default());
            let (mut sum, mut n, mut stretch) = (Vec3::ZERO, 0, 0.0f32);
            run(&mut rope, &wind, 60.0, |r, t| {
                stretch = stretch.max(worst_stretch(r));
                if t > 10.0 {
                    sum += r.end();
                    n += 1;
                }
            });
            (sum / n as f32, stretch)
        };
        let (gentle, s1) = lean(2.0);
        let (strong, s2) = lean(6.0);
        let dir = Vec3::new(0.6, 0.0, 0.8);
        assert!(gentle.dot(dir) > 0.02, "mean offset {gentle}");
        assert!(
            strong.dot(dir) > 2.0 * gentle.dot(dir),
            "{strong} vs {gentle}"
        );
        assert!(s1 < 0.01 && s2 < 0.01, "stretch {s1}, {s2}");
        // Control: still air doesn't lean.
        let mut rope = Rope::new(Vec3::ZERO, RopeParams::default());
        run(&mut rope, &Wind::STILL, 20.0, |_, _| {});
        assert!(rope.end().x.abs() < 1e-4 && rope.end().z.abs() < 1e-4);
    }

    #[test]
    fn it_bends_rather_than_swinging_rigid() {
        // A rope with no weight pulling it straight bows in the wind: its
        // middle leaves the line from the anchor to its end.
        let bow = |segments: usize| {
            let params = RopeParams {
                segments,
                ..RopeParams::default()
            };
            let mut rope = Rope::new(Vec3::ZERO, params);
            let n = rope.points.len() - 1;
            rope.inv_mass[n] = 1.0; // no lantern
            let wind = Wind {
                dir: Vec3::X,
                speed: 4.0,
            };
            let mut most = 0.0f32;
            run(&mut rope, &wind, 20.0, |r, _| {
                let (a, b) = (r.points[0], r.end());
                let line = (b - a).normalize();
                for p in &r.points {
                    let off = (*p - a) - line * (*p - a).dot(line);
                    most = most.max(off.length());
                }
            });
            most
        };
        assert!(bow(12) > 0.03, "a 1.5 m rope bowed only {} m", bow(12));
        // Control: one segment is a straight line.
        assert!(bow(1) < 1e-5);
    }

    #[test]
    fn segments_and_item_sit_on_the_points() {
        let (a, b, r) = (Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.3, 1.1, 3.2), 0.01);
        let m = segment_matrix(a, b, r);
        let dir = (b - a).normalize();
        let bottom = m.transform_point3(Vec3::new(0.0, -0.5, 0.0));
        let top = m.transform_point3(Vec3::new(0.0, 0.5, 0.0));
        assert!((bottom - (a - dir * r)).length() < 1e-5, "{bottom}");
        assert!((top - (b + dir * r)).length() < 1e-5, "{top}");
        // A point on the cylinder's side is `radius` off the axis.
        let side = m.transform_point3(Vec3::new(0.5, 0.0, 0.0));
        let from = side - a;
        assert!(((from - dir * from.dot(dir)).length() - r).abs() < 1e-5);

        // At rest the item is drawn exactly as authored.
        let local = Mat4::from_scale_rotation_translation(
            Vec3::splat(1.2),
            Quat::from_rotation_y(0.7),
            Vec3::ZERO,
        );
        let rope = Rope::new(Vec3::new(0.0, 4.0, 0.0), RopeParams::default());
        let at_rest = item_matrix(&rope.points, local);
        let authored = Mat4::from_translation(rope.end()) * local;
        assert!(at_rest.abs_diff_eq(authored, 1e-5));
        // Tilted, its +Y runs up the last segment.
        let pts = [Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.3, 0.0, 0.0)];
        let up = item_matrix(&pts, Mat4::IDENTITY).transform_vector3(Vec3::Y);
        assert!((up - (pts[0] - pts[1]).normalize()).length() < 1e-5);
    }

    #[test]
    fn it_is_deterministic_and_interpolates_between_steps() {
        let wind = Wind::default();
        let mut a = Rope::new(Vec3::new(3.0, 7.0, 1.0), RopeParams::default());
        let mut b = a.clone();
        run(&mut a, &wind, 3.0, |_, _| {});
        run(&mut b, &wind, 3.0, |_, _| {});
        assert_eq!(a, b);
        assert_eq!(a.interpolated(0.0), a.prev);
        assert_eq!(a.interpolated(1.0), a.points);
        assert_ne!(a.prev, a.points, "it should be moving");
    }
}
