//! The hitscan weapon and its targets: one press, one ray; an orb that
//! flashes, pops, and comes back.

use crate::components::{FrameCount, Material, Mesh, Transform, MESH_SPHERE};
use crate::controller::{InputState, Look, Player, EYE_HEIGHT};
use crate::lights::PointLight;
use crate::physics::{to_rapier, Physics};
use bevy_ecs::prelude::*;
use feather_render::MeshId;
use glam::{Mat4, Vec3};
use rapier3d::prelude::QueryFilter;

/// How far a shot reaches, in metres — past the fog's useful range anyway.
pub const MAX_RANGE: f32 = 100.0;
/// The orb's palette material (level.rs's table: 0..PALETTE always exist).
/// Index 16 is the brightest entry (sRGB .91/.90/.20), so it reads by day;
/// the glow at night and the hit-flash come from the `PointLight`.
pub const TARGET_MATERIAL: u32 = 16;
/// Fixed ticks a popped orb takes to return (5 s at FIXED_DT).
pub const RESPAWN_TICKS: u32 = 300;
/// What a shot gives a dynamic prop it hits (§15), N·s, along the shot at
/// the hit point: 1.5 m/s to a 20 kg barrel.
pub const SHOT_IMPULSE: f32 = 30.0;
/// Per-tick flash decay: 0.82^12 ≈ 0.09, so a hit glows for about 0.2 s.
const FLASH_DECAY: f32 = 0.82;

/// The shooting score: what the HUD shows and the shot sound counts.
#[derive(Resource, Default)]
pub struct Score {
    pub shots: u32,
    pub hits: u32,
    pub kills: u32,
    /// `FrameCount` at the last hit, for the HUD's hit marker.
    pub last_hit_frame: u64,
}

/// A shootable orb (§18's `target` prefab): `hits_left` hits pop it. No
/// collider — a shot tests the ray against `radius` analytically and leaves
/// rapier for walls.
#[derive(Component)]
pub struct Target {
    pub hits_left: u32,
    pub hits_max: u32,
    /// Sphere radius: the hit test's reach, and the visual size via the scale
    /// baked into `base`.
    pub radius: f32,
    /// 1.0 on the tick a hit lands, decays after; drives the light flash and
    /// the size pulse.
    pub flash: f32,
    /// `PointLight.intensity` with no flash on it, what `flash` multiplies.
    pub base_intensity: f32,
    /// Authored placement with the size scale baked in: the pulse scale's
    /// identity, and what a respawn restores.
    pub base: Mat4,
}

/// A popped orb waiting for its respawn.
#[derive(Clone)]
pub struct PendingTarget {
    pub transform: Mat4,
    pub hits: u32,
    pub radius: f32,
    pub color: Vec3,
    pub intensity: f32,
    pub ticks: u32,
}

#[derive(Resource, Default)]
pub struct TargetRespawns(pub Vec<PendingTarget>);

/// The components an orb is, wherever it spawns from (the §18 prefab, the
/// respawn queue). An orb is a `Transform` entity, not a dynamic `Position`
/// one: it holds still, and the light extract (§12) only reads `Transform`,
/// so a `Position`-based orb would lose its glow.
pub fn target_bundle(
    transform: Mat4,
    hits: u32,
    radius: f32,
    color: Vec3,
    intensity: f32,
) -> impl Bundle {
    // The built-in sphere has radius 0.5 (level.rs), so 2*radius scales it to
    // the hit-test size; the pulse re-multiplies from `base`.
    let base = transform * Mat4::from_scale(Vec3::splat(radius * 2.0));
    (
        Transform(base),
        Mesh(MeshId(MESH_SPHERE)),
        Material(TARGET_MATERIAL),
        PointLight {
            color,
            intensity,
            radius: (radius * 12.0).max(1.0),
            source_radius: radius * 0.4,
        },
        Target {
            hits_left: hits,
            hits_max: hits,
            radius,
            flash: 0.0,
            base_intensity: intensity,
            base,
        },
    )
}

/// Ray vs sphere, `dir` unit: the nearest positive t whose point is inside.
fn ray_sphere(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc = origin - center;
    let b = oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - c; // |dir| = 1
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    (t > 0.0).then_some(t)
}

/// One hitscan shot from `eye` along `dir`. Walls are the rapier world (so a
/// shot can't pass geometry); targets are analytic spheres; whichever is
/// nearer wins. A hit flashes the orb and, at `hits_left == 0`, pops it onto
/// the respawn queue; a dynamic prop (§15) is knocked by `SHOT_IMPULSE`.
/// Returns whether an orb was hit.
pub fn fire(world: &mut World, eye: Vec3, dir: Vec3) -> bool {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return false;
    }
    // Walls: the same per-call query pipeline the controller builds for its
    // casts, so a shot respects every collider. The shooter's own capsule is
    // excluded — the ray starts inside it.
    let wall = {
        let mut players = world.query::<&Player>();
        let body = players.iter(world).next().map(|p| p.body);
        let physics = world.resource::<Physics>();
        let queries = physics.broad_phase.as_query_pipeline(
            physics.narrow_phase.query_dispatcher(),
            &physics.bodies,
            &physics.colliders,
            match body {
                Some(body) => QueryFilter::default().exclude_rigid_body(body),
                None => QueryFilter::default(),
            },
        );
        let ray = rapier3d::prelude::Ray::new(to_rapier(eye), to_rapier(dir));
        queries.cast_ray(&ray, MAX_RANGE, true)
    };
    // Targets: the nearest sphere the ray reaches before the wall (if any).
    let mut hit: Option<Entity> = None;
    let mut hit_t = wall.map_or(MAX_RANGE, |(_, t)| t);
    {
        let mut targets = world.query::<(Entity, &Transform, &Target)>();
        for (entity, t, target) in targets.iter(world) {
            let center = t.0.transform_point3(Vec3::ZERO);
            if let Some(t) = ray_sphere(eye, dir, center, target.radius) {
                if t < hit_t {
                    hit = Some(entity);
                    hit_t = t;
                }
            }
        }
    }
    let Some(entity) = hit else {
        // The rapier world was nearest: knock it, if it moves.
        if let Some((collider, t)) = wall {
            let mut physics = world.resource_mut::<Physics>();
            if let Some(body) = physics.colliders[collider].parent() {
                let body = &mut physics.bodies[body];
                if body.is_dynamic() {
                    body.apply_impulse_at_point(
                        to_rapier(dir * SHOT_IMPULSE),
                        to_rapier(eye + dir * t),
                        true,
                    );
                }
            }
        }
        return false;
    };
    let frame = world.resource::<FrameCount>().0;
    {
        let mut score = world.resource_mut::<Score>();
        score.hits += 1;
        score.last_hit_frame = frame;
    }
    // Knock a hit off; pop at zero. The respawn rebuilds from what the orb
    // carries, so it keeps its authored look.
    let mut popped = None;
    {
        let mut e = world.entity_mut(entity);
        let (done, base, hits_max, radius, intensity) = {
            let mut target = e.get_mut::<Target>().expect("queried above");
            target.hits_left -= 1;
            target.flash = 1.0;
            (
                target.hits_left == 0,
                target.base,
                target.hits_max,
                target.radius,
                target.base_intensity,
            )
        };
        if done {
            let color = e.get::<PointLight>().expect("target bundle").color;
            popped = Some(PendingTarget {
                transform: base,
                hits: hits_max,
                radius,
                color,
                intensity,
                ticks: RESPAWN_TICKS,
            });
        }
    }
    if let Some(p) = popped {
        world.despawn(entity);
        world.resource_mut::<Score>().kills += 1;
        world.resource_mut::<TargetRespawns>().0.push(p);
    }
    true
}

/// Fire one shot on the latched edge, then clear it — the jump contract (§14).
/// An exclusive system: a shot reads the player, the physics world and the
/// orb list, and can despawn, which wants the whole `World`.
pub fn weapon_sys(world: &mut World) {
    if !world.resource::<InputState>().fire {
        return;
    }
    world.resource_mut::<InputState>().fire = false;
    world.resource_mut::<Score>().shots += 1;
    let (pos, fwd) = {
        let mut q = world.query::<(&Player, &Look)>();
        q.iter(world)
            .next()
            .map(|(p, l)| (p.pos, l.forward()))
            .unwrap_or((Vec3::ZERO, Vec3::NEG_Z))
    };
    fire(world, pos + Vec3::new(0.0, EYE_HEIGHT, 0.0), fwd);
}

/// Orbs live: the flash decays (light + size pulse), and popped ones count
/// down to a respawn at their authored spot.
pub fn targets_sys(
    mut q: Query<(&mut Target, &mut Transform, &mut PointLight)>,
    mut pending: ResMut<TargetRespawns>,
    mut commands: Commands,
) {
    for (mut target, mut t, mut light) in &mut q {
        target.flash *= FLASH_DECAY;
        // Snap the asymptote to exactly zero, so the light returns to exactly
        // its base intensity instead of lingering a thousandth above it.
        if target.flash < 0.001 {
            target.flash = 0.0;
        }
        t.0 = target.base * Mat4::from_scale(Vec3::splat(1.0 + 0.35 * target.flash));
        light.intensity = target.base_intensity * (1.0 + 3.0 * target.flash);
    }
    let mut ready = Vec::new();
    pending.0.retain_mut(|p| {
        p.ticks = p.ticks.saturating_sub(1);
        if p.ticks == 0 {
            ready.push(p.clone());
        }
        p.ticks > 0
    });
    for p in ready {
        commands.spawn(target_bundle(
            p.transform,
            p.hits,
            p.radius,
            p.color,
            p.intensity,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::GROUND_Y;
    use crate::testing::setup;

    const EYE: Vec3 = Vec3::new(0.0, GROUND_Y + 1.6, 0.0);

    /// Ground slab, a player (its capsule registered, so the ray must exclude
    /// it), and one orb 5 m dead ahead of the eye.
    fn world_with_target() -> (World, Entity) {
        let (physics, player) = setup(&[], Vec3::new(0.0, GROUND_Y, 0.0));
        let mut world = World::new();
        world.insert_resource(physics);
        world.insert_resource(FrameCount::default());
        world.insert_resource(Score::default());
        world.insert_resource(TargetRespawns::default());
        world.spawn((player, Look::new()));
        let orb = world
            .spawn(target_bundle(
                Mat4::from_translation(Vec3::new(0.0, GROUND_Y + 1.6, -5.0)),
                3,
                0.5,
                Vec3::ONE,
                2.0,
            ))
            .id();
        (world, orb)
    }

    #[test]
    fn a_shot_at_an_orb_hits_it() {
        let (mut world, orb) = world_with_target();
        assert!(fire(&mut world, EYE, Vec3::new(0.0, 0.0, -1.0)));
        let t = world.get::<Target>(orb).expect("the orb survives one hit");
        assert_eq!(t.hits_left, 2);
        assert_eq!(t.flash, 1.0);
        assert_eq!(world.resource::<Score>().hits, 1);
        assert_eq!(
            world.resource::<Score>().last_hit_frame,
            world.resource::<FrameCount>().0
        );
    }

    #[test]
    fn a_shot_off_axis_misses() {
        let (mut world, orb) = world_with_target();
        assert!(!fire(
            &mut world,
            EYE,
            Vec3::new(0.3, 0.0, -1.0).normalize()
        ));
        assert_eq!(world.resource::<Score>().hits, 0);
        assert_eq!(world.get::<Target>(orb).expect("untouched").hits_left, 3);
    }

    #[test]
    fn a_wall_eats_the_shot() {
        let (mut world, _) = world_with_target();
        // A tall slab dead ahead, well between the eye and the orb.
        world.resource_mut::<Physics>().add_static_box(
            Vec3::new(0.0, GROUND_Y + 1.6, -2.5),
            Vec3::new(4.0, 4.0, 0.5),
        );
        // New colliders join the broad phase on the next step (physics.rs).
        world.resource_mut::<Physics>().step();
        assert!(!fire(&mut world, EYE, Vec3::new(0.0, 0.0, -1.0)));
        assert_eq!(world.resource::<Score>().hits, 0);
    }

    /// A 20 kg unit box at `at`.
    fn add_box(world: &mut World, at: Vec3) -> rapier3d::prelude::RigidBodyHandle {
        let cube: Vec<_> = feather_assets::MeshData::cube(1.0)
            .vertices
            .iter()
            .map(|v| to_rapier(Vec3::from(v.pos)))
            .collect();
        let mut physics = world.resource_mut::<Physics>();
        let (h, _) = physics
            .add_dynamic_hull(&cube, at, glam::Quat::IDENTITY, Some(20.0), 1.0)
            .expect("a cube has a hull");
        physics.step();
        h
    }

    #[test]
    fn a_shot_knocks_a_prop_along_it() {
        let (mut world, _) = world_with_target();
        let at = Vec3::new(0.0, GROUND_Y + 0.5, -3.0);
        let h = add_box(&mut world, at);
        let dir = (at - EYE).normalize();
        assert!(!fire(&mut world, EYE, dir), "no orb hit");
        let v = crate::physics::from_rapier(world.resource::<Physics>().bodies[h].linvel());
        let along = v.dot(dir);
        assert!(along > 1.0, "knocked along the shot: {v:?}");
        assert!(
            (v - dir * along).length() < 0.5 * along,
            "mostly along it: {v:?}"
        );
        for _ in 0..120 {
            world.resource_mut::<Physics>().step();
        }
        let moved =
            crate::physics::from_rapier(world.resource::<Physics>().bodies[h].translation()) - at;
        assert!(
            moved.z < -0.05 && moved.z > -1.0,
            "slid a little: {moved:?}"
        );
    }

    #[test]
    fn an_orb_in_front_of_a_prop_takes_the_shot() {
        let (mut world, orb) = world_with_target();
        let h = add_box(&mut world, Vec3::new(0.0, GROUND_Y + 1.6, -8.0));
        assert!(fire(&mut world, EYE, Vec3::new(0.0, 0.0, -1.0)));
        assert_eq!(world.get::<Target>(orb).expect("orb").hits_left, 2);
        // It's falling (it floats at eye height), but nothing pushed it.
        let v = crate::physics::from_rapier(world.resource::<Physics>().bodies[h].linvel());
        assert_eq!((v.x, v.z), (0.0, 0.0), "untouched: {v:?}");
    }

    #[test]
    fn three_hits_pop_the_orb_and_it_respawns() {
        let (mut world, orb) = world_with_target();
        for _ in 0..3 {
            assert!(fire(&mut world, EYE, Vec3::new(0.0, 0.0, -1.0)));
        }
        assert!(world.get::<Target>(orb).is_none(), "popped");
        assert_eq!(world.resource::<Score>().kills, 1);
        assert_eq!(world.resource::<TargetRespawns>().0.len(), 1);
        let mut schedule = Schedule::default();
        schedule.add_systems(targets_sys);
        for _ in 0..RESPAWN_TICKS {
            schedule.run(&mut world);
        }
        assert!(world.resource::<TargetRespawns>().0.is_empty());
        let mut q = world.query::<&Target>();
        assert_eq!(q.iter(&world).count(), 1, "the orb is back");
    }

    #[test]
    fn a_hit_flashes_then_settles() {
        let (mut world, orb) = world_with_target();
        fire(&mut world, EYE, Vec3::new(0.0, 0.0, -1.0));
        let mut schedule = Schedule::default();
        schedule.add_systems(targets_sys);
        schedule.run(&mut world);
        let t = world.get::<Target>(orb).expect("orb");
        assert!(t.flash < 1.0 && t.flash > 0.3, "decaying: {}", t.flash);
        assert!(
            world.get::<PointLight>(orb).expect("light").intensity > t.base_intensity,
            "a hit glows"
        );
        for _ in 0..60 {
            schedule.run(&mut world);
        }
        let t = world.get::<Target>(orb).expect("orb");
        assert!(t.flash < 0.01, "settled: {}", t.flash);
        assert_eq!(
            world.get::<PointLight>(orb).expect("light").intensity,
            t.base_intensity
        );
    }

    /// The respawn restores the authored placement, not wherever the popped
    /// orb last was — a knocked orb comes back where the level put it.
    #[test]
    fn respawn_restores_the_authored_spot() {
        let (mut world, _) = world_with_target();
        for _ in 0..3 {
            fire(&mut world, EYE, Vec3::new(0.0, 0.0, -1.0));
        }
        let mut schedule = Schedule::default();
        schedule.add_systems(targets_sys);
        for _ in 0..RESPAWN_TICKS {
            schedule.run(&mut world);
        }
        let mut q = world.query::<(&Transform, &Target)>();
        let (t, _) = q.iter(&world).next().expect("respawned");
        assert!(
            (t.0.transform_point3(Vec3::ZERO) - Vec3::new(0.0, GROUND_Y + 1.6, -5.0)).length()
                < 1e-5,
            "back at the authored spot"
        );
    }
}
