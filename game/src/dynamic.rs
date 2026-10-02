//! Dynamic bodies (§15): props rapier moves. The `dynamic` prefab, the
//! rapier → ECS readback of their poses, the interpolated matrix they're
//! drawn with, and the kill plane for ones that fall out of the world.

use crate::components::{Material, Mesh, NoShadowCast};
use crate::controller::KillPlane;
use crate::physics::{from_rapier, quat_from_rapier, tag_surface, to_rapier, Physics};
use crate::prefab::{spawn_static_prop, ColliderStats, SpawnArgs};
use bevy_ecs::prelude::*;
use glam::{Mat4, Quat, Vec3};
use rapier3d::prelude::{ColliderBuilder, RigidBodyHandle, Vector};

/// A `dynamic` node's mass when it names neither `mass` nor `density`: the
/// shape's volume at 150 kg/m³, about a hollow steel drum or a crate.
pub const DEFAULT_DENSITY: f32 = 150.0;

/// The parameters a `dynamic` node may carry.
pub const DYNAMIC_PARAMS: [&str; 4] = ["mass", "density", "collider", "shadow"];

/// What a `dynamic` prop collides as (its `collider` param), in its own
/// frame. A scanned mesh's hull has a bottom of many near-coplanar facets,
/// and a barrel stood on one kept rocking and walking for seconds; a
/// primitive fitted to the mesh's bounds rests still (§15).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyShape {
    /// The convex hull of the mesh.
    Hull,
    /// A cuboid round its bounds.
    Box,
    /// An upright (local Y) cylinder round its bounds: a barrel, a drum.
    Cylinder,
}

/// A `dynamic` node's params over the defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicParams {
    /// kg; `None` takes the mass from `density`.
    pub mass: Option<f32>,
    pub density: f32,
    pub shape: BodyShape,
}

impl Default for DynamicParams {
    fn default() -> Self {
        Self {
            mass: None,
            density: DEFAULT_DENSITY,
            shape: BodyShape::Hull,
        }
    }
}

/// The entity's rigid body, and the node's scale, which the body's pose
/// can't hold: the collider was built scaled, and the draw re-applies it.
#[derive(Component)]
pub struct Body {
    pub handle: RigidBodyHandle,
    pub scale: Vec3,
}

/// A body's pose after the latest fixed step (rapier → ECS).
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct BodyPose {
    pub pos: Vec3,
    pub rot: Quat,
}

/// `BodyPose` at the start of the latest fixed step, the other end extract
/// interpolates from.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct PrevBodyPose(pub BodyPose);

/// A `dynamic` node's params over the defaults, and the ones it couldn't
/// use: unknown ones, and ones out of range or of the wrong kind. `mass`
/// wins over `density` when both are given.
pub fn dynamic_params(spec: Option<&feather_assets::PrefabSpec>) -> (DynamicParams, Vec<String>) {
    let mut p = DynamicParams::default();
    let Some(spec) = spec else {
        return (p, Vec::new());
    };
    let mut bad: Vec<String> = spec
        .params
        .as_object()
        .into_iter()
        .flat_map(|o| o.keys())
        .filter(|k| !DYNAMIC_PARAMS.contains(&k.as_str()))
        .cloned()
        .collect();
    let mut num = |key: &str, ok: fn(f32) -> bool| match spec.params.get(key) {
        None => None,
        Some(_) => match spec.f32(key).filter(|&v| ok(v)) {
            Some(v) => Some(v),
            None => {
                bad.push(key.to_string());
                None
            }
        },
    };
    p.mass = num("mass", |v| v > 0.0 && v <= 100_000.0);
    if let Some(d) = num("density", |v| v > 0.0 && v <= 20_000.0) {
        p.density = d;
    }
    if spec.params.get("collider").is_some() {
        match spec.str("collider") {
            Some("hull") => p.shape = BodyShape::Hull,
            Some("box") => p.shape = BodyShape::Box,
            Some("cylinder") => p.shape = BodyShape::Cylinder,
            _ => bad.push("collider".to_string()),
        }
    }
    (p, bad)
}

/// A body's collider in its own frame, from its mesh's vertices with the
/// node's scale applied (a rapier pose can't scale). `None` when a hull
/// has nothing to wrap.
pub fn body_collider(
    data: &feather_assets::MeshData,
    scale: Vec3,
    shape: BodyShape,
) -> Option<ColliderBuilder> {
    let (a, b) = data.bounds();
    // A negative scale mirrors the bounds; min/max puts them back in order.
    let (lo, hi) = ((a * scale).min(b * scale), (a * scale).max(b * scale));
    let (half, center) = ((hi - lo) * 0.5, to_rapier((lo + hi) * 0.5));
    match shape {
        BodyShape::Hull => {
            let points: Vec<Vector> = data
                .vertices
                .iter()
                .map(|v| to_rapier(Vec3::from(v.pos) * scale))
                .collect();
            ColliderBuilder::convex_hull(&points)
        }
        BodyShape::Box => Some(ColliderBuilder::cuboid(half.x, half.y, half.z).translation(center)),
        BodyShape::Cylinder => {
            Some(ColliderBuilder::cylinder(half.y, half.x.max(half.z)).translation(center))
        }
    }
}

/// A prop rapier moves (§15): it falls, settles, and can be pushed. It
/// collides as its `collider` shape (default the convex hull of its mesh),
/// in its own frame: the node's rotation and translation become the body's
/// pose, and its scale is baked into the shape, since a rapier pose can't
/// scale. A node with no mesh, or one whose points have no hull, falls back
/// to static geometry.
pub fn spawn_dynamic(world: &mut World, args: &SpawnArgs) {
    let (params, bad) = dynamic_params(args.spec);
    for key in bad {
        eprintln!("[scene] dynamic: can't use param {key:?}; ignored");
    }
    let (Some(mesh), Some(data)) = (args.mesh, args.mesh_data) else {
        eprintln!("[scene] dynamic: a node without a mesh; spawning as static geometry");
        spawn_static_prop(world, args);
        return;
    };
    let (scale, rot, pos) = args.transform.to_scale_rotation_translation();
    let Some(shape) = body_collider(data, scale, params.shape) else {
        eprintln!("[scene] dynamic: no hull for its mesh; spawning as static geometry");
        spawn_static_prop(world, args);
        return;
    };
    let (handle, collider) =
        world
            .resource_mut::<Physics>()
            .add_dynamic(shape, pos, rot, params.mass, params.density);
    tag_surface(
        &mut world.resource_mut::<Physics>().colliders[collider],
        args.surface,
    );
    if let Some(mut stats) = world.get_resource_mut::<ColliderStats>() {
        stats.dynamic += 1;
        stats.surfaces[args.surface.index()] += 1;
    }
    let pose = BodyPose { pos, rot };
    let mut e = world.spawn((
        Body { handle, scale },
        pose,
        PrevBodyPose(pose),
        Mesh(mesh),
        Material(args.material),
    ));
    if !args.flag("shadow", true) {
        e.insert(NoShadowCast);
    }
}

/// **rapier → ECS** for the bodies: after the step, the pose they had
/// becomes the previous one and rapier's is the current one.
pub fn bodies_readback(
    q: &mut Query<(&Body, &mut BodyPose, &mut PrevBodyPose)>,
    physics: &Physics,
) {
    for (body, mut pose, mut prev) in q.iter_mut() {
        prev.0 = *pose;
        let b = &physics.bodies[body.handle];
        *pose = BodyPose {
            pos: from_rapier(b.translation()),
            rot: quat_from_rapier(*b.rotation()),
        };
    }
}

pub fn bodies_readback_sys(
    mut q: Query<(&Body, &mut BodyPose, &mut PrevBodyPose)>,
    physics: Res<Physics>,
) {
    bodies_readback(&mut q, &physics);
}

/// The matrix a body is drawn with, `alpha` of the way from its previous
/// pose to its current one: position lerped, rotation slerped, then the
/// node's scale.
pub fn body_model(prev: &BodyPose, cur: &BodyPose, scale: Vec3, alpha: f32) -> Mat4 {
    Mat4::from_scale_rotation_translation(
        scale,
        prev.rot.slerp(cur.rot, alpha),
        prev.pos.lerp(cur.pos, alpha),
    )
}

/// A body that has fallen below the kill plane is gone: out of rapier and
/// out of the world. Otherwise a barrel pushed off the edge would fall
/// forever, awake, costing a solver step each tick.
pub fn bodies_kill_plane_sys(
    q: Query<(Entity, &Body, &BodyPose)>,
    kill: Res<KillPlane>,
    mut physics: ResMut<Physics>,
    mut commands: Commands,
) {
    for (entity, body, pose) in &q {
        if pose.pos.y < kill.0 {
            physics.remove_body(body.handle);
            commands.entity(entity).despawn();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::{physics_step_sys, GROUND_Y};
    use crate::testing::node;
    use crate::Surface;
    use feather_assets::MeshData;
    use feather_render::MeshId;

    /// The 80x80 ground the app builds, a kill plane under it, and the fixed
    /// tick's body half: the step, the readback, the kill plane.
    fn body_world() -> (World, Schedule) {
        let mut w = World::new();
        let mut physics = Physics::new();
        physics.add_static_box(
            Vec3::new(0.0, GROUND_Y - 0.5, 0.0),
            Vec3::new(80.0, 1.0, 80.0),
        );
        w.insert_resource(physics);
        w.insert_resource(KillPlane(GROUND_Y - 4.0));
        w.insert_resource(ColliderStats::default());
        let mut s = Schedule::default();
        s.add_systems((physics_step_sys, bodies_readback_sys, bodies_kill_plane_sys).chain());
        (w, s)
    }

    fn spawn(w: &mut World, transform: Mat4, params: serde_json::Value, mesh: &MeshData) {
        let n = node(Some("dynamic"), params);
        spawn_dynamic(
            w,
            &SpawnArgs {
                transform,
                mesh: Some(MeshId(0)),
                material: 7,
                mesh_data: Some(mesh),
                baked: None,
                spec: n.prefab.as_ref(),
                surface: Surface::Wood,
            },
        );
    }

    fn the_body(w: &mut World) -> (Body, BodyPose) {
        let mut q = w.query::<(&Body, &BodyPose)>();
        let (b, p) = q.single(w).expect("one body");
        (
            Body {
                handle: b.handle,
                scale: b.scale,
            },
            *p,
        )
    }

    fn run(w: &mut World, s: &mut Schedule, ticks: u32) {
        for _ in 0..ticks {
            s.run(w);
        }
    }

    #[test]
    fn a_dropped_body_falls_rests_and_sleeps() {
        let (mut w, mut s) = body_world();
        let cube = MeshData::cube(1.0);
        let start = Vec3::new(0.0, GROUND_Y + 1.5, 0.0);
        spawn(
            &mut w,
            Mat4::from_translation(start),
            serde_json::json!({}),
            &cube,
        );
        run(&mut w, &mut s, 30);
        let (_, mid) = the_body(&mut w);
        assert!(mid.pos.y < start.y - 0.3, "falling: {}", mid.pos.y);
        run(&mut w, &mut s, 270);
        let (body, rest) = the_body(&mut w);
        assert!(
            (rest.pos.y - (GROUND_Y + 0.5)).abs() < 0.01,
            "resting on the ground at its half-height: {}",
            rest.pos.y
        );
        assert!(
            w.resource::<Physics>().bodies[body.handle].is_sleeping(),
            "asleep after 5 s"
        );
        // The collider carries the node's surface (§20), as static ones do.
        let physics = w.resource::<Physics>();
        let c = physics.bodies[body.handle].colliders()[0];
        assert_eq!(
            crate::physics::collider_surface(&physics.colliders[c]),
            Surface::Wood
        );
        assert_eq!(w.resource::<ColliderStats>().dynamic, 1);
    }

    /// A node authored resting on the ground, scaled and turned, stays put
    /// and is drawn exactly as authored: the scale reached the hull, and the
    /// rotation and translation the body.
    #[test]
    fn a_scaled_turned_body_authored_at_rest_holds_still() {
        let (mut w, mut s) = body_world();
        let cube = MeshData::cube(1.0);
        let scale = Vec3::new(1.5, 0.5, 0.8);
        let authored = Mat4::from_scale_rotation_translation(
            scale,
            Quat::from_rotation_y(0.5),
            Vec3::new(2.0, GROUND_Y + scale.y * 0.5, -1.0),
        );
        spawn(&mut w, authored, serde_json::json!({}), &cube);
        let (_, start) = the_body(&mut w);
        run(&mut w, &mut s, 300);
        let (body, end) = the_body(&mut w);
        assert!(
            end.pos.distance(start.pos) < 0.001,
            "moved {} m",
            end.pos.distance(start.pos)
        );
        assert!(end.rot.angle_between(start.rot) < 0.001);
        assert!(w.resource::<Physics>().bodies[body.handle].is_sleeping());
        let drawn = body_model(&end, &end, body.scale, 0.5);
        assert!(
            drawn.abs_diff_eq(authored, 1e-3),
            "drawn {drawn:?}\nauthored {authored:?}"
        );
    }

    #[test]
    fn a_body_is_drawn_between_its_poses() {
        let prev = BodyPose {
            pos: Vec3::new(0.0, 1.0, 0.0),
            rot: Quat::IDENTITY,
        };
        let cur = BodyPose {
            pos: Vec3::new(2.0, 1.0, 0.0),
            rot: Quat::from_rotation_z(1.0),
        };
        let scale = Vec3::new(1.0, 2.0, 3.0);
        let m = |a| body_model(&prev, &cur, scale, a);
        let want = |pos, rot| Mat4::from_scale_rotation_translation(scale, rot, pos);
        assert!(m(0.0).abs_diff_eq(want(prev.pos, prev.rot), 1e-6));
        assert!(m(1.0).abs_diff_eq(want(cur.pos, cur.rot), 1e-6));
        assert!(m(0.25).abs_diff_eq(
            want(Vec3::new(0.5, 1.0, 0.0), Quat::from_rotation_z(0.25)),
            1e-5
        ));
    }

    #[test]
    fn a_body_below_the_kill_plane_is_removed() {
        let (mut w, mut s) = body_world();
        let cube = MeshData::cube(1.0);
        // One falling past the plane, one resting on the ground.
        spawn(
            &mut w,
            Mat4::from_translation(Vec3::new(50.0, GROUND_Y - 3.0, 0.0)),
            serde_json::json!({}),
            &cube,
        );
        spawn(
            &mut w,
            Mat4::from_translation(Vec3::new(0.0, GROUND_Y + 0.5, 0.0)),
            serde_json::json!({}),
            &cube,
        );
        run(&mut w, &mut s, 60);
        assert_eq!(
            w.query::<&Body>().iter(&w).count(),
            1,
            "the fallen one is gone"
        );
        let physics = w.resource::<Physics>();
        assert_eq!(physics.bodies.len(), 1, "and out of rapier");
        assert_eq!(
            physics.colliders.len(),
            2,
            "the ground and the resting body"
        );
    }

    #[test]
    fn dynamic_params_set_the_mass_and_bad_ones_are_named() {
        let spec = |v| node(Some("dynamic"), v).prefab;
        assert_eq!(dynamic_params(None), (DynamicParams::default(), vec![]));
        let s = spec(serde_json::json!({
            "mass": 20.0, "density": 500.0, "collider": "cylinder", "shadow": false
        }));
        let want = DynamicParams {
            mass: Some(20.0),
            density: 500.0,
            shape: BodyShape::Cylinder,
        };
        assert_eq!(dynamic_params(s.as_ref()), (want, vec![]));
        let s = spec(serde_json::json!({
            "mass": -1.0, "density": "x", "collider": "sphere", "colour": 1
        }));
        let (p, mut bad) = dynamic_params(s.as_ref());
        bad.sort();
        assert_eq!(p, DynamicParams::default());
        assert_eq!(bad, ["collider", "colour", "density", "mass"]);

        // What reaches rapier: `mass` wins, else the 1 m³ cube's volume times
        // the density.
        let cube = MeshData::cube(1.0);
        for (params, want) in [
            (serde_json::json!({ "mass": 20.0, "density": 500.0 }), 20.0),
            (serde_json::json!({ "density": 500.0 }), 500.0),
            (serde_json::json!({}), DEFAULT_DENSITY),
        ] {
            let (mut w, _) = body_world();
            spawn(&mut w, Mat4::IDENTITY, params, &cube);
            let (body, _) = the_body(&mut w);
            let mass = w.resource::<Physics>().bodies[body.handle].mass();
            assert!((mass - want).abs() < 0.01, "mass {mass}, want {want}");
        }
    }

    /// `collider` picks the shape, fitted to the mesh's bounds with the
    /// node's scale: a cube scaled (0.6, 0.9, 0.4) as a box is that box, and
    /// as a cylinder stands 0.9 tall and 0.6 across (the wider side).
    #[test]
    fn the_collider_param_shapes_the_body() {
        use rapier3d::parry::shape::ShapeType;
        let cube = MeshData::cube(1.0);
        let scale = Vec3::new(0.6, 0.9, 0.4);
        for (collider, want) in [
            ("hull", ShapeType::ConvexPolyhedron),
            ("box", ShapeType::Cuboid),
            ("cylinder", ShapeType::Cylinder),
        ] {
            let (mut w, _) = body_world();
            spawn(
                &mut w,
                Mat4::from_scale(scale),
                serde_json::json!({ "collider": collider }),
                &cube,
            );
            let (body, _) = the_body(&mut w);
            let physics = w.resource::<Physics>();
            let c = &physics.colliders[physics.bodies[body.handle].colliders()[0]];
            assert_eq!(c.shape().shape_type(), want, "{collider}");
            let a = c.compute_aabb();
            let size = from_rapier(a.maxs - a.mins);
            let want_size = match want {
                ShapeType::Cylinder => Vec3::new(0.6, 0.9, 0.6),
                _ => scale,
            };
            assert!(size.abs_diff_eq(want_size, 1e-3), "{collider}: {size}");
        }
    }

    #[test]
    fn shadow_param_and_a_meshless_node() {
        let cube = MeshData::cube(1.0);
        let (mut w, _) = body_world();
        spawn(
            &mut w,
            Mat4::IDENTITY,
            serde_json::json!({ "shadow": false }),
            &cube,
        );
        assert_eq!(w.query::<(&Body, &NoShadowCast)>().iter(&w).count(), 1);

        // A marker node has nothing to collide with: no body, nothing drawn.
        let (mut w, _) = body_world();
        let n = node(Some("dynamic"), serde_json::json!({}));
        spawn_dynamic(
            &mut w,
            &SpawnArgs {
                transform: Mat4::IDENTITY,
                mesh: None,
                material: 0,
                mesh_data: None,
                baked: None,
                spec: n.prefab.as_ref(),
                surface: Surface::default(),
            },
        );
        assert_eq!(w.query::<&Body>().iter(&w).count(), 0);
        assert_eq!(w.resource::<Physics>().bodies.len(), 0);
    }

    // ---- the player and props (§15) ----

    /// The player 3 m behind a unit box of `mass` kg resting on the ground,
    /// facing it along -Z.
    fn player_and_box(mass: f32) -> (Physics, crate::controller::Player, RigidBodyHandle) {
        let (mut ph, p) = crate::testing::setup(&[], Vec3::new(0.0, GROUND_Y, 3.0));
        let cube: Vec<Vector> = MeshData::cube(1.0)
            .vertices
            .iter()
            .map(|v| to_rapier(Vec3::from(v.pos)))
            .collect();
        let at = Vec3::new(0.0, GROUND_Y + 0.5, 0.0);
        let (h, _) = ph
            .add_dynamic_hull(&cube, at, Quat::IDENTITY, Some(mass), DEFAULT_DENSITY)
            .expect("a cube has a hull");
        ph.step();
        (ph, p, h)
    }

    #[test]
    fn walking_into_a_light_prop_pushes_it() {
        let (mut ph, mut p, h) = player_and_box(20.0);
        let start = from_rapier(ph.bodies[h].translation());
        crate::testing::run(&mut p, &mut ph, 120, Vec3::NEG_Z);
        let moved = from_rapier(ph.bodies[h].translation()) - start;
        assert!(moved.z < -1.0, "pushed along the walk: {moved:?}");
        assert!(moved.x.abs() < 0.05, "and not sideways: {moved:?}");
        assert!(p.pos.z < 1.0, "the player followed it: {}", p.pos.z);
    }

    /// A prop whose friction on the ground beats `PUSH_FORCE` doesn't move,
    /// and holds the player: 500 kg, about a full drum.
    #[test]
    fn a_heavy_prop_holds_the_player() {
        let (mut ph, mut p, h) = player_and_box(500.0);
        let start = from_rapier(ph.bodies[h].translation());
        crate::testing::run(&mut p, &mut ph, 120, Vec3::NEG_Z);
        let moved = from_rapier(ph.bodies[h].translation()) - start;
        assert!(moved.length() < 0.01, "moved {moved:?}");
        assert!(p.pos.z > 0.5 + crate::controller::PLAYER_RADIUS - 0.05);
    }

    #[test]
    fn the_player_stands_on_a_prop() {
        let (mut ph, mut p, h) = player_and_box(20.0);
        // Drop the player onto the box's top.
        p.pos = Vec3::new(0.0, GROUND_Y + 1.5, 0.0);
        p.prev_pos = p.pos;
        p.on_ground = false;
        let body = p.body;
        ph.bodies[body].set_translation(to_rapier(p.pos + crate::controller::Player::CENTER), true);
        crate::testing::run(&mut p, &mut ph, 120, Vec3::ZERO);
        assert!(p.on_ground);
        assert!(
            (p.pos.y - (GROUND_Y + 1.0)).abs() < 0.03,
            "feet on its top: {}",
            p.pos.y
        );
        let rest = from_rapier(ph.bodies[h].translation());
        assert!((rest - Vec3::new(0.0, GROUND_Y + 0.5, 0.0)).length() < 0.01);
    }
}
