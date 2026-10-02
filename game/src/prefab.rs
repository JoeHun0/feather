//! Prefabs (§18): what a scene node's `prefab` name spawns, and the
//! colliders built for it.

use crate::components::{ColliderRef, Hanging, Material, Mesh, NoShadowCast, Transform};
use crate::lights::PointLight;
use crate::physics::{tag_surface, to_rapier, Physics};
use crate::{rope, Surface};
use bevy_ecs::prelude::*;
use feather_assets::MeshData;
use feather_render::MeshId;
use glam::{Mat4, Vec3};
use rapier3d::prelude::{ColliderHandle, Vector};
use std::collections::HashMap;

/// What a prefab spawn function gets: the node's placement plus whatever
/// geometry and parameters it carries.
pub struct SpawnArgs<'a> {
    pub transform: Mat4,
    /// `None` for a marker node — geometry-free, there only to place a prefab.
    pub mesh: Option<MeshId>,
    pub material: u32,
    /// Local-space geometry, for building a collider.
    pub mesh_data: Option<&'a MeshData>,
    /// Its bake (§17), whose LODs can stand in for an over-budget mesh.
    pub baked: Option<&'a feather_assets::bake::BakedMesh>,
    pub spec: Option<&'a feather_assets::PrefabSpec>,
    /// What the mesh's material sounds like underfoot (§20), for its collider.
    pub surface: Surface,
}

impl SpawnArgs<'_> {
    /// A boolean param, falling back to `default` when absent or the wrong type.
    pub fn flag(&self, key: &str, default: bool) -> bool {
        self.spec.and_then(|s| s.bool(key)).unwrap_or(default)
    }
}

/// §18's `HashMap<PrefabId, SpawnFn>`: a new kind of thing is a new function
/// here, not a change to the scene format.
pub type SpawnFn = fn(&mut World, &SpawnArgs);

/// Static level geometry, with two switches read from `params`:
///
/// - `collide` (default true) — off closes §26's "no per-node opt-out"; a few
///   hundred decorative props otherwise each build a trimesh at load.
/// - `shadow` (default true) — off applies `NoShadowCast`. The demo ground
///   already does this (a flat slab casts nothing useful but rasterizes the
///   whole shadow map); this makes it authorable per node instead of hardcoded.
///
/// One parameterised prefab rather than a `no_collide` / `no_shadow` pair: the
/// switches are independent, so separate ids would need one per combination.
pub fn spawn_prop(world: &mut World, args: &SpawnArgs) {
    let collide = args.flag("collide", true);
    let shadow = args.flag("shadow", true);
    spawn_scene_node(world, args, collide, !shadow);
}

/// A punctual light (§12), optionally attached to geometry. Params `color`
/// (linear rgb), `intensity` and `radius`, each falling back to a sane default
/// so a bare `{"prefab": "point_light"}` still lights something.
///
/// Also honours `prop`'s `collide` and `shadow` switches with the same defaults:
/// a lamp with geometry is still a physical object, so it collides and casts a
/// sun shadow unless the scene says otherwise. Special-casing it to never cast
/// would make `point_light` the one prefab where `shadow` silently did nothing.
///
/// This is the payoff of §18's registry: a new kind of thing is one more
/// function here, with no change to the scene format — the test scene has been
/// carrying `point_light` nodes since before there was a light system.
pub fn spawn_point_light(world: &mut World, args: &SpawnArgs) {
    let d = PointLight::default();
    let radius = args.spec.and_then(|s| s.f32("radius")).unwrap_or(d.radius);
    let light = PointLight {
        color: args.spec.and_then(|s| s.vec3("color")).unwrap_or(d.color),
        intensity: args
            .spec
            .and_then(|s| s.f32("intensity"))
            .unwrap_or(d.intensity),
        radius,
        // A source larger than the light's reach is meaningless, and a negative
        // one would invert the representative-point clamp in the shader.
        source_radius: args
            .spec
            .and_then(|s| s.f32("source_radius"))
            .unwrap_or(d.source_radius)
            .clamp(0.0, radius.max(0.0)),
    };
    // Geometry is optional: a marker node lights without being visible, a mesh
    // node is a lamp that both emits and renders.
    match spawn_scene_node(
        world,
        args,
        args.flag("collide", true),
        !args.flag("shadow", true),
    ) {
        Some(e) => {
            world.entity_mut(e).insert(light);
        }
        None => {
            world.spawn((Transform(args.transform), light));
        }
    }
}

/// A node with no prefab at all: geometry that collides and casts, which is what
/// every scene node did before prefabs existed.
pub fn spawn_static_prop(world: &mut World, args: &SpawnArgs) {
    spawn_scene_node(world, args, true, false);
}

/// A shootable orb (the playable loop's target): a glowing sphere that
/// flashes on a hit and pops after `hits` of them, respawning a few seconds
/// later where the level put it. The node's own geometry is ignored — an orb
/// is its own mesh (weapon.rs). Params: `hits` (default 3), `radius`
/// (metres, default 0.5), `color` (linear rgb, default a warm amber),
/// `intensity` (its light, default 3).
pub fn spawn_target(world: &mut World, args: &SpawnArgs) {
    let hits = args
        .spec
        .and_then(|s| s.f32("hits"))
        .filter(|v| *v >= 1.0)
        .map(|v| v as u32)
        .unwrap_or(3);
    let radius = args
        .spec
        .and_then(|s| s.f32("radius"))
        .filter(|v| *v > 0.0)
        .unwrap_or(0.5);
    let color = args
        .spec
        .and_then(|s| s.vec3("color"))
        .filter(|c| c.min_element() >= 0.0)
        .unwrap_or(Vec3::new(1.0, 0.62, 0.25));
    let intensity = args
        .spec
        .and_then(|s| s.f32("intensity"))
        .filter(|v| *v >= 0.0)
        .unwrap_or(3.0);
    world.spawn(crate::weapon::target_bundle(
        args.transform,
        hits,
        radius,
        color,
        intensity,
    ));
}

/// Shared body of the geometry prefabs.
/// Triangle budget above which a prop collides as a convex hull instead of its
/// exact mesh (§15). Measured on a 34k-triangle Poly Haven lantern: standing on
/// its trimesh cost 27 ms per physics tick in debug (213 ms stepping off the
/// rim) and 2.5 ms in release, and the fixed-step catch-up multiplied that
/// into seconds per frame. Props of a few hundred triangles cost nothing
/// measurable, so they keep exact collision.
pub const TRIMESH_MAX_TRIS: usize = 2048;
/// Largest geometric error, in world units, of a baked LOD used as a prop's
/// collider in place of its over-budget full mesh (§17). Far below what the
/// player can feel (capsule radius 0.35, autostep 0.4), and far closer than
/// the convex hull it replaces, whose error is the depth of every concavity.
pub const COLLISION_TOLERANCE: f32 = 0.05;

/// The finest baked LOD usable as a collider: within `TRIMESH_MAX_TRIS` and
/// within `COLLISION_TOLERANCE` once its error is scaled to the node
/// (`world_scale`, its largest axis scale). `None` if none is. LOD errors
/// never decrease and triangle counts never grow along the chain, so the first
/// level under the budget is the only candidate: if it's too coarse, every
/// later one is too.
pub fn collision_lod(lods: &[feather_assets::bake::Lod], world_scale: f32) -> Option<usize> {
    let i = lods
        .iter()
        .position(|l| l.indices.len() / 3 <= TRIMESH_MAX_TRIS)?;
    (lods[i].error * world_scale <= COLLISION_TOLERANCE).then_some(i)
}

/// A transform's largest axis scale: how much it can stretch an error.
pub fn max_axis_scale(m: &Mat4) -> f32 {
    m.x_axis
        .truncate()
        .length()
        .max(m.y_axis.truncate().length())
        .max(m.z_axis.truncate().length())
}

/// What `build_collider` actually built, for the load-time `[scene]` line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltCollider {
    /// The full mesh (or a baked LOD0, which is the same triangles).
    Mesh,
    /// A coarser baked LOD standing in for an over-budget mesh.
    Lod(usize),
    Hull,
    Box,
}

/// Collider counts for a session, logged once after spawning.
#[derive(Resource, Default, Debug)]
pub struct ColliderStats {
    pub mesh: usize,
    pub lod: usize,
    pub hull: usize,
    pub boxes: usize,
    /// `dynamic` props' hulls (§15), not counted in `hull`.
    pub dynamic: usize,
    /// Per surface (§20), indexed by `Surface::index`.
    pub surfaces: [usize; Surface::ALL.len()],
}

impl ColliderStats {
    /// "12 concrete, 400 grass": the surfaces that have colliders.
    pub fn surfaces_line(&self) -> String {
        let parts: Vec<String> = Surface::ALL
            .into_iter()
            .filter(|s| self.surfaces[s.index()] > 0)
            .map(|s| format!("{} {}", self.surfaces[s.index()], s.name()))
            .collect();
        parts.join(", ")
    }
}

/// How a scene node collides (§15), from the `collider` prefab param.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColliderKind {
    /// Exact mesh within `TRIMESH_MAX_TRIS`, a convex hull above it.
    Auto,
    Mesh,
    Hull,
    /// Oriented bounding box: the hull of the local bounds' eight corners.
    Box,
    None,
}

impl ColliderKind {
    /// `collide: false` wins over `collider`, so older scenes keep meaning.
    pub fn from_args(args: &SpawnArgs, collide: bool) -> Self {
        if !collide {
            return Self::None;
        }
        match args.spec.and_then(|s| s.str("collider")) {
            None | Some("auto") => Self::Auto,
            Some("mesh") => Self::Mesh,
            Some("hull") => Self::Hull,
            Some("box") => Self::Box,
            Some("none") => Self::None,
            Some(other) => {
                eprintln!("[scene] unknown collider \"{other}\"; using auto");
                Self::Auto
            }
        }
    }
}

/// Build one primitive's static collider in world space, and say what it
/// built. `auto` uses the exact mesh within `TRIMESH_MAX_TRIS`; above it, a
/// baked LOD within `COLLISION_TOLERANCE` if there is one (§17), and a convex
/// hull otherwise. Flat geometry still gets a (zero-thickness) hull, which
/// collides fine (tested). Only points with no hull at all, too few or
/// collinear, fall back to the exact mesh.
pub fn build_collider(
    physics: &mut Physics,
    data: &MeshData,
    baked: Option<&feather_assets::bake::BakedMesh>,
    transform: Mat4,
    kind: ColliderKind,
) -> Option<(ColliderHandle, BuiltCollider)> {
    let world = |p: Vec3| to_rapier(transform.transform_point3(p));
    let trimesh = |physics: &mut Physics, vertices: &[feather_assets::Vertex], indices: &[u32]| {
        let verts: Vec<Vector> = vertices.iter().map(|v| world(Vec3::from(v.pos))).collect();
        let tris: Vec<[u32; 3]> = indices.as_chunks::<3>().0.to_vec();
        physics.add_static_trimesh(verts, tris)
    };
    let exact = |physics: &mut Physics| {
        trimesh(physics, &data.vertices, &data.indices).map(|h| (h, BuiltCollider::Mesh))
    };
    let kind = match kind {
        ColliderKind::Auto if data.indices.len() / 3 <= TRIMESH_MAX_TRIS => ColliderKind::Mesh,
        ColliderKind::Auto => {
            let lod = baked
                .and_then(|b| collision_lod(&b.lods, max_axis_scale(&transform)).map(|i| (b, i)));
            if let Some((b, i)) = lod {
                let built = if i == 0 {
                    BuiltCollider::Mesh
                } else {
                    BuiltCollider::Lod(i)
                };
                return trimesh(physics, &b.vertices, &b.lods[i].indices).map(|h| (h, built));
            }
            ColliderKind::Hull
        }
        k => k,
    };
    let (points, built): (Vec<Vector>, _) = match kind {
        ColliderKind::None => return None,
        ColliderKind::Mesh | ColliderKind::Auto => return exact(physics),
        ColliderKind::Hull => (
            data.vertices
                .iter()
                .map(|v| world(Vec3::from(v.pos)))
                .collect(),
            BuiltCollider::Hull,
        ),
        ColliderKind::Box => {
            let (lo, hi) = data.bounds();
            (
                (0..8)
                    .map(|i| {
                        let pick = |bit: usize, a: f32, b: f32| if i & bit == 0 { a } else { b };
                        world(Vec3::new(
                            pick(1, lo.x, hi.x),
                            pick(2, lo.y, hi.y),
                            pick(4, lo.z, hi.z),
                        ))
                    })
                    .collect(),
                BuiltCollider::Box,
            )
        }
    };
    match physics.add_static_hull(&points) {
        Some(h) => Some((h, built)),
        None => {
            eprintln!("[scene] degenerate {kind:?} collider; using the exact mesh");
            exact(physics)
        }
    }
}

pub fn spawn_scene_node(
    world: &mut World,
    args: &SpawnArgs,
    collide: bool,
    no_shadow: bool,
) -> Option<Entity> {
    let mesh = args.mesh?;
    let kind = ColliderKind::from_args(args, collide);
    let collider = args.mesh_data.and_then(|data| {
        build_collider(
            &mut world.resource_mut::<Physics>(),
            data,
            args.baked,
            args.transform,
            kind,
        )
    });
    if let Some((h, _)) = collider {
        tag_surface(
            &mut world.resource_mut::<Physics>().colliders[h],
            args.surface,
        );
    }
    if let (Some((_, built)), Some(mut stats)) =
        (collider, world.get_resource_mut::<ColliderStats>())
    {
        match built {
            BuiltCollider::Mesh => stats.mesh += 1,
            BuiltCollider::Lod(_) => {
                stats.mesh += 1;
                stats.lod += 1;
            }
            BuiltCollider::Hull => stats.hull += 1,
            BuiltCollider::Box => stats.boxes += 1,
        }
        stats.surfaces[args.surface.index()] += 1;
    }
    let collider = collider.map(|(h, _)| h);
    let mut e = world.spawn((
        Transform(args.transform),
        Mesh(mesh),
        Material(args.material),
    ));
    if let Some(c) = collider {
        e.insert(ColliderRef(c));
    }
    if no_shadow {
        e.insert(NoShadowCast);
    }
    Some(e.id())
}

/// `player_start` places the player and is consumed before the world is built,
/// so it has no spawn function of its own — see `Session::new`.
pub fn prefab_registry() -> HashMap<&'static str, SpawnFn> {
    let mut r: HashMap<&'static str, SpawnFn> = HashMap::new();
    r.insert("prop", spawn_prop as SpawnFn);
    r.insert("point_light", spawn_point_light as SpawnFn);
    r.insert("hanging", spawn_hanging as SpawnFn);
    r.insert("target", spawn_target as SpawnFn);
    r.insert("dynamic", crate::dynamic::spawn_dynamic as SpawnFn);
    r
}

/// The parameters a `hanging` node (§15) may carry.
pub const HANGING_PARAMS: [&str; 6] = ["length", "segments", "radius", "wind", "shadow", "light"];

/// What a `hanging` node's `light` object may carry.
pub const HANGING_LIGHT_PARAMS: [&str; 5] = ["color", "intensity", "radius", "source_radius", "at"];

/// A `hanging` node's light (§12), if its params have a `light` object:
/// `point_light`'s params over its defaults, plus `at`, where the light sits
/// in the item's own space (default its origin). Also the keys it couldn't
/// use, as `light.<key>`.
pub fn hanging_light(
    spec: Option<&feather_assets::PrefabSpec>,
) -> (Option<(PointLight, Vec3)>, Vec<String>) {
    let Some(value) = spec.and_then(|s| s.params.get("light")) else {
        return (None, Vec::new());
    };
    let Some(object) = value.as_object() else {
        return (None, vec!["light".to_string()]);
    };
    let l = feather_assets::PrefabSpec {
        id: "light".into(),
        params: value.clone(),
    };
    let mut bad: Vec<String> = object
        .keys()
        .filter(|k| !HANGING_LIGHT_PARAMS.contains(&k.as_str()))
        .map(|k| format!("light.{k}"))
        .collect();
    let mut light = PointLight::default();
    let mut at = Vec3::ZERO;
    for key in HANGING_LIGHT_PARAMS {
        if !object.contains_key(key) {
            continue;
        }
        let ok = match key {
            "color" => l
                .vec3(key)
                .filter(|c| c.min_element() >= 0.0)
                .map(|c| light.color = c),
            "at" => l.vec3(key).map(|v| at = v),
            "intensity" => l
                .f32(key)
                .filter(|v| *v >= 0.0)
                .map(|v| light.intensity = v),
            "radius" => l.f32(key).filter(|v| *v > 0.0).map(|v| light.radius = v),
            _ => l
                .f32(key)
                .filter(|v| *v >= 0.0)
                .map(|v| light.source_radius = v),
        };
        if ok.is_none() {
            bad.push(format!("light.{key}"));
        }
    }
    light.source_radius = light.source_radius.min(light.radius);
    (Some((light, at)), bad)
}

/// A `hanging` node's rope, over `RopeParams::default()`, and the params it
/// couldn't use: unknown ones, and ones out of range or not numbers.
pub fn hanging_params(
    spec: Option<&feather_assets::PrefabSpec>,
) -> (rope::RopeParams, Vec<String>) {
    let mut p = rope::RopeParams::default();
    let Some(spec) = spec else {
        return (p, Vec::new());
    };
    let mut bad: Vec<String> = spec
        .params
        .as_object()
        .into_iter()
        .flat_map(|o| o.keys())
        .filter(|k| !HANGING_PARAMS.contains(&k.as_str()))
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
    if let Some(v) = num("length", |v| v > 0.0 && v <= 50.0) {
        p.length = v;
    }
    if let Some(v) = num("segments", |v| {
        v.fract() == 0.0 && (1.0..=64.0).contains(&v)
    }) {
        p.segments = v as usize;
    }
    if let Some(v) = num("radius", |v| v > 0.0 && v <= 0.5) {
        p.radius = v;
    }
    if let Some(v) = num("wind", |v| (0.0..=10.0).contains(&v)) {
        p.wind = v;
    }
    (p, bad)
}

/// Something on a rope (§15): the node is the item at rest, hung by its
/// mesh's origin, and the rope's anchor is `length` straight above it. The
/// rope sways in the level's wind; neither it nor the item collides.
pub fn spawn_hanging(world: &mut World, args: &SpawnArgs) {
    let (params, mut bad) = hanging_params(args.spec);
    let (light, bad_light) = hanging_light(args.spec);
    bad.extend(bad_light);
    for key in bad {
        eprintln!("[scene] hanging: can't use param {key:?}; ignored");
    }
    let (scale, rotation, at) = args.transform.to_scale_rotation_translation();
    let local = Mat4::from_scale_rotation_translation(scale, rotation, Vec3::ZERO);
    world.spawn(Hanging {
        rope: rope::Rope::new(at + Vec3::Y * params.length, params),
        item: args.mesh.map(|m| (m, args.material, local)),
        casts: args.flag("shadow", true),
        light,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::{Player, GROUND_Y};
    use crate::level::mesh_surfaces;

    use crate::physics::collider_surface;
    use crate::testing::{node, setup, step};

    // ---- §18 prefabs ----

    /// A world with just enough in it to spawn scene nodes into.
    fn prefab_world() -> World {
        let mut w = World::new();
        w.insert_resource(Physics::new());
        w
    }

    fn spawn_one(w: &mut World, spec: Option<&feather_assets::PrefabSpec>, cube: &MeshData) {
        let args = SpawnArgs {
            transform: Mat4::IDENTITY,
            mesh: Some(MeshId(0)),
            material: 0,
            mesh_data: Some(cube),
            baked: None,
            spec,
            surface: mesh_surfaces(std::slice::from_ref(cube)).0[0],
        };
        match spec.and_then(|s| prefab_registry().get(s.id.as_str()).copied()) {
            Some(f) => f(w, &args),
            None => spawn_static_prop(w, &args),
        }
    }

    #[test]
    fn collide_param_controls_whether_a_collider_is_built() {
        let cube = MeshData::cube(1.0);

        // Default (no prefab): geometry collides, exactly as before prefabs.
        let mut w = prefab_world();
        spawn_one(&mut w, None, &cube);
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 1);
        assert_eq!(w.resource::<Physics>().colliders.len(), 1);

        // collide: false skips the trimesh — the §26 per-node opt-out.
        let n = node(Some("prop"), serde_json::json!({ "collide": false }));
        let mut w = prefab_world();
        spawn_one(&mut w, n.prefab.as_ref(), &cube);
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 0);
        assert_eq!(
            w.resource::<Physics>().colliders.len(),
            0,
            "collider was still built"
        );
        // It must still render.
        assert_eq!(w.query::<&Mesh>().iter(&w).count(), 1);
    }

    #[test]
    fn shadow_param_controls_the_no_cast_marker() {
        let cube = MeshData::cube(1.0);
        let n = node(Some("prop"), serde_json::json!({ "shadow": false }));
        let mut w = prefab_world();
        spawn_one(&mut w, n.prefab.as_ref(), &cube);
        assert_eq!(w.query::<&NoShadowCast>().iter(&w).count(), 1);
        // Independent of collision: this one still collides.
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 1);

        // Absent param defaults to casting.
        let n = node(Some("prop"), serde_json::json!({}));
        let mut w = prefab_world();
        spawn_one(&mut w, n.prefab.as_ref(), &cube);
        assert_eq!(w.query::<&NoShadowCast>().iter(&w).count(), 0);
    }

    #[test]
    fn target_spawns_a_glowing_orb() {
        use crate::weapon::Target;
        let n = node(
            Some("target"),
            serde_json::json!({ "hits": 2.0, "color": [1.0, 0.0, 0.0] }),
        );
        let mut w = prefab_world();
        let args = SpawnArgs {
            transform: Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
            mesh: None,
            material: 0,
            mesh_data: None,
            baked: None,
            spec: n.prefab.as_ref(),
            surface: Surface::default(),
        };
        prefab_registry().get("target").copied().unwrap()(&mut w, &args);
        // One orb: its own sphere and light, no collider (weapon.rs tests the
        // ray against `radius`), placed at the node's transform.
        let mut q = w.query::<(&Transform, &Mesh, &PointLight, &Target)>();
        assert_eq!(q.iter(&w).count(), 1, "exactly one orb");
        let (t, _, light, target) = q.iter(&w).next().unwrap();
        assert_eq!(target.hits_left, 2, "the hits param");
        assert_eq!(light.color, Vec3::new(1.0, 0.0, 0.0), "the color param");
        assert!((t.0.transform_point3(Vec3::ZERO) - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
        assert_eq!(
            w.resource::<Physics>().colliders.len(),
            0,
            "orbs don't collide"
        );
    }

    #[test]
    fn unknown_prefab_falls_back_to_static_geometry() {
        let cube = MeshData::cube(1.0);
        // `trigger` has no implementation yet (§18 lists it; nothing consumes it).
        // It must still appear as geometry rather than vanishing or aborting the
        // load. This deliberately names a prefab that does not exist — when one
        // is added, point this at another unimplemented id rather than deleting
        // the test, since the fallback is what keeps scenes forward-compatible.
        let n = node(Some("trigger"), serde_json::json!({ "radius": 8.0 }));
        assert!(prefab_registry().get("trigger").is_none());
        let mut w = prefab_world();
        spawn_one(&mut w, n.prefab.as_ref(), &cube);
        assert_eq!(w.query::<&Mesh>().iter(&w).count(), 1);
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 1);
    }

    /// `hanging` (§15): params land, defaults fill in, and bad ones are named.
    #[test]
    fn hanging_params_apply_and_bad_ones_are_named() {
        let d = rope::RopeParams::default();
        assert_eq!(hanging_params(None), (d, Vec::new()));
        let spec = |v| feather_assets::PrefabSpec {
            id: "hanging".into(),
            params: v,
        };
        let (p, bad) = hanging_params(Some(&spec(serde_json::json!({
            "length": 2.5, "segments": 16, "radius": 0.01, "wind": 0.35, "shadow": false
        }))));
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(
            p,
            rope::RopeParams {
                length: 2.5,
                segments: 16,
                radius: 0.01,
                wind: 0.35
            }
        );
        let (p, mut bad) = hanging_params(Some(&spec(serde_json::json!({
            "length": -1.0, "segments": 2.5, "radius": "thin", "wind": 0.5, "colour": 1
        }))));
        bad.sort();
        assert_eq!(bad, ["colour", "length", "radius", "segments"]);
        assert_eq!(p, rope::RopeParams { wind: 0.5, ..d });
    }

    /// The node is the item at rest; the anchor is `length` above it, and
    /// nothing collides.
    #[test]
    fn a_hanging_node_hangs_its_item_below_the_anchor() {
        let cube = MeshData::cube(1.0);
        let mut w = prefab_world();
        let spec = feather_assets::PrefabSpec {
            id: "hanging".into(),
            params: serde_json::json!({ "length": 2.0 }),
        };
        let at = Vec3::new(3.0, 4.0, -1.0);
        let placed = Mat4::from_scale_rotation_translation(
            Vec3::splat(1.5),
            glam::Quat::from_rotation_y(0.6),
            at,
        );
        let args = SpawnArgs {
            transform: placed,
            mesh: Some(MeshId(0)),
            material: 7,
            mesh_data: Some(&cube),
            baked: None,
            spec: Some(&spec),
            surface: Surface::default(),
        };
        spawn_hanging(&mut w, &args);
        let h = w.query::<&Hanging>().single(&w).unwrap();
        assert_eq!(h.rope.anchor(), at + Vec3::Y * 2.0);
        assert!((h.rope.end() - at).length() < 1e-5);
        let (mesh, material, local) = h.item.unwrap();
        assert_eq!((mesh, material), (MeshId(0), 7));
        // Drawn at rest exactly where the node put it.
        let drawn = rope::item_matrix(&h.rope.points, local);
        assert!(drawn.abs_diff_eq(placed, 1e-5), "{drawn} vs {placed}");
        assert!(h.casts);
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 0);
        assert_eq!(
            w.query::<&Mesh>().iter(&w).count(),
            0,
            "drawn by its rope, not as a prop"
        );
    }

    /// A `hanging` node's `light` (§12): off without one; its params over
    /// `point_light`'s defaults, `at` in the item's space; bad keys named.
    #[test]
    fn a_hanging_light_is_parsed() {
        let spec = |v| feather_assets::PrefabSpec {
            id: "hanging".into(),
            params: v,
        };
        assert_eq!(
            hanging_light(Some(&spec(serde_json::json!({ "length": 2.0 })))).0,
            None
        );
        let (l, bad) = hanging_light(Some(&spec(serde_json::json!({ "light": {
            "color": [1.0, 0.7, 0.4], "intensity": 6.0, "radius": 11.0,
            "source_radius": 0.05, "at": [0.0, -0.17, 0.0]
        } }))));
        assert!(bad.is_empty(), "{bad:?}");
        let (l, at) = l.unwrap();
        assert_eq!(l.color, Vec3::new(1.0, 0.7, 0.4));
        assert_eq!((l.intensity, l.radius, l.source_radius), (6.0, 11.0, 0.05));
        assert_eq!(at, Vec3::new(0.0, -0.17, 0.0));
        // An empty object is a default light at the item's origin.
        let (l, bad) = hanging_light(Some(&spec(serde_json::json!({ "light": {} }))));
        assert!(bad.is_empty());
        assert_eq!(l, Some((PointLight::default(), Vec3::ZERO)));
        let (l, mut bad) = hanging_light(Some(&spec(serde_json::json!({ "light": {
            "radius": -1.0, "color": [1, 1], "flicker": true, "intensity": 3.0
        } }))));
        bad.sort();
        assert_eq!(bad, ["light.color", "light.flicker", "light.radius"]);
        assert_eq!(l.unwrap().0.intensity, 3.0);
        let (l, bad) = hanging_light(Some(&spec(serde_json::json!({ "light": true }))));
        assert_eq!((l, bad), (None, vec!["light".to_string()]));
        // hanging_params knows the key, so it isn't unknown there.
        assert!(
            hanging_params(Some(&spec(serde_json::json!({ "light": {} }))))
                .1
                .is_empty()
        );
    }

    #[test]
    fn a_material_surface_tags_the_collider() {
        use Surface::{Concrete, Grass, Snow};
        let cube = |surface: Option<&str>| {
            let mut m = MeshData::cube(1.0);
            m.material.surface = surface.map(str::to_string);
            m
        };
        let meshes = [
            cube(Some("grass")),
            cube(Some("Snow")),
            cube(None),
            cube(Some("lava")),
            cube(Some("lava")),
        ];
        let (surfaces, unknown) = mesh_surfaces(&meshes);
        assert_eq!(surfaces, [Grass, Snow, Concrete, Concrete, Concrete]);
        assert_eq!(unknown, ["lava"], "each unknown name is reported once");

        // Spawned as a scene node, the collider carries it.
        for (mesh, want) in meshes.iter().zip(surfaces) {
            let mut w = prefab_world();
            spawn_one(&mut w, None, mesh);
            let c = w
                .query::<&ColliderRef>()
                .iter(&w)
                .next()
                .expect("collider")
                .0;
            let got = collider_surface(&w.resource::<Physics>().colliders[c]);
            assert_eq!(got, want);
        }

        let mut stats = ColliderStats::default();
        stats.surfaces[Concrete.index()] = 2;
        stats.surfaces[Snow.index()] = 1;
        assert_eq!(stats.surfaces_line(), "2 concrete, 1 snow");
    }

    /// The shape rapier built for the single collider-carrying prop in `w`.
    fn collider_shape(w: &mut World) -> Option<rapier3d::parry::shape::ShapeType> {
        let c = w.query::<&ColliderRef>().iter(w).next()?.0;
        Some(
            w.resource::<Physics>()
                .colliders
                .get(c)?
                .shape()
                .shape_type(),
        )
    }

    #[test]
    fn collider_param_selects_the_proxy() {
        use rapier3d::parry::shape::ShapeType::{ConvexPolyhedron, TriMesh};
        let cube = MeshData::cube(1.0); // 12 triangles
        let dense = MeshData::uv_sphere(64, 64, 1.0); // well past TRIMESH_MAX_TRIS
        assert!(dense.indices.len() / 3 > TRIMESH_MAX_TRIS);
        let cases: [(&MeshData, serde_json::Value, Option<_>); 8] = [
            // Auto: exact under the budget, a hull over it.
            (&cube, serde_json::json!({}), Some(TriMesh)),
            (&dense, serde_json::json!({}), Some(ConvexPolyhedron)),
            // Explicit choices override the budget either way.
            (
                &cube,
                serde_json::json!({ "collider": "hull" }),
                Some(ConvexPolyhedron),
            ),
            (
                &dense,
                serde_json::json!({ "collider": "mesh" }),
                Some(TriMesh),
            ),
            (
                &dense,
                serde_json::json!({ "collider": "box" }),
                Some(ConvexPolyhedron),
            ),
            (&dense, serde_json::json!({ "collider": "none" }), None),
            // `collide: false` still wins, so older scenes keep their meaning.
            (
                &cube,
                serde_json::json!({ "collide": false, "collider": "mesh" }),
                None,
            ),
            // An unknown value warns and falls back to auto.
            (
                &cube,
                serde_json::json!({ "collider": "bogus" }),
                Some(TriMesh),
            ),
        ];
        for (i, (mesh, params, want)) in cases.into_iter().enumerate() {
            let spec = feather_assets::PrefabSpec {
                id: "prop".into(),
                params,
            };
            let mut w = prefab_world();
            spawn_one(&mut w, Some(&spec), mesh);
            assert_eq!(collider_shape(&mut w), want, "case {i}");
        }
    }

    #[test]
    fn a_flat_hull_still_holds_the_player() {
        // Coplanar points: parry still builds a (zero-thickness) hull rather
        // than failing. What matters is that it collides, so check that
        // directly: a floor tile asked for a hull, raised 1 m, must hold the
        // player instead of letting them fall through to the ground.
        let v = |x: f32, z: f32| feather_assets::Vertex {
            pos: [x, 0.0, z],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
        };
        let quad = MeshData {
            vertices: vec![v(-2.0, -2.0), v(2.0, -2.0), v(2.0, 2.0), v(-2.0, 2.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            material: Default::default(),
        };
        let at = Mat4::from_translation(Vec3::new(0.0, GROUND_Y + 1.0, 0.0));
        let (mut physics, _) = setup(&[], Vec3::new(20.0, GROUND_Y, 20.0));
        build_collider(&mut physics, &quad, None, at, ColliderKind::Hull).expect("collider");
        let mut p = Player::new(&mut physics, Vec3::new(0.0, GROUND_Y + 2.0, 0.0));
        physics.step();
        for _ in 0..90 {
            step(&mut p, &mut physics, Vec3::ZERO, false, 0.0, false);
        }
        assert!(p.on_ground, "should stand on the tile");
        assert!(
            (p.pos.y - (GROUND_Y + 1.0)).abs() < 0.05,
            "stood at {}",
            p.pos.y
        );
    }

    #[test]
    fn a_dense_prop_is_walkable_as_a_hull() {
        // The lantern case, synthetic: a 30 cm ball of ~32k triangles on the
        // ground. Auto makes it a hull; the player can stand on it and walk off.
        let ball = MeshData::uv_sphere(128, 128, 0.3);
        assert!(ball.indices.len() / 3 > TRIMESH_MAX_TRIS);
        let at = Mat4::from_translation(Vec3::new(0.0, GROUND_Y + 0.3, 0.0));
        let (mut physics, _) = setup(&[], Vec3::new(20.0, GROUND_Y, 20.0));
        let (c, built) =
            build_collider(&mut physics, &ball, None, at, ColliderKind::Auto).expect("collider");
        assert_eq!(built, BuiltCollider::Hull, "no bake: over budget is a hull");
        assert_eq!(
            physics.colliders.get(c).unwrap().shape().shape_type(),
            rapier3d::parry::shape::ShapeType::ConvexPolyhedron
        );
        let mut p = Player::new(&mut physics, Vec3::new(0.0, GROUND_Y + 1.0, 0.0));
        physics.step();
        for _ in 0..60 {
            step(&mut p, &mut physics, Vec3::ZERO, false, 0.0, false);
        }
        assert!(p.on_ground, "should stand on the ball");
        assert!(
            p.pos.y > GROUND_Y + 0.4,
            "stood at {}, not on top of the ball",
            p.pos.y
        );
        for _ in 0..90 {
            step(&mut p, &mut physics, Vec3::X, false, 0.0, false);
        }
        assert!(
            p.on_ground && (p.pos.y - GROUND_Y).abs() < 0.05,
            "landed at {}",
            p.pos.y
        );
    }

    fn lod(error: f32, tris: usize) -> feather_assets::bake::Lod {
        feather_assets::bake::Lod {
            error,
            indices: vec![0; tris * 3],
        }
    }

    #[test]
    fn collision_lod_takes_the_finest_level_within_budget_and_tolerance() {
        let chain = [
            lod(0.0, 8000),
            lod(0.01, 4000),
            lod(0.03, 2000),
            lod(0.2, 1000),
        ];
        // LOD1 is too big; LOD2 is the first under the budget, 3 cm off.
        assert_eq!(collision_lod(&chain, 1.0), Some(2));
        // At 2x its error is 6 cm, over tolerance, and so is every later level.
        assert_eq!(collision_lod(&chain, 2.0), None);
        // A small mesh qualifies as itself.
        assert_eq!(
            collision_lod(&[lod(0.0, 500), lod(0.02, 250)], 10.0),
            Some(0)
        );
        // Nothing under the budget at all.
        assert_eq!(
            collision_lod(&[lod(0.0, 9000), lod(0.001, 5000)], 1.0),
            None
        );
        assert_eq!(collision_lod(&[], 1.0), None);
    }

    /// A dense dish, 6 m wide and 1 m deep: a concave prop, over the triangle
    /// budget. Returns the fine mesh and a bake whose LOD1 is every 4th grid
    /// line (the same vertex array, as a real bake shares it).
    fn dish() -> (MeshData, feather_assets::bake::BakedMesh) {
        const N: usize = 64;
        let (r, depth) = (3.0f32, 1.0f32);
        let mut m = MeshData::cube(1.0);
        m.vertices.clear();
        m.indices.clear();
        for i in 0..=N {
            for j in 0..=N {
                let x = -r + 2.0 * r * j as f32 / N as f32;
                let z = -r + 2.0 * r * i as f32 / N as f32;
                let y = depth * ((x * x + z * z) / (r * r)).min(1.0);
                m.vertices.push(feather_assets::Vertex {
                    pos: [x, y, z],
                    normal: [0.0, 1.0, 0.0],
                    uv: [0.0, 0.0],
                });
            }
        }
        // Counter-clockwise from above, so every triangle faces up.
        let grid = |step: usize| {
            let mut idx = Vec::new();
            for i in (0..N).step_by(step) {
                for j in (0..N).step_by(step) {
                    let v = |i: usize, j: usize| (i * (N + 1) + j) as u32;
                    let (a, b, c, d) = (
                        v(i, j),
                        v(i + step, j),
                        v(i, j + step),
                        v(i + step, j + step),
                    );
                    idx.extend_from_slice(&[a, b, c, c, b, d]);
                }
            }
            idx
        };
        m.indices = grid(1);
        let baked = feather_assets::bake::BakedMesh {
            vertices: m.vertices.clone(),
            lods: vec![
                feather_assets::bake::Lod {
                    error: 0.0,
                    indices: m.indices.clone(),
                },
                feather_assets::bake::Lod {
                    error: 0.01,
                    indices: grid(4),
                },
            ],
        };
        (m, baked)
    }

    /// Drop the player into the dish; where do its feet come to rest?
    fn rest_height_in_dish(
        baked: Option<&feather_assets::bake::BakedMesh>,
    ) -> (f32, BuiltCollider) {
        let (mesh, bake) = dish();
        assert!(mesh.indices.len() / 3 > TRIMESH_MAX_TRIS);
        let baked = baked.map(|_| &bake);
        let at = Mat4::from_translation(Vec3::new(0.0, GROUND_Y + 0.2, 0.0));
        let (mut physics, _) = setup(&[], Vec3::new(20.0, GROUND_Y, 20.0));
        let (_, built) =
            build_collider(&mut physics, &mesh, baked, at, ColliderKind::Auto).expect("collider");
        let mut p = Player::new(&mut physics, Vec3::new(0.0, GROUND_Y + 2.5, 0.0));
        physics.step();
        for _ in 0..90 {
            step(&mut p, &mut physics, Vec3::ZERO, false, 0.0, false);
        }
        assert!(p.on_ground, "never landed ({built:?})");
        (p.pos.y - GROUND_Y, built)
    }

    /// The point of LOD collision: a hull fills the dish up to its rim, a LOD
    /// trimesh lets you stand in it.
    #[test]
    fn a_baked_lod_keeps_a_concave_prop_concave() {
        let (y, built) = rest_height_in_dish(Some(&dish().1));
        println!("dish, LOD1 trimesh: feet at {y:.3} above ground");
        assert_eq!(built, BuiltCollider::Lod(1));
        assert!(
            y < 0.5,
            "stood at {y}, not down in the dish (bottom at 0.2)"
        );
        let (y, built) = rest_height_in_dish(None);
        println!("dish, hull: feet at {y:.3} above ground");
        assert_eq!(built, BuiltCollider::Hull);
        assert!(
            y > 1.1,
            "stood at {y}, but the hull's lid is at the rim (1.2)"
        );
    }

    #[test]
    fn explicit_mesh_collider_stays_exact() {
        let (mesh, bake) = dish();
        let (mut physics, _) = setup(&[], Vec3::new(20.0, GROUND_Y, 20.0));
        let (_, built) = build_collider(
            &mut physics,
            &mesh,
            Some(&bake),
            Mat4::IDENTITY,
            ColliderKind::Mesh,
        )
        .expect("collider");
        assert_eq!(built, BuiltCollider::Mesh);
    }

    #[test]
    fn point_light_prefab_reads_params_with_defaults() {
        let cube = MeshData::cube(1.0);
        let spec = feather_assets::PrefabSpec {
            id: "point_light".into(),
            params: serde_json::json!({
                "color": [1.0, 0.5, 0.25], "intensity": 7.5, "radius": 3.0
            }),
        };
        let mut w = prefab_world();
        spawn_one(&mut w, Some(&spec), &cube);
        let l = *w.query::<&PointLight>().single(&w).unwrap();
        assert_eq!(l.color, Vec3::new(1.0, 0.5, 0.25));
        assert_eq!(l.intensity, 7.5);
        assert_eq!(l.radius, 3.0);
        assert_eq!(l.source_radius, PointLight::default().source_radius);

        // source_radius: read when given, clamped to [0, radius].
        for (given, want) in [(0.4, 0.4), (0.0, 0.0), (-1.0, 0.0), (9.0, 3.0)] {
            let sized = feather_assets::PrefabSpec {
                id: "point_light".into(),
                params: serde_json::json!({ "radius": 3.0, "source_radius": given }),
            };
            let mut w = prefab_world();
            spawn_one(&mut w, Some(&sized), &cube);
            let l = *w.query::<&PointLight>().single(&w).unwrap();
            assert_eq!(l.source_radius, want, "source_radius {given}");
        }

        // The geometry switches behave as they do on `prop`, rather than
        // point_light being the one prefab where `shadow` silently does nothing.
        let dark = feather_assets::PrefabSpec {
            id: "point_light".into(),
            params: serde_json::json!({ "shadow": false, "collide": false }),
        };
        let mut w = prefab_world();
        spawn_one(&mut w, Some(&dark), &cube);
        assert_eq!(w.query::<&NoShadowCast>().iter(&w).count(), 1);
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 0);
        assert_eq!(w.query::<&PointLight>().iter(&w).count(), 1);

        // Defaults match `prop`: a lamp is still a physical object.
        let plain = feather_assets::PrefabSpec {
            id: "point_light".into(),
            params: serde_json::json!({}),
        };
        let mut w = prefab_world();
        spawn_one(&mut w, Some(&plain), &cube);
        assert_eq!(w.query::<&NoShadowCast>().iter(&w).count(), 0);
        assert_eq!(w.query::<&ColliderRef>().iter(&w).count(), 1);

        // A bare prefab still lights something, rather than being a black light.
        let bare = feather_assets::PrefabSpec {
            id: "point_light".into(),
            params: serde_json::json!({}),
        };
        let mut w = prefab_world();
        spawn_one(&mut w, Some(&bare), &cube);
        let l = *w.query::<&PointLight>().single(&w).unwrap();
        let d = PointLight::default();
        assert_eq!(
            (l.color, l.intensity, l.radius),
            (d.color, d.intensity, d.radius)
        );
        assert!(l.intensity > 0.0 && l.radius > 0.0);
    }
}
