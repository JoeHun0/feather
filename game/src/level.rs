//! Building a level (§18): loading the scenes, reading their markers, and
//! `build_world` — a session's world, before the GPU upload.

use crate::components::{
    integrate, rand01, ropes, tick, ColliderRef, FrameCount, LevelWind, Material, Mesh,
    NoShadowCast, Position, PrevPosition, PrevRotation, Rotation, Scale, Spin, Velocity, GRID,
    MESH_BUILTIN_COUNT, MESH_CUBE, MESH_LEVEL_CUBE, MESH_SPHERE, PALETTE,
};
use crate::controller::{
    physics_step_sys, player_readback_sys, player_target_sys, InputState, Look, Player, GROUND_Y,
};
use crate::physics::Physics;
use crate::prefab::{prefab_registry, spawn_static_prop, ColliderStats, SpawnArgs};
use crate::{rope, weather, Surface};
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ExecutorKind;
use feather_assets::MeshData;
use feather_render::{Environment, MeshId};
use glam::{Mat4, Vec3};
use std::time::Instant;

/// Load the mesh registry and every CLI scene. Runs before the world exists,
/// because `player_start` decides where the player is built.
///
/// Built-ins occupy the fixed MESH_* slots first (the demo sphere/cube, then the
/// unit cube every static level piece is scaled from); each scene's meshes are
/// appended and its node indices rebased onto them.
/// Every CLI scene's meshes and nodes, merged after the built-in meshes, plus
/// the first loaded scene's sky-visibility key (§13): one volume per session.
pub fn load_scenes(
    scenes: &[String],
) -> (Vec<MeshData>, Vec<feather_assets::SceneNode>, Option<u64>) {
    let mut meshes: Vec<MeshData> = vec![
        MeshData::uv_sphere(16, 24, 0.5),
        MeshData::cube(1.0),
        MeshData::cube(1.0),
        rope_mesh(),
    ];
    let mut nodes: Vec<feather_assets::SceneNode> = Vec::new();
    let mut sky_key = None;
    for path in scenes {
        match feather_assets::load_gltf_scene(path) {
            Ok(scene) => {
                if sky_key.is_none() {
                    sky_key = Some(feather_assets::bake::sky_key(&scene));
                }
                eprintln!(
                    "loaded {path}: {} meshes, {} nodes",
                    scene.meshes.len(),
                    scene.nodes.len()
                );
                let base = meshes.len();
                meshes.extend(scene.meshes);
                nodes.extend(scene.nodes.into_iter().map(|n| feather_assets::SceneNode {
                    mesh: n.mesh.map(|m| base + m),
                    transform: n.transform,
                    prefab: n.prefab,
                }));
            }
            Err(e) => eprintln!("failed to load {path}: {e} (skipped)"),
        }
    }
    (meshes, nodes, sky_key)
}

/// Each mesh's footstep surface (§20), from its material's `surface` tag,
/// plus the tag names that aren't surfaces (each once, for the caller to
/// report). Resolved per mesh rather than per node, so a typo on a tile used
/// 400 times is one warning. Unknown and untagged both mean concrete.
pub fn mesh_surfaces(meshes: &[MeshData]) -> (Vec<Surface>, Vec<&str>) {
    let mut unknown: Vec<&str> = Vec::new();
    let surfaces = meshes
        .iter()
        .map(|m| {
            let Some(name) = m.material.surface.as_deref() else {
                return Surface::default();
            };
            Surface::from_name(name).unwrap_or_else(|| {
                if !unknown.contains(&name) {
                    unknown.push(name);
                }
                Surface::default()
            })
        })
        .collect();
    (surfaces, unknown)
}

/// Where the player starts: a `player_start` marker's position and yaw, or the
/// hardcoded default when a scene does not place one.
pub fn player_start(nodes: &[feather_assets::SceneNode]) -> (Vec3, Option<f32>) {
    for n in nodes {
        let Some(spec) = n.prefab.as_ref() else {
            continue;
        };
        if spec.id == "player_start" {
            return (
                n.transform.transform_point3(Vec3::ZERO),
                spec.f32("yaw").map(|d| d.to_radians()),
            );
        }
    }
    (Vec3::new(0.0, GROUND_Y, 8.0), None)
}

/// The parameters an `environment` marker (§13, §18) may carry.
pub const ENVIRONMENT_PARAMS: [&str; 27] = [
    "ground",
    "wind_speed",
    "wind_azimuth",
    "weather",
    "time",
    "time_speed",
    "latitude",
    "day_of_year",
    "sun_elevation",
    "sun_azimuth",
    "sun_color",
    "sun_intensity",
    "sky_zenith",
    "sky_horizon",
    "sky_ground",
    "sky_sun_color",
    "sky_intensity",
    "sun_glow",
    "sun_disk",
    "fog_density",
    "fog_height",
    "fog_falloff",
    "fog_color",
    "fog_sun",
    "exposure",
    "exposure_min",
    "exposure_max",
];

/// Whether the level stands on the engine's ground, the 80×80 slab whose
/// top is `GROUND_Y`: the first `environment` marker's `ground`, true
/// unless it says false. A level that brings its own (relief, or a hole for
/// a pond, §18) says false, and nothing is under it but what it built.
/// Also the params it couldn't read.
pub fn level_ground(nodes: &[feather_assets::SceneNode]) -> (bool, Vec<String>) {
    let spec = nodes
        .iter()
        .filter_map(|n| n.prefab.as_ref())
        .find(|s| s.id == "environment");
    match spec.filter(|s| s.params.get("ground").is_some()) {
        None => (true, Vec::new()),
        Some(s) => match s.bool("ground") {
            Some(ground) => (ground, Vec::new()),
            None => (true, vec!["ground".to_string()]),
        },
    }
}

/// The level's wind (§15's ropes sway in it): the first `environment`
/// marker's `wind_speed` (m/s, at least 0) and `wind_azimuth` (the direction
/// it blows towards, degrees clockwise from north, -Z, as for the sun), over
/// `Wind::default()`. Also the params it couldn't read.
pub fn level_wind(nodes: &[feather_assets::SceneNode]) -> (rope::Wind, Vec<String>) {
    let mut wind = rope::Wind::default();
    let mut bad = Vec::new();
    let Some(spec) = nodes
        .iter()
        .filter_map(|n| n.prefab.as_ref())
        .find(|s| s.id == "environment")
    else {
        return (wind, bad);
    };
    if spec.params.get("wind_speed").is_some() {
        match spec.f32("wind_speed").filter(|v| *v >= 0.0) {
            Some(v) => wind.speed = v,
            None => bad.push("wind_speed".to_string()),
        }
    }
    if spec.params.get("wind_azimuth").is_some() {
        match spec.f32("wind_azimuth") {
            Some(deg) => {
                let (s, c) = deg.to_radians().sin_cos();
                wind.dir = Vec3::new(s, 0.0, -c);
            }
            None => bad.push("wind_azimuth".to_string()),
        }
    }
    (wind, bad)
}

/// The level's weather (§13): the first `environment` marker's `weather`
/// (a name, or `level` for the level's own look), `time` (the hour a new game
/// starts at, 0–24), `time_speed` (game seconds per real second, at least 0),
/// `latitude` (degrees, −89 to 89) and `day_of_year` (1–366), over
/// `LevelWeather::default()`. Also the params it couldn't read.
pub fn level_weather(nodes: &[feather_assets::SceneNode]) -> (weather::LevelWeather, Vec<String>) {
    let mut w = weather::LevelWeather::default();
    let mut bad = Vec::new();
    let Some(spec) = nodes
        .iter()
        .filter_map(|n| n.prefab.as_ref())
        .find(|s| s.id == "environment")
    else {
        return (w, bad);
    };
    let has = |key: &str| spec.params.get(key).is_some();
    if has("weather") {
        match spec.str("weather").and_then(weather::choice_named) {
            Some(c) => w.choice = c,
            None => bad.push("weather".to_string()),
        }
    }
    let mut num = |key: &str, ok: fn(f32) -> bool| {
        let v = spec.f32(key).filter(|v| ok(*v));
        if has(key) && v.is_none() {
            bad.push(key.to_string());
        }
        v
    };
    if let Some(t) = num("time", |v| (0.0..24.0).contains(&v)) {
        w.time = Some(t);
    }
    if let Some(v) = num("time_speed", |v| v >= 0.0) {
        w.speed = v;
    }
    if let Some(v) = num("latitude", |v| (-89.0..=89.0).contains(&v)) {
        w.path.latitude = v;
    }
    if let Some(v) = num("day_of_year", |v| (1.0..=366.0).contains(&v)) {
        w.path.day_of_year = v;
    }
    (w, bad)
}

/// The level's atmosphere (§13): the first `environment` marker's params over
/// `Environment::default()`, which is also what a level without one gets.
/// Like `player_start`, the marker is read before the world is built. Also
/// returns the params it couldn't use, unknown or unreadable, to warn about.
///
/// The sun is placed by `sun_elevation` (degrees above the horizon) and
/// `sun_azimuth` (degrees clockwise from north, -Z, seen from above: 90 is
/// east, +X); give one and the other keeps the default sun's.
pub fn environment(nodes: &[feather_assets::SceneNode]) -> (Environment, Vec<String>) {
    let mut env = Environment::default();
    let Some(spec) = nodes
        .iter()
        .filter_map(|n| n.prefab.as_ref())
        .find(|s| s.id == "environment")
    else {
        return (env, Vec::new());
    };
    let mut bad: Vec<String> = spec
        .params
        .as_object()
        .into_iter()
        .flat_map(|o| o.keys())
        .filter(|k| !ENVIRONMENT_PARAMS.contains(&k.as_str()))
        .cloned()
        .collect();
    let has = |key: &str| spec.params.get(key).is_some();
    let mut num = |key: &str, field: &mut f32| {
        if has(key) {
            match spec.f32(key) {
                Some(v) => *field = v,
                None => bad.push(key.to_string()),
            }
        }
    };
    let (mut elevation, mut azimuth) = sun_angles(env.sun_dir);
    num("sun_elevation", &mut elevation);
    num("sun_azimuth", &mut azimuth);
    num("sun_intensity", &mut env.sun_intensity);
    num("sky_intensity", &mut env.sky_intensity);
    num("sun_glow", &mut env.sun_glow);
    num("sun_disk", &mut env.sun_disk);
    num("fog_density", &mut env.fog_density);
    num("fog_height", &mut env.fog_height);
    num("fog_falloff", &mut env.fog_falloff);
    num("fog_sun", &mut env.fog_sun);
    num("exposure", &mut env.exposure);
    num("exposure_min", &mut env.exposure_min);
    num("exposure_max", &mut env.exposure_max);
    if has("sun_elevation") || has("sun_azimuth") {
        env.sun_dir = sun_travel(elevation, azimuth);
    }
    let colours = [
        ("sun_color", &mut env.sun_color),
        ("sky_zenith", &mut env.sky_zenith),
        ("sky_horizon", &mut env.sky_horizon),
        ("sky_ground", &mut env.sky_ground),
        ("sky_sun_color", &mut env.sky_sun_color),
    ];
    for (key, field) in colours {
        if has(key) {
            match spec.vec3(key) {
                Some(v) => *field = v,
                None => bad.push(key.to_string()),
            }
        }
    }
    if has("fog_color") {
        match spec.vec3("fog_color") {
            Some(v) => env.fog_color = Some(v),
            None => bad.push("fog_color".to_string()),
        }
    }
    (env, bad)
}

/// The elevation and azimuth (degrees, as `environment` takes them) of the
/// sun whose light travels along `travel`.
pub fn sun_angles(travel: Vec3) -> (f32, f32) {
    let to_sun = -travel.normalize();
    let elevation = to_sun.y.clamp(-1.0, 1.0).asin().to_degrees();
    let azimuth = to_sun.x.atan2(-to_sun.z).to_degrees();
    (elevation, azimuth)
}

/// The direction sunlight travels from a sun at `elevation` and `azimuth`
/// (degrees, as `environment` takes them).
pub fn sun_travel(elevation: f32, azimuth: f32) -> Vec3 {
    let (se, ce) = elevation.to_radians().sin_cos();
    let (sa, ca) = azimuth.to_radians().sin_cos();
    -Vec3::new(ce * sa, se, -ce * ca)
}

/// A session before it touches the GPU: the scenes it loaded, the simulated
/// world with its level, and the tables the renderer is built from.
/// `Session::new` is this plus the upload, so a test that builds this runs
/// the game's own scene → world path, without a device.
pub struct WorldBuild {
    pub world: World,
    pub schedule: Schedule,
    /// The player entity in `world`.
    pub player: Entity,
    pub meshes: Vec<MeshData>,
    pub baked: Vec<Option<feather_assets::bake::BakedMesh>>,
    pub materials: Vec<feather_assets::Material>,
    pub fits: Vec<Mat4>,
    pub mesh_spheres: Vec<(Vec3, f32)>,
    /// The level's atmosphere (§13), from its `environment` marker.
    pub environment: Environment,
    /// Its weather and sun's path (§13), from the same marker.
    pub level_weather: weather::LevelWeather,
    /// The first scene's baked sky visibility (§13), if there is one.
    pub sky: Option<feather_assets::bake::SkyVolume>,
    /// Time spent loading the scenes and bake, for the `[load]` line.
    pub t_scenes: std::time::Duration,
}

/// The orb demo's obstacle boxes, as (centre, size), resting on the ground.
/// `tools/gen_testscene.py`'s `LEVEL_BOXES` repeats them: the generators
/// still keep their scenes clear of where they were, so regenerating an old
/// scene gives the same level.
pub fn demo_boxes() -> [(Vec3, Vec3); 5] {
    [
        (Vec3::new(-3.0, 0.75, 2.0), Vec3::new(1.5, 1.5, 1.5)),
        (Vec3::new(3.5, 1.0, -1.0), Vec3::new(2.0, 2.0, 2.0)),
        (Vec3::new(0.0, 0.5, -4.5), Vec3::new(3.0, 1.0, 1.0)),
        (Vec3::new(-5.0, 1.5, -3.0), Vec3::new(1.0, 3.0, 1.0)),
        (Vec3::new(5.0, 0.5, 4.0), Vec3::new(1.0, 1.0, 4.0)),
    ]
    .map(|(offset, size)| (Vec3::new(offset.x, GROUND_Y + offset.y, offset.z), size))
}

/// The CPU half of `Session::new`: load the CLI `scenes` (empty means the orb
/// demo) and the bake from `bake_dir` (`None` ignores it), then build the
/// world, the player, the level and the schedule.
pub fn build_world(scenes: &[String], bake_dir: Option<&std::path::Path>) -> WorldBuild {
    // Scenes load *first*, because a `player_start` marker (§18) decides
    // where the player goes and the player is built below.
    let t_start = Instant::now();
    let (meshes, scene_nodes, sky_key) = load_scenes(scenes);
    // The bake (§17), read once: the renderer draws its LODs and the
    // colliders built below can use them too.
    let baked = bake_dir.map_or_else(Vec::new, |d| {
        feather_assets::bake::load_baked_meshes(d, &meshes)
    });
    // The first scene's sky visibility (§13), if it's been baked.
    let sky = bake_dir.zip(sky_key).and_then(|(d, key)| {
        let path =
            feather_assets::bake::baked_sky_path(&d.join(feather_assets::bake::SKY_DIR), key);
        match feather_assets::bake::SkyVolume::read(&path) {
            Ok(v) => {
                let [x, y, z] = v.dims;
                eprintln!(
                    "[sky] visibility: {x}x{y}x{z} cells of {:.2} m, {:.1} MB",
                    v.cell,
                    (v.texels.len() * 8) as f64 / 1_048_576.0
                );
                Some(v)
            }
            Err(_) => {
                eprintln!(
                    "[sky] no sky visibility baked for {}: ambient light is unoccluded; \
                     `feather-bake SCENE` bakes it",
                    scenes[0]
                );
                None
            }
        }
    });
    let t_scenes = t_start.elapsed();
    let (start_pos, start_yaw) = player_start(&scene_nodes);
    let (environment, mut bad_params) = environment(&scene_nodes);
    let (wind, bad_wind) = level_wind(&scene_nodes);
    bad_params.extend(bad_wind);
    let (level_weather, bad_weather) = level_weather(&scene_nodes);
    bad_params.extend(bad_weather);
    let (ground, bad_ground) = level_ground(&scene_nodes);
    bad_params.extend(bad_ground);
    for key in bad_params {
        eprintln!("[scene] environment: can't use param {key:?}; ignored");
    }
    let (surfaces, unknown_surfaces) = mesh_surfaces(&meshes);
    for name in unknown_surfaces {
        eprintln!(
            "[scene] unknown surface {name:?}; using {}",
            Surface::default().name()
        );
    }

    let mut world = World::new();
    world.insert_resource(FrameCount::default());
    world.insert_resource(LevelWind(wind));
    world.insert_resource(ColliderStats::default());
    let mut physics = Physics::new();
    // Feet on the ground. The player is a normal ECS entity: sim state in
    // `Player`, render-rate angles in `Look` (§15).
    let body = Player::new(&mut physics, start_pos);
    world.insert_resource(physics);
    world.insert_resource(InputState::default());
    let mut look = Look::new();
    if let Some(yaw) = start_yaw {
        look.yaw = yaw;
    }
    let player = world.spawn((body, look)).id();

    // The drifting orb demo only runs when no scene was given — a loaded level
    // is what you want to look at, and 1000 orbs would bury it.
    let half = (GRID as f32 - 1.0) / 2.0;
    let orb_count = if scenes.is_empty() {
        GRID * GRID * GRID
    } else {
        0
    };
    for i in 0..orb_count {
        let (x, y, z) = (i % GRID, (i / GRID) % GRID, i / (GRID * GRID));
        let pos = Vec3::new(x as f32 - half, y as f32 - half, z as f32 - half) * 1.6;
        let u = i as u32;
        let vel = Vec3::new(
            rand01(u * 3) - 0.5,
            rand01(u * 3 + 1) - 0.5,
            rand01(u * 3 + 2) - 0.5,
        ) * 1.5;
        let spin = (rand01(u * 7 + 11) - 0.5) * 3.0;
        // Procedural sphere or cube, with a random shared palette material.
        let mesh = if rand01(u * 17 + 5) < 0.5 {
            MESH_SPHERE
        } else {
            MESH_CUBE
        };
        let material = (rand01(u * 23 + 7) * PALETTE as f32) as u32 % PALETTE;
        world.spawn((
            Position(pos),
            PrevPosition(pos), // prev == curr on frame 0: first interp is a no-op
            Velocity(vel),
            Rotation(0.0),
            PrevRotation(0.0),
            Spin(spin),
            Scale(Vec3::splat(0.6)),
            Mesh(MeshId(mesh)),
            Material(material),
        ));
    }

    let mut schedule = Schedule::default();
    schedule.set_executor_kind(ExecutorKind::MultiThreaded);
    schedule.add_systems((integrate, tick, ropes));
    // §15's coupling: ECS -> rapier, the step, then rapier -> ECS. Chained so
    // the bracket order is explicit (they all touch `Physics`, so bevy_ecs
    // would serialise them regardless).
    schedule.add_systems((player_target_sys, physics_step_sys, player_readback_sys).chain());

    // Built-ins get the demo fit (centre + unit-scale); scene meshes keep their
    // authored transform, so identity.
    let fits: Vec<Mat4> = meshes
        .iter()
        .enumerate()
        .map(|(i, m)| {
            if (i as u32) < MESH_BUILTIN_COUNT {
                fit_transform(m)
            } else {
                Mat4::IDENTITY
            }
        })
        .collect();
    let mesh_spheres: Vec<(Vec3, f32)> = meshes.iter().map(local_sphere).collect();

    // Material table: shared palette first (indices 0..PALETTE), then each
    // mesh's own material (index PALETTE + mesh_id) — matches the ids that
    // the world build above assigned to entities.
    let mut materials: Vec<feather_assets::Material> = (0..PALETTE).map(palette_material).collect();
    materials.extend(meshes.iter().map(|m| m.material.clone()));

    // Two dedicated level materials, appended after everything else.
    let ground_mat = materials.len() as u32;
    materials.push(level_material([0.20, 0.21, 0.23], 0.95));
    let box_mat = materials.len() as u32;
    materials.push(level_material([0.45, 0.22, 0.14], 0.7));

    // Static level. The ground and the obstacle boxes are each rendered as a
    // scaled unit cube and given a matching cuboid collider (`spawn_static`).
    // Level pieces carry no Velocity/Spin, so `integrate` skips them and they
    // never wrap.
    // Unless the level brings its own ground.
    if ground {
        let ground = spawn_static(
            &mut world,
            Vec3::new(0.0, GROUND_Y - 0.5, 0.0),
            Vec3::new(80.0, 1.0, 80.0),
            MESH_LEVEL_CUBE,
            ground_mat,
        );
        // This ground is a flat slab: it casts nothing useful but would
        // rasterize the whole shadow map. It still *receives* shadows
        // (receiving is sampling the map, not being in it).
        world.entity_mut(ground).insert(NoShadowCast);
    }
    // The demo's obstacle boxes, like its orbs, only when no scene was given:
    // a loaded level is the level, and five brown boxes in its middle aren't
    // part of it.
    if scenes.is_empty() {
        for (center, size) in demo_boxes() {
            spawn_static(&mut world, center, size, MESH_LEVEL_CUBE, box_mat);
        }
    }
    // Scene geometry from the CLI glTF files (§18): one entity per node,
    // carrying that node's world transform, so nodes sharing a mesh draw as
    // instances. Each also gets a fixed trimesh collider so the level is
    // walkable. Material ids follow the same `PALETTE + mesh index` rule the
    // table below is built with.
    let registry = prefab_registry();
    let mut unknown: Vec<&str> = Vec::new();
    // Counted so the effect of `collide: false` is visible in the log rather
    // than having to be taken on trust.
    let colliders_before = world.resource::<Physics>().colliders.len();
    let mut spawned = 0usize;
    for node in &scene_nodes {
        // `player_start` and `environment` were consumed before the world
        // was built.
        if node
            .prefab
            .as_ref()
            .is_some_and(|s| s.id == "player_start" || s.id == "environment")
        {
            continue;
        }
        let mesh_idx = node.mesh;
        let args = SpawnArgs {
            transform: node.transform,
            mesh: mesh_idx.map(|i| MeshId(i as u32)),
            material: mesh_idx.map_or(0, |i| PALETTE + i as u32),
            mesh_data: mesh_idx.map(|i| &meshes[i]),
            baked: mesh_idx.and_then(|i| baked.get(i)).and_then(Option::as_ref),
            spec: node.prefab.as_ref(),
            surface: mesh_idx.map_or(Surface::default(), |i| surfaces[i]),
        };
        match node.prefab.as_ref() {
            Some(spec) => match registry.get(spec.id.as_str()) {
                Some(f) => f(&mut world, &args),
                None => {
                    // Unknown ids are authorable-ahead-of-time, not errors:
                    // fall back to static geometry and say so once.
                    if !unknown.contains(&spec.id.as_str()) {
                        eprintln!("unknown prefab {:?}; spawning as static geometry", spec.id);
                        unknown.push(spec.id.as_str());
                    }
                    spawn_static_prop(&mut world, &args);
                }
            },
            None => spawn_static_prop(&mut world, &args),
        }
        spawned += 1;
    }
    if spawned > 0 {
        let colliders = world.resource::<Physics>().colliders.len() - colliders_before;
        eprintln!("scene: {spawned} nodes spawned, {colliders} colliders built");
        let st = world.resource::<ColliderStats>();
        eprintln!(
            "[scene] colliders: {} mesh ({} from LODs), {} hull, {} box",
            st.mesh, st.lod, st.hull, st.boxes
        );
        eprintln!("[scene] surfaces: {}", st.surfaces_line());
    }

    // One step so the broad-phase BVH the character controller shape-casts
    // against contains the level before the first fixed tick.
    world.resource_mut::<Physics>().step();

    WorldBuild {
        world,
        schedule,
        player,
        meshes,
        baked,
        materials,
        fits,
        mesh_spheres,
        environment,
        level_weather,
        sky,
        t_scenes,
    }
}

/// Spawn one static level piece: rendered through the normal instanced path but
/// carrying no `Velocity`/`Spin`, so `integrate` skips it (it never moves or
/// wraps). `prev == curr` makes it a no-op through the interpolation path. Also
/// registers a matching fixed cuboid collider with `Physics` (it must already be
/// a resource).
pub fn spawn_static(
    world: &mut World,
    center: Vec3,
    size: Vec3,
    mesh: u32,
    material: u32,
) -> Entity {
    let collider = world.resource_mut::<Physics>().add_static_box(center, size);
    world
        .spawn((
            Position(center),
            PrevPosition(center),
            Rotation(0.0),
            PrevRotation(0.0),
            Scale(size),
            Mesh(MeshId(mesh)),
            Material(material),
            ColliderRef(collider),
        ))
        .id()
}

/// A plain untextured material for level geometry. Base color is treated as
/// already-linear (the values here are low, so no sRGB decode needed).
/// `MESH_ROPE`: an 8-sided unit tube in a dark, rough hemp, which every
/// rope's segments share.
pub fn rope_mesh() -> MeshData {
    MeshData {
        material: level_material([0.12, 0.085, 0.05], 0.92),
        ..MeshData::cylinder(8)
    }
}

pub fn level_material(base_linear: [f32; 3], roughness: f32) -> feather_assets::Material {
    feather_assets::Material {
        base_color: [base_linear[0], base_linear[1], base_linear[2], 1.0],
        metallic: 0.0,
        roughness,
        emissive: [0.0, 0.0, 0.0],
        normal_scale: 1.0,
        base_color_texture: None,
        normal_texture: None,
        metallic_roughness_texture: None,
        ..Default::default()
    }
}

/// A mesh's **local** (pre-fit) bounding sphere `(centre, radius)`. Extract maps
/// it through the model matrix — which already includes the fit — to get the
/// world sphere the frustum test uses (§8).
pub fn local_sphere(mesh: &MeshData) -> (Vec3, f32) {
    let (min, max) = mesh.bounds();
    let c = (min + max) * 0.5;
    (c, (max - c).length())
}

/// Center the mesh at the origin and scale its largest extent to ~1 unit, so an
/// arbitrarily-sized glTF drops into the demo grid at the same scale as the
/// sphere. Applied as the innermost factor of each instance's model matrix.
pub fn fit_transform(mesh: &MeshData) -> Mat4 {
    let (min, max) = mesh.bounds();
    let center = (min + max) * 0.5;
    let extent = (max - min).max_element().max(1e-4);
    Mat4::from_scale(Vec3::splat(1.0 / extent)) * Mat4::from_translation(-center)
}

/// A varied shared material for the procedural demo. Base color is generated in
/// sRGB then stored linear (the renderer shades in linear space).
pub fn palette_material(k: u32) -> feather_assets::Material {
    let srgb_to_linear = |c: f32| {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let r = 0.1 + 0.85 * rand01(k * 13 + 1);
    let g = 0.1 + 0.85 * rand01(k * 13 + 2);
    let b = 0.1 + 0.85 * rand01(k * 13 + 3);
    feather_assets::Material {
        base_color: [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), 1.0],
        metallic: if rand01(k * 13 + 4) > 0.7 { 1.0 } else { 0.0 },
        roughness: 0.3 + 0.6 * rand01(k * 13 + 5),
        emissive: [0.0, 0.0, 0.0],
        normal_scale: 1.0,
        base_color_texture: None,
        normal_texture: None,
        metallic_roughness_texture: None,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::node;

    /// The level's wind comes from its `environment` marker.
    #[test]
    fn the_environment_sets_the_wind() {
        let (w, bad) = level_wind(&[]);
        assert!(bad.is_empty());
        assert_eq!(
            (w.dir, w.speed),
            (rope::Wind::default().dir, rope::Wind::default().speed)
        );
        let marker = node(
            Some("environment"),
            serde_json::json!({ "wind_speed": 3.5, "wind_azimuth": 180.0 }),
        );
        let (w, bad) = level_wind(&[marker]);
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(w.speed, 3.5);
        // Azimuth 180 blows south, +Z.
        assert!((w.dir - Vec3::Z).length() < 1e-6, "{}", w.dir);
        let marker = node(
            Some("environment"),
            serde_json::json!({ "wind_speed": -1.0, "wind_azimuth": "north" }),
        );
        let (w, bad) = level_wind(std::slice::from_ref(&marker));
        assert_eq!(bad, ["wind_speed", "wind_azimuth"]);
        assert_eq!(w.speed, rope::Wind::default().speed);
        // environment() knows the keys, so it doesn't call them unknown.
        assert!(environment(&[marker]).1.is_empty());
    }

    /// The level's weather comes from its `environment` marker; bad values
    /// are reported and leave the defaults.
    #[test]
    fn the_environment_sets_the_weather() {
        assert_eq!(
            level_weather(&[]),
            (weather::LevelWeather::default(), Vec::new())
        );
        let marker = node(
            Some("environment"),
            serde_json::json!({ "weather": "Overcast", "time": 6.5, "time_speed": 60.0,
                                "latitude": 45.0, "day_of_year": 172.0 }),
        );
        let (w, bad) = level_weather(&[marker]);
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(
            w,
            weather::LevelWeather {
                path: weather::SolarPath {
                    latitude: 45.0,
                    day_of_year: 172.0
                },
                choice: 2,
                time: Some(6.5),
                speed: 60.0,
            }
        );
        let marker = node(
            Some("environment"),
            serde_json::json!({ "weather": "stormy", "time": 24.0, "time_speed": -1.0,
                                "latitude": 95.0, "day_of_year": 0.0 }),
        );
        let (w, bad) = level_weather(std::slice::from_ref(&marker));
        assert_eq!(
            bad,
            ["weather", "time", "time_speed", "latitude", "day_of_year"]
        );
        assert_eq!(w, weather::LevelWeather::default());
        // environment() knows the keys, so it doesn't call them unknown.
        assert!(environment(&[marker]).1.is_empty());
    }

    /// A level without an `environment` marker keeps the look every level had
    /// before levels could choose one, and so does one whose marker is empty.
    #[test]
    fn a_level_without_an_environment_keeps_the_default_look() {
        let prop = || node(Some("prop"), serde_json::json!({}));
        assert_eq!(environment(&[prop()]), (Environment::default(), Vec::new()));
        let empty = node(Some("environment"), serde_json::json!({}));
        assert_eq!(
            environment(&[prop(), empty]),
            (Environment::default(), Vec::new())
        );
    }

    #[test]
    fn an_environment_marker_sets_the_atmosphere() {
        let marker = node(
            Some("environment"),
            serde_json::json!({
                "sun_elevation": 90.0, "sun_azimuth": 0.0,
                "sun_color": [1.0, 0.5, 0.25], "sun_intensity": 3.0,
                "sky_zenith": [0.1, 0.2, 0.3], "sky_horizon": [0.4, 0.5, 0.6],
                "sky_ground": [0.7, 0.8, 0.9], "sky_sun_color": [0.9, 0.9, 0.8],
                "sky_intensity": 1.5, "sun_glow": 0.2, "sun_disk": 0.0,
                "fog_density": 0.02, "fog_height": -9.0, "fog_falloff": 0.08,
                "fog_color": [0.5, 0.52, 0.48], "fog_sun": 0.3, "exposure": 1.2,
                "exposure_min": 0.25, "exposure_max": 4.0,
            }),
        );
        let (env, bad) = environment(&[marker]);
        assert!(bad.is_empty(), "{bad:?}");
        assert!(
            (env.sun_dir - Vec3::NEG_Y).length() < 1e-6,
            "{:?}",
            env.sun_dir
        );
        let want = Environment {
            sun_dir: env.sun_dir,
            sun_color: Vec3::new(1.0, 0.5, 0.25),
            sun_intensity: 3.0,
            sky_zenith: Vec3::new(0.1, 0.2, 0.3),
            sky_horizon: Vec3::new(0.4, 0.5, 0.6),
            sky_ground: Vec3::new(0.7, 0.8, 0.9),
            sky_sun_color: Vec3::new(0.9, 0.9, 0.8),
            sky_intensity: 1.5,
            sun_glow: 0.2,
            sun_disk: 0.0,
            fog_density: 0.02,
            fog_height: -9.0,
            fog_falloff: 0.08,
            fog_color: Some(Vec3::new(0.5, 0.52, 0.48)),
            fog_sun: 0.3,
            exposure: 1.2,
            exposure_min: 0.25,
            exposure_max: 4.0,
        };
        assert_eq!(env, want);
    }

    /// Params left out keep their defaults; ones it doesn't know, or can't
    /// read, are reported (for a warning) and change nothing.
    #[test]
    fn a_partial_environment_keeps_the_rest() {
        let marker = node(
            Some("environment"),
            serde_json::json!({
                "fog_density": 0.05, "fog": 1.0, "sky_zenith": "grey", "fog_color": 0.5,
                "exposure_max": "bright",
            }),
        );
        let (env, mut bad) = environment(&[marker]);
        bad.sort();
        assert_eq!(bad, ["exposure_max", "fog", "fog_color", "sky_zenith"]);
        let want = Environment {
            fog_density: 0.05,
            ..Default::default()
        };
        assert_eq!(env, want);
    }

    /// Elevation is above the horizon and azimuth clockwise from north (-Z),
    /// seen from above; the direction is the way the light *travels*, away
    /// from the sun. Giving one angle keeps the default sun's other one.
    #[test]
    fn the_sun_is_placed_by_elevation_and_azimuth() {
        let close = |a: Vec3, b: Vec3| (a - b).length() < 1e-5;
        assert!(close(sun_travel(90.0, 123.0), Vec3::NEG_Y));
        assert!(close(sun_travel(0.0, 0.0), Vec3::Z)); // sun in the north
        assert!(close(sun_travel(0.0, 90.0), Vec3::NEG_X)); // sun in the east
        let d = Environment::default().sun_dir;
        let (el, az) = sun_angles(d);
        assert!(close(sun_travel(el, az), d));
        let marker = node(
            Some("environment"),
            serde_json::json!({ "sun_elevation": 10.0 }),
        );
        let (env, _) = environment(&[marker]);
        let (el2, az2) = sun_angles(env.sun_dir);
        assert!(
            (el2 - 10.0).abs() < 1e-3 && (az2 - az).abs() < 1e-3,
            "{el2} {az2}"
        );
    }

    /// `ground` (§18): the engine's slab unless the level says false; a
    /// value that isn't a bool is reported and keeps the slab.
    #[test]
    fn the_environment_says_whether_the_level_brings_its_ground() {
        assert_eq!(level_ground(&[]), (true, Vec::new()), "no marker: the slab");
        let marker = |params| feather_assets::SceneNode {
            mesh: None,
            transform: Mat4::IDENTITY,
            prefab: Some(feather_assets::PrefabSpec {
                id: "environment".into(),
                params,
            }),
        };
        let with = |p| level_ground(&[marker(p)]);
        assert_eq!(with(serde_json::json!({})), (true, Vec::new()));
        assert_eq!(
            with(serde_json::json!({ "ground": false })),
            (false, Vec::new())
        );
        assert_eq!(
            with(serde_json::json!({ "ground": true })),
            (true, Vec::new())
        );
        assert_eq!(
            with(serde_json::json!({ "ground": "no" })),
            (true, vec!["ground".to_string()])
        );
        assert!(ENVIRONMENT_PARAMS.contains(&"ground"));
    }

    #[test]
    fn player_start_marker_places_the_player() {
        let marker = feather_assets::SceneNode {
            mesh: None,
            transform: Mat4::from_translation(Vec3::new(4.0, -2.0, 7.0)),
            prefab: Some(feather_assets::PrefabSpec {
                id: "player_start".into(),
                params: serde_json::json!({ "yaw": 180.0 }),
            }),
        };
        let (pos, yaw) = player_start(std::slice::from_ref(&marker));
        assert_eq!(pos, Vec3::new(4.0, -2.0, 7.0));
        assert_eq!(yaw, Some(std::f32::consts::PI));

        // No marker: the hardcoded default, so unmarked scenes and the orb demo
        // behave exactly as before.
        let (pos, yaw) = player_start(&[]);
        assert_eq!(pos, Vec3::new(0.0, GROUND_Y, 8.0));
        assert_eq!(yaw, None);

        // A marker without a yaw keeps the default facing.
        let mut m = marker;
        m.prefab.as_mut().unwrap().params = serde_json::json!({});
        let (_, yaw) = player_start(std::slice::from_ref(&m));
        assert_eq!(yaw, None);
    }
}
