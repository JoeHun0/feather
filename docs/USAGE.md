# Feather — Running, Testing, Measuring

A practical guide: how to build the engine, run it, feed it scenes, test it,
and time it. For *why* anything works the way it does, see
[ARCHITECTURE.md](ARCHITECTURE.md). Its §26 records what is actually
implemented.

---

## 1. First-time setup

| Need | Why | Ubuntu |
|---|---|---|
| Rust via rustup | `rust-toolchain.toml` pins `stable` + rustfmt + clippy | [rustup.rs](https://rustup.rs) |
| Vulkan 1.3 driver | the renderer targets 1.3 (dynamic rendering, maintenance4) | `mesa-vulkan-drivers` |
| Validation layers | correctness signal on debug builds (optional since `025d3c6`) | `vulkan-validationlayers` |
| cmake, python3, C++ compiler, ninja | `shaderc` builds from source | `cmake python3 g++ ninja-build` |
| python3 (stdlib only) | the test-scene generator | — |
| ALSA headers | audio (kira → cpal) links ALSA; without them the build fails in `alsa-sys` | `libasound2-dev` |

Two gotchas:

- **An old `stable` toolchain is not upgraded automatically.** The dependencies
  need rustc ≥ 1.90. If the build stops with `rustc 1.xx is not supported by
  the following packages`, run `rustup update stable`.
- **To skip the shaderc source build**, point `SHADERC_LIB_DIR` at a Vulkan SDK's
  shaderc. A missing cmake/python/compiler otherwise shows up as an opaque
  build-script failure.

`scratch/` is gitignored, so a fresh clone has no scenes. Generate the default
one:

```bash
mkdir -p scratch && python3 tools/gen_testscene.py --check
```

---

## 2. Build and test

```bash
cargo build --workspace
cargo test --workspace
```

**Always pass `--workspace`.** Building or testing one crate alone
(`-p feather-gfx`, `-p feather-render`) fails with a raw-window-handle
`HandleError`. That is a feature-unification quirk, not a real bug. To run a
subset of tests, filter by name instead:

```bash
cargo test --workspace menu          # every test with "menu" in its name
cargo test --workspace shader_cluster
```

Running single tests, the harnesses the tests use, and writing a new one:
§5.

**Debug builds optimise dependencies** (`[profile.dev.package."*"]` in
`Cargo.toml`); the engine's own crates stay at full debug. Without it, rapier
and the image decoder ran ~10x slower in debug. It costs a slower *clean*
build (~41 s to ~62 s); incremental builds are unaffected.

Release build (the `[gpu]` timings measured the same as debug; shaders compile
optimised):

```bash
cargo build --workspace --release
```

---

## 3. Running

```bash
cargo run -- [FLAGS] [SCENE.gltf|.glb ...]
```

The app opens on the **main menu** with nothing loaded. **NEW GAME** loads the
scenes named on the command line; with no scene it runs the procedural
drifting-orb demo instead. The orb demo is pathological by design, so never
use it for timing. The window opens at **640×480** (logical pixels, so larger
on a HiDPI desktop) and is resizable; `--bench` uses a fixed 1920×1080.

| Flag | Effect |
|---|---|
| `SCENE` (bare path, repeatable) | glTF scene(s) NEW GAME loads. Each node becomes an entity; glTF `extras` become prefabs (§4). |
| `--msaa N` | Geometry-pass MSAA sample count: 1 / 2 / 4 / 8, clamped to what the device supports. Overrides `config/graphics.toml` for this run only (never saved). Also changeable from the main menu's OPTIONS > GRAPHICS (not mid-game), which *is* saved. |
| `--no-bake` | Ignore the bake: raw RGBA8 textures and raw meshes (no LODs), for A/B against it (§4). |
| `--no-lod` | Draw every mesh at LOD0 but keep the baked vertex order, to A/B the LOD win alone (§4). |
| `--no-bloom` | Turn bloom off for this run (never saved). Mainly for `--bench` A/B runs, which ignore the config files. |
| `--no-auto-exposure` | Fixed exposure for this run (never saved), likewise. |
| `--no-ao` | GTAO off for this run (never saved), for `--bench` A/B runs like `--no-bloom`. |
| `--no-taa` | TAA off for this run (never saved), for `--bench` A/B runs like `--no-bloom`: `--bench` runs with it on otherwise. |
| `--no-sky-occlusion` | Ignore the level's baked sky visibility, so the ambient light is unoccluded, as before it existed. For A/B runs; there's no menu option, as it's part of the look. |
| `--bench` | Scripted timing run: skips the menu, sweeps the camera 360°, prints a summary and exits. Ignores the config files. See §7. |

Typical runs:

```bash
cargo run -- scratch/testscene.gltf                  # walk the test level
cargo run -- --msaa 4 scratch/testscene.gltf         # with 4x MSAA
cargo run -- --bench scratch/lights120.gltf          # timing sweep
```

### Config files — `config/`

Settings persist in `config/` (gitignored, relative to the working directory,
so run from the repo root), one file per concern. The first normal run writes
each with defaults and comments; delete one to reset it. All are a flat
`key = value` subset of TOML parsed by hand (`app/src/config/`), with no
`[sections]`.

**`graphics.toml`** — saved when you change a setting in the menu or with the
F1 / F2 / F3 bindings. Only that key's value is rewritten; comments, other lines and
hand edits made while the game runs are kept. (It was `settings.toml` before
controls got their own file; an old one is renamed on first run.)

| Key | Values | Default |
|---|---|---|
| `display` | `"windowed"`, `"fullscreen"` (borderless, current monitor) | `"windowed"` |
| `fov` | vertical field of view, whole degrees 30–120 (60 ≈ 91° horizontal at 16:9) | `60` |
| `shadows` | `"high"`, `"medium"`, `"low"`, `"off"` | `"high"` |
| `msaa` | `1`, `2`, `4`, `8` (clamped to the device) | `1` |
| `fxaa` | `true`, `false` | `false` |
| `taa` | `true`, `false` (temporal AA: smooths edges, foliage and shimmer, slightly softens textures) | `true` |
| `bloom` | `true`, `false` | `true` |
| `auto_exposure` | `true`, `false` | `true` |
| `ambient_occlusion` | `true`, `false` (GTAO: contact shadows in corners and under objects) | `true` |

**`controls.toml`** — key bindings and mouse look. Saved when you change
something in OPTIONS > CONTROLS or OPTIONS > GAMEPLAY (same in-place editing as
`graphics.toml`), or edit it by hand; it's read at startup.

- `action = ["Key", ...]`: one or more keys, or `[]` to unbind. Actions:
  `forward`, `back`, `left`, `right`, `jump` (also fly up in noclip), `down`,
  `noclip`, `exposure_down`, `exposure_up`, `cycle_shadows`, `toggle_fxaa`, `toggle_taa`,
  `toggle_fullscreen`. A file from before an action existed still gets its
  default key.
  Defaults are in the Controls table below.
- Key names are physical positions named as on a US QWERTY keyboard, whatever
  your layout, case-insensitive: `A`–`Z`, `0`–`9`, `F1`–`F12`, `Space`, `Tab`,
  `Backspace`, `CapsLock`, `LeftShift`/`RightShift`, `LeftCtrl`/`RightCtrl`,
  `LeftAlt`/`RightAlt`, `Left`/`Right` (arrows), `Insert`, `Delete`, `Home`,
  `End`, `PageUp`, `PageDown`, `Numpad0`–`Numpad9`, and punctuation by word:
  `LeftBracket`, `RightBracket`, `Semicolon`, `Quote`, `Comma`, `Period`,
  `Slash`, `Backslash`, `Minus`, `Equal`, `Backquote`.
- **Escape, Up, Down and Enter are reserved for the menu** and can't be bound,
  so a bad file can't lock you out of it.
- A key on two actions does both (with a warning). Actions missing from the
  file keep their defaults.
- `sensitivity` (multiplier, default `1.0`, up to `20`) and `invert_y`
  (`true` / `false`). The GAMEPLAY menu steps sensitivity through presets and
  shows it as a percentage; any other value can still be typed here.
- Rebinding in the menu gives the action exactly one key, so a hand-written
  multi-key line (`["W", "I"]`) becomes a single key once you rebind it.

**`audio.toml`** — volumes in whole percent (0–100), saved when you change
them in OPTIONS > SOUND.

| Key | What it scales | Default |
|---|---|---|
| `master` | everything | `80` |
| `sfx` | footsteps, jumps, landings | `100` |
| `ambience` | the hum of visible lamps | `100` |

**All files:**

- **Precedence:** defaults < file < CLI flags. A flag applies to that run only.
- **Mistakes aren't fatal:** a bad line logs `[config] … line N: …; ignored`
  and that key (or action) keeps its default. An unreadable file (e.g. not
  UTF-8) is left as-is and the run uses defaults.
- **`--bench` ignores all of them** (neither reads nor creates them), and runs
  silent, so timings never depend on personal settings or an audio thread.

### Controls

Defaults, rebindable in OPTIONS > CONTROLS or `config/controls.toml` (Esc isn't).

| In game | |
|---|---|
| WASD + mouse | move / look |
| Space | jump |
| V | noclip (free flight) on/off |
| Left Ctrl | fly down (noclip) |
| `[` / `]` | exposure down / up (compensation, with auto-exposure on) |
| F1 | cycle shadow quality: OFF → LOW (4×512) → MEDIUM (4×1024) → HIGH (4×2048) |
| F2 | FXAA on/off |
| F3 | TAA on/off |
| F11 | windowed / fullscreen |
| Esc | pause |

**Menus:** mouse (hover + click) or arrows + Enter. **Esc** steps back one
screen, and resumes play only from the pause menu's top level.

- **Main menu:** NEW GAME / OPTIONS / QUIT
- **Pause menu:** CONTINUE / OPTIONS / MAIN MENU / EXIT
- **OPTIONS:** GRAPHICS / CONTROLS / SOUND / GAMEPLAY / BACK, and in game
  WEATHER.
- **OPTIONS > WEATHER** (in game only; the weather engine's temporary
  controls, ARCHITECTURE §13). All rows are live and never saved; a new game
  starts from the level's.
  - WEATHER: LEVEL, CLEAR, OVERCAST, FOGGY. LEVEL is the level's own look.
    Between two weathers the change takes 30 game minutes (`ARRIVING` while
    it blends); to and from LEVEL it's at once.
  - TIME: +1 h a press, as `HH MM` in local solar time. `LEVEL` means the
    level's own fixed sun. Under LEVEL a clock moves the sun only
    (`SUN ONLY`).
  - SPEED: PAUSED, 1X, 10X, 60X, 600X (game seconds per real second).
  - FOG DENSITY (WEATHER, OFF, THIN, LIGHT, MEDIUM, THICK, HEAVY) and FOG
    HEIGHT (WEATHER, EVEN, TALL, MEDIUM, LOW, GROUND: how fast it thins
    upwards, EVEN the same at every height) override the weather's.
- **OPTIONS > GRAPHICS:** DISPLAY (windowed / fullscreen), shadows, FXAA,
  TAA, BLOOM and AUTO EXPOSURE are live; MSAA can change only from
  the main menu, because the mesh and sky pipelines bake the sample count
  (in-game it reads `MENU ONLY`).
- **OPTIONS > CONTROLS:** one row per action with its keys (`JUMP  SPACE`).
  Enter or click a row, and it reads `PRESS A KEY`: the next key you press
  becomes that action's only key. Esc cancels; Up, Down and Enter can't be
  bound, so they're ignored while it waits. If another action had that key, it
  loses it and shows `NONE`. RESET KEYS restores the defaults (sensitivity and
  invert are kept). The list scrolls when the window is too short for it.
- **OPTIONS > GAMEPLAY:** SENSITIVITY cycles 25 → 50 → 75 → 100 → 125 → 150
  → 200 → 300 → 400 (percent) and wraps; INVERT Y toggles; FIELD OF VIEW
  cycles 50 → 60 → 70 → 80 → 90 (vertical degrees) and wraps. All three are
  live in game.
- **OPTIONS > SOUND:** MASTER VOLUME, SFX and AMBIENCE each cycle 0 → 25 → 50
  → 75 → 100 (percent) and wrap, live, saved to `config/audio.toml`.
- Changes save straight away: CONTROLS, SENSITIVITY and INVERT Y to
  `config/controls.toml`, FIELD OF VIEW (a view setting) to
  `config/graphics.toml`, volumes to `config/audio.toml`.

**What you hear:** footsteps every ~1.6 m walked, sounding like what you
stand on (concrete, grass, wood, carpet or snow, from the material's
`surface` tag, §4; untagged is concrete), a jump sound, a thump on landings
from more than a small step down (louder the harder), nothing in noclip, and
a 60 Hz hum from every *visible* lamp (a point light on geometry, e.g. the
testscene's two glowing orbs) that pans as you turn and fades out at the
light's radius. Footsteps, jump and landing are **recorded** (Kenney Impact
Sounds, CC0: footsteps for each surface, soft impacts) once
`python3 tools/fetch_assets.py kenney_impact_sounds` has run (0.76 MB); without
the pack, or for any file that won't decode, each falls back to a synthesised
sound, so audio never needs the download. A surface other than concrete
whose files are missing uses the concrete steps instead, with an `[audio]`
note naming the fetch command. The lamp hum is always synthesised.
Every clip is normalised to its event's level, so recorded and synthesised
sounds play equally loud. With no audio device the game logs
`[audio] unavailable` and runs silent.

Defaults: shadows HIGH, FXAA off, TAA on, bloom on, auto-exposure on, MSAA 1×.

---

## 4. Test scenes — `tools/gen_testscene.py`

Stdlib-only Python that writes a self-contained `.gltf` (buffer embedded as a
data URI). The level is laid out in sectors, each aimed at one system:

| Zone | Exercises |
|---|---|
| `shadow` | pillars running past the old single-map boundary (CSM), and a 70 m tower at (30, 30) whose shadow ends beside spawn (caster pancaking) |
| `traversal` | stairs and ramps around the 0.4 autostep and 45° slope limit |
| `aa` | thin poles, a lattice, a picket fence at graded distances |
| `field` | hundreds of nodes over a few meshes (dedup, instancing, culling) |
| `pbr` | metallic × roughness sphere grid (IBL, tonemap reference) |
| `normals` | three pairs west of spawn (x −19…−5, z 8): a rotated, non-uniformly scaled node beside the same shape baked into its vertices. Each pair must shade identically; the last pair is mirrored |
| `surfaces` | footsteps (§3): a strip of five 5 cm pads east of spawn (x 3.5…27.5, z ≈ 9.75; spawn faces west, so turn around), west to east concrete (grey), grass (green), wood (brown), carpet (red), snow (white), 1 m of plain ground between them. Walk along it and each pad's steps should sound different |

| Option | Default | Meaning |
|---|---|---|
| `-o, --out PATH` | `scratch/testscene.gltf` | output file |
| `--zones LIST` | `all` | comma list of the zones above |
| `--density low\|med\|high` | `med` | field size: 120 / 320 / 800 nodes |
| `--seed N` | `7` | RNG seed; same seed + options → byte-identical file |
| `--lights N` | `0` | scatter N extra point lights as geometry-free markers |
| `--light-radius R` | `14` | radius of the scattered lights. 14 ≈ 8.5 lights per pixel (heavy overlap, clustering's worst case); 4 ≈ 1 per pixel. Must be > 0; authored lights are unaffected |
| `--textures` | off | procedural base-color / normal / MR textures |
| `--many-textures` | off | also gives each of the 30 PBR spheres its own base-colour and MR texture: 66 unique images, past the engine's old 64-slot cap (implies `--textures`) |
| `--no-extras` | off | omit all prefab `extras` (incompatible with `--lights`) |
| `--check` | off | validate the output against the engine's constraints |

**Always pass `--check` when you change the generator.** It enforces the
constraints that are otherwise silent: bounds, ground contact, no overlap with
the app's built-in level boxes.

Recipes:

```bash
# The light benchmark set (only the light count differs; same seed = same level)
for n in 0 60 120; do python3 tools/gen_testscene.py --lights $n -o scratch/lights$n.gltf; done

# Many small lights: the case clustering is built for
python3 tools/gen_testscene.py --lights 120 --light-radius 4 -o scratch/lights120_r4.gltf --check

# Culling / instancing load
python3 tools/gen_testscene.py --density high -o scratch/dense.gltf --check

# One zone in isolation, textured
python3 tools/gen_testscene.py --zones pbr --textures -o scratch/pbr.gltf --check
```

### Authoring your own scene (glTF `extras` → prefabs)

Any glTF works: triangles only, `POSITION` required, and normals are computed
if absent. A node's `extras` can name a prefab:

```json
"extras": { "prefab": "point_light", "params": { "radius": 6.0 } }
```

| Prefab | Params (defaults) | Notes |
|---|---|---|
| `player_start` | `yaw` degrees (keep the default look direction) | where NEW GAME spawns the player; first one wins |
| `prop` | `collide` (true), `shadow` (true), `collider` (`auto`) | a mesh with optional collider / shadow casting. `collider`: `auto` (exact mesh up to 2048 triangles; above that, the finest baked LOD under 2048 triangles within 5 cm, else a convex hull), `mesh` (always the full mesh), `hull`, `box`, `none`. Bake a scene so its detailed props get LOD collision instead of hulls, which fill every hollow |
| `environment` | `sun_elevation` 63° and `sun_azimuth` 127° (degrees above the horizon, and clockwise from north, -Z, seen from above), `sun_color` [1,1,1], `sun_intensity` 8, `sky_zenith` [0.10,0.22,0.55], `sky_horizon` [0.55,0.65,0.85], `sky_ground` [0.17,0.18,0.19], `sky_sun_color` [1,0.95,0.85] (the tint of the sun's glow and disk), `sky_intensity` 1 (also the ambient light), `sun_glow` 0.6, `sun_disk` 60 (0 hides the sun), `fog_density` 0.010 per metre (at `fog_height` 0; OPTIONS > WEATHER can override it and `fog_falloff` in game), `fog_falloff` 0 (per metre; above 0 the fog thins with height, by e every 1/falloff metres, and the sky background fogs through it too), `fog_color` (unset: the sky's colour), `fog_sun` 0 (a glow towards the sun in the fog), `exposure` 1 (with auto-exposure on, compensation: a multiplier on the metered exposure; off, the fixed exposure), `exposure_min` 0.125 and `exposure_max` 8 (the range auto-exposure may choose from), `wind_speed` 2 m/s and `wind_azimuth` 90° (the direction it blows towards, clockwise from north like the sun's; ropes sway in it), `weather` `level` (or `clear`, `overcast`, `foggy`: a new game's weather, any case), `time` unset (the hour a new game starts at, 0–24 local solar time; unset, LEVEL keeps the level's own fixed sun and a weather starts where the sun is nearest it), `time_speed` 10 (game seconds per real second), `latitude` 51.3 and `day_of_year` 120 (the sun's path: sunrise 04:44, sunset 19:16) | the level's atmosphere, on a bare marker; first one wins. Colours are linear. Left-out params keep the default look; unknown or unreadable ones print `[scene] environment: can't use param …` and are ignored |
| `hanging` | `length` 1.5 m, `segments` 12, `radius` 0.008 m, `wind` 1 (how strongly it feels the level's wind), `shadow` (true), `light` (none: an object of `point_light`'s params, same defaults, plus `at` [0,0,0], the light's position in the item's own space) | something on a rope (§15 in ARCHITECTURE.md): the node is the item at rest, hung by its mesh's origin, and the rope is tied `length` straight above it. It sways in the level's wind; neither rope nor item collides. Bad or unknown params print `[scene] hanging: can't use param …` |
| `point_light` | `color` [1,1,1], `intensity` 12, `radius` 10, `source_radius` 0.1 | on a bare marker or on geometry (a lamp that also renders). `source_radius` is the emitter's physical size, clamped to [0, radius]: it sets the size of the highlight on shiny surfaces. Match it to the lamp's geometry; 0 is a true point, which makes a pinprick-bright highlight on smooth metal |

An unknown prefab name falls back to static geometry. Up to 128 point lights
can be *visible* at once; lights past that are dropped for the frame with a
`[light]` warning.

**Cutouts** (grass, chain-link, leaves) come from the material, as glTF
defines them: `"alphaMode": "MASK"` cuts out wherever the base colour's
alpha (texture × factor) is below `alphaCutoff` (default 0.5), in the
image and in shadows alike. Add `"doubleSided": true` for cards seen from
both sides; their back faces are then lit like their fronts. The alpha must
be in the base colour texture (a PNG with alpha): glTF has no separate
opacity map. `"alphaMode": "BLEND"` still draws opaque (there's no
transparency yet); the load prints `[mesh] N BLEND materials drawn opaque`.

**Footstep surfaces** belong to *materials*, not nodes: a material's `extras`
names one, and everything drawn with it (every placement, every face) steps
like it:

```json
"materials": [ { "name": "planks", "extras": { "surface": "wood" } } ]
```

Surfaces: `concrete` (the default, also for untagged materials and the
built-in ground), `grass`, `wood`, `carpet`, `snow`, in any case. An unknown
name logs `[scene] unknown surface "…"; using concrete` once. Only geometry
that collides matters: the surface is read from the collider under your feet.

### A scene from real CC0 models — Kenney Nature Kit

```bash
python3 tools/fetch_assets.py                 # once: ~10.5 MB, SHA-256 pinned
python3 tools/gen_naturescene.py --check      # writes scratch/nature.glb
cargo run -- scratch/nature.glb
```

`fetch_assets.py` downloads the pinned archive, **refuses it if the SHA-256
doesn't match**, and extracts only the `.glb` models and the licence into
`scratch/assets/kenney_nature_kit/` (gitignored, like all of `scratch/`). Run
it again and it's a no-op, unless the tool now keeps more of a pack than your
copy has (it checks for a file per kept prefix): then it fetches that pack
again and says why. `--zip PATH` uses an archive you already have (still
hash-checked); `--force` re-extracts.

The kit is **CC0** (Creative Commons Zero, see its `License.txt`): free for any
use, with credit to Kenney (kenney.nl) appreciated but not required.

The same command also fetches **Kenney Impact Sounds** (0.76 MB, CC0) into
`scratch/assets/kenney_impact_sounds/`: only its footsteps (carpet, concrete,
grass, snow, wood) and soft impacts, 35 sounds, plus the licence. Fetch just
that pack with `python3 tools/fetch_assets.py kenney_impact_sounds`.

`gen_naturescene.py` builds a grass meadow, forest, rocks, a cliff ridge on the
east edge, a camp with a fire light (you spawn at its edge, facing the fire), and a
stone path lit by lamps. Options: `--density low|med|high`, `--seed N`,
`-o PATH`, `--check` (the same engine constraints as the procedural
generator). Grass tufts, flowers and the stone path have no collider; the
ground tiles collide (as flat boxes) because they are the meadow you walk on
and carry its grass surface, and none of those cast a shadow. Trees, rocks,
the tent and fences do both. Footsteps: grass on the meadow and on the grass
tops of rocks and cliffs, wood on logs and stumps (jump on), concrete on bare
rock and the campfire stones. The path sounds like grass, since its stones
don't collide.

### A detail stress scene — Poly Haven scans

```bash
python3 tools/fetch_assets.py                          # also fetches ~16 MB of Poly Haven models
python3 tools/gen_detailscene.py --density med --check # writes scratch/detail.glb
cargo run --release -- scratch/detail.glb
```

Four CC0 photoscans from [Poly Haven](https://polyhaven.com) (marble bust,
brass lantern, mossy rock set, grass clumps), each with 2048² PBR textures,
placed many times over: `low` / `med` / `high` put about 1.8M / 6M / 14.5M
triangles in the level. Each file is pinned by the MD5 Poly Haven publishes.
Unlike the nature scene, materials pass through as authored (textures,
`alphaMode`, `doubleSided`), because they are what's being tested. Debug
loads it in ~2 s and release in under half a second; each image is decoded
and uploaded once however many materials share it.

### An industrial compound — the STALKER look

```bash
python3 tools/fetch_assets.py                          # also fetches ~101 MB of Poly Haven textures and props, and a 9 MB ambientCG fence
python3 tools/gen_zonescene.py --check                 # writes scratch/zone.glb
cargo run --release -p feather-bake -- scratch/zone.glb   # optional: LODs + BC7, loads in ~0.4 s
cargo run -- scratch/zone.glb
```

An abandoned industrial compound under an overcast sky: a yard walled by
Soviet precast-panel fence, a brick and rusty-iron hangar with holes in the
roof, a two-storey office block you can walk into and climb, broken asphalt
and mud, road barriers, barrels, tyres, pipework, a covered car and lines of
electricity poles. Kerosene lanterns hang on ropes from the hangar's roof
beams, swaying in the draught, and are its only lamps. You spawn on the road outside the gate,
facing it.

- **Assets:** 8 tiling Poly Haven textures at 2K (precast concrete, factory
  brick, rusty corrugated iron, rusty metal, damaged concrete floor, damaged
  road, muddy leaves, worn plaster) and 9 props at 1K. All are CC0, and each
  file is pinned by Poly Haven's MD5.
- **Surfaces** tile at their real-world size.
- **Material AO:** each Poly Haven material names its ARM map as its
  `occlusionTexture` as well, since Poly Haven packs AO in the map's red.
  That covers the tiling textures here and the props through
  `gen_detailscene.py`'s import. `--check` fails if one is missing.
- **Footsteps:** grass on the mud, concrete everywhere else.
- **Atmosphere:** the level's `environment` marker (see the prefab table)
  makes it grey and hazy. Its values are starting points; tune them in
  `ENVIRONMENT` at the top of the script and regenerate.
- **The app's slab:** the grey ground slab the app puts under every level is
  under the mud. (The orb demo's brown boxes don't appear in loaded scenes.)
- **Options:** `--seed N` (props, missing fence panels, roof holes, and the
  trees' and bushes' shapes and places), `-o PATH`, `--check`. On top of the
  shared checks, the zone's `--check` reads the written vertices: no leaf or
  branch inside a solid (crowns may top the fence and poles), no bush
  through a trunk, no trunk in a solid, and every tree and bush placed.
  Seeds 1–200 all pass.
- **Cutouts:** grass tufts on the mud, and a rusty chain-link enclosure round
  the compressor (ambientCG's Fence006, CC0, SHA-256 pinned). The first run
  merges the fence's colour and opacity maps into
  `scratch/assets/ambientcg_Fence006/merged_rgba.png`; later runs reuse it.
- **Interiors** get the sky's light only through their openings once the
  zone is baked: the hangar is dim and lit mostly by its door, roof holes and
  lamp, and the office is darker still. Unbaked, they're as bright as the yard.
- **Trees and bushes:** birch-like trees and bushes the script draws and
  builds, thick outside the fence and scattered in the yard (41 trees and 65
  bushes at the default seed). Leaf cards are single-sided MASK with normals
  pointing out of the crown (ARCHITECTURE.md §5). Trunks block you; crowns
  and bushes don't. `--check` fails if a leaf goes double-sided, a crown or
  bush collides, a trunk doesn't, or a trunk stands on the road or near the
  spawn.
- **What to look for:**
  - crowns lit on the sun side and darker away from it, not flat;
  - dappled cutout shadows under the trees and on the fence;
  - no dark or light fringes round the leaves;
  - no leaves popping or crowns thinning as you walk away (LODs);
  - shadows under trees 9 m or more away are slightly lighter than under
    near ones: their casters use a quarter of the leaf cards (the foliage
    LOD floor);
  - trunks stop you, bushes don't.

### Baking — `feather-bake`

```bash
cargo run --release -p feather-bake -- scratch/detail.glb   # any scenes; incremental
cargo run --release -- scratch/detail.glb                    # now uses the bake
```

The bake writes to `scratch/bake/` (`--out DIR` to change it). Anything not
baked still loads raw, so baking is optional. Re-running skips everything
already baked, since files are keyed on content; delete `scratch/bake/` to
start over. Use `--release`: the BC7 encoder is slow in debug.

- **Textures** → `tex/`: every texture a scene uses becomes a **BC7** mip
  chain. That's 4× less GPU memory (the detail scene: 64 MB instead of 256 MB)
  and no load-time mip building.
- **Meshes** → `mesh/`: vertices reordered for the GPU's caches, plus a
  **LOD chain**. Each level has about half the triangles of the one before, and
  small disconnected parts such as grass blades get dropped. Meshes under 256
  triangles keep full detail only. The bake prints each mesh's chain
  (`tris 30830 > 15415 > ...` and each level's error).
- **At runtime** the engine picks, per instance and per view, the coarsest LOD
  whose error stays under **1 pixel** on screen, or under **one shadow texel**
  for each shadow cascade. On `detail_high` that cut the triangles drawn by
  ~92% and the frame from 1.83 to 0.50 ms (§17 in ARCHITECTURE.md).
  `--no-lod` turns it off for comparison.
- **Sky visibility** → `sky/`: one volume per scene, holding how much of the
  sky each point of the level sees, on a grid of 0.5 m cells. It's traced
  against the mesh LODs, so it comes after them, and it's fast: the zone
  takes 0.7 s. The bake prints the grid, the cells found inside geometry and
  the mean sky seen. The engine uses it to darken the ambient light where
  little sky reaches (inside buildings, under roofs). At load,
  `[sky] visibility: 142x21x131 cells …` says it was found, and `[sky] no
  sky visibility baked …` says to bake. `--no-sky-occlusion` turns it off.

The `[mesh]` lines at load say what you got, e.g.
`30 meshes (27 baked, 135 LODs): 2.7 MB vertices, 3.0 MB indices` and
`12 textures uploaded ... (12 baked BC7, 0 raw; 0 images decoded), 64.0 MB`.
A baked texture is found by its source file's bytes and never decoded, which
is most of why a baked scene loads fast (`[load]` below). After an engine
update that changes the bake format, the first load reports raw textures
until you re-run the bake; stale files in `scratch/bake/` are simply ignored.

**What to look for** when judging LODs: distant objects should look the same
with `--no-lod` as without, with no visible popping as you walk towards them,
and distant shadows should look unchanged.

**What to look for** in material AO (the zone): darker brick mortar,
corrugation grooves and rust flakes, and seams on the barrels and the
compressor, with no black blotches. It stays with GTAO off, which only removes
the contact shadowing.

**What to look for** in OPTIONS > WEATHER (in game).

The fog rows change the view at once:
- OFF clears the air;
- HEAVY buries the hangar;
- GROUND keeps the haze low, with the walls clear above it.

For the weather engine, pick CLEAR and set SPEED to 600X: a day passes in
2.4 minutes.
- **Dawn** (about 04:45 on the default day) warms from the east.
- **Noon** is the old default look.
- **Dusk** sets in the west, with an orange glow that lingers over the
  horizon after sunset. It follows the sun, not the moon.
- **Night:** moonlit and dark. The moon's shadows come from the side
  opposite the sun, the lanterns carry the hangar, and the moon's disk is a
  faint soft glow.
- **Two points to judge:**
  - where the light switches (sunrise and sunset), nothing should jump;
  - between two weathers, OVERCAST should arrive over 30 game minutes with
    no pop.

Known gaps:
- a bright interior haze by day (the fog ignores occlusion, as it always
  has);
- no stars or clouds.

All the keys are first seeds, to be tuned by eye.

**What to look for** in the hangar's lanterns (the zone; one hangs near
the door, visible from outside):
- They sway gently and differently, not in lockstep, and the rope bends a
  little in gusts rather than staying a rigid rod. Their shadows sway too.
- The rope stays joined along its length, with no gaps at its bends, and
  meets the beam and the lantern's handle.
- With TAA on, the thin rope reads fainter than with it off (it's about a
  pixel wide from the floor), and nothing trails the moving lantern.
- Their light is the hangar's: warm pools under each, glowing glass, and the
  light shifting slightly with the sway. The middle of the floor should no
  longer be the darkest part.

**What to look for** when judging TAA (OPTIONS > GRAPHICS > TAA, or F3,
live):
- Edges, fence wire, pole crossarms and the sun disk: smooth, and steady as
  you turn slowly. Leaves and grass should shimmer much less than with it
  off.
- Textures read a little softer than without it (it averages samples across
  each pixel); a sharpen isn't built yet.
- No trails behind trunks, poles or the fence as you strafe past them, and
  none when you stop. The orb demo's moving orbs do trail a little: only the
  camera's motion is followed.
- Nothing lags or smears after a resize or a level load (the history
  restarts).

**What to look for** when judging GTAO (OPTIONS > GRAPHICS > AMBIENT
OCCLUSION, live):
- A soft darkening along wall-floor and wall-ceiling creases, in the corners
  of door and window openings, and under barrels, tyres and the car.
- No dark or light rims round objects against the sky or a far background
  (poles, fence posts, grass), no 4×4 or 8×8 grain on flat walls, and no
  stair-steps in the creases. Thin crevices, such as under the car, read a
  little softer than they did at full resolution.
- Flat surfaces shouldn't darken towards the screen's edges as you turn.

**What to look for** when judging sky occlusion (A/B with
`--no-sky-occlusion`):
- Open ground should look the same either way.
- Inside, walls facing an opening should be brighter than walls facing away
  from it, and surfaces should brighten towards the door.
- There should be no bright band along the tops of the hangar's walls under
  the roof, or at the base of walls. Those would be light leaking through
  thin geometry.
- The exposure adapts, so a dark room brightens again after a second.

---

## 5. Tests

`cargo test --workspace` runs 259 tests, in a few seconds once built (the
ones that step rapier, the ropes and the mixer take most of it).
All of them are CPU-side: none needs a GPU, a window, a sound card or
anything in `scratch/`, so they pass on a fresh clone.

### Running them

```bash
cargo test --workspace                     # everything
cargo test --workspace surface             # every test with "surface" in its name
cargo test --workspace tests::a_scene_steps_like_its_materials -- --exact   # just that one
cargo test --workspace a_scene_steps -- --nocapture                         # and show what it printed
```

- A filter matches the full test name, module included, as the output
  prints it (`test surface::tests::surface_names_round_trip ... ok`).
  `--exact` needs that full name. And always `--workspace` (§2).
- Each crate is its own test binary with its own `test result:` line. To
  total them:
  ```bash
  cargo test --workspace 2>&1 | grep '^test result' | awk '{p+=$4; f+=$6} END {print p, "passed,", f, "failed"}'
  ```
- `--nocapture` shows the engine's own log lines too. In the end-to-end
  tests those are the real load's, such as
  `[scene] unknown surface "lava"; using concrete`.

### How tests reach the engine without a GPU

The part of each system that decides what happens sits behind a seam that
needs no device (ARCHITECTURE.md §1), and the tests use those seams
directly. Reuse them when you add one:

| Seam | What a test can do with it | Helpers |
|---|---|---|
| The controller is plain functions (`player_target`, `player_readback`) that the ECS systems wrap | step the real controller against real rapier colliders | `setup(boxes, start)` builds `Physics` with the ground and boxes; `step(…)` runs one tick, `run(…)` several (`game/src/testing.rs`) |
| Prefabs spawn from a `World` and `SpawnArgs` | spawn one node the way a scene does, then inspect its entity and collider | `prefab_world()`, `spawn_one(…)` (`game/src/prefab.rs`), `node(…)` (`game/src/testing.rs`) |
| `Audio<B: Backend>` is generic over kira's backend | play through the real mixer into a buffer and measure what came out | the `Capture` backend, `audio(…)`, `peak(…)` (`app/src/audio.rs`) |
| `Menu` needs neither the renderer nor the event loop | drive every screen, row and rebind; hit-test the layout at any window size | the menu tests in `app/src/menu.rs` |
| `build_world` (`game/src/level.rs`) is `Session::new` without the GPU upload | load a scene file into the game's own world and run the real schedule | `write_pad_level(…)`, `walk_pad_level(…)` (`app/src/main.rs`); next section |
| glTF can be written at test time | use real files without committing binaries | `write_fixture(…)` (`assets/src/lib.rs`), `write_pad_level(…)` |

Temp directories come from `config::test_dir(name)` in `app` and
`fixture_dir(name)` in `assets`. Tests run in parallel, so give each its own
name, and remove the directory at the end.

### The end-to-end tests

`build_world(scenes, bake_dir)` is the CPU half of `Session::new`:
- it loads the scenes, and the bake if given a directory;
- it resolves each mesh's footstep surface;
- it builds the world, the player, the built-in level, the scene's nodes and
  colliders, and the schedule.

`Session::new` is that plus the GPU upload, so a test that calls it runs
exactly what NEW GAME runs.

The two tests in `app/src/main.rs` work like this:

1. `write_pad_level(dir, pads)` writes a real `.gltf` + `.bin`: a row of flat
   pads (4 × 0.05 × 3.5 m) along +X at z = 9.75, one material per pad with an
   optional `extras.surface`, and a `player_start` marker west of the row,
   facing +X.
2. `walk_pad_level(path)` builds it with `build_world` and checks that the
   player starts at the marker. It then sets `InputState.wish` to +X and runs
   the real schedule one fixed tick at a time. After each tick it feeds the
   `Player` component to a `StepTracker`, as the app does each frame. It
   returns each step's x and surface, and the load's colliders per surface.
3. `check_steps` judges the steps:
   - well inside a pad, a step must have the pad's surface;
   - well clear of the pads, concrete;
   - every pad must get at least one step, so an empty walk can't pass;
   - within 0.4 m of a pad edge (the capsule's radius plus a margin) either
     answer is right, so those steps aren't judged.

The tests themselves:

- **`a_scene_steps_like_its_materials`** tags the pads grass, wood, `Carpet`
  (names are case-insensitive), snow, `lava` (not a surface) and nothing.
  That covers the whole footstep chain: material extras → the loader →
  per-mesh surfaces → spawned colliders → the controller's probe →
  `StepTracker`. From the step event on, `each_surface_plays_its_own_steps`
  (`app/src/audio.rs`) takes over.
- **`an_untagged_scene_steps_on_concrete`** walks the same level with no tags,
  and must hear only concrete. It's the counter-case: what the first test
  hears has to come from the tags.

To test something else end to end, build a scene and drive the schedule the
same way:

```rust
let mut b = build_world(&[path], None);                 // what NEW GAME builds
b.world.resource_mut::<InputState>().wish = Vec3::X;    // also jump, vertical, noclip
for _ in 0..120 {
    b.schedule.run(&mut b.world);                       // one fixed tick
}
let p = b.world.get::<Player>(b.player).unwrap();       // read back what you need
```

- **What runs:** the game's real schedule, multi-threaded as in the game:
  `integrate`, `tick`, and the §15 bracket (ECS → rapier, the physics step,
  rapier → ECS).
- **What `build_world` leaves out:** the GPU upload and everything `App`
  does per frame (input, menus, audio playback). The test stands in for
  those, as `walk_pad_level` does for `StepTracker`.

### Writing a test

- **CPU only, and nothing from `scratch/`.** A test must pass on a fresh
  clone, so write fixtures to a temp directory (above).
- **Test the real path, not a copy.** If a test would have to duplicate
  engine code, split a seam out of the engine instead; that's why
  `build_world` exists.
- **Make sure it can fail.** Before trusting a new test, break the code it
  guards for a moment and watch it fail: a negative control.
  - The end-to-end test was checked that way: a no-op `tag_surface`, and a
    probe that never updates `Player.surface`, each fail it.
  - When the expected result could also come from a bug (every step
    concrete, say), write the counter-case as a test of its own.
- **Rule out empty passes.** Assert that the thing happened at all (every
  pad got a step), not only that nothing went wrong.
- **Take tolerances from the physics** (the 0.4 m edge margin is the
  capsule's 0.35 m radius plus a little), not from whatever the first run
  printed.
- **One-off checks are temporary.** A check against real assets or timings
  (walking the `scratch/` scenes, the real sound pack through the mixer, a
  probe's cost) can be a temporary `#[test]` run with `--nocapture`. Delete
  it before committing, and write what it found into ARCHITECTURE.md.
- **Keep this section current:** add the test to the table below, and update
  the count at the top, in the same commit.

### What they guard

| Area | Crate | What the tests pin down |
|---|---|---|
| Character controller | game | settling, walking speed, jumps and head bumps, no air-jump, autostep lip vs wall, sliding along box faces, noclip; a low overhang (a ball hung below head height, hull or trimesh) holds a head-on push with at most 3 slide passes a tick (rapier's loop used to run all 20), and a glancing push slides past it; reaching it mid-tick (starts spread over one tick's travel, head-on and glancing) takes at most 3 passes on every tick; a wall leaning 9° wedges the slide at most once, held in an inside corner and sliding off its open end; hopping at a tree still climbs its branch stubs (two triangles cut from a real Kenney tree), because the overhang clip leaves jumps alone |
| Audio | app | a missing sound pack falls back to 15 synthesised clips with one note (naming the fetch command), and every surface steps like concrete; a file that doesn't decode falls back for that sound only, with a note; a surface whose files are missing or broken uses the concrete steps, one note each; surface names round-trip (any case; unknown is `None`, index 0 is concrete); every clip is normalised to its event's peak; synthesised sounds are finite, within ±1, end at zero (no click) and are deterministic, and the hum loops without a seam; footsteps per stride, each carrying the surface underfoot, silent in noclip, jump vs walking off a ledge, the landing threshold, teleports ignored; percent → dB; **through the real kira mixer** (a capture backend, no sound card): a landing is audible, master and SFX at 50% halve its peak and 0% silences it, a lamp to the listener's right is louder in the right ear (and vice versa), and one past its radius is silent, and each surface's step plays that surface's clip; `audio.toml` parse/warn/save; the SOUND rows step and save |
| Collision proxies | game | the `collider` param picks mesh/hull/box/none, `auto` switches to a hull past 2048 triangles without a bake, `collide: false` still wins; a dense 30 cm prop is walkable as a hull; a flat hull still holds the player; with a bake, `auto` picks the finest LOD within the triangle budget and 5 cm (scaled by the node), and a concave dish keeps the player *in* it as a LOD trimesh but on its rim as a hull; `mesh` stays exact |
| Surfaces underfoot | game | a material's surface tag reaches its collider (untagged and unknown are concrete, each unknown name reported once); walking over a tagged pad reads concrete → pad → concrete, and over an untagged one only concrete; on a ledge held by the rim (centre over the drop, where a ray finds nothing) the probe still finds the ledge |
| End to end | app | a glTF level written by the test (flat pads, each material tagged grass, wood, `Carpet`, snow, an unknown name or nothing) goes through the game's own world build (`build_world`, the CPU half of `Session::new`) and the real schedule: the player starts at the scene's marker, the load counts colliders per surface, and walking the row, every step sounds like the pad under it (unknown and untagged: concrete), and every pad gets one; the same level untagged steps only on concrete |
| Menus | app | row layout and hit-testing at several window sizes, including scrolling a screen taller than the window (the selection stays visible, hovering a visible row never scrolls, hits map back to the right row); wraparound, Esc/back behaviour, OPTIONS reachable from both menus, MSAA only outside a session, DISPLAY toggles windowed/fullscreen, FIELD OF VIEW steps its presets (and the projection really uses it); SENSITIVITY / INVERT Y change and save; rebinding waits for a key, ignores menu keys, takes the key from its old action (which shows NONE), and is cancelled by Esc, moving away or BACK; RESET KEYS; **every label drawable by the 5×7 A–Z/0–9 font** |
| Shadows (CSM) | render, app | split distances, texel snapping, cascade spheres cover their frustum slice at every FOV from 30° to 120°; caster pancaking (the tower's top is culled by the full cascade frustum but kept by caster culling, and sits up-light of the near plane) |
| Pass list | gfx, render | the barrier tracker: a clear waits for last frame's readers; one transition serves every later read in its layout (GTAO's depth barrier covers the main pass's test and TAA's read); a write waits for every reader; a tested-and-sampled depth merges into one layout, and one image in two layouts in one pass is refused; the swapchain chains after the acquire; a reset starts from `UNDEFINED`; a resolve writes its target; a buffer's read waits for its write. The frame plan, for every MSAA/TAA/FXAA/caster combination: in a steady frame every barrier waits for some earlier use; single-sample readers take the resolves under MSAA; the post chain reads TAA's output and writes where FXAA says |
| Lights | game, app | falloff reaches exactly 0 at the radius; frustum culling by sphere, not point; sphere-light specular (a CPU reference of the shader): src = 0 is the old point light, the smooth-metal singularity goes away, the highlight is the source's size, energy roughly conserved |
| Prefabs | game, assets, app | `player_start` placement, `prop`/`point_light` params and defaults, extras parsing, fallback for unknown prefabs; `environment`: no marker and an empty one keep the default look, every param lands, left-out params keep defaults and bad ones are reported, the sun's elevation/azimuth convention, and (end-to-end) a level's marker reaches `build_world`; the orb demo's obstacle boxes stand in the demo but not in a loaded level (end-to-end) |
| Materials | assets, render | glTF `alphaMode`/`alphaCutoff`/`doubleSided` load as `AlphaMode::Mask`(cutoff, default 0.5)/`Blend`/`Opaque` and the flag; the GPU record carries the cutoff only for MASK and the double-sided bit, which matches `mesh.frag`'s; glTF `occlusionTexture` and its strength load (an ARM map shares MR's image, one conversion), and the GPU record packs its slot above the flags (MR's own slot for an ARM map, 0 for none) with the strength in `params.z`, at the shift `mesh.frag` declares; `mesh.frag`'s occlusion follows glTF's strength rule and combines with GTAO by `min`; masked instances' runs follow every opaque one (one pipeline switch a pass), with the same instances and triangles, and a scene without masked materials builds the runs it always did; `mesh.frag` never discards or demotes (masked materials rely on its early-Z, testing depth for EQUAL against the prepass's cut) |
| Environment | render | nothing is baked (no specialization constant in either compiled shader); `Environment::default()`'s atmosphere is the old defaults slot by slot, and every field reaches it; mesh.frag's `Globals` block is the Rust struct's size with `Atmosphere` at its tail, and sky.frag declares it and the `#define`s over it text for text; height fog's closed form equals numeric integration of its density (rays up, down and level), uniform fog is exactly `density·dist`, the sky's infinite-ray limit; `sky()` and the fog functions are textually identical in both shaders; sky.frag's push block is the 96 bytes the pass pushes |
| Weather | game | the sun's path (noon elevation and bearing, morning east, an equinox rising at 6 and setting at 18, sunrise on the horizon, polar days, a level's sun found again); the reference-day warp; log-space key blending with exact ends; sampling at, between and across keys; every weather's keys sorted and usable, OVERCAST's 15:00 key the zone's `environment` (read from `gen_zonescene.py`); the sun/moon switch continuous through sunrise and sunset; the clock's speed and wrap; LEVEL exactly the level's look, and with a clock the level's palette under the moving sun (faded in at sunrise, no moon by night); a weather's frame (key, fog height, sun and moon); transitions start from the screen and stay continuous through a re-pick; SPEED presets; the WEATHER rows cycle and wrap, and it's in game only; the marker's `weather`/`time`/`time_speed`/`latitude`/`day_of_year` and their bad values |
| Scene loading | assets | mesh dedup, transforms accumulate, meshes stay in local space; materials sharing an image share one texture, which isn't decoded until asked and then matches the old RGB→RGBA expansion; a material's `extras.surface` is read (a non-string one warns and is dropped) |
| Mip chains | gfx | level count per texture size (square, non-square, non-power-of-two) |
| Texture bake | assets, bake | keys separate content and kind (sRGB colour vs data), and are computed from the encoded bytes without decoding; BC7 level sizes round partial blocks up; baked files round-trip and reject truncation, wrong sizes or an unknown kind; sRGB mips average *light* (black/white → 188, not 128); chains end at 1×1; every baked level has exactly the blocks Vulkan copies |
| Mesh bake + LOD | assets, bake, render | baked meshes round-trip and reject damage (truncation, trailing bytes, bad magic, out-of-range index, decreasing or NaN error, partial triangle, no LODs); the mesh key ignores the material; a sphere gets a chain with fewer triangles and growing error per level and LOD0 is the input reordered; small meshes stay LOD0; disconnected parts prune; the pixel and texel rules (distance, scale, non-uniform scale, inside the sphere); runs split per (mesh, LOD) |
| Sky visibility | assets, bake, render, app | the BVH finds what brute force finds; open ground sees the whole sky; a closed room with 0.2 m walls is dark inside, nothing leaking in even at corners; a door lets sky in from its side; a canopy shades only what's under it; the probe finds the inside of a closed box, not of a double-sided one; inside cells take their darkest neighbour, ring by ring; mirrored placements face outwards; occluders skip cutouts and `shadow: false`, and take the coarsest LOD within 5 cm; big levels get bigger cells; open-sky weights equal `sky_irradiance`'s for every normal and specular occlusion is 1 in the open, 0 shut in, and smooth through the horizon; the encoding is exact for open sky, and free distances round down; a sample leaves out a cell behind a wall and fades to open sky outside the grid; the key covers occluders and nothing else; files round-trip and reject damage; `mesh.frag` declares the same constants; (end to end) `build_world` finds a level's volume by its key, the first scene's of several, and not after the level changes |
| GTAO | render, app | a Rust reference of the three passes (the depth levels, half-resolution GTAO and denoise) and of `mesh.frag`'s upsample, run over depth buffers ray-cast from known scenes: distances come back from depth as glam's projection put them; the slice integral equals Simpson's rule; an open floor reads 1 (0.9998 mean), and so do face-on and 40° walls filling the view, out to the screen's edges; a wall darkens the floor at its foot (0.4–0.8) and nowhere else; the denoise and the upsample keep to their side of an edge; the upsample copies its texels and blends bilinearly on a plane; each depth level holds every 2^k-th pixel, and is exact on a plane; a step's level follows its length; half resolution keeps the full-resolution result at a wall's foot; every 4×4 window holds each jitter once; the multi-bounce and specular-occlusion fits leave open surfaces alone; the shaders declare the reference's constants, gtao and the denoise share `view_pos` / `depth_normal` text for text, and `mesh.frag` converts to their space as the view test pins it; the GRAPHICS row, `ambient_occlusion` key and `--no-ao` |
| Bloom | gfx, render, app | the chain's level sizes (half resolution, halving to an 8-texel side, at most 7 levels, never zero, odd and tiny windows); a Rust reference of the chain: a flat image blooms to itself, a bright pixel spreads with falloff, a plain downsample keeps energy within 5%, and the shaders declare the reference's weights; the steps go down then up; the tonemap's push block matches its Rust struct; the GRAPHICS row toggles it; `--no-bloom` (with the other flags) parses |
| Auto-exposure | render, app | a Rust reference of the metering, whose constants and state layout the shaders must declare: luminance round-trips through its bin (black to bin 0), the 10–90% trimmed mean ignores the tails, adaptation converges, brightens faster than it darkens and is frame-rate independent, the exposure maps `KEY` and clamps, the centre weight falls off but never to 0; `exposure_min`/`exposure_max` land and bad values are reported; the GRAPHICS row and `--no-auto-exposure` |
| Config files | app | both templates parse back to the defaults; `display` parses and rejects anything but `"windowed"` / `"fullscreen"`; `fov` accepts 30–120 whole degrees only; each bad line warns and keeps the default; saving edits one value in place; create-once, the `settings.toml` → `graphics.toml` migration, an unreadable file is left alone |
| Controls | app | key names round-trip; reserved menu keys are refused; a key on two actions warns; two keys on one action hold until both are released; toggles ignore auto-repeat but exposure repeats; a rebound jump moves; `toggle_fullscreen` defaults to F11 (also in files that predate it) and ignores auto-repeat; `rebind` steals the key and refuses menu/unnamed keys; sensitivity presets step and wrap; saved literals parse back, and saving every binding into the template keeps its comments; every key's menu label is drawable |
| Texture slots | render | one slot per unique image × colour space; sRGB and UNORM uses of the same pixels stay separate; overflow past the capacity is counted and falls back to the defaults |
| Light clusters | render | GLSL grid constants + `MAX_LIGHTS` match the Rust ones |
| Ropes | assets, game | still, a rope hangs straight at its length (control: one pass, no attachment, stretches); released, it swings with a pendulum's period, dies down and never grows (control: 4× gravity halves the period); wind leans it downwind within 1% of its length (control: still air); an unweighted rope bows (control: one segment); segment and item matrices sit on the points, and at rest the item is drawn as authored; determinism and interpolation; the unit tube; `hanging` params, spawn (anchor above the node, no collider) and the level's wind params; a `light` on it (params, bad keys) rides the item: at rest where the node puts it, swung with it (control: unlit, no light) |
| TAA | render, app | a Rust reference of `taa.comp`: the Halton jitter covers the pixel evenly; jittering moves the image by exactly its offset and keeps depth; the reprojection finds last frame's pixel; the colour transforms round-trip; Catmull-Rom is exact at texel centres and along rows; the clip keeps the inside and pulls in the outside; a still edge converges to its coverage (control: no jitter, hard edge); a pan keeps its image (control: an unfollowed history smears); the shader declares the reference's constants and formulas; the GRAPHICS row, `taa` key, `--no-taa` and F3 |

### What tests cannot cover

And how each is checked instead:

- **GPU correctness:** Vulkan validation errors on a debug run (§8 below).
- **Anything visual:** a person looking at the screen. Say what to look for
  when handing off a render change.
- **Performance:** `--bench` A/B runs (§7 below).
- **Shader syntax:** shaders compile at *build* time, so an error there is a
  build error.
- **Real assets** (the generated `scratch/` scenes, the recorded sound pack):
  running the game, and one-off harnesses while developing (above).

---

## 6. Log lines

Everything goes to stderr.

| Prefix | Meaning |
|---|---|
| `[gfx] device: …` | the GPU picked at startup. Check it: machines with an iGPU + dGPU list both |
| `[gfx] MSAA: …` | requested vs actual sample count |
| `[vulkan] …` | validation messages (debug builds), or the "layer not found" warning |
| `[gpu] shadow … cluster … geo … ao … taa … bloom … expo … post … frame …` | smoothed per-pass GPU ms, about once a second |
| `[quality] shadows / fxaa / taa / bloom / auto exposure: …` | a setting changed |
| `[config] …` | a config file created / loaded / renamed, a line ignored or a key bound twice, plus the effective graphics settings at startup |
| `[light] N visible lights exceeds MAX_LIGHTS` | lights past 128 were dropped this frame |
| `[audio] started …; sounds: R recorded, S synthesised` / `… not found (…fetch_assets.py…)` / `footstep_grass_000.ogg: …; grass steps use the concrete ones (…)` / `unavailable: …` / `N lamps humming` | the mixer came up (or why not; the game then runs silent), which surfaces lack their own steps, and how many lamps a session gave a hum. With the whole pack: 35 recorded, 0 synthesised |
| `[mesh] N meshes (B baked, L LODs): …` | geometry uploaded at load, and how much of it came from the bake |
| `[mesh] N textures uploaded … (B baked BC7, R raw; D images decoded)` | textures at load; with a complete bake D is 0 |
| `[scene] colliders: N mesh (M from LODs), H hull, B box` | the colliders a session built; with a bake, detailed props should be LODs, not hulls |
| `[scene] surfaces: N concrete, M grass, …` / `[scene] unknown surface "x"; using concrete` | the same colliders by footstep surface (only surfaces that have any), and any material tag that isn't a surface |
| `[load] scenes … world … renderer … total …` | where a session's load time went: parsing + reading images + meshes, spawning + colliders, GPU upload |
| `[bench] …` | the `--bench` summary, including triangles submitted per frame (`Mtris main/shadow`), the share of instances at each LOD (`LOD mix`), how many instances of cutout materials were drawn (`masked main/shadow`), and, with auto-exposure on, the exposure it metered over the sweep (`exposure`) |

---

## 7. Measuring performance

### Pin the GPU clocks first

A fast GPU running at vsync idles most of each frame and **downclocks**. That
inflates light passes more than heavy ones, so even back-to-back A/B comparisons
come out biased. On AMD/amdgpu:

```bash
echo profile_standard | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
```

Find the right `cardN` with `grep -H . /sys/class/drm/card[0-9]/device/device`: the
RX 7800 XT is `0x747e`. The setting resets on reboot, or write `auto`.

It may not hold the **memory** clock. On 2026-09-30, `pp_dpm_mclk` wandered
456–1218 MHz within single runs, and bandwidth-bound passes (bloom, exposure,
post) came out bimodal in both binaries of an A/B (bloom median 0.07 or 0.10 ms
from run to run). Sample it during a run
(`grep '\*' /sys/class/drm/card1/device/pp_dpm_mclk` once a second), and judge
those passes over more rounds, or not at all.

### `--bench`

```bash
cargo run -- --bench scratch/lights120.gltf
```

What it does: skips the menu and opens 1920×1080 (the compositor may shrink it;
the report prints the real size). It then waits 120 frames at the spawn point,
turns 360° at level pitch over 720 frames, and records **raw** per-frame times,
not the smoothed `[gpu]` line. It prints min / p10 / median / p90 / max for
each pass plus the visible-light count, then exits. Don't touch the window
while it runs. It ignores `config/`: settings come from the
defaults and CLI flags only.

Bloom and auto-exposure are on by default, so a bench's `frame` includes
them (~0.10 and ~0.06 ms with pinned clocks). So is sky occlusion when the
scene's volume is baked (`geo` +0.04 ms on the zone, +0.06 on `lights120`),
and so is GTAO (its own `ao` line, 0.12 ms on the zone; its upsample runs in
the main pass, inside `geo`), and TAA (its own `taa` line, 0.15 ms pinned;
frames +0.18 ms on the zone, +0.19 on `lights120`). TAA is on by default in
`--bench` too, so it's in every frame from its commit on. It reads a lot of
memory, and unpinned the memory clock wanders (96 to 1218 MHz within one run
here): its median held at 0.12–0.14 ms while its mean doubled. So time it
pinned. `geo` also times the
depth prepass, which now runs on its own before GTAO. Pinned clocks run below
boost, so these read higher than unpinned numbers. Compare pinned only with
pinned; debug and release measure the same. To compare with numbers from before they existed, add
`--no-bloom --no-auto-exposure --no-sky-occlusion --no-ao --no-taa`. The zone gained trees
and bushes after most of ARCHITECTURE.md's zone figures were taken (frame 0.72 →
1.10 ms pinned, 1.01 once far foliage shadows took coarser LODs, 1.19 with TAA). Regenerating an older zone isn't possible from this script, so
compare only zones generated by the same version.

### A/B recipe

1. State the prediction *before* measuring, so it can be wrong.
2. Build the committed version and keep its binary. **Only with uncommitted
   changes:** on a clean tree `git stash` saves nothing, and the `pop` would
   then apply an *older* stash. The `git diff HEAD --quiet` guard below refuses in
   that case.
   ```bash
   git diff HEAD --quiet && echo "nothing to stash" || { git stash && cargo build --workspace && cp target/debug/feather /tmp/feather_base; git stash pop; }
   ```
3. Build yours, then alternate runs back-to-back at the same size:
   ```bash
   cargo build --workspace
   for v in /tmp/feather_base target/debug/feather; do $v --bench scratch/lights120.gltf 2>&1 | grep '^\[bench\]'; done
   ```
4. Compare medians and the p10–p90 band. **Only compare numbers taken on this
   machine, in one session.** ARCHITECTURE.md's milliseconds come from specific
   hardware, and only the ratios carry over.
5. **Before blaming a ≤ 0.02 ms `geo` change on your change's maths,** check
   the compiled shader. Any edit to `mesh.frag` can shift register allocation
   in its light loop and move `lights120`'s `geo` by ~0.02 ms on its own
   (ARCHITECTURE.md §13, the fog finding).
   - `RADV_DEBUG=shaderstats` prints each shader's VGPRs, instructions and
     waves.
   - `RADV_DEBUG=shaders,nonir` prints the ISA.
   - If the instruction counts match and only register numbers differ, it's
     allocation.

---

## 8. Validation

| Check | How |
|---|---|
| Core validation | automatic on debug builds, if `VK_LAYER_KHRONOS_validation` is installed; otherwise a `[vulkan] … not found` warning and it runs unvalidated |
| Sync validation | `VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true cargo run -- --bench scratch/lights120.gltf` |
| Loader debugging | `VK_LOADER_DEBUG=layer` shows which layers actually load |
| Simulate a missing layer | `VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation` |

**Sync validation is clean: expect zero messages.** Anything it reports after
your change is yours (ARCHITECTURE.md §21 has the rule that fixed the last
batch). To see the messages grouped by type:

```bash
VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true target/debug/feather --bench scratch/lights120.gltf 2>&1 \
  | grep -o "\[ [A-Za-z0-9_-]* \]" | sort | uniq -c
```

**One exception, under MSAA (2× and up):** the installed layer (Ubuntu 24.04's
1.3.275) reports one `SYNC-HAZARD-READ-AFTER-WRITE` a frame on the depth
prepass's resolve. It's a false positive, fixed upstream (ARCHITECTURE.md §13,
GTAO Limits). This drops exactly that message and counts what's left, which
should be 0:

```bash
VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true target/debug/feather --bench --msaa 4 scratch/zone.glb 2>&1 \
  | grep -E 'VUID|SYNC-' \
  | awk '!(/vkCmdEndRendering\(\): pDepthAttachment->imageView/ && /during resolve/)' | wc -l
```

It matches the message's text because its ID (`0xe4d96472`) is shared by every
read-after-write hazard. With a GTAO barrier deliberately removed, the real
hazards all came through.

Don't truncate these messages with `cut`: the useful detail is at the end.

---

## 9. Pitfalls that bite

- **UI text is 5×7, A–Z and 0–9 only.** Anything else renders blank. A test
  guards the menu labels, so run it after renaming a row.
- **`App` field order is load-bearing.** `session` must stay declared before
  `renderer`, so GPU resources drop while the device is alive. Nothing in the
  type system enforces this.
- **rustfmt:** the repo is not fmt-clean. Keep your own new lines clean and
  never reformat existing code. Compare the diff count before and after:
  `cargo fmt -p feather-<crate> -- --check | grep -c '^Diff in'`.
