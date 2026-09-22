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
use it for timing.

| Flag | Effect |
|---|---|
| `SCENE` (bare path, repeatable) | glTF scene(s) NEW GAME loads. Each node becomes an entity; glTF `extras` become prefabs (§4). |
| `--msaa N` | Geometry-pass MSAA sample count: 1 / 2 / 4 / 8, clamped to what the device supports. Overrides `config/graphics.toml` for this run only (never saved). Also changeable from the main menu's OPTIONS > GRAPHICS (not mid-game), which *is* saved. |
| `--no-bake` | Ignore baked textures and upload raw RGBA8, for A/B against the bake (§4). |
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
each with defaults and comments; delete one to reset it. Both are a flat
`key = value` subset of TOML parsed by hand (`app/src/config/`), with no
`[sections]`.

**`graphics.toml`** — saved when you change a setting in the menu or with the
F1 / F2 bindings. Only that key's value is rewritten; comments, other lines and
hand edits made while the game runs are kept. (It was `settings.toml` before
controls got their own file; an old one is renamed on first run.)

| Key | Values | Default |
|---|---|---|
| `shadows` | `"high"`, `"medium"`, `"low"`, `"off"` | `"high"` |
| `msaa` | `1`, `2`, `4`, `8` (clamped to the device) | `1` |
| `fxaa` | `true`, `false` | `false` |

**`controls.toml`** — key bindings and mouse look. Read-only (edit it by hand;
there's no rebinding screen yet), and read at startup.

- `action = ["Key", ...]`: one or more keys, or `[]` to unbind. Actions:
  `forward`, `back`, `left`, `right`, `jump` (also fly up in noclip), `down`,
  `noclip`, `exposure_down`, `exposure_up`, `cycle_shadows`, `toggle_fxaa`.
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
  (`true` / `false`).

**Both files:**

- **Precedence:** defaults < file < CLI flags. A flag applies to that run only.
- **Mistakes aren't fatal:** a bad line logs `[config] … line N: …; ignored`
  and that key (or action) keeps its default. An unreadable file (e.g. not
  UTF-8) is left as-is and the run uses defaults.
- **`--bench` ignores both** (neither reads nor creates them), so timings
  never depend on personal settings.

### Controls

Defaults, rebindable in `config/controls.toml` (Esc isn't).

| In game | |
|---|---|
| WASD + mouse | move / look |
| Space | jump |
| V | noclip (free flight) on/off |
| Left Ctrl | fly down (noclip) |
| `[` / `]` | exposure down / up |
| F1 | cycle shadow quality: OFF → LOW (4×512) → MEDIUM (4×1024) → HIGH (4×2048) |
| F2 | FXAA on/off |
| Esc | pause |

**Menus:** mouse (hover + click) or arrows + Enter. **Esc** steps back one
screen, and resumes play only from the pause menu's top level.

- **Main menu:** NEW GAME / OPTIONS / QUIT
- **Pause menu:** CONTINUE / OPTIONS / MAIN MENU / EXIT
- **OPTIONS > GRAPHICS:** shadows and FXAA are live; MSAA can change only from
  the main menu, because the mesh and sky pipelines bake the sample count
  (in-game it reads `MENU ONLY`).
- **SOUND, GAMEPLAY:** placeholders; their rows do nothing yet.

Defaults: shadows HIGH, FXAA off, MSAA 1×.

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
| `prop` | `collide` (true), `shadow` (true), `collider` (`auto`) | a mesh with optional collider / shadow casting. `collider`: `auto` (exact mesh up to 2048 triangles, convex hull above), `mesh`, `hull`, `box`, `none`. Detailed props should never collide as exact meshes: a 34k-triangle one cost 27 ms per physics tick |
| `point_light` | `color` [1,1,1], `intensity` 12, `radius` 10, `source_radius` 0.1 | on a bare marker or on geometry (a lamp that also renders). `source_radius` is the emitter's physical size, clamped to [0, radius]: it sets the size of the highlight on shiny surfaces. Match it to the lamp's geometry; 0 is a true point, which makes a pinprick-bright highlight on smooth metal |

An unknown prefab name falls back to static geometry. Up to 128 point lights
can be *visible* at once; lights past that are dropped for the frame with a
`[light]` warning.

### A scene from real CC0 models — Kenney Nature Kit

```bash
python3 tools/fetch_assets.py                 # once: ~10.5 MB, SHA-256 pinned
python3 tools/gen_naturescene.py --check      # writes scratch/nature.glb
cargo run -- scratch/nature.glb
```

`fetch_assets.py` downloads the pinned archive, **refuses it if the SHA-256
doesn't match**, and extracts only the `.glb` models and the licence into
`scratch/assets/kenney_nature_kit/` (gitignored, like all of `scratch/`). Run
it again and it's a no-op. `--zip PATH` uses an archive you already have
(still hash-checked); `--force` re-extracts.

The kit is **CC0** (Creative Commons Zero, see its `License.txt`): free for any
use, with credit to Kenney (kenney.nl) appreciated but not required.

`gen_naturescene.py` builds a grass meadow, forest, rocks, a cliff ridge on the
east edge, a camp with a fire light (you spawn at its edge, facing the fire), and a
stone path lit by lamps. Options: `--density low|med|high`, `--seed N`,
`-o PATH`, `--check` (the same engine constraints as the procedural
generator). Grass, flowers and ground tiles have no collider and cast no
shadow; trees, rocks, the tent and fences do both.

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

### Compressing textures — `feather-bake`

```bash
cargo run --release -p feather-bake -- scratch/detail.glb   # any scenes; incremental
cargo run --release -- scratch/detail.glb                    # now uploads BC7
```

The bake turns every texture a scene uses into a **BC7** mip chain in
`scratch/bake/tex/`: 4× less GPU memory (the detail scene: 64 MB instead of
256 MB) and no load-time mip building. The engine picks the baked version up
automatically, keyed on the texture's pixels. Anything not baked still loads
raw, so baking is optional. The `[mesh]` line says which you got:
`12 textures uploaded ... (12 baked BC7, 0 raw), 64.0 MB`. Re-running skips
textures already baked; delete `scratch/bake/` to start over. Use `--release`
for the bake: the encoder is slow in debug.

---

## 5. Tests — what exists and what it guards

`cargo test --workspace`: 61 tests, all CPU-side (none needs a GPU).

| Area | Crate | What the tests pin down |
|---|---|---|
| Character controller | app | settling, walking speed, jumps and head bumps, no air-jump, autostep lip vs wall, sliding along box faces, noclip |
| Collision proxies | app | the `collider` param picks mesh/hull/box/none, `auto` switches to a hull past 2048 triangles, `collide: false` still wins; a dense 30 cm prop is walkable as a hull; a flat hull still holds the player |
| Menus | app | row layout and hit-testing, wraparound, Esc/back behaviour, OPTIONS reachable from both menus, MSAA only outside a session, **every label drawable by the 5×7 A–Z/0–9 font** |
| Shadows (CSM) | app | split distances, texel snapping, cascade spheres cover their frustum slice; caster pancaking (the tower's top is culled by the full cascade frustum but kept by caster culling, and sits up-light of the near plane) |
| Lights | app | falloff reaches exactly 0 at the radius; frustum culling by sphere, not point; sphere-light specular (a CPU reference of the shader): src = 0 is the old point light, the smooth-metal singularity goes away, the highlight is the source's size, energy roughly conserved |
| Prefabs | app, assets | `player_start` placement, `prop`/`point_light` params and defaults, extras parsing, fallback for unknown prefabs |
| Scene loading | assets | mesh dedup, transforms accumulate, meshes stay in local space; materials sharing an image share one decoded copy |
| Mip chains | gfx | level count per texture size (square, non-square, non-power-of-two) |
| Texture bake | assets, bake | keys separate content and colour space; BC7 level sizes round partial blocks up; baked files round-trip and reject truncation or wrong sizes; sRGB mips average *light* (black/white → 188, not 128); chains end at 1×1; every baked level has exactly the blocks Vulkan copies |
| Texture slots | render | one slot per unique image × colour space; sRGB and UNORM uses of the same pixels stay separate; overflow past the capacity is counted and falls back to the defaults |
| Light clusters | render | GLSL grid constants + `MAX_LIGHTS` match the Rust ones |

What tests **cannot** cover, and how it is checked instead:

- **GPU correctness:** Vulkan validation errors on a debug run (§8 below).
- **Anything visual:** a person looking at the screen. Say what to look for
  when handing off a render change.
- **Performance:** `--bench` A/B runs (§7 below).
- **Shader syntax:** shaders compile at *build* time, so an error there is a
  build error.

---

## 6. Log lines

Everything goes to stderr.

| Prefix | Meaning |
|---|---|
| `[gfx] device: …` | the GPU picked at startup. Check it: machines with an iGPU + dGPU list both |
| `[gfx] MSAA: …` | requested vs actual sample count |
| `[vulkan] …` | validation messages (debug builds), or the "layer not found" warning |
| `[gpu] shadow … cluster … geo … post … frame …` | smoothed per-pass GPU ms, about once a second |
| `[quality] shadows / fxaa: …` | a setting changed |
| `[config] …` | a config file created / loaded / renamed, a line ignored or a key bound twice, plus the effective graphics settings at startup |
| `[light] N visible lights exceeds MAX_LIGHTS` | lights past 128 were dropped this frame |
| `[bench] …` | the `--bench` summary |

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
