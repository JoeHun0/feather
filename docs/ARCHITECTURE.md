# Feather — Engine Architecture

Status: design reference (pre-implementation beyond the bootstrap).
Scope: a from-scratch Rust + Vulkan engine for a minimalist open-world FPS.
Visual floor is a 2010-era look; the baseline actually targets ~2016 image
quality where it costs little. Long-term goal: an original Zone-flavored
survival FPS; a literal STALKER: Shadow of Chernobyl rebuild is a stretch
dream, not a dependency.

This document is the single source of truth for engine design decisions. Code
lives elsewhere; this is the *why* and the *shape*.

---

## 1. Guiding principles

- **The ECS `World` is the single source of truth for simulation.** The
  renderer is a *consumer* of a per-frame snapshot, never a reader of live sim
  state.
- **Decouple sim from render** enough to parallelize; keep it **simple enough
  for one person** to hold in their head. Every decision serves both.
- **Data-oriented from day one.** IDs/handles instead of pointers; contiguous
  component arrays; systems as transforms over data.
- **Defer the expensive-but-optional.** Design the *seam* now, build the
  overlap/streaming/GPU-driven path later. (Same instinct as bounded-arena
  before streaming.)
- **Open formats at every authoring boundary; one custom format at exactly one
  point** — the offline bake output.

## 2. Technology stack (committed)

| Concern            | Choice                                             |
|--------------------|----------------------------------------------------|
| Language           | Rust                                               |
| Graphics API       | Vulkan via `ash` (thin, unsafe 1:1 bindings)       |
| GPU allocation     | `vk-mem` (VMA)                                      |
| ECS                | `bevy_ecs` (standalone, `multi_threaded`)          |
| Data parallelism   | `rayon`                                             |
| Math               | `glam` (SIMD)                                       |
| Windowing/input    | `winit` (+ `gilrs` for gamepad)                    |
| Physics            | `rapier` (kinematic character controller)          |
| Audio              | `kira`                                             |
| Dev UI             | `egui` (ash integration)                           |
| CPU profiling      | Tracy (`tracy-client`)                             |

Rationale summary: Vulkan for native Linux/Steam Deck reach + best validation
tooling; Rust because a solo dev's hardest bug class (threaded data races) is a
compile error, and ECS neutralizes the borrow checker's main ergonomic cost. No
multi-backend RHI abstraction — Vulkan-only, so keep all `vk::` types inside the
`gfx`/`render` crates.

## 3. Rendering strategy

**Clustered forward**, not deferred. Open-world sightlines mean low overdraw;
transparency (water, foliage, glass) is first-class; MSAA stays free (period-
accurate and clean); one lighting path for a solo dev. Clustered light
assignment gives the many-lights capability that was deferred's main selling
point without its G-buffer VRAM/bandwidth tax or its transparency/MSAA pain.

Hybrid escalation (later, optional): add a **deferred-opaque** path only if
dense indoor many-light scenes demand it. Because material/light data is
structured and handle-indexed, a deferred pass consumes the same data with no
asset/extract rework. Starting forward and adding deferred is additive; the
reverse is a rewrite.

### Quality tiers

- **Baseline (build from the start, near-zero extra cost):**
  metallic-roughness **PBR**; **linear-space lighting + HDR + tonemapping**
  (AgX/ACES); **IBL** ambient (prefiltered env + BRDF LUT / SH irradiance);
  **GTAO**; **CSM** sun shadows.
- **Maybe (opt-in passes, real cost / real gain; hooks preserved):**
  **SSR**; **volumetric fog** (froxel); light **probes / irradiance volumes**
  for dynamic-object bounce.
- **Skip (the UE5-cost trap):** realtime GI (Lumen-style), ray-traced
  reflections/GI, virtualized geometry (Nanite-like).

The "maybe" tier is cheap to keep open because both consume data already
produced: SSR needs a **pre-transparent HDR resolve** (also a baseline step);
volumetrics needs the **CSM + clustered light buffer** readable from compute
(both already built).

## 4. Frame pipeline

```
Input ──▶ Fixed-step sim ──▶ Extract (double-buffered) ──▶ Cull ──▶ Record + submit
          (bevy_ecs,          (sim↔render seam)            (rayon)   (per-thread
           parallel systems)                                          command buffers)
                                                                          │
                                                              GPU executes frame N-1
                                                              (overlaps CPU frame N)
```

- **Extract seam (most important boundary).** Sim writes gameplay components;
  the renderer never reads them directly. Once per frame, extract copies only
  render-relevant data (world matrix, mesh/material handles, light params) into
  a **double-buffered `RenderFrame`**. This owned snapshot is what later allows
  GPU frame N-1 to overlap CPU frame N without sharing mutable sim state.
- **Fixed timestep, interpolated render.** Sim runs at fixed `dt` (accumulator);
  render interpolates between the last two sim states by
  `alpha = accumulator / FIXED_DT`. Non-negotiable for stable/deterministic
  physics. Interpolation is computed at extract.

### Threading model (three levels)

1. **Across systems** — `bevy_ecs` scheduler runs disjoint systems on multiple
   threads automatically from declared component access.
2. **Within a system** — `rayon`/`par_iter` for heavy loops (cull, particle
   integration, skinning).
3. **Across stages (pipelining)** — sim N+1 overlapping GPU N. **Deferred by
   design**: build levels 1–2 now, run stages sequentially first; the
   double-buffered `RenderFrame` is the seam that makes level 3 addable later.

## 5. Materials and bindless

PBR + bindless makes materials **plain data indexed by an integer**, not
pipeline/descriptor state.

### Texture set (glTF metallic-roughness aligned)

| Map        | Color space | Compression | Notes                              |
|------------|-------------|-------------|------------------------------------|
| Base color | sRGB        | BC7         | RGBA; A = cutout alpha             |
| Normal     | linear      | BC5         | 2-channel, reconstruct Z in shader |
| ORM        | linear      | BC7/BC1     | occlusion=R, roughness=G, metal=B  |
| Emissive   | sRGB        | BC7         | optional                           |

BCn + mips generated at bake time. This is where the real VRAM/bandwidth budget
is spent.

### GPU material (SSBO, 64 bytes, std430, indexed by `material_id`)

```glsl
struct GpuMaterial {
    vec4  base_color_factor;   // linear multiplier
    vec4  emissive;            // rgb = emissive, a = metallic_factor
    vec4  params;              // x=roughness y=normal_scale z=occlusion w=alpha_cutoff
    uvec4 tex;                 // bindless indices: x=baseColor y=normal z=orm w=emissive
};
```

CPU-side only (routing, not shading): alpha mode (opaque/mask/blend),
double-sided, cull — used to pick pipeline and draw bucket.

### Bindless mechanics

- One descriptor set with a large `textures[]` sampled-image array (4096+) and a
  small sampler array.
- Features: `runtimeDescriptorArray`,
  `shaderSampledImageArrayNonUniformIndexing`,
  `descriptorBindingSampledImageUpdateAfterBind`,
  `descriptorBindingPartiallyBound`, pool `UPDATE_AFTER_BIND`.
- Resident **default textures** at fixed low indices (white, flat-normal,
  default-ORM, black) so missing maps resolve without a branch.
- Indirection chain: instance carries `material_id` →
  `materials[material_id]` → `textures[nonuniformEXT(mat.tex.x)]`. Result:
  distinct pipelines only per **shading-model × vertex-layout × blend-state**
  (~4–6 total), not per material. This is the biggest solo-dev workload cut.

## 6. Geometry and instanced draws

Geometry and per-object data arrive at **two different rates**.

### Vertex layout (static, device-local, shared buffers)

```glsl
struct Vertex {            // per-vertex
    vec3 position; vec3 normal; vec4 tangent; vec2 uv;
};
```

All meshes share one vertex + one index buffer; a mesh is a slice
`{ first_index, index_count, vertex_offset }`. Skinned meshes add
`joint_indices` (u8/u16 ×4) + `weights` (f32 ×4) → a second vertex layout →
skinned pipeline variants.

### Instance record (per-frame SSBO, indexed by `gl_InstanceIndex`)

```glsl
struct GpuInstance {
    mat4 model;
    uint material_id;      // -> materials[]
    uint joint_offset;     // skinned only; else unused
    uint _pad0, _pad1;
};
```

Pulled from an SSBO (not vertex attributes — a `mat4` would eat 4 attribute
slots). Vulkan `gl_InstanceIndex` includes `firstInstance`.

### Extract/batch (what turns ECS transforms into draws)

Per frame, over the culled visible set:
1. Each visible entity → `(mesh_id, material_id, model)`.
2. Sort by **pipeline bucket** (opaque/masked/transparent/skinned), then by
   `mesh_id` within bucket.
3. Write `GpuInstance` records in sorted order → contiguous runs per mesh.
4. Per contiguous same-mesh run, emit one
   `vkCmdDrawIndexed(index_count, run_length, first_index, vertex_offset, run_start)`.

500 crate entities → **one** draw, `instanceCount = 500`. Transparents can't
batch freely — they sort **back-to-front** for blend correctness (smaller
draws). Opaque/masked are where instancing pays.

Escalation (later): write `VkDrawIndexedIndirectCommand[]` and use
`vkCmdDrawIndexedIndirect`, populated by GPU-side culling. CPU loop is the
correct first version and stays as the weak-hardware fallback.

## 7. Load-time bindless allocator

Maps **stable handles** (content-hash/path, cross-session) → **resident slots**
(descriptor index, buffer offset), once, at load. Downstream only sees resolved
slots.

- **Texture slots** — free-list over `textures[]`; slots `0..K` reserved for
  defaults.
- **Material slots** — free-list into `materials[]`; `material_id` = the index.
- **Mesh residency** — offset sub-allocator over the shared vertex/index
  buffers; returns the slice. Free-list of ranges, upgrade to slab/arena per
  vertex-format if fragmentation bites.
- **Dedup + refcounts** — `HashMap<AssetId, (slot, refcount)>`; upload once,
  refcount for correct unload/streaming.
- **Load flow** (device steps on the allocator/descriptor-owning thread, drained
  at frame start under a per-frame budget so bursts can't hitch): `alloc()` →
  staging→device copy on transfer queue → `vkUpdateDescriptorSets` at
  `dstArrayElement = slot` (legal because update-after-bind + partially-bound) →
  patch `GpuMaterial.tex` → publish handle→slot.
- **Non-blocking resolve** — a `Handle` is `(index, generation)`; the table
  entry holds the resident slot or a loading marker that resolves to a default.
  Systems/extract never wait. This is what lets async streaming drop in later.

## 8. Culling

Sits between sim and extract; turns "every entity" into "visible set, per view."

- **Broad → narrow.** Test chunk AABBs vs frustum first (reject whole chunks);
  only survivors get per-entity tests. Per-entity: bounding **sphere** vs six
  frustum planes (Gribb–Hartmann from `viewProj`); upgrade to AABB only where
  sphere over-inclusion costs draws.
- **Parallel (rayon fold-reduce).** `par_iter` over `(Transform, Bounds,
  RenderRef)`; each task builds a thread-local `Vec<VisibleItem>`, then
  concatenate. Global bucket sort after merge (`par_sort_unstable` if hot). No
  shared `Mutex<Vec>`.
- **LOD + interpolation happen here** (distance already computed): LOD picks
  `mesh_id`; extract stays pure sort-and-write.
- **Per view.** Main camera + each CSM cascade (ortho frustum) reuse the same
  function. Shadow views use a stripped `{mesh_id, model}` item and a depth-only
  pipeline.

```rust
struct VisibleItem { mesh_id: MeshId, material_id: u32, model: Mat4, depth: f32, bucket: Bucket }
```

## 9. GPU resource layer

### Per-frame memory (three classes)

- **Static device-local** — meshes, baked textures. Uploaded once via staging →
  transfer queue, never moved.
- **Per-frame dynamic** — camera/view UBO, `instances[]`, light list, cluster
  grid. Host-visible, persistently mapped, **N-buffered** (= frames-in-flight).
  Allocated via a **per-frame linear (bump) allocator**: one mapped buffer,
  bump from 0 each frame, reset when that frame's fence signals. Respect
  `min*BufferOffsetAlignment`; prefer `HOST_VISIBLE | HOST_COHERENT` (take
  `DEVICE_LOCAL | HOST_VISIBLE` on ReBAR/UMA).
- **Staging** — host-visible ring for streamed uploads, drained on transfer
  queue with a per-frame byte/count budget.

Render targets (HDR color, depth, shadow atlas, bloom mips, froxel texture) are
**engine-owned**, allocated once, reallocated only on resize — not per-frame.

Upload: dedicated transfer queue if available (overlaps rendering; needs
queue-family ownership release/acquire barrier), else graphics queue. Completion
signaled via timeline semaphore; loader flips handle to resident.

### Descriptor layout (organized by update frequency)

- **Set 0 — resident:** bindless `textures[]`, sampler array, `materials[]`.
  Changes only at asset load.
- **Set 1 — per-frame:** camera UBO, `instances[]`, light list, cluster grid.
  Bind the ring once; select this frame's suballocation via **dynamic offsets**
  (no per-frame set reallocation).
- **Set 2 — per-view:** shadow cascade matrices + params; rebound per view
  (main vs each cascade); stripped for shadow passes.

No per-material/per-draw set (bindless). Push constants (≤128 B) carry tiny
per-draw/per-view scalars (instance base offset, cascade index, debug toggles).
One shared pipeline layout spans nearly all pipelines.

## 10. Frame graph / pass ordering

Decision: **no automatic frame graph now.** Explicit hardcoded pass sequence +
a thin barrier helper (`Pass` declares `{reads, writes}`, emits transitions).
Refactor to a real graph only if pass permutations explode.

Per-frame sequence:

1. **Transfer/upload** — streamed copies; per-frame buffer writes are memcpy to
   the mapped ring (no pass).
2. **Shadow passes** — per CSM cascade, depth-only render of casters (stripped
   instance data).
3. **Depth prepass** — opaque depth into main depth buffer (enables `EQUAL`
   test + overdraw kill in the opaque pass).
4. **Cluster light assignment (compute)** — build grid, bin lights; writes
   cluster/light-index buffers.
5. **Opaque + masked forward** — into MSAA HDR; samples shadow maps + cluster
   light list + bindless materials.
6. **Sky/atmosphere** — depth-tested, after opaque, before transparents.
7. **HDR resolve** — MSAA HDR → single-sample readable HDR (the seam enabling
   transparents + SSR to read scene color).
8. **(maybe) SSR / volumetrics** — consume depth + resolved HDR + clusters.
9. **Transparent forward** — back-to-front, blend into HDR; reads resolved
   opaque color for water refraction.
10. **Post** — tonemap, bloom, SMAA → swapchain.
11. **UI/HUD**, then **present**.

Barriers: Vulkan 1.3 **sync2** (`VkImageMemoryBarrier2`, timeline semaphores).
Key hazards: shadow depth W→R before opaque; HDR color W→R at resolve; cluster
buffer W(compute)→R(fragment). Queues: graphics for all passes; cluster
(and later SSR/volumetric) compute on graphics for now — async compute only when
profiling shows a bubble to fill.

## 11. Shadows (CSM)

- **Splits:** practical/parallel-split scheme, 4 cascades (3 acceptable),
  `λ≈0.75`, blend of log + uniform.
- **Fit each cascade to a bounding sphere** (rotation-invariant → no
  size-change shimmer). Ortho centered on sphere, extents = radius.
- **Texel-snap** the ortho center to whole shadow-map texel increments each
  frame (kills edge crawl). Non-negotiable.
- **Atlas** = texture2D array, one layer/cascade, 2048² (far cascade 1024²).
- **Bias:** slope-scaled depth bias (`vkCmdSetDepthBias`) + **normal-offset
  bias** in the receiver (scaled by cascade world-texel size) — keeps depth
  bias small so contact shadows don't detach.
- **Selection + blend:** pick cascade by view-space depth vs splits; lerp across
  a small band at boundaries. 3×3 PCF baseline; PCSS-lite later.
- **Per-cascade cull** with the cascade ortho, volume extended toward the light
  (caster pancaking) so off-screen casters still shadow. Depth-only pipeline
  (alpha-mask discard only).
- **Sun is separate from clustering** — a global directional light + its
  cascades, not binned. Clusters are for local lights only.

Set 2 carries per-cascade light-space `viewProj`, split depths, world-texel
size.

## 12. Clustered lighting

- **Grid:** 16×9×24 = 3456 clusters start (tile ≈ 80–120 px). Depth sliced
  **exponentially** (Doom 2016):
  `slice = floor(numSlices · log(z/near) / log(far/near))`.
- **Cluster AABBs (rarely run):** one compute dispatch, one thread/cluster,
  cached; rebuilt only on resize/FOV change.
- **Light assignment (per frame, pass 4):** one thread/cluster; test each
  light's bound (sphere / cone-in-sphere) vs cluster AABB; append index.
  Brute-force cluster×light is fine at hundreds of lights.
- **Buffers (Set 1):**
  - `lights[]` — `GpuLight { vec4 pos_radius; vec4 color_intensity; vec4
    dir_cone; uint type; ... }`.
  - `clusterGrid[]` — `uvec2 {offset, count}` per cluster.
  - `lightIndexList[]` — flat `u32`; a global atomic counter hands each cluster a
    contiguous range. Cap lights/cluster (e.g. 128); clamp on overflow.
- **Fragment lookup:** reconstruct cluster from `gl_FragCoord.xy` + view depth →
  `clusterGrid` → loop `count` entries of `lightIndexList` → accumulate BRDF;
  then add the directional sun (with CSM) unconditionally.

## 13. Image pipeline: IBL + post

### IBL bake (ambient/indirect term; split-sum)

- **Source cubemap** — scene capture (6 faces) or HDR sky projected to cube. One
  sky-dominated env to start.
- **Diffuse irradiance** — cosine-convolved; store as **9 SH coefficients**
  (near-lossless for low-frequency, trivial in shader).
- **Prefiltered specular** — GGX-convolved per mip (mip = roughness), ~128²–256²
  base.
- **BRDF LUT** — 2D R16G16 512², axes (N·V, roughness); **scene-independent**,
  baked once, shipped as a constant.
- **Runtime term:**
  ```
  ambient = (irradiance(N)·albedo·(1-metallic)
           + prefiltered(R, rough·maxMip)·(F0·brdf.x + brdf.y)) · (ORM.ao × GTAO)
  ```
  Added to direct sun+cluster. GTAO modulates **ambient only**. Consumes exactly
  the material data already defined. All bake-time; runtime cost ≈ a few texture
  reads. Later: local probes per-object for interiors; irradiance volumes for
  dynamic bounce.

### Post chain (after transparents; HDR until tonemap)

1. **HDR resolve** (done as pass 7).
2. **Bloom** — threshold → progressive dual-filter down/up mip chain; in HDR,
   before tonemap; persistent mip pyramid.
3. **Exposure** — fixed constant to start; auto-exposure (luminance compute) is a
   hook in the camera UBO, not built now.
4. **Composite + tonemap** — scene + bloom, exposure, then **AgX**/ACES (in
   linear).
5. **OETF/gamma** — let the `_SRGB` swapchain encode; do **not** also gamma in
   shader (no double-correction).
6. **Anti-alias** — **SMAA** on LDR (post-tonemap). MSAA on geometry stays.
   (TAA is a later, bigger shift that replaces MSAA and moves before tonemap.)
7. **Output** → swapchain; **UI/HUD after tonemap** in LDR/sRGB.

Maybe-hooks placed: **SSR** between resolve and transparents (HDR, pre-tonemap);
**volumetric fog** composite in HDR after opaque/transparent, before bloom.
Intermediates in `R16G16B16A16_SFLOAT`; full-screen passes prefer compute; all
post targets engine-owned, resized on window change.

## 14. Input

Two layers (raw events and game meaning change at different rates).

- **Raw collection:** winit `PhysicalKey`/scancode (layout-independent);
  `DeviceEvent::MouseMotion` for raw unaccelerated look (never `CursorMoved`);
  cursor grabbed + hidden. `gilrs` as a parallel gamepad source.
- **Action mapping:** a bindings table (open config format) maps inputs →
  abstract actions. **Button actions** track edges
  (`pressed`/`released`/`held`); **axis actions** are analog (WASD synthesizes
  ±1). Output = one `InputState` resource, platform-independent.
- **Rate seam:** accumulate look deltas + **latch** button edges between sim
  ticks; each fixed tick consumes latched state then clears edges. UI consumes
  input before gameplay via a focus flag (which also releases the cursor grab).

## 15. FPS controller + physics coupling

- **Kinematic character controller** (rapier `KinematicCharacterController`) —
  predictable movement, slopes, step-over, no bounce/tumble. It applies **no
  gravity**: you own vertical velocity (add gravity/tick, zero on grounded, jump
  on edge when grounded).
- **rapier in ECS (raw, not bevy_rapier):** sets/pipeline/params as `Resource`s;
  entities carry `RigidBodyHandle`/`ColliderHandle`. Two sync systems bracket
  the step: ECS→rapier (push kinematic targets), rapier→ECS (write isometry to
  `Transform`). Collision layers via `InteractionGroups` bitmasks.
- **Fixed-step loop (the coupling):**
  ```
  accumulator += frame_dt
  while accumulator >= FIXED_DT {
      snapshot previous_transform for each body     // for interpolation
      input = latched_input.consume()
      step_gameplay(input)                          // desired motion, jump, gravity
      physics.step(FIXED_DT)
      accumulator -= FIXED_DT
  }
  alpha = accumulator / FIXED_DT                     // extract lerps prev↔current
  ```
  Deterministic given identical input/order → replay/netcode-ready later.
- **Camera look decoupled to render rate** (responsive aim) while body position
  interpolates from the fixed step (stable physics). Root-motion, if ever used,
  must sample at sim rate (drives collision).

## 16. Skinned / animated meshes

- **Second vertex layout** (joints+weights) → skinned-opaque / skinned-masked
  pipelines (already budgeted).
- **GPU skinning in the vertex shader**, linear blend skinning baseline
  (dual-quaternion later if twist artifacts matter):
  ```glsl
  mat4 skin = w.x*joints[i.x] + w.y*joints[i.y] + w.z*joints[i.z] + w.w*joints[i.w];
  // joints[j] = globalJointTransform * inverseBindMatrix[j]
  ```
- **Animation sampling** (CPU or compute, only for visible animated instances):
  glTF T/R/S channels → local transforms → hierarchical compose → × inverse-bind
  → joint palette. Baseline: single-clip + **cross-fade** (per-joint lerp);
  state machine later.
- **Buffers:** palette uploaded to a per-frame joint buffer (linear allocator);
  instance carries `joint_offset`. Skinned = own bucket/pipeline. Advance anim
  on the **render clock** (presentation), except gameplay-significant frames
  (hit windows) which key off sim time.

## 17. Assets: formats and bake

**Interchange = open; runtime = one custom format (bake output only).**

- **Import (open):** glTF 2.0 (meshes/materials/skins/anim/scene — already
  metallic-roughness PBR, 1:1 with the material model); PNG/TGA or KTX2
  textures; WAV→Ogg Vorbis audio.
- **Runtime blob (custom, necessity):** BCn textures + mips ready; meshes in the
  exact `Vertex` layout as shared-buffer slices; materials as the 64-byte
  `GpuMaterial`; mmap-and-go, minimal parse. Authored in *nothing* — pure bake
  output; re-bake from open sources if it changes. Keep it a thin
  header + raw GPU-ready payloads (or a stable binary serialization) rather than
  a bespoke parser.
- **Level/scene stays open:** glTF scene carries placements + hierarchy;
  per-node gameplay data in glTF `extras` or an open sidecar (RON/TOML). Bakes
  into a runtime scene blob like everything else.

### Bake tool (`bake/`, standalone CLI, offline)

- glTF import (`gltf` crate).
- Mesh: reinterleave to `Vertex`; tangents via mikktspace if absent; `meshopt`
  (vertex-cache/fetch optimization + LOD simplification); compute sphere + AABB
  bounds; emit slice data.
- Textures: decode → mips → BCn with correct color space per slot → pack ORM.
  Parallelize with rayon (the slow step).
- Materials: glTF MR → `GpuMaterial`; references as content-hash asset IDs.
- Skins/anim: inverse-bind + skeleton + compact clips.
- Scene: placements + hierarchy → scene blob; `extras`/sidecar → spawn data.
- IBL: env → SH irradiance + prefiltered specular; BRDF LUT baked once
  separately.
- **Build behavior:** content-addressed (hash source + settings) → incremental +
  cross-asset dedup; emit a **manifest** (asset ID → blob location) the runtime
  handle table loads; baked output is gitignored build cache, open sources are
  truth; `--watch` is the hot-reload seam.

## 18. Scene spawning + save/load

- **Data-driven spawn via prefab registry:** node = static part (transform,
  mesh/material handles) + gameplay part (`{ prefab, params }` in
  `extras`/sidecar). Runtime `HashMap<PrefabId, SpawnFn>`; loading a scene =
  walk nodes → resolve handles → look up prefab → spawn archetype. New types =
  new spawn fn, not a format change. Chunk membership computed at bake, written
  into the blob.
- **Save/load is separate from scene loading.** Do **not** serialize the whole
  `World` (GPU/physics handles aren't serializable). Mark a **saveable subset**:
  `serde` on plain-data gameplay components + a `Saveable` marker; save queries
  those; load re-spawns from scene then overlays saved deltas. Small,
  versionable, decoupled — and consistent with the replayable fixed step.

## 19. UI / HUD

- **Dev UI → egui** (immediate-mode): stats, entity inspector, tunables,
  debug toggles, cascade/cluster visualizers. Biggest solo-dev productivity win.
- **Game HUD/menus → lightweight custom** renderer: alpha-blended textured quads
  + text, own pipeline, drawn **after tonemap in LDR/sRGB**. May start on egui
  and build the custom path when the look demands it.
- **Text → SDF/MSDF atlas** (crisp at any scale, one atlas; baked in the tool).
- **Input routing:** UI consumes input before gameplay via the focus flag.

## 20. Audio

- **`kira`** — game-oriented mixer (tweens, spatial, streaming); above `rodio`,
  below FMOD/Wwise weight.
- **ECS-shaped:** mixer = `Resource`; emitters carry `AudioEmitter`; one system
  syncs emitter positions + listener (camera) per frame for attenuation /
  panning / doppler. Run on the **render clock** (freshest camera pose).
- **Assets:** WAV source → Ogg shipped; stream music/ambience, decode-to-memory
  short SFX.
- **Zone feel first:** positional 3D SFX + layered ambient beds. Occlusion,
  reverb zones, dynamic music are later refinements.

## 21. Debug / profiling

- **CPU:** Tracy (`tracy-client`), scoped zones on systems + passes.
- **GPU:** timestamp queries (`vkCmdWriteTimestamp2`) bracketing passes → per-
  pass ms in Tracy GPU zones + egui overlay. This is how SSR/volumetrics cost
  gets judged.
- **RenderDoc:** in-application API, capture on a keybind.
- **Object naming:** `vkSetDebugUtilsObjectName` on buffers/images/pipelines
  from the start (readable validation + captures).
- **Validation:** core on debug builds; **sync validation** + best-practices as
  toggles (sync validation essential for hand-written sync2 barriers).
- **Debug draw:** immediate line/shape renderer (bounds, frustums, rapier
  colliders) + fullscreen debug modes via push-constant flags (wireframe,
  cascade tint, cluster-light heatmap, overdraw).

## 22. Crate / workspace layout

Split for compile times, not purity. Keep Vulkan entirely inside `gfx`/`render`
— no `vk::` types leak into `game`.

```
platform   winit, input, action mapping
gfx        ash, vk-mem, device/swapchain/queues, low-level Vulkan
render     frame graph, passes, RenderFrame consumer, materials/bindless
assets     bake formats + runtime loaders + handle tables
game       components, systems, gameplay, controller
bake/      standalone offline CLI (import → runtime blobs)
app        thin binary wiring it together
```

## 23. Build order (milestones)

1. Clear screen (bootstrap, done) → triangle → textured mesh with camera
   (validates buffers, descriptors, depth).
2. ECS-driven scene: extract stage + instanced draw of many meshes from `World`.
3. Lighting: CSM sun + GTAO + a few clustered lights + IBL ambient (the "better
   than 2010" look lands here).
4. Physics + FPS controller (rapier), fixed timestep + interpolation → walkable.
5. Bounded arena content, chunked culling, bindless materials, PBR bake path.
6. Deferred-by-design items as needed: streaming, stage pipelining, GPU-driven
   indirect culling, SSR/volumetrics, probes.

**Smoke-test scene (the debut):** flat PBR ground + shadow-casting bunker/
building (interior overdraw + point lights) + many instanced props (the
instancing thesis, visible) + **water** (transparent bucket, custom material,
pre-transparent resolve — the highest-value test) + masked foliage/fence + one
dynamic light. Walkable in first person. Zone flavor (overcast, ground fog,
concrete/rust) doubles as qualitative feature tests. Use original/CC0 props, not
ripped SoC assets, so a public showcase is clean.

## 24. Reversible-only-with-pain forks (decide before the dependent step)

- **Deferred vs clustered forward** → clustered forward (before step 3).
- **Bindless vs traditional descriptors** → bindless (before step 2).
- **PBR authoring** → commit before building the asset/material layer (fixes the
  texture set + material buffer layout).

## 25. Deferred by design (hooks left in place)

GPU-driven indirect culling; async compute; auto-exposure; dual-quaternion
skinning; animation state machines; local reflection probes / irradiance
volumes; audio occlusion + reverb zones; TAA; streaming + stage pipelining;
X-Ray (`.ogf`/level) importer for the SoC-rebuild stretch dream (becomes just
another importer feeding the same bake).

## 26. Implementation status

Snapshot of what's actually built vs. the design above. The design sections are
unchanged targets; this records reality so the doc doesn't drift. Update as
milestones land.

### Done (milestones 0–2 + basic lighting)

- **Workspace/toolchain**: 6 crates, `rust-toolchain.toml` (stable), GLSL→SPIR-V
  at build time via `shaderc` in `render/build.rs`, embedded from `OUT_DIR`.
- **Deps in use**: ash 0.38, ash-window 0.13, raw-window-handle 0.6, winit 0.30,
  vk-mem 0.4, glam 0.29, bevy_ecs 0.16, serde 1. Not yet added: rapier, kira,
  egui, tracy.
- **gfx**: instance/debug messenger/surface/device/queues; VK 1.3 **dynamic
  rendering** (feature enabled); swapchain + image views + resize; **depth**
  (D32_SFLOAT) created with the swapchain; **vk-mem** allocator held as
  `Arc<Allocator>`; RAII `Buffer`/`Image`/`MappedBuffer` (self-freeing);
  device-local staging upload (`create_device_local_buffer`) and persistently-
  mapped host buffers (`create_host_visible_buffer`); per-frame command buffers;
  sync with per-image `render_finished`; `wait_idle` + ordered teardown
  (resources → allocator → device). `FRAMES_IN_FLIGHT = 2`. Engine-owned
  **HDR scene-color target** (RGBA16F) + linear sampler, created with the
  swapchain and recreated on resize; `draw_frame` runs two passes — geometry
  into the HDR target, then a caller-supplied post pass into the swapchain —
  with the HDR write→read barrier between.
- **render**: instanced `MeshRenderer` — **several meshes share one** device-local
  vertex + index buffer, each a `{first_index, index_count, vertex_offset}` slice
  keyed by `MeshId`; per-mesh indices stay 0-based (rebased via `vertexOffset`).
  Per-frame instance **SSBO** `InstanceData { model, material_id }` (std430, 80 B)
  at set0/binding0 (vertex), plus a **resident materials SSBO** `GpuMaterial`
  (§5: 64 B — base color, metallic, roughness, emissive; `tex` zeroed) at
  set0/binding1 (fragment). `view_proj` + `light_dir` in a push constant,
  **linear-space** ambient + Lambert **directional light** on the material's base
  color, written unclamped to the HDR target, `cull NONE` + depth. Draw sorts
  `(MeshId, instance)` by mesh, uploads in that order, emits **one
  `cmd_draw_indexed` per contiguous run** (`firstInstance` = run start).
  `TonemapPass` — attributeless fullscreen triangle sampling the HDR target,
  **exposure + Narkowicz ACES**, output left linear for the `_SRGB` swapchain to
  encode (no double gamma); exposure via push constant.
- **assets**: Vulkan-free CPU mesh types (`Vertex`, `MeshData` + bounds +
  `Material`), procedural `uv_sphere` + `cube`, and a minimal **glTF/GLB loader**
  (`load_gltf`) — merges all triangle primitives across the node hierarchy with
  transforms, computes normals when absent, reads the first primitive's
  **material** (base color / metallic / roughness / emissive factors), .glb blob
  + external `.bin` (gltf crate, no image/base64 deps).
- **app**: winit loop; bevy_ecs world + multi-threaded schedule (`integrate`,
  `tick`); render-rate **fly camera** (WASD/mouse/Esc, pointer locked); inline
  **extract** (`World` query → sorted `Vec<(MeshId, InstanceData)>` each frame);
  `[`/`]` adjust tonemap exposure at runtime; meshes are a procedural sphere +
  cube by default or one per glTF path on the CLI (each auto-fitted to the grid);
  each entity assigned a mesh + `material_id` at random (glTF meshes use their
  file material; procedural meshes use a shared generated palette).
- **Build order (§23)**: step 1 done; step 2 done; step 3 only partially — basic
  directional lighting, *not* CSM/GTAO/IBL. Steps 4+ not started.

### Current simplifications to revisit

- **Extract seam (§4)**: extract is inline in `app`, writing straight to the
  instance SSBO each frame. No double-buffered `RenderFrame` struct yet; stages
  run sequentially (stage pipelining deferred by design).
- **Timestep (§4)**: sim runs once per frame with a hardcoded `dt = 1/60`. No
  accumulator and **no interpolation** yet — needed before physics. Camera is
  already render-rate (matches design).
- **Descriptors (§9)**: one set with two bindings — per-frame instances
  (binding 0) + resident materials (binding 1), as N discrete per-frame sets.
  Not yet the Set 0 (resident/bindless) / Set 1 (per-frame ring + dynamic
  offsets) / Set 2 (per-view) scheme; materials are a plain SSBO, **not bindless
  textures**.
- **Push constants**: currently carry `view_proj` + `light_dir` (provisional).
  Design reserves push constants for tiny per-draw scalars and puts camera in a
  per-frame UBO — revisit when Set 1 lands.
- **Vertex layout (§6)**: pos+normal only; tangent/uv deferred until textures.
- **Lighting (§3, §13)**: now **linear-space** ambient+Lambert into an RGBA16F
  HDR target, resolved by an ACES **tonemap** pass — the color-management seam is
  correct. Still **not** the full baseline: no PBR BRDF and no IBL yet, exposure
  is a fixed constant (no auto-exposure), and no bloom. The tonemap curve is a
  drop-in point for AgX. This unblocks bloom/SSR/transparents, which all read the
  resolved HDR.
- **HDR/depth targets (§9)**: single engine-owned images shared across both
  frames-in-flight (matches the design: render targets are engine-owned, not
  per-frame). With `FRAMES_IN_FLIGHT = 2` this carries a latent cross-frame WAW
  hazard on the shared targets — pending the sync2 barrier + timeline-semaphore
  pass (§10). The intra-frame HDR write→read hazard *is* handled.
- **Geometry / assets (§6, §7)**: a **mesh registry** and a **material table**
  now exist — several meshes in shared vertex/index buffers drawn by sorted
  per-mesh runs, and a resident `materials[]` SSBO indexed by per-instance
  `material_id` (base-color/metallic/roughness/emissive factors from glTF or a
  generated palette). Still missing: **textures + bindless**, pipeline buckets
  (one pipeline for everything), UVs/tangents, a PBR BRDF (shading is Lambert on
  base color — metallic/roughness are stored but unused), skinning/animation, and
  the offline **bake** path (runtime blob, handle tables, load-time allocator).
  glTF loads directly each run; buffers are never freed/streamed; one material
  per merged glTF (first primitive wins).

### Not yet started

Culling; textures + bindless + PBR BRDF + pipeline buckets (material factor
table landed); shadows (CSM); clustered lighting; IBL; bloom + auto-exposure
(HDR target + tonemap now in place); transparents; asset bake pipeline (runtime
glTF + multi-mesh registry landed); scene format / spawning / save; physics + FPS
controller (rapier);
skinning; UI/HUD; audio; debug/profiling tooling (Tracy/RenderDoc/timestamp
queries); GPU-driven culling; streaming; stage pipelining.
