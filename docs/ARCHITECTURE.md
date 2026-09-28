# Feather — Engine Architecture

Status: in active implementation — a textured-PBR forward renderer with IBL,
4-cascade sun shadows with caster pancaking, clustered punctual point lights
with sphere-light specular, ambient light occluded by a baked sky-visibility
volume and GTAO, a depth prepass, per-view frustum culling,
MSAA/FXAA, mipmapped textures (deduplicated, a per-level bindless array, and
an offline BC7 bake), GPU per-pass timing, a rapier FPS controller in the ECS
with collision proxies for detailed props, glTF scene loading with §18
prefabs, and a main menu + pause menu with an options tree are up;
validation, including sync validation, is clean; see §26 for exactly what's built vs.
still designed. How to build, run, test and measure it: [USAGE.md](USAGE.md).
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
- **Testable without a GPU.** The part of each system that decides what
  happens sits behind a seam that needs no device or window:
  - the controller as plain functions under its ECS systems (§26);
  - menus that only return outcomes (§19);
  - a mixer generic over its audio backend (§20);
  - `build_world`, the CPU half of starting a session (§26).

  `cargo test` covers those, down to loading a glTF file into the game's own
  world and running its schedule. Only what reaches the GPU is left to
  validation and to eyes (USAGE.md §5).

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
| Normal     | linear      | BC7         | 3-channel; BC5 measured and rejected (§17) |
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

(As built: `tex.w` is flags in bits 0–7 and the occlusion texture's slot
above them; there's no emissive texture yet. `params.z` is the occlusion
strength. See material AO in §13.)

CPU-side only (routing, not shading): alpha mode (opaque/mask/blend),
double-sided, cull — used to pick pipeline and draw bucket.

**Landed (§26): alpha cutout.**
- **Loader:** it reads glTF `alphaMode`, `alphaCutoff` (default 0.5) and
  `doubleSided` into `Material`.
  - MASK materials cut out.
  - BLEND materials still draw opaque, since there's no transparent pass;
    the load prints how many.
  - On the GPU, `params.w` is the cutoff (0 unless MASK), and `tex.w` holds
    flags (bit 0: double-sided).
- **Draw order:** runs sort with the masked bit on top of the
  `(mesh, LOD)` key, so each view's list is all opaque runs, then all masked
  ones. Each pass binds its opaque pipeline, draws, and switches once.
  Scenes without masked materials record exactly the runs they did.
- **Three masked pipelines,** each with the same layout and descriptors:
  - **Depth prepass:** `mesh.vert` + `mask.frag`, which only cuts. Same
    vertex shader, so depths still match the main pass bit for bit.
  - **Shadow:** `mesh.vert` + `mask.frag`. `mesh.vert` reads its matrix
    from the first 64 push-constant bytes, which is where the shadow pass
    pushes the light's, so no new vertex shader was needed. It has **no face
    culling**, unlike the opaque shadow pass (front-face culling): a card is
    usually one quad, which would drop out of the map whenever it faced the
    sun.
  - **Main:** `mesh.frag` specialized with `MASKED` (constant ID 100).
    - It has to cut too: where the prepass cut a hole, the depth behind it
      passes `LESS_OR_EQUAL`.
    - It **demotes** rather than discards (Vulkan 1.3's mandatory
      `shaderDemoteToHelperInvocation`, now enabled). Normal mapping and mip
      selection still need derivatives in the cut fragments' quads, which
      `discard` would leave undefined.
    - Opaque materials keep a pipeline with no cut at all, and so their
      early-Z.
- **The alpha test** is `cutout_alpha()`, written out in both shaders (a
  test keeps the texts identical). It's base-colour alpha × factor, scaled up
  by `1 + 0.25 · mip level`: box-filtered mips average alpha down, so without
  that a cutout thins away with distance. It works the same for raw and baked
  (BC7 with alpha) textures.
- **Double-sided:** a back face of a double-sided material flips its normal,
  which also fixes its shadow lookup's normal offset. Culling stays NONE for
  everything, as before.
- **Measured** (debug, clocks unpinned, 3 interleaved runs):
  - `lights120`, which has no masked materials: `geo` 0.58 ms with HEAD's
    binary, 0.56–0.57 with this one. No change, as predicted.
  - The zone's 600 grass tufts and chain-link enclosure (up to 451 masked
    instances in view, 519–833 casters a frame) cost `shadow` +0.02 ms
    (predicted +0.02–0.05) and `geo` ≤ 0.01 ms. I'd predicted +0.05–0.15;
    they're cheap cards.
  - Validation with sync: 0 messages with masked draws, and the new
    `[bench] masked main/shadow` lines show they ran.
- **Not done:** alpha-to-coverage under MSAA, transparency for BLEND,
  sorting masked draws front to back.

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

**Landed (§26): normal transform.** `mesh.vert` transforms normals by the
**cofactor** of `mat3(model)` (three cross products; it equals det × the
inverse-transpose, and the magnitude drops out in `normalize`), negated when
det < 0 so mirrored nodes keep outward normals. It is computed per vertex
rather than stored in the instance record as a normal matrix, as originally
planned: ~15 ALU ops per vertex instead of +48 B per instance (80 → 128) and a
CPU inverse per instance. Measured on `detail_high` (3 interleaved runs,
pinned clocks): `geo` median 1.02 ms both ways. The old `mat3(model)` was wrong
under *any* non-uniform scale, not only rotated + scaled: an unrotated
(3, 1, 1) scale tilts a 45° normal by 53°. It only held because every
non-uniformly scaled node was a box, whose normals lie on the scale axes.

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

**Landed (§26):** the per-view **narrow** step — a `Frustum` (six Gribb–Hartmann
planes from any `viewProj`) tests each entity's conservative bounding **sphere**
(center = interpolated position, radius = `√3/2 · max(scale)` from the unit-cube
fit). Run inline in the app's extract loop, **per view**: the camera frustum trims
the main pass and the sun light ortho trims the shadow pass, each producing its own
instance set. Still **single-threaded** (no rayon fold-reduce), **sphere-only** (no
AABB refinement), and with **no broad phase / chunk culling** yet (LOD landed in
§17, chosen in the renderer rather than here). Note the
observed behavior: at a dense fragment-bound view the *frame time* barely moves
(the culled objects were already off-screen — zero pixels), so the win is in
vertex/draw work and shows up when the view is sparse or geometry/overdraw-bound;
the shadow pass (vertex-bound) would benefit most but its casters are all inside
the fixed light frustum today.

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

**Landed (§26):** steps 2, 3, 5 and 6 in a reduced form — a sun shadow pass, then
one geometry pass containing **depth prepass** → opaque → sky. The prepass (step 3) is
depth-only over the camera-culled set, after which the opaque pass runs with
**depth-write off and `LESS_OR_EQUAL`**, so early-Z discards occluded fragments
instead of shading them. Measured **geo 5.2 ms → 3.0 ms (−42 %)**, frame 10.3 → 8.0 ms
in a back-to-back A/B. Two implementation notes worth keeping: the prepass reuses
**`mesh.vert`, the same module as the opaque pass**, so `gl_Position` is
bit-identical and the depth test always matches (a separate depth-only shader
associating its matrix multiply differently would round differently and drop
fragments); and because it runs inside a render pass that has a colour attachment,
its pipeline must declare that **same attachment count** with an **empty colour
write mask** — omitting the attachment is a validation error.

The **sky** (step 6) then draws last, as a far-plane (`z = 1.0`) fullscreen
triangle with `LESS_OR_EQUAL` and no depth write, so it survives only where the
depth buffer is still the cleared 1.0 and shades just the visible background
instead of the whole screen — **geo 3.0 → 2.8 ms**. It shares `fullscreen.vert`
with the tonemap pass, which is unaffected because that pass renders with no depth
attachment. Still pending here: no MSAA resolve or transparent pass. (The
cluster pass landed — §12 — as a compute dispatch between shadow and geometry,
guarded by a legacy `vkCmdPipelineBarrier` until the sync2 pass.)

**Landed (§26): the prepass on its own.** GTAO (§13) needs the depth before
anything is shaded, so the geometry pass is now three steps:
1. **Depth prepass**, a depth-only rendering. Depth is stored, and under MSAA
   also resolved (`SAMPLE_ZERO`, always supported) into a single-sample D32
   image. Its pipelines lost their empty colour attachment, since the
   rendering has none, and still share `mesh.vert`.
2. **GTAO**, compute, reading that depth in `DEPTH_STENCIL_READ_ONLY_OPTIMAL`.
3. **Main pass** (opaque + sky), which loads the depth, tests it and never
   writes it. At 1× it uses that same image in the read-only layout, under
   MSAA the multisampled buffer.

`geo` still times steps 1 and 3; GTAO has its own `ao` slot. Measured (debug,
clocks unpinned, 3 interleaved runs against the old binary, GTAO off): `geo`
0.21–0.23 → 0.22–0.23 ms on the zone and 0.64 → 0.62 ms on `lights120`. I
predicted at most +0.02 ms, and it held.

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

**Landed (§26):** **4 cascades**, the design above less PCSS (pancaking landed
later; see the end of this section). One
D32 depth image with one **array layer per cascade** (`SHADOW_CASCADES = 4`),
`gfx` rendering a depth-only pass per layer and `mesh.frag` sampling all of them
through a `sampler2DArrayShadow`. Splits use the practical scheme (λ = 0.75) over
`[0.1, SHADOW_DISTANCE]`, each cascade **fitted to the bounding sphere** of its
frustum slice and **texel-snapped** (the ortho center is quantized to whole shadow-map texels in
light space, so edges don't crawl as you walk; the along-light depth needs no
snap). Centering on the player rather than the view direction means turning the
camera never disturbs the map. Dense texels matter — a wider frustum or lower
resolution looks pixelated. A depth-only shadow pass (`shadow.vert`,
front-face cull + slope-scaled
`vkCmdSetDepthBias`) renders each cascade's own culled caster set; the mesh fragment shader samples it with a comparison sampler and
**3×3 PCF**, occluding the **direct sun term only** (ambient/IBL stays lit). The
light-space matrix rides a per-frame globals UBO (set 0 binding 4) since it won't
fit the 96 B push constant. Casters are frustum-culled against the light ortho
(§8), which only bites once the moving frustum leaves them behind, and an opt-out
`NoShadowCast` marker drops individual entities from the caster set. The demo's
flat ground carries it: a flat slab casts nothing useful but rasterizes the whole
shadow map — measured at **~1.9 ms, 29 % of the shadow pass**. This is per-entity
by design; terrain with relief must cast (hills shadow valleys) and simply omits
the marker, which also means that cost returns then. Note the shadow pass is
substantially but **not purely fill-bound** — see the measured breakdown under
the cascade notes below, which corrects an earlier claim here that cost tracks
texels alone. Shadow resolution is still a quality knob, like AA (§13). Three implementation notes worth keeping.

**Selection is by projection containment**, not by view-space depth against the
splits: the fragment shader has no view matrix, and the push constant is already
96 B, so adding one would pass the 128 B floor Vulkan guarantees. Testing each
cascade's box in order needs neither and is correct by construction, since
cascade 0 is the tightest. A small band at each cascade's edge blends into the
next so the resolution change is a gradient, not a line.

**The sphere fit is what stops shimmer.** Centroid and radius are invariant
under rigid motion, so turning cannot change the world size a cascade covers; a
box fit would resize as you rotate and the texels would crawl with it. A unit
test sweeps a full turn and asserts the radius holds.

**Measured cost, and a corrected assumption.** 4×2048² is exactly the texel count
*and* the 64 MiB of the single 4096² map it replaced, so on the old "fill-bound"
assumption the pass should have cost the same. It did not: **0.78 ms → 1.48 ms**
(1.89×) on the test scene at near-full-screen, with `geo` 1.73 → 2.19 ms and the
frame 3.02 → 4.10 ms.

The main menu isolates why. With **no casters at all**, both configurations clear
the identical 16.78 M texels, yet four passes cost 0.87 ms against one pass at
0.48 ms. That +0.39 ms cannot be fill; it is **fixed per-pass overhead**, ~0.13 ms
a pass. The remaining +0.31 ms is casters drawn into several overlapping
cascades. So the pass is substantially fill-bound but per-pass cost is a third of
it at this resolution — which is why `set_shadow_casters` collapses the four
passes to a single layered clear when nothing will be drawn, and why
**multiview** (one pass, `gl_ViewIndex` selecting the cascade) is the obvious
follow-up. Multiview would trade away per-cascade culling, so it is worth
measuring rather than assuming.

**Range is paid for in sharpness.** The practical split scheme equalises texel
density in *screen* space — every cascade lands at a similar texels-per-pixel
ratio — so spreading a fixed budget over ~4x the distance makes the mid-field
coarser than the old over-dense ±16 map: sharper than before inside 4 units,
~1.4× coarser from 4–9, ~3.1× coarser from 9–20, and shadowed at all from 20–60
where previously there was nothing. Shortening `SHADOW_DISTANCE` does **not**
recover the mid-field (at 30 the 5–11 band comes out *worse*, 1.7×), because the
scheme just re-shuffles which cascade covers where. **Per-cascade resolution is
the only real lever**, and it costs fill linearly.

**Landed (§26): caster pancaking.** Each cascade's light ortho starts
`radius + SHADOW_BACK` (r + 40 m) up-light of its sphere. A caster further
toward the sun was lost twice over: the light frustum's near plane culled it,
and the rasteriser would have clipped whatever got through. Now casters are
culled against the cascade frustum **with its near plane dropped**
(`Frustum::without_near`). The side planes are parallel to the light, so
anything outside them could never shadow the box and stays culled. The shadow
pipeline enables **hardware depth clamp**, which flattens up-light geometry
onto depth 0 so everything behind it compares as shadowed. That is exact per
fragment, unlike clamping z in `shadow.vert`, which distorts triangles that
cross the near plane. `depthClamp` is optional in core Vulkan: it is enabled
when supported, and otherwise startup warns and the old clipping stays.
`SHADOW_BACK` stays 40 even though it now only spends depth precision,
because `SHADOW_DEPTH_BIAS` is in normalised depth over `back + radius`, so
shrinking the range would silently shrink the world-space bias.

The test level's 70 m tower at (30, 30) exercises it: its top is ~78 m
up-light, which clips cascades 0–2 (radii ~5 / 11 / 25 m) but not the 73 m
cascade 3. So before the fix its shadow's far end vanished exactly where you
stood on it, near spawn, and reappeared from further away. A unit test builds
the real cascades with the player on that shadow tip and asserts both the bug
and the fix. Cost: none measurable (`--bench` identical to 0.01 ms).

**Pending:** PCSS; and per-cascade resolution variation (array
layers must share an extent, so §11's "far cascade 1024²" is not expressible in
a single array — the sphere fit already gives far cascades more world per texel,
which is that line's intent).

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

**Landed (§26): punctual lights (stage A), then clustering (stage B, below).** Stage A of the above:
a `PointLight` component placed by §18's `point_light` prefab (on a marker node,
or on geometry so a lamp both emits and renders), extracted per frame into a
`GpuLight[]` SSBO at set 0 binding 5, and shaded in `mesh.frag` with the same
Cook-Torrance terms as the sun — reusing the BRDF rather than adding a second
lighting path. Falloff is windowed inverse-square, the window driving the light
to *exactly* zero at its radius so the cutoff is not a visible sphere edge.
Lights are culled per frame against the camera frustum **by sphere, not point**,
so one whose centre is off-screen still lights what is on-screen.

Stage A tested **every visible light per fragment**, with a distance early-out.
The reasoning then was that cost tracks lights-*in-frame*, not
lights-touching-*this-pixel*, and that this was the whole case for the grid. The
stage-B measurement below shows that was mostly wrong for the test scene. The
component, prefab, extract, SSBO and BRDF all carried over unchanged; only the
loop's source changed.

**Measured — the baseline stage B has to beat.** `geo` on the test scene with
`--lights N` scattered as geometry-free markers, so light count was the only
variable (`shadow` held flat at ~1.5 ms across all three, confirming it):

| lights | `geo` median | spread |
|---|---|---|
| 0 | 2.39 ms | 1.6 ms |
| 60 | 5.53 ms | 5.0 ms |
| 120 | 9.97 ms | 7.1 ms |

4.2× at 120 lights, and **superlinear** — ~52 µs per light for the first 60,
~74 µs for the next — because more lights survive the frustum cull and overlap.
The spread is itself the symptom: at 120 lights `geo` swings 5.1–12.3 ms purely
with view direction, because cost tracks lights *in frustum* rather than lights
*touching the pixel*. Clustering is precisely what converts that into a stable
per-pixel cost, so this is the case for building it, backed by a number rather
than assumed. *(Stage B, below, found this only partly true: most of the growth
was lights genuinely overlapping, which no grid removes.)* *(Measured on the original dev laptop; re-baseline on new hardware
before comparing — absolute ms do not transfer between machines.)*

**Re-baselined on the desktop (RX 7800 XT)** with `--bench` (§21): 1920×1045,
360° sweep at the spawn point, 720 raw per-frame samples, GPU clocks pinned
with `profile_standard`, debug build (release measured identical):

| lights | visible (median) | `geo` median | p10–p90 |
|---|---|---|---|
| 0 | 2 | 0.19 ms | 0.16–0.23 |
| 60 | 34 | 0.45 ms | 0.37–0.53 |
| 120 | 66 | 0.76 ms | 0.62–0.88 |

The **ratio carries over** — 4.0× at 120 lights against the laptop's 4.2×, still
superlinear (~8.1 µs per visible light for the first 32, ~9.7 µs for the next)
— while the absolute cost is ~13× lower. "0 lights" is not zero: the base
scene's own `point_light` markers put up to 6 in view. Without pinned clocks
the same sweep read 3.3× and 0.52 ms at 120: the idle GPU downclocks, which
inflates light workloads more than heavy ones (see §21). On this hardware the
whole light loop is well under a millisecond, so stage B's case rests on
laptop/Deck-class GPUs, and its A/B here must account for a compute dispatch's
fixed cost, which at these sizes may not be negligible.

`MAX_LIGHTS = 128`, clamped with a warning. `GpuLight` is 32 bytes:
`pos_radius` and `radiance_source` (rgb = colour × intensity, premultiplied on
the CPU to free the alpha for the source radius). §12's `dir_cone` and `type`
are omitted rather than padded, since unused fields cost bandwidth every
frame.

**Landed (§26): stage B, the cluster grid.** 16×9×24, exponential slices over the
camera's own near/far (0.1 / 200 — `CAMERA_NEAR`/`CAMERA_FAR` in the app, shared
with the projection so the two cannot drift). One compute dispatch per frame
(`cluster.comp`, one thread per cluster, 64-thread groups) — the engine's first
compute pipeline — runs in its own `draw_frame` slot between the shadow and
geometry passes, is timed as `[gpu] … cluster`, and is followed by a
compute-write → fragment-read barrier. It reuses the mesh renderer's set 0 and
pipeline layout (masks at binding 6). `pick_device` now requires
`GRAPHICS | COMPUTE` in one family, and `maintenance4` is enabled because glslang
emits `LocalSizeId` for Vulkan 1.3 compute (mandatory in 1.3, so no device is
excluded). Three deliberate departures from the design above:

- **A per-cluster bitmask, not `{offset,count}` + `lightIndexList` + an atomic
  counter.** With `MAX_LIGHTS = 128` a mask is *exact* — 4 words per cluster,
  55 KB per frame in flight — so there is no per-cluster overflow to clamp, no
  atomics and no counter reset, and the fragment visits lights in ascending
  order (`findLSB`). It grows linearly with the cap. Revisit only if lights
  reach the thousands.
- **No cached AABB pass.** Each thread builds its cluster's AABB inline (a few
  ALU ops), which removes a buffer, a pass, and the resize/FOV invalidation
  path.
- **Tiles defined by view-space slope, not `gl_FragCoord`.** The fragment derives
  its tile from x/d, y/d against tan(fov/2), exactly as the compute side bounds
  it. There is one shared definition, no framebuffer size in the UBO, and no
  one-frame skew during a resize. The AABB faces are padded slightly
  (`SLOPE_PAD`, `DEPTH_PAD`) because the two sides reach boundaries through
  different float ops.

**Correctness is exact, and was checked as such.** A light outside the cluster
contributes exactly zero (the falloff window reaches 0 at the radius), so with
conservative clusters the clustered sum *is* the brute-force sum. A temporary
harness ran the brute loop alongside in `mesh.frag` and atomically counted
contributing-but-unmasked lights over every fragment of the full `--bench` sweep:
**0** at 60 and at 120 lights. The negative control (radius halved in the compute
test) reported 2.9 billion. Under sync validation, removing the new barrier raises
`SYNC-HAZARD-READ-AFTER-WRITE` on the mask buffer; with it, stage B adds no
hazards (pre-existing ones: §21).

**Measured (RX 7800 XT, `profile_standard`, `--bench`, back-to-back A/B), and a
premise corrected.** `--lights N` (r = 14, heavily overlapping):

| lights | `geo` stage A → B | `cluster` pass | `frame` A → B |
|---|---|---|---|
| 0 | 0.19 → 0.20 | 0.00 | 0.32 → 0.35 |
| 60 | 0.45 → 0.40 | 0.02 | 0.59 → 0.56 |
| 120 | 0.76 → 0.61 | 0.03 | 0.89 → 0.79 |

Only −20% at 120, against a predicted ~−50%, because **cost tracks lights that
actually contribute, not lights tested.** The harness counted 4.11 / 8.46
contributing lights per fragment at 60 / 120. After clustering, the light cost
(`geo` minus the 0-light run) is 0.048 ms per contributing light at *both*
counts, so it is linear. Stage A was ~0.065, meaning the brute loop's rejects
were only ~25% of its light cost, and that 25% is all clustering removes. Stage
A's "superlinear" growth was mostly real overlap (8.46 > 2 × 4.11), not wasted
tests. The clusters are ~2.6× conservative (22.2 lights walked per fragment vs
8.46 contributing). Tightening them would only trim the cheap reject part.

Clustering pays where lights-in-frame ≫ lights-touching-the-pixel. Same scene
with the scattered lights shrunk to r = 4 (≈1 per pixel): `geo` 0.37 → **0.22**
(−41%), p10–p90 narrowing from 51% to 41% of the median. (That run predates
`--light-radius` and used a hand-edited copy, which also shrank the scene's two
authored r = 14 lights: 2 of 124 lights. Regenerate it with
`--lights 120 --light-radius 4`.) `--lights` at its default r = 14 (~8.5 per
pixel) is deliberately clustering's *worst* case. With dense overlap,
the next cost to attack is the BRDF itself, not assignment.

**Landed (§26): sphere-light specular (Karis 2013, representative point).** A
true point light on smooth metal (roughness clamps at 0.04, α = 0.0016) makes a
near-singular, sun-bright dot. Each light now has a `source_radius` (prefab
param, default 0.1 m, clamped to [0, radius]). Specular is lit from the point
on that sphere closest to the reflection ray, which flattens D into a disc the
size of the source. D at the *original* α is then scaled by (α/α')², with
α' = α + src/(2d). Diffuse keeps the centre direction. **Falloff stays on the
centre distance**, so a light still reaches exactly zero at its radius and the
stage-B clustering stays exact. The generator gives its emissive orbs their
real 0.7 m, so a mirror sphere's highlight matches the orb's reflection.

Two corrections to the textbook form, both found by integrating the lobe in a
CPU reference (`sphere_light_*` tests, which pin these properties):
- **D must stay at α.** Evaluating D at α' *and* multiplying by (α/α')² widens
  twice and kept ~1% of the energy.
- **The widening constant is `2d`, not the `3d` some write-ups give.** Half the
  source's angular radius, since half-vectors move half as far. At `/2` the
  energy is within ±12% of the point light at roughness 0.3 and +15–49% on
  mirror-smooth metal, the approximation's known looseness. `/3` overshoots up
  to 3.3×.

The highlight half-width is 0.9–1.3× the source's angular radius (tested). src
= 0 reduces to the old point light (tested against a transcription of it).

**Measured cost** (A/B, `profile_standard`): +0.16 ms `geo` at `--lights 120`
(0.61 → 0.77), but +0.01 ms at `--light-radius 4` (0.22 → 0.23). The cost is per
*contributing* light (~0.019 ms each at full screen, ~+40% on the per-light
BRDF): the extra `length` + `normalize` are real work. Two cuts are already in:
`reflect(-V, N)` is hoisted per fragment, and a whole-sphere-below-horizon test
(`dot(N, delta) <= -src`) runs before the representative point. Together they
recovered 0.05 ms of a first +0.21. It is expensive only where ~8.5 lights
overlap every pixel.

**Pending:** spot lights (cone-vs-AABB in `cluster.comp`); shadowed point lights
(cube maps).

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
           + prefiltered(R, rough·maxMip)·(F0·brdf.x + brdf.y)) · min(ORM.ao, GTAO)
  ```
  Added to direct sun+cluster. GTAO modulates **ambient only**. (The design
  said `ORM.ao × GTAO`; material AO landed as their `min`, below.) Consumes exactly
  the material data already defined. All bake-time; runtime cost ≈ a few texture
  reads. Later: local probes per-object for interiors; irradiance volumes for
  dynamic bounce.

**Landed (§26): sky visibility occludes the ambient.** The formula above has
no term for a sky that isn't there. So a room lit only through its door got
the whole sky's light on every wall, and the zone's hangar and office looked
as bright inside as the yard. The level's baked sky-visibility volume (§17)
now scales the sky term. GTAO (below) adds the small-scale contact.
- **Shading** (`mesh.frag`, mirroring `SkyVis`, whose tests pin it):
  - Diffuse is `ground·(½ − ½n.y)·2w0 + sky_avg·max(w0 + w·n, 0)`: the sky
    part by its directional weight, and the ground's bounce by the fraction
    of sky seen. A floor lit through a door has a lit ceiling only as far
    as the door lets light in.
  - Specular IBL is multiplied by `SkyVis::specular(r)`: the sky weight
    round the reflection vector against open sky's, blended into the
    ground-bounce fraction below the horizon over 0.25 in `r.y`.
  - With open sky both are exactly the old terms, so a level without a
    volume shades as before (`sky_dims.w = 0` skips the lookup).
- **Sampling** (`sky_visibility`, mirroring `SkyVolume::sample`):
  - The point sampled is one cell off the surface along its geometric
    normal.
  - Eight `texelFetch`es of an RG32_UINT 3D image at binding 7, each the
    `SkyVis` moments and the `SkyFree` nibbles.
  - The cells blend by hand, leaving out those whose path to the point is
    blocked (the leak fix, §17).
  - Outside the grid it fades to open sky over one cell. The volume's
    placement rides in `Globals` (`sky_origin`, `sky_dims`).
- **Loading:** the first CLI scene's volume, found by its `sky_key` under
  `scratch/bake/sky/`; `[sky] visibility: …` at load says so. Without one,
  `[sky] no sky visibility baked …` suggests the bake. `--no-bake` or
  `--no-sky-occlusion` skip it.
- **Measured** (debug, clocks unpinned, 3 interleaved runs,
  `--no-sky-occlusion` vs on, sync validation 0 messages throughout):
  - `geo`: zone 0.17 → 0.20–0.21 ms, `lights120` 0.56–0.57 → 0.63 ms, and
    frames +0.04–0.05 ms.
  - I predicted +0.02–0.06 ms. The zone held; `lights120` landed at the top
    edge and just over it.
  - Bench-sweep exposure: the zone median stayed at 1.36 and `lights120`
    moved 1.00 → 1.02. Both predictions (≤ 10% and ≤ 3%) held, but the zone's
    says the sweep, from the gate, hardly looks into a building.
  - From inside (a temporary start-position override): hangar 2.88 → 3.28,
    office 4.07 → 5.78. **I predicted several-fold, up to the cap of 8, and
    was wrong.** The lamps, the door and the windows carry most of the light
    the meter sees there.
  - To check the GPU really applies the volume, a temporary shader wrote
    only the occlusion ratio (occluded over open irradiance). The meter then
    read ratios of ≈ 0.82 at the gate, ≈ 0.075 in the hangar and < 0.07 in
    the office (clamped). The CPU volume's values are ≈ 0.8, 0.04–0.11 and
    0.006, and with `--no-sky-occlusion` every spot metered alike.
- **Re-measured with pinned clocks** (`profile_standard`, release, 3
  interleaved rounds, `--no-sky-occlusion` vs on): `geo` 0.24 → 0.29 ms on
  the zone and 0.82 → 0.92 ms on `lights120` (+12%). I predicted
  +0.02–0.05 ms: the zone held at the edge, `lights120` didn't. Its 25 MB
  volume (below) was the first suspect: 8 fetches a fragment spread across it
  can miss the cache much more than in the zone's 3 MB.
- **Found: it was latency, not the volume's size.** At a2c5e27 (pinned,
  release, interleaved rounds; `geo` with it off: 0.26–0.27 zone / 0.86
  `lights120`, on: 0.31 / 0.96):
  - **All 8 fetches from one cell** (the maths kept): 0.28 / 0.89. So the
    fetches' addresses cost ~0.07 ms of `lights120`'s 0.10, and the maths
    little.
  - **Fetches wrapped into a 16³ block,** 32 KB and cache-resident but still
    spread: 0.31 / 0.96, unchanged. **I predicted ≈ 0.88 and was wrong:** it
    isn't cache capacity, so a sparse grid wouldn't help the time.
  - **4 fetches instead of 8:** 0.31 / 0.95. I predicted 0.92–0.93: wrong
    again; fewer fetches barely help.
  - **Latency is left.** `sky_visibility()` ran after the light loop, and
    its blend waits on the loads at once, with nothing to overlap them.
  - **Sampling it right after the normal** is known, before the lights
    (predicted 0.89–0.91, held): 0.30 / 0.90. The loads' latency now
    overlaps the albedo and metal-roughness samples.
  - Earlier still, before those samples: 0.30 / 0.93, worse.
- **Now** (3 interleaved rounds against a2c5e27):
  - `lights120` `geo` 0.96 → 0.90–0.91 ms, frame 1.43 → 1.37–1.38 ms; the
    zone 0.31 → 0.30 and 0.75 → 0.73–0.74 ms.
  - On vs `--no-sky-occlusion`: +0.04 ms (zone) and +0.06 ms (`lights120`),
    from +0.05 / +0.10.
  - `mesh.frag` keeps 72 VGPRs and 20 waves.
  - The off path read 0.84 ms against 0.86 before, which is within
    §13's register-allocation noise (the fog finding).
- **Memory:** 8 bytes a cell. That's 3.0 MB for the zone, but 25 MB for
  `lights120`, whose occluders reach 70 m up: its 3.3M cells are nearly all
  open sky. A sparse or two-level grid would fix the memory; it wouldn't
  change the time (above). Nothing needs it yet.
- **Limits:**
  - one volume per session, the first scene's;
  - static geometry only (a moving object neither occludes nor, beyond
    sampling where it is, is occluded correctly);
  - cutouts don't occlude;
  - lamps and the sun are direct light and unaffected;
  - thin walls within a cell of each other can still exchange a little
    light where the per-axis distances miss a diagonal gap.

**Landed (§26): material AO** (glTF `occlusionTexture`). A material's own
texture-scale occlusion (mortar lines, corrugation grooves, the seams of a
barrel), for the ambient only, as glTF specifies.
- **Content:** Poly Haven's ARM maps carry AO in red, but their glTFs
  don't reference it.
  - `gen_zonescene.py` names the ARM map as each tiling material's
    `occlusionTexture`.
  - `gen_detailscene.py`'s import does the same for any model whose MR image
    is a `*_arm_<res>k` file with no occlusion of its own.
  - The zone's `--check` requires all 25 of its MR materials to carry one.
  - Their red channel means run 0.77 (rusty metal) to 0.98 (concrete floor)
    for the tiling textures.
- **Loader and bake:** `Material` gains `occlusion_texture` and
  `occlusion_strength`. An ARM map is one image for both, so one `Arc`, one
  BC7 bake (all four channels) and one bindless slot.
- **GPU:** `tex.w` holds the flags in bits 0–7 and the occlusion slot above
  them (`MATERIAL_OCCLUSION_SHIFT` = 8); `params.z` is the strength.
- **`mesh.frag`:**
  - it samples MR as `.rgb`;
  - if the occlusion slot is MR's, it's that sample's red, at no extra
    fetch;
  - slot 0 (none) is 1; any other slot gets a sample of its own;
  - then glTF's `1 + strength·(sample − 1)`.
- **With GTAO: the `min`, not the product** (as Filament does). The
  models' baked AO already holds some of what GTAO sees (under the car,
  inside the tyre), and multiplying would darken those crevices twice.
  With GTAO off, it's the material AO alone.
- **Checked on the GPU:** a temporary harness wrote the material AO to the
  HDR target from the three zone views.
  - The means were 0.89–0.93, against the textures' 0.77–0.98.
  - Brick mortar, corrugation grooves, the asphalt's patches and the tyre's
    tread were all visible.
  - Materials without occlusion read exactly 1.
- **Measured** (pinned, release, 3 interleaved rounds against 0954247):
  - the zone `geo` 0.30 ms either way, frames 0.73–0.74;
  - `lights120` `geo` 0.90 → 0.93 ms, frames 1.37 → 1.40, though none of its
    materials has occlusion.
  - I predicted ±0.02 ms for both. The zone held; `lights120` didn't.
  - The bench-sweep exposure on the zone moved 1.36 → 1.38 (darker
    ambient).
- **Chasing the `lights120` cost** (each variant measured):
  - It isn't occupancy: 72 VGPRs and 20 waves throughout.
  - It isn't the extra branch: the rule reduced to `arm.r` or 1 costs the
    same 0.93.
  - It isn't the fetch order: the code before the sky fetches is the same
    ISA as before.
  - Compiling the occlusion out (`material_ao = 1`) gives back 0.90. So
    it's one more value live across the light loop, which is `lights120`'s
    hot spot.
  - The light loop's bank conflicts *fell* (27 → 21) while it slowed, so
    that model doesn't explain this one (the fog finding's caveat).
  - Accepted: the zone doesn't pay it.

**Landed (§26): GTAO** (Jimenez et al. 2016, after Intel's XeGTAO): contact
shadowing within 0.8 m, which the sky volume's 0.5 m cells, sampled a cell off
the surface, can't see. It uses the depth prepass alone (§10).
- **Half resolution, without an averaged depth.** Half-resolution texel
  `h` *is* full-resolution pixel `2h`: its position and normal come from the
  full-resolution depth, so they're exact, and on a plane every blend weight
  is 1. (A downsampled depth would have to pick one of each 2×2, a bias the
  four failures below show is easy to reach.)
- **`gtao_depth.comp`:** point-sampled depth levels for gtao's longer steps.
  Level `k` (1–4, ½ to 1/16 resolution) holds pixel `2^k·g` at texel `g`,
  sized `ceil(W/2^k)`: exact depths, never averaged, so interpolating
  between them is still exact on a plane. One pass over the half grid, each
  thread writing its pixel to every level whose grid it's on. They're the
  mips of one R32F image, whose base is padded because mip sizes round
  down.
- **`gtao.comp`** (`render::AoPass`), over the half-resolution grid:
  - Normals are rebuilt from depth.
  - 2 slices × 4 steps each way, both jittered by 4×4 patterns over the
    half-resolution grid.
  - A step `len` px long reads level `floor(log2(len) − 3.3)`, clamped to
    0–4 (XeGTAO's rule), so its samples are about a tenth of its length
    apart. Steps under ~20 px read the depth itself. A point at screen
    position `q` is at texel coordinate `(q − 0.5)/2^k` in level `k`, the
    same rule as `textureGather` at level 0. Above level 0 it's four
    `texelFetch`es, since `textureGather` has no LOD.
  - Radius 0.8 m, with occluders fading over its last 60%.
  - Pixels whose radius spans under 2 px are left open; the radius is
    clamped to 200 px (full-resolution pixels, as are the steps).
  - The horizons are integrated against the normal projected into each slice.
- **`gtao_denoise.comp`:** a 4×4 blur at half resolution (8×8 pixels), so
  every texel averages all 16 jitter variants (32 directions). Each neighbour
  is weighted by its distance off the centre's tangent plane, and gets nothing
  past 10% of the view distance.
- **`mesh.frag` upsamples** (`gtao_upsample`), with the fragment's own
  position and geometric normal, and no pass of its own:
  - A pixel at even coordinates copies its texel.
  - Any other pixel blends the 2 or 4 texels round it bilinearly. Each is
    weighted by its distance off the fragment's tangent plane, as in the
    denoise.
  - If no texel lies on the fragment's surface (a feature too thin to have a
    texel), it takes the one nearest in distance.
  - A texel's position comes from depth level 1, which holds exactly that
    texel's pixel.
  - Under MSAA, each surface at an edge pixel gets its own value.
- **`mesh.frag` then applies it:** the diffuse ambient is multiplied by
  Jimenez's multi-bounce fit (per albedo), the specular ambient by Lagarde's
  specular occlusion. Sky visibility times GTAO, ambient only.
- **Targets:** gfx owns three R32F images: the depth levels, and raw and
  denoised AO at half size (rounded up). R32F because it's on the mandatory
  storage list. A `targets_generation` counter tells
  `AoPass` and `MeshRenderer::set_ao` when a resize or MSAA change has moved
  them. A view handle can't tell, since a new view may reuse a freed one's
  handle.
- **Four ways it went wrong first.** A flat floor must read 1, and didn't:
  - **Clamping** each pixel to ≤ 1 before the denoise. One pixel's slice pair
    reads above 1 as often as below; only their average over directions is
    exact. Clamped, the floor read 0.982. The raw target is now unclamped.
  - **Snapping steps to texel centres,** as XeGTAO does. The first 1–2 px
    steps land up to 20° off their slice, and the horizon keeps the error:
    0.995 mean, 0.936 at worst. Steps now sit at their exact sub-pixel point,
    with depth interpolated from a `textureGather`, which is exact on a plane
    (depth is linear in screen space across one).
  - **Weighting the denoise by depth difference** instead of plane distance.
    That weighs the jitter variants unevenly on a receding floor: 0.957 at the
    worst pixel.
  - **Spreading slices evenly round the screen** (XeGTAO's choice) rather than
    round the view vector, which is what the integral assumes. Off-centre
    they bunch up. A face-on wall filling the view read 0.926 in the corners,
    and a wall at 40° read 0.81.
  - A fifth suspect, steps off the screen reading clamped depth, measured as
    **no difference** on either wall. Where steps leave the screen the
    surface comes towards the camera, so a clamped depth lies behind it. No
    rule for it was kept.

  Now: floor mean 0.9998, and 0.995 or more away from the screen's edge
  columns. Face-on and 40° walls read ≥ 0.99 except the two edge columns on
  the 86° side.
- **Checked on the GPU:** a temporary harness dumped one frame's depth and
  GTAO output from three zone viewpoints. The CPU reference, run over that
  depth, matches the GPU to a median of 1e-5 and a 99th percentile of
  1e-3 (0.03 at worst, a few pixels).
- **Measured at full resolution** (its first version; debug, clocks unpinned,
  3 interleaved runs, `--no-ao` vs on):
  - `ao` 0.26–0.27 ms on the zone (frames +0.25–0.28 ms) and 0.37–0.38 ms on
    `lights120`. I predicted 0.15–0.30: the zone held, `lights120` didn't.
  - The gtao pass is most of it: 0.23 / 0.31 ms, against 0.05–0.07 for the
    denoise.
  - The orb demo, with meshes up close at the 200 px radius clamp, costs
    0.47 ms.
  - Release (not interleaved with debug, clocks unpinned): 0.30 / 0.39 ms.
  - The metered exposure doesn't move (1.36 → 1.37): corners are a small
    share of any view.
  - Validation with sync: 0 messages at MSAA 1/2/4×, with GTAO off, on the
    nature scene and the orb demo, and across two resizes mid-session.
    (Not reproduced since: under MSAA the prepass's depth resolve reports
    one `SYNC-HAZARD-READ-AFTER-WRITE` a frame, at that commit too, a known
    false positive of this validation layer. See Limits.)
- **Re-measured with pinned clocks** (`profile_standard`, release, 3
  interleaved rounds; every median repeated to 0.01 ms):
  - `ao` 0.38 ms on the zone and 0.54 ms on `lights120` (frames +0.40 /
    +0.59 ms). I predicted 0.22–0.28 / 0.30–0.38 and was wrong: pinned is a
    fixed clock *below* boost, so every pass reads slower than unpinned.
  - Skipping the denoise (a temporary harness) leaves 0.33 / 0.46 ms, so
    the gtao pass is 85–87% of it. I predicted ~85%, which held.
  - Debug and release measure the same (0.39 / 0.54 ms): GPU work doesn't
    depend on the Rust profile.
- **Half resolution, measured** (pinned, release, 3 interleaved rounds
  against the full-resolution binary; every median repeated to 0.01 ms):
  - `ao` 0.38 → 0.21 ms on the zone and 0.54 → 0.29 ms on `lights120`
    (−45%; frames 1.01 → 0.83 and 1.81 → 1.52 ms). **I predicted 0.12–0.18
    and 0.17–0.25 ms, and was wrong:** too optimistic.
  - A temporary harness skipping passes split it: gtao 0.13 / 0.19 ms,
    denoise 0.03 / 0.03, upsample 0.05 / 0.07. With a quarter of the
    threads, gtao fell only to ~40%. Its steps still read the
    full-resolution depth, from threads now 2 px apart, so the cache helps
    less. I'd guessed ~0.03 for the upsample.
- **Half resolution, checked:**
  - On the GPU: the same dump harness, from three zone viewpoints (in the
    hangar, its door, the office). The CPU reference over that depth
    matches to a median of 1e-5, a p99 of 2e-3 and 0.03 at worst, as at
    full resolution.
  - Against the full-resolution reference, on the same depth: mean
    difference 0.0007–0.002, p99 0.015–0.056. Mean AO is the same to 4
    decimals. The differences are a pixel-wide line along creases and
    silhouettes, and the largest (0.5) is in the thin crevice under the car,
    which reads lighter and softer.
  - Validation with sync: 0 messages at 1×, with GTAO off, on `lights120`,
    the nature scene and the orb demo, and at 1921×1046 (odd halves). At
    MSAA 2/4× only the prepass resolve's false positive (Limits).
- **The steps' depth reads.** An experiment first (pinned, release, both
  results predicted and both held):
  - with steps at a quarter of the radius, the gtao pass fell 0.13 → 0.07
    and 0.19 → 0.09 ms;
  - with no spread at all, 0.05 / 0.07 ms.

  So about 60% of it was step reads spread across the full-resolution depth.
  The depth levels above fix that:
  - `ao` 0.21 → 0.16 ms (zone) and 0.29 → 0.23 ms (`lights120`), frames
    −0.05 / −0.06 ms. Pinned, release, 3 interleaved rounds, identical
    medians.
  - I predicted 0.16–0.18 / 0.20–0.23: both held, at the edge.
  - The split (skipping passes): copy 0.02, gtao 0.07 / 0.11 (predicted
    0.07–0.09 / 0.09–0.12), denoise 0.02 / 0.03, upsample 0.05 / 0.07.
  - The zone's p10 rose 0.10 → 0.12 ms: the copy costs the same in views
    that are mostly sky.
  - On the GPU (the dump harness, the same three views): the reference
    matches to a median of 6e-6 and a p99 of 2e-3.
  - Against 44dfec4's GPU output: mean difference ≤ 1e-4, p99 ≤ 0.002, at
    worst 0.12 along the car's lower edge (close detail, read coarser by the
    long steps).
  - Validation with sync: as for half resolution above.
- **The upsample, moved into `mesh.frag`.** It started as its own
  full-resolution pass, at 0.05 / 0.07 ms. An experiment timed variants of
  that pass:

  | variant | zone | `lights120` |
  |---|---|---|
  | as it was | 0.05 | 0.07 |
  | only its own texel | 0.04 | 0.04 |
  | plain bilinear | 0.04 | 0.05 |
  | sky check + store 1.0 | 0.03 | 0.03 |

  The last row is the pass itself: dispatch, barrier, a depth read and an
  8 MB write. It was most of the cost, so a cheaper blend couldn't win much.
  (I'd predicted 0.02 for that floor, and 0.03 for plain bilinear: both
  wrong.)
  - Measured (pinned, release, 3 interleaved rounds against 635d898;
    identical medians):
    - `ao` 0.16 → 0.11 ms (zone) and 0.23 → 0.16 ms (`lights120`);
    - `geo` +0.02 / +0.04 ms;
    - frames 0.78 → 0.75 and 1.46 → 1.43 ms.
    - I predicted `geo` +0.01–0.02 and frames −0.03 to −0.04 / −0.04 to
      −0.06: the zone held, `lights120` didn't. More of it is geometry, and
      the fetches land in a heavier shader.
  - `mesh.frag` keeps 72 VGPRs and 20 waves a SIMD (RADV shader stats);
    it grew from 1531 to 1826 instructions.
  - On the GPU: a temporary harness wrote `mesh.frag`'s upsampled value to
    the HDR target, from the same three zone views.
    - Against the reference, the median difference is 4.7e-4, which is the
      RGBA16F target's step near 1, and the p99 ≤ 2.4e-3.
    - The mean AO is the same as 44dfec4's to 4 decimals.
    - The largest differences (≤ 0.23, a few pixels) are on the car, whose
      smooth vertex normals differ from the depth-rebuilt ones the reference
      uses.
  - Validation with sync: as for half resolution above.
- **Next lever:** the AO slot is now depth copy 0.02, gtao 0.07 / 0.11 and
  denoise 0.02 / 0.03 ms. gtao is near its 0.05 / 0.07 floor, so what's
  left is small fixed costs. Not pursued.
- **Limits:**
  - screen-space: what's off-screen or behind a silhouette doesn't occlude;
  - no thickness heuristic, so thin poles and grass cards shadow as if
    solid;
  - no temporal filtering, so the result is noise-free only where the 4×4
    blur averages a smooth surface;
  - the screen's outermost ~4 columns and rows can read a few percent dark
    where a surface is seen near grazing: the last two half-resolution
    texels, whose clamped 4×4 window repeats some jitter variants (at full
    resolution it was 2 columns; ignoring steps that leave the screen
    halves it at the edge, an unkept experiment);
  - a step past a level's last grid point, in the last `2^k − 1` px of the
    right or bottom edge, reads that texel's depth (clamped, like a step off
    the screen);
  - an occluder narrower than a level's stride (up to 16 px at 1/16) can
    fall between a long step's samples;
  - under MSAA, sync validation reports one `SYNC-HAZARD-READ-AFTER-WRITE` a
    frame on the prepass's depth resolve, since e246707 gave the prepass its
    own rendering. **It's a false positive** of the installed layer
    (Ubuntu 24.04's `vulkan-validationlayers` 1.3.275):
    - Vulkan-ValidationLayers issue #7441 reports this exact message on
      `vkCmdEndRendering` depth resolves. It appeared with 1.3.275 and not
      with 1.3.268.
    - Upstream fixed it in PR #7476, "sync: Fix ordered accesses for
      depth-stencil resolve". Sync-val orders a resolve's reads after the
      attachment's earlier accesses, but it only knew the colour accesses,
      so a depth attachment's own writes looked unordered.
    - The engine was left alone. USAGE §8 filters exactly this message: its
      ID can't be used, since every read-after-write hazard shares it. A
      layer with the fix won't report it at all;
  - a faint large-scale tint (≤ 2%) was seen on the hangar's far wall in one
    dump and not chased.
- **Switch:** OPTIONS > GRAPHICS > AMBIENT OCCLUSION, saved as
  `ambient_occlusion`, and `--no-ao` for bench A/B runs. Off, the passes
  don't run and `mesh.frag` skips the fetch (`Globals::ao_params`).
- **Tests** (`render::ao`'s reference is both shaders step for step):
  - distances back from depth, against glam's projection;
  - the slice integral against Simpson's rule;
  - an open floor, and face-on and 40° walls filling the view, read 1;
  - a wall darkens the floor at its foot and nowhere else;
  - the denoise keeps to its side of an edge, and so does `mesh.frag`'s
    upsample;
  - the upsample copies its texels and is a plain bilinear blend on a plane;
  - each depth level holds every `2^k`-th pixel, rounded-up sizes included,
    as the copy pass writes them;
  - every level is exact on a plane, wherever its grid reaches;
  - a step's level follows its length;
  - half resolution keeps full resolution's result at a wall's foot (mean
    difference < 0.01, the foot within 0.05);
  - every 4×4 window holds each jitter once;
  - the ambient fits leave open surfaces alone;
  - the shaders declare the reference's constants (and gfx the level
    count); gtao and the denoise share `view_pos` and `depth_normal` text for
    text; `mesh.frag` converts to the passes' space as the view test pins
    it;
  - the menu row, config key and flag.

  Each was shown to fail against a deliberately broken copy.

### Environment: a level's atmosphere

**Landed (§26).** A level chooses its sun, sky, fog and starting exposure with
an `environment` marker (§18). Without one it gets `Environment::default()`,
which is exactly the look every level had before.
- **What it sets:** the sun (direction, colour × intensity); the analytic sky
  (zenith, horizon, below-horizon colours, overall intensity, which is also
  the ambient light, since the IBL is that sky); the tint and strength of the
  sun's glow and of its disk (0 hides the sun, as behind cloud); exponential
  fog density (fog blends towards the sky, so it takes the sky's colour); and
  the tonemap's starting exposure.
- **How it reaches the GPU: specialization constants.** `render::Environment`
  fills 19 `constant_id`s that `mesh.frag` and `sky.frag` declare, baked when
  a session builds its pipelines (`MeshRenderer::new`, `SkyPass::new`).
  - A level's atmosphere is fixed for its session, so this costs nothing
    per frame and needs no descriptor or push-constant space. The sky pass's
    push constants were already 96 of the guaranteed 128 bytes.
  - One struct feeds both shaders, which used to repeat the palette by
    hand. Their `sky()` functions still have to agree by hand.
  - A test parses the compiled SPIR-V: every constant a shader declares must
    be in the map at the same id, under the expected name, with a GLSL
    default equal to `Environment::default()`. A mismatch compiles fine and
    shades silently wrong, so without that test nothing would catch it. Both
    kinds of mismatch were tried; the test failed on each.
- The sun's **direction** and the **exposure** aren't baked. They already
  travel per frame, so the app takes them from the environment at session
  start. The exposure debug keys still adjust it, but no longer carry over
  from one session to the next.
- **Measured:** `--bench scratch/lights120.gltf`, old and new binaries
  interleaved (clocks not pinned): frame median 0.72/0.71 ms either way, as
  predicted. The constants fold as the literals did.
- **Limit:** changing the atmosphere *during* a session (weather, time of
  day) means rebuilding two pipelines. The runtime route is the globals UBO
  (§25).

**Landed (§26): height fog.** Fog was uniform (`1 − e^{−density·dist}`,
towards the sky's colour), only on geometry, so the horizon stayed crisp and
the ground's edge showed. Now it can pool on the ground.
- **Model:**
  - The density is `fog_density · e^{−fog_falloff·(y − fog_height)}`.
  - Along each view ray it is integrated exactly:
    `τ = ρ(eye)·(1 − e^{−k·d·v.y}) / (k·v.y)`, with the series
    `1 − x/2 + x²/6` for `|k·d·v.y| < 0.01`, where the exact form cancels
    badly in f32 (with the first cut-over, at 1e-4, the integration test
    caught a 1.2e-4 relative error just above it).
  - Fog colour is `fog_color` if given, else the sky's in that direction,
    plus an optional `fog_sun` glow towards the sun.
  - `fog_falloff = 0`, the default, is exactly the old uniform fog.
- **The sky background fogs too, but only through a height layer.**
  - Looking up, the fog to infinity is finite (`ρ(eye)/(k·v.y)`), so the
    zenith stays clearer than the horizon.
  - Below the horizon it's total, which hides the void past the ground's
    edge.
  - Uniform fog leaves the sky alone, as it always did: fogging an
    infinitely distant sky uniformly would have dimmed the default sun disk
    to ~14%.
- `sky()`, `fog_optical_depth()` and `fog_color()` are written out in both
  shaders, since the build has no `#include`. A test asserts their texts are
  identical, so the sky and the fog on geometry can't drift into a seam.
  `sky.frag`'s disk moved from `sky()` into `main` for that; the same maths.
- **Tests:** the closed form against Simpson integration (climbing,
  descending and level rays, four falloffs, three eye heights, four
  distances); uniform fog equals `density·dist` exactly; the sky limit; the
  shared text. Each was shown to fail on a deliberately broken copy.
- **Measured** (debug, clocks not pinned, old and new binaries interleaved):
  - `zone.glb`, which uses the height fog: `geo` 0.21 ms in both, 3/3 runs.
  - `lights120` (default fog): `geo` 0.55 → 0.57–0.58 ms, reproducible
    (3/3). I predicted no change, and was wrong; the cause isn't found.
    - The compiled `mesh.frag` (RADV shader stats) is 977 vs 975
      instructions with the same 60 VGPRs, so the same occupancy, and no
      spills.
    - Folding the default path on its specialization constants changed
      nothing.
    - A bisect found that any rewrite of the fog tail costs the same, and
      only restoring the old lines removes it. That points at code layout
      or scheduling in the driver's output, not arithmetic.
    - Worth re-measuring with pinned clocks and in release before acting on
      it.
  - **Re-measured with pinned clocks** (`profile_standard`, 3 interleaved
    rounds of fb3e789 vs e533010, debug and release): it's real and larger.
    `lights120` `geo` 0.77 → 0.81 ms in debug and 0.76 → 0.81 ms in release
    (+5–6%, frames 0.92 → 0.97 ms), 3/3 each; the zone 0.19 ms either way. I
    predicted it would persist at +0.01–0.02 ms, and it came out larger. It
    lands on the default fog path, which `lights120` uses, so it's worth
    finding.
  - **Found: it isn't the fog, it's register allocation.** Variants of
    `mesh.frag` at a2c5e27 (pinned, release, 3 interleaved rounds;
    `lights120` `geo`):

    | variant | `geo` ms | light-loop bank conflicts |
    |---|---|---|
    | as it is | 0.96 | 30 of 155 VALU |
    | the three pre-e533010 fog lines | 0.94 | 25 |
    | `normalize` for the fog direction | 0.96 | 30 |
    | no fog at all | 0.95 | 30 (and fewer elsewhere) |

    - The zone read 0.31 ms in every variant.
    - I predicted the old lines would win back 0.03–0.05 ms (it was 0.02),
      and no fog 0.01 more (it was *slower*): both wrong.
    - The regression shrank from +0.04–0.05 to +0.02 ms since the re-measure
      above, with no fog change in between.
    - **Evidence.** In the ISA (`RADV_DEBUG=shaders`), the current and the
      old-fog shaders are the same length (1852). Their clustered light loop
      is the identical instruction sequence with the registers renumbered:
      0 structural differences, 328 lines differing only in register
      numbers.
    - **The likely mechanism.** RDNA's VGPRs sit in four banks (index mod 4),
      and a VALU instruction that reads two sources from one bank stalls. A
      count of those in the loop ranks the variants as their times do. (It's
      a simplified model of RDNA3: consistent with the evidence, not proof.
      Material AO later gave a counterexample: fewer conflicts and slower,
      so treat it as one factor among several.)
    - **A check that it predicts, not just fits:** three edits that change
      no maths (swapping two AO lines, a sentinel constant, one
      `inversesqrt`) left the loop's allocation as it was, and each timed
      0.96, as predicted.
    - **What follows.** Any edit to `mesh.frag` can move `lights120`'s `geo`
      by ~0.02 ms through allocation alone. The fog wording that allocates
      better today does so by dropping height fog, and matching it with some
      other wording would be tuning against one driver version, so nothing
      was changed.
    - **Next, if it matters:** a lighter light loop (fewer live values), or
      a driver update, not the fog.

### Post chain (after transparents; HDR until tonemap)

1. **HDR resolve** (done as pass 7).
2. **Bloom** — threshold → progressive dual-filter down/up mip chain; in HDR,
   before tonemap; persistent mip pyramid.
3. **Exposure** — fixed constant to start; auto-exposure (luminance compute) is a
   hook in the camera UBO, not built now. **Landed** (below), with its state
   in a storage buffer the tonemap reads rather than the camera UBO.
4. **Composite + tonemap** — scene + bloom, exposure, then **AgX**/ACES (in
   linear).
5. **OETF/gamma** — let the `_SRGB` swapchain encode; do **not** also gamma in
   shader (no double-correction).

**Landed (§26): bloom, physically based, without a threshold** (Jimenez 2014,
Call of Duty: AW).
- **How it works:** the whole HDR image is filtered down a mip chain and back
  up. The tonemap mixes the result in at `BLOOM_STRENGTH` = 0.04, so things
  glow in proportion to how bright they are and dark ones stay crisp.
- **Target:** gfx owns the chain: one RGBA16F image at half resolution, mips
  down to an 8-texel side, at most 7 levels (`bloom_mips`), one view per
  level, rebuilt with the other targets. It's RGBA16F because the passes
  write it as a storage image and RGBA16F is on the mandatory storage-format
  list, unlike B10G11R11.
- **`draw_frame`** gets a compute slot between geometry and tonemap, timed on
  its own (`[gpu] … bloom`, `[bench] bloom`).
  - It moves the chain UNDEFINED → GENERAL each frame. Following §21's rule,
    that waits for the previous frame's bloom writes and tonemap reads, and
    it happens even with bloom off, since the tonemap's descriptor names that
    layout.
  - The HDR→read barrier now also covers compute.
- **`render::BloomPass`:** 13 dispatches for 7 levels, each its own
  descriptor set, with a compute→compute barrier between steps.
  - **Down:** a 13-tap filter (five overlapping 2×2 boxes, weights ½ + 4 × ⅛).
    The first step is a Karis average: boxes weighted 1/(1 + luma), so a lone
    very bright pixel can't flicker the bloom.
  - **Up:** each level adds a 3×3 tent of the one below, in place, so level 0
    ends as the sum of every level.
- **The tonemap** divides that sum by the level count before mixing. A flat
  image then blooms to itself exactly, which makes the mix energy-conserving.
  It skips bloom entirely at strength 0.
- **Tests:** a Rust reference of the chain, whose weights the shaders must
  declare (a text check):
  - a flat image stays flat;
  - a bright pixel spreads with falloff;
  - a plain downsample keeps energy within 5%.

  Also the chain's sizes and the tonemap's push layout.
- **Measured** (debug, clocks unpinned, 3 interleaved runs, `--no-bloom` vs
  on):
  - `bloom` 0.15–0.16 ms on both `zone.glb` and `lights120`, tonemap +0.01–
    0.02 ms (the extra sample), frame +0.12–0.16 ms. **I predicted
    0.05–0.10 ms, and was wrong.**
  - It isn't the barriers between the many small dispatches: capping the
    chain at 4 levels still cost 0.14 ms. The cost is in the big levels, the
    first downsample (13 taps of the full-resolution HDR image) and the last
    upsample.
  - A single-pass downsample (FidelityFX SPD-style) or a cheaper first tap
    pattern is where to look if it matters.
  - **Re-measured with pinned clocks** (`profile_standard`, release, 3
    interleaved rounds): `bloom` 0.10 ms on both scenes, frame +0.10–
    0.12 ms. A temporary harness that stopped the chain early split it:
    the first downsample is 0.03 ms, the last upsample 0.02 ms, and the
    other 11 steps share 0.05 ms. I predicted 0.08–0.12 in all, ~0.04 for
    the first downsample and 0.02–0.03 for the last upsample, and all held.
    So the unpinned 0.15 was mostly downclocking, and no single step is
    worth rewriting on its own.
  - Validation with sync: 0 messages, with bloom on, off, and at MSAA 4×
    (where it reads the resolve target).
- **Switch:** OPTIONS > GRAPHICS > BLOOM, saved as `bloom` in
  `graphics.toml`, and `--no-bloom` for bench A/B runs (the bench ignores
  config). Bloom runs only over a world.

**Landed (§26): auto-exposure, centre-weighted.** The eye adapting: the dark
hangar brightens, bright sky darkens.
- **Metering:** two compute passes after bloom, owned by
  `render::ExposurePass`.
  - **Histogram:** log2 luminance of a quarter-resolution grid of HDR
    samples in 256 bins over 2⁻¹⁰…2⁶. Each sample is weighted by
    `exp(−r²/0.6²)`, with r in half-heights from the screen's centre (1…64,
    never 0), so what you look at drives exposure. A workgroup accumulates
    in shared memory, then adds into a global histogram.
  - **Average:** one group. It takes the weighted mean between the 10th and
    90th percentiles, adapts towards it in log space
    (`adapted += (target − adapted)·(1 − e^{−dt·speed})`, 3 EV/s to bright,
    1 EV/s to dark), and computes `exposure = clamp(KEY / 2^adapted, min,
    max)`. It zeroes the histogram for the next frame.
  - A level's first frame, and turning auto-exposure back on, snap to the
    metered value instead of fading.
- **State:** it stays on the GPU. A persistent buffer (the histogram,
  `adapted`, `exposure`) is read straight by the tonemap, so there are no
  frames of CPU readback lag.
  - The buffer is shared across frames in flight, so the pass's first
    barrier waits for the previous frame's compute writes and tonemap reads
    (§21's rule), and its last makes its writes visible to the tonemap.
  - Each frame also copies the result into a small host buffer, which the
    app reads after that frame's fence. That's what `[bench] exposure`
    prints: the evidence that it responds.
- **Compensation:** with auto-exposure on, the level's `exposure` and the
  `[`/`]` keys are compensation (a multiplier on the metered value).
  `exposure_min`/`exposure_max` (defaults ⅛ and 8) bound it per level.
  With auto-exposure off, `exposure` is the fixed exposure, as before.
- **`KEY` = 0.55, calibrated, not guessed:** with 0.18 (the textbook middle
  grey), `lights120`'s sweep metered a median exposure of 0.33, so its look
  would have darkened by 3×. 0.18 / 0.33 ≈ 0.55 puts that median at **1.00**
  (0.84–1.14 over the sweep), today's fixed look, as the neutral point.
  - The zone then meters 0.98–1.88 (median 1.36), since it's a darker,
    overcast scene. Its `exposure` compensation went from 1.25, a
    fixed-exposure guess, back to 1.0.
  - Predicted: exposure varies as the camera turns, and the zone settles
    higher than `lights120`. Both are right.
- **Measured** (debug, clocks unpinned, 3 interleaved runs,
  `--no-auto-exposure` vs on):
  - the passes cost 0.05 ms and frames grow 0.04–0.05 ms, on the zone and
    `lights120`. I predicted ≤ 0.04 ms, slightly low.
  - Validation with sync: 0 messages, on, off, and at MSAA 4× (which
    meters the resolve target, and gives the same exposures).
- **Tests:** a Rust reference of the maths, whose constants and state layout
  the shaders must declare (a text check, which caught a stray comment in
  the layout on its first run):
  - bins round-trip;
  - the trimmed mean ignores tails under 10%;
  - adaptation converges, brightens faster than it darkens, and is frame-rate
    independent;
  - the exposure maps `KEY` and clamps;
  - the centre weight falls off but never to 0.
- **Switch:** OPTIONS > GRAPHICS > AUTO EXPOSURE, saved as `auto_exposure`,
  and `--no-auto-exposure` for bench A/B runs. It runs only over a world.
**Landed (§26):** a **graphics settings layer** with a live **shadow-quality**
preset (`F1` cycles; `GraphicsSettings` on `App`, deliberately not an ECS resource
since §1 scopes the `World` to simulation). Measured on the orb demo at one window
size, one run:

| preset | shadow map | shadow pass | frame |
|--------|-----------|-------------|-------|
| Off    | 512² (no casters) | 0.01 ms | 1.10 ms |
| Low    | 1024²     | 0.47 ms | 1.72 ms |
| Medium | 2048²     | 0.86 ms | 1.97 ms |
| High   | 4096²     | 2.00 ms | 2.97 ms |

`Off` needs no shader branch: extract stops feeding casters, so the (small) map is
merely cleared and every fragment compares against 1.0 as lit. Switching happens
with the device idle — the shadow image is freed and the mesh renderer's
descriptor re-pointed, which is unsound mid-flight.

**Landed (§26): persistence.** Display mode, field of view, shadows, MSAA and FXAA live in
`config/graphics.toml` (gitignored, cwd-relative like the bake dir; module
`app/src/config/`; controls have their own file, §14. It was `settings.toml`
until then, and an old one is renamed on first run). A missing file is written with commented defaults.
Precedence is defaults < file < CLI, and only a key the player changes (menu,
F1, F2, F11) is written back, so a `--msaa 4` override never leaks into the file.
Saving edits that key's value in place and leaves every other byte alone
(comments, unknown lines, hand edits made mid-run). A bad line is a warning;
an unreadable file is left untouched and the run saves nothing. `--bench`
neither reads nor creates it, so timings don't depend on someone's settings.
Hand-parsed flat TOML rather than a crate: a few scalar keys don't justify a
dependency. One file per concern is what keeps each file flat, so the bindings
(§14) didn't need tables or a crate either. Found on the way: the renderer ignored the *initial* settings (it starts
at the High shadow size with FXAA off), which was invisible while those were
also the defaults; startup now pushes both in.

**Landed (§26): display mode.** `display = "windowed" | "fullscreen"`, live
from GRAPHICS > DISPLAY and the rebindable `toggle_fullscreen` action (F11).
Fullscreen is **borderless** on the current monitor
(`Fullscreen::Borderless(None)`). Exclusive fullscreen (a video-mode change)
is ignored by winit on Wayland, so it would be a setting that does nothing
there. Switching is `window.set_fullscreen`; the compositor's `Resized` event
recreates the swapchain like any resize, so the renderer didn't change. The
window opens straight into the saved mode (`with_fullscreen`), default
windowed at 640×480 logical; `--bench` has no config and stays windowed at its
fixed size. Checked with a temporary harness that toggled it live:
640×480 → 1920×1080 → 640×480, sync validation clean.

**MSAA landed** and is selected at startup with `--msaa N` (1/2/4/8), clamped to
device support. The geometry pass renders into a multisampled HDR + depth pair and
dynamic rendering resolves (`AVERAGE`) into a single-sample image at
`cmd_end_rendering`; `Renderer::hdr_view` hands that resolve out, so the tonemap
pass needed no changes — `TonemapPass::update` already re-points when the handle
changes. The `render` crate needed **no changes at all**, because the mesh, sky and
prepass pipelines already read `Renderer::samples()`. Measured on the orb demo at
one window size:

| MSAA | geometry pass | frame |
|------|---------------|-------|
| 1×   | 0.80 ms | 2.88 ms |
| 2×   | 1.33 ms | 3.38 ms |
| 4×   | 1.59 ms | 3.64 ms |
| 8×   | 2.20 ms | 4.03 ms |

The shadow (~1.9 ms) and tonemap (~0.11 ms) passes are unchanged across all four,
confirming the shadow map stays single-sample and the tonemap stays 1×.

**Startup-only, not live:** the sample count is baked into every geometry
pipeline, so changing it means rebuilding them all — the usual "applies on
restart". Live switching is a follow-up.

**Known caveat — HDR resolve.** The hardware resolve averages *radiance* and the
tonemap runs afterwards, which is not the same as tonemapping then averaging. With
`SUN_RADIANCE = 8.0` and a `pow(s, 4000) * 60` sky disk, very bright edges can
sparkle even as geometry edges smooth out. The fix is a custom tonemapped resolve
(a small fullscreen/compute pass replacing the hardware one); not done here.

**FXAA landed too** (`F2` toggles it live, default off), so there is now a real
choice of AA mode. The chain becomes `HDR → tonemap → LDR → FXAA → swapchain`;
without it the tonemap writes the swapchain directly as before. Measured cost:
the `post` pass goes **0.12 ms → 0.24 ms**, i.e. ~0.12 ms for FXAA, and unlike
MSAA that is **flat with scene complexity** — it tracks resolution only. For
comparison, 4× MSAA cost ~0.8 ms of geometry on the same scene, so FXAA is
roughly 6× cheaper here at lower edge quality: it softens edges but slightly
blurs fine detail, and cannot recover subpixel geometry the way multisampling
can. That is the trade, and it is why the weaker machine wants FXAA.

Implementation notes: the LDR intermediate uses the **swapchain's `_SRGB`
format**, so one tonemap pipeline serves both targets (dynamic rendering requires
the pipeline's colour format to match the attachment, so a UNORM intermediate
would have forced a second pipeline). Sampling an `_SRGB` image hands back
**linear**, so FXAA blends in linear — physically the right place to average —
and only its edge *thresholds* need perceptual luma, approximated with `sqrt()`
(gamma 2.0) at one instruction per tap rather than a full OETF. The `[gpu] post`
timer now spans tonemap *and* FXAA. Unlike MSAA this toggles live, because
nothing about it is baked into a pipeline; the LDR intermediate is always
allocated (~8 MB at 1080p) so the toggle needs no resource churn.

SMAA is still the higher-quality target (§13) and still absent: it needs
precomputed `AreaTex`/`SearchTex` lookup data that would have to be vendored.

6. **Anti-alias** — **user-selectable**: SMAA on LDR (post-tonemap) or MSAA
   2×/4× on geometry (the geometry sample count is already a single knob —
   `Renderer::samples`, see §26). SMAA is the cheaper default on bandwidth-bound
   hardware (e.g. Steam Deck); MSAA the higher geometry-edge quality where the
   budget allows. (TAA is a later, bigger shift that replaces both and moves
   before tonemap.)
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

**Landed (§26):** the `InputState` **resource** and the rate seam — the app fills
it at render rate (abstract `wish` direction, latched `jump` edge, vertical axis,
noclip) and the fixed step consumes it, the controller system clearing the jump
edge so one press feeds exactly one tick. Look deltas are applied at render rate
straight to the `Look` component. No gamepad, no UI focus flag.

**Landed (§26): the bindings table**, in `config/controls.toml`
(`app/src/config/controls.rs`), separate from graphics by function. Physical
keys map to 11 actions (movement, jump, noclip down, noclip, exposure ±, the
shadow and FXAA toggles), each taking a list of keys or `[]`. Keys are named by
US-QWERTY position with punctuation spelled out (`LeftBracket`), which keeps
TOML escapes out of the file. **Escape, Up, Down and Enter stay hard-wired to
the menu** and are refused as bindings, so a broken file can't strand the
player outside the menu the rebinding screen lives in. The file also
carries `sensitivity` (a multiplier on the old hard-coded 0.0025 rad per
count) and `invert_y`. The menu edits it (§19: OPTIONS > CONTROLS and
GAMEPLAY) and saves each change in place, like `graphics.toml`. Held state comes from a set of keys
that are down, so two keys on one action don't cancel each other. An action
fires when it becomes active, so **key auto-repeat no longer re-toggles** noclip
or FXAA (holding V used to flicker). Exposure is the deliberate exception and
repeats. Focus loss clears the held set, since a key released while unfocused
never reports its release. All of it (parse errors, reserved keys, multi-key
hold, repeat, rebinding) is unit-tested without a GPU through
`Controls::key`, a pure function of (bindings, held set, event).

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
- **Collision proxies, never detailed render meshes.** **Landed (§26):** each
  scene primitive collides by the `collider` prefab param: `mesh` (exact
  trimesh), `hull` (convex hull), `box` (oriented bounds, as a hull of their
  eight corners) or `none`. The default `auto` keeps the exact mesh within
  `TRIMESH_MAX_TRIS` = 2048 triangles and uses a hull above it. The measurement
  behind it: standing on a 34k-triangle Poly Haven lantern's trimesh cost
  **27 ms per physics tick in debug** (213 ms stepping off its rim; 2.5 / 27 ms
  in release), and the fixed-step catch-up (up to `MAX_STEPS` ticks a frame)
  turned that into seconds per frame. As a hull the same lantern costs 0.01 ms
  a tick. Hulls are per primitive, so a rock *set* is one hull per rock. They
  fill concavities, which an author overrides with `collider: "mesh"`. Flat
  geometry still gets a zero-thickness hull that collides fine (tested); only
  points with no hull at all fall back to the exact mesh. `collide: false`
  still wins, so older scenes keep their meaning.

  **Landed (§26): colliders from the baked LODs.** Between those two, `auto`
  now tries the mesh bake (§17) first. An over-budget mesh uses the *finest*
  LOD with at most `TRIMESH_MAX_TRIS` triangles whose error, scaled by the
  node's largest axis scale, is within `COLLISION_TOLERANCE` = 5 cm (the
  capsule is 35 cm, autostep 40 cm), as an exact trimesh. Only when no LOD
  qualifies (no bake, `--no-bake`, or a node scaled up past the tolerance)
  does it fall back to the hull. `collider: "mesh"` still means the full mesh.
  The baked meshes are read once per session and shared with the renderer.
  A `[scene] colliders: …` line reports what a scene got. On `detail_high`,
  all 373 colliders (busts, lanterns, rocks) went from hulls to LOD trimeshes
  (lantern and bust at LOD4, rocks at LOD2–3). The load's "world" phase went
  0.50 → 0.33 s (predicted −10…40%): quickhull over every vertex of a full
  mesh costs more than a BVH over ~2k triangles. A unit test drops the
  player into a dense dish: as a LOD trimesh its feet rest at 0.22 m, at the
  bottom (0.2); as a hull, at 1.22 m, on the lid the hull puts at rim height
  (1.2).

  Per-tick cost, standing on the lantern scaled 8× (debug, a temporary
  harness): full trimesh 0.25 ms, LOD4 trimesh 0.023 ms, hull 0.012 ms. The
  LOD is well under a millisecond and 1.9× the hull (predicted: under 1 ms,
  within 2×). But the full mesh is no longer the 27 ms quoted above, which
  predates optimising dependencies in debug builds (`[profile.dev.package."*"]`
  in `Cargo.toml`). So the 2048 budget's
  original rationale is stale. It's still a sensible bound (11× cheaper than
  the full lantern), but worth re-measuring before anyone relies on the 27 ms
  figure.

  **Landed (§26): what's underfoot.** rapier's controller reports *that* the
  player is grounded, not on what (its ground contact is private). So after
  each grounded move, `player_target` sweeps the capsule's bottom sphere
  `GROUND_PROBE` = 10 cm down and stores the hit collider's §20 surface in
  `Player.surface`; airborne ticks keep the last one. A sphere rather than a
  ray from the centre, because on a ledge held by the rim, centre over the
  drop, the ray finds nothing (tested). And rather than the whole capsule,
  which also tests whatever trunk it leans on: along three walks through the
  nature scene the sphere cost 20–40% less than the capsule and hit the same
  collider at 573–600 of 600 poses. The rest were walks that ended on the seam
  between two floor tiles, where either tile is right. A probe costs 1.3 µs
  in the open on the cuboid ground, 3.6 µs on the nature scene's floor tiles
  and 6–12 µs beside props (debug, a temporary harness). I predicted ≤ 5 µs:
  right in the open, wrong beside props.

  The surface lives in the collider's rapier `user_data`, as its `Surface`
  index. Concrete is 0, rapier's default, so anything untagged (the built-in
  level) is concrete. It's there because the probe runs where there is
  `Physics` but no `World`. If something needs collider → entity later, the
  entity belongs in `user_data` and the surface moves to a component.

  **Landed (§26): low overhangs no longer stall the controller.**
  - **Symptom:** in the nature scene, holding a direction into a tree ran
    the controller's slide loop to its internal cap of 20 passes every
    tick: 0.41 ms a tick in debug (0.56 ms with floor tiles in range),
    against 8–18 µs walking in the open. The player didn't move.
  - **Cause, traced pass by pass:** it wasn't the trunk.
    - The canopy hangs lower than the 1.8 m capsule, so the capsule wedged
      between the ground and the canopy's underside, whose normal points
      down (y = −0.90).
    - rapier's slide projects the motion onto one surface at a time.
      Sliding along the underside tilts it into the ground; sliding along the
      ground tilts it forward into the underside again.
    - In that ~26° wedge each round only shrinks what's left a little, so all
      20 passes run and nothing moves.
    - Any overhang below head height does it: a ball or a tilted slab, as a
      hull or a trimesh. Walls don't: sliding along a wall and along the
      floor don't undo each other.
  - **Fix:** before calling the controller, `player_target` removes the
    horizontal part of the motion that goes into any downward-facing surface
    the capsule touches (now `overhangs_near`, `clip_horizontal`; it also
    looks ahead, see below).
    - Head-on, nothing is left, so the loop never starts.
    - At an angle, the player slides along the overhang as along a wall.
      Before, it stood still, because the loop gave up.
    - Only on the ground, and not on a jump. The stall needs the floor as
      one side of the wedge, and in the air sliding up over undersides is
      how hopping at a tree climbs it, stub by stub. The first version also
      clipped in the air, and that stopped the climb; it was noticed in
      play.
      - A sweep hopped at every prop in `nature.glb` from four sides. With
        the ground-only rule, every prop gets exactly as high as with the old
        code. Clipping in the air had changed the outcome for 12 tree
        models (2–12 runs each), mostly by losing the climb.
      - `hopping_at_a_tree_climbs_its_branches` pins it, with two
        triangles cut from a real Kenney tree.
    - The check is a ball at the capsule's top sphere, the only part a
      downward-facing surface can touch: on the straight part every contact
      normal is horizontal. Contacts come from contact manifolds, as in
      rapier's own ground test. Only things at head height cost anything.
  - **Measured (debug, temporary harness, the same binary with the fix on
    and off):**
    - pushing into the tree: 419 → 16 µs a tick (1.4 passes), and 560 →
      26 µs with the floor tiles;
    - walking in the open: +0.3–1.1 µs.
    - Predicted ≤ 50 µs, and unchanged in the open.
  - **What doesn't help:**
    - rapier's `normal_nudge_factor`: up to 1e-3 nothing changes, and from
      1e-2 the player squeezes under the canopy (tunnelling, not a fix);
    - rapier 0.36, which doesn't touch the slide loop.
  - `Player.slide_hits` counts a tick's passes. `a_low_overhang_holds_the_player_cheaply`
    and `a_low_overhang_is_slid_along` pin the behaviour down.

  **Landed (§26): the rest of the grounded stalls.**
  - **Re-measured:** a new sweep walked into every `nature.glb` prop.
    - 8 directions, each aimed at the centre and 0.4 m to one side:
      2,832 runs of 180 ticks, 509,760 ticks in all (debug, a temporary
      harness).
    - 257 ticks ran all 20 passes.
    - **Correction:** about 3,000 more came from 19 runs that *started
      inside* a neighbouring tree. A capsule inside a trimesh bounces
      between the two faces of the same triangles for the whole run.
      That's an artefact of placing starts blindly, and probably most of
      the 7,133 this section used to quote. Turning noclip off inside a
      tree would do the same.
  - **Cause, traced:** all 257 were the ground + underside wedge above.
    The clip missed it in three ways:
    - the underside was only reached *during* the tick. 161 ticks were
      the slide carrying the player from one canopy facet onto the next,
      and 44 were arriving at a canopy. The clip only knew what touched
      when the tick began.
    - the underside was shallower than the clip's ~12° (30 ticks): 9–10°
      past vertical, like a fat trunk's bulge. It wedges only when the
      push along it is blocked too.
    - 22 unseen by the head ball at the tick's start. They went with the
      first fix below.
  - **Predicted wrong:** I expected creases between two side walls (rocks,
    cliff blocks) to dominate and trees to add little. No stall had two
    side walls. All but 5 happened pushing at a tree, and the one of those
    5 traced (a log) wedged under a neighbouring tree's canopy.
  - **Fix 1, look ahead.** `around_overhangs` bends the tick's motion
    round what it would reach.
    - `overhangs_near` also returns undersides within this tick's travel,
      each with the room left before the head is `OVERHANG_REACH` away.
    - A leg goes that far *in its own direction*. The rest is then clipped
      against that underside, now touching, as before.
    - Up to 3 bends, summed into one move for the controller.
    - A grounded player now stops 4–5 cm short of an underside instead of
      the controller's ~2 cm (3.9 cm further out in the canopy test).
  - **Versions that failed**, each caught by a test or by the sweep's
    outcome check (did each run get past its prop?):
    - clipping far undersides like touching ones steered the player
      sideways along facets it hadn't reached. On a ball canopy that drift
      fed itself until the player slid round it
      (`a_low_overhang_holds_the_player_cheaply` failed);
    - stopping at the underside ahead, without going on along it, held
      612 runs that used to slide round pines and cones;
    - a leg aimed exactly at the reach left float noise. The next tick
      then found that underside "still ahead" with microscopic room, and
      stopped the player dead. Legs now aim 1 cm inside the reach.
  - **Fix 2, shallow undersides, only after a wedge.**
    - Clipping them up front (to ~1°) removed their stalls, but held
      players whom the slide used to carry along a trunk: 12 of 21 newly
      held runs, all at `tree_fat`, half of them glancing pushes.
    - Instead, `Player.wedged` latches after a grounded tick whose slide
      gave up. While it's set, shallow undersides that touch count for the
      clip, and it stays set as long as the clip keeps changing the
      motion. So each wedge costs one 20-pass tick.
  - **Measured (debug, the same binary with the new code on and off, two
    repeats each):**
    - 20-pass ticks: 257 and 233 → 5 and 7, all shallow wedges, about
      one per wedge (8 in 7 runs on a traced pass);
    - mean tick: 46.6–46.9 → 47.2–47.4 µs;
    - walking in the open (over 3 m from any prop): 3.90–3.93 → 3.86–3.90
      µs.
    - Predicted: at least 90% fewer 20-pass ticks (right, 97–98%) and
      at most +3 µs in the open (right, no measurable change). My first
      open-walking figure read +6 µs because it counted ticks held by the
      clip beside canopies as "open": a wrong measure, not a cost.
    - The 20-pass count moves by about ±20 between identical runs of the
      old code: a few runs depend on the physics world's history.
  - **Behaviour:**
    - Walking: 2,368 of the 2,832 runs got past their prop before, and
      2,399 now: 36 newly get past, and 5 are newly held (3 head-on at
      round trees, 2 glancing). Two runs of the old code differ in none.
    - Hopping at every prop from four sides (708 runs): 704 reach the
      same height as before and one reaches 2 mm higher. The other 3 also
      differ between two runs of the old code. 512 runs climb above 1.8 m
      either way. Predicted: all within 1 mm (off by that one run).
  - Tests (both fail on the old code, checked):
    - `reaching_a_low_overhang_mid_tick_is_cheap`: starts spread over one
      tick's travel, head-on and glancing, hull and trimesh. At most 3
      passes on every tick; the old code hit 20.
    - `a_shallow_overhang_wedges_at_most_once`: a wall leaning 9°, with an
      inside corner (83 of 120 ticks at 20 passes before) and at its open
      end, where the old controller stuck (67). Now at most one each, and
      the player slides off the open end.
  - **Still open:**
    - each shallow wedge still costs one 20-pass tick, about 0.3–0.4 ms
      in debug;
    - when the 3 bends run out, the rest of the motion is dropped: the
      player stops short rather than wedging.
  - **Correction:** this section used to blame a second stall on trimesh
    floor tiles plus a camp prop (0.24 ms), and the nature tiles became flat
    boxes for it. That obstacle was a player capsule the measuring harness
    had left in the world. What it did show: pushing into another
    *kinematic* body can still stall, because rapier's moving-platform code
    strips the retry's nudge on every pass. Nothing in the game is one yet;
    NPCs would be. The tiles stay boxes, which measured the same as
    trimeshes.

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

**Landed (§26): the texture part of the bake.** `bake/` (`feather-bake`)
loads scenes with the runtime's own loader and collects every (image, kind)
pair actually used (`TexKind`): base colour as sRGB, normal and MR as data. For
each it builds a mip chain on the CPU (2×2 box, **sRGB averaged in linear
space**, edge-clamped for odd sizes), pads each level to 4×4 blocks, and
**BC7-encodes** it with Intel's ISPC encoder (`intel_tex_2`, bake crate only,
so the engine never links it). It writes a thin header + raw blocks per level
to `scratch/bake/tex/<key>.bc7`. The key (`feather_assets::bake::texture_key`)
is xxh3 of the image's **encoded source bytes** + size + kind + `BAKE_VERSION`
(2; it was the decoded pixels until the change below), so it's
content-addressed and incremental, and cross-scene dedup comes free. The
runtime computes the same key per unique texture and uploads the baked chain
when one exists and the device has `textureCompressionBC`. **Otherwise raw
RGBA8 with GPU-built mips, never an error.** `--no-bake` forces raw for A/B.
Baked files are validated on read (magic, version, per-level sizes against
the dimensions) and written via temp + rename, so an interrupted bake is
never trusted. Measured on the detail scene: **64 MB of BC7 vs 256 MB raw**
(4×), 12 textures baked in 4.3 s, a re-run skips all 12, validation clean
(including 1×1/2×2 partial-block levels), and `geo` identical.
**Landed (§26): baked textures are never decoded.** The loader no longer uses
`gltf::import`, which decoded every image up front. It resolves buffers
(`import_buffers`) and reads each image's **encoded** bytes (buffer view,
`data:` URI or file), taking only the size from the header. `TextureData`
holds those bytes and decodes to RGBA8 on the first `pixels()` call, with
the `image` crate's own conversion, optimised even in debug, replacing our
per-pixel RGB→RGBA loop. The key hashes the source bytes, so a baked texture
is found and uploaded without ever being decoded; the `[mesh]` line counts
decoded images to prove it. A side effect: 16-bit PNGs, which used to fall
back to the factors, now load. On `detail_high` (baked, 3 interleaved runs):

| | before | after |
|---|---|---|
| load, debug | 2.66–2.72 s | **0.60–0.61 s** |
| load, release | 0.63–0.64 s | **0.41–0.42 s** |
| peak RSS, debug / release | 469 / 421 MB | **231 / 182 MB** |
| texture VRAM (exact) | 64.0 MB | 64.0 MB |
| GPU frame | 0.51 ms | 0.51 ms |

Predictions: debug −50% or more (held), release −0.2…0.5 s (held), and RSS
−190 MB, which was wrong: it's −239 MB, because the image crate's
intermediate RGB decode was also alive at peak. The "world" phase (spawning +
colliders), which this doesn't touch, got ~0.05 s slower in both builds.
Unexplained; plausibly CPU clocks no longer warmed by the decode that used to
run just before it. The driver's VRAM counter agrees within noise (+390 MB
around the sweep both ways).

**BC5 for normal maps was measured and rejected.** The design (§5 table) had
normal maps as BC5: X and Y in two independent channels, Z rebuilt in the
shader, the same 8 bpp as BC7. Decoding both bakes of the detail scene's four
normal maps and measuring angular error against the source (mean / p99):

| map | BC7, XYZ (kept) | BC5, Z rebuilt | source, Z rebuilt |
|---|---|---|---|
| bust | 0.31° / 1.85° | 0.23° / 2.03° | 0.11° / 1.45° |
| lantern | 0.68° / 3.13° | 0.69° / 5.53° | 0.38° / 4.85° |
| grass | 1.43° / 10.1° | 5.19° / 70.4° | 4.87° / 70.4° |
| rocks | 1.13° / 5.63° | 17.8° / 71.4° | 17.8° / 71.4° |

The last column is the uncompressed source through the same Z rebuild. It
shows the loss isn't compression: the rock and grass maps hold inward-pointing
(Z < 0: 25% of the rock texels) and non-unit (24% of the grass texels) normals,
common in scan bakes, which only a third channel reproduces. Against BC7
through the *same* Z rebuild, BC5 is 25–35% better on the clean maps, which is
the textbook claim. But it's not the engine's comparison, and the prediction
("about half BC7's error") was wrong for this content. Normal maps stay BC7.
Renormalising averaged normals in their mips was measured too (< 0.06° mean
after BC7; the shader normalises anyway) and dropped.

**Landed (§26): the mesh part of the bake, and LODs.** Same shape as textures:
the runtime loads the glTF, keys each mesh (`mesh_key`: xxh3 of the loader's
vertices + indices + `MESH_BAKE_VERSION`, *not* the material, so a shape
used with several materials bakes once), and uses
`scratch/bake/mesh/<key>.fbm` when it exists. Otherwise it uses the raw mesh
as a single LOD, never an error. The bake root is now `scratch/bake/`, with
`tex/` beside `mesh/`. meshoptimizer (`meshopt`, bake crate only) does the
work:
- **LOD0** is the input triangles, vertex-cache ordered.
- **Each further level** is simplified *from LOD0* towards half the previous
  level's triangles, with normals as a weighted attribute (so shading creases
  hold) and **`Prune`**, which drops disconnected parts smaller than the
  error. Grass blades can't be merged by edge collapse at all, and they're two
  thirds of `detail_high`'s triangles.
- **The chain stops** when a step removes < 10%, below 64 triangles, or at 8
  levels. Meshes under 256 triangles keep LOD0 only.
- **All levels share one vertex array**, fetch-ordered over the concatenated
  index lists, so LODs cost index memory only.
- **Each level stores its error** in mesh-local units: meshopt's relative
  error × `simplify_scale`, forced non-decreasing.
- **Files are validated on read:** every index in range, whole triangles,
  finite non-decreasing errors, no trailing bytes.

**Selection is a render concern.** `MeshRenderer` keeps a LOD list and a local
bounding sphere per mesh, and `prepare_frame` picks, per instance per view,
the coarsest level whose error fits the view's budget. For the camera that's
**1 pixel** at the instance's nearest point:
`err · scale · (h / 2tan(fov/2)) / (dist − r)`, with the largest axis scale so
a stretched instance is never under-estimated. For each shadow cascade it's
**one shadow texel** (`texel_world`, distance-free, since the ortho can't
resolve anything finer). Runs are keyed (mesh, LOD), and the app's culling
didn't change. `--no-lod` pins LOD0 with the baked vertex order.

Measured on `detail_high` (pinned clocks, 3 interleaved runs, medians; every
run gave the same numbers):

| | before (no mesh bake) | `--no-lod` | LOD |
|---|---|---|---|
| main Mtris / frame | 3.66 | 3.66 | **0.29** (−92%) |
| shadow Mtris / frame | 7.06 | 7.06 | **0.61** (−91%) |
| `shadow` | 0.76 ms | 0.78 ms | **0.18 ms** (−76%) |
| `geo` | 1.03 ms | 1.03 ms | **0.28 ms** (−73%) |
| frame | 1.83 ms | 1.86 ms | **0.50 ms** (−73%) |

The **vertex order alone did nothing** (predicted: within ±5%; Poly Haven's
exports were already cache-friendly). The **LOD win was far larger than
predicted** (geo −25…45%, shadow −30…60%). The prediction assumed pixel cost
would dominate once triangles dropped. In fact most of the time went to
rasterising sub-pixel triangles. That also corrects the earlier reading that
14.5M triangles "isn't the bottleneck": they were ~¾ of the geometry and
shadow time. Cost: indices 1.5 → 3.0 MB (predicted 1.5–2×), vertices
unchanged at 2.7 MB. The testscene (nothing ≥ 256 triangles) is unchanged,
and the bake takes 0.1 s for `detail_high`'s 27 meshes.
Open risk, not measured: a caster drawn coarser than its own receiver could
self-shadow. It's bounded to one texel, inside the existing bias, but it's
a thing to look at. Colliders use these LODs too (§15): over-budget props get
a LOD trimesh within 5 cm instead of a convex hull.
Not yet: materials and scenes as baked blobs (the rest of this section).

**Landed (§26): the sky-visibility part of the bake.** Ambient light was
unoccluded, so a room lit only through its door was as bright as the yard
outside. The fix starts here: for each scene, the bake measures how much sky
each point of the level sees. The renderer's side is §13.
- **What a cell stores:** `bake/src/sky.rs` covers the scene with 0.5 m
  cells. Each cell centre casts 128 rays over the upper hemisphere (a
  Fibonacci spiral, turned about the vertical by a per-cell hash so thin
  poles become noise rather than structure) and keeps two moments of the
  visibility V:
  - `w0 = (1/4π)∫V dω` and `w = (1/2π)∫V ω dω` (`SkyVis`);
  - they are what the clamped-cosine L1 convolution needs, so the sky light
    reaching a normal n is `max(w0 + w·n, 0)`;
  - for open sky that is exactly the `½ + ½n.y` that `sky_irradiance`
    already gives the sky, so an open point shades as before.
- **Occluders** (`sky_occludes`): every mesh node the sun's shadow sees, i.e.
  not `shadow: false`, and not cutouts, whose alpha the CPU doesn't test
  (grass and chain-link are mostly holes). Each is traced at its coarsest
  baked LOD within 5 cm at its placed scale. That takes the zone from 825k
  placed triangles to 114k.
- **Ray casting:** our own BVH (binned SAH, any-hit and nearest-hit walks),
  about 200 lines. No crate: a test compares it with brute force.
- **Cells inside geometry:** they would read as closed and darken what
  samples near them.
  - A 16-ray full-sphere probe finds them: a quarter or more of the rays
    hit back faces, judged by the vertex normals, since the renderer draws
    with culling off and winding means nothing.
  - They take their **darkest** valid neighbour's value, for up to 3 rings.
  - Darkest, not the mean: a cell inside a wall between a room and the open
    would otherwise carry the open sky into the room.
- **Leaks through thin geometry:** cells are coarser than the thinnest
  geometry.
  - First version: a plain trilinear sample, one cell off the surface along
    its normal. Measured against ray-traced truth in the zone, interiors
    matched (hangar floor 0.107 vs 0.118 of open sky, office 0.006 vs
    0.004). But the top 30 cm of the hangar's walls read 0.10–0.50 against
    a truth of 0: cells above the 0.1 m roof panels were among the samples'
    neighbours.
  - Fix: each cell also stores how far it sees along ±x, ±y, ±z (`SkyFree`:
    4 bits each, up to one cell, rounded down). Sampling
    (`SkyVolume::sample`, which mesh.frag mirrors) leaves out any of the
    eight neighbours whose path to the point is blocked, fading over 0.1
    cell, and renormalises. If it can see none, it uses the plain blend.
  - After the fix the wall tops read 0.000. The office floor strip next to
    its west wall reads 0.014 against 0–0.007.
  - A closed room with 0.2 m walls stays under 0.05 everywhere inside,
    corners included. The plain blend failed that test at a floor corner.
- **File:** `scratch/bake/sky/<key>.fsv`: a header (origin, cell, dims)
  plus 8 bytes per cell (the encoded `SkyVis`, then the `SkyFree`
  nibbles), one RG32_UINT texel on the GPU.
  - Validated on read, and written via temp + rename.
  - Keyed (`sky_key`) on every occluder's `mesh_key`, transform and
    sidedness, plus `SKY_BAKE_VERSION`, which covers the bake's settings.
    Moving a wall re-bakes; retexturing doesn't.
  - One volume per scene file: each is its own level.
  - Cells grow past 4M, which caps the file at 32 MB.
- **Measured** (release, 24 threads): the zone (142×21×131 cells, 390k) bakes
  in **0.7 s** (the BVH build 0.1 s), and the file is 3.1 MB.
  - I predicted 5–30 s and was wrong by an order of magnitude: the LODs
    shrank the triangle count 7×, and most rays leave the level after a few
    boxes.
  - 5,062 cells are inside geometry, and all are filled.
  - Mean sky seen is 0.78.
- **Tests:**
  - the BVH against brute force (any-hit and nearest);
  - open ground sees the whole sky;
  - a closed room is dark inside with no leaks, corners included;
  - a door lets sky in from its side;
  - a canopy shades only what is under it;
  - the probe finds the inside of a closed box, and not of a double-sided
    one;
  - the fill rule;
  - mirrored placements face outwards;
  - occluder selection and LOD choice;
  - the grid grows its cells.

  In `assets`:
  - open-sky weights equal `sky_irradiance`'s;
  - exact encoding of open sky;
  - free distances round down;
  - sampling past a wall;
  - what the key covers;
  - file damage.

  Each test was shown to fail against a deliberately broken copy of the code.

## 18. Scene spawning + save/load

- **Data-driven spawn via prefab registry:** node = static part (transform,
  mesh/material handles) + gameplay part (`{ prefab, params }` in
  `extras`/sidecar). Runtime `HashMap<PrefabId, SpawnFn>`; loading a scene =
  walk nodes → resolve handles → look up prefab → spawn archetype. New types =
  new spawn fn, not a format change. Chunk membership computed at bake, written
  into the blob.
**Landed (§26):** data-driven spawning, both halves. `load_gltf_scene` returns
meshes in local space plus one `SceneNode` per placement, deduplicated by
`(mesh, primitive)` so a mesh referenced by many nodes is stored once and drawn
as instances. Each node also carries a `PrefabSpec` read from its glTF
**`extras`** in this section's `{ prefab, params }` shape, and the app resolves
it against a `HashMap<&str, SpawnFn>` — a new kind of thing is a new function,
not a format change.

A node with a prefab but **no mesh** is emitted as a *marker*, which is what
makes spawn points (and later triggers) expressible at all.

Implemented prefabs are deliberately only those that do something today:

- **`player_start`** — a marker whose position and `yaw` place the player.
  Consumed *before* the world is built, so scene loading now runs ahead of
  player creation in `Session::new`. Absent, the hardcoded spawn is used, so the
  orb demo and unmarked scenes are unchanged.
- **`prop`** — static geometry with `collide` and `shadow` switches (both
  default true), and a `collider` choice (`auto`/`mesh`/`hull`/`box`/`none`,
  §15). `collide: false` closes the per-node collider opt-out §26
  listed as missing; `shadow: false` applies `NoShadowCast`, previously
  hardcoded for the demo ground alone. One parameterised prefab rather than a
  `no_collide`/`no_shadow` pair, since the switches are independent and separate
  ids would need one per combination.

- **`point_light`** — a §12 punctual light, with `color`, `intensity` and
  `radius`, and `prop`'s `collide`/`shadow` switches on the same defaults: a
  lamp with geometry is still a physical object. Special-casing it never to cast
  would make it the one prefab where `shadow` silently did nothing. Added once
  there was a light system to feed; it needed one new function and no change to
  the scene format, which is this section's claim holding up in practice.

- **`environment`** — a marker for the level's atmosphere (§13): sun
  elevation/azimuth, colour and intensity, the sky palette, the sun's glow
  and disk, fog density, exposure. Consumed before the world is built, like
  `player_start`; first one wins. Params left out keep the default look, and
  unknown or unreadable ones are warned about and ignored.

**Unknown ids warn once and fall back to static geometry** rather than failing,
which is what lets a scene be authored ahead of the engine. Extras parsing is
lenient for the same reason: malformed data costs one prop, not the level.

**Materials carry gameplay too:** a glTF material's `extras` can name its
footstep `surface` (§20), as in `{"surface": "wood"}`. It's per material, not
per node, so it follows the geometry into every placement. The loader passes
the name through unchecked (`Material::surface`) and the app resolves it once
per mesh, just as leniently: an unknown name warns once and means concrete.

**Pending:** chunk membership; the bake path (§17), since glTF is still parsed
at runtime; and prefabs for lights and triggers, both blocked on the systems
that would consume them. The registry lives in `app` beside the components it
spawns; §22 wants it in `game`, which is blocked on moving the components
there.

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

**Landed (§26):** the **lightweight custom renderer**, as a `UiPass` drawing
alpha-blended textured quads into the swapchain *after* tonemap/FXAA (LDR/sRGB,
where §13 and §19 both place UI). One pipeline serves both rectangles and text:
the atlas is a **5×7 pixel font** plus a single solid texel, sampled NEAREST so
scaled glyphs stay crisp and cannot bleed into their neighbours. Colours are
specified **linear**, since the `_SRGB` swapchain encodes on store and Vulkan
blends `_SRGB` attachments in linear space.

Its first consumer is the **Esc pause menu**, now a small screen tree: root
(CONTINUE / OPTIONS / EXIT) → OPTIONS (GRAPHICS / CONTROLS / SOUND / GAMEPLAY) →
each submenu. **GRAPHICS carries real settings** — display mode toggles (§13),
shadow quality cycles and FXAA
toggles live, both sharing the exact code path F1/F2 use, so the two cannot
drift. MSAA is shown with its current sample count and a RESTART note but is
**inert**: the count is baked into every geometry pipeline, so changing it live
means rebuilding them all. Showing the value honestly beats offering a control
that silently does nothing. SOUND holds live volumes since audio landed (§20).

**Landed (§26): CONTROLS and GAMEPLAY are live.** GAMEPLAY's SENSITIVITY cycles
presets and shows a percentage (the font has no decimal point); INVERT Y
toggles; FIELD OF VIEW cycles vertical degrees (50–90; the file takes 30–120)
and is saved to `graphics.toml`, since it's a view setting. The FOV used to be
a constant; now the projection, the light clusters (§12), the LOD pixel budget
(§17) and the cascade fit (§11) all read `GraphicsSettings::fov_y()` once per
frame, so they can't drift apart. `--bench` ignores the config and stays at
60°, and its numbers were unchanged by this. CONTROLS lists every action
(§14) with its keys. Rebinding is a small state machine inside `Menu`:
activating a row sets `capturing`, the app hands the next key press to
`Menu::capture_key` **and consumes it**, so pressing V to bind JUMP doesn't
also toggle noclip. Escape cancels. Up, Down and Enter are ignored while it
waits, so a held Enter can't bind itself, and so are keys the file couldn't
name. Any other key becomes the action's only key, taken from whichever
action had it, which then shows NONE (the conflict rule chosen over swapping or
double-binding). Leaving the row, activating another or BACK ends the wait;
hovering the same row doesn't, so mouse jitter can't. Every change returns
`MenuOutcome::SaveControls(part)`, and the app writes just that part of
`controls.toml` in place. The save machinery (`ConfigFile`, `set_value`) moved
from `graphics.rs` to `config/mod.rs` so both files share it.

CONTROLS has 13 rows, more than fit a small window, so the layout gained
**scrolling**. `menu_layout` shows as many rows as fit below the title, and
moves the first visible row only as far as needed to keep the selection in
view. `Menu` stores that position, so hovering a visible row never scrolls
the list under the mouse. Dim bars mark rows hidden above or below. The title
now sits just above the rows instead of at a fixed 28% of the height, so a
tall screen can't run into it. Drawing, hover and click all use the one
layout, and tests cover every screen at three window sizes and every selected
row.

Navigation state lives in a `Menu` struct that depends on **neither the renderer
nor the event loop**: it mutates settings and returns a `MenuOutcome` the app
acts on. That keeps the whole thing unit-testable without a GPU, which is how
the screen tree, wrap-around, BACK-restores-selection, the inert rows and the
rebind flow are all covered. Esc walks back one screen at a time and only unpauses from the root.

Row labels use **A-Z, 0-9 and spaces only** — the 5x7 font renders anything else
blank, so a colon would silently become whitespace; a test guards this.

Pausing stops the fixed step, releases the cursor and ignores mouselook (§14's
focus flag), while rendering continues so the frozen scene shows behind the
overlay.

**Main menu and session lifetime.** The app now opens on a pre-game menu with
**nothing loaded** (NEW GAME / OPTIONS / QUIT); the pause root gains MAIN MENU
alongside EXIT, so leaving a level and quitting are separate actions. Esc pauses
from play and otherwise walks back one screen, stopping at the main root where
there is nothing to resume into.

Everything world-scoped lives in a `Session` — the ECS `World`, the schedule,
the player, the `MeshRenderer`, the `SkyPass`, the mesh fits and bounding
spheres, the accumulator — which `App` holds as an `Option`. Its presence *is*
the app state, so there is no second state enum to keep in sync, and **"no world
loaded" is representable** for the first time. Dropping it is the engine's first
real teardown path.

Two consequences fall out. The main menu needs **no background path**: with no
session the shadow and geometry passes are skipped, and the geometry
attachment's existing clear flows through the normal tonemap with the UI over
it. And **MSAA becomes changeable** — `MeshRenderer` and `SkyPass` are the only
things that bake `Renderer::samples()`, and both are session-scoped, so a sample
count picked in the main menu simply applies when a session starts. No live
pipeline rebuild, which is why the GRAPHICS row is active out of session and
locked (`MENU ONLY`) in it.

Teardown idles the device before dropping the session, since a queued frame may
still be reading those buffers. Relatedly, `App`'s **field order is
load-bearing**: Rust drops fields in declaration order, so `session` is declared
before `renderer` to preserve §26's resources → allocator → device teardown.

The menu is driven by **keyboard and mouse**: arrows/Enter, or hover to select
and left-click to activate. Both share one `menu_index` — hover moves the same
selection the arrows do, so the two are interchangeable mid-interaction rather
than fighting over two notions of "current". Entry rectangles come from a single
pure function of the framebuffer size that the renderer draws and the hit-test
queries, so the visible highlight and the clickable target cannot drift apart;
being pure, it is also unit-tested without a GPU. A click off the entries does
nothing, and hovering off them leaves the selection alone so Enter always has a
target.

Still pending: **egui** for the dev UI (this is the *game* HUD path, not a
replacement for it), SDF/MSDF text for scale-independent glyphs, and lower-case
and punctuation.

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

**Landed (§26): the mixer, first sounds, and the SOUND menu.**
- **Setup:** kira 0.12 on cpal, with default features off: Ogg Vorbis
  decoding only (below) and no realtime-dbus, which would be a second system
  library. On Linux it needs `libasound2-dev`.
- **Buses:** SFX and AMBIENCE sub-tracks under the main (master) track.
  Volumes are built into the tracks at creation, because a `set_volume` tweens
  from 0 dB, so the first sounds would play too loud; live menu changes use
  the 10 ms tween to avoid zipper clicks.
- **The mixer isn't an ECS `Resource` yet:** it sits on `App` next to the
  renderer, since sessions come and go and the mixer spans them.
- **Render clock:** the listener follows the camera, and a pure `StepTracker`
  turns the player's motion into footsteps (per 1.6 m on the ground), jump
  (leaving the ground while rising; walking off a ledge is silent) and landing
  (after falling faster than 3 m/s, louder the harder). All non-spatial, on SFX.
- **Lamps:** every visible lamp (a point light on geometry) gets a *spatial*
  AMBIENCE sub-track looping a seamless 60 Hz hum, attenuated out to the
  light's radius. Bare light markers stay silent, which also keeps the
  120-light stress scenes to zero emitters.
- **No device, no problem:** the run logs `[audio] unavailable` and stays
  silent. `--bench` is silent too, and its timings were unchanged.
- **Tests:** `Audio<B: Backend>` is generic, so tests drive the *real* mixer
  through a capture backend (kira's `Backend` trait is public) and assert on
  its output. A landing is audible; master and SFX at 50% halve the peak; a
  lamp on the right is louder in the right ear; past its radius it's silent.
  That test caught two real bugs before any listening: the start-of-run
  volume tween (master 50% measured 0.82) and a click at the end of the
  landing sound (it now fades its last 5 ms).
- **Checked on the real device:** PipeWire lists a `feather` stream on the
  analog output while the game runs.
- **Recorded SFX (landed):** footsteps, jump and landing come from Kenney
  Impact Sounds (CC0, 0.76 MB, pinned in `tools/fetch_assets.py`: footsteps
  for five surfaces, soft medium/heavy impacts, five variants each, cycled)
  when the pack is fetched. `SoundSet` falls back per file to the synthesised sound, so
  audio never needs the download; a missing pack is one log line naming the
  fetch command. Every clip, recorded or not, is normalised to its event's
  target peak (0.5 / 0.35 / 0.8) at load. A temporary harness put the real
  clips through the mixer at 0.503 / 0.350 / 0.800, so they sit exactly where
  the synthesised ones did. The hum stays synthesised (the pack has no loops).
- **Footsteps per surface (landed).** There are five surfaces, the pack's
  footstep sets: concrete (the default), grass, wood, carpet and snow.
  - **Where the surface comes from:** a glTF material names one in its
    `extras` (§18). Every scene collider carries its mesh's surface, the
    grounded player probes what it stands on each tick (§15), and
    `StepTracker` stamps each step with it (`SoundEvent::Step(Surface)`).
  - **Why per material, not per node:** one tag covers every placement, and
    a multi-material model gets the answer per face (Kenney's cliff blocks
    have grass tops and rock sides).
  - **Load line:** `[scene] surfaces: …` counts the colliders per surface.
  - **Clips:** each surface has five recorded steps, normalised to the step
    peak like concrete. A surface with any file missing or broken reuses the
    concrete steps (recorded or synthesised), with one note per surface
    naming the fetch command. So a pack fetched before the surfaces were
    kept sounds exactly as it did, and a missing pack is still one note.
    There are no synthesised surfaces.
  - **Measured (temporary harnesses):**
    - The 25 recorded steps peak within 0.3 dB of each other. Kenney
      peak-normalises to about −1 dBFS; I predicted within 3 dB. Through the
      mixer they all play at 0.494–0.503, against 0.87–0.91 without the gain.
    - Grass steps last 0.6–0.8 s, but their energy is in the first 50 ms, so
      a walk (a step every 0.2 s) peaks at 0.50 on every surface.
    - Density differs, though. A walk's steady RMS is 0.034–0.041 on concrete,
      grass and carpet, but 0.061 on wood and 0.072 on snow (5.1 and 6.5 dB
      above concrete). If those sound too loud, a per-surface trim is the fix.
  - **Content:**
    - The test level's `surfaces` zone: a strip of five pads east of spawn.
    - The nature scene: materials tagged by name, and grass tiles that now
      collide, as flat boxes (§15).
    - End to end, a harness walked the real scenes through the real spawn
      path. The strip steps concrete → grass → wood → carpet → snow; the
      meadow is grass, stumps and logs wood, stone concrete. The pre-change
      files stay concrete throughout (the negative control). Loading nature
      builds 400 more (box) colliders: world time 0.01 → 0.02 s, the log
      line's resolution (predicted +5–15 ms).
- **Not yet:** music, per-surface jump and landing sounds, stone or metal
  steps (not in the pack), occlusion, reverb.

## 21. Debug / profiling

- **CPU:** Tracy (`tracy-client`), scoped zones on systems + passes.
- **GPU:** timestamp queries bracketing passes → per-pass ms. This is how
  SSR/volumetrics cost gets judged. **Landed** (§26): a timestamp query pool in
  `gfx` brackets the shadow, geometry, and post passes, reads back after the frame
  fence (no stall), and logs smoothed per-pass ms to stderr (`[gpu] shadow … cluster … geo …
  ao … bloom … expo … post … frame …`), with a `Renderer::gpu_times` accessor for a future overlay.
  **Caveat when reading these numbers:** absolute per-pass ms shift with overall
  GPU load/clock state — the fixed-size shadow pass measured 2.5 ms with a small
  window and 4.7 ms with a large one, unchanged work. Only compare A/B runs taken
  back-to-back at the same window size. Uses core `vkCmdWriteTimestamp` for now; the
  `vkCmdWriteTimestamp2` form arrives with the §10 sync2 barrier pass. Tracy GPU
  zones + an egui overlay are the remaining upgrades.
  **Also landed:** startup logs the chosen device (`[gfx] device: …`) — machines
  with an iGPU + dGPU enumerate both, and a timing is meaningless without
  knowing which ran it. `Renderer::gpu_times_raw` exposes the unsmoothed times
  (lagging by `FRAMES_IN_FLIGHT`). **`--bench`** is the repeatable A/B harness:
  it skips the menu, opens a 1920×1080 window, settles 120 frames at the spawn
  point, sweeps yaw through 360° at level pitch over 720 frames, prints
  min/p10/median/p90/max of the raw per-pass times plus the visible-light
  count, and exits. Raw rather than EMA because the view-dependent spread is
  often the thing being measured.
  **Pin GPU clocks before measuring.** Under FIFO a fast GPU idles most of the
  frame and downclocks; on the RX 7800 XT, going uncapped (busier GPU) cut
  `geo` ~30% at 0 lights but only ~13% at 120 — a load-dependent bias that
  back-to-back A/B does not cancel. On amdgpu:
  `echo profile_standard | sudo tee /sys/class/drm/cardN/device/power_dpm_force_performance_level`
  (resets on reboot, or write `auto`). With it set, vsync on vs off measured
  identical, confirming clock state was the only distortion. It fixes the
  clock *below* boost, so pinned times read higher than unpinned ones
  (the zone's frame is 1.01 ms pinned against ~0.87 unpinned): compare
  pinned only with pinned. Pinned, 3 interleaved rounds repeat every median
  to 0.01 ms, and debug and release builds measure the same.
- **Test content:** `tools/gen_testscene.py` generates the measurement
  workload. The orb demo is pathological on purpose (huge overdraw, every object
  a caster, no occlusion) and so is useless for judging cost; this produces a
  walkable glTF level in cardinal sectors, each aimed at one system: pillars
  running from inside to well past the old ±16 single-map shadow boundary
  (shadows/CSM), stairs and ramps
  bracketing the 0.4 autostep and rapier's 45° slope limit (§15), thin poles, a
  lattice and a picket fence at graded distances (AA), a few hundred nodes over a
  handful of meshes (dedup/instancing/culling), and a metallic×roughness sphere
  grid (IBL/tonemap reference). Nodes carry §18 prefabs: the field scatter is
  `prop` with `collide: false`, a `player_start` marker places the player, and
  point lights sit on the emissive spheres and down the pillar rows.
  `--density low|med|high` scales the field for A/B timing, `--lights N` scatters
  N extra point lights as geometry-free markers (the §12 light-loop benchmark:
  mesh count is unchanged, so light count is the only variable) and
  `--light-radius R` sets their overlap (default 14), `--textures`
  adds procedural textures, `--seed` gives reproducible variants, `--check`
  re-parses the output and
  asserts the engine's invariants (bounds, ground contact, no intersection with
  the app's own level geometry, the normal-transform rule below).
  Stdlib-only Python; the generator is committed and its output is not, since
  `/scratch/` is gitignored. **Not** the §17 bake tool — that is the offline
  asset-blob pipeline, this is dev content.
  **Real-asset scene:** `tools/fetch_assets.py` pins third-party CC0 packs by URL
  + SHA-256 (refusing a mismatch) into the gitignored `scratch/assets/`, and
  `tools/gen_naturescene.py` composes the **Kenney Nature Kit** into
  `scratch/nature.glb`: a meadow of grass tiles, a forest, rocks, a cliff
  ridge, a lit camp and a lamp-lit path. It is one level `.glb` with prefab
  `extras`, like the generator's, so the engine needed no change. What it has
  to do to the kit:
  - Kenney's materials are `KHR_materials_unlit` with metallic = 1, roughness
    = 1, which the engine (ignoring `unlit`) would draw as dark tinted metal.
    They become dielectric (metallic 0, roughness 0.9) with the colour kept.
  - Inner transforms are baked into the vertices, so placements are
    translation + yaw + uniform scale, which satisfies the normal rule by
    construction.
  - Every model is instanced.
  - Models are scaled ×4 (Kenney trees are 1.7 units).
  It reuses `gen_testscene.check()`. That check now exempts flat ground cover
  (under 5 cm, lying on the ground) from the keep-out test: a floor decal
  cannot block the app's boxes or the spawn. The procedural level has none.
  (Loaded scenes no longer get the demo's boxes. The keep-out still includes
  them so every generator gives the same level for a given seed; dropping
  them is a separate, output-changing step.)
  Measured (RX 7800 XT, `profile_standard`, `--bench`, `med`, ~1480 nodes):
  `shadow` 0.08 ms, `geo` 0.23 ms, `frame` 0.36 ms.
  **Detail stress scene:** `tools/gen_detailscene.py` places Poly Haven CC0
  scans (17k–58k triangles, 2048² JPEG PBR textures, alpha-blended grass) at
  1.8M / 6M / 14.5M placed triangles. It exists to find where detailed content
  breaks the engine. Measured:
  - **Triangles are not the wall.** 14.5M placed costs `frame` 2.65 ms at
    pinned clocks, scaling ~linearly, and culling kept instances under the
    8192 budget.
  - **Textures are.** The loader converts each material's textures *per
    primitive* (81 conversions of 12 images: 13.5 s of a 13.9 s debug load)
    and the renderer uploads per material. So ~1 GB of VRAM goes on
    duplicates of 192 MB of unique images, and past the then-fixed 64 slots some
    materials silently fall back to default textures. **Fixed:** the loader
    converts each glTF *image* once and shares it by `Arc` across materials,
    and the renderer uploads one slot per unique image × colour space
    (`plan_texture_slots`). Measured: debug load 13.8 → 2.2 s (release 0.8 →
    0.35 s), 12 uploads for 81 references, `detail_low` VRAM 1,412 → 638 MB
    above idle. Overflow past the cap is now reported instead of silent.
  - **Collision against render meshes** froze the game (see §15's collision
    proxies).
  - No mipmaps and no alpha cutout, visible as shimmer and opaque grass cards.
    **Mipmaps fixed** (full chains + trilinear + anisotropy, §26); memory
    **fixed** by the §17 texture bake (BC7: 64 vs 256 MB). **Alpha cutout
    landed** (§5), but this scene's grass is BLEND with JPEG textures, which
    have no alpha, so it still draws opaque (the load says so).
  **Industrial compound (the STALKER look):** `tools/gen_zonescene.py` writes
  `scratch/zone.glb`, the first scene aimed at the game's intended look
  rather than at a subsystem.
  - **Content:**
    - a yard walled by Soviet precast-panel fence (a gate, missing, fallen and
      leaning panels);
    - a 24 × 22 × 8 m hangar of brick and rusty corrugated iron, with a big
      door, a side door and holes in the roof;
    - a two-storey office block with windows, a door and a walkable stair
      (0.2 m rises);
    - broken asphalt, concrete aprons, mud;
    - Poly Haven props (barriers, barrels, tyres, pipework, utility boxes, a
      compressor, a covered car, assembled electricity poles);
    - dim lamps, and an overcast, hazy `environment` (§13).
  - **Architecture is generated boxes** with a material per face. UVs are
    world metres over each texture's real size (from Poly Haven's metadata),
    so surfaces tile at true scale and adjacent boxes continue one texture.
  - **Textures are pinned as diffuse + GL normal + ARM**, since ARM's G/B are
    the roughness/metal layout `metallicRoughnessTexture` reads. The
    greyscale roughness map the older packs use would read as metallic too.
  - **The hangar stands round the orb demo's five boxes and the old spawn
    point**, which `check()` keeps clear. The boxes themselves no longer
    appear in a loaded scene (below).
  - **Measured** (debug, clocks not pinned, `--bench`, 295 nodes, 818k
    placed triangles):
    - without the bake: load 1.36–1.41 s, frame median 0.59–0.74 ms, 1.8 M
      shadow triangles;
    - baked (LODs + BC7): load 0.41–0.42 s, frame 0.37–0.38 ms, 0.22 M shadow
      triangles, and 58 props collide as LOD trimeshes instead of hulls.
    - Predicted under 3 s / under 1 s to load and ≤ 1.0 ms a frame; all
      right. I'd estimated ~0.5 M placed triangles; it is 818k, since the
      barriers and pole crossarms are dense scans.
  - A throwaway harness walked the game's own schedule from the spawn
    through the gate, the hangar, out of its side door, across the mud, into
    the office and up the stair to the first floor. No slide ran out of
    passes, nothing stuck, and the footsteps were concrete, then grass on the
    mud. Walking it into one of the app's boxes, as a negative control,
    registered as stuck.
  - **Cutout content** (§5): 600 grass tufts (three crossed cards each, a
    procedural texture) on the mud, and a rusty chain-link enclosure round
    the compressor from ambientCG's Fence006 (CC0). glTF wants the cut in
    the base colour's alpha, so the generator merges the fence's colour and
    opacity maps into one RGBA PNG with a stdlib PNG decoder, cached beside
    the pack. A cold run, merge included, takes the same 1.7 s as a warm one;
    I'd expected the pure-Python decode to be slow. The enclosure's panels block the player (a harness walked into
    one and stopped 0.37 m short, and walked through the gate bay).
    - `put()` now centres each prop's *bounds* on its spot, not its origin:
      the compressor's origin is ~4 m from its geometry, which the enclosure
      exposed.
  - **Known look limits:** the ambient light isn't occluded, so interiors are
    too bright; there are no leafy trees yet; and shadows are sharp even
    under an overcast sky (no PCSS).
- **RenderDoc:** in-application API, capture on a keybind.
- **Object naming:** `vkSetDebugUtilsObjectName` on buffers/images/pipelines
  from the start (readable validation + captures).
- **Validation:** core on debug builds; **sync validation** + best-practices as
  toggles (sync validation essential for hand-written sync2 barriers).
  **Landed (§26):** core validation on debug builds, enabled only if the layer is
  installed (otherwise a startup warning, not a failure). Sync validation has
  no in-app toggle yet, but runs via the layer's env var:
  `VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true`. **It is clean — keep it so**
  (under MSAA, but for one known false positive of layer 1.3.275 on the
  prepass's depth resolve, §13 GTAO Limits, which USAGE §8 filters). The
  first run found ~3.4k cross-frame hazards per `--bench` run. The shadow,
  depth, HDR, resolve and LDR images are *single* images shared by every frame
  in flight, yet their `UNDEFINED` transitions used `TOP_OF_PIPE` as the
  source stage. That ordered nothing against the previous frame still writing
  or sampling them. Without FXAA the same mistake left the swapchain
  transition unchained from the acquire semaphore (which waits at
  `COLOR_ATTACHMENT_OUTPUT`). Each transition now takes as its source the
  stages of that image's previous-frame uses. A barrier's first scope covers
  earlier submissions on the queue, so this orders frame N+1 after frame N.
  Verified at 0 messages across default, MSAA 4×, FXAA, shadows-off, all
  three combined, and the main menu; `--bench` timings were unchanged. Any
  *new* per-frame image must follow the same rule.
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
   than 2010" look lands here). (Landed: PBR + textures, analytic-sky IBL,
   4-cascade CSM with 3×3 PCF, and clustered punctual point lights (§12 stage
   B, the engine's first compute pass), caster pancaking, a baked sky-visibility
   volume and GTAO occluding the ambient (§13). Remaining: cubemap IBL.)
4. Physics + FPS controller (rapier), fixed timestep + interpolation → walkable.
   (Landed: fixed timestep + interpolation, the rapier kinematic FPS controller
   against static colliders, and the ECS↔rapier sync systems with the player as an
   ECS entity. Remaining: dynamic bodies — see §26.)
5. Bounded arena content, chunked culling, bindless materials, PBR bake path.
   (Landed: per-view frustum culling, bindless-lite materials, and glTF scene
   loading, the offline bake for textures and meshes, and mesh LODs. Remaining:
  chunked/broad-phase culling.)
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

GPU-driven indirect culling; async compute; dual-quaternion
skinning; animation state machines; local reflection probes / irradiance
volumes; audio occlusion + reverb zones; user-selectable anti-aliasing mode
(SMAA / MSAA 2×/4×; the geometry sample-count seam is in place, §26); TAA;
streaming + stage pipelining; X-Ray (`.ogf`/level) importer for the SoC-rebuild
stretch dream (becomes just another importer feeding the same bake); changing
a level's atmosphere during a session (weather, time of day: §13's constants
would move into the globals UBO).

## 26. Implementation status

Snapshot of what's actually built vs. the design above. The design sections are
unchanged targets; this records reality so the doc doesn't drift. Update as
milestones land.

**On the millisecond figures throughout this document:** they were measured on
the original dev laptop. They are only meaningful *relative to each other* — as
A/B deltas taken back-to-back at one window size on one machine. On different
hardware, re-measure both sides of any comparison before drawing conclusions;
the ratios and the reasoning should carry over, the absolute numbers will not.

### Done (milestones 0–2, most of 3–4: PBR + IBL + sky, CSM, punctual lights, rapier controller, prefabs, menus)

- **Workspace/toolchain**: 7 crates (`bake` is the offline §17 tool;
  `default-members = ["app"]` so plain `cargo run` means the game),
  `rust-toolchain.toml` (stable), GLSL→SPIR-V
  at build time via `shaderc` in `render/build.rs`, embedded from `OUT_DIR`.
- **Deps in use**: ash 0.38, ash-window 0.13, raw-window-handle 0.6, winit 0.30,
  vk-mem 0.4, glam 0.29, bevy_ecs 0.16, serde 1, rapier3d `=0.35.3` (exact
  pin; default features). rapier 0.35 builds on its own newer glam (0.33, via
  `glamx`) rather than nalgebra, so its `Vector`/`Pose` are *not* the workspace's
  glam 0.29 `Vec3` — `app` converts at the boundary (`to_rapier`/`from_rapier`)
  until the workspace glam is bumped to match. Also `gltf 1` (`import`, `utils`,
  `extras` and `names` features, pulling in the `image` crate) and `serde_json 1`
  in `assets` — the latter for §18 prefab params, already in the tree via gltf.
  `app` has `serde_json` as a dev-dependency only. `xxhash-rust` (xxh3) in
  `assets` for the baked-texture content key; `intel_tex_2` (Intel's ISPC BCn
  encoder) in `bake` only, so the engine never links it. `kira =0.12.5` (on
  cpal, Ogg Vorbis decoding only) and `mint` in `app`, for §20 audio.
  Dependencies build optimised even in debug (`[profile.dev.package."*"]`).
  Not yet added: egui, tracy.
- **gfx**: instance/debug messenger/surface/device/queues (on debug builds the
  validation layer + messenger are enabled only when
  `VK_LAYER_KHRONOS_validation` is installed; if missing, a warning is printed
  and the engine runs unvalidated rather than failing instance creation);
  VK 1.3 **dynamic rendering** (feature enabled); swapchain + image views +
  resize; **depth**
  (D32_SFLOAT) created with the swapchain; **vk-mem** allocator held as
  `Arc<Allocator>`; RAII `Buffer`/`Image`/`MappedBuffer` (self-freeing);
  device-local staging upload (`create_device_local_buffer`), persistently-
  mapped host buffers (`create_host_visible_buffer`), and sampled-image upload
  (`create_texture` → RGBA8, single mip, `SHADER_READ_ONLY`); per-frame command
  buffers; sync with per-image `render_finished`; `wait_idle` + ordered teardown
  (resources → allocator → device). `FRAMES_IN_FLIGHT = 2`. VK 1.2
  `shaderSampledImageArrayNonUniformIndexing` enabled (bindless textures).
  Engine-owned **HDR scene-color target** (RGBA16F) + linear sampler, created
  with the swapchain and recreated on resize; `draw_frame` runs two passes —
  geometry into the HDR target, then a caller-supplied post pass into the
  swapchain — with the HDR write→read barrier between.
- **render**: instanced `MeshRenderer` — **several meshes share one** device-local
  vertex + index buffer, each a `{first_index, index_count, vertex_offset}` slice
  keyed by `MeshId`; per-mesh indices stay 0-based (rebased via `vertexOffset`).
  Per-frame instance **SSBO** `InstanceData { model, material_id }` (std430, 80 B)
  at set0/binding0 (vertex); a **resident materials SSBO** `GpuMaterial` (§5: 64 B
  — base color, metallic, roughness, emissive, normal_scale, `tex` = base /
  normal / MR slots) at binding1 (fragment); and a **bindless-lite texture
  array** — a runtime-sized `sampler2D textures[]` at binding2 (the
  `runtimeDescriptorArray` feature), non-uniformly indexed by material. It is
  **sized per level to exactly the textures it uses**: slot 0 = white, slot 1
  = flat normal, then one slot per unique image × colour space, with no
  padding. The only ceiling is the device's combined-image-sampler limit
  (8.4M per stage here). The fixed `textures[64]` it replaced let a level hold
  only ~20 Poly Haven-style models (3 textures each); `gen_testscene.py
  --many-textures` (66 textures) proves the difference: the old binary
  dropped 4 references to defaults, the new one uploads all 66. GPU cost is
  identical (`--bench` A/B). Textures carry a **full mip
  chain**, built on upload by a GPU blit chain (`_SRGB` blits filter in linear
  space, so base-colour mips darken correctly). They're sampled **trilinear
  with 16× anisotropy** (when `samplerAnisotropy` is supported), since mips
  alone blur the ground at grazing angles. Cost measured on the detail scene:
  texture memory 192 → 258 MB (the expected +33%), and **no measurable GPU
  time**. `geo` was flat at `med` and −2% at `high`, because that scene is
  bound by rasterising millions of triangles, not by texture fetch. So it's a
  quality fix (no distant shimmer), not a speed-up there.
  Vertex is pos+normal+uv. `view_proj` + `light_dir` + `camera_pos` in a push
  constant (96 B), **linear-space** Cook-Torrance **PBR** (metallic-roughness)
  directional light with **base-color + normal + metallic-roughness textures**
  (normal mapping via screen-space-derivative TBN, no per-vertex tangent), plus
  **analytic-sky environment ambient** (split-sum IBL: hemisphere irradiance +
  reflection + Karis env-BRDF, no cubemap), written unclamped to the HDR target,
  `cull NONE` + depth. Draw
  sorts `(MeshId, instance)` by (mesh, LOD), the LOD picked per view (§17), and
  emits **one `cmd_draw_indexed` per contiguous run** (`firstInstance` = run start). `TonemapPass` — attributeless
  fullscreen triangle sampling the HDR target, **exposure + Narkowicz ACES**,
  output left linear for the `_SRGB` swapchain to encode; exposure via push
  constant. `SkyPass` — far-plane fullscreen procedural-sky background drawn
  **last** in the geometry pass, depth-tested (`LESS_OR_EQUAL`, no write) so it
  only shades pixels the opaque geometry left uncovered; it reconstructs view rays
  from the inverse view-projection. The reflection path (`mesh.frag`) and the background
  (`sky.frag`) share one palette + soft-glow tuning so IBL reflections match the
  visible sky; only the background adds a sharp sun disk (a disk in the
  reflection would double-count against the analytic sun on smooth metals).
  Opaque `mesh.frag` also applies **exp distance fog** toward that same sky along
  the view ray, hiding the finite ground edge and reading as depth. A **directional
  sun shadow** (§11) precedes the geometry pass: a depth-only pass renders the
  scene from a fixed light ortho into a 4096² map, which `mesh.frag` samples with
  3×3 PCF to occlude the direct sun term. `draw_frame` now runs three passes
  (shadow → geometry → post); the mesh renderer splits into `prepare_frame`
  (CPU sort/stage) + `draw_shadow`/`draw_main` replaying the same instance runs.
  Normals use the cofactor (inverse-transpose) of the model matrix, so any node
  TRS shades correctly, mirrored included (§6); the test level's `normals` zone
  pairs transformed nodes with baked twins as the visual check.
- **assets**: Vulkan-free CPU mesh types (`Vertex` = pos+normal+uv, `MeshData` +
  bounds + `Material` with optional decoded base-color / normal / MR textures),
  procedural `uv_sphere` + `cube`, and a **glTF/GLB scene loader**
  (`load_gltf_scene`, via `gltf::import_buffers`) returning `SceneData { meshes, nodes }`
  — one `MeshData` per referenced (mesh, primitive) pair in **local** space with
  **its own material**, plus one `SceneNode` per placement carrying that node's
  world transform. Primitives are deduplicated, so a mesh used by N nodes is
  stored once and drawn as N instances. Reads UVs, computes normals when absent,
  reads base-color/normal/MR images as encoded bytes (decoded lazily, §17), and
  resolves .glb / external / data-URI buffers and images. Covered by the crate's first unit tests (a
  synthesized glTF fixture asserting dedup and transform inheritance).
- **app**: winit loop; bevy_ecs world + multi-threaded schedule (`integrate`,
  `tick`) driven on a **fixed timestep** (accumulator + `FIXED_DT`, frame delta
  clamped and steps capped as a spiral-of-death guard) so sim speed no longer
  tracks framerate; a **first-person controller on rapier** stepped in the same
  loop (WASD walk with accel toward a target speed, own gravity, edge-latched
  jump when grounded; a `KinematicCharacterController` moves a capsule against
  the level's colliders and `PhysicsPipeline::step` runs once per fixed tick;
  `V` toggles a noclip fly) — the player is an **ECS entity** whose fixed-step
  state is driven by the two §15 sync systems, with input arriving as the §14
  `InputState` resource; look is render-rate for responsive aim (its own `Look`
  component), body position is fixed-step and camera-**interpolated**; inline **extract** (`World` query → sorted
  `Vec<(MeshId, InstanceData)>` each frame) that **interpolates** each entity's
  double-buffered sim state (prev/curr position + spin angle) by
  `alpha = accumulator / FIXED_DT` and reads a per-entity `Scale`; a small
  **static level** (ground box + obstacle boxes, each with a fixed cuboid
  collider in rapier; the boxes only in the orb demo, since a loaded scene is
  the level, which `a_loaded_level_has_no_demo_boxes` checks; no `Velocity`/`Spin` so `integrate` skips them) drawn
  through the same instanced path; a **depth prepass** (§10) ahead of the opaque
  draws, which then run depth-write-off + `LESS_OR_EQUAL` (−42% on the geometry
  pass);
  `[`/`]` adjust tonemap exposure at runtime; the mesh registry always begins with
  three built-ins (demo sphere, demo cube, and the unit cube every static level
  piece is scaled from, auto-fitted to the grid), and **glTF paths on the CLI are
  loaded as scenes** appended after them — one entity per node with a `Transform`,
  its primitive's own material, and a trimesh `ColliderRef`, so a loaded level is
  walkable. The app opens on the main menu with nothing loaded (§19); NEW GAME
  builds a session from the CLI scenes. A session is `build_world` (scenes,
  bake, world, player, level, schedule: all CPU) plus the GPU upload, so the
  end-to-end tests build exactly the game's world from a glTF file with no
  device. Giving a scene suppresses the orb demo
  (1000 orbs would bury it); with no scene arguments NEW GAME builds the orb demo,
  each orb taking a random built-in mesh + palette material. The orb demo is a *pathological* workload and is no
  longer what perf work should be measured against — `tools/gen_testscene.py`
  (§21) generates the realistic one. Culling uses a **per-mesh local bounding sphere**
  mapped through the model matrix, rather than a blanket radius — required because
  scene meshes are not unit-fitted. Limits: unique textures are
  bounded only by the device's descriptor limits (deduplicated per image ×
  colour space; past the limit, references use the defaults with a `[mesh]`
  warning), a collider is built per node at load (an
  exact trimesh up to 2048 triangles; above that, a baked LOD trimesh within
  5 cm, else a convex hull; overridable per node, §15; no convex
  *decomposition*), scenes land at their authored
  coordinates so a model authored around the origin floats above the demo ground
  at `GROUND_Y`, and there is still no broad phase, so a large scene leans on
  the per-entity cull (and on LODs, §17, for its triangle count).
- **Build order (§23)**: step 1 done; step 2 done; step 3 mostly — PBR direct
  lighting + full textures + analytic-sky IBL + 4-cascade CSM + punctual lights,
  sky visibility and GTAO (§13), *not* cubemap IBL;
  step 4 mostly landed — fixed timestep + interpolation and the rapier kinematic
  FPS controller against static colliders (ECS↔rapier sync systems and dynamic
  bodies still pending, see below). Steps 5+ not started.

### Current simplifications to revisit

- **Extract seam (§4)**: extract is inline in `app`, writing straight to the
  instance SSBO each frame; sim-state interpolation (prev↔curr by `alpha`) now
  happens here, as the design specifies. No double-buffered `RenderFrame` struct
  yet; stages run sequentially (stage pipelining deferred by design).
- **Timestep (§4)**: **landed.** Sim runs on a fixed-`FIXED_DT` accumulator with
  render **interpolation** (`alpha = accumulator / FIXED_DT`, computed at
  extract) over per-entity double-buffered state (`Prev*` / current). Toroidal
  wrap shifts `prev` with `curr` so the interpolated segment never streaks across
  the seam; frame delta is clamped and steps capped (spiral-of-death guard).
  Camera stays render-rate (matches design). The remaining §4 gap is the
  double-buffered **`RenderFrame` snapshot** for stage pipelining — extract is
  still inline. This was the groundwork physics needed, and rapier now runs on it
  for the player (spin is still a cosmetic angular velocity, not a rigid body).
- **FPS controller (§15)**: **landed on rapier.** rapier's state is one raw ECS
  `Resource` (`Physics`: pipeline, integration params, islands, broad/narrow
  phase, body/collider/joint sets, CCD — no gravity, the controller is
  kinematic). The player is a position-based kinematic body with a **capsule**
  (r 0.35, h 1.8); each fixed tick `step_player` snapshots `prev_pos`, chases the
  target speed, applies gravity **only while airborne** (zeroed when grounded;
  jump on the press edge when grounded), shape-casts the desired motion with
  `KinematicCharacterController::move_shape` (slide, snap-to-ground, autostep up
  to 0.4), sets the kinematic target, calls `physics.step()`, and reads the body
  position back. Horizontal velocity is re-derived from the motion that actually
  happened, so blocked axes lose their speed. The ground and every obstacle box
  (the demo's only) are real fixed cuboid colliders (`spawn_static`); one warm-up step publishes
  them to the broad-phase BVH the controller queries. Look is render-rate; the
  body interpolates like any other entity. The player is a **normal ECS entity**
  (§1): sim state in a `Player` component, render-rate angles in `Look`, driven by
  the **two sync systems §15 asks for**, chained as `player_target_sys` (ECS→rapier:
  run the controller, push the kinematic target) → `physics_step_sys` →
  `player_readback_sys` (rapier→ECS: body translation back into the component).
  Input arrives as the §14 `InputState` resource rather than being read off `App`.
  Level entities carry a `ColliderRef` handle. Scene colliders also carry their
  §20 footstep surface in rapier's `user_data`, and a grounded player sweeps its
  bottom sphere down each tick to read it into `Player.surface` (§15). Before
  each grounded move, the motion is bent round any downward-facing surface
  the top of the capsule touches or would reach this tick (shallow ones only
  once a slide has given up), which otherwise stalled the controller (§15). The controller maths stay plain functions (`player_target`/`player_readback`) that
  the systems wrap, so the physics tests drive the real logic. Still open: no `InteractionGroups` layers
  (nothing to separate yet); the drifting orbs and their spin are still
  non-physical (cosmetic, not rigid bodies). Known limits: the ground is a finite 80×80 box, so walking
  off the edge falls forever (no kill-plane/respawn); slope climbing/sliding uses
  rapier's 45° defaults but no level geometry exercises it; a jump that clips a
  box rim counts as grounded (vertical speed zeroed) and the capsule clambers
  up rather than continuing the arc; turning noclip off while standing inside
  geometry only depenetrates when not moving; default rapier features (no
  `enhanced-determinism`), so cross-machine replay isn't guaranteed yet; pushing
  into another kinematic body can still run the controller's slide loop to its
  20-pass cap (§15), though there are none yet.
- **Descriptors (§9)**: one set with three bindings — per-frame instances
  (binding 0), resident materials (binding 1), and a resident
  `sampler2D textures[]` (binding 2) sized per level, as N discrete per-frame
  sets. It has since grown the shadow map, globals, lights and cluster masks
  (bindings 3–6), the sky-visibility volume (binding 7) and GTAO's
  half-resolution result and depth levels (bindings 8 and 9). Not yet the
  Set 0 (resident) / Set 1 (per-frame ring) / Set 2 (per-view) split, and the
  texture array is sized and filled at load — **not** update-after-bind /
  partially-bound (fine until streaming; no runtime texture loading yet).
- **Push constants**: currently carry `view_proj` + `light_dir` + `camera_pos`
  (96 B, provisional). Design reserves push constants for tiny per-draw scalars
  and puts camera in a per-frame UBO — revisit when Set 1 lands.
- **Vertex layout (§6)**: pos+normal+uv. Normal mapping uses a screen-space
  **derivative TBN** (no per-vertex tangent); MikkTSpace vertex tangents are the
  higher-quality follow-up (§6 reserves the tangent attribute).
- **Lighting (§3, §13)**: **linear-space** Cook-Torrance **PBR** (metallic-
  roughness) for one directional light plus **analytic-sky IBL** ambient (split-sum:
  hemisphere irradiance for diffuse, reflection-vector sky sample for specular,
  Karis analytic env-BRDF), into an RGBA16F HDR target resolved by an ACES
  **tonemap** pass. The environment is a **procedural sky evaluated in-shader**
  (its palette, the sun and the fog are now per level, §13's environment),
  not a precomputed cubemap — so no arbitrary HDR environments and the specular
  "prefilter" is a crude roughness lerp (no real GGX convolution/mips). The sky
  *is* now drawn as a visible background (SkyPass) matching the reflected
  environment. Real cubemap IBL (equirect→cube, irradiance/prefilter passes,
  BRDF LUT) is the follow-up. Sun shadows (§11 CSM) and punctual point lights
  (§12) now exist. Bloom and auto-exposure landed (§13). The ambient is
  occluded by a baked per-level sky-visibility volume (§13, §17) and by the
  lower of half-resolution GTAO and the material's own AO (glTF
  `occlusionTexture`, §13). The tonemap curve is a drop-in point for AgX.
  **Known artifact — specular singularity on smooth metal.** A punctual light has
  zero area, so on low-roughness metal (the PBR grid bottoms out at 0.06) its
  specular lobe collapses to a near-singular bright dot. With a geometry-free
  marker light the source itself is invisible, so it reads as the reflection of a
  sun that isn't there. Correct for the model, wrong-looking in practice. The
  standard fix is Karis's representative-point approximation: treat each light as
  a sphere of some source radius and renormalise the specular lobe, which spreads
  the highlight into a believable disc. Independent of clustering.
- **HDR/depth targets (§9)**: single engine-owned images shared across both
  frames-in-flight (matches the design: render targets are engine-owned, not
  per-frame). With `FRAMES_IN_FLIGHT = 2` this carries a latent cross-frame WAW
  hazard on the shared targets — pending the sync2 barrier + timeline-semaphore
  pass (§10). The intra-frame HDR write→read hazard *is* handled.
- **Anti-aliasing (§3, §10, §13)**: none yet — rendering is single-sample
  throughout. The groundwork is a **single knob**, `MSAA_SAMPLES` /
  `Renderer::samples()`, that the HDR + depth targets and the mesh/sky pipelines
  all read (the tonemap pass to the swapchain stays 1× on purpose). Flipping it
  is *not* sufficient on its own: a multisampled HDR target needs a **resolve
  attachment** (multisample → single-sample) before the tonemap pass can sample
  it. SMAA (the §13 post-AA alternative) would slot in as tonemap → LDR → SMAA →
  swapchain. **MSAA now exists** (§13): `--msaa N` at startup, clamped to device
  support, with a multisampled HDR/depth pair resolved into a single-sample image
  by dynamic rendering. The `samples()` seam did its job — the `render` crate
  needed no edits, since its pipelines already read it. **Switchable from the
  main menu** (§19): the two pipelines that bake the sample count belong to a
  `Session`, so changing it with no world loaded needs only
  `Renderer::set_msaa` (idle, clamp, recreate the depth/HDR/resolve targets) —
  the tonemap re-binds itself, since it refreshes per frame and early-outs when
  the view is unchanged. Still pending: **SMAA** (so there is no AA *mode
  choice* yet, only a sample count), switching *during* a session, and a
  **tonemapped resolve** to stop bright HDR edges sparkling through the
  averaging resolve.
- **Geometry / assets (§6, §7)**: a **mesh registry**, a **material table**, and a
  full **base-color + normal + metallic-roughness texture** path now exist —
  several meshes in shared vertex/index buffers drawn by sorted per-mesh runs; a
  resident `materials[]` SSBO indexed by per-instance `material_id`; and a fixed
  bindless `textures[]` array (glTF images or white/flat-normal defaults),
  consumed by a Cook-Torrance PBR BRDF. Mips landed (GPU-built, or baked BC7,
  §17). Still missing: MikkTSpace vertex tangents (normal mapping is
  derivative-based), pipeline buckets (one pipeline for everything),
  skinning/animation, and the rest of the offline **bake** path (textures
  landed; runtime blob, handle tables, load-time allocator not). glTF loads directly each
  run; a session's meshes/materials/textures are now **freed on returning to the
  main menu** (§19) — the first teardown path — but nothing is *streamed*, and
  freeing is still all-or-nothing per session; one material per merged glTF (first primitive
  wins).

### Not yet started

Culling (per-view bounding-sphere frustum cull landed for the camera + shadow
views, §8 — broad-phase/chunk cull, rayon parallelism, AABB refinement, and LOD
still pending; LOD landed, §17); MikkTSpace vertex tangents (mips landed); pipeline buckets (PBR BRDF +
base-color/normal/MR textures landed); clustered lighting (**landed** — a `point_light` prefab, a lights SSBO, and a
16×9×24 cluster grid assigned by one compute dispatch into per-cluster light
bitmasks, plus sphere-light specular via `source_radius`, §12; spot lights and
point-light shadows pending);
precomputed cubemap/HDR IBL (analytic-sky IBL landed);
shadows: **4-cascade CSM + 3×3 PCF landed** (§11) — practical splits, sphere-fit
and texel-snapped per cascade, a depth array layer each, per-cascade caster
culling, projection-based selection with an edge blend, and normal-offset bias;
caster pancaking (depth clamp + near-plane-free caster culling); **PCSS still
pending**, and since array layers share an extent all
cascades are the same resolution. Shadows now reach `SHADOW_DISTANCE` (60) rather
than a ±16 box;
transparents (alpha
*cutout* landed, §5; BLEND still draws opaque); asset
bake pipeline (runtime glTF scene loading + multi-mesh registry landed, and
the **texture bake** — BC7 mip chains in a content-addressed cache, §17 — then
the mesh LOD bake and a per-scene **sky-visibility volume**, §17; material /
scene bake, runtime blob and handle tables are not); **scene spawning landed**
(§18: `extras` → `PrefabSpec`, a prefab registry, marker nodes, `player_start`
and a parameterised `prop` with per-node collider/shadow opt-outs; unknown ids
fall back to static geometry; chunk membership and light/trigger prefabs
pending) / save; rapier beyond the player (kinematic FPS controller, static
colliders and the ECS↔rapier sync systems landed — dynamic bodies and collision
layers pending);
skinning; UI/HUD (§19's lightweight quad/text renderer + the Esc pause menu
landed, with keyboard *and* mouse navigation, an OPTIONS screen tree, and live
display-mode/shadow-quality/FXAA controls under GRAPHICS — MSAA is changeable
there only from the main menu, since the session's mesh and sky pipelines bake
the sample count; all of them, plus the GAMEPLAY field of view, persist in
`config/graphics.toml` (§13), and key bindings + mouse look live in
`config/controls.toml` (§14), edited from OPTIONS > CONTROLS / GAMEPLAY, whose
lists scroll when the window is short; egui dev UI, SDF text and
lower-case/punctuation pending); audio (**landed**, §20: kira mixer, recorded
footsteps/jump/landing (Kenney, CC0) with synthesised fallbacks, footsteps per
surface from glTF material tags, spatial lamp hums, live SOUND volumes; music
and occlusion pending); debug/profiling tooling (per-pass GPU timestamp timing
landed — stderr log + `Renderer::gpu_times`, plus the `--bench` sweep harness;
Tracy / RenderDoc / egui overlay and
CPU-side zones pending); GPU-driven culling; streaming; stage pipelining;
anti-aliasing (**MSAA** `--msaa N` at startup and **FXAA** on `F2` both landed,
§13 — **SMAA** is still absent, as are live MSAA switching and a tonemapped
resolve to stop bright HDR edges sparkling).
