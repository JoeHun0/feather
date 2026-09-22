# Feather — Engine Architecture

Status: in active implementation — a textured-PBR forward renderer with IBL,
4-cascade sun shadows with caster pancaking, clustered punctual point lights
with sphere-light specular, a depth prepass, per-view frustum culling,
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
AABB refinement), and with **no broad phase / chunk culling or LOD** yet. Note the
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
straight to the `Look` component. Still raw winit key polling rather than a
bindings table; no gamepad, no UI focus flag.

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
loads scenes with the runtime's own loader and collects every (image, colour
space) pair actually used: base colour as sRGB, normal and MR as data. For
each it builds a mip chain on the CPU (2×2 box, **sRGB averaged in linear
space**, edge-clamped for odd sizes), pads each level to 4×4 blocks, and
**BC7-encodes** it with Intel's ISPC encoder (`intel_tex_2`, bake crate only,
so the engine never links it). It writes a thin header + raw blocks per level
to `scratch/bake/tex/<key>.bc7`. The key (`feather_assets::bake::texture_key`)
is xxh3 of the *decoded* pixels + size + colour space + `BAKE_VERSION`, so
it's content-addressed and incremental, and cross-scene dedup comes free. The
runtime computes the same key per unique texture and uploads the baked chain
when one exists and the device has `textureCompressionBC`. **Otherwise raw
RGBA8 with GPU-built mips, never an error.** `--no-bake` forces raw for A/B.
Baked files are validated on read (magic, version, per-level sizes against
the dimensions) and written via temp + rename, so an interrupted bake is
never trusted. Measured on the detail scene: **64 MB of BC7 vs 256 MB raw**
(4×), 12 textures baked in 4.3 s, a re-run skips all 12, validation clean
(including 1×1/2×2 partial-block levels), and `geo` identical.
Not yet: BC5 for normal maps (needs z reconstruction in the shader); keying on
*source* bytes so baked scenes can skip JPEG decode at load (today the decode
still runs to compute the key); meshes, materials and scenes (the rest of
this section).

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

**Unknown ids warn once and fall back to static geometry** rather than failing,
which is what lets a scene be authored ahead of the engine. Extras parsing is
lenient for the same reason: malformed data costs one prop, not the level.

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
(CONTINUE / OPTIONS / EXIT) → OPTIONS (GRAPHICS / SOUND / GAMEPLAY) → each
submenu. **GRAPHICS carries real settings** — shadow quality cycles and FXAA
toggles live, both sharing the exact code path F1/F2 use, so the two cannot
drift. MSAA is shown with its current sample count and a RESTART note but is
**inert**: the count is baked into every geometry pipeline, so changing it live
means rebuilding them all. Showing the value honestly beats offering a control
that silently does nothing. SOUND and GAMEPLAY are placeholder rows — §20 audio
is unstarted and there are no gameplay settings to bind to yet.

Navigation state lives in a `Menu` struct that depends on **neither the renderer
nor the event loop**: it mutates settings and returns a `MenuOutcome` the app
acts on. That keeps the whole thing unit-testable without a GPU, which is how
the screen tree, wrap-around, BACK-restores-selection and the inert rows are all
covered. Esc walks back one screen at a time and only unpauses from the root.

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

## 21. Debug / profiling

- **CPU:** Tracy (`tracy-client`), scoped zones on systems + passes.
- **GPU:** timestamp queries bracketing passes → per-pass ms. This is how
  SSR/volumetrics cost gets judged. **Landed** (§26): a timestamp query pool in
  `gfx` brackets the shadow, geometry, and post passes, reads back after the frame
  fence (no stall), and logs smoothed per-pass ms to stderr (`[gpu] shadow … cluster … geo …
  post … frame …`), with a `Renderer::gpu_times` accessor for a future overlay.
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
  identical, confirming clock state was the only distortion.
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
    **fixed** by the §17 texture bake (BC7: 64 vs 256 MB); alpha cutout
    pending until content needs it (no current asset has alpha).
- **RenderDoc:** in-application API, capture on a keybind.
- **Object naming:** `vkSetDebugUtilsObjectName` on buffers/images/pipelines
  from the start (readable validation + captures).
- **Validation:** core on debug builds; **sync validation** + best-practices as
  toggles (sync validation essential for hand-written sync2 barriers).
  **Landed (§26):** core validation on debug builds, enabled only if the layer is
  installed (otherwise a startup warning, not a failure). Sync validation has
  no in-app toggle yet, but runs via the layer's env var:
  `VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true`. **It is clean — keep it so.** The
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
   B, the engine's first compute pass), and caster pancaking. Remaining: GTAO
   and cubemap IBL.)
4. Physics + FPS controller (rapier), fixed timestep + interpolation → walkable.
   (Landed: fixed timestep + interpolation, the rapier kinematic FPS controller
   against static colliders, and the ECS↔rapier sync systems with the player as an
   ECS entity. Remaining: dynamic bodies — see §26.)
5. Bounded arena content, chunked culling, bindless materials, PBR bake path.
   (Landed: per-view frustum culling, bindless-lite materials, and glTF scene
   loading. Remaining: chunked/broad-phase culling, LOD, and the offline bake.)
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
volumes; audio occlusion + reverb zones; user-selectable anti-aliasing mode
(SMAA / MSAA 2×/4×; the geometry sample-count seam is in place, §26); TAA;
streaming + stage pipelining; X-Ray (`.ogf`/level) importer for the SoC-rebuild
stretch dream (becomes just another importer feeding the same bake).

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
  encoder) in `bake` only, so the engine never links it. Dependencies build
  optimised even in debug (`[profile.dev.package."*"]`). Not yet added: kira,
  egui, tracy.
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
  sorts `(MeshId, instance)` by mesh, emits **one `cmd_draw_indexed` per
  contiguous run** (`firstInstance` = run start). `TonemapPass` — attributeless
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
  (`load_gltf_scene`, via `gltf::import`) returning `SceneData { meshes, nodes }`
  — one `MeshData` per referenced (mesh, primitive) pair in **local** space with
  **its own material**, plus one `SceneNode` per placement carrying that node's
  world transform. Primitives are deduplicated, so a mesh used by N nodes is
  stored once and drawn as N instances. Reads UVs, computes normals when absent,
  decodes base-color/normal/MR textures to RGBA8, and resolves .glb / external /
  data-URI buffers and images. Covered by the crate's first unit tests (a
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
  collider in rapier; no `Velocity`/`Spin` so `integrate` skips them) drawn
  through the same instanced path; a **depth prepass** (§10) ahead of the opaque
  draws, which then run depth-write-off + `LESS_OR_EQUAL` (−42% on the geometry
  pass);
  `[`/`]` adjust tonemap exposure at runtime; the mesh registry always begins with
  three built-ins (demo sphere, demo cube, and the unit cube every static level
  piece is scaled from, auto-fitted to the grid), and **glTF paths on the CLI are
  loaded as scenes** appended after them — one entity per node with a `Transform`,
  its primitive's own material, and a trimesh `ColliderRef`, so a loaded level is
  walkable. The app opens on the main menu with nothing loaded (§19); NEW GAME
  builds a session from the CLI scenes. Giving a scene suppresses the orb demo
  (1000 orbs would bury it); with no scene arguments NEW GAME builds the orb demo,
  each orb taking a random built-in mesh + palette material. The orb demo is a *pathological* workload and is no
  longer what perf work should be measured against — `tools/gen_testscene.py`
  (§21) generates the realistic one. Culling uses a **per-mesh local bounding sphere**
  mapped through the model matrix, rather than a blanket radius — required because
  scene meshes are not unit-fitted. Limits: unique textures are
  bounded only by the device's descriptor limits (deduplicated per image ×
  colour space; past the limit, references use the defaults with a `[mesh]`
  warning), a collider is built per node at load (an
  exact trimesh up to 2048 triangles and a convex hull above that, overridable
  per node, §15; no convex *decomposition*), scenes land at their authored
  coordinates so a model authored around the origin floats above the demo ground
  at `GROUND_Y`, and there is still no broad-phase/LOD, so a large scene leans on
  the per-entity cull.
- **Build order (§23)**: step 1 done; step 2 done; step 3 mostly — PBR direct
  lighting + full textures + analytic-sky IBL + 4-cascade CSM + punctual lights,
  *not* the cluster grid, GTAO or cubemap IBL;
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
  are real fixed cuboid colliders (`spawn_static`); one warm-up step publishes
  them to the broad-phase BVH the controller queries. Look is render-rate; the
  body interpolates like any other entity. The player is a **normal ECS entity**
  (§1): sim state in a `Player` component, render-rate angles in `Look`, driven by
  the **two sync systems §15 asks for**, chained as `player_target_sys` (ECS→rapier:
  run the controller, push the kinematic target) → `physics_step_sys` →
  `player_readback_sys` (rapier→ECS: body translation back into the component).
  Input arrives as the §14 `InputState` resource rather than being read off `App`.
  Level entities carry a `ColliderRef` handle. The controller maths stay plain
  functions (`player_target`/`player_readback`) that the systems wrap, so the
  physics tests drive the real logic. Still open: no `InteractionGroups` layers
  (nothing to separate yet); the drifting orbs and their spin are still
  non-physical (cosmetic, not rigid bodies). Known limits: the ground is a finite 80×80 box, so walking
  off the edge falls forever (no kill-plane/respawn); slope climbing/sliding uses
  rapier's 45° defaults but no level geometry exercises it; a jump that clips a
  box rim counts as grounded (vertical speed zeroed) and the capsule clambers
  up rather than continuing the arc; turning noclip off while standing inside
  geometry only depenetrates when not moving; default rapier features (no
  `enhanced-determinism`), so cross-machine replay isn't guaranteed yet.
- **Descriptors (§9)**: one set with three bindings — per-frame instances
  (binding 0), resident materials (binding 1), and a resident
  `sampler2D textures[]` (binding 2) sized per level, as N discrete per-frame
  sets. Not yet the
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
  **tonemap** pass. The environment is a **procedural sky evaluated in-shader**,
  not a precomputed cubemap — so no arbitrary HDR environments and the specular
  "prefilter" is a crude roughness lerp (no real GGX convolution/mips). The sky
  *is* now drawn as a visible background (SkyPass) matching the reflected
  environment. Real cubemap IBL (equirect→cube, irradiance/prefilter passes,
  BRDF LUT) is the follow-up. Sun shadows (§11 CSM) and punctual point lights
  (§12) now exist. Still missing: auto-exposure and bloom. The tonemap curve is a
  drop-in point for AgX.
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
still pending); MikkTSpace vertex tangents (mips landed); pipeline buckets (PBR BRDF +
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
bloom + auto-exposure (HDR target + tonemap now in place); transparents; asset
bake pipeline (runtime glTF scene loading + multi-mesh registry landed, and
the **texture bake** — BC7 mip chains in a content-addressed cache, §17; mesh /
material / scene bake, runtime blob and handle tables are not); **scene spawning landed**
(§18: `extras` → `PrefabSpec`, a prefab registry, marker nodes, `player_start`
and a parameterised `prop` with per-node collider/shadow opt-outs; unknown ids
fall back to static geometry; chunk membership and light/trigger prefabs
pending) / save; rapier beyond the player (kinematic FPS controller, static
colliders and the ECS↔rapier sync systems landed — dynamic bodies and collision
layers pending);
skinning; UI/HUD (§19's lightweight quad/text renderer + the Esc pause menu
landed, with keyboard *and* mouse navigation, an OPTIONS screen tree, and live
shadow-quality/FXAA controls under GRAPHICS — MSAA is changeable there only
from the main menu, since the session's mesh and sky pipelines bake the sample
count; egui dev UI, SDF text and
lower-case/punctuation pending); audio; debug/profiling tooling (per-pass GPU timestamp timing
landed — stderr log + `Renderer::gpu_times`, plus the `--bench` sweep harness;
Tracy / RenderDoc / egui overlay and
CPU-side zones pending); GPU-driven culling; streaming; stage pipelining;
anti-aliasing (**MSAA** `--msaa N` at startup and **FXAA** on `F2` both landed,
§13 — **SMAA** is still absent, as are live MSAA switching and a tonemapped
resolve to stop bright HDR edges sparkling).
