#!/usr/bin/env python3
"""Compose a STALKER-like test level: an abandoned industrial compound.

Why this exists
---------------
The game is meant to look like a STALKER-style shooter, and no test scene
did: the others are procedural boxes, a stylised Kenney forest and a detail
stress test. This one is the look under test: weathered precast concrete,
brick, rust, broken asphalt and mud (Poly Haven CC0 scans), a grey overcast
sky with haze (an `environment` marker, ARCHITECTURE.md §13), and buildings
you can walk into and climb. Like the other generators it writes one
self-contained level with §18 prefab `extras`, so the engine needs no changes
to load it.

    python3 tools/fetch_assets.py
    python3 tools/gen_zonescene.py --check
    cargo run --release -p feather-bake -- scratch/zone.glb   # optional: LODs
    cargo run -- scratch/zone.glb

What is in it
-------------
* A yard walled by Soviet precast-panel fence (4 m bays of 2.5 m panels on
  posts), with a gate on the south side, two missing panels, one fallen flat
  and one leaning.
* A hangar (24 x 22 x 8 m): brick to 3 m, rusty corrugated iron above and on
  the roof, a big door facing the gate, a side door, holes in the roof. It
  stands round where the orb demo's boxes and spawn point are, which
  `gen_testscene.check` keeps clear.
* A two-storey office block (12 x 8 m): precast outside, worn plaster inside,
  window and door openings, a stair (0.2 m rises, within autostep) up to the
  first floor, a parapet roof.
* Broken asphalt from the map's edge through the gate to the hangar, concrete
  aprons, mud everywhere else.
* Props: road barriers at the gate, barrel clusters (some tipped over), tyre
  stacks, utility boxes and pipework on the walls, a compressor, a covered car
  in the hangar, and the kit's three assembled electricity poles in lines
  outside the fence.
* Cutouts (ARCHITECTURE.md §5): ~600 grass tufts on the mud, three crossed
  cards each, from a texture this script draws, and a rusty chain-link
  enclosure round the compressor, from ambientCG's Fence006. glTF takes a
  cutout's alpha from the base colour, so the fence's colour and opacity maps
  are merged into one RGBA PNG (a stdlib decoder; the result is cached beside
  the pack).
* Trees and bushes (§5 cutouts): birch-like trees and bushes, drawn and built
  here, thick in the belt outside the fence and scattered in the yard. Leaf
  cards are single-sided MASK with normals pointing out of the crown: the
  main pass doesn't cull, so both sides draw, lit as one soft volume. Trunks
  collide; crowns and bushes don't.
* A pond in the yard west of the hangar (ARCHITECTURE.md §10's water): a
  hollow in the mud, its bank sloping to a bed ~1.1 m down, under dark
  water 15 cm below the mud. The level brings its own ground
  (`environment`'s `ground: false`), since the app's slab would fill the
  hollow.
* Dim warm lamps in the hangar and the office, `player_start` outside the
  gate, and the overcast `environment`.

How surfaces are built
----------------------
Architecture is axis-aligned boxes, each face with its own material. UVs are
**world metres divided by the texture's real size** (from Poly Haven's
metadata: the precast panel is 4 m, the brick 1.5 m), so every surface tiles
at true scale and adjacent boxes continue one texture. Each texture is used as
diffuse (sRGB base colour), GL normal, and ARM, whose G/B channels are the
roughness/metal layout glTF's metallicRoughnessTexture reads. Materials carry
their footstep surface (§20): mud steps like grass, the rest like concrete.

Props are Poly Haven models copied with their own materials, merged per model
(gen_detailscene.Level.model), seated on whatever they stand on.

--check reuses gen_testscene.check(), so every generator enforces the same
engine constraints.
"""

import argparse
import math
import os
import random
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_detailscene as gds  # noqa: E402  (Level: glTF writing, model import)
import gen_testscene as gts  # noqa: E402  (constants + the shared check)

ASSETS = gds.ASSETS
G = gts.GROUND_Y
HALF = gts.GROUND_HALF

# Tiling textures: Poly Haven id, real size in metres (api.polyhaven.com/info),
# footstep surface (§20).
TEXTURES = {
    "precast": ("preconcrete_wall_001", 4.0, "concrete"),
    "brick": ("factory_brick", 1.5, "concrete"),
    "corrugated": ("rusty_corrugated_iron", 2.0, "concrete"),
    "rust": ("rusty_metal_02", 1.0, "concrete"),
    "floor": ("concrete_floor_damaged_01", 5.0, "concrete"),
    "asphalt": ("road_damaged", 2.2, "concrete"),
    "mud": ("brown_mud_leaves_01", 1.3, "grass"),
    "plaster": ("worn_plaster_wall", 1.8, "concrete"),
}

# Props: Poly Haven 1K glTFs under the asset cache.
PROPS = {
    "barrier": "polyhaven_concrete_road_barrier/concrete_road_barrier_1k.gltf",
    "barrel_a": "polyhaven_Barrel_01/Barrel_01_1k.gltf",
    "barrel_b": "polyhaven_barrel_03/barrel_03_1k.gltf",
    "tyre": "polyhaven_old_tyre/old_tyre_1k.gltf",
    "utility_box": "polyhaven_utility_box_01/utility_box_01_1k.gltf",
    "pipes": "polyhaven_modular_industrial_pipes_01/modular_industrial_pipes_01_1k.gltf",
    "compressor": "polyhaven_old_military_compressor/old_military_compressor_1k.gltf",
    "car": "polyhaven_covered_car/covered_car_1k.gltf",
    "poles": "polyhaven_modular_electricity_poles/modular_electricity_poles_1k.gltf",
}
# The pole kit's assembled presets: node-name prefix and the pole's x in the
# kit (its base goes to the origin).
POLE_PRESETS = {"pole_1": ("preset_01_", -2.5), "pole_2": ("preset_02_", -4.5),
                "pole_3": ("preset_03_", -6.5)}
# Kit parts smaller than this (bolts, nuts, rings) aren't worth a draw.
POLE_MIN_PART = 0.05

# Ground-level slabs: thin enough to count as ground cover for the check, and
# stacked so they never z-fight: mud on the app's slab, paving on the mud.
MUD_TOP = G + 0.02
PAVE_TOP = G + 0.04

# Buildings (x0, x1, z0, z1), outer faces.
HANGAR = (-12.0, 12.0, -9.0, 13.0)
HANGAR_WALL = 0.4
HANGAR_BRICK = 3.0  # brick up to here, corrugated iron above
HANGAR_HEIGHT = 8.0
HANGAR_DOOR = (-4.0, 4.0, 6.0)  # x0, x1, height (south wall)
OFFICE = (16.0, 28.0, -20.0, -12.0)
OFFICE_WALL = 0.3
STOREY = 3.2  # floor to floor
SLAB = 0.2
STEP_RISE, STEP_RUN = 0.2, 0.3
FENCE = (-32.0, 32.0, -28.0, 28.0)
FENCE_BAY = 4.0
PANEL = (3.7, 2.5, 0.16)  # length, height, thickness
POST = (0.3, 2.8)  # side, height
GATE = (-4.0, 4.0)  # x span of the gate in the south fence

# Overcast, grey-green, hazy (§13): starting values, to be tuned by eye.
ENVIRONMENT = {
    # The level brings its own ground: the app's slab would fill the pond.
    "ground": False,
    # ... and digs below GROUND_Y (the pond bed), so the kill plane is lower
    # than the demo default: a fall past the mud kills, wading doesn't.
    "kill_y": G - 5.0,
    "sun_elevation": 35.0, "sun_azimuth": 210.0,
    "sun_color": [1.0, 0.92, 0.80], "sun_intensity": 2.5,
    "sky_zenith": [0.36, 0.39, 0.40], "sky_horizon": [0.58, 0.60, 0.56],
    "sky_ground": [0.16, 0.15, 0.12], "sky_sun_color": [1.0, 0.95, 0.85],
    "sky_intensity": 1.0, "sun_glow": 0.25, "sun_disk": 0.0,
    # Ground haze: thickest on the ground, thinning by e every 12.5 m up.
    "fog_density": 0.035, "fog_height": G, "fog_falloff": 0.08,
    "fog_color": [0.50, 0.52, 0.48], "fog_sun": 0.3,
    "exposure": 1.0,
}
LAMP = {"color": [1.0, 0.78, 0.5], "intensity": 10.0, "radius": 12.0,
        "source_radius": 0.1}
# The playable loop's orbs (game/src/weapon.rs's `target` prefab): one press
# one shot, three hits pop one. Default colour/intensity; tune by eye.
TARGET_PARAMS = {"hits": 3.0}
# The barrels are props rapier moves (ARCHITECTURE.md §15's `dynamic`
# prefab): empty 200 l steel drums, about 20 kg, so the player can shove
# one along and a shot knocks it. As cylinders: the scans' hulls rocked on
# their facets. Tyres and barriers stay put.
BARREL = {"prefab": "dynamic", "params": {"mass": 20.0, "collider": "cylinder"}}

# Alpha cutout (ARCHITECTURE.md §5). A chain-link enclosure round the
# compressor, from ambientCG's Fence006 (rusty diamond mesh), and grass tufts
# on the mud from a texture this script draws.
FENCE_PACK = os.path.join(ASSETS, "ambientcg_Fence006")
FENCE_MAPS = "Fence006_1K-PNG"
CHAINLINK_TILE = 1.0  # metres per texture repeat
ENCLOSURE = (16.8, 23.2, -6.2, 0.2)  # x0, x1, z0, z1
ENCLOSURE_BAY, ENCLOSURE_HEIGHT = 3.2, 2.0
# Trees and bushes (§5 cutouts): how many to place, outside the fence and
# in the yard, and the variants built.
TREES_OUT, TREES_IN = 35, 6
BUSHES_OUT, BUSHES_IN = 40, 25
TREE_VARIANTS, BUSH_VARIANTS = 3, 2
BELT = 38.0  # trees and bushes stay inside |x|, |z| <= this (the ground is ±40)
SPAWN_XZ = (0.0, 37.0)  # player_start, which foliage keeps 6 m from
ROAD_X = 6.0  # the road and gate apron, |x| < this south of the hangar
GRASS_TUFTS = 600
GRASS_TUFT = (0.9, 0.55)  # width, height of each of a tuft's three cards

# The pond: a hollow in the mud west of the hangar, where nothing else stands
# (the app's slab is off, ENVIRONMENT's `ground`). A basin mesh fills POND's
# rectangle: flat mud at its edges, a bank sloping to a bed POND_DEPTH down
# in the middle of a wobbly oval POND_RADII across; the water lies
# POND_WATER under the mud. Nothing else is placed in the rectangle, and
# every such test runs after its candidate's random draws, so the rest of a
# seed's level is exactly as it was without the pond.
POND = (-26.0, -16.0, -7.0, 1.0)  # x0, x1, z0, z1
POND_RADII = (3.6, 2.8)
POND_DEPTH = 1.1
POND_WATER = 0.15
POND_CELL = 0.25  # the basin's grid
POND_WATER_COLOR = [0.045, 0.05, 0.03]  # the water where it's deep (linear)
POND_CLARITY = 0.35  # metres to 1/e: murky, the bed gone by about a metre

FACES = {
    # face: (normal, the two in-plane axes (u, v) for UVs, as axis indices)
    "+x": ((1, 0, 0), (2, 1)), "-x": ((-1, 0, 0), (2, 1)),
    "+y": ((0, 1, 0), (0, 2)), "-y": ((0, -1, 0), (0, 2)),
    "+z": ((0, 0, 1), (0, 1)), "-z": ((0, 0, -1), (0, 1)),
}


def quat_axis(axis, degrees):
    """Unit quaternion (x, y, z, w) for a rotation about a unit axis."""
    h = math.radians(degrees) / 2
    s = math.sin(h)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h)]


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def yaw_q(degrees):
    return quat_axis((0, 1, 0), degrees)


class Zone(gds.Level):
    """gen_detailscene's glTF writer plus boxes with per-face materials,
    tiling textures, arbitrary rotations and placement bookkeeping."""

    def __init__(self, rng):
        super().__init__()
        self.doc["asset"]["generator"] = "feather tools/gen_zonescene.py"
        self.rng = rng
        self.mats = {}
        self.textures = {}
        self.local_boxes = {}
        # Everything solid placed so far, plus what the app puts in every
        # level: props look for room among these.
        self.solid = list(gts.keep_out_boxes())
        # ids of the solids a tree crown may grow through: the fence line
        # (it's lower), trunks (crowns interleave) and poles (overgrown).
        self.crown_through = set()

    # --- materials ---------------------------------------------------------

    def _image(self, path):
        if path not in self.textures:
            with open(path, "rb") as f:
                data = f.read()
            self.doc["images"].append({"bufferView": self._view(data), "mimeType": "image/jpeg"})
            self.doc["textures"].append({"source": len(self.doc["images"]) - 1})
            self.textures[path] = len(self.doc["textures"]) - 1
        return self.textures[path]

    def material(self, key):
        if key not in self.mats:
            pid, _, surface = TEXTURES[key]
            base = os.path.join(ASSETS, f"polyhaven_{pid}", pid)
            arm = self._image(f"{base}_arm_2k.jpg")
            self.doc["materials"].append({
                "name": key,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": self._image(f"{base}_diff_2k.jpg")},
                    "metallicRoughnessTexture": {"index": arm},
                },
                "normalTexture": {"index": self._image(f"{base}_nor_gl_2k.jpg")},
                # The ARM map's red is ambient occlusion (§13).
                "occlusionTexture": {"index": arm},
                "extras": {"surface": surface},
            })
            self.mats[key] = len(self.doc["materials"]) - 1
        return self.mats[key]

    def _png(self, data, key):
        if key not in self.textures:
            self.doc["images"].append({"bufferView": self._view(data), "mimeType": "image/png"})
            self.doc["textures"].append({"source": len(self.doc["images"]) - 1})
            self.textures[key] = len(self.doc["textures"]) - 1
        return self.textures[key]

    def cutout_material(self, key, base_png, normal_png=None, metallic=0.0, roughness=0.9,
                        surface=None, double_sided=True):
        """A MASK material (§5) from an RGBA base colour: double-sided by
        default. Leaves are single-sided on purpose (see `foliage`)."""
        if key not in self.mats:
            m = {
                "name": key,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": self._png(base_png, key + "/base")},
                    "metallicFactor": metallic,
                    "roughnessFactor": roughness,
                },
                "alphaMode": "MASK",
                "alphaCutoff": 0.5,
                "doubleSided": double_sided,
            }
            if normal_png is not None:
                m["normalTexture"] = {"index": self._png(normal_png, key + "/normal")}
            if surface:
                m["extras"] = {"surface": surface}
            self.doc["materials"].append(m)
            self.mats[key] = len(self.doc["materials"]) - 1
        return self.mats[key]

    def png_material(self, key, base_png, roughness=0.9, surface=None):
        """An opaque material from a PNG base colour (bark)."""
        if key not in self.mats:
            m = {
                "name": key,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": self._png(base_png, key + "/base")},
                    "metallicFactor": 0.0,
                    "roughnessFactor": roughness,
                },
            }
            if surface:
                m["extras"] = {"surface": surface}
            self.doc["materials"].append(m)
            self.mats[key] = len(self.doc["materials"]) - 1
        return self.mats[key]

    def mesh(self, prims, name):
        """One mesh from primitives (positions, normals, uvs, indices,
        material), triangle lists."""
        out = []
        for pos, nrm, uv, idx, material in prims:
            out.append({
                "attributes": {
                    "POSITION": self._accessor(pos, "f", "VEC3", 5126, 34962, minmax=True),
                    "NORMAL": self._accessor(nrm, "f", "VEC3", 5126, 34962),
                    "TEXCOORD_0": self._accessor(uv, "f", "VEC2", 5126, 34962),
                },
                "indices": self._accessor(idx, "I", "SCALAR", 5125, 34963),
                "material": material,
            })
        self.doc["meshes"].append({"name": name, "primitives": out})
        return len(self.doc["meshes"]) - 1

    def cards(self, quads, material, name):
        """One mesh of flat double-sided quads, one primitive. Each quad is
        (origin, u axis, v axis, uv scale): corners origin, +u, +u+v, +v, with
        UV (0,1)..(s,1-t) so the texture stands upright."""
        pos, nrm, uv, idx = [], [], [], []
        for o, u, v, (su, sv) in quads:
            n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
            ln = math.sqrt(sum(c * c for c in n))
            n = [c / ln for c in n]
            base = len(pos)
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                pos.append([o[i] + a * u[i] + b * v[i] for i in range(3)])
                nrm.append(n)
                uv.append([a * su, 1.0 - b * sv])
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
        self.doc["meshes"].append({"name": name, "primitives": [{
            "attributes": {
                "POSITION": self._accessor(pos, "f", "VEC3", 5126, 34962, minmax=True),
                "NORMAL": self._accessor(nrm, "f", "VEC3", 5126, 34962),
                "TEXCOORD_0": self._accessor(uv, "f", "VEC2", 5126, 34962),
            },
            "indices": self._accessor(idx, "I", "SCALAR", 5125, 34963),
            "material": material,
        }]})
        return len(self.doc["meshes"]) - 1

    # --- boxes -------------------------------------------------------------

    def _box_mesh(self, lo, hi, mats, uv_origin, name):
        """A box mesh from `lo` to `hi` (local space), one primitive per
        material. `mats` maps a face ("+x" ...) to a TEXTURES key, with "*"
        for the rest. UVs are (local + uv_origin) metres over the texture's
        size, v flipped so textures stand upright."""
        groups = {}
        for face, (n, (ua, va)) in FACES.items():
            key = mats.get(face, mats.get("*"))
            axis = next(i for i in range(3) if n[i])
            fixed = hi[axis] if n[axis] > 0 else lo[axis]
            corners = []
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis] = fixed
                p[ua] = (lo[ua], hi[ua])[a]
                p[va] = (lo[va], hi[va])[b]
                corners.append(p)
            size = TEXTURES[key][1]
            uvs = [((p[ua] + uv_origin[ua]) / size, -(p[va] + uv_origin[va]) / size)
                   for p in corners]
            g = groups.setdefault(key, ([], [], [], []))
            base = len(g[0])
            g[0].extend(corners)
            g[1].extend([list(n)] * 4)
            g[2].extend(uvs)
            g[3].extend([base, base + 1, base + 2, base, base + 2, base + 3])
        prims = []
        for key, (pos, nrm, uv, idx) in groups.items():
            prims.append({
                "attributes": {
                    "POSITION": self._accessor(pos, "f", "VEC3", 5126, 34962, minmax=True),
                    "NORMAL": self._accessor(nrm, "f", "VEC3", 5126, 34962),
                    "TEXCOORD_0": self._accessor(uv, "f", "VEC2", 5126, 34962),
                },
                "indices": self._accessor(idx, "I", "SCALAR", 5125, 34963),
                "material": self.material(key),
            })
        self.doc["meshes"].append({"name": name, "primitives": prims})
        return len(self.doc["meshes"]) - 1

    def place(self, mesh, t, rot=None, extras=None, name="node", scale=None):
        n = {"name": name, "translation": [float(c) for c in t]}
        if mesh is not None:
            n["mesh"] = mesh
        if rot is not None:
            n["rotation"] = [float(c) for c in rot]
        if scale is not None:
            n["scale"] = [float(scale)] * 3
        if extras:
            n["extras"] = extras
        self.doc["nodes"].append(n)
        self.doc["scenes"][0]["nodes"].append(len(self.doc["nodes"]) - 1)

    def box(self, lo, hi, mats, name, collide=True, shadow=True):
        """An axis-aligned box from world `lo` to `hi`, placed at its centre,
        with world-continuous UVs (so it's its own mesh)."""
        if isinstance(mats, str):
            mats = {"*": mats}
        c = [(a + b) / 2 for a, b in zip(lo, hi)]
        mesh = self._box_mesh([a - m for a, m in zip(lo, c)], [b - m for b, m in zip(hi, c)],
                              mats, c, name)
        params = {"collider": "box"} if collide else {"collide": False}
        if not shadow:
            params["shadow"] = False
        self.place(mesh, c, extras={"prefab": "prop", "params": params}, name=name)
        if collide and hi[1] - lo[1] >= 0.05:
            self.solid.append((list(lo), list(hi)))

    def local_box(self, size, mats, name):
        """A box mesh centred on its origin, UVs from local coordinates: one
        mesh for many instances (fence panels, posts)."""
        key = (tuple(size), name)
        if key not in self.local_boxes:
            h = [s / 2 for s in size]
            self.local_boxes[key] = self._box_mesh([-a for a in h], h, {"*": mats}, [0, 0, 0], name)
        return self.local_boxes[key]

    # --- props -------------------------------------------------------------

    def prop(self, key, **kw):
        return self.model(key, path=PROPS.get(key, kw.pop("path", None)), **kw)

    def put(self, key, x, z, floor=MUD_TOP, rot=None, extras=None, lift=0.0, check=True):
        """Place prop `key` with its rotated bounds centred on (x, z), seated
        on `floor` (the bounds' bottom on it). Centred on the bounds, not the
        origin: some models sit metres from theirs (the compressor, ~4 m).
        Returns False, placing nothing, if it would overlap something solid
        (when `check`)."""
        m = self.prop(key)
        aabb = gts.node_aabb((m["lo"], m["hi"]), (0.0, 0.0, 0.0), rot)
        x -= (aabb[0][0] + aabb[1][0]) / 2
        z -= (aabb[0][2] + aabb[1][2]) / 2
        y = floor + lift - aabb[0][1]
        aabb = gts.node_aabb((m["lo"], m["hi"]), (x, y, z), rot)
        if check and any(gts.overlaps(aabb, s, 0.02) for s in self.solid):
            return False
        self.place(m["mesh"], (x, y, z), rot, extras, name=key)
        self.solid.append(aabb)
        return True


# --- the level ---------------------------------------------------------------

def wall(z, axis, fixed, a0, a1, y0, y1, thick, mats, openings=(), name="wall"):
    """A straight wall along `axis` ("x" or "z"), centred on `fixed` across
    it, from a0 to a1, heights y0..y1 above the ground, with rectangular
    `openings` (b0, b1, oy0, oy1) cut out of it."""
    def seg(s0, s1, h0, h1):
        if s1 - s0 < 1e-3 or h1 - h0 < 1e-3:
            return
        if axis == "x":
            lo, hi = (s0, G + h0, fixed - thick / 2), (s1, G + h1, fixed + thick / 2)
        else:
            lo, hi = (fixed - thick / 2, G + h0, s0), (fixed + thick / 2, G + h1, s1)
        z.box(lo, hi, mats, name)

    def clamp(h):
        return min(max(h, y0), y1)

    pos = a0
    for b0, b1, oy0, oy1 in sorted(openings):
        seg(pos, b0, y0, y1)
        seg(b0, b1, y0, clamp(oy0))  # below the opening
        seg(b0, b1, clamp(oy1), y1)  # above it
        pos = b1
    seg(pos, a1, y0, y1)


def in_pond(x, zz, r=0.0):
    """Whether a footprint of radius `r` at (x, zz) touches the pond's
    rectangle."""
    x0, x1, z0, z1 = POND
    return x0 - r < x < x1 + r and z0 - r < zz < z1 + r


def pond_depth(x, zz):
    """How far the basin lies below the mud at (x, zz): 0 outside the
    hollow (and so on the rectangle's edges), POND_DEPTH at its middle."""
    x0, x1, z0, z1 = POND
    dx = (x - (x0 + x1) / 2) / POND_RADII[0]
    dz = (zz - (z0 + z1) / 2) / POND_RADII[1]
    a = math.atan2(dz, dx)
    # A hand-dug outline, not an ellipse. At most 1.25 radii out, which
    # stays inside the rectangle.
    wobble = 1.0 + 0.12 * math.sin(2 * a + 0.7) + 0.08 * math.sin(3 * a - 1.9) \
        + 0.05 * math.sin(5 * a + 2.3)
    s = 1.0 - math.hypot(dx, dz) / wobble  # 1 in the middle, 0 on the outline
    t = min(max(s / 0.6, 0.0), 1.0)  # the bank is the outer 60%
    return POND_DEPTH * t * t * (3 - 2 * t)


def pond(z):
    """The basin, a heightfield of mud over POND's rectangle (world-space
    vertices, a trimesh collider, casting), and the water over it."""
    x0, x1, z0, z1 = POND
    nx, nz = round((x1 - x0) / POND_CELL), round((z1 - z0) / POND_CELL)
    size = TEXTURES["mud"][1]
    h = lambda x, zz: MUD_TOP - pond_depth(x, zz)
    pos, nrm, uv, idx = [], [], [], []
    for j in range(nz + 1):
        for i in range(nx + 1):
            x, zz = x0 + i * POND_CELL, z0 + j * POND_CELL
            e = 0.05
            n = (-(h(x + e, zz) - h(x - e, zz)) / (2 * e), 1.0,
                 -(h(x, zz + e) - h(x, zz - e)) / (2 * e))
            ln = math.sqrt(sum(c * c for c in n))
            pos.append([x, h(x, zz), zz])
            nrm.append([c / ln for c in n])
            # The mud boxes' top-face UVs, so the texture runs on.
            uv.append([x / size, -zz / size])
    for j in range(nz):
        for i in range(nx):
            a = j * (nx + 1) + i
            idx += [a, a + 1, a + nx + 2, a, a + nx + 2, a + nx + 1]
    basin = z.mesh([(pos, nrm, uv, idx, z.material("mud"))], "pond_basin")
    # Relief: a trimesh (it's concave, and too dense for `auto`), and it
    # casts, unlike the flat ground.
    z.place(basin, (0.0, 0.0, 0.0), extras={"prefab": "prop", "params": {"collider": "mesh"}},
            name="pond_basin")
    z.doc["materials"].append({
        "name": "pond_water",
        "pbrMetallicRoughness": {"baseColorFactor": POND_WATER_COLOR + [1.0],
                                 "metallicFactor": 0.0, "roughnessFactor": 0.05},
        "extras": {"shader": "water", "clarity": POND_CLARITY},
    })
    water_mat = len(z.doc["materials"]) - 1
    y = MUD_TOP - POND_WATER
    # One quad over the whole rectangle: where the basin is above it, the
    # mud hides it.
    corners = [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]]
    water = z.mesh([(corners, [[0.0, 1.0, 0.0]] * 4, [[c[0], -c[2]] for c in corners],
                     [0, 1, 2, 0, 2, 3], water_mat)], "pond_water")
    z.place(water, (0.0, 0.0, 0.0),
            extras={"prefab": "prop", "params": {"collide": False, "shadow": False}},
            name="pond_water")


def ground(z):
    # Mud everywhere but the pond's rectangle, which the basin fills; then
    # asphalt and concrete on the mud. The flat parts are shadowless (they'd
    # cast nothing but fill the shadow map).
    x0, x1, z0, z1 = POND
    for lo, hi in (((-HALF, G, -HALF), (HALF, MUD_TOP, z0)),
                   ((-HALF, G, z1), (HALF, MUD_TOP, HALF)),
                   ((-HALF, G, z0), (x0, MUD_TOP, z1)),
                   ((x1, G, z0), (HALF, MUD_TOP, z1))):
        z.box(lo, hi, "mud", "mud", shadow=False)
    pond(z)
    x0, x1, z0, z1 = HANGAR
    z.box((GATE[0], MUD_TOP, z1), (GATE[1], PAVE_TOP, HALF), "asphalt", "road", shadow=False)
    for a, b in ((-16.0, GATE[0]), (GATE[1], 16.0)):
        z.box((a, MUD_TOP, z1), (b, PAVE_TOP, 22.0), "floor", "apron", shadow=False)
    z.box((x0, MUD_TOP, z0), (x1, PAVE_TOP, z1), "floor", "hangar_floor", shadow=False)


def hangar(z):
    x0, x1, z0, z1 = HANGAR
    t = HANGAR_WALL
    dx0, dx1, dh = HANGAR_DOOR
    side_door = (1.0, 2.4, 0.0, 2.4)  # east wall, z span and height
    for y0, y1, mat in ((0.0, HANGAR_BRICK, "brick"), (HANGAR_BRICK, HANGAR_HEIGHT, "corrugated")):
        # North and south walls span between the east and west ones.
        wall(z, "x", z0, x0 + t / 2, x1 - t / 2, y0, y1, t, mat, name="hangar_wall")
        wall(z, "x", z1, x0 + t / 2, x1 - t / 2, y0, y1, t, mat,
             [(dx0, dx1, 0.0, dh)], name="hangar_wall")
        wall(z, "z", x0, z0 - t / 2, z1 + t / 2, y0, y1, t, mat, name="hangar_wall")
        wall(z, "z", x1, z0 - t / 2, z1 + t / 2, y0, y1, t, mat, [side_door], name="hangar_wall")
    # Roof: a grid of corrugated panels on rusty beams, a few missing.
    cols, rows = 6, 5
    w, d = (x1 - x0 + 0.4) / cols, (z1 - z0 + 0.4) / rows
    holes = set(z.rng.sample([(c, r) for c in range(cols) for r in range(rows)], 3))
    top = HANGAR_HEIGHT
    for c in range(cols):
        for r in range(rows):
            if (c, r) in holes:
                continue
            a, b = x0 - 0.2 + c * w, z0 - 0.2 + r * d
            z.box((a, G + top, b), (a + w, G + top + 0.1, b + d),
                  {"*": "corrugated", "-y": "rust"}, "hangar_roof")
    for r in range(1, rows):
        b = z0 - 0.2 + r * d
        z.box((x0 + t / 2, G + top - 0.4, b - 0.12), (x1 - t / 2, G + top, b + 0.12),
              "rust", "hangar_beam")


def office(z):
    x0, x1, z0, z1 = OFFICE
    t = OFFICE_WALL
    ix0, ix1, iz0, iz1 = x0 + t / 2, x1 - t / 2, z0 + t / 2, z1 - t / 2  # inside faces
    f1 = SLAB + STOREY  # top of the first floor
    roof = SLAB + 2 * STOREY  # underside of the roof slab
    parapet = roof + SLAB + 0.6

    def windows(centres, floor_top):
        sill = floor_top + 0.9
        return [(c - 0.7, c + 0.7, sill, sill + 1.4) for c in centres]

    door = [(-16.6, -15.4, 0.0, SLAB + 2.2)]
    walls = {
        # (axis, fixed, a0, a1, outside face, openings)
        "south": ("x", z1, ix0, ix1, "+z",
                  windows((18.5, 22.0, 25.5), SLAB) + windows((18.5, 22.0, 25.5), f1)),
        "north": ("x", z0, ix0, ix1, "-z",
                  windows((18.5, 22.0, 25.5), SLAB) + windows((18.5, 22.0, 25.5), f1)),
        "west": ("z", x0, z0 - t / 2, z1 + t / 2, "-x",
                 door + windows((-18.6, -13.4), SLAB) + windows((-18.0, -14.0), f1)),
        "east": ("z", x1, z0 - t / 2, z1 + t / 2, "+x", windows((-18.0,), f1)),
    }
    for name, (axis, fixed, a0, a1, out, opens) in walls.items():
        inside = ("-" if out[0] == "+" else "+") + out[1]
        mats = {"*": "precast", inside: "plaster"}
        wall(z, axis, fixed, a0, a1, 0.0, parapet, t, mats, opens, name=f"office_{name}")

    floor_mats = {"*": "plaster", "+y": "floor"}
    # Ground floor slab, then the stair up the inside of the east wall, from
    # the south end northwards, and the first floor round its stairwell.
    z.box((ix0, G, iz0), (ix1, G + SLAB, iz1), floor_mats, "office_floor")
    sx0, sx1 = ix1 - 1.2, ix1
    steps = round(STOREY / STEP_RISE)
    s_start = iz1 - 1.2  # a landing at the foot, to turn onto the stair
    s_end = s_start - steps * STEP_RUN
    for i in range(steps):
        za, zb = s_start - (i + 1) * STEP_RUN, s_start - i * STEP_RUN
        z.box((sx0, G + SLAB, za), (sx1, G + SLAB + (i + 1) * STEP_RISE, zb),
              {"*": "plaster", "+y": "floor"}, "office_step")
    y0, y1 = G + f1 - SLAB, G + f1
    z.box((ix0, y0, iz0), (sx0, y1, iz1), floor_mats, "office_floor")
    z.box((sx0, y0, iz0), (ix1, y1, s_end), floor_mats, "office_floor")
    z.box((sx0, y0, s_start), (ix1, y1, iz1), floor_mats, "office_floor")
    z.box((ix0, G + roof, iz0), (ix1, G + roof + SLAB, iz1),
          {"*": "plaster", "+y": "floor"}, "office_roof")


def fence(z):
    x0, x1, z0, z1 = FENCE
    panel = z.local_box(PANEL, "precast", "fence_panel")
    post = z.local_box((POST[0], POST[1], POST[0]), "precast", "fence_post")
    bays = []  # (centre x, centre z, yaw)
    for x in range(int(x0), int(x1), int(FENCE_BAY)):
        for fz in (z0, z1):
            cx = x + FENCE_BAY / 2
            if fz == z1 and GATE[0] < cx < GATE[1]:
                continue
            bays.append((cx, fz, 0.0))
    for zz in range(int(z0), int(z1), int(FENCE_BAY)):
        for fx in (x0, x1):
            bays.append((fx, zz + FENCE_BAY / 2, 90.0))
    rng = z.rng
    gone = set(rng.sample(range(len(bays)), 4))
    fallen, leaning = sorted(gone)[:2]
    extras = {"prefab": "prop", "params": {"collider": "box"}}
    for i, (cx, cz, yaw) in enumerate(bays):
        if i in gone and i not in (fallen, leaning):
            continue
        rot = yaw_q(yaw)
        y = MUD_TOP + 0.05 + PANEL[1] / 2
        if i == fallen:
            # Flat on the ground beside its gap, face up.
            rot = quat_mul(yaw_q(yaw), quat_axis((1, 0, 0), 90.0))
            y = MUD_TOP + PANEL[2] / 2
            # Beside its gap, on the inside.
            if yaw == 0.0:
                cz -= math.copysign(1.6, cz)
            else:
                cx -= math.copysign(1.6, cx)
        elif i == leaning:
            rot = quat_mul(yaw_q(yaw), quat_axis((1, 0, 0), 12.0))
            y = MUD_TOP + math.cos(math.radians(12)) * PANEL[1] / 2
        z.place(panel, (cx, y, cz), rot, extras, name="fence_panel")
        if i == fallen:
            # Out in the yard, past the fence line's solid below.
            z.solid.append(gts.node_aabb(([-a / 2 for a in PANEL], [a / 2 for a in PANEL]),
                                         (cx, y, cz), rot))
    posts = set()
    for cx, cz, yaw in bays:
        for s in (-1, 1):
            p = (cx + s * FENCE_BAY / 2, cz) if yaw == 0.0 else (cx, cz + s * FENCE_BAY / 2)
            posts.add((round(p[0], 3), round(p[1], 3)))
    for px, pz in sorted(posts):
        z.place(post, (px, MUD_TOP + POST[1] / 2, pz), None, extras, name="fence_post")
    # Everything along the fence line is solid for the props.
    t = 0.4
    for lo, hi in (((x0 - t, G, z0 - t), (x1 + t, G + 3, z0 + t)),
                   ((x0 - t, G, z1 - t), (x1 + t, G + 3, z1 + t)),
                   ((x0 - t, G, z0 - t), (x0 + t, G + 3, z1 + t)),
                   ((x1 - t, G, z0 - t), (x1 + t, G + 3, z1 + t))):
        z.solid.append((list(lo), list(hi)))
        z.crown_through.add(id(z.solid[-1]))


def props(z):
    rng = z.rng
    counts = {}

    def lying():
        return quat_mul(yaw_q(rng.uniform(0, 360)), quat_axis((1, 0, 0), 90.0))

    def count(key, ok):
        counts[key] = counts.get(key, 0) + (1 if ok else 0)

    # Barriers staggered across the gate, a couple pushed aside inside.
    for x, zz, yaw in ((-2.6, 30.5, 8), (0.4, 32.2, -5), (2.8, 30.2, 12),
                       (-6.5, 25.0, 80), (6.8, 24.5, 95)):
        count("barrier", z.put("barrier", x, zz, rot=yaw_q(yaw), floor=MUD_TOP))
    # Barrel clusters: yard, behind the hangar, inside it.
    for cx, cz, floor in ((-20.0, 18.0, MUD_TOP), (10.0, 19.0, PAVE_TOP),
                          (-4.0, -14.0, MUD_TOP), (24.8, -8.6, MUD_TOP),
                          (-9.5, -6.5, PAVE_TOP), (9.0, 10.0, PAVE_TOP)):
        for _ in range(rng.randint(3, 6)):
            key = rng.choice(("barrel_a", "barrel_b"))
            x, zz = cx + rng.uniform(-1.4, 1.4), cz + rng.uniform(-1.4, 1.4)
            rot = lying() if rng.random() < 0.2 else yaw_q(rng.uniform(0, 360))
            count(key, z.put(key, x, zz, floor=floor, rot=rot, extras=BARREL))
    # Tyre stacks, and a few lying about.
    for cx, cz, floor in ((-15.0, 24.0, MUD_TOP), (14.0, -24.0, MUD_TOP), (-10.0, 10.5, PAVE_TOP)):
        for k in range(rng.randint(2, 5)):
            count("tyre", z.put("tyre", cx, cz, floor=floor, rot=lying(),
                                lift=k * 0.165, check=(k == 0)))
    for _ in range(4):
        x, zz = rng.uniform(-28, 28), rng.uniform(-24, 26)
        rot = lying()
        count("tyre", not in_pond(x, zz, 0.6) and z.put("tyre", x, zz, rot=rot))
    # On the walls: utility boxes (backs to the wall) and pipework.
    ox0, _, _, _ = OFFICE
    m = z.prop("utility_box")
    depth = m["hi"][2] - m["lo"][2]
    for zz in (-13.2, -18.8):
        count("utility_box", z.put("utility_box", ox0 - OFFICE_WALL / 2 - depth / 2 - 0.01, zz,
                                   rot=yaw_q(-90), lift=0.9, check=False))
    hx0, hx1, hz0, hz1 = HANGAR
    count("utility_box", z.put("utility_box", hx1 + HANGAR_WALL / 2 + depth / 2 + 0.01, 4.0,
                               rot=yaw_q(90), lift=0.9, check=False))
    m = z.prop("pipes")
    depth = m["hi"][2] - m["lo"][2]
    for x in (-8.0, -6.6, 3.0, 4.4):
        count("pipes", z.put("pipes", x, hz0 - HANGAR_WALL / 2 - depth / 2 - 0.01,
                             rot=yaw_q(180), lift=0.3, check=False))
    count("compressor", z.put("compressor", 20.0, -3.0, rot=yaw_q(rng.uniform(0, 360))))
    count("car", z.put("car", -8.5, 6.5, floor=PAVE_TOP, rot=yaw_q(4)))
    # Electricity poles in lines outside the fence, crossarms across the line.
    for key, (prefix, px) in POLE_PRESETS.items():
        keep = (lambda p: lambda name, size: name.startswith(p) and max(size) >= POLE_MIN_PART)(prefix)
        z.model(key, path=PROPS["poles"], keep=keep, shift=(-px, 0.0, 0.0))
    for i, x in enumerate((-30.0, -18.0, -6.0, 6.0, 18.0, 30.0)):
        count("pole", z.put(f"pole_{i % 3 + 1}", x, 34.5, rot=yaw_q(90)))
        z.crown_through.add(id(z.solid[-1]))
    for i, zz in enumerate((-24.0, -8.0, 8.0)):
        count("pole", z.put(f"pole_{(i + 1) % 3 + 1}", -36.0, zz, rot=yaw_q(0)))
        z.crown_through.add(id(z.solid[-1]))
    return counts


# --- alpha cutout ------------------------------------------------------------

def png_decode(path):
    """An 8-bit, non-interlaced PNG as (width, height, channels, rows), rows
    being bytearrays. Stdlib only, like gen_testscene's encoder."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos, idat, head = 8, [], None
    while pos < len(data):
        n, tag = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if tag == b"IHDR":
            head = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
    w, h, depth, ctype, _, _, interlace = head
    if depth != 8 or interlace:
        raise ValueError(f"{path}: only 8-bit non-interlaced PNGs are supported")
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    raw = zlib.decompress(b"".join(idat))
    stride = w * ch
    rows, prev, i = [], bytearray(stride), 0
    for _ in range(h):
        f, line = raw[i], bytearray(raw[i + 1:i + 1 + stride])
        i += 1 + stride
        if f == 1:
            for x in range(ch, stride):
                line[x] = (line[x] + line[x - ch]) & 255
        elif f == 2:
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 255
        elif f == 3:
            for x in range(stride):
                left = line[x - ch] if x >= ch else 0
                line[x] = (line[x] + ((left + prev[x]) >> 1)) & 255
        elif f == 4:
            for x in range(stride):
                a = line[x - ch] if x >= ch else 0
                b = prev[x]
                c = prev[x - ch] if x >= ch else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[x] = (line[x] + pred) & 255
        rows.append(line)
        prev = line
    return w, h, ch, rows


def chainlink_rgba():
    """Fence006's colour with its opacity as alpha, one RGBA PNG: glTF takes
    a cutout's alpha from the base colour. Cached beside the pack."""
    cached = os.path.join(FENCE_PACK, "merged_rgba.png")
    if os.path.exists(cached):
        with open(cached, "rb") as f:
            return f.read()
    w, h, ch, color = png_decode(os.path.join(FENCE_PACK, f"{FENCE_MAPS}_Color.png"))
    w2, h2, ch2, alpha = png_decode(os.path.join(FENCE_PACK, f"{FENCE_MAPS}_Opacity.png"))
    if (w, h, ch, ch2) != (w2, h2, 3, 1):
        raise ValueError("Fence006: expected RGB colour and grey opacity of one size")
    rows = []
    for c, a in zip(color, alpha):
        row = bytearray(w * 4)
        row[0::4], row[1::4], row[2::4] = c[0::3], c[1::3], c[2::3]
        row[3::4] = a
        rows.append(row)
    data = gts.png_rgba(w, h, rows)
    with open(cached, "wb") as f:
        f.write(data)
    return data


def grass_rgba(rng, size=256, blades=46):
    """A grass card: tapered, gently bent blades from the bottom edge, from a
    dark base to dry yellow-green tips, alpha 0 between them. Transparent
    texels carry a mid grass colour, not black: mips average colour across the
    cut edge too, and black would fringe the blades at a distance."""
    fill = (92, 88, 48, 0)
    px = [[list(fill) for _ in range(size)] for _ in range(size)]
    for _ in range(blades):
        x0 = rng.uniform(0.08, 0.92) * size
        height = rng.uniform(0.45, 0.98) * size
        width = rng.uniform(2.5, 6.0)
        bend = rng.uniform(-0.25, 0.25) * height
        tip = rng.choice([(150, 146, 78), (118, 128, 58), (160, 150, 92), (96, 112, 50)])
        base = (52, 50, 26)
        steps = int(height)
        for k in range(steps):
            t = k / max(1, steps - 1)
            y = size - 1 - k
            cx = x0 + bend * t * t
            half = width * (1.0 - t) / 2 + 0.5
            col = [int(b + (c - b) * t) for b, c in zip(base, tip)]
            for x in range(int(cx - half), int(cx + half) + 1):
                if 0 <= x < size:
                    px[y][x] = col + [255]
    rows = [bytearray(v for p in row for v in p) for row in px]
    return gts.png_rgba(size, size, rows)


def enclosure(z):
    """Chain-link round the compressor: rusty posts, double-sided cutout
    panels (they collide, as flat boxes), a gap on the west as its gate."""
    mat = z.cutout_material("chainlink", chainlink_rgba(),
                            open(os.path.join(FENCE_PACK, f"{FENCE_MAPS}_NormalGL.png"), "rb").read(),
                            metallic=0.6, roughness=0.6)
    w, h = ENCLOSURE_BAY, ENCLOSURE_HEIGHT
    tile = (w / CHAINLINK_TILE, h / CHAINLINK_TILE)
    panel = z.cards([((-w / 2, 0.0, 0.0), (w, 0.0, 0.0), (0.0, h, 0.0), tile)], mat, "chainlink")
    post = z.local_box((0.08, h + 0.2, 0.08), "rust", "chainlink_post")
    x0, x1, z0, z1 = ENCLOSURE
    extras = {"prefab": "prop", "params": {"collider": "box"}}
    bays = []  # (centre x, centre z, yaw)
    for i in range(round((x1 - x0) / w)):
        cx = x0 + (i + 0.5) * w
        bays += [(cx, z0, 0.0), (cx, z1, 0.0)]
    for i in range(round((z1 - z0) / w)):
        cz = z0 + (i + 0.5) * w
        bays += [(x1, cz, 90.0)]
        # West: only the northern bay; the southern one is the gate.
        if i == 0:
            bays += [(x0, cz, 90.0)]
    posts = set()
    for cx, cz, yaw in bays:
        z.place(panel, (cx, MUD_TOP + 0.05, cz), yaw_q(yaw), extras, name="chainlink")
        for s in (-1, 1):
            p = (cx + s * w / 2, cz) if yaw == 0.0 else (cx, cz + s * w / 2)
            posts.add((round(p[0], 3), round(p[1], 3)))
    for px, pz in sorted(posts):
        z.place(post, (px, MUD_TOP + (h + 0.2) / 2, pz), None, extras, name="chainlink_post")
    # Solid for the props: the panel lines.
    t = 0.3
    for lo, hi in (((x0 - t, G, z0 - t), (x1 + t, G + 3, z0 + t)),
                   ((x0 - t, G, z1 - t), (x1 + t, G + 3, z1 + t)),
                   ((x0 - t, G, z0 - t), (x0 + t, G + 3, z1 + t)),
                   ((x1 - t, G, z0 - t), (x1 + t, G + 3, z1 + t))):
        z.solid.append((list(lo), list(hi)))


def grass(z):
    """Tufts of three crossed cards on the mud, clear of paving, buildings,
    the fence lines and every prop. They cast shadows but don't collide."""
    rng = z.rng
    mat = z.cutout_material("grass_card", grass_rgba(rng), surface="grass")
    w, h = GRASS_TUFT
    quads = []
    for a in (0.0, 60.0, 120.0):
        c, s_ = math.cos(math.radians(a)), math.sin(math.radians(a))
        quads.append(((-c * w / 2, 0.0, -s_ * w / 2), (c * w, 0.0, s_ * w), (0.0, h, 0.0), (1.0, 1.0)))
    tuft = z.cards(quads, mat, "grass_tuft")
    hx0, hx1, hz0, hz1 = HANGAR
    ox0, ox1, oz0, oz1 = OFFICE
    fx0, fx1, fz0, fz1 = FENCE
    keep_off = [  # (x0, x1, z0, z1), with a margin
        (hx0 - 0.6, hx1 + 0.6, hz0 - 0.6, hz1 + 0.6),
        (ox0 - 0.6, ox1 + 0.6, oz0 - 0.6, oz1 + 0.6),
        (GATE[0] - 0.4, GATE[1] + 0.4, hz1, HALF),  # the road
        (-16.4, 16.4, hz1, 22.4),  # the aprons
    ] + [(fx - 0.5, fx + 0.5, fz0 - 0.5, fz1 + 0.5) for fx in (fx0, fx1)] \
      + [(fx0 - 0.5, fx1 + 0.5, fz - 0.5, fz + 0.5) for fz in (fz0, fz1)]
    placed = dropped = 0
    for _ in range(GRASS_TUFTS * 4):
        if placed == GRASS_TUFTS:
            break
        x, zz = rng.uniform(-HALF + 1, HALF - 1), rng.uniform(-HALF + 1, HALF - 1)
        if any(a <= x <= b and c <= zz <= d for a, b, c, d in keep_off):
            continue
        scale = rng.uniform(0.7, 1.35)
        r = w * scale / 2
        aabb = ([x - r, MUD_TOP, zz - r], [x + r, MUD_TOP + h * scale, zz + r])
        if any(gts.overlaps(aabb, s_, 0.05) for s_ in z.solid):
            continue
        yaw = rng.uniform(0, 360)
        placed += 1
        if in_pond(x, zz, r):
            # Dropped, not drawn again: every other tuft stays where it was.
            dropped += 1
            continue
        z.place(tuft, (x, MUD_TOP, zz), yaw_q(yaw),
                {"prefab": "prop", "params": {"collide": False}}, name="grass", scale=scale)
    return placed - dropped


# --- trees and bushes ---------------------------------------------------------

def _add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _mul(a, k):
    return [a[0] * k, a[1] * k, a[2] * k]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    ln = math.sqrt(sum(c * c for c in a)) or 1.0
    return [c / ln for c in a]


def leaf_rgba(rng, size=512, leaves=52):
    """A leaf cluster card: pointed oval birch-like leaves, each with a
    paler midrib, on thin twigs from the bottom centre. Muted greens for an
    overcast day, a few yellowing. Transparent texels carry the mid leaf
    colour, as grass_rgba's do, so mips don't fringe the cut edge."""
    fill = (74, 86, 42, 0)
    px = [[list(fill) for _ in range(size)] for _ in range(size)]

    def dot(x, y, col):
        if 0 <= x < size and 0 <= y < size:
            px[y][x] = list(col) + [255]

    # Twigs: from the bottom centre, fanning up.
    tips = []
    for _ in range(7):
        a = math.radians(rng.uniform(-70, 70))
        length = rng.uniform(0.45, 0.85) * size
        x0, y0 = size / 2 + rng.uniform(-10, 10), size - 4
        for k in range(int(length)):
            t = k / length
            x = x0 + math.sin(a) * k + math.sin(a * 2) * 12 * t * t
            y = y0 - math.cos(a) * k
            w = max(1, int(3 * (1 - t)))
            for dx in range(-w // 2, w // 2 + 1):
                dot(int(x) + dx, int(y), (62, 50, 36))
            if k % 9 == 0 and t > 0.2:
                tips.append((x, y, a))
    # Leaves along the twigs, and a few loose ones to fill the cluster.
    spots = rng.sample(tips, min(len(tips), leaves - 10))
    spots += [(rng.uniform(0.2, 0.8) * size, rng.uniform(0.15, 0.7) * size,
               rng.uniform(-1.2, 1.2)) for _ in range(leaves - len(spots))]
    for cx, cy, a in spots:
        length = rng.uniform(34, 58)
        width = length * rng.uniform(0.42, 0.55)
        ang = a + rng.uniform(-1.3, 1.3) - math.pi / 2  # along x', pointing out
        ca, sa = math.cos(ang), math.sin(ang)
        base = rng.choice([(72, 92, 38), (84, 100, 44), (62, 80, 34), (96, 104, 46),
                           (120, 118, 52)])
        r = int(length) + 2
        for y in range(int(cy) - r, int(cy) + r + 1):
            for x in range(int(cx) - r, int(cx) + r + 1):
                dx, dy = x - cx, y - cy
                u = dx * ca + dy * sa  # along the leaf, 0 at its stalk
                v = -dx * sa + dy * ca
                if not 0.0 <= u <= length:
                    continue
                half = width / 2 * math.sin(math.pi * u / length) ** 0.8
                if abs(v) > half:
                    continue
                edge = abs(v) / max(half, 1e-3)
                shade = 1.0 - 0.28 * edge - 0.10 * (u / length)
                col = [min(255, int(c * shade)) for c in base]
                if abs(v) < 0.9:
                    col = [min(255, c + 22) for c in col]  # the midrib
                dot(x, y, col)
    rows = [bytearray(v for p in row for v in p) for row in px]
    return gts.png_rgba(size, size, rows)


def bark_rgba(rng, w=128, h=256):
    """Birch-like bark, tiling round (u) and along (v) the stem: pale grey
    with dark horizontal lenticels and a few dark patches."""
    px = []
    for y in range(h):
        row = []
        for x in range(w):
            n = rng.uniform(-10, 10)
            g = 196 + n
            row.append([int(g), int(g - 2), int(g - 10)])
        px.append(row)
    for _ in range(140):
        y, x0 = rng.randrange(h), rng.randrange(w)
        length, thick = rng.randint(5, 24), rng.choice([1, 1, 2])
        tone = rng.randint(30, 70)
        for k in range(length):
            for t in range(thick):
                px[(y + t) % h][(x0 + k) % w] = [tone, tone - 2, tone - 4]
    for _ in range(10):
        cy, cx, r = rng.randrange(h), rng.randrange(w), rng.randint(4, 10)
        for y in range(cy - r, cy + r):
            for x in range(cx - r, cx + r):
                if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                    px[y % h][x % w] = [52, 50, 46]
    rows = [bytearray(c for p in row for c in p + [255]) for row in px]
    return gts.png_rgba(w, h, rows)


def tube(points, radii, sides, u_repeat, v_len, out):
    """A tube along `points` with `radii`, appended to `out` = (pos, nrm,
    uv, idx). UV u runs round it `u_repeat` times, v along it per `v_len`
    metres. No caps: trunk tops end inside crowns, branch tips are thin."""
    pos, nrm, uv, idx = out
    along = 0.0
    ring0 = len(pos)
    for i, p in enumerate(points):
        t = _norm([points[min(i + 1, len(points) - 1)][k] - points[max(i - 1, 0)][k]
                   for k in range(3)])
        ref = [0.0, 0.0, 1.0] if abs(t[2]) < 0.9 else [1.0, 0.0, 0.0]
        x = _norm(_cross(t, ref))
        y = _cross(t, x)
        if i:
            along += math.dist(points[i], points[i - 1])
        for j in range(sides + 1):
            a = 2 * math.pi * j / sides
            d = _add(_mul(x, math.cos(a)), _mul(y, math.sin(a)))
            pos.append(_add(p, _mul(d, radii[i])))
            nrm.append(d)
            uv.append([u_repeat * j / sides, -along / v_len])
    for i in range(len(points) - 1):
        for j in range(sides):
            a = ring0 + i * (sides + 1) + j
            b = a + sides + 1
            idx += [a, b, a + 1, a + 1, b, b + 1]


def leaf_cards(rng, centres, sizes, out):
    """Leaf cards at `centres`, appended to `out`. Each faces roughly out
    of the crown, turned at random about that; every vertex's normal points
    out of the crown's ellipsoid (from the cards' own spread), so the crown
    shades as one volume. The material is single-sided and the main pass
    doesn't cull, so both faces draw with that normal."""
    pos, nrm, uv, idx = out
    n = len(centres)
    c = [sum(p[k] for p in centres) / n for k in range(3)]
    spread = [max(0.3, math.sqrt(sum((p[k] - c[k]) ** 2 for p in centres) / n)) for k in range(3)]
    for p, s in zip(centres, sizes):
        out_dir = _norm([(p[k] - c[k]) / spread[k] for k in range(3)])
        facing = _norm(_add(out_dir, [rng.uniform(-0.9, 0.9) for _ in range(3)]))
        side = _norm(_cross(facing, _norm([rng.uniform(-1, 1) for _ in range(3)])))
        up = _cross(facing, side)
        base = len(pos)
        for a, b in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
            q = _add(p, _add(_mul(side, a * s), _mul(up, b * s)))
            pos.append(q)
            nrm.append(_norm([(q[k] - c[k]) / spread[k] ** 2 for k in range(3)]))
            uv.append([a + 0.5, 0.5 - b])
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]


def tree_meshes(z, rng, leaf, bark, variant):
    """One tree: (trunk mesh, crown mesh). Birch-like, 7-12 m: a leaning,
    tapered trunk, branches reaching up, leaf cards on their outer parts and
    round the top. The crown mesh holds the branches too."""
    height = rng.uniform(7.0, 12.0)
    r0 = rng.uniform(0.13, 0.22)
    lean = [rng.uniform(-0.06, 0.06), 0.0, rng.uniform(-0.06, 0.06)]
    segs = 10
    trunk_pts = [[lean[0] * height * (i / segs) ** 1.5, height * i / segs,
                  lean[2] * height * (i / segs) ** 1.5] for i in range(segs + 1)]
    trunk_r = [r0 * (1.0 - 0.8 * i / segs) for i in range(segs + 1)]
    trunk = ([], [], [], [])
    tube(trunk_pts, trunk_r, 8, 2.0, 1.2, trunk)
    branches = ([], [], [], [])
    centres, sizes = [], []
    n_br = rng.randint(6, 10)
    for b in range(n_br):
        t0 = rng.uniform(0.38, 0.85)
        start = [trunk_pts[0][k] + (trunk_pts[-1][k] - trunk_pts[0][k]) * t0 for k in range(3)]
        az = 2 * math.pi * (b + rng.uniform(-0.3, 0.3)) / n_br
        el = math.radians(rng.uniform(30, 55))
        length = height * rng.uniform(0.28, 0.42) * (1.15 - t0)
        d = [math.cos(az) * math.cos(el), math.sin(el), math.sin(az) * math.cos(el)]
        pts = [_add(start, _add(_mul(d, length * i / 5), [0.0, 0.25 * length * (i / 5) ** 2, 0.0]))
               for i in range(6)]
        tube(pts, [r0 * 0.35 * (1 - 0.8 * i / 5) for i in range(6)], 5, 1.0, 0.8, branches)
        for _ in range(rng.randint(22, 34)):
            t = rng.uniform(0.35, 1.0)
            i = min(4, int(t * 5))
            f = t * 5 - i
            p = [pts[i][k] + (pts[i + 1][k] - pts[i][k]) * f for k in range(3)]
            spread = 0.35 + 0.75 * t
            centres.append(_add(p, [rng.uniform(-spread, spread) for _ in range(3)]))
            sizes.append(rng.uniform(0.8, 1.4))
    top = trunk_pts[-1]
    for _ in range(rng.randint(40, 60)):
        centres.append(_add(top, [rng.uniform(-1.3, 1.3), rng.uniform(-1.8, 0.9),
                                  rng.uniform(-1.3, 1.3)]))
        sizes.append(rng.uniform(0.8, 1.3))
    leaves = ([], [], [], [])
    leaf_cards(rng, centres, sizes, leaves)
    trunk_mesh = z.mesh([(*trunk, bark)], f"tree_trunk_{variant}")
    crown_mesh = z.mesh([(*branches, bark), (*leaves, leaf)], f"tree_crown_{variant}")
    # How far the crown reaches from the trunk's foot, horizontally.
    crown_pts = branches[0] + leaves[0]
    reach = max(math.hypot(p[0], p[2]) for p in crown_pts)
    return trunk_mesh, crown_mesh, r0, reach, crown_pts


def bush_mesh(z, rng, leaf, bark, variant):
    """One bush, 1-2.5 m: a squashed ball of leaf cards on a few stems."""
    radius = rng.uniform(0.7, 1.2)
    height = radius * rng.uniform(1.1, 1.6)
    stems = ([], [], [], [])
    for _ in range(rng.randint(3, 5)):
        a = rng.uniform(0, 2 * math.pi)
        tip = [math.cos(a) * radius * 0.6, height * rng.uniform(0.5, 0.8), math.sin(a) * radius * 0.6]
        pts = [_mul(tip, i / 3) for i in range(4)]
        tube(pts, [0.04, 0.03, 0.02, 0.012], 5, 1.0, 0.8, stems)
    # A leaning stem's foot ring dips below its origin; level it on the
    # ground, or scaled-up bushes sink under the check's tolerance.
    for p in stems[0]:
        p[1] = max(p[1], 0.0)
    centres, sizes = [], []
    for _ in range(rng.randint(60, 90)):
        while True:
            q = [rng.uniform(-1, 1) for _ in range(3)]
            if sum(c * c for c in q) <= 1.0:
                break
        size = rng.uniform(0.55, 0.95)
        # Above the ground whichever way the card turns.
        y = max(size * 0.72, height * 0.55 + q[1] * height * 0.45)
        centres.append([q[0] * radius, y, q[2] * radius])
        sizes.append(size)
    leaves = ([], [], [], [])
    leaf_cards(rng, centres, sizes, leaves)
    return z.mesh([(*stems, bark), (*leaves, leaf)], f"bush_{variant}"), stems[0] + leaves[0]


def foliage(z):
    """Trees and bushes: thick in the belt between the fence and the
    ground's edge, a few in the yard. Trunks collide (a trimesh of the trunk
    alone) and join the solids; crowns and bushes don't collide."""
    rng = z.rng
    leaf = z.cutout_material("leaf", leaf_rgba(rng), roughness=0.7, surface="grass",
                             double_sided=False)
    bark = z.png_material("bark", bark_rgba(rng), roughness=0.85)
    trees = [tree_meshes(z, rng, leaf, bark, v) for v in range(TREE_VARIANTS)]
    bushes = [bush_mesh(z, rng, leaf, bark, v) for v in range(BUSH_VARIANTS)]
    hx0, hx1, hz0, hz1 = HANGAR
    ox0, ox1, oz0, oz1 = OFFICE
    fx0, fx1, fz0, fz1 = FENCE
    ex0, ex1, ez0, ez1 = ENCLOSURE

    def clear(x, zz, r):
        if abs(x) > BELT or abs(zz) > BELT:
            return False
        if abs(x) < ROAD_X + r and zz > hz1 - 1.0:
            return False  # the road, the gate and its apron
        if math.dist((x, zz), SPAWN_XZ) < 6.0 + r:
            return False
        for a, b, c, d, m in ((hx0, hx1, hz0, hz1, 1.5), (ox0, ox1, oz0, oz1, 1.5),
                              (ex0, ex1, ez0, ez1, 1.0), (-16.0, 16.0, hz1, 22.0, 0.5)):
            if a - m - r <= x <= b + m + r and c - m - r <= zz <= d + m + r:
                return False
        on_fence_x = min(abs(x - fx0), abs(x - fx1)) < 0.8 + r and fz0 - 1 <= zz <= fz1 + 1
        on_fence_z = min(abs(zz - fz0), abs(zz - fz1)) < 0.8 + r and fx0 - 1 <= x <= fx1 + 1
        return not (on_fence_x or on_fence_z)

    def clear_of_buildings(x, zz, r):
        for a, b, c, d in ((hx0, hx1, hz0, hz1), (ox0, ox1, oz0, oz1), (ex0, ex1, ez0, ez1)):
            if a - r <= x <= b + r and c - r <= zz <= d + r:
                return False
        return True

    def inside(x, zz):
        return fx0 < x < fx1 and fz0 < zz < fz1

    def spot(want_inside, near_edge):
        for _ in range(400):
            if want_inside:
                x, zz = rng.uniform(fx0 + 1, fx1 - 1), rng.uniform(fz0 + 1, fz1 - 1)
                # Along the fence and the buildings' walls, where the mower never went.
                if near_edge and min(x - fx0, fx1 - x, zz - fz0, fz1 - zz) > 4.0 \
                        and rng.random() < 0.8:
                    continue
            else:
                x, zz = rng.uniform(-BELT, BELT), rng.uniform(-BELT, BELT)
                if inside(x, zz):
                    continue
                # Thicker towards the corners.
                corner = min(abs(abs(x) - BELT), abs(abs(zz) - BELT))
                if corner > 4.0 and rng.random() < 0.35:
                    continue
            yield x, zz

    counts = {"trees": 0, "bushes": 0}

    def place_tree(want_inside, n):
        placed = 0
        for x, zz in spot(want_inside, False):
            if placed == n:
                break
            k = rng.randrange(len(trees))
            trunk, crown, r0, reach, crown_pts = trees[k]
            scale = rng.uniform(0.8, 1.2)
            r = r0 * scale + 0.15
            # The whole crown on the ground (the check's 80 x 80) and clear
            # of the buildings; over the fence is fine, it's higher.
            if max(abs(x), abs(zz)) + reach * scale > gts.GROUND_HALF - 0.1:
                continue
            if not clear(x, zz, 1.0) or not clear_of_buildings(x, zz, reach * scale):
                continue
            aabb = ([x - r, G, zz - r], [x + r, G + 3.0, zz + r])
            if any(gts.overlaps(aabb, s, 0.8) for s in z.solid):
                continue
            rot = yaw_q(rng.uniform(0, 360))
            t = (x, MUD_TOP, zz)
            # Tested once the draws are made, so a candidate that passes
            # leaves the random stream as it was.
            if not on_ground(z, (trunk, crown), t, rot, scale):
                continue
            if pokes_into(placed_pts(crown_pts, t, rot, scale),
                          [s for s in z.solid if id(s) not in z.crown_through]):
                continue
            if in_pond(x, zz, r):
                continue  # a crown may lean over the water; a trunk stays out
            z.place(trunk, t, rot, {"prefab": "prop", "params": {}}, name="tree_trunk", scale=scale)
            z.place(crown, t, rot, {"prefab": "prop", "params": {"collide": False}},
                    name="tree_crown", scale=scale)
            z.solid.append(aabb)
            z.crown_through.add(id(aabb))
            placed += 1
        return placed

    def place_bush(want_inside, n):
        placed = 0
        for x, zz in spot(want_inside, True):
            if placed == n:
                break
            scale = rng.uniform(0.8, 1.25)
            if not clear(x, zz, 1.2 * scale):
                continue
            r = 1.0 * scale
            aabb = ([x - r, G, zz - r], [x + r, G + 1.5, zz + r])
            if any(gts.overlaps(aabb, s, 0.1) for s in z.solid):
                continue
            bush, pts = bushes[rng.randrange(len(bushes))]
            rot = yaw_q(rng.uniform(0, 360))
            t = (x, MUD_TOP, zz)
            # That box is the bush's middle; its leaves reach further.
            if not on_ground(z, (bush,), t, rot, scale) \
                    or pokes_into(placed_pts(pts, t, rot, scale), z.solid) \
                    or in_pond(x, zz, r):
                continue
            z.place(bush, t, rot, {"prefab": "prop", "params": {"collide": False}},
                    name="bush", scale=scale)
            placed += 1
        return placed

    counts["trees"] = place_tree(False, TREES_OUT) + place_tree(True, TREES_IN)
    counts["bushes"] = place_bush(False, BUSHES_OUT) + place_bush(True, BUSHES_IN)
    return counts


def on_ground(z, meshes, t, rot, scale):
    """What gts.check asks of a placed node: its rotated bounds on the
    80 x 80 ground and not below it. Those bounds reach further than the
    vertices do, but they're what the check tests."""
    for m in meshes:
        lo, hi = [1e30] * 3, [-1e30] * 3
        for p in z.doc["meshes"][m]["primitives"]:
            a = z.doc["accessors"][p["attributes"]["POSITION"]]
            lo = [min(u, v) for u, v in zip(lo, a["min"])]
            hi = [max(u, v) for u, v in zip(hi, a["max"])]
        lo, hi = gts.node_aabb((lo, hi), t, rot, [scale] * 3)
        if min(lo[0], lo[2]) < -gts.GROUND_HALF or max(hi[0], hi[2]) > gts.GROUND_HALF \
                or lo[1] < gts.GROUND_Y - 0.01:
            return False
    return True


def placed_pts(pts, t, rot, scale):
    """Local points scaled, rotated and moved to `t`."""
    m = gts.quat_matrix(rot)
    return [[(m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2]) * scale + t[i] for i in range(3)]
            for p in pts]


def pokes_into(pts, boxes):
    """Whether any point lies inside any of `boxes` (checking only the boxes
    that the points' bounds overlap)."""
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    near = [b for b in boxes if gts.overlaps((lo, hi), b)]
    return any(all(b[0][i] < p[i] < b[1][i] for i in range(3)) for b in near for p in pts)


def check_foliage(doc, blob):
    """Leaves are single-sided MASK (the crown-normal trick needs no
    back-face flip); crowns and bushes don't collide, trunks do; no trunk
    stands on the road or near the spawn. From the written vertices (`blob`
    is the binary chunk): no crown or bush point inside a solid's bounds,
    bar the concrete fence's parts and, for crowns, the poles they may
    overgrow; no trunk's lower 3 m in a solid, fence included; no bush
    grows through a trunk."""
    bad = []
    for m in doc["materials"]:
        if m.get("name") == "leaf" and (m.get("alphaMode") != "MASK" or m.get("doubleSided")):
            bad.append("leaf material must be single-sided MASK")
    for n in doc["nodes"]:
        name, params = n.get("name"), n.get("extras", {}).get("params", {})
        if name in ("tree_crown", "bush") and params.get("collide", True):
            bad.append(f"{name} at {n['translation']} collides")
        if name == "tree_trunk":
            x, _, zz = n["translation"]
            if params.get("collide", True) is False:
                bad.append(f"tree_trunk at {n['translation']} doesn't collide")
            if (abs(x) < ROAD_X and zz > HANGAR[3] - 1.0) or math.dist((x, zz), SPAWN_XZ) < 6.0:
                bad.append(f"tree_trunk at {n['translation']} on the road or spawn")
    bad += foliage_overlaps(doc, blob)
    if bad:
        print(f"check: {len(bad)} foliage problems: {bad[:5]}", file=sys.stderr)
    return not bad


def node_points(doc, blob, node):
    """A mesh node's vertices, in the world, read from the binary chunk."""
    m = gts.quat_matrix(node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    s, t = node.get("scale", [1.0] * 3), node["translation"]
    out = []
    for p in doc["meshes"][node["mesh"]]["primitives"]:
        a = doc["accessors"][p["attributes"]["POSITION"]]
        v = doc["bufferViews"][a["bufferView"]]
        off, stride = v.get("byteOffset", 0) + a.get("byteOffset", 0), v.get("byteStride", 12)
        for i in range(a["count"]):
            q = struct.unpack_from("<3f", blob, off + i * stride)
            out.append([sum(m[r][c] * q[c] * s[c] for c in range(3)) + t[r] for r in range(3)])
    return out


def node_bounds(doc, node):
    """A mesh node's world bounds, from its accessors' min/max, as gts.check."""
    lo, hi = [1e30] * 3, [-1e30] * 3
    for p in doc["meshes"][node["mesh"]]["primitives"]:
        a = doc["accessors"][p["attributes"]["POSITION"]]
        lo = [min(u, w) for u, w in zip(lo, a["min"])]
        hi = [max(u, w) for u, w in zip(hi, a["max"])]
    return gts.node_aabb((lo, hi), node["translation"], node.get("rotation"),
                         node.get("scale"))


def solid_nodes(doc, exempt=()):
    """(bounds, name) of every colliding mesh node that isn't foliage, a flat
    ground piece, something hanging, or in `exempt`."""
    skip = {"tree_trunk", "tree_crown", "bush", "grass", "apron", "road", "mud",
            "hangar_floor", "office_floor", "lantern"} | set(exempt)
    return [(node_bounds(doc, n), n["name"]) for n in doc["nodes"] if "mesh" in n
            and n.get("name") not in skip
            and n.get("extras", {}).get("prefab") != "hanging"
            and n.get("extras", {}).get("params", {}).get("collide", True)]


def foliage_overlaps(doc, blob):
    """check_foliage's vertex-level half: a list of problems."""
    points = lambda node: node_points(doc, blob, node)
    bounds = lambda node: node_bounds(doc, node)
    named = lambda n: [x for x in doc["nodes"] if x.get("name") == n]
    bad = []
    solids = solid_nodes(doc, exempt=("fence_post", "fence_panel"))
    fence = [(bounds(n), n["name"]) for n in named("fence_panel") + named("fence_post")]
    trunks = []
    for n in named("tree_trunk"):
        # Its bounds, not its points: rings a metre apart straddle a
        # fallen panel.
        foot = [p for p in points(n) if p[1] < MUD_TOP + 3.0]
        foot = ([min(p[i] for p in foot) for i in range(3)],
                [max(p[i] for p in foot) for i in range(3)])
        for box, name in solids + fence:
            if gts.overlaps(foot, box):
                bad.append(f"tree_trunk at ({n['translation'][0]:.1f}, "
                           f"{n['translation'][2]:.1f}) stands in {name}")
        low = [p for p in points(n) if p[1] < MUD_TOP + 2.0]
        c = [sum(p[0] for p in low) / len(low), sum(p[2] for p in low) / len(low)]
        trunks.append((c, max(math.dist(c, (p[0], p[2])) for p in low)))
    for n in named("tree_crown") + named("bush"):
        pts = points(n)
        where = f"{n['name']} at ({n['translation'][0]:.1f}, {n['translation'][2]:.1f})"
        for box, name in solids:
            if n["name"] == "tree_crown" and name.startswith("pole"):
                continue
            if pokes_into(pts, [box]):
                bad.append(f"{where} has leaves inside {name}")
        if n["name"] == "bush":
            low = [p for p in pts if p[1] < MUD_TOP + 2.0]
            for c, r in trunks:
                if any(math.dist(c, (p[0], p[2])) < r for p in low):
                    bad.append(f"{where} grows through the trunk at ({c[0]:.1f}, {c[1]:.1f})")
    return bad


# --- things on ropes (§15) ------------------------------------------------------

# Under the hangar's roof beams: (x, beam row 1-4, rope length). A draught
# through the door and the roof's holes sways them, weaker than outside.
LANTERNS = ((-5.0, 1, 2.0), (3.0, 2, 1.4), (-2.0, 3, 2.4), (1.0, 4, 2.2))
LANTERN_WIND = 0.35
LANTERN_SCALE = 1.3
# Each lantern's flame (§12's point light, carried as it swings), at the
# chimney's centre in the mesh's own units. Four of these light the hangar,
# about as much in all as the two roof lamps they replaced (2 at 10).
LANTERN_LIGHT = {"color": [1.0, 0.72, 0.42], "intensity": 6.0, "radius": 11.0,
                 "source_radius": 0.05, "at": [0.0, -0.17, 0.0]}
# The glass glows with it (glTF emissiveFactor, linear).
LANTERN_GLOW = [1.0, 0.72, 0.38]


def lantern_mesh(z):
    """A kerosene lantern hung by its handle: a wire loop, a conical cap, a
    glass chimney in four guard wires, a base with a floor. Its origin is the
    top of the handle, where the rope ties on; it hangs down -Y."""
    rust = z.material("rust")
    glass = z.png_material("lantern_glass", gts.png_rgba(
        4, 4, [bytearray([214, 205, 170, 255] * 4) for _ in range(4)]), roughness=0.25)
    z.doc["materials"][glass]["emissiveFactor"] = LANTERN_GLOW
    metal, pane = ([], [], [], []), ([], [], [], [])
    # The handle: a half circle standing up from the cap.
    arc = [[0.07 * math.cos(a), -0.07 + 0.07 * math.sin(a), 0.0]
           for a in [math.pi * k / 10 for k in range(11)]]
    tube(arc, [0.004] * len(arc), 6, 1.0, 0.3, metal)
    tube([[0, -0.10, 0], [0, -0.075, 0]], [0.075, 0.02], 16, 2.0, 0.3, metal)  # cap
    tube([[0, -0.235, 0], [0, -0.20, 0], [0, -0.135, 0], [0, -0.10, 0]],
         [0.045, 0.06, 0.06, 0.045], 16, 1.0, 0.3, pane)  # chimney
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        x, zz = 0.068 * math.cos(a), 0.068 * math.sin(a)
        tube([[x, -0.24, zz], [x, -0.10, zz]], [0.004, 0.004], 5, 1.0, 0.3, metal)
    tube([[0, -0.30, 0], [0, -0.235, 0]], [0.07, 0.07], 16, 2.0, 0.3, metal)  # base
    pos, nrm, uv, idx = metal
    centre = len(pos)
    pos.append([0.0, -0.30, 0.0])
    nrm.append([0.0, -1.0, 0.0])
    uv.append([0.5, 0.5])
    for k in range(17):
        a = 2 * math.pi * k / 16
        pos.append([0.07 * math.cos(a), -0.30, 0.07 * math.sin(a)])
        nrm.append([0.0, -1.0, 0.0])
        uv.append([0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a)])
    idx += [i for k in range(16) for i in (centre, centre + 1 + k, centre + 2 + k)]
    return z.mesh([(*metal, rust), (*pane, glass)], "lantern")


def lanterns(z):
    """The hangar's lanterns, each a `hanging` node (§15) at its rest pose:
    the lantern's handle `length` below the underside of a roof beam. Built
    last, drawing nothing from the seed, so the rest of a level is unchanged."""
    x0, x1, z0, z1 = HANGAR
    d = (z1 - z0 + 0.4) / 5  # the roof's rows, as in hangar()
    mesh = lantern_mesh(z)
    for k, (x, row, length) in enumerate(LANTERNS):
        beam_z = z0 - 0.2 + row * d
        underside = G + HANGAR_HEIGHT - 0.4
        z.place(mesh, (x, underside - length, beam_z), yaw_q(35.0 * k),
                {"prefab": "hanging", "params": {"length": length, "wind": LANTERN_WIND,
                                                 "light": LANTERN_LIGHT}},
                name="lantern", scale=LANTERN_SCALE)
    return len(LANTERNS)


def check_hanging(doc, blob):
    """Each `hanging` node's rope is tied to the underside of something solid
    (within 5 cm, inside its footprint), is a positive length, and at rest
    neither the rope nor what hangs from it is inside a solid. A light it
    carries has a positive intensity and radius and sits within the item."""
    solids = solid_nodes(doc)
    bad = []
    for n in doc["nodes"]:
        if n.get("extras", {}).get("prefab") != "hanging":
            continue
        length = n["extras"].get("params", {}).get("length", 1.5)
        x, y, zz = n["translation"]
        where = f"{n.get('name')} at ({x:.1f}, {zz:.1f})"
        if not isinstance(length, (int, float)) or length <= 0:
            bad.append(f"{where}: rope length {length!r}")
            continue
        light = n["extras"].get("params", {}).get("light")
        if light is not None:
            if not (light.get("intensity", 12.0) > 0 and light.get("radius", 10.0) > 0):
                bad.append(f"{where}: its light has no intensity or radius")
            if "mesh" in n:
                (lo, hi), c = node_bounds(doc, n), light.get("at", [0.0, 0.0, 0.0])
                sc = n.get("scale", [1.0] * 3)
                m = gts.quat_matrix(n.get("rotation", (0.0, 0.0, 0.0, 1.0)))
                p = [sum(m[r][k] * c[k] * sc[k] for k in range(3)) + n["translation"][r]
                     for r in range(3)]
                if not all(lo[i] <= p[i] <= hi[i] for i in range(3)):
                    bad.append(f"{where}: its light is outside it")
        top = y + length
        if not any(lo[0] < x < hi[0] and lo[2] < zz < hi[2] and 0 <= lo[1] - top <= 0.05
                   for (lo, hi), _ in solids):
            bad.append(f"{where}: its rope's top ({top:.2f}) isn't under anything solid")
        # The rope at rest, from 5 mm under its knot, and the item.
        rope = [[x, top - 0.005 - (length - 0.005) * k / 16, zz] for k in range(17)]
        for box, name in solids:
            if pokes_into(rope, [box]):
                bad.append(f"{where}: its rope passes through {name}")
            if "mesh" in n and pokes_into(node_points(doc, blob, n), [box]):
                bad.append(f"{where}: it's inside {name}")
    if bad:
        print(f"check: {len(bad)} hanging problems: {bad[:5]}", file=sys.stderr)
    return not bad


def targets(z):
    # Floating at roughly eye height, spread across the yard so a new game
    # opens with one in sight from the gate.
    for x, y, zz in (
        (0.0, PAVE_TOP + 2.0, 16.0),    # the hangar door
        (-8.0, MUD_TOP + 1.7, 20.0),    # the yard, left of the road
        (-14.0, MUD_TOP + 1.7, -4.0),   # the pond bank
        (2.5, PAVE_TOP + 2.1, 24.0),    # the gate apron
        (-18.0, MUD_TOP + 1.6, -16.0),  # behind the hangar
        (21.0, MUD_TOP + 1.9, -9.5),    # the office front
        (14.0, MUD_TOP + 1.8, 18.0),    # the east yard
    ):
        z.place(None, (x, y, zz), name="target",
                extras={"prefab": "target", "params": TARGET_PARAMS})


def markers(z):
    lamp = {"prefab": "point_light", "params": LAMP}
    # The office's; the hangar is lit by its lanterns (see `lanterns`).
    for x, y, zz in ((22.0, SLAB + STOREY - 0.4, -16.0), (21.0, SLAB + 2 * STOREY - 0.4, -16.0)):
        z.place(None, (x, G + y, zz), extras=lamp, name="lamp")
    z.place(None, (0.0, PAVE_TOP, 37.0), name="player_start",
            extras={"prefab": "player_start", "params": {"yaw": 270.0}})
    z.place(None, (0.0, G, 0.0), name="environment",
            extras={"prefab": "environment", "params": ENVIRONMENT})


def build(seed):
    z = Zone(random.Random(seed))
    ground(z)
    hangar(z)
    office(z)
    fence(z)
    enclosure(z)
    counts = props(z)
    counts.update(foliage(z))
    counts["grass"] = grass(z)
    markers(z)
    targets(z)
    counts["lanterns"] = lanterns(z)
    return z, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", default=os.path.join("scratch", "zone.glb"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--check", action="store_true", help="validate after writing")
    args = ap.parse_args()
    needed = [os.path.join(ASSETS, p) for p in PROPS.values()]
    needed += [os.path.join(ASSETS, f"polyhaven_{pid}", f"{pid}_arm_2k.jpg")
               for pid, _, _ in TEXTURES.values()]
    needed += [os.path.join(FENCE_PACK, f"{FENCE_MAPS}_{m}.png")
               for m in ("Color", "Opacity", "NormalGL")]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        print(f"missing {len(missing)} asset files (e.g. {missing[0]}): "
              "run `python3 tools/fetch_assets.py` first", file=sys.stderr)
        return 1
    z, counts = build(args.seed)
    z.write(args.out)
    tris = sum(z.doc["accessors"][p["indices"]]["count"] // 3
               for n in z.doc["nodes"] if "mesh" in n
               for p in z.doc["meshes"][n["mesh"]]["primitives"])
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB): "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
          + f"\n  {tris:,} triangles placed, {len(z.doc['images'])} images")
    # Placement gives up after so many candidates, silently; check it didn't.
    short = [f"{k} {counts[k]} of {want}" for k, want in
             (("trees", TREES_OUT + TREES_IN), ("bushes", BUSHES_OUT + BUSHES_IN))
             if counts[k] < want]
    if short:
        print(f"check: foliage fell short: {', '.join(short)}", file=sys.stderr)
    if args.check and not (gts.check(z.doc) and check_occlusion(z.doc)
                           and check_foliage(z.doc, z.bin) and check_hanging(z.doc, z.bin)
                           and not short):
        return 1
    return 0


def check_occlusion(doc):
    """Every material here with an MR texture takes it from a Poly Haven
    ARM map, so each must name the same image as its occlusion."""
    source = lambda slot: doc["textures"][slot["index"]]["source"]
    bad = [m.get("name", i) for i, m in enumerate(doc["materials"])
           if "metallicRoughnessTexture" in m.get("pbrMetallicRoughness", {})
           and ("occlusionTexture" not in m
                or source(m["occlusionTexture"])
                != source(m["pbrMetallicRoughness"]["metallicRoughnessTexture"]))]
    if bad:
        print(f"check: {len(bad)} ARM materials without their occlusion: {bad[:5]}",
              file=sys.stderr)
    return not bad


if __name__ == "__main__":
    sys.exit(main())
