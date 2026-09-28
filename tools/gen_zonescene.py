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

# Alpha cutout (ARCHITECTURE.md §5). A chain-link enclosure round the
# compressor, from ambientCG's Fence006 (rusty diamond mesh), and grass tufts
# on the mud from a texture this script draws.
FENCE_PACK = os.path.join(ASSETS, "ambientcg_Fence006")
FENCE_MAPS = "Fence006_1K-PNG"
CHAINLINK_TILE = 1.0  # metres per texture repeat
ENCLOSURE = (16.8, 23.2, -6.2, 0.2)  # x0, x1, z0, z1
ENCLOSURE_BAY, ENCLOSURE_HEIGHT = 3.2, 2.0
GRASS_TUFTS = 600
GRASS_TUFT = (0.9, 0.55)  # width, height of each of a tuft's three cards

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
                        surface=None):
        """A double-sided MASK material (§5) from an RGBA base colour."""
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
                "doubleSided": True,
            }
            if normal_png is not None:
                m["normalTexture"] = {"index": self._png(normal_png, key + "/normal")}
            if surface:
                m["extras"] = {"surface": surface}
            self.doc["materials"].append(m)
            self.mats[key] = len(self.doc["materials"]) - 1
        return self.mats[key]

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


def ground(z):
    # Mud over the app's slab, then asphalt and concrete on the mud. All flat
    # and shadowless (they cast nothing but would fill the shadow map).
    z.box((-HALF, G, -HALF), (HALF, MUD_TOP, HALF), "mud", "mud", shadow=False)
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
            count(key, z.put(key, x, zz, floor=floor, rot=rot))
    # Tyre stacks, and a few lying about.
    for cx, cz, floor in ((-15.0, 24.0, MUD_TOP), (14.0, -24.0, MUD_TOP), (-10.0, 10.5, PAVE_TOP)):
        for k in range(rng.randint(2, 5)):
            count("tyre", z.put("tyre", cx, cz, floor=floor, rot=lying(),
                                lift=k * 0.165, check=(k == 0)))
    for _ in range(4):
        x, zz = rng.uniform(-28, 28), rng.uniform(-24, 26)
        count("tyre", z.put("tyre", x, zz, rot=lying()))
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
    for i, zz in enumerate((-24.0, -8.0, 8.0)):
        count("pole", z.put(f"pole_{(i + 1) % 3 + 1}", -36.0, zz, rot=yaw_q(0)))
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
    placed = 0
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
        z.place(tuft, (x, MUD_TOP, zz), yaw_q(rng.uniform(0, 360)),
                {"prefab": "prop", "params": {"collide": False}}, name="grass", scale=scale)
        placed += 1
    return placed


def markers(z):
    lamp = {"prefab": "point_light", "params": LAMP}
    for x, y, zz in ((-6.0, 6.8, 0.0), (6.0, 6.8, 6.0),
                     (22.0, SLAB + STOREY - 0.4, -16.0), (21.0, SLAB + 2 * STOREY - 0.4, -16.0)):
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
    counts["grass"] = grass(z)
    markers(z)
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
    if args.check and not (gts.check(z.doc) and check_occlusion(z.doc)):
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
