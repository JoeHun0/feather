#!/usr/bin/env python3
"""Compose a nature test level from the Kenney Nature Kit (CC0).

Why this exists
---------------
Every other test scene is procedural boxes and spheres. This one is built from
real authored models: closer to the game, and a realistic workload for
culling, instancing, shadows and lights. Like gen_testscene.py it writes one
self-contained level file with §18 prefab `extras`, since a level is a single
glTF in the engine's design, so the engine needs no changes to load it.

The models come from the gitignored cache that tools/fetch_assets.py fills:
    python3 tools/fetch_assets.py
    python3 tools/gen_naturescene.py --check
    cargo run -- scratch/nature.glb

What the composer does to the kit, and why
------------------------------------------
* **Materials become dielectric.** Every Kenney material is
  KHR_materials_unlit with metallic = 1, roughness = 1: flat colours meant to
  be shown unlit. The engine ignores `unlit`, so read literally the forest
  would render as dark, tinted, fully rough *metal*. Each material keeps its
  baseColorFactor and becomes metallic 0 / roughness 0.9. They are deduped
  by name across files (the kit shares 23 of them).
* **Inner transforms are baked into the vertices**, so every placed node is a
  plain translation + yaw + *uniform* scale. That keeps the engine's
  mat3(model) normal rule true by construction.
* **Each model is instanced**: its geometry is written once, and every
  placement references it.
* **Models are seated on the ground** from their true lowest vertex. The
  kit's roots sit a few centimetres low, which --check would reject.
* **Scale x4**: Kenney units are small (a tree is 1.7), and at x4 a tree is
  ~7 m, a tent 2.2 m and a fence 1.4 m next to the 1.8 m player.

--check reuses gen_testscene.check(), so both tools enforce the same engine
constraints.
"""

import argparse
import json
import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_testscene as gts  # noqa: E402  (constants + the shared check)

KIT = os.path.join("scratch", "assets", "kenney_nature_kit")
GROUND_Y = gts.GROUND_Y
HALF = gts.GROUND_HALF
SCALE = 4.0
JITTER = 0.15

# Everything placed in the level: (count by density, candidate models).
DENSITY = {
    "low": {"trees": 60, "rocks": 18, "small": 160},
    "med": {"trees": 130, "rocks": 36, "small": 420},
    "high": {"trees": 260, "rocks": 70, "small": 900},
}
TREES = [
    "tree_default", "tree_oak", "tree_detailed", "tree_fat", "tree_tall",
    "tree_pineDefaultA", "tree_pineDefaultB", "tree_pineRoundA", "tree_pineRoundC",
    "tree_pineTallA", "tree_pineTallB", "tree_cone", "tree_default_dark",
    "tree_oak_dark", "tree_default_fall", "tree_oak_fall",
]
ROCKS = ["rock_largeA", "rock_largeB", "rock_largeC", "rock_largeD", "rock_tallA",
         "rock_tallB", "rock_tallE", "stone_largeA", "stone_largeC"]
SMALL = ["grass", "grass_large", "grass_leafs", "flower_redA", "flower_yellowB",
         "flower_purpleC", "plant_bush", "plant_bushSmall", "mushroom_red",
         "mushroom_tanGroup", "rock_smallA", "rock_smallFlatB"]
# Small scatter that is big enough to matter keeps its shadow.
SMALL_CASTS = {"plant_bush"}

CAMP = (0.0, 20.0)  # the clearing, north of the default spawn (z = 8)
CAMP_CLEAR = 9.0
PATH = [(0.0, 12.0), (-6.0, 0.0), (-8.0, -14.0), (-4.0, -26.0), (0.0, -36.0)]


# --- GLB in -------------------------------------------------------------------

def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, _, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"{path}: not a GLB")
    jlen, _ = struct.unpack_from("<II", data, 12)
    doc = json.loads(data[20:20 + jlen])
    off = 20 + jlen
    blen, _ = struct.unpack_from("<II", data, off)
    return doc, data[off + 8:off + 8 + blen]


COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
FORMATS = {5126: "f", 5125: "I", 5123: "H", 5121: "B"}


def read_accessor(doc, bin_, index):
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    n = COMPONENTS[acc["type"]]
    fmt = FORMATS[acc["componentType"]]
    size = struct.calcsize(fmt)
    stride = view.get("byteStride", n * size)
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return [struct.unpack_from(f"<{n}{fmt}", bin_, base + i * stride)
            for i in range(acc["count"])]


def trs_matrix(node):
    """Row-major 4x4 from a node's TRS (or matrix)."""
    if "matrix" in node:
        m = node["matrix"]  # column-major
        return [[m[c * 4 + r] for c in range(4)] for r in range(4)]
    t = node.get("translation", [0, 0, 0])
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = node.get("scale", [1, 1, 1])
    r = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return [[r[i][0] * s[0], r[i][1] * s[1], r[i][2] * s[2], t[i]] for i in range(3)] + [[0, 0, 0, 1]]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def xform_point(m, p):
    return tuple(m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3] for i in range(3))


def xform_normal(m, n):
    # Correct for rotation + uniform scale (renormalised below). load_model
    # refuses non-uniform inner scale, which would need the inverse-transpose.
    v = tuple(m[i][0] * n[0] + m[i][1] * n[1] + m[i][2] * n[2] for i in range(3))
    length = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / length for c in v)


def load_model(name):
    """A model as a list of primitives (positions, normals, uvs, indices,
    material name), with every inner node transform baked into the vertices."""
    path = os.path.join(KIT, name + ".glb")
    doc, bin_ = read_glb(path)
    prims = []

    def walk(ni, parent):
        node = doc["nodes"][ni]
        sc = node.get("scale", [1, 1, 1])
        if "matrix" in node or max(sc) - min(sc) > 1e-6:
            raise ValueError(f"{path}: node {ni} has a matrix or non-uniform scale; "
                             "baking its normals would need the inverse-transpose")
        world = matmul(parent, trs_matrix(node))
        if "mesh" in node:
            for p in doc["meshes"][node["mesh"]]["primitives"]:
                if p.get("mode", 4) != 4:
                    continue
                a = p["attributes"]
                pos = [xform_point(world, v) for v in read_accessor(doc, bin_, a["POSITION"])]
                nrm = ([xform_normal(world, v) for v in read_accessor(doc, bin_, a["NORMAL"])]
                       if "NORMAL" in a else None)
                uv = read_accessor(doc, bin_, a["TEXCOORD_0"]) if "TEXCOORD_0" in a else None
                idx = ([i[0] for i in read_accessor(doc, bin_, p["indices"])]
                       if "indices" in p else list(range(len(pos))))
                mat = doc["materials"][p["material"]] if "material" in p else {"name": "_default"}
                prims.append((pos, nrm, uv, idx, mat))
        for c in node.get("children", []):
            walk(c, world)

    ident = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
    scene = doc["scenes"][doc.get("scene", 0)]
    for ni in scene["nodes"]:
        walk(ni, ident)
    if not prims:
        raise ValueError(f"{path}: no triangle geometry")
    lo = [min(v[i] for p in prims for v in p[0]) for i in range(3)]
    hi = [max(v[i] for p in prims for v in p[0]) for i in range(3)]
    return {"prims": prims, "lo": lo, "hi": hi}


# --- GLB out ------------------------------------------------------------------

class Level:
    def __init__(self):
        self.bin = bytearray()
        self.doc = {
            "asset": {"version": "2.0", "generator": "feather tools/gen_naturescene.py"},
            "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
            "accessors": [], "bufferViews": [], "materials": [], "buffers": [],
        }
        self.materials = {}  # name -> index
        self.meshes = {}     # model name -> mesh index, once written
        self.models = {}     # model name -> loaded model (bounds for placement)

    def model(self, name):
        """A kit model's geometry and bounds, loaded once. Measuring a candidate
        does not write it: only placed models end up in the level."""
        if name not in self.models:
            self.models[name] = load_model(name)
        return self.models[name]

    def _view(self, data, target):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.doc["bufferViews"].append({"buffer": 0, "byteOffset": len(self.bin),
                                        "byteLength": len(data), "target": target})
        self.bin += data
        return len(self.doc["bufferViews"]) - 1

    def _accessor(self, values, fmt, gtype, ctype, target, minmax=False):
        n = COMPONENTS[gtype]
        flat = [c for v in values for c in (v if n > 1 else (v,))]
        view = self._view(struct.pack(f"<{len(flat)}{fmt}", *flat), target)
        acc = {"bufferView": view, "componentType": ctype, "count": len(values), "type": gtype}
        if minmax:
            acc["min"] = [min(v[i] for v in values) for i in range(n)]
            acc["max"] = [max(v[i] for v in values) for i in range(n)]
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    def _material(self, mat):
        name = mat.get("name", "_default")
        if name not in self.materials:
            pbr = mat.get("pbrMetallicRoughness", {})
            # Unlit flat colour -> lit dielectric (see the module docstring).
            self.doc["materials"].append({
                "name": name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": pbr.get("baseColorFactor", [1, 1, 1, 1]),
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                },
            })
            self.materials[name] = len(self.doc["materials"]) - 1
        return self.materials[name]

    def mesh(self, name):
        """The level mesh for a kit model, written on first use only."""
        if name not in self.meshes:
            model = self.model(name)
            prims = []
            for pos, nrm, uv, idx, mat in model["prims"]:
                attrs = {"POSITION": self._accessor(pos, "f", "VEC3", 5126, 34962, minmax=True)}
                if nrm:
                    attrs["NORMAL"] = self._accessor(nrm, "f", "VEC3", 5126, 34962)
                if uv:
                    attrs["TEXCOORD_0"] = self._accessor(uv, "f", "VEC2", 5126, 34962)
                prims.append({
                    "attributes": attrs,
                    "indices": self._accessor(idx, "I", "SCALAR", 5125, 34963),
                    "material": self._material(mat),
                })
            self.doc["meshes"].append({"name": name, "primitives": prims})
            self.meshes[name] = len(self.doc["meshes"]) - 1
        return self.meshes[name]

    def node(self, name, translation, yaw=0.0, scale=1.0, extras=None, label=None):
        entry = {"translation": [float(c) for c in translation]}
        if name is not None:
            entry["mesh"] = self.mesh(name)
        if yaw:
            entry["rotation"] = [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]
        if scale != 1.0:
            entry["scale"] = [scale, scale, scale]
        if extras:
            entry["extras"] = extras
        entry["name"] = label or name
        self.doc["nodes"].append(entry)
        self.doc["scenes"][0]["nodes"].append(len(self.doc["nodes"]) - 1)

    def write(self, path):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.doc["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(self.doc, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942) + bytes(self.bin))


# --- Layout -------------------------------------------------------------------

class Placer:
    """Placement on the ground with spacing. Placed things reserve discs from
    each other. The app's own boxes and the spawn clearance are tested with
    exactly --check's box test, since a disc approximation disagrees with it
    at box corners."""

    def __init__(self, level, rng):
        self.level, self.rng = level, rng
        self.discs = []
        self.keep_out = gts.keep_out_boxes()

    def clear_of_app(self, name, x, z, yaw, scale):
        model = self.level.model(name)
        y = GROUND_Y - model["lo"][1] * scale
        rot = [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]
        aabb = gts.node_aabb((model["lo"], model["hi"]), (x, y, z), rot, (scale,) * 3)
        return not any(gts.overlaps(aabb, k) for k in self.keep_out)

    def footprint(self, name, scale):
        model = self.level.model(name)
        lo, hi = model["lo"], model["hi"]
        # Radius of the model's footprint under any yaw.
        return scale * max(math.hypot(x, z) for x in (lo[0], hi[0]) for z in (lo[2], hi[2]))

    def free(self, x, z, r, gap=0.0, bound=None):
        # `bound` is the full footprint when spacing is reduced, so a canopy
        # never overhangs the ground slab even if trunks pack closer.
        b = r if bound is None else bound
        if abs(x) + b > HALF - 0.5 or abs(z) + b > HALF - 0.5:
            return False
        return all(math.hypot(x - dx, z - dz) >= r + dr + gap for dx, dz, dr in self.discs)

    def reserve(self, x, z, r):
        self.discs.append((x, z, r))

    def put(self, name, x, z, yaw=None, scale=None, extras=None, reserve=True, lift=0.0):
        scale = scale or SCALE * (1 + self.rng.uniform(-JITTER, JITTER))
        yaw = self.rng.uniform(0, math.tau) if yaw is None else yaw
        model = self.level.model(name)
        # Seat the lowest vertex on the ground (plus `lift`, for decals).
        y = GROUND_Y - model["lo"][1] * scale + lift
        self.level.node(name, (x, y, z), yaw, scale, extras)
        if reserve:
            self.reserve(x, z, self.footprint(name, scale))

    def scatter(self, names, count, gap, avoid=(), extras=None, reserve=True,
                spacing=1.0, tries=60):
        """`spacing` scales the disc each placement is spaced by. Trees use
        ~0.5, roughly the trunk, so canopies can overlap as they do in a
        forest; only trunks need walking room."""
        placed = 0
        for _ in range(count):
            name = self.rng.choice(names)
            scale = SCALE * (1 + self.rng.uniform(-JITTER, JITTER))
            full = self.footprint(name, scale)
            r = full * spacing
            for _ in range(tries):
                x, z = self.rng.uniform(-HALF, HALF), self.rng.uniform(-HALF, HALF)
                yaw = self.rng.uniform(0, math.tau)
                if any(math.hypot(x - ax, z - az) < ar for ax, az, ar in avoid):
                    continue
                if self.free(x, z, r, gap, full) and self.clear_of_app(name, x, z, yaw, scale):
                    ex = extras(name) if callable(extras) else extras
                    self.put(name, x, z, yaw=yaw, scale=scale, extras=ex, reserve=False)
                    if reserve:
                        self.reserve(x, z, r)
                    placed += 1
                    break
        return placed


def path_points(step):
    """Points every `step` metres along the PATH polyline."""
    out = []
    for (ax, az), (bx, bz) in zip(PATH, PATH[1:]):
        n = max(1, int(math.hypot(bx - ax, bz - az) / step))
        out += [(ax + (bx - ax) * i / n, az + (bz - az) * i / n) for i in range(n)]
    return out + [PATH[-1]]


def prop(collide=True, shadow=True):
    return {"prefab": "prop", "params": {"collide": collide, "shadow": shadow}}


def build(density, seed):
    rng = random.Random(seed)
    level = Level()
    pl = Placer(level, rng)
    counts = DENSITY[density]

    # Ground cover: 4 m grass tiles over the whole 80 x 80 slab. Lifted 2 cm so
    # they don't z-fight the app's ground; no collider (the slab already
    # collides) and no shadow casting (a flat floor shadows nothing; see §11's
    # note on the ground rasterising the whole map).
    tile = SCALE
    n = int(2 * HALF / tile)
    for i in range(n):
        for j in range(n):
            x, z = -HALF + tile * (i + 0.5), -HALF + tile * (j + 0.5)
            pl.put("ground_grass", x, z, yaw=0.0, scale=SCALE, reserve=False,
                   extras=prop(collide=False, shadow=False), lift=0.02)

    # The camp: a tent, a fire with a warm light, seats around it.
    cx, cz = CAMP
    pl.put("campfire_stones", cx, cz, yaw=0.0, scale=SCALE)
    level.node(None, (cx, GROUND_Y + 0.8, cz), label="campfire_light", extras={
        "prefab": "point_light",
        "params": {"color": [1.0, 0.55, 0.25], "intensity": 30.0, "radius": 14.0,
                   "source_radius": 0.4},
    })
    pl.put("tent_detailedOpen", cx - 5.5, cz + 3.0, yaw=math.radians(120))
    for i, ang in enumerate((200, 290, 20)):
        a = math.radians(ang)
        pl.put("log" if i % 2 else "stump_round", cx + 3.2 * math.cos(a), cz + 3.2 * math.sin(a),
               yaw=a + math.pi / 2)
    pl.put("log_stack", cx + 5.0, cz + 4.5, yaw=math.radians(-30))
    level.node(None, (0.0, GROUND_Y, 13.0), label="player_start",
               extras={"prefab": "player_start", "params": {"yaw": 90.0}})

    # The path: stone slabs from the camp through the forest, a fence along one
    # stretch, and dim lamps (light-only markers) to exercise §12.
    for i, (x, z) in enumerate(path_points(3.2)):
        yaw = rng.uniform(0, math.tau)
        if pl.free(x, z, 1.2) and pl.clear_of_app("path_stone", x, z, yaw, SCALE):
            pl.put("path_stone", x, z, yaw=yaw, scale=SCALE, reserve=False,
                   extras=prop(collide=False, shadow=False))
        pl.reserve(x, z, 2.5)  # keep the trail clear of trees
        if i % 5 == 2:
            level.node(None, (x + 2.2, GROUND_Y + 2.2, z), label=f"path_lamp_{i}", extras={
                "prefab": "point_light",
                "params": {"color": [1.0, 0.8, 0.5], "intensity": 10.0, "radius": 9.0,
                           "source_radius": 0.15},
            })
    fence_yaw = math.radians(75)
    for x, z in path_points(3.9)[4:12]:
        if pl.free(x - 3.0, z, 1.6) and pl.clear_of_app("fence_simple", x - 3.0, z, fence_yaw, SCALE):
            pl.put("fence_simple", x - 3.0, z, yaw=fence_yaw, scale=SCALE)

    # A rock ridge along the east edge: cliff blocks, the occasional one
    # stacked, for silhouette and big shadow casters.
    for k in range(9):
        z = -30.0 + k * 4.2
        x = 34.0 + rng.uniform(-1.0, 1.0)
        name = rng.choice(["cliff_block_rock", "cliff_block_stone", "cliff_blockHalf_rock"])
        yaw = rng.choice([0, math.pi / 2])
        if pl.free(x, z, 2.9) and pl.clear_of_app(name, x, z, yaw, SCALE):
            pl.put(name, x, z, yaw=yaw, scale=SCALE)

    avoid = [(cx, cz, CAMP_CLEAR)]
    # Rocks before trees: fewer and bigger, they would find no room left.
    placed = {
        "rocks": pl.scatter(ROCKS, counts["rocks"], gap=1.0, avoid=avoid),
        "trees": pl.scatter(TREES, counts["trees"], gap=1.0, avoid=avoid, spacing=0.5),
        # Small stuff: no collider; only bushes cast. Not reserved, so it can
        # nestle against trees; it still avoids trees and rocks itself.
        "small": pl.scatter(SMALL, counts["small"], gap=0.2, reserve=False,
                            extras=lambda nm: prop(collide=False, shadow=nm in SMALL_CASTS)),
    }
    return level, placed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", default=os.path.join("scratch", "nature.glb"))
    ap.add_argument("--density", choices=sorted(DENSITY), default="med")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--check", action="store_true", help="validate after writing")
    args = ap.parse_args()

    if not os.path.isdir(KIT):
        print(f"{KIT} is missing: run `python3 tools/fetch_assets.py` first", file=sys.stderr)
        return 1
    level, placed = build(args.density, args.seed)
    level.write(args.out)
    want = DENSITY[args.density]
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KiB): "
          + ", ".join(f"{k} {placed[k]}/{want[k]}" for k in placed)
          + f", {len(level.meshes)} distinct models")
    if args.check and not gts.check(level.doc):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
