#!/usr/bin/env python3
"""Compose a *detail stress* level from Poly Haven CC0 scans.

Why this exists
---------------
The engine's content so far is low-poly and untextured. This scene is the
opposite: photoscanned models (17k-58k triangles each) with 2048x2048 PBR
textures, alpha-blended grass cards, and enough instances to reach millions
of triangles. It exists to *measure* where the engine stops coping with
detailed objects (loading, texture memory and shimmer, alpha foliage,
triangle throughput, the instance cap) before deciding what to build.

    python3 tools/fetch_assets.py
    python3 tools/gen_detailscene.py --density med --check
    cargo run -- --bench scratch/detail.glb

Unlike gen_naturescene.py it keeps the materials as authored: textures,
alphaMode and doubleSided pass through untouched, since those are what is
under test. Non-required extensions (the lantern glass's
KHR_materials_transmission) are dropped, which the engine ignores anyway.
Images are embedded once each, shared by every material that uses them.
"""

import argparse
import json
import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_testscene as gts  # noqa: E402
from gen_naturescene import (  # noqa: E402
    COMPONENTS, matmul, read_accessor, trs_matrix, xform_normal, xform_point,
)

ASSETS = os.path.join("scratch", "assets")
GROUND_Y, HALF = gts.GROUND_Y, gts.GROUND_HALF

MODELS = {
    "bust": "polyhaven_marble_bust_01/marble_bust_01_2k.gltf",
    "lantern": "polyhaven_Lantern_01/Lantern_01_2k.gltf",
    "rocks": "polyhaven_rock_moss_set_02/rock_moss_set_02_2k.gltf",
    "grass": "polyhaven_grass_medium_01/grass_medium_01_2k.gltf",
}
# Placements per model. `high` is sized to exceed the engine's 8192-instance
# budget (main + four cascades share it), since that limit is under test.
DENSITY = {
    "low": {"bust": 10, "lantern": 10, "rocks": 6, "grass": 40},
    "med": {"bust": 25, "lantern": 30, "rocks": 15, "grass": 150},
    "high": {"bust": 50, "lantern": 60, "rocks": 30, "grass": 400},
}
TEXTURE_SLOTS = ("baseColorTexture", "metallicRoughnessTexture")
MATERIAL_TEXTURES = ("normalTexture", "occlusionTexture", "emissiveTexture")


def load_gltf(path):
    """A .gltf with one external buffer, as (doc, bin, directory)."""
    with open(path) as f:
        doc = json.load(f)
    if len(doc.get("buffers", [])) != 1:
        raise ValueError(f"{path}: expected exactly one buffer")
    base = os.path.dirname(path)
    with open(os.path.join(base, doc["buffers"][0]["uri"]), "rb") as f:
        bin_ = f.read()
    return doc, bin_, base


class Level:
    def __init__(self):
        self.bin = bytearray()
        self.doc = {
            "asset": {"version": "2.0", "generator": "feather tools/gen_detailscene.py"},
            "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
            "accessors": [], "bufferViews": [], "buffers": [], "materials": [],
            "images": [], "textures": [], "samplers": [],
        }
        self.images = {}  # source path -> image index
        self.models = {}  # key -> {"mesh": index, "lo": [...], "hi": [...]}

    def _view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            view["target"] = target
        self.doc["bufferViews"].append(view)
        self.bin += data
        return len(self.doc["bufferViews"]) - 1

    def _accessor(self, values, fmt, gtype, ctype, target, minmax=False):
        n = COMPONENTS[gtype]
        flat = [c for v in values for c in (v if n > 1 else (v,))]
        acc = {"bufferView": self._view(struct.pack(f"<{len(flat)}{fmt}", *flat), target),
               "componentType": ctype, "count": len(values), "type": gtype}
        if minmax:
            acc["min"] = [min(v[i] for v in values) for i in range(n)]
            acc["max"] = [max(v[i] for v in values) for i in range(n)]
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    def _texture(self, src_doc, base, tex_index):
        """Copy one texture, embedding its image once however many materials
        (or models) use it."""
        src_tex = src_doc["textures"][tex_index]
        img = src_doc["images"][src_tex["source"]]
        path = os.path.normpath(os.path.join(base, img["uri"]))
        if path not in self.images:
            with open(path, "rb") as f:
                data = f.read()
            mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
            self.doc["images"].append({"bufferView": self._view(data), "mimeType": mime})
            self.images[path] = len(self.doc["images"]) - 1
        tex = {"source": self.images[path]}
        if "sampler" in src_tex:
            self.doc["samplers"].append(src_doc["samplers"][src_tex["sampler"]])
            tex["sampler"] = len(self.doc["samplers"]) - 1
        self.doc["textures"].append(tex)
        return len(self.doc["textures"]) - 1

    def _material(self, src_doc, base, index, cache):
        if index in cache:
            return cache[index]
        m = json.loads(json.dumps(src_doc["materials"][index]))  # deep copy
        m.pop("extensions", None)  # KHR_materials_transmission: not required
        pbr = m.get("pbrMetallicRoughness", {})
        for slot in TEXTURE_SLOTS:
            if slot in pbr:
                pbr[slot]["index"] = self._texture(src_doc, base, pbr[slot]["index"])
        for slot in MATERIAL_TEXTURES:
            if slot in m:
                m[slot]["index"] = self._texture(src_doc, base, m[slot]["index"])
        self.doc["materials"].append(m)
        cache[index] = len(self.doc["materials"]) - 1
        return cache[index]

    def model(self, key):
        """One level mesh per source model: every mesh node baked into the
        vertices (Poly Haven sets carry per-piece transforms), one primitive
        per source primitive, materials and textures carried over."""
        if key in self.models:
            return self.models[key]
        doc, bin_, base = load_gltf(os.path.join(ASSETS, MODELS[key]))
        prims, mats = [], {}
        lo, hi = [1e30] * 3, [-1e30] * 3

        def walk(ni, parent):
            node = doc["nodes"][ni]
            sc = node.get("scale", [1, 1, 1])
            if "matrix" in node or max(sc) - min(sc) > 1e-6:
                raise ValueError(f"{key}: node {ni} has a matrix or non-uniform scale")
            world = matmul(parent, trs_matrix(node))
            for p in doc["meshes"][node["mesh"]]["primitives"] if "mesh" in node else []:
                a = p["attributes"]
                pos = [xform_point(world, v) for v in read_accessor(doc, bin_, a["POSITION"])]
                for v in pos:
                    for i in range(3):
                        lo[i], hi[i] = min(lo[i], v[i]), max(hi[i], v[i])
                attrs = {"POSITION": self._accessor(pos, "f", "VEC3", 5126, 34962, minmax=True)}
                if "NORMAL" in a:
                    nrm = [xform_normal(world, v) for v in read_accessor(doc, bin_, a["NORMAL"])]
                    attrs["NORMAL"] = self._accessor(nrm, "f", "VEC3", 5126, 34962)
                if "TEXCOORD_0" in a:
                    uv = read_accessor(doc, bin_, a["TEXCOORD_0"])
                    attrs["TEXCOORD_0"] = self._accessor(uv, "f", "VEC2", 5126, 34962)
                idx = [i[0] for i in read_accessor(doc, bin_, p["indices"])]
                prim = {"attributes": attrs,
                        "indices": self._accessor(idx, "I", "SCALAR", 5125, 34963)}
                if "material" in p:
                    prim["material"] = self._material(doc, base, p["material"], mats)
                prims.append(prim)
            for c in node.get("children", []):
                walk(c, world)

        ident = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
        for ni in doc["scenes"][doc.get("scene", 0)]["nodes"]:
            walk(ni, ident)
        self.doc["meshes"].append({"name": key, "primitives": prims})
        self.models[key] = {"mesh": len(self.doc["meshes"]) - 1, "lo": lo, "hi": hi}
        return self.models[key]

    def node(self, mesh, translation, yaw=0.0, extras=None, name=None):
        n = {"translation": [float(c) for c in translation], "name": name or "node"}
        if mesh is not None:
            n["mesh"] = mesh
        if yaw:
            n["rotation"] = [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]
        if extras:
            n["extras"] = extras
        self.doc["nodes"].append(n)
        self.doc["scenes"][0]["nodes"].append(len(self.doc["nodes"]) - 1)

    def write(self, path):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.doc["buffers"] = [{"byteLength": len(self.bin)}]
        if not self.doc["samplers"]:
            del self.doc["samplers"]
        js = json.dumps(self.doc, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(self.bin)))
            f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942) + bytes(self.bin))


def build(density, seed):
    rng = random.Random(seed)
    level = Level()
    keep_out = gts.keep_out_boxes()
    placed_discs = []
    counts = {}
    # Big things first, so they find room.
    for key in ("rocks", "lantern", "bust", "grass"):
        m = level.model(key)
        radius = max(math.hypot(x, z) for x in (m["lo"][0], m["hi"][0])
                     for z in (m["lo"][2], m["hi"][2]))
        # Grass clumps may overlap each other (it's ground cover); solid things
        # keep a gap.
        gap = 0.0 if key == "grass" else 0.5
        n = 0
        for _ in range(DENSITY[density][key]):
            for _ in range(80):
                x = rng.uniform(-HALF + radius + 0.5, HALF - radius - 0.5)
                z = rng.uniform(-HALF + radius + 0.5, HALF - radius - 0.5)
                yaw = rng.uniform(0, math.tau)
                if key != "grass" and any(math.hypot(x - dx, z - dz) < radius + dr + gap
                                          for dx, dz, dr in placed_discs):
                    continue
                y = GROUND_Y - m["lo"][1]
                rot = [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]
                aabb = gts.node_aabb((m["lo"], m["hi"]), (x, y, z), rot)
                if any(gts.overlaps(aabb, k) for k in keep_out):
                    continue
                extras = ({"prefab": "prop", "params": {"collide": False, "shadow": False}}
                          if key == "grass" else None)
                level.node(m["mesh"], (x, y, z), yaw, extras, name=key)
                if key != "grass":
                    placed_discs.append((x, z, radius))
                n += 1
                break
        counts[key] = n
    level.node(None, (0.0, GROUND_Y, 8.0), name="player_start",
               extras={"prefab": "player_start", "params": {"yaw": 270.0}})
    return level, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", default=os.path.join("scratch", "detail.glb"))
    ap.add_argument("--density", choices=sorted(DENSITY), default="med")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--check", action="store_true", help="validate after writing")
    args = ap.parse_args()
    missing = [k for k, rel in MODELS.items() if not os.path.exists(os.path.join(ASSETS, rel))]
    if missing:
        print(f"missing {', '.join(missing)}: run `python3 tools/fetch_assets.py` first",
              file=sys.stderr)
        return 1
    level, counts = build(args.density, args.seed)
    level.write(args.out)
    tris = 0
    for n in level.doc["nodes"]:
        if "mesh" in n:
            for p in level.doc["meshes"][n["mesh"]]["primitives"]:
                tris += level.doc["accessors"][p["indices"]]["count"] // 3
    prims = sum(len(level.doc["meshes"][n["mesh"]]["primitives"])
                for n in level.doc["nodes"] if "mesh" in n)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB): "
          + ", ".join(f"{k} {v}/{DENSITY[args.density][k]}" for k, v in counts.items())
          + f"\n  {tris:,} triangles placed, {prims} primitive instances, "
          f"{len(level.doc['images'])} images")
    if args.check and not gts.check(level.doc):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
