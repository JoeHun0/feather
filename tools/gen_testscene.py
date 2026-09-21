#!/usr/bin/env python3
"""Generate a populated glTF test level for the Feather engine.

Why this exists
---------------
The two workloads we had were both unfit for measurement. The orb demo is
pathological by design (1000 orbs, huge overdraw, every object a caster, no
occlusion), and the old `scratch/testscene.gltf` was a 5-node fixture written to
prove primitive dedup -- not a level. This produces a walkable scene laid out
around the systems we actually measure: shadows, AA, culling/instancing,
materials, and the character controller.

This is NOT the offline bake tool. ARCHITECTURE.md §17 reserves `bake/` for the
asset-blob pipeline; this is a dev content generator, hence `tools/`.

Output is a self-contained .gltf (data-URI buffer) written to scratch/, which is
gitignored -- so the generator is committed and the asset is not.

Engine constraints this respects (all verified against the source)
------------------------------------------------------------------
* Ground is a finite 80x80 slab whose top surface is GROUND_Y = -9.0
  (app/src/main.rs:67, :885), so everything must sit within x,z in [-40, 40].
* The player spawns at (0, -9, 8) (app/src/main.rs:665) and the static level --
  ground plus five obstacle boxes -- spawns *unconditionally*; only the orb demo
  is suppressed by passing a scene (app/src/main.rs:673). So we must not
  intersect those.
* The loader takes triangles only, requires POSITION, computes normals when
  absent and reads TEXCOORD_0 when present (assets/src/lib.rs:322, :352-361).
* MAX_TEXTURES = 64 (render/src/mesh.rs:22); past that, materials fall back to
  the default textures.
* NORMAL TRANSFORM: mesh.vert uses mat3(model), which is only correct for
  uniform scale, or for non-uniform scale with no rotation
  (render/shaders/mesh.vert:33). So: rotated nodes must be uniformly scaled, and
  non-uniformly scaled nodes must be unrotated. Meshes are therefore baked at
  their true size, letting rotated nodes keep scale 1. --check enforces this.
"""

import argparse
import base64
import json
import math
import os
import random
import struct
import sys
import zlib

# --- Engine constants, mirrored from app/src/main.rs ------------------------
GROUND_Y = -9.0
GROUND_HALF = 40.0
SPAWN = (0.0, -9.0, 8.0)
SHADOW_RADIUS = 16.0
AUTOSTEP = 0.4
MAX_TEXTURES = 64

# The five obstacle boxes the app always spawns (app/src/main.rs:898-904), as
# (center_x, center_y_offset_above_ground, center_z, size_x, size_y, size_z).
LEVEL_BOXES = [
    (-3.0, 0.75, 2.0, 1.5, 1.5, 1.5),
    (3.5, 1.0, -1.0, 2.0, 2.0, 2.0),
    (0.0, 0.5, -4.5, 3.0, 1.0, 1.0),
    (-5.0, 1.5, -3.0, 1.0, 3.0, 1.0),
    (5.0, 0.5, 4.0, 1.0, 1.0, 4.0),
]


# --- Geometry builders ------------------------------------------------------
# Each returns (positions, normals, uvs, indices) with flat per-face normals
# where it matters, in LOCAL space. Meshes are baked at true size so that any
# node needing rotation can keep scale 1 (see the normal-transform note above).

def box(sx, sy, sz):
    """Axis-aligned box centred on the origin, 24 verts / 36 indices."""
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    faces = [
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)]),
        ((1, 0, 0), [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)]),
        ((-1, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
        ((0, 1, 0), [(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)]),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
    ]
    pos, nrm, uv, idx = [], [], [], []
    for n, quad in faces:
        base = len(pos)
        for k, v in enumerate(quad):
            pos.append(v)
            nrm.append(n)
            uv.append([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)][k])
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return pos, nrm, uv, idx


def sphere(radius=0.5, stacks=16, slices=24):
    """UV sphere with smooth normals -- curved silhouettes alias differently
    from box edges, which is useful for the AA comparison."""
    pos, nrm, uv, idx = [], [], [], []
    for i in range(stacks + 1):
        v = i / stacks
        phi = v * math.pi
        for j in range(slices + 1):
            u = j / slices
            theta = u * 2.0 * math.pi
            n = (
                math.sin(phi) * math.cos(theta),
                math.cos(phi),
                math.sin(phi) * math.sin(theta),
            )
            pos.append((n[0] * radius, n[1] * radius, n[2] * radius))
            nrm.append(n)
            uv.append((u, v))
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            idx += [a, a + 1, b, a + 1, b + 1, b]
    return pos, nrm, uv, idx


def cylinder(radius, height, sides=12):
    """Capped cylinder centred on the origin, axis +Y. Thin instances of this
    are the hardest case for post-AA: a sub-pixel curved silhouette."""
    hy = height / 2.0
    pos, nrm, uv, idx = [], [], [], []
    for j in range(sides + 1):
        u = j / sides
        theta = u * 2.0 * math.pi
        cx, cz = math.cos(theta), math.sin(theta)
        for k, y in enumerate((-hy, hy)):
            pos.append((cx * radius, y, cz * radius))
            nrm.append((cx, 0.0, cz))
            uv.append((u, float(k)))
    for j in range(sides):
        a = j * 2
        idx += [a, a + 2, a + 1, a + 1, a + 2, a + 3]
    for y, ny, wind in ((hy, 1.0, True), (-hy, -1.0, False)):
        centre = len(pos)
        pos.append((0.0, y, 0.0))
        nrm.append((0.0, ny, 0.0))
        uv.append((0.5, 0.5))
        ring = len(pos)
        for j in range(sides + 1):
            theta = (j / sides) * 2.0 * math.pi
            cx, cz = math.cos(theta), math.sin(theta)
            pos.append((cx * radius, y, cz * radius))
            nrm.append((0.0, ny, 0.0))
            uv.append((cx * 0.5 + 0.5, cz * 0.5 + 0.5))
        for j in range(sides):
            a, b = ring + j, ring + j + 1
            idx += [centre, a, b] if wind else [centre, b, a]
    return pos, nrm, uv, idx


def wedge(run, rise, width):
    """Right triangular prism: a ramp rising along +X over `run`, `width` deep.

    Baked as real geometry rather than a rotated box so the slope normal is
    correct under mat3(model) and the collider matches what you see.
    """
    hw = width / 2.0
    # Profile in XY, extruded along Z.
    slope_n = (-rise, run, 0.0)
    ln = math.hypot(rise, run)
    slope_n = (slope_n[0] / ln, slope_n[1] / ln, 0.0)
    faces = [
        ((0, -1, 0), [(0, 0, hw), (run, 0, hw), (run, 0, -hw), (0, 0, -hw)]),
        ((1, 0, 0), [(run, 0, -hw), (run, rise, -hw), (run, rise, hw), (run, 0, hw)]),
        (slope_n, [(0, 0, -hw), (run, rise, -hw), (run, rise, hw), (0, 0, hw)]),
    ]
    pos, nrm, uv, idx = [], [], [], []
    for n, quad in faces:
        base = len(pos)
        for k, v in enumerate(quad):
            pos.append(tuple(float(c) for c in v))
            nrm.append(tuple(float(c) for c in n))
            uv.append([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)][k])
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    for n, tri, flip in (
        ((0, 0, 1), [(0, 0, hw), (run, rise, hw), (run, 0, hw)], False),
        ((0, 0, -1), [(0, 0, -hw), (run, 0, -hw), (run, rise, -hw)], False),
    ):
        base = len(pos)
        for k, v in enumerate(tri):
            pos.append(tuple(float(c) for c in v))
            nrm.append(tuple(float(c) for c in n))
            uv.append([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0)][k])
        idx += [base, base + 1, base + 2]
    return pos, nrm, uv, idx


# --- Minimal PNG writer (only needed for --textures) ------------------------

def png_rgba(width, height, pixels):
    """Encode RGBA8 rows as a PNG. Avoids a Pillow dependency -- this tool is
    deliberately stdlib-only so it runs anywhere the repo does."""
    raw = b"".join(b"\x00" + bytes(row) for row in pixels)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def checker_tex(size=64, a=(200, 200, 200, 255), b=(60, 60, 70, 255), cells=8):
    step = max(1, size // cells)
    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            row += list(a if ((x // step) + (y // step)) % 2 == 0 else b)
        rows.append(row)
    return png_rgba(size, size, rows)


def normal_tex(size=64, bumps=6):
    """Sine-bump normal map, encoded the usual (n*0.5+0.5) way."""
    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            u, v = x / size * math.tau * bumps, y / size * math.tau * bumps
            nx, nz = math.sin(u) * 0.35, math.sin(v) * 0.35
            ny = math.sqrt(max(0.0, 1.0 - nx * nx - nz * nz))
            row += [
                int((nx * 0.5 + 0.5) * 255),
                int((nz * 0.5 + 0.5) * 255),
                int((ny * 0.5 + 0.5) * 255),
                255,
            ]
        rows.append(row)
    return png_rgba(size, size, rows)


def mr_tex(size=64, cells=4):
    """glTF packs occlusion/roughness/metallic into R/G/B."""
    step = max(1, size // cells)
    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            rough = int(40 + 200 * ((x // step) / max(1, cells - 1)))
            metal = 255 if ((y // step) % 2 == 0) else 0
            row += [255, min(255, rough), metal, 255]
        rows.append(row)
    return png_rgba(size, size, rows)


# --- glTF assembly ----------------------------------------------------------

class Gltf:
    def __init__(self):
        self.blob = bytearray()
        self.views = []
        self.accessors = []
        self.meshes = []
        self.materials = []
        self.nodes = []
        self.images = []
        self.samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
        self.textures = []
        self.mesh_bounds = {}  # mesh index -> (min xyz, max xyz) over all prims

    def _view(self, data, target=None):
        while len(self.blob) % 4:
            self.blob.append(0)
        offset = len(self.blob)
        self.blob += data
        v = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            v["target"] = target
        self.views.append(v)
        return len(self.views) - 1

    def _accessor(self, data, count, comp_type, type_, target, minmax=None):
        view = self._view(data, target)
        a = {"bufferView": view, "componentType": comp_type, "count": count, "type": type_}
        if minmax:
            a["min"], a["max"] = minmax
        self.accessors.append(a)
        return len(self.accessors) - 1

    def add_material(self, name, base, metallic, roughness, emissive=None, tex=None):
        m = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": list(base),
                "metallicFactor": metallic,
                "roughnessFactor": roughness,
            },
        }
        if emissive:
            m["emissiveFactor"] = list(emissive)
        if tex:
            base_t, normal_t, mr_t = tex
            if base_t is not None:
                m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": base_t}
            if mr_t is not None:
                m["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": mr_t}
            if normal_t is not None:
                m["normalTexture"] = {"index": normal_t}
        self.materials.append(m)
        return len(self.materials) - 1

    def add_texture(self, png_bytes):
        uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        self.images.append({"uri": uri})
        self.textures.append({"sampler": 0, "source": len(self.images) - 1})
        return len(self.textures) - 1

    def add_mesh(self, name, prims):
        """prims: list of (geometry tuple, material index). Several primitives on
        one mesh exercises the loader's (mesh, primitive) dedup key."""
        out = []
        lo = [1e30] * 3
        hi = [-1e30] * 3
        for (pos, nrm, uv, idx), mat in prims:
            pmin = [min(p[i] for p in pos) for i in range(3)]
            pmax = [max(p[i] for p in pos) for i in range(3)]
            lo = [min(lo[i], pmin[i]) for i in range(3)]
            hi = [max(hi[i], pmax[i]) for i in range(3)]
            p_acc = self._accessor(
                struct.pack("<%df" % (len(pos) * 3), *[c for v in pos for c in v]),
                len(pos), 5126, "VEC3", 34962, (pmin, pmax),
            )
            n_acc = self._accessor(
                struct.pack("<%df" % (len(nrm) * 3), *[c for v in nrm for c in v]),
                len(nrm), 5126, "VEC3", 34962,
            )
            t_acc = self._accessor(
                struct.pack("<%df" % (len(uv) * 2), *[c for v in uv for c in v]),
                len(uv), 5126, "VEC2", 34962,
            )
            i_acc = self._accessor(
                struct.pack("<%dI" % len(idx), *idx), len(idx), 5125, "SCALAR", 34963,
            )
            out.append({
                "attributes": {"POSITION": p_acc, "NORMAL": n_acc, "TEXCOORD_0": t_acc},
                "indices": i_acc,
                "material": mat,
                "mode": 4,
            })
        self.meshes.append({"name": name, "primitives": out})
        i = len(self.meshes) - 1
        self.mesh_bounds[i] = (lo, hi)
        return i

    def add_node(self, mesh, translation, rotation=None, scale=None, extras=None, name=None):
        n = {"mesh": mesh, "translation": [float(c) for c in translation]}
        if rotation:
            n["rotation"] = [float(c) for c in rotation]
        if scale:
            n["scale"] = [float(c) for c in scale]
        if extras:
            n["extras"] = extras
        if name:
            n["name"] = name
        self.nodes.append(n)
        return len(self.nodes) - 1

    def add_marker(self, translation, extras, name):
        """A mesh-less node. Today the loader skips it (it only emits SceneNodes
        for nodes with a mesh); once §18 lands it is a prefab spawn."""
        self.nodes.append({
            "name": name,
            "translation": [float(c) for c in translation],
            "extras": extras,
        })
        return len(self.nodes) - 1

    def document(self):
        uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(self.blob)).decode("ascii")
        doc = {
            "asset": {"version": "2.0", "generator": "feather tools/gen_testscene.py"},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(self.nodes)))}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.views,
            "buffers": [{"byteLength": len(self.blob), "uri": uri}],
        }
        if self.textures:
            doc["images"] = self.images
            doc["samplers"] = self.samplers
            doc["textures"] = self.textures
        return doc


def yaw(degrees):
    """Quaternion (xyzw) about +Y. Only ever paired with uniform scale."""
    h = math.radians(degrees) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def axis_z(degrees):
    """Quaternion (xyzw) about +Z, for the leaning lattice beams."""
    h = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(h), math.cos(h))


# --- Placement helpers ------------------------------------------------------

def quat_matrix(q):
    x, y, z, w = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def node_aabb(bounds, translation, rotation=None, scale=None):
    """World AABB of a placed mesh, by transforming the eight local corners."""
    lo, hi = bounds
    s = scale or (1.0, 1.0, 1.0)
    m = quat_matrix(rotation) if rotation else ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    out_lo = [1e30] * 3
    out_hi = [-1e30] * 3
    for cx in (lo[0], hi[0]):
        for cy in (lo[1], hi[1]):
            for cz in (lo[2], hi[2]):
                v = (cx * s[0], cy * s[1], cz * s[2])
                for i in range(3):
                    w = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] + translation[i]
                    out_lo[i] = min(out_lo[i], w)
                    out_hi[i] = max(out_hi[i], w)
    return out_lo, out_hi


def overlaps(a, b, margin=0.0):
    return all(a[0][i] - margin < b[1][i] and b[0][i] - margin < a[1][i] for i in range(3))


def keep_out_boxes():
    """What the app already puts in the world, which we must not intersect:
    the five obstacle boxes plus a clear cylinder around the spawn point."""
    out = []
    for cx, cy, cz, sx, sy, sz in LEVEL_BOXES:
        c = (cx, GROUND_Y + cy, cz)
        out.append((
            [c[0] - sx / 2, c[1] - sy / 2, c[2] - sz / 2],
            [c[0] + sx / 2, c[1] + sy / 2, c[2] + sz / 2],
        ))
    out.append((
        [SPAWN[0] - 2.5, GROUND_Y - 0.5, SPAWN[2] - 2.5],
        [SPAWN[0] + 2.5, GROUND_Y + 3.0, SPAWN[2] + 2.5],
    ))
    return out


class Scene:
    """Wraps the glTF builder with the placement bookkeeping the zones need."""

    def __init__(self, g, rng):
        self.g = g
        self.rng = rng
        self.placed = list(keep_out_boxes())

    def place(self, mesh, translation, rotation=None, scale=None, extras=None, name=None):
        box_ = node_aabb(self.g.mesh_bounds[mesh], translation, rotation, scale)
        self.placed.append(box_)
        return self.g.add_node(mesh, translation, rotation, scale, extras, name)

    def free(self, mesh, translation, rotation=None, scale=None, margin=0.3):
        box_ = node_aabb(self.g.mesh_bounds[mesh], translation, rotation, scale)
        return not any(overlaps(box_, p, margin) for p in self.placed)


# --- Zones ------------------------------------------------------------------
# Laid out in cardinal sectors so each can be framed on its own from spawn.

def zone_shadow(s, pal, extras_on):
    """North (-Z). Pillars spanning the shadow box boundary.

    The player spawns at z=8 and SHADOW_RADIUS is 16, so shadows stop at z~=-8.
    These rows run from z=-5 (inside) to z=-36 (far outside), which makes the
    cut-off edge directly visible -- and is the concrete argument for CSM. Thin
    pillars are included on purpose: they are what shows shadow-map texel
    density, i.e. the blockiness visible on the ground today.
    """
    rows = [-6.5, -9.5, -14.0, -18.5, -23.0, -28.0, -33.0, -36.5]
    widths = [0.15, 0.3, 0.55, 1.1]
    for ri, z in enumerate(rows):
        for ci, x in enumerate(range(-17, 5, 3)):
            w = widths[(ri + ci) % len(widths)]
            h = 3.0 + ((ri * 7 + ci * 3) % 5) * 1.5
            extras = None
            if extras_on and ci % 5 == 0:
                # Thin pillars read poorly in the shadow map and cost caster
                # fill; a few are marked as non-casters to show the switch.
                extras = {"prefab": "prop", "params": {"shadow": False}}
            t = (float(x), GROUND_Y + h / 2.0, z)
            sc = (w, h, w)
            # Skip rather than intersect: the app's own obstacle boxes sit near
            # the origin, and this zone's first row passes close to them.
            if not s.free(pal["box"], t, None, sc, margin=0.0):
                continue
            s.place(pal["box"], t, scale=sc, extras=extras, name=f"pillar_{ri}_{ci}")
    # A line of lights down the pillar rows. They overlap deliberately: the
    # brute-force loop costs lights-in-frame, so overlap is what clustering has
    # to beat later.
    if extras_on:
        for i, z in enumerate(rows[::2]):
            s.g.add_marker(
                (-19.0, GROUND_Y + 3.0, z),
                {
                    "prefab": "point_light",
                    "params": {
                        "color": [1.0, 0.85, 0.6],
                        "intensity": 30.0,
                        "radius": 12.0,
                        # Emitter size for the sphere-light specular (§12);
                        # these are geometry-free, so "a small lamp".
                        "source_radius": 0.15,
                    },
                },
                f"lamp_{i}",
            )

    # One large blocker: a big soft-edged shadow to contrast with the thin ones.
    s.place(pal["box"], (-6.0, GROUND_Y + 5.0, -20.0), scale=(7.0, 10.0, 1.5), name="blocker")


def zone_traversal(s, pal):
    """East (+X). Geometry that exercises the controller.

    §26 says outright that "slope climbing/sliding uses rapier's 45° defaults
    but no level geometry exercises it". These flights and ramps bracket both
    limits: risers either side of the 0.4 autostep, slopes either side of 45°.
    """
    # Stair flights. Risers bracket AUTOSTEP = 0.4: 0.25/0.35 should walk up,
    # 0.45/0.6 should not (0.45 is the interesting near-miss).
    for fi, riser in enumerate((0.25, 0.35, 0.45, 0.6)):
        z = -9.0 + fi * 5.0
        for step in range(6):
            h = riser * (step + 1)
            s.place(
                pal["box"], (9.0 + step * 0.7, GROUND_Y + h / 2.0, z),
                scale=(0.7, h, 3.0), name=f"stair_{riser}_{step}",
            )
    # Ramps bracketing rapier's 45° slope limit. Baked geometry (see `wedge`),
    # placed unrotated so the slope normal stays exact.
    for ai, angle in enumerate((20, 40, 44, 50)):
        z = -9.0 + ai * 5.0
        s.place(pal[f"ramp{angle}"], (17.0, GROUND_Y, z), name=f"ramp_{angle}")
    # Jump gaps. MOVE_SPEED is 8.0 and the jump apex is ~1.5 units, so the
    # 2.0/3.0 gaps should clear and 4.5/6.0 should not.
    for gi, gap in enumerate((2.0, 3.0, 4.5, 6.0)):
        z = -9.0 + gi * 5.0
        s.place(pal["box"], (25.0, GROUND_Y + 0.5, z), scale=(4.0, 1.0, 3.0), name=f"ledge_a{gi}")
        s.place(
            pal["box"], (25.0 + 4.0 + gap, GROUND_Y + 0.5, z),
            scale=(4.0, 1.0, 3.0), name=f"ledge_b{gi}",
        )
    # Low tunnel: the capsule is 1.8 tall, so 2.2 clears and 1.5 blocks.
    for ti, clear in enumerate((2.2, 1.5)):
        z = 7.0 + ti * 4.0
        s.place(pal["box"], (34.0, GROUND_Y + 1.5, z - 1.6), scale=(6.0, 3.0, 0.4), name="tun_w0")
        s.place(pal["box"], (34.0, GROUND_Y + 1.5, z + 1.6), scale=(6.0, 3.0, 0.4), name="tun_w1")
        s.place(
            pal["box"], (34.0, GROUND_Y + clear + 0.25, z),
            scale=(6.0, 0.5, 3.6), name=f"tun_roof_{clear}",
        )


def zone_aa(s, pal):
    """South (+Z). Thin high-contrast geometry at graded distances.

    Spawn is at z=8, so these rows sit ~4, ~14 and ~24 units out. Within each
    row the spacing tightens, so one screenshot contains edges from
    comfortably-resolved down to sub-pixel -- which is exactly the near/far
    split that separates MSAA from FXAA (FXAA's SPAN_MAX is 8 pixels, so it
    only helps once the staircase risers are small).
    """
    for row, z in enumerate((12.0, 22.0, 32.0)):
        x = -15.0
        for spacing in (1.0, 0.8, 0.6, 0.45, 0.35, 0.28):
            for _ in range(6):
                s.place(pal["pole"], (x, GROUND_Y + 3.0, z), name=f"pole_{row}")
                x += spacing
            x += 1.2
    # Diagonal lattice: rotated, so it keeps scale 1 and the mesh is baked at
    # true size -- mat3(model) stays correct (render/shaders/mesh.vert:33).
    for i in range(10):
        x = -12.0 + i * 2.6
        for ang in (40, -40):
            s.place(
                pal["beam"], (x, GROUND_Y + 2.0, 17.0),
                rotation=axis_z(ang), name="lattice",
            )
    # Picket fence: near-field sub-pixel edges when viewed down its length.
    for i in range(40):
        s.place(
            pal["box"], (-10.0 + i * 0.5, GROUND_Y + 0.9, 27.0),
            scale=(0.08, 1.8, 0.08), name="picket",
        )


def zone_pbr(s, pal, extras_on):
    """North-east. Metallic x roughness sphere grid, a stable reference for
    IBL / tonemap / exposure changes.

    Each cell needs its own material, and the app derives one material per
    loaded mesh -- so these are distinct meshes, not instances of one.
    """
    for mi in range(6):
        for ri in range(5):
            mesh = pal["pbr"][mi * 5 + ri]
            s.place(
                mesh,
                (12.0 + mi * 2.6, GROUND_Y + 1.6, -14.0 - ri * 2.6),
                scale=(1.4, 1.4, 1.4), name=f"pbr_{mi}_{ri}",
            )
    for ei, mesh in enumerate(pal["emissive"]):
        s.place(
            mesh, (12.0 + ei * 5.0, GROUND_Y + 1.6, -27.0),
            scale=(1.4, 1.4, 1.4),
            # A lamp: geometry that also emits (§12). The emissive material
            # makes the sphere itself glow; the prefab makes it light what is
            # around it.
            extras={
                "prefab": "point_light",
                "params": {
                    "color": [1.0, 0.6, 0.2] if ei == 0 else [0.4, 0.7, 1.0],
                    "intensity": 40.0,
                    "radius": 14.0,
                    # The orb's own radius (0.5 sphere x 1.4), so a mirror
                    # sphere's highlight is the size of the orb's reflection.
                    "source_radius": 0.7,
                },
            }
            if extras_on
            else None,
            name=f"emissive_{ei}",
        )


def scatter_lights(s, count, radius=14.0):
    """Extra punctual lights spread over the walkable area (§12).

    Marker nodes with no geometry, so they add lighting cost and nothing else —
    which is what makes them a clean independent variable when measuring the
    light loop. `radius` sets how much they overlap: the default 14 puts ~8.5
    lights on every ground pixel, clustering's worst case, while ~4 leaves about
    one — where clustering pays (ARCHITECTURE.md §12 has both measured).
    """
    for i in range(count):
        x = s.rng.uniform(-36.0, 36.0)
        z = s.rng.uniform(-34.0, 34.0)
        h = s.rng.uniform(1.5, 5.0)
        # Warm-to-cool spread so it is obvious which light lit what.
        t = i / max(1, count - 1)
        s.g.add_marker(
            (x, GROUND_Y + h, z),
            {
                "prefab": "point_light",
                "params": {
                    "color": [1.0, 0.55 + 0.35 * t, 0.3 + 0.6 * t],
                    "intensity": 18.0,
                    "radius": radius,
                    "source_radius": 0.15,
                },
            },
            f"scatter_light_{i}",
        )


def zone_field(s, pal, count):
    """West (-X). Many nodes over few meshes, spread to the ground edge.

    This is the realistic draw workload: it exercises primitive dedup, the
    per-mesh bounding-sphere frustum cull and the sorted per-mesh draw runs,
    without the orb demo's pathological overdraw.

    Tagged `prop` with `collide: false` (§18): this is scenery at the map edge,
    and a trimesh collider per node is the single biggest load-time cost in the
    scene. The traversal and shadow zones keep their collision, since walking on
    them is the entire point of those.
    """
    kinds = ["box", "sphere", "pole"]
    placed = 0
    attempts = 0
    while placed < count and attempts < count * 40:
        attempts += 1
        kind = s.rng.choice(kinds)
        mesh = pal[kind]
        x = s.rng.uniform(-38.0, -20.0)
        z = s.rng.uniform(-34.0, 34.0)
        if kind == "box":
            # Non-uniform scale is fine here only because there is no rotation.
            sc = (s.rng.uniform(0.5, 2.5), s.rng.uniform(0.6, 4.0), s.rng.uniform(0.5, 2.5))
            t = (x, GROUND_Y + sc[1] / 2.0, z)
            rot = None
        else:
            # Rotated, so the scale must be uniform.
            u = s.rng.uniform(0.6, 2.0)
            sc = (u, u, u)
            t = (x, GROUND_Y + (u * 0.5 if kind == "sphere" else u * 3.0), z)
            rot = yaw(s.rng.uniform(0, 360))
        if not s.free(mesh, t, rot, sc):
            continue
        s.place(
            mesh, t, rot, sc,
            extras={"prefab": "prop", "params": {"collide": False}},
            name=f"field_{placed}",
        )
        placed += 1
    return placed


# --- Palette ----------------------------------------------------------------

def build_palette(g, use_textures):
    """Mesh + material palette.

    Deliberately few meshes for most of the scene, so hundreds of nodes collapse
    onto a handful of draw runs -- that is the dedup/instancing path under test.
    The PBR grid is the exception: the app derives one material per *mesh*, so a
    grid of distinct materials must be distinct meshes.
    """
    # Several *distinct* base-color textures, not one shared set: the point is
    # to populate multiple slots of the bindless textures[] array so the
    # non-uniform indexing path is actually exercised. Still far under
    # MAX_TEXTURES (render/src/mesh.rs:22).
    tex_box = tex_sphere = tex_metal = tex_ramp = None
    if use_textures:
        normal_t = g.add_texture(normal_tex())
        mr_t = g.add_texture(mr_tex())
        tex_box = (g.add_texture(checker_tex(cells=8)), normal_t, mr_t)
        tex_sphere = (
            g.add_texture(checker_tex(cells=4, a=(210, 180, 140, 255), b=(90, 70, 50, 255))),
            normal_t,
            None,
        )
        tex_metal = (
            g.add_texture(checker_tex(cells=16, a=(170, 175, 185, 255), b=(70, 74, 82, 255))),
            None,
            mr_t,
        )
        tex_ramp = (
            g.add_texture(checker_tex(cells=2, a=(120, 130, 140, 255), b=(55, 60, 66, 255))),
            normal_t,
            None,
        )

    m_box = g.add_material("concrete", (0.62, 0.60, 0.57, 1.0), 0.0, 0.85, tex=tex_box)
    m_sphere = g.add_material("plaster", (0.75, 0.72, 0.68, 1.0), 0.0, 0.6, tex=tex_sphere)
    m_metal = g.add_material("steel", (0.78, 0.79, 0.82, 1.0), 1.0, 0.28, tex=tex_metal)
    m_ramp = g.add_material("ramp", (0.45, 0.48, 0.52, 1.0), 0.0, 0.7, tex=tex_ramp)

    pal = {
        "box": g.add_mesh("box", [(box(1.0, 1.0, 1.0), m_box)]),
        "sphere": g.add_mesh("sphere", [(sphere(0.5), m_sphere)]),
        "pole": g.add_mesh("pole", [(cylinder(0.06, 6.0), m_metal)]),
        "beam": g.add_mesh("beam", [(box(0.12, 5.0, 0.12), m_metal)]),
    }
    for angle in (20, 40, 44, 50):
        run = 4.0
        rise = run * math.tan(math.radians(angle))
        pal[f"ramp{angle}"] = g.add_mesh(f"ramp{angle}", [(wedge(run, rise, 3.0), m_ramp)])

    # Metallic x roughness grid: 6 x 5 distinct meshes, each with its own
    # material. Geometry is duplicated per cell rather than sharing accessors --
    # simpler, and a few hundred KB in a gitignored scratch file is not worth
    # the complexity of accessor reuse.
    pal["pbr"] = []
    geom = sphere(0.5)
    for mi in range(6):
        for ri in range(5):
            metallic = mi / 5.0
            roughness = 0.06 + (ri / 4.0) * 0.9
            mat = g.add_material(
                f"pbr_m{mi}_r{ri}", (0.82, 0.80, 0.76, 1.0), metallic, roughness
            )
            pal["pbr"].append(g.add_mesh(f"pbr_m{mi}_r{ri}", [(geom, mat)]))

    pal["emissive"] = []
    for i, e in enumerate(((3.0, 1.6, 0.5), (0.4, 1.2, 3.0))):
        mat = g.add_material(f"emissive_{i}", (0.05, 0.05, 0.05, 1.0), 0.0, 0.5, emissive=e)
        pal["emissive"].append(g.add_mesh(f"emissive_{i}", [(geom, mat)]))
    return pal


DENSITY = {"low": 120, "med": 320, "high": 800}


# --- Validation -------------------------------------------------------------

def check(doc, strict_dedup=True):
    """Re-parse the emitted file and assert the invariants that matter.

    This is a permanent feature, not a scratch harness: regenerating with a new
    seed or density must not silently produce a scene that floats, intersects
    the app's own level geometry, or shades wrongly.
    """
    errors = []
    n_acc, n_mat = len(doc["accessors"]), len(doc["materials"])
    bounds = {}
    tris = 0

    for mi, mesh in enumerate(doc["meshes"]):
        lo, hi = [1e30] * 3, [-1e30] * 3
        for pi, prim in enumerate(mesh["primitives"]):
            where = f"mesh {mi} ({mesh.get('name')}) prim {pi}"
            if prim.get("mode", 4) != 4:
                errors.append(f"{where}: mode {prim.get('mode')} is not TRIANGLES; "
                              "the loader skips it (assets/src/lib.rs:322)")
            pos = prim["attributes"].get("POSITION")
            if pos is None:
                errors.append(f"{where}: no POSITION")
                continue
            if pos >= n_acc:
                errors.append(f"{where}: POSITION accessor {pos} out of range")
                continue
            acc = doc["accessors"][pos]
            if "min" not in acc or "max" not in acc:
                errors.append(f"{where}: POSITION accessor lacks min/max "
                              "(required by spec; the per-mesh bounding sphere "
                              "used for culling depends on it)")
            else:
                lo = [min(lo[i], acc["min"][i]) for i in range(3)]
                hi = [max(hi[i], acc["max"][i]) for i in range(3)]
            if prim.get("material", 0) >= n_mat:
                errors.append(f"{where}: material {prim.get('material')} out of range")
            idx = prim.get("indices")
            if idx is not None and idx < n_acc:
                tris += doc["accessors"][idx]["count"] // 3
        bounds[mi] = (lo, hi)

    keep = keep_out_boxes()
    mesh_nodes = 0
    referenced = set()
    for ni, node in enumerate(doc["nodes"]):
        if "mesh" not in node:
            continue
        mesh_nodes += 1
        referenced.add(node["mesh"])
        name = node.get("name", f"node{ni}")
        rot = node.get("rotation")
        sc = node.get("scale")
        # mesh.vert uses mat3(model): correct only for uniform scale, or for
        # non-uniform scale with no rotation (render/shaders/mesh.vert:33).
        if rot and sc and not (abs(sc[0] - sc[1]) < 1e-6 and abs(sc[1] - sc[2]) < 1e-6):
            errors.append(f"node {ni} ({name}): rotated AND non-uniformly scaled "
                          f"{sc} -- mat3(model) would skew its normals")
        aabb = node_aabb(bounds[node["mesh"]], node["translation"], rot, sc)
        for axis in (0, 2):
            if aabb[0][axis] < -GROUND_HALF or aabb[1][axis] > GROUND_HALF:
                errors.append(f"node {ni} ({name}): extends past the {GROUND_HALF*2:.0f}"
                              f"x{GROUND_HALF*2:.0f} ground on axis {'xyz'[axis]}")
        if aabb[0][1] < GROUND_Y - 0.01:
            errors.append(f"node {ni} ({name}): sinks below GROUND_Y "
                          f"({aabb[0][1]:.2f} < {GROUND_Y})")
        for k in keep:
            if overlaps(aabb, k):
                errors.append(f"node {ni} ({name}): intersects the app's own level "
                              "geometry or the spawn clearance")
                break

    n_tex = len(doc.get("textures", []))
    if n_tex >= MAX_TEXTURES:
        errors.append(f"{n_tex} textures >= MAX_TEXTURES ({MAX_TEXTURES}); "
                      "the surplus would fall back to defaults (render/src/mesh.rs:22)")
    ratio = mesh_nodes / max(1, len(referenced))
    if strict_dedup and ratio <= 1.0:
        errors.append(f"{mesh_nodes} mesh-bearing nodes over {len(referenced)} referenced "
                      "meshes -- dedup/instancing is not being exercised")

    print(f"  meshes      {len(doc['meshes'])}")
    print(f"  materials   {n_mat}")
    print(f"  nodes       {len(doc['nodes'])} ({mesh_nodes} with a mesh, "
          f"{len(doc['nodes']) - mesh_nodes} markers)")
    print(f"  instancing  {ratio:.1f} nodes per referenced mesh "
          f"({len(referenced)}/{len(doc['meshes'])} meshes used)")
    print(f"  triangles   {tris}")
    print(f"  textures    {n_tex}")
    if errors:
        print(f"\n{len(errors)} problem(s):", file=sys.stderr)
        for e in errors[:25]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 25:
            print(f"  ... and {len(errors) - 25} more", file=sys.stderr)
        return False
    print("  checks      all passed")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", default="scratch/testscene.gltf")
    ap.add_argument("--zones", default="all",
                    help="all, or a comma list of: shadow,traversal,aa,field,pbr")
    ap.add_argument("--density", choices=sorted(DENSITY), default="med",
                    help="scales the culling/instancing field only")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lights", type=int, default=0, metavar="N",
                    help="extra scattered point lights, for measuring the §12 "
                         "clustered light loop (engine caps at 128 visible)")
    ap.add_argument("--light-radius", type=float, default=14.0, metavar="R",
                    help="radius of the --lights scatter (default 14: heavy "
                         "overlap; ~4: about one light per pixel)")
    ap.add_argument("--textures", action="store_true",
                    help="procedural base-color/normal/MR textures (off by default "
                         "to stay well clear of MAX_TEXTURES)")
    ap.add_argument("--no-extras", action="store_true",
                    help="omit the §18 prefab extras (player_start, props, "
                         "point lights); the engine then spawns at its default")
    ap.add_argument("--check", action="store_true", help="validate after writing")
    args = ap.parse_args()

    ALL_ZONES = {"shadow", "traversal", "aa", "field", "pbr"}
    want = ALL_ZONES if args.zones == "all" else set(args.zones.split(","))
    unknown = want - ALL_ZONES
    if unknown:
        ap.error(f"unknown zone(s): {', '.join(sorted(unknown))}")

    g = Gltf()
    s = Scene(g, random.Random(args.seed))
    pal = build_palette(g, args.textures)
    extras_on = not args.no_extras

    if "shadow" in want:
        zone_shadow(s, pal, extras_on)
    if "traversal" in want:
        zone_traversal(s, pal)
    if "aa" in want:
        zone_aa(s, pal)
    if "pbr" in want:
        zone_pbr(s, pal, extras_on)
    if "field" in want:
        n = zone_field(s, pal, DENSITY[args.density])
        if n < DENSITY[args.density]:
            print(f"note: field placed {n}/{DENSITY[args.density]} "
                  "(ran out of free space)", file=sys.stderr)
    if args.light_radius <= 0.0:
        ap.error("--light-radius must be > 0 (a zero-radius light lights nothing)")
    if args.lights > 0:
        if not extras_on:
            ap.error("--lights needs extras; drop --no-extras")
        scatter_lights(s, args.lights, args.light_radius)
    if extras_on:
        g.add_marker(SPAWN, {"prefab": "player_start", "params": {"yaw": 180.0}}, "player_start")

    doc = g.document()
    with open(args.out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KiB)")

    if args.check:
        if not check(doc, strict_dedup=(want == ALL_ZONES)):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
