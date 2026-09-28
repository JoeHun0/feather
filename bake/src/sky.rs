//! Sky-visibility bake (§13): how much of the sky each point of a level sees,
//! on a grid of cells, so the renderer can occlude the ambient light it
//! otherwise adds everywhere at full strength.
//!
//! Each cell centre casts `RAYS` rays over the upper hemisphere against the
//! level's occluders (a BVH over their world-space triangles) and stores the
//! two moments of the visibility that [`SkyVis`] describes, plus how far it
//! sees along each axis ([`SkyFree`]). The runtime blends the eight cells
//! round a point one cell off the surface, along its normal, leaving out the
//! cells it can't see: without that, cells above a 0.1 m roof lit the top
//! 30 cm of the zone hangar's walls at up to half the open sky's light.
//!
//! Cells whose centre is inside geometry would read as fully occluded and
//! darken whatever samples near them, so a short full-sphere probe finds
//! them (a quarter of its rays or more hit back faces), and they take the
//! **darkest** valid neighbour's value. The darkest, not the mean: a cell
//! inside a wall between a room and the open would otherwise carry the open
//! sky into the room.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use feather_assets::bake::{sky_occludes, BakedMesh, SkyFree, SkyVis, SkyVolume};
use feather_assets::{SceneData, Vertex};
use glam::{Mat3, Mat4, Vec3};

/// Edge of a cell, in metres: half the smallest rooms the zone has (3.2 m
/// storeys, 8 m offices) would still be 3 cells.
const CELL: f32 = 0.5;
/// Past this many cells, the cells grow instead: the bake's time and the
/// volume's memory are both linear in it.
const MAX_CELLS: f64 = 4.0e6;
/// Empty cells round the occluders' bounds, so that a surface's sample, a
/// cell off it, stays inside the grid.
const PAD: f32 = 2.0;
/// Visibility rays per cell, over the upper hemisphere.
const RAYS: usize = 128;
/// Rays of the inside-geometry probe, over the whole sphere...
const PROBE_RAYS: usize = 16;
/// ...and how many of them must hit a back face to call the cell inside.
const PROBE_INSIDE: usize = PROBE_RAYS / 4;
/// How many rings of neighbours an inside cell may borrow from.
const FILL_PASSES: usize = 3;
/// Coarsest baked LOD a mesh may be traced at, as world-space error. Far
/// below a cell, and it takes the zone's 825k placed triangles down a lot.
const LOD_TOLERANCE: f32 = 0.05;
/// Rays start this far along, so a cell centre lying exactly on a surface
/// doesn't hit it at t = 0.
const T_MIN: f32 = 1e-4;

/// One world-space triangle, ready for Möller–Trumbore.
#[derive(Clone, Copy, Debug)]
pub struct Tri {
    v0: Vec3,
    e1: Vec3,
    e2: Vec3,
    /// Which way it faces, from its vertex normals (the renderer draws with
    /// culling off, so winding says nothing), or zero when both sides are a
    /// front: a hit from the side it faces away from is a back face.
    facing: Vec3,
}

impl Tri {
    pub fn new(p: [Vec3; 3], facing: Vec3) -> Self {
        Self {
            v0: p[0],
            e1: p[1] - p[0],
            e2: p[2] - p[0],
            facing,
        }
    }

    fn centroid(&self) -> Vec3 {
        self.v0 + (self.e1 + self.e2) / 3.0
    }

    fn bounds(&self) -> (Vec3, Vec3) {
        let (a, b, c) = (self.v0, self.v0 + self.e1, self.v0 + self.e2);
        (a.min(b).min(c), a.max(b).max(c))
    }

    /// Distance along the ray to the hit, if within (T_MIN, t_max).
    fn hit(&self, o: Vec3, d: Vec3, t_max: f32) -> Option<f32> {
        let p = d.cross(self.e2);
        let det = self.e1.dot(p);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        let s = o - self.v0;
        let u = s.dot(p) * inv;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = s.cross(self.e1);
        let v = d.dot(q) * inv;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = self.e2.dot(q) * inv;
        (t > T_MIN && t < t_max).then_some(t)
    }
}

/// A bounding volume hierarchy over triangles: binned-SAH build, flat nodes.
pub struct Bvh {
    nodes: Vec<Node>,
    tris: Vec<Tri>,
}

#[derive(Clone, Copy)]
struct Node {
    min: Vec3,
    max: Vec3,
    /// Leaf: first triangle. Interior: the left child (the right is next).
    start: u32,
    /// Triangles in a leaf; 0 for an interior node.
    count: u32,
}

const LEAF_TRIS: usize = 4;
const BINS: usize = 16;

impl Bvh {
    pub fn build(mut tris: Vec<Tri>) -> Self {
        let mut nodes = Vec::with_capacity(2 * tris.len().div_ceil(LEAF_TRIS).max(1));
        let bounds = |t: &[Tri]| {
            t.iter().fold(
                (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                |(lo, hi), t| {
                    let (a, b) = t.bounds();
                    (lo.min(a), hi.max(b))
                },
            )
        };
        let (min, max) = bounds(&tris);
        nodes.push(Node {
            min,
            max,
            start: 0,
            count: tris.len() as u32,
        });
        let mut todo = vec![0usize];
        while let Some(n) = todo.pop() {
            let (start, count) = (nodes[n].start as usize, nodes[n].count as usize);
            if count <= LEAF_TRIS {
                continue;
            }
            let Some(mid) = split(&mut tris[start..start + count]) else {
                continue;
            };
            let left = nodes.len();
            for (s, c) in [(start, mid), (start + mid, count - mid)] {
                let (min, max) = bounds(&tris[s..s + c]);
                nodes.push(Node {
                    min,
                    max,
                    start: s as u32,
                    count: c as u32,
                });
            }
            nodes[n].start = left as u32;
            nodes[n].count = 0;
            todo.extend([left, left + 1]);
        }
        Self { nodes, tris }
    }

    /// Whether anything lies along the ray before `t_max` (any hit).
    pub fn occluded(&self, o: Vec3, d: Vec3, t_max: f32) -> bool {
        let mut hit = false;
        self.walk(o, d, t_max, |tri, t_max| {
            hit = tri.hit(o, d, t_max).is_some();
            hit.then_some(0.0)
        });
        hit
    }

    /// The nearest hit before `t_max`: its distance and triangle.
    pub fn nearest(&self, o: Vec3, d: Vec3, t_max: f32) -> Option<(f32, &Tri)> {
        let mut best: Option<(f32, &Tri)> = None;
        self.walk(o, d, t_max, |tri, t_max| {
            let t = tri.hit(o, d, t_max)?;
            best = Some((t, tri));
            Some(t)
        });
        best
    }

    /// Visit the triangles whose leaves the ray reaches, nearest boxes first.
    /// `visit` returns a new, shorter `t_max` on a hit; 0 stops the walk.
    fn walk<'a>(
        &'a self,
        o: Vec3,
        d: Vec3,
        mut t_max: f32,
        mut visit: impl FnMut(&'a Tri, f32) -> Option<f32>,
    ) {
        if self.tris.is_empty() {
            return;
        }
        let inv = d.recip();
        let mut stack = [0u32; 64];
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let node = &self.nodes[stack[top] as usize];
            if slab(node, o, inv, t_max).is_none() {
                continue;
            }
            if node.count > 0 {
                let s = node.start as usize;
                for tri in &self.tris[s..s + node.count as usize] {
                    if let Some(t) = visit(tri, t_max) {
                        if t <= 0.0 {
                            return;
                        }
                        t_max = t;
                    }
                }
                continue;
            }
            // Push the farther child first, so the nearer one pops next.
            let (a, b) = (node.start, node.start + 1);
            let ta = slab(&self.nodes[a as usize], o, inv, t_max);
            let tb = slab(&self.nodes[b as usize], o, inv, t_max);
            let order = match (ta, tb) {
                (Some(x), Some(y)) if y < x => [a, b],
                _ => [b, a],
            };
            for c in order {
                if top < stack.len() {
                    stack[top] = c;
                    top += 1;
                }
            }
        }
    }
}

/// Entry distance of the ray into a node's box, if it enters before `t_max`.
fn slab(n: &Node, o: Vec3, inv: Vec3, t_max: f32) -> Option<f32> {
    let t1 = (n.min - o) * inv;
    let t2 = (n.max - o) * inv;
    // f32::min/max drop a NaN (0 × ∞, a ray in a box's face plane), which
    // leaves that axis unconstrained instead of poisoning the test.
    let near =
        t1.x.min(t2.x)
            .max(t1.y.min(t2.y))
            .max(t1.z.min(t2.z))
            .max(0.0);
    let far =
        t1.x.max(t2.x)
            .min(t1.y.max(t2.y))
            .min(t1.z.max(t2.z))
            .min(t_max);
    (near <= far).then_some(near)
}

/// Partition `tris` by the cheapest binned-SAH split along the widest axis
/// of their centroids. The split point, or `None` when no split beats a leaf.
fn split(tris: &mut [Tri]) -> Option<usize> {
    let (lo, hi) = tris.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), t| (lo.min(t.centroid()), hi.max(t.centroid())),
    );
    let extent = hi - lo;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    if extent[axis] <= 0.0 {
        return None;
    }
    let bin = |t: &Tri| {
        (((t.centroid()[axis] - lo[axis]) / extent[axis] * BINS as f32) as usize).min(BINS - 1)
    };
    let empty = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    let mut bins = [(empty, 0usize); BINS];
    for t in tris.iter() {
        let (a, b) = t.bounds();
        let (bounds, n) = &mut bins[bin(t)];
        *bounds = (bounds.0.min(a), bounds.1.max(b));
        *n += 1;
    }
    let area = |(lo, hi): (Vec3, Vec3)| {
        let e = (hi - lo).max(Vec3::ZERO);
        e.x * e.y + e.y * e.z + e.z * e.x
    };
    let grow = |a: (Vec3, Vec3), b: (Vec3, Vec3)| (a.0.min(b.0), a.1.max(b.1));
    // Cost of each split plane k (bins [0, k) left): left and right sweeps.
    let mut left = [(0.0f32, 0usize); BINS];
    let (mut acc, mut n) = (empty, 0);
    for k in 1..BINS {
        acc = grow(acc, bins[k - 1].0);
        n += bins[k - 1].1;
        left[k] = (area(acc), n);
    }
    let (mut acc, mut n) = (empty, 0);
    let mut best: Option<(f32, usize)> = None;
    for k in (1..BINS).rev() {
        acc = grow(acc, bins[k].0);
        n += bins[k].1;
        let (la, ln) = left[k];
        if ln == 0 || n == 0 {
            continue;
        }
        let cost = la * ln as f32 + area(acc) * n as f32;
        if best.is_none_or(|(c, _)| cost < c) {
            best = Some((cost, k));
        }
    }
    let (cost, k) = best?;
    let parent = area(tris.iter().fold(empty, |acc, t| grow(acc, t.bounds())));
    // A leaf costs its triangles; a split costs roughly a traversal step
    // plus its children's triangles weighted by the chance of entering them.
    if cost / parent.max(1e-12) + 1.0 >= tris.len() as f32 {
        return None;
    }
    let mut mid = 0;
    for i in 0..tris.len() {
        if bin(&tris[i]) < k {
            tris.swap(i, mid);
            mid += 1;
        }
    }
    (mid > 0 && mid < tris.len()).then_some(mid)
}

/// One mesh placed in the world, as the bake traces it.
pub struct Occluder<'a> {
    pub vertices: &'a [Vertex],
    pub indices: &'a [u32],
    pub transform: Mat4,
    pub double_sided: bool,
}

/// The scene's occluders ([`sky_occludes`]), each at its coarsest baked LOD
/// within `LOD_TOLERANCE` at its placed scale when it has a bake, else as
/// loaded.
pub fn occluders<'a>(scene: &'a SceneData, baked: &'a [Option<BakedMesh>]) -> Vec<Occluder<'a>> {
    scene
        .nodes
        .iter()
        .filter_map(|node| {
            let m = node.mesh?;
            let mesh = &scene.meshes[m];
            if !sky_occludes(node, mesh) {
                return None;
            }
            let m3 = Mat3::from_mat4(node.transform);
            let scale = m3
                .x_axis
                .length()
                .max(m3.y_axis.length())
                .max(m3.z_axis.length());
            let (vertices, indices) = match baked.get(m).and_then(Option::as_ref) {
                Some(b) => {
                    let lod = b
                        .lods
                        .iter()
                        .rev()
                        .find(|l| l.error * scale <= LOD_TOLERANCE)
                        .unwrap_or(&b.lods[0]);
                    (&b.vertices[..], &lod.indices[..])
                }
                None => (&mesh.vertices[..], &mesh.indices[..]),
            };
            Some(Occluder {
                vertices,
                indices,
                transform: node.transform,
                double_sided: mesh.material.double_sided,
            })
        })
        .collect()
}

/// World-space triangles of the occluders. Facing comes from the vertex
/// normals through the cofactor matrix, like mesh.vert, so a mirrored
/// placement still faces outwards.
pub fn triangles(occluders: &[Occluder]) -> Vec<Tri> {
    let mut out = Vec::new();
    for o in occluders {
        let m = Mat3::from_mat4(o.transform);
        let cof = Mat3::from_cols(
            m.y_axis.cross(m.z_axis),
            m.z_axis.cross(m.x_axis),
            m.x_axis.cross(m.y_axis),
        );
        let flip = if m.determinant() < 0.0 { -1.0 } else { 1.0 };
        for t in o.indices.as_chunks::<3>().0 {
            let v = t.map(|i| &o.vertices[i as usize]);
            let p = v.map(|v| o.transform.transform_point3(Vec3::from(v.pos)));
            if (p[1] - p[0]).cross(p[2] - p[0]).length_squared() < 1e-14 {
                continue;
            }
            let facing = if o.double_sided {
                Vec3::ZERO
            } else {
                flip * (cof * v.iter().map(|v| Vec3::from(v.normal)).sum::<Vec3>())
            };
            out.push(Tri::new(p, facing));
        }
    }
    out
}

/// Where the cells are: the occluders' bounds padded by `PAD` cells, in
/// `CELL`-sized cells unless that would be more than `MAX_CELLS`.
pub fn grid(tris: &[Tri]) -> (Vec3, f32, [u32; 3]) {
    let (lo, hi) = tris.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), t| {
            let (a, b) = t.bounds();
            (lo.min(a), hi.max(b))
        },
    );
    if tris.is_empty() {
        return (Vec3::ZERO, CELL, [1, 1, 1]);
    }
    let size = hi - lo;
    let mut cell = CELL;
    let count = |c: f32| {
        let n = (size / c).ceil() + 2.0 * PAD;
        n.x as f64 * n.y as f64 * n.z as f64
    };
    if count(cell) > MAX_CELLS {
        cell *= (count(cell) / MAX_CELLS).cbrt() as f32;
        while count(cell) > MAX_CELLS {
            cell *= 1.01;
        }
    }
    let n = (size / cell).ceil() + 2.0 * PAD;
    (lo - PAD * cell, cell, [n.x as u32, n.y as u32, n.z as u32])
}

/// `n` directions spread evenly over the sphere, or over the upper
/// hemisphere: a Fibonacci spiral, uniform in y (so in solid angle).
fn spiral(n: usize, hemisphere: bool) -> Vec<Vec3> {
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    (0..n)
        .map(|j| {
            let s = (j as f32 + 0.5) / n as f32;
            let y = if hemisphere { 1.0 - s } else { 1.0 - 2.0 * s };
            let r = (1.0 - y * y).max(0.0).sqrt();
            let phi = j as f32 * golden;
            Vec3::new(r * phi.cos(), y, r * phi.sin())
        })
        .collect()
}

/// A per-cell turn about the vertical, so neighbouring cells don't all miss
/// the same thin pole between the same two rays: structure becomes noise,
/// which the trilinear filter then averages.
fn cell_turn(x: u32, y: u32, z: u32) -> Mat3 {
    let mut h =
        x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77) ^ z.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    Mat3::from_rotation_y(h as f32 / u32::MAX as f32 * std::f32::consts::TAU)
}

/// What one bake found, for its report.
#[derive(Debug, Default, Clone, Copy)]
pub struct Stats {
    pub triangles: usize,
    /// Cells whose centre is inside geometry...
    pub inside: usize,
    /// ...and of those, how many no valid neighbour reached (left closed).
    pub unfilled: usize,
    /// Mean `2·w0` over the valid cells: the average fraction of sky seen.
    pub mean_sky: f32,
}

/// Bake a volume over `bvh`'s triangles on the given grid, on every core.
pub fn bake(bvh: &Bvh, origin: Vec3, cell: f32, dims: [u32; 3]) -> (SkyVolume, Stats) {
    let [nx, ny, nz] = dims;
    let cells = (nx * ny * nz) as usize;
    let hemi = spiral(RAYS, true);
    let sphere = spiral(PROBE_RAYS, false);
    // Rows of x, handed out one at a time: the open rows above a level are
    // far cheaper than the rows through it.
    let rows = (ny * nz) as usize;
    let next = AtomicUsize::new(0);
    let out = Mutex::new(vec![(SkyVis::CLOSED, SkyFree::CLEAR, false); cells]);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let row = next.fetch_add(1, Ordering::Relaxed);
                if row >= rows {
                    break;
                }
                let (y, z) = (row as u32 % ny, row as u32 / ny);
                let mut done = Vec::with_capacity(nx as usize);
                for x in 0..nx {
                    let p = origin + (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * cell;
                    let (vis, valid) = cell_value(bvh, p, &cell_turn(x, y, z), &hemi, &sphere);
                    done.push((vis, free_distances(bvh, p, cell), valid));
                }
                let base = row * nx as usize;
                out.lock().unwrap()[base..base + nx as usize].copy_from_slice(&done);
            });
        }
    });
    let cells_out = out.into_inner().unwrap();
    let mut values: Vec<(SkyVis, bool)> = cells_out.iter().map(|c| (c.0, c.2)).collect();
    let inside = values.iter().filter(|v| !v.1).count();
    let unfilled = fill(&mut values, dims);
    let valid: Vec<&SkyVis> = values.iter().filter(|v| v.1).map(|v| &v.0).collect();
    let stats = Stats {
        triangles: bvh.tris.len(),
        inside,
        unfilled,
        mean_sky: valid.iter().map(|v| 2.0 * v.w0).sum::<f32>() / valid.len().max(1) as f32,
    };
    let volume = SkyVolume {
        origin,
        cell,
        dims,
        texels: values
            .iter()
            .zip(&cells_out)
            .map(|(v, c)| {
                let (a, b) = (v.0.encode(), c.1.encode());
                [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
            })
            .collect(),
    };
    (volume, stats)
}

/// One cell: its visibility, and whether it's valid (not inside geometry).
fn cell_value(bvh: &Bvh, p: Vec3, turn: &Mat3, hemi: &[Vec3], sphere: &[Vec3]) -> (SkyVis, bool) {
    let back = sphere
        .iter()
        .filter(|&&d| {
            let d = *turn * d;
            bvh.nearest(p, d, f32::INFINITY)
                .is_some_and(|(_, t)| t.facing.dot(d) > 0.0)
        })
        .count();
    if back >= PROBE_INSIDE {
        return (SkyVis::CLOSED, false);
    }
    let (mut seen, mut w) = (0usize, Vec3::ZERO);
    for &d in hemi {
        let d = *turn * d;
        if !bvh.occluded(p, d, f32::INFINITY) {
            seen += 1;
            w += d;
        }
    }
    let m = hemi.len() as f32;
    (
        SkyVis {
            w0: 0.5 * seen as f32 / m,
            w: w / m,
        },
        true,
    )
}

/// How far `p` sees along ±x, ±y, ±z, in cells, up to one.
fn free_distances(bvh: &Bvh, p: Vec3, cell: f32) -> SkyFree {
    let reach = |d: Vec3| bvh.nearest(p, d, cell).map_or(1.0, |(t, _)| t / cell);
    SkyFree {
        plus: Vec3::new(reach(Vec3::X), reach(Vec3::Y), reach(Vec3::Z)),
        minus: Vec3::new(reach(-Vec3::X), reach(-Vec3::Y), reach(-Vec3::Z)),
    }
}

/// Give each invalid cell its darkest valid neighbour's value, ring by ring
/// for `FILL_PASSES` rings. Cells still unreached stay closed; the count of
/// them is returned.
fn fill(values: &mut [(SkyVis, bool)], [nx, ny, nz]: [u32; 3]) -> usize {
    let index = |x: u32, y: u32, z: u32| (x + nx * (y + ny * z)) as usize;
    for _ in 0..FILL_PASSES {
        let before = values.to_vec();
        let mut changed = false;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if before[index(x, y, z)].1 {
                        continue;
                    }
                    let mut darkest: Option<SkyVis> = None;
                    for dz in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let (qx, qy, qz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                                if qx < 0 || qy < 0 || qz < 0 {
                                    continue;
                                }
                                let (qx, qy, qz) = (qx as u32, qy as u32, qz as u32);
                                if qx >= nx || qy >= ny || qz >= nz {
                                    continue;
                                }
                                let (v, ok) = before[index(qx, qy, qz)];
                                if ok && darkest.is_none_or(|d| v.w0 < d.w0) {
                                    darkest = Some(v);
                                }
                            }
                        }
                    }
                    if let Some(v) = darkest {
                        values[index(x, y, z)] = (v, true);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    values.iter().filter(|v| !v.1).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use feather_assets::bake::Lod;
    use feather_assets::{MeshData, PrefabSpec, SceneNode};

    /// A deterministic stream in [0, 1).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn vec(&mut self, s: f32) -> Vec3 {
            Vec3::new(self.next(), self.next(), self.next()) * 2.0 * s - s
        }
    }

    /// Axis-aligned boxes from `lo` to `hi`, as the unit cube placed.
    fn boxes(b: &[(Vec3, Vec3)]) -> Vec<Tri> {
        let cube = MeshData::cube(1.0);
        let occ: Vec<Occluder> = b
            .iter()
            .map(|&(lo, hi)| Occluder {
                vertices: &cube.vertices,
                indices: &cube.indices,
                transform: Mat4::from_translation((lo + hi) / 2.0) * Mat4::from_scale(hi - lo),
                double_sided: false,
            })
            .collect();
        triangles(&occ)
    }

    fn baked(b: &[(Vec3, Vec3)]) -> (SkyVolume, Stats) {
        let tris = boxes(b);
        let (origin, cell, dims) = grid(&tris);
        bake(&Bvh::build(tris), origin, cell, dims)
    }

    /// A closed room: floor, roof and four walls, `t` thick, inside faces at
    /// x, z in [0, w] and y in [0, h]. `door` cuts a 1 × 2 m hole in the +x
    /// wall.
    fn room(w: f32, h: f32, t: f32, door: bool) -> Vec<(Vec3, Vec3)> {
        let v = Vec3::new;
        let mut b = vec![
            (v(-t, -t, -t), v(w + t, 0.0, w + t)),
            (v(-t, h, -t), v(w + t, h + t, w + t)),
            (v(-t, 0.0, -t), v(0.0, h, w + t)),
            (v(-t, 0.0, -t), v(w + t, h, 0.0)),
            (v(-t, 0.0, w), v(w + t, h, w + t)),
        ];
        let z0 = w / 2.0 - 0.5;
        if door {
            b.push((v(w, 0.0, -t), v(w + t, h, z0)));
            b.push((v(w, 0.0, z0 + 1.0), v(w + t, h, w + t)));
            b.push((v(w, 2.0, z0), v(w + t, h, z0 + 1.0)));
        } else {
            b.push((v(w, 0.0, -t), v(w + t, h, w + t)));
        }
        b
    }

    #[test]
    fn bvh_finds_what_brute_force_finds() {
        let mut r = Lcg(7);
        let tris: Vec<Tri> = (0..600)
            .map(|_| {
                let c = r.vec(5.0);
                Tri::new([c + r.vec(0.6), c + r.vec(0.6), c + r.vec(0.6)], r.vec(1.0))
            })
            .collect();
        let bvh = Bvh::build(tris.clone());
        let (mut hits, mut agree) = (0, 0);
        for _ in 0..3000 {
            let o = r.vec(7.0);
            let d = r.vec(1.0).normalize();
            let t_max = if r.next() < 0.3 { 4.0 } else { f32::INFINITY };
            let brute = tris
                .iter()
                .filter_map(|t| t.hit(o, d, t_max))
                .min_by(f32::total_cmp);
            let near = bvh.nearest(o, d, t_max).map(|h| h.0);
            assert_eq!(brute.is_some(), bvh.occluded(o, d, t_max), "any-hit");
            match (brute, near) {
                (Some(a), Some(b)) => {
                    assert!((a - b).abs() < 1e-5, "{a} vs {b}");
                    hits += 1;
                }
                (None, None) => {}
                other => panic!("nearest disagrees: {other:?}"),
            }
            agree += 1;
        }
        // Enough of both outcomes that the test means something.
        assert!(hits > 300 && agree - hits > 300, "{hits} hits of {agree}");
        // And the tree really is a tree, not one big leaf.
        assert!(bvh.nodes.len() > 100);
    }

    #[test]
    fn open_ground_sees_the_whole_sky() {
        let (vol, stats) = baked(&[(Vec3::new(-5.0, -0.2, -5.0), Vec3::new(5.0, 0.0, 5.0))]);
        let v = vol.sample(Vec3::new(0.0, 1.0, 0.0));
        let open = SkyVis::OPEN;
        assert!((v.w0 - open.w0).abs() < 0.005, "{v:?}");
        assert!((v.w - open.w).abs().max_element() < 0.01, "{v:?}");
        assert_eq!(stats.unfilled, 0);
    }

    #[test]
    fn a_closed_room_is_dark_inside_and_nothing_leaks_in() {
        let (vol, _) = baked(&room(4.0, 3.0, 0.2, false));
        let cell = vol.cell;
        // Surfaces as the shader samples them: one cell off along the normal.
        // Wall and floor points include ones a few cm from a perpendicular
        // wall, where the sample's neighbours reach across it.
        let v = Vec3::new;
        let surfaces = [
            (v(2.0, 0.0, 2.0), Vec3::Y),
            (v(2.0, 3.0, 2.0), -Vec3::Y),
            (v(0.0, 1.5, 2.0), Vec3::X),
            (v(0.03, 0.0, 2.0), Vec3::Y),
            (v(0.03, 0.0, 0.03), Vec3::Y),
            (v(0.0, 2.97, 0.03), Vec3::X),
            (v(4.0, 0.05, 3.97), -Vec3::X),
        ];
        for (p, n) in surfaces {
            let s = vol.sample(p + n * cell);
            assert!(s.weight(n) < 0.05, "{p} facing {n}: {s:?}");
        }
    }

    #[test]
    fn a_door_lets_sky_in_from_its_side() {
        let (vol, _) = baked(&room(4.0, 3.0, 0.2, true));
        let (closed, _) = baked(&room(4.0, 3.0, 0.2, false));
        let p = Vec3::new(3.0, 1.0, 2.0);
        let (s, c) = (vol.sample(p), closed.sample(p));
        // Some sky, less than half of it, and it comes from +x.
        assert!(s.w0 > c.w0 + 0.005 && s.w0 < 0.25, "{s:?} vs {c:?}");
        assert!(s.w.x > 0.0 && s.w.x > s.w.z.abs(), "{s:?}");
        assert!(s.weight(Vec3::X) > s.weight(-Vec3::X), "{s:?}");
    }

    #[test]
    fn a_canopy_shades_what_is_under_it_only() {
        let v = Vec3::new;
        // A floor too, which blocks no sky but stretches the grid out to
        // where the comparison is.
        let (vol, _) = baked(&[
            (v(-3.0, 3.0, -3.0), v(3.0, 3.2, 3.0)),
            (v(-3.0, -0.2, -3.0), v(22.0, 0.0, 3.0)),
        ]);
        let under = vol.sample(v(0.0, 0.5, 0.0));
        // 17 m past its edge the canopy is 8° above the horizon.
        let beside = vol.sample(v(20.0, 0.5, 0.0));
        assert!(under.weight(Vec3::Y) < 0.5, "{under:?}");
        assert!(beside.weight(Vec3::Y) > 0.95, "{beside:?}");
        // Under the edge, the open side is brighter than the covered side.
        let edge = vol.sample(v(2.5, 0.5, 0.0));
        assert!(edge.weight(Vec3::X) > edge.weight(-Vec3::X), "{edge:?}");
    }

    #[test]
    fn the_probe_finds_cells_inside_closed_geometry() {
        let hemi = spiral(RAYS, true);
        let sphere = spiral(PROBE_RAYS, false);
        let solid = Bvh::build(boxes(&[(Vec3::splat(-1.0), Vec3::splat(1.0))]));
        let turn = Mat3::IDENTITY;
        assert!(!cell_value(&solid, Vec3::ZERO, &turn, &hemi, &sphere).1);
        assert!(cell_value(&solid, Vec3::new(3.0, 0.0, 0.0), &turn, &hemi, &sphere).1);
        // A double-sided box has no inside to find.
        let cube = MeshData::cube(2.0);
        let open = Bvh::build(triangles(&[Occluder {
            vertices: &cube.vertices,
            indices: &cube.indices,
            transform: Mat4::IDENTITY,
            double_sided: true,
        }]));
        assert!(cell_value(&open, Vec3::ZERO, &turn, &hemi, &sphere).1);
    }

    #[test]
    fn inside_cells_take_their_darkest_valid_neighbour() {
        let lit = SkyVis {
            w0: 0.4,
            w: Vec3::new(0.0, 0.4, 0.0),
        };
        let dim = SkyVis {
            w0: 0.1,
            w: Vec3::new(0.0, 0.1, 0.0),
        };
        let invalid = (SkyVis::OPEN, false);
        // A wall cell between the open and a dim room takes the room's value.
        let mut v = vec![(lit, true), invalid, (dim, true)];
        assert_eq!(fill(&mut v, [3, 1, 1]), 0);
        assert_eq!(v[1], (dim, true));
        // Filling walks one ring per pass, up to FILL_PASSES...
        let mut v = vec![(lit, true), invalid, invalid, invalid, (lit, true)];
        assert_eq!(fill(&mut v, [5, 1, 1]), 0);
        assert!(v.iter().all(|c| *c == (lit, true)));
        // ...and what it can't reach stays closed.
        let mut v = vec![invalid; 2 * FILL_PASSES + 3];
        v[0] = (lit, true);
        let n = v.len() as u32;
        assert_eq!(fill(&mut v, [n, 1, 1]), n as usize - 1 - FILL_PASSES);
    }

    #[test]
    fn a_mirrored_placement_still_faces_outwards() {
        let cube = MeshData::cube(1.0);
        for transform in [
            Mat4::IDENTITY,
            Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0)),
            Mat4::from_scale(Vec3::new(2.0, -0.5, 3.0)),
        ] {
            let tris = triangles(&[Occluder {
                vertices: &cube.vertices,
                indices: &cube.indices,
                transform,
                double_sided: false,
            }]);
            assert_eq!(tris.len(), 12);
            for t in &tris {
                assert!(t.facing.dot(t.centroid()) > 0.0, "{transform}: {t:?}");
            }
        }
    }

    #[test]
    fn occluders_skip_cutouts_and_shadowless_nodes_and_pick_a_close_lod() {
        let wall = MeshData::cube(1.0);
        let mut grass = MeshData::cube(1.0);
        grass.material.alpha_mode = feather_assets::AlphaMode::Mask(0.5);
        let lod = |error, n| Lod {
            error,
            indices: wall.indices[..n].to_vec(),
        };
        let bake = BakedMesh {
            vertices: wall.vertices.clone(),
            lods: vec![lod(0.0, 36), lod(0.01, 24), lod(0.2, 12)],
        };
        let node = |mesh, scale: f32, shadow: Option<bool>| SceneNode {
            mesh: Some(mesh),
            transform: Mat4::from_scale(Vec3::splat(scale)),
            prefab: shadow.map(|s| PrefabSpec {
                id: "prop".into(),
                params: serde_json::json!({ "shadow": s }),
            }),
        };
        let scene = SceneData {
            meshes: vec![wall.clone(), grass],
            nodes: vec![
                node(0, 1.0, None),
                node(0, 10.0, Some(true)),
                node(0, 1.0, Some(false)),
                node(1, 1.0, None),
                SceneNode {
                    mesh: None,
                    transform: Mat4::IDENTITY,
                    prefab: None,
                },
            ],
        };
        let baked = vec![Some(bake), None];
        let occ = occluders(&scene, &baked);
        // The two casting walls: at scale 1 the 1 cm LOD, at scale 10 that
        // is 10 cm, too coarse, so LOD0.
        let tris: Vec<usize> = occ.iter().map(|o| o.indices.len() / 3).collect();
        assert_eq!(tris, [8, 12]);
    }

    #[test]
    fn big_levels_get_bigger_cells() {
        let v = Vec3::new;
        let small = boxes(&[(v(0.0, 0.0, 0.0), v(10.0, 3.0, 10.0))]);
        let (origin, cell, dims) = grid(&small);
        assert_eq!(cell, CELL);
        assert_eq!(origin, v(-1.0, -1.0, -1.0));
        assert_eq!(dims, [24, 10, 24]);
        let big = boxes(&[(v(0.0, 0.0, 0.0), v(2000.0, 50.0, 2000.0))]);
        let (_, cell, dims) = grid(&big);
        let cells = dims.iter().map(|&d| d as f64).product::<f64>();
        assert!(cell > CELL && cells <= MAX_CELLS, "{cell} m, {cells} cells");
    }
}
