//! Baked assets (§17): the half of the bake that the runtime and the `bake`
//! tool share, which is the content keys and the file formats.
//!
//! The bake turns each (image, colour space) a scene uses into a BC7 mip chain,
//! and each mesh into an optimised vertex order plus a LOD chain, stored under
//! keys derived from the decoded content. The runtime computes the same keys
//! and uses a baked file when it finds one, falling back to the raw asset when
//! it doesn't. Open sources stay the truth, and the cache is disposable.
//!
//! Layout under the bake root: `tex/<key>.bc7`, `mesh/<key>.fbm` and
//! `sky/<key>.fsv`.

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use glam::Vec3;

use crate::{AlphaMode, MeshData, SceneData, SceneNode, TextureData, Vertex};

/// Subdirectory of the bake root holding baked textures.
pub const TEX_DIR: &str = "tex";
/// Subdirectory of the bake root holding baked meshes.
pub const MESH_DIR: &str = "mesh";

/// Bump when the bake's output changes meaning (mip filter, encoder settings,
/// file layout): stale cache entries then simply aren't found.
pub const BAKE_VERSION: u32 = 2;
const MAGIC: [u8; 4] = *b"FBTX";

/// What a texture is used as, which decides how it's baked.
///
/// Normal maps are `Data`, i.e. three-channel BC7, on purpose. BC5 (X, Y with Z
/// rebuilt in the shader) was measured on the detail scene's four normal maps
/// and rejected: the rock and grass maps hold inward-pointing and non-unit
/// texels (25% and 24% of them) that only a third channel reproduces, so BC5
/// moved their mean error from 1.1° to 17.8° (rocks) and 1.4° to 5.2° (grass),
/// while gaining under 0.1° on the clean bust map. Same VRAM either way.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TexKind {
    /// Base colour: BC7, sampled as sRGB.
    Color,
    /// Linear data (metallic-roughness, normal maps): BC7, UNORM.
    Data,
}

impl TexKind {
    /// Sampled with sRGB decoding (only base colour).
    pub fn srgb(self) -> bool {
        self == TexKind::Color
    }

    fn tag(self) -> u32 {
        match self {
            TexKind::Data => 0,
            TexKind::Color => 1,
        }
    }

    fn from_tag(tag: u32) -> Option<Self> {
        [TexKind::Data, TexKind::Color]
            .into_iter()
            .find(|k| k.tag() == tag)
    }
}

/// Content key of one texture as the bake sees it: the image's **encoded**
/// bytes (`source_key`), so the runtime can find the bake without decoding
/// anything, plus the kind, since the same image baked as sRGB colour and as
/// linear data gives different outputs.
pub fn texture_key(tex: &TextureData, kind: TexKind) -> u64 {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    h.update(&BAKE_VERSION.to_le_bytes());
    h.update(&kind.tag().to_le_bytes());
    h.update(&tex.width.to_le_bytes());
    h.update(&tex.height.to_le_bytes());
    h.update(&tex.source_key.to_le_bytes());
    h.digest()
}

/// Where a key's baked file lives in the cache directory.
pub fn baked_path(dir: &Path, key: u64) -> PathBuf {
    dir.join(format!("{key:016x}.bc7"))
}

/// Bump when the mesh bake's output changes meaning (simplifier settings, LOD
/// rules, file layout). Separate from `BAKE_VERSION` so re-tuning LODs doesn't
/// invalidate the (slow) texture bake.
pub const MESH_BAKE_VERSION: u32 = 1;
/// Meshes under this many triangles bake to LOD0 only: their cost is the
/// draw, not the triangles. Shared so the runtime only suggests baking meshes
/// that would gain LODs.
pub const MIN_LOD_TRIS: usize = 256;
const MESH_MAGIC: [u8; 4] = *b"FBMS";

/// Content key of one mesh: its vertices and indices exactly as the loader
/// produced them. The material isn't part of it, so a shape reused with
/// several materials bakes once.
pub fn mesh_key(mesh: &MeshData) -> u64 {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    h.update(&MESH_BAKE_VERSION.to_le_bytes());
    h.update(&(mesh.vertices.len() as u64).to_le_bytes());
    h.update(pod_bytes(&mesh.vertices));
    h.update(pod_bytes(&mesh.indices));
    h.digest()
}

/// Where a key's baked mesh lives in the mesh directory.
pub fn baked_mesh_path(dir: &Path, key: u64) -> PathBuf {
    dir.join(format!("{key:016x}.fbm"))
}

fn pod_bytes<T: Copy>(s: &[T]) -> &[u8] {
    // Vertex is repr(C) f32s and u32 has no padding: every byte is initialised.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// The baked version of each mesh (`None` where there isn't a valid one),
/// read from the mesh directory of a bake root. Loaded once per session: the
/// renderer draws from it and the app builds colliders from its LODs.
pub fn load_baked_meshes(root: &Path, meshes: &[MeshData]) -> Vec<Option<BakedMesh>> {
    let dir = root.join(MESH_DIR);
    meshes
        .iter()
        .map(|m| BakedMesh::read(&baked_mesh_path(&dir, mesh_key(m))).ok())
        .collect()
}

/// One level of detail: an index list into the shared vertex array.
#[derive(Debug, Clone, PartialEq)]
pub struct Lod {
    /// Geometric error of this level against the original, in mesh-local
    /// units (0 for LOD0). The runtime projects it to decide when the level
    /// is good enough.
    pub error: f32,
    pub indices: Vec<u32>,
}

/// A baked mesh: vertices reordered for fetch locality, shared by every LOD,
/// and LODs from full detail (0) to coarsest.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedMesh {
    pub vertices: Vec<Vertex>,
    pub lods: Vec<Lod>,
}

impl BakedMesh {
    /// Magic, version, vertex count, raw `Vertex` array, LOD count, then per
    /// LOD its error, index count and indices. Temp + rename like textures.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let tmp = path.with_extension("tmp");
        {
            let mut f = io::BufWriter::new(std::fs::File::create(&tmp)?);
            f.write_all(&MESH_MAGIC)?;
            f.write_all(&MESH_BAKE_VERSION.to_le_bytes())?;
            f.write_all(&(self.vertices.len() as u32).to_le_bytes())?;
            f.write_all(pod_bytes(&self.vertices))?;
            f.write_all(&(self.lods.len() as u32).to_le_bytes())?;
            for lod in &self.lods {
                f.write_all(&lod.error.to_le_bytes())?;
                f.write_all(&(lod.indices.len() as u32).to_le_bytes())?;
                f.write_all(pod_bytes(&lod.indices))?;
            }
            f.flush()?;
        }
        std::fs::rename(tmp, path)
    }

    /// Read and *validate* a baked mesh. Anything the renderer would trust
    /// blindly is checked: sizes, whole triangles, every index in range, at
    /// least one LOD, and errors that never decrease (the selection walks the
    /// chain assuming coarser means worse). Any failure is an error and the
    /// caller uses the raw mesh.
    pub fn read(path: &Path) -> io::Result<Self> {
        let mut data = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut data)?;
        let mut r = Cursor { data: &data, at: 0 };
        if r.take(4)? != MESH_MAGIC {
            return Err(invalid("not a baked mesh"));
        }
        if r.u32()? != MESH_BAKE_VERSION {
            return Err(invalid("baked by another version"));
        }
        let vertex_count = r.u32()? as usize;
        let raw = r.take(vertex_count * std::mem::size_of::<Vertex>())?;
        // 8 little-endian f32s per vertex: pos, normal, uv.
        let (floats, _) = raw.as_chunks::<4>();
        let vertices: Vec<Vertex> = floats
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| {
                let f = v.map(f32::from_le_bytes);
                Vertex {
                    pos: [f[0], f[1], f[2]],
                    normal: [f[3], f[4], f[5]],
                    uv: [f[6], f[7]],
                }
            })
            .collect();
        let lod_count = r.u32()?;
        if lod_count == 0 {
            return Err(invalid("no LODs"));
        }
        let mut lods: Vec<Lod> = Vec::with_capacity(lod_count as usize);
        for _ in 0..lod_count {
            let error = f32::from_le_bytes(r.take(4)?.try_into().unwrap());
            let count = r.u32()? as usize;
            if count == 0 || !count.is_multiple_of(3) {
                return Err(invalid("LOD isn't whole triangles"));
            }
            let indices: Vec<u32> = r
                .take(count * 4)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            if indices.iter().any(|&i| i as usize >= vertex_count) {
                return Err(invalid("index out of range"));
            }
            let prev = lods.last().map_or(0.0, |l| l.error);
            if !error.is_finite() || error < prev {
                return Err(invalid("LOD errors must be finite and non-decreasing"));
            }
            lods.push(Lod { error, indices });
        }
        if r.at != data.len() {
            return Err(invalid("trailing bytes"));
        }
        Ok(Self { vertices, lods })
    }
}

/// Bytes of one BC7 mip level: 16 per 4×4 block, partial blocks rounded up,
/// which is exactly what Vulkan expects.
pub fn bc7_level_size(width: u32, height: u32, level: u32) -> usize {
    let w = (width >> level).max(1);
    let h = (height >> level).max(1);
    (w.div_ceil(4) * h.div_ceil(4) * 16) as usize
}

/// A baked BC7 mip chain, level 0 first, down to 1×1.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedTexture {
    pub kind: TexKind,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Vec<u8>>,
}

impl BakedTexture {
    /// Thin header + raw GPU-ready payloads (§17): magic, version, kind,
    /// width, height, level count, then each level's length and blocks.
    /// Written to a temporary name and renamed, so an interrupted bake never
    /// leaves a truncated file that a later run would trust.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let tmp = path.with_extension("tmp");
        {
            let mut f = io::BufWriter::new(std::fs::File::create(&tmp)?);
            f.write_all(&MAGIC)?;
            for v in [
                BAKE_VERSION,
                self.kind.tag(),
                self.width,
                self.height,
                self.levels.len() as u32,
            ] {
                f.write_all(&v.to_le_bytes())?;
            }
            for level in &self.levels {
                f.write_all(&(level.len() as u32).to_le_bytes())?;
                f.write_all(level)?;
            }
            f.flush()?;
        }
        std::fs::rename(tmp, path)
    }

    /// Read and *validate* a baked file: a wrong magic, a version from another
    /// bake, or a level whose size doesn't match its dimensions is an error,
    /// and the caller falls back to the raw texture.
    pub fn read(path: &Path) -> io::Result<Self> {
        let mut data = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut data)?;
        let mut r = Cursor { data: &data, at: 0 };
        if r.take(4)? != MAGIC {
            return Err(invalid("not a baked texture"));
        }
        let (version, kind, width, height, count) =
            (r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?);
        if version != BAKE_VERSION {
            return Err(invalid("baked by another version"));
        }
        let kind = TexKind::from_tag(kind).ok_or_else(|| invalid("unknown texture kind"))?;
        let mut levels = Vec::with_capacity(count as usize);
        for level in 0..count {
            let len = r.u32()? as usize;
            if len != bc7_level_size(width, height, level) {
                return Err(invalid("level size doesn't match its dimensions"));
            }
            levels.push(r.take(len)?.to_vec());
        }
        Ok(Self {
            kind,
            width,
            height,
            levels,
        })
    }
}

/// Subdirectory of the bake root holding sky-visibility volumes.
pub const SKY_DIR: &str = "sky";
/// Bump when the sky bake's output changes meaning: its cell size, ray
/// counts, LOD tolerance, fill rule or file layout all live in the bake tool,
/// and only this version tells the runtime a cached volume is stale.
pub const SKY_BAKE_VERSION: u32 = 1;
const SKY_MAGIC: [u8; 4] = *b"FBSV";
/// Largest volume a file may declare, in cells: far above what the bake
/// writes (it grows its cells to stay under 4M), low enough that a damaged
/// header can't ask for gigabytes.
const SKY_MAX_CELLS: u64 = 1 << 26;

/// Whether a scene node's geometry blocks the sky in the sky bake: what the
/// sun's shadow sees (a node with `shadow: false` doesn't), except cutouts,
/// whose alpha the CPU bake doesn't test. Grass and chain-link are mostly
/// holes, so leaving them out is closer than treating them as solid.
pub fn sky_occludes(node: &SceneNode, mesh: &MeshData) -> bool {
    let casts = node
        .prefab
        .as_ref()
        .and_then(|p| p.bool("shadow"))
        .unwrap_or(true);
    casts && !matches!(mesh.material.alpha_mode, AlphaMode::Mask(_))
}

/// Content key of a scene's sky volume: every occluder's mesh (by
/// [`mesh_key`], which also covers the LODs the bake traces) with its world
/// transform and sidedness, in node order. Materials otherwise don't matter,
/// so retexturing a level keeps its volume.
pub fn sky_key(scene: &SceneData) -> u64 {
    let mut mesh_keys: Vec<Option<u64>> = vec![None; scene.meshes.len()];
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    h.update(&SKY_BAKE_VERSION.to_le_bytes());
    for node in &scene.nodes {
        let Some(m) = node.mesh else { continue };
        let mesh = &scene.meshes[m];
        if !sky_occludes(node, mesh) {
            continue;
        }
        let key = *mesh_keys[m].get_or_insert_with(|| mesh_key(mesh));
        h.update(&key.to_le_bytes());
        for f in node.transform.to_cols_array() {
            h.update(&f.to_bits().to_le_bytes());
        }
        h.update(&[mesh.material.double_sided as u8]);
    }
    h.digest()
}

/// Where a key's sky volume lives in the sky directory.
pub fn baked_sky_path(dir: &Path, key: u64) -> PathBuf {
    dir.join(format!("{key:016x}.fsv"))
}

/// How much sky one point sees (§13), as the two moments of the visibility
/// V(ω) (1 where a ray escapes the level) over the **upper** hemisphere:
/// `w0 = (1/4π)∫V dω` and `w = (1/2π)∫V ω dω`.
///
/// They're what the clamped-cosine convolution of V's first two spherical
/// harmonic bands needs, so the sky light a surface with normal `n` receives,
/// relative to the whole sky's, is `max(w0 + w·n, 0)`. For open sky that is
/// `½ + ½n.y`, exactly the weight the shader's `sky_irradiance` already gives
/// the sky, so an unoccluded point shades as it always did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyVis {
    /// In [0, ½]: ½ is all of the upper hemisphere visible.
    pub w0: f32,
    /// Each component in [−¼, ¼] across (x, z), and [0, ½] up (y).
    pub w: Vec3,
}

impl SkyVis {
    /// Nothing in the way.
    pub const OPEN: SkyVis = SkyVis {
        w0: 0.5,
        w: Vec3::new(0.0, 0.5, 0.0),
    };
    /// No sky at all.
    pub const CLOSED: SkyVis = SkyVis {
        w0: 0.0,
        w: Vec3::ZERO,
    };

    /// The sky weight for a surface facing `n` (unit): the fraction of the
    /// whole sky's light it receives, `½ + ½n.y` in the open.
    pub fn weight(self, n: Vec3) -> f32 {
        (self.w0 + self.w.dot(n)).max(0.0)
    }

    /// RGBA8, scaled so every component spans its range and the open sky is
    /// exact: `r = 510·w0`, `g = 127 + 508·w.x`, `b = 510·w.y`,
    /// `a = 127 + 508·w.z`. mesh.frag decodes the same numbers (a test
    /// checks), and decoding is linear, so trilinear filtering is exact.
    pub fn encode(self) -> [u8; 4] {
        let q = |v: f32| v.round().clamp(0.0, 255.0) as u8;
        [
            q(510.0 * self.w0),
            q(127.0 + 508.0 * self.w.x),
            q(510.0 * self.w.y),
            q(127.0 + 508.0 * self.w.z),
        ]
    }

    /// Inverse of [`encode`](Self::encode), from UNORM values in [0, 1] as a
    /// shader samples them (so it also decodes filtered texels).
    pub fn decode(t: [f32; 4]) -> Self {
        SkyVis {
            w0: t[0] * 255.0 / 510.0,
            w: Vec3::new(
                (t[1] * 255.0 - 127.0) / 508.0,
                t[2] * 255.0 / 510.0,
                (t[3] * 255.0 - 127.0) / 508.0,
            ),
        }
    }
}

/// How far one cell centre can see along each axis before geometry, in cells
/// (0 to 1; 1 means clear all the way to the neighbouring centre).
///
/// Sampling needs it because a grid's cells are coarser than the thinnest
/// geometry: a point under a 0.1 m roof has cells above the roof among its
/// eight neighbours, and a plain trilinear blend would carry their open sky
/// through it. A neighbour whose path to the sampled point is blocked is left
/// out instead ([`SkyVolume::sample`], mesh.frag's `sky_visibility`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyFree {
    /// Towards +x, +y, +z.
    pub plus: Vec3,
    /// Towards −x, −y, −z.
    pub minus: Vec3,
}

impl SkyFree {
    /// Clear in every direction.
    pub const CLEAR: SkyFree = SkyFree {
        plus: Vec3::ONE,
        minus: Vec3::ONE,
    };

    /// Four bits per direction, rounded **down** (a wall is never reported
    /// further than it is): byte k holds +axis k in its low nibble and −axis
    /// k in its high one, and the fourth byte is 0.
    pub fn encode(self) -> [u8; 4] {
        let q = |v: f32| (v.clamp(0.0, 1.0) * 15.0).floor() as u8;
        let b = |a: usize| q(self.plus[a]) | q(self.minus[a]) << 4;
        [b(0), b(1), b(2), 0]
    }

    pub fn decode(b: [u8; 4]) -> Self {
        let n = |a: usize, shift: u32| ((b[a] >> shift) & 15) as f32 / 15.0;
        SkyFree {
            plus: Vec3::new(n(0, 0), n(1, 0), n(2, 0)),
            minus: Vec3::new(n(0, 4), n(1, 4), n(2, 4)),
        }
    }

    /// How much a neighbour this far away (in cells, per axis, signed from
    /// this centre) is visible from it: 1 within the free distance on every
    /// axis, falling to 0 over `SKY_FREE_SOFT` cells past it. Soft, so a
    /// sample sliding past a prop's edge changes smoothly; short, so no wall
    /// of 0.1 cell or more lets anything through.
    pub fn visible(self, delta: Vec3) -> f32 {
        let free = Vec3::select(delta.cmpge(Vec3::ZERO), self.plus, self.minus);
        let over = (delta.abs() - free).max_element();
        (1.0 - over / SKY_FREE_SOFT).clamp(0.0, 1.0)
    }
}

/// See [`SkyFree::visible`], in cells. mesh.frag uses the same (a test
/// checks).
pub const SKY_FREE_SOFT: f32 = 0.1;

/// A baked sky-visibility volume: per cubic cell of a grid over the level,
/// the [`SkyVis`] at its centre and the [`SkyFree`] distances round it.
#[derive(Debug, Clone, PartialEq)]
pub struct SkyVolume {
    /// World position of the grid's minimum corner.
    pub origin: Vec3,
    /// Edge of one cell, in metres.
    pub cell: f32,
    /// Cells along x, y, z.
    pub dims: [u32; 3],
    /// Per cell, x fastest, then y, then z (a 3D image's layout): bytes 0–3
    /// the encoded `SkyVis`, 4–7 the encoded `SkyFree`. One RG32_UINT texel
    /// on the GPU.
    pub texels: Vec<[u8; 8]>,
}

impl SkyVolume {
    /// Magic, version, origin (3 f32), cell (f32), dims (3 u32), then the
    /// texels. Temp + rename like the other bakes.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let tmp = path.with_extension("tmp");
        {
            let mut f = io::BufWriter::new(std::fs::File::create(&tmp)?);
            f.write_all(&SKY_MAGIC)?;
            f.write_all(&SKY_BAKE_VERSION.to_le_bytes())?;
            for v in self.origin.to_array().into_iter().chain([self.cell]) {
                f.write_all(&v.to_le_bytes())?;
            }
            for d in self.dims {
                f.write_all(&d.to_le_bytes())?;
            }
            f.write_all(self.texels.as_flattened())?;
            f.flush()?;
        }
        std::fs::rename(tmp, path)
    }

    /// Read and *validate* a volume: finite origin, a positive finite cell,
    /// non-empty and bounded dimensions, and exactly one texel per cell.
    pub fn read(path: &Path) -> io::Result<Self> {
        let mut data = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut data)?;
        let mut r = Cursor { data: &data, at: 0 };
        if r.take(4)? != SKY_MAGIC {
            return Err(invalid("not a sky volume"));
        }
        if r.u32()? != SKY_BAKE_VERSION {
            return Err(invalid("baked by another version"));
        }
        let mut f = [0.0f32; 4];
        for v in &mut f {
            *v = f32::from_bits(r.u32()?);
        }
        let origin = Vec3::new(f[0], f[1], f[2]);
        let cell = f[3];
        if !origin.is_finite() || !cell.is_finite() || cell <= 0.0 {
            return Err(invalid("bad origin or cell size"));
        }
        let dims = [r.u32()?, r.u32()?, r.u32()?];
        let cells = dims.iter().map(|&d| d as u64).product::<u64>();
        if cells == 0 || cells > SKY_MAX_CELLS {
            return Err(invalid("bad dimensions"));
        }
        let texels = r.take(cells as usize * 8)?.as_chunks::<8>().0.to_vec();
        if r.at != data.len() {
            return Err(invalid("trailing bytes"));
        }
        Ok(Self {
            origin,
            cell,
            dims,
            texels,
        })
    }

    /// Cell `(x, y, z)`, decoded.
    pub fn at(&self, x: u32, y: u32, z: u32) -> (SkyVis, SkyFree) {
        let [nx, ny, _] = self.dims;
        let t = self.texels[(x + nx * (y + ny * z)) as usize];
        let vis = [t[0], t[1], t[2], t[3]].map(|v| v as f32 / 255.0);
        (
            SkyVis::decode(vis),
            SkyFree::decode([t[4], t[5], t[6], t[7]]),
        )
    }

    /// The sky visibility at world point `p`: a trilinear blend of the eight
    /// cells round it (clamped to the grid, like a clamp-to-edge sampler),
    /// leaving out those it can't see ([`SkyFree::visible`]) and renormalising.
    /// If it can see none, the plain blend. mesh.frag's `sky_visibility` is
    /// this, step for step.
    pub fn sample(&self, p: Vec3) -> SkyVis {
        let dims = Vec3::new(
            self.dims[0] as f32,
            self.dims[1] as f32,
            self.dims[2] as f32,
        );
        // Texel space, centres on integers.
        let t = ((p - self.origin) / self.cell - 0.5).clamp(Vec3::ZERO, dims - 1.0);
        let i = t.floor();
        let f = t - i;
        let (mut seen, mut plain) = ((0.0, 0.0, Vec3::ZERO), (0.0, Vec3::ZERO));
        for corner in 0..8u32 {
            let o = Vec3::new(
                (corner & 1) as f32,
                ((corner >> 1) & 1) as f32,
                ((corner >> 2) & 1) as f32,
            );
            let c = (i + o).min(dims - 1.0);
            let tri = Vec3::select(o.cmpeq(Vec3::ONE), f, 1.0 - f);
            let tri = tri.x * tri.y * tri.z;
            let (v, free) = self.at(c.x as u32, c.y as u32, c.z as u32);
            let w = tri * free.visible(f - o);
            seen = (seen.0 + w, seen.1 + w * v.w0, seen.2 + w * v.w);
            plain = (plain.0 + tri * v.w0, plain.1 + tri * v.w);
        }
        if seen.0 > 1e-3 {
            SkyVis {
                w0: seen.1 / seen.0,
                w: seen.2 / seen.0,
            }
        } else {
            SkyVis {
                w0: plain.0,
                w: plain.1,
            }
        }
    }
}

fn invalid(why: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, why.to_string())
}

/// Bounds-checked reads over a baked file.
struct Cursor<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let s = self
            .data
            .get(self.at..self.at + n)
            .ok_or_else(|| invalid("truncated"))?;
        self.at += n;
        Ok(s)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tex(fill: u8) -> TextureData {
        TextureData::from_rgba8(vec![fill; 8 * 8 * 4], 8, 8)
    }

    #[test]
    fn keys_are_content_and_kind_addressed() {
        let k = |t: &TextureData, kind| texture_key(t, kind);
        assert_eq!(k(&tex(1), TexKind::Color), k(&tex(1), TexKind::Color));
        assert_ne!(k(&tex(1), TexKind::Color), k(&tex(2), TexKind::Color));
        // Same image, different use: different bakes.
        assert_ne!(k(&tex(1), TexKind::Color), k(&tex(1), TexKind::Data));
        // Keyed on the source, not on decoding: two textures with the same
        // encoded bytes share a key without either being decoded.
        let png = |c: u8| {
            let mut out = Vec::new();
            image::RgbImage::from_pixel(2, 2, image::Rgb([c, 0, 0]))
                .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
                .unwrap();
            TextureData::from_encoded(out).unwrap()
        };
        let (a, b) = (png(7), png(7));
        assert_eq!(k(&a, TexKind::Data), k(&b, TexKind::Data));
        assert_ne!(k(&a, TexKind::Data), k(&png(8), TexKind::Data));
        assert!(!a.is_decoded() && !b.is_decoded());
    }

    #[test]
    fn bc7_level_sizes_round_partial_blocks_up() {
        assert_eq!(bc7_level_size(2048, 2048, 0), 2048 * 2048); // 1 byte/pixel
        assert_eq!(bc7_level_size(8, 8, 1), 16); // 4x4: one block
        assert_eq!(bc7_level_size(8, 8, 3), 16); // 1x1 still costs a block
        assert_eq!(bc7_level_size(6, 2, 0), 32); // 2 blocks by 1
    }

    fn mesh() -> BakedMesh {
        let v = |x: f32| Vertex {
            pos: [x, 0.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            uv: [x, 0.5],
        };
        BakedMesh {
            vertices: (0..4).map(|i| v(i as f32)).collect(),
            lods: vec![
                Lod {
                    error: 0.0,
                    indices: vec![0, 1, 2, 0, 2, 3],
                },
                Lod {
                    error: 0.25,
                    indices: vec![0, 1, 3],
                },
            ],
        }
    }

    #[test]
    fn mesh_keys_ignore_material_but_not_geometry() {
        let m = mesh();
        let data = |indices: Vec<u32>| MeshData {
            vertices: m.vertices.clone(),
            indices,
            material: crate::Material::default(),
        };
        let a = data(vec![0, 1, 2]);
        let mut b = data(vec![0, 1, 2]);
        b.material.roughness = 0.1;
        assert_eq!(mesh_key(&a), mesh_key(&b));
        assert_ne!(mesh_key(&a), mesh_key(&data(vec![0, 2, 1])));
    }

    #[test]
    fn baked_meshes_round_trip_and_reject_damage() {
        let dir = std::env::temp_dir().join(format!("feather_bake_mesh_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = baked_mesh_path(&dir, 0x42);
        mesh().write(&path).unwrap();
        assert_eq!(BakedMesh::read(&path).unwrap(), mesh());

        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(BakedMesh::read(&path).is_err(), "truncated");
        let mut extra = bytes.clone();
        extra.push(0);
        std::fs::write(&path, &extra).unwrap();
        assert!(BakedMesh::read(&path).is_err(), "trailing bytes");
        let mut magic = bytes.clone();
        magic[0] = b'X';
        std::fs::write(&path, &magic).unwrap();
        assert!(BakedMesh::read(&path).is_err(), "bad magic");

        for (why, damage) in [
            ("index out of range", {
                let mut m = mesh();
                m.lods[1].indices[2] = 4;
                m
            }),
            ("decreasing error", {
                let mut m = mesh();
                m.lods[1].error = -1.0;
                m
            }),
            ("NaN error", {
                let mut m = mesh();
                m.lods[1].error = f32::NAN;
                m
            }),
            ("partial triangle", {
                let mut m = mesh();
                m.lods[1].indices.pop();
                m
            }),
            ("no LODs", {
                let mut m = mesh();
                m.lods.clear();
                m
            }),
        ] {
            damage.write(&path).unwrap();
            assert!(BakedMesh::read(&path).is_err(), "{why} was accepted");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn baked_files_round_trip_and_reject_damage() {
        let dir = std::env::temp_dir().join(format!("feather_bake_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let baked = BakedTexture {
            kind: TexKind::Color,
            width: 8,
            height: 4,
            levels: (0..4)
                .map(|l| vec![l as u8; bc7_level_size(8, 4, l)])
                .collect(),
        };
        let path = baked_path(&dir, 0x1234);
        for kind in [TexKind::Color, TexKind::Data] {
            let b = BakedTexture {
                kind,
                ..baked.clone()
            };
            b.write(&path).unwrap();
            assert_eq!(BakedTexture::read(&path).unwrap(), b);
        }
        // An unknown kind tag is rejected (it's the u32 after magic + version).
        baked.write(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[8] = 9;
        std::fs::write(&path, &bytes).unwrap();
        assert!(BakedTexture::read(&path).is_err());
        baked.write(&path).unwrap();

        // A truncated file is rejected, not trusted.
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(BakedTexture::read(&path).is_err());
        // So is a level whose size doesn't match its dimensions.
        let wrong = BakedTexture {
            levels: vec![vec![0; 3]],
            ..baked
        };
        wrong.write(&path).unwrap();
        assert!(BakedTexture::read(&path).is_err());
    }

    #[test]
    fn open_sky_weights_like_the_shaders_sky_irradiance() {
        // sky_irradiance(n) gives the sky ½ + ½n.y of its light; the open
        // volume must reproduce that for every normal, or baking a level
        // would change how its open ground shades.
        for i in 0..64 {
            let a = i as f32 * 0.7;
            let n = Vec3::new(
                a.cos() * (a * 0.3).sin(),
                (a * 0.3).cos(),
                a.sin() * (a * 0.3).sin(),
            )
            .normalize();
            let want = 0.5 + 0.5 * n.y;
            assert!((SkyVis::OPEN.weight(n) - want).abs() < 1e-6, "{n}");
        }
        assert_eq!(SkyVis::CLOSED.weight(Vec3::Y), 0.0);
    }

    #[test]
    fn sky_texels_encode_the_open_sky_exactly() {
        let decode = |t: [u8; 4]| SkyVis::decode(t.map(|v| v as f32 / 255.0));
        assert_eq!(SkyVis::OPEN.encode(), [255, 127, 255, 127]);
        let open = decode(SkyVis::OPEN.encode());
        assert!((open.w0 - 0.5).abs() < 1e-6 && (open.w - SkyVis::OPEN.w).length() < 1e-6);
        assert_eq!(decode(SkyVis::CLOSED.encode()), SkyVis::CLOSED);
        // Anything in range round-trips to within half a step.
        for i in 0..200 {
            let s = i as f32 / 199.0;
            let v = SkyVis {
                w0: 0.5 * s,
                w: Vec3::new(0.25 - 0.5 * s, 0.5 * (1.0 - s), -0.25 + 0.5 * s),
            };
            let back = decode(v.encode());
            assert!((back.w0 - v.w0).abs() <= 0.5 / 510.0 + 1e-6, "{v:?}");
            assert!(
                (back.w - v.w).abs().max_element() <= 0.5 / 508.0 + 1e-6,
                "{v:?}"
            );
        }
    }

    fn texel(v: SkyVis, f: SkyFree) -> [u8; 8] {
        let (a, b) = (v.encode(), f.encode());
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
    }

    fn volume() -> SkyVolume {
        // 2×1×1: open on the left, closed on the right.
        SkyVolume {
            origin: Vec3::new(-1.0, 0.0, 0.0),
            cell: 1.0,
            dims: [2, 1, 1],
            texels: vec![
                texel(SkyVis::OPEN, SkyFree::CLEAR),
                texel(SkyVis::CLOSED, SkyFree::CLEAR),
            ],
        }
    }

    #[test]
    fn sky_volumes_sample_like_a_clamped_linear_sampler() {
        let v = volume();
        let at = |x: f32| v.sample(Vec3::new(x, 0.5, 0.5)).w0;
        // Cell centres are exact, halfway is the mean, beyond the centres it
        // clamps to the edge cell, and y/z past a 1-cell axis don't matter.
        assert!((at(-0.5) - 0.5).abs() < 1e-6);
        assert!(at(0.5).abs() < 1e-6);
        assert!((at(0.0) - 0.25).abs() < 1e-6);
        assert!((at(-9.0) - 0.5).abs() < 1e-6 && at(9.0).abs() < 1e-6);
        assert!((v.sample(Vec3::new(-0.5, 7.0, -3.0)).w0 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_wall_between_cells_keeps_the_far_one_out() {
        // The open cell's view towards +x ends 0.3 cells out: a wall. Points
        // past it (plus the soft margin) see only the closed cell; points
        // before it still blend; the closed cell's side is unaffected.
        let mut v = volume();
        let walled = SkyFree {
            plus: Vec3::new(0.3, 1.0, 1.0),
            ..SkyFree::CLEAR
        };
        v.texels[0] = texel(SkyVis::OPEN, walled);
        let at = |v: &SkyVolume, x: f32| v.sample(Vec3::new(x, 0.5, 0.5)).w0;
        let plain = |x: f32| 0.5 * (0.5 - x);
        assert!(
            at(&v, 0.0).abs() < 1e-6,
            "halfway, past the wall: {}",
            at(&v, 0.0)
        );
        assert!(
            (at(&v, -0.3) - plain(-0.3)).abs() < 1e-3,
            "before it: {}",
            at(&v, -0.3)
        );
        assert!(at(&v, 0.4).abs() < 1e-6);
        // The soft margin: just past the wall, partly visible.
        let edge = at(&v, -0.5 + 0.3 + 0.05);
        assert!(edge > 0.0 && edge < plain(-0.15), "{edge}");
        // Seeing nothing falls back to the plain blend rather than dividing
        // by zero.
        v.texels[1] = texel(
            SkyVis::CLOSED,
            SkyFree {
                minus: Vec3::ZERO,
                ..SkyFree::CLEAR
            },
        );
        v.texels[0] = texel(
            SkyVis::OPEN,
            SkyFree {
                plus: Vec3::ZERO,
                ..SkyFree::CLEAR
            },
        );
        assert!((at(&v, 0.0) - 0.25).abs() < 1e-6, "{}", at(&v, 0.0));
    }

    #[test]
    fn free_distances_round_down_and_round_trip() {
        let f = SkyFree {
            plus: Vec3::new(1.0, 0.5, 0.0),
            minus: Vec3::new(0.99, 0.2, 1.0),
        };
        let b = SkyFree::decode(f.encode());
        for (got, want) in [(b.plus, f.plus), (b.minus, f.minus)] {
            assert!((got - want).max_element() <= 0.0);
            assert!((want - got).max_element() < 1.0 / 15.0 + 1e-6);
        }
        assert_eq!(SkyFree::decode(SkyFree::CLEAR.encode()), SkyFree::CLEAR);
        assert_eq!(f.encode()[3], 0);
    }

    #[test]
    fn sky_volumes_round_trip_and_reject_damage() {
        let dir = std::env::temp_dir().join(format!("feather_bake_sky_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = baked_sky_path(&dir, 0x5c);
        volume().write(&path).unwrap();
        assert_eq!(SkyVolume::read(&path).unwrap(), volume());

        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(SkyVolume::read(&path).is_err(), "truncated");
        let mut extra = bytes.clone();
        extra.push(0);
        std::fs::write(&path, &extra).unwrap();
        assert!(SkyVolume::read(&path).is_err(), "trailing bytes");
        let mut magic = bytes.clone();
        magic[0] = b'X';
        std::fs::write(&path, &magic).unwrap();
        assert!(SkyVolume::read(&path).is_err(), "bad magic");
        for (why, damage) in [
            ("zero cells", {
                let mut v = volume();
                v.dims = [0, 1, 1];
                v.texels.clear();
                v
            }),
            ("texels short of the dims", {
                let mut v = volume();
                v.dims = [3, 1, 1];
                v
            }),
            ("cell size 0", {
                let mut v = volume();
                v.cell = 0.0;
                v
            }),
            ("NaN origin", {
                let mut v = volume();
                v.origin.y = f32::NAN;
                v
            }),
        ] {
            damage.write(&path).unwrap();
            assert!(SkyVolume::read(&path).is_err(), "{why} was accepted");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sky_keys_cover_occluders_and_nothing_else() {
        use crate::{PrefabSpec, SceneNode};
        use glam::Mat4;
        let wall = MeshData::cube(1.0);
        let mut grass = MeshData::cube(0.5);
        grass.material.alpha_mode = AlphaMode::Mask(0.5);
        let node = |mesh, x: f32, shadow: Option<bool>| SceneNode {
            mesh: Some(mesh),
            transform: Mat4::from_translation(Vec3::new(x, 0.0, 0.0)),
            prefab: shadow.map(|s| PrefabSpec {
                id: "prop".into(),
                params: serde_json::json!({ "shadow": s }),
            }),
        };
        let scene = |nodes| SceneData {
            meshes: vec![wall.clone(), grass.clone()],
            nodes,
        };
        let base = sky_key(&scene(vec![node(0, 0.0, None)]));
        // Moving an occluder changes the key...
        assert_ne!(base, sky_key(&scene(vec![node(0, 1.0, None)])));
        // ...but adding a cutout or a shadowless node doesn't, and neither
        // does recolouring.
        assert_eq!(
            base,
            sky_key(&scene(vec![node(0, 0.0, None), node(1, 3.0, None)]))
        );
        assert_eq!(
            base,
            sky_key(&scene(vec![
                node(0, 0.0, Some(true)),
                node(0, 5.0, Some(false))
            ]))
        );
        let mut recoloured = scene(vec![node(0, 0.0, None)]);
        recoloured.meshes[0].material.base_color = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(base, sky_key(&recoloured));
        let mut two_sided = scene(vec![node(0, 0.0, None)]);
        two_sided.meshes[0].material.double_sided = true;
        assert_ne!(base, sky_key(&two_sided));
    }
}
