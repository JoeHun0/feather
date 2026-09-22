//! Mesh bake (§17): an optimised vertex order and a LOD chain, via
//! meshoptimizer.
//!
//! Every LOD indexes the *same* vertex array (simplification only drops
//! triangles and reuses existing vertices), so LODs cost index memory only.
//! LOD0 is the input triangles, reordered for the post-transform vertex cache;
//! each further level is simplified from LOD0 towards half the previous
//! level's triangles. Each level records its geometric error in mesh-local
//! units, which the runtime projects to screen pixels (or shadow texels) to
//! pick the coarsest level that still looks the same.

use feather_assets::bake::{BakedMesh, Lod, MIN_LOD_TRIS};
use feather_assets::{MeshData, Vertex};
use meshopt::{SimplifyOptions, VertexDataAdapter};

/// No level coarser than this many triangles.
const MIN_TRIS: usize = 64;
const MAX_LODS: usize = 8;
/// A level must remove at least this fraction of the previous level's
/// triangles, or the chain stops: the simplifier has stalled against the
/// error cap or the topology, and a near-duplicate level is pure memory.
const MIN_STEP_REDUCTION: f32 = 0.10;
/// Error cap for one simplification, relative to the mesh extent. Loose on
/// purpose: the runtime decides when a level is usable from its error; the
/// cap only stops the chain collapsing into garbage.
const MAX_RELATIVE_ERROR: f32 = 0.25;
/// Weight of normal differences against position error. Keeps simplification
/// from flattening shading creases it can't see in positions alone.
const NORMAL_WEIGHT: f32 = 0.5;

fn bytes<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// Bake one mesh. Never fails: a mesh the simplifier can't reduce simply ends
/// up with fewer LODs (at least LOD0).
pub fn bake_mesh(mesh: &MeshData) -> BakedMesh {
    let n = mesh.vertices.len();
    let adapter = VertexDataAdapter::new(bytes(&mesh.vertices), std::mem::size_of::<Vertex>(), 0)
        .expect("Vertex is a whole number of f32s with the position first");
    let mut lods = vec![Lod {
        error: 0.0,
        indices: meshopt::optimize_vertex_cache(&mesh.indices, n),
    }];

    if mesh.indices.len() / 3 >= MIN_LOD_TRIS {
        // Relative error × this = mesh-local units.
        let scale = meshopt::simplify_scale(&adapter);
        let normals: Vec<f32> = mesh.vertices.iter().flat_map(|v| v.normal).collect();
        // Passed straight to C, which reads one per vertex: never empty.
        let locks = vec![false; n];
        while lods.len() < MAX_LODS {
            let prev = lods.last().expect("LOD0 exists");
            let prev_tris = prev.indices.len() / 3;
            if prev_tris / 2 < MIN_TRIS {
                break;
            }
            let mut relative = 0.0f32;
            let simplified = meshopt::simplify_with_attributes_and_locks(
                &mesh.indices,
                &adapter,
                &normals,
                &[NORMAL_WEIGHT; 3],
                3 * std::mem::size_of::<f32>(),
                &locks,
                (prev_tris / 2) * 3,
                MAX_RELATIVE_ERROR,
                // Prune: drop disconnected parts (grass blades) smaller than
                // the error, which simplification alone can never merge.
                SimplifyOptions::Prune,
                Some(&mut relative),
            );
            let tris = simplified.len() / 3;
            if tris < MIN_TRIS || tris as f32 > prev_tris as f32 * (1.0 - MIN_STEP_REDUCTION) {
                break;
            }
            // Simplifying from LOD0 each time makes errors grow with the
            // target anyway; the max makes it a guarantee.
            let error = (relative * scale).max(prev.error);
            lods.push(Lod {
                error,
                indices: meshopt::optimize_vertex_cache(&simplified, n),
            });
        }
    }

    // One fetch-order pass over every LOD's indices, LOD0 first, since they
    // share the vertex array. It also drops vertices no LOD references.
    let mut all: Vec<u32> = lods
        .iter()
        .flat_map(|l| l.indices.iter().copied())
        .collect();
    let vertices = meshopt::optimize_vertex_fetch(&mut all, &mesh.vertices);
    let mut at = 0;
    for lod in &mut lods {
        let len = lod.indices.len();
        lod.indices.copy_from_slice(&all[at..at + len]);
        at += len;
    }
    BakedMesh { vertices, lods }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Triangles as position triples, rotated to a canonical start (winding
    /// kept) and sorted: equal iff the same triangles, however reordered.
    fn triangle_set(vertices: &[Vertex], indices: &[u32]) -> Vec<[[u32; 3]; 3]> {
        let key = |i: u32| vertices[i as usize].pos.map(f32::to_bits);
        let mut tris: Vec<[[u32; 3]; 3]> = indices
            .chunks_exact(3)
            .map(|t| {
                let t = [key(t[0]), key(t[1]), key(t[2])];
                let r = (0..3).min_by_key(|&k| t[k]).unwrap();
                [t[r], t[(r + 1) % 3], t[(r + 2) % 3]]
            })
            .collect();
        tris.sort();
        tris
    }

    #[test]
    fn sphere_gets_a_valid_lod_chain() {
        let sphere = MeshData::uv_sphere(32, 64, 1.0); // 4032 triangles
        let baked = bake_mesh(&sphere);
        let counts: Vec<usize> = baked.lods.iter().map(|l| l.indices.len() / 3).collect();
        let errors: Vec<f32> = baked.lods.iter().map(|l| l.error).collect();
        println!("tris per LOD {counts:?}, errors {errors:?}");
        assert!(
            baked.lods.len() >= 4,
            "a 4k-triangle sphere should reduce well"
        );
        assert_eq!(errors[0], 0.0);
        for w in baked.lods.windows(2) {
            assert!(w[1].indices.len() < w[0].indices.len(), "{counts:?}");
            assert!(w[1].error >= w[0].error, "{errors:?}");
        }
        // Coarser levels must actually deviate a little, but stay under the cap.
        assert!(errors[1] > 0.0 && *errors.last().unwrap() <= MAX_RELATIVE_ERROR * 2.0 + 1e-6);
        for lod in &baked.lods {
            assert!(lod
                .indices
                .iter()
                .all(|&i| (i as usize) < baked.vertices.len()));
        }
        // LOD0 is the input, only reordered.
        assert_eq!(
            triangle_set(&baked.vertices, &baked.lods[0].indices),
            triangle_set(&sphere.vertices, &sphere.indices),
        );
    }

    #[test]
    fn small_meshes_keep_lod0_only() {
        let cube = MeshData::cube(1.0);
        let baked = bake_mesh(&cube);
        assert_eq!(baked.lods.len(), 1);
        assert_eq!(
            triangle_set(&baked.vertices, &baked.lods[0].indices),
            triangle_set(&cube.vertices, &cube.indices),
        );
    }

    /// Disconnected parts of varying size, like grass blades: simplification
    /// can't merge them (each flat-shaded cube face is its own component), so
    /// only Prune reduces them, dropping the smallest first. (Identical parts
    /// would all go at once, which the bake rejects as an empty LOD.)
    #[test]
    fn many_small_parts_still_reduce() {
        let one = MeshData::cube(1.0);
        let mut m = MeshData::cube(0.0);
        m.vertices.clear();
        m.indices.clear();
        for i in 0..200 {
            let size = 0.005 + 0.05 * (i as f32 / 200.0);
            let base = m.vertices.len() as u32;
            m.vertices.extend(one.vertices.iter().map(|v| {
                let mut v = *v;
                v.pos = v.pos.map(|c| c * size);
                v.pos[0] += (i % 20) as f32 * 0.1;
                v.pos[2] += (i / 20) as f32 * 0.1;
                v
            }));
            m.indices.extend(one.indices.iter().map(|i| i + base));
        }
        let baked = bake_mesh(&m);
        let counts: Vec<usize> = baked.lods.iter().map(|l| l.indices.len() / 3).collect();
        println!("200 cubes: tris per LOD {counts:?}");
        assert!(baked.lods.len() >= 2, "{counts:?}");
    }
}
