//! The frame as a list of passes (§10): what each pass reads and writes, and
//! the barriers that follow from that.
//!
//! `render` decides which passes run and in what order (`render::frame`);
//! gfx owns the images they name and executes the list
//! (`Renderer::draw_frame`). Nobody writes a barrier: a [`Tracker`]
//! remembers, per image, its layout, its last write and which stages have
//! read it since, and before each pass plans the barriers the pass's uses
//! need. The state outlives the frame, so a frame's first use of an image
//! waits for the previous frame's last one (§21's rule for images shared by
//! the frames in flight) without anyone writing that barrier either.

use std::collections::HashMap;
use std::hash::Hash;

use ash::vk;

/// What the frame's passes name. The images are gfx's (it recreates them
/// with the window and the sample count); `Clusters` stands for a buffer
/// the mesh renderer owns, one per frame in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Res {
    /// The sun's cascaded shadow map (§11), one layer per cascade.
    Shadow,
    /// The geometry depth buffer, multisampled under MSAA.
    Depth,
    /// Its sample-0 resolve, under MSAA only.
    DepthResolve,
    /// The lit HDR image, multisampled under MSAA.
    Hdr,
    /// Its average resolve, under MSAA only.
    HdrResolve,
    /// GTAO's targets (§13): raw and denoised half-resolution visibility,
    /// and the depth levels.
    AoRaw,
    AoHalf,
    AoDepth,
    /// TAA's history (§13): last frame's output, and this frame's.
    TaaPrev,
    TaaOut,
    /// The bloom mip chain (§13).
    Bloom,
    /// The tonemap's LDR output, when FXAA follows it.
    Ldr,
    /// This frame's swapchain image.
    Swapchain,
    /// The light clusters (§12): compute writes them, the main pass reads.
    Clusters,
}

/// An attachment's load op. `Clear` and `DontCare` drop what was there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Load {
    Clear,
    Load,
    DontCare,
}

/// Which shaders read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Compute,
    Fragment,
}

impl Stage {
    fn flags(self) -> vk::PipelineStageFlags {
        match self {
            Stage::Compute => vk::PipelineStageFlags::COMPUTE_SHADER,
            Stage::Fragment => vk::PipelineStageFlags::FRAGMENT_SHADER,
        }
    }
}

/// The layout a sampled image's descriptor names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampled {
    /// `SHADER_READ_ONLY_OPTIMAL`.
    ReadOnly,
    /// `DEPTH_STENCIL_READ_ONLY_OPTIMAL`: a depth buffer that may also be
    /// the pass's read-only depth attachment.
    DepthReadOnly,
    /// `GENERAL`: an image compute writes as storage (bloom, GTAO).
    General,
}

/// How a pass uses one resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Use {
    /// A colour attachment, in declaration order. `resolve` receives the
    /// average of its samples when the rendering ends.
    Color {
        load: Load,
        resolve: Option<Res>,
    },
    /// The depth attachment. `write: false` only tests it, in the read-only
    /// layout; `resolve` receives sample 0 when the rendering ends.
    Depth {
        load: Load,
        write: bool,
        resolve: Option<Res>,
    },
    /// Sampled by `stage`'s shaders.
    Sample {
        stage: Stage,
        layout: Sampled,
    },
    /// A compute storage image; `discard` when the pass overwrites all of it.
    Storage {
        discard: bool,
    },
    /// Copied from and to (`Kind::Copy`).
    CopySrc,
    CopyDst,
    /// A buffer compute writes, and one `stage` reads.
    BufferWrite,
    BufferRead {
        stage: Stage,
    },
    /// Handed to the presentation engine: the swapchain's last use.
    Present,
}

/// What a pass does with the command buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A rendering over its `Color`/`Depth` uses. `per_layer` renders each
    /// layer of a layered attachment in turn (the shadow cascades), calling
    /// the recorder with the layer's index; otherwise one rendering covers
    /// every layer.
    Graphics { per_layer: bool },
    /// Compute, or anything else outside a rendering.
    Compute,
    /// A whole-image copy, recorded by gfx itself.
    Copy { src: Res, dst: Res },
}

/// Records a pass: `(cmd, extent, frame, index)`, where `index` is the
/// layer of a per-layer pass and 0 otherwise.
pub type Record<'a> = Box<dyn FnMut(vk::CommandBuffer, vk::Extent2D, usize, usize) + 'a>;

/// One pass of the frame.
pub struct Pass<'a> {
    pub name: &'static str,
    pub kind: Kind,
    pub uses: Vec<(Res, Use)>,
    /// Timestamp slots written before the pass's barriers and after its
    /// work (see [`slot`]).
    pub timer: (Option<u32>, Option<u32>),
    pub record: Option<Record<'a>>,
}

/// Timestamp slots (§21), in pairs: `GpuTimes` reads them back. Every slot
/// is written every frame (gfx stamps whatever no pass did at the end), or
/// the readback would wait on it forever.
pub mod slot {
    pub const SHADOW: (u32, u32) = (0, 1);
    pub const CLUSTER: (u32, u32) = (2, 3);
    pub const PREPASS: (u32, u32) = (4, 5);
    pub const AO: (u32, u32) = (6, 7);
    pub const MAIN: (u32, u32) = (8, 9);
    pub const TAA: (u32, u32) = (10, 11);
    pub const BLOOM: (u32, u32) = (12, 13);
    pub const EXPOSURE: (u32, u32) = (14, 15);
    /// The tonemap, and FXAA when it's on (it takes the end).
    pub const POST: (u32, u32) = (16, 17);
    pub const COUNT: u32 = 18;
}

const WRITES: vk::AccessFlags = vk::AccessFlags::from_raw(
    vk::AccessFlags::SHADER_WRITE.as_raw()
        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE.as_raw()
        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
        | vk::AccessFlags::TRANSFER_WRITE.as_raw(),
);

/// One resource's requirement in one pass: the layout it must be in, and
/// the stages and accesses that will touch it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Need {
    pub layout: vk::ImageLayout,
    pub stages: vk::PipelineStageFlags,
    pub access: vk::AccessFlags,
    /// The pass doesn't care what was there: a transition may start from
    /// `UNDEFINED`.
    pub discard: bool,
}

impl Need {
    fn writes(&self) -> bool {
        self.access.intersects(WRITES)
    }
}

/// What `res` needs for `u`: one entry, or two for an attachment that
/// resolves into another image.
pub fn needs(res: Res, u: Use) -> Vec<(Res, Need)> {
    use vk::AccessFlags as A;
    use vk::ImageLayout as L;
    use vk::PipelineStageFlags as S;
    let need = |layout, stages, access, discard| Need {
        layout,
        stages,
        access,
        discard,
    };
    let dropped = |load| load != Load::Load;
    match u {
        Use::Color { load, resolve } => {
            let access = match load {
                Load::Load => A::COLOR_ATTACHMENT_READ | A::COLOR_ATTACHMENT_WRITE,
                _ => A::COLOR_ATTACHMENT_WRITE,
            };
            let mut v = vec![(
                res,
                need(
                    L::COLOR_ATTACHMENT_OPTIMAL,
                    S::COLOR_ATTACHMENT_OUTPUT,
                    access,
                    dropped(load),
                ),
            )];
            if let Some(r) = resolve {
                v.push((
                    r,
                    need(
                        L::COLOR_ATTACHMENT_OPTIMAL,
                        S::COLOR_ATTACHMENT_OUTPUT,
                        A::COLOR_ATTACHMENT_WRITE,
                        true,
                    ),
                ));
            }
            v
        }
        Use::Depth {
            load,
            write,
            resolve,
        } => {
            let tests = S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS;
            let mut v = vec![(
                res,
                if write {
                    need(
                        L::DEPTH_ATTACHMENT_OPTIMAL,
                        tests,
                        A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dropped(load),
                    )
                } else {
                    need(
                        L::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                        tests,
                        A::DEPTH_STENCIL_ATTACHMENT_READ,
                        false,
                    )
                },
            )];
            // A depth resolve writes in the colour-output stage, as colour
            // resolves do.
            if let Some(r) = resolve {
                v.push((
                    r,
                    need(
                        L::DEPTH_ATTACHMENT_OPTIMAL,
                        S::EARLY_FRAGMENT_TESTS
                            | S::LATE_FRAGMENT_TESTS
                            | S::COLOR_ATTACHMENT_OUTPUT,
                        A::DEPTH_STENCIL_ATTACHMENT_WRITE | A::COLOR_ATTACHMENT_WRITE,
                        true,
                    ),
                ));
            }
            v
        }
        Use::Sample { stage, layout } => {
            let layout = match layout {
                Sampled::ReadOnly => L::SHADER_READ_ONLY_OPTIMAL,
                Sampled::DepthReadOnly => L::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                Sampled::General => L::GENERAL,
            };
            vec![(res, need(layout, stage.flags(), A::SHADER_READ, false))]
        }
        Use::Storage { discard } => vec![(
            res,
            need(
                L::GENERAL,
                S::COMPUTE_SHADER,
                A::SHADER_READ | A::SHADER_WRITE,
                discard,
            ),
        )],
        Use::CopySrc => vec![(
            res,
            need(
                L::TRANSFER_SRC_OPTIMAL,
                S::TRANSFER,
                A::TRANSFER_READ,
                false,
            ),
        )],
        Use::CopyDst => vec![(
            res,
            need(
                L::TRANSFER_DST_OPTIMAL,
                S::TRANSFER,
                A::TRANSFER_WRITE,
                true,
            ),
        )],
        Use::BufferWrite => vec![(
            res,
            need(L::UNDEFINED, S::COMPUTE_SHADER, A::SHADER_WRITE, true),
        )],
        Use::BufferRead { stage } => {
            vec![(
                res,
                need(L::UNDEFINED, stage.flags(), A::SHADER_READ, false),
            )]
        }
        Use::Present => vec![(
            res,
            need(L::PRESENT_SRC_KHR, S::BOTTOM_OF_PIPE, A::empty(), false),
        )],
    }
}

/// What the tracker knows about one resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub layout: vk::ImageLayout,
    /// The last write (or layout transition): later uses wait for these
    /// stages...
    pub write_stages: vk::PipelineStageFlags,
    /// ...and need these accesses made visible.
    pub write_access: vk::AccessFlags,
    /// Stages that have read since the last write: the next write waits
    /// for them too.
    pub read_stages: vk::PipelineStageFlags,
    /// Where the last write is already visible: reads there need nothing.
    pub visible_stages: vk::PipelineStageFlags,
    pub visible_access: vk::AccessFlags,
}

impl State {
    /// Never used, or its contents lost (a new image).
    pub const FRESH: State = State {
        layout: vk::ImageLayout::UNDEFINED,
        write_stages: vk::PipelineStageFlags::empty(),
        write_access: vk::AccessFlags::empty(),
        read_stages: vk::PipelineStageFlags::empty(),
        visible_stages: vk::PipelineStageFlags::empty(),
        visible_access: vk::AccessFlags::empty(),
    };

    /// A swapchain image just acquired: nothing in it, and first usable in
    /// the stage the submit waits on the acquire semaphore at.
    pub const ACQUIRED: State = State {
        write_stages: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ..State::FRESH
    };
}

/// One planned barrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Barrier<K> {
    pub key: K,
    pub old: vk::ImageLayout,
    pub new: vk::ImageLayout,
    pub src_stages: vk::PipelineStageFlags,
    pub src_access: vk::AccessFlags,
    pub dst_stages: vk::PipelineStageFlags,
    pub dst_access: vk::AccessFlags,
}

/// Remembers each resource's [`State`] across passes and frames, and plans
/// the barriers a list of passes needs. Pure: gfx keys it by `vk::Image`
/// and turns the plan into `vk::` barriers; the tests key it by name.
#[derive(Debug, Default)]
pub struct Tracker<K> {
    states: HashMap<K, State>,
}

impl<K: Copy + Eq + Hash> Tracker<K> {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }

    /// Forget everything: the images were recreated.
    pub fn reset(&mut self) {
        self.states.clear();
    }

    /// Set one resource's state: a swapchain image just acquired, or a
    /// buffer at the start of its frame.
    pub fn set(&mut self, key: K, state: State) {
        self.states.insert(key, state);
    }

    pub fn state(&self, key: K) -> State {
        self.states.get(&key).copied().unwrap_or(State::FRESH)
    }

    /// Plan the barriers before each of `passes` (each a list of needs),
    /// updating the states as if they ran. A resource named twice in one
    /// pass needs one layout for both (a read-only depth attachment that is
    /// also sampled), and its needs merge.
    pub fn plan(&mut self, passes: &[Vec<(K, Need)>]) -> Vec<Vec<Barrier<K>>> {
        let merged: Vec<Vec<(K, Need)>> = passes.iter().map(|p| merge(p)).collect();
        let mut out = Vec::with_capacity(merged.len());
        for (i, pass) in merged.iter().enumerate() {
            let mut barriers = Vec::new();
            for &(key, need) in pass {
                if let Some(b) = self.step(key, need, &merged[i + 1..]) {
                    barriers.push(b);
                }
            }
            out.push(barriers);
        }
        out
    }

    fn step(&mut self, key: K, need: Need, later: &[Vec<(K, Need)>]) -> Option<Barrier<K>> {
        let st = self.states.entry(key).or_insert(State::FRESH);
        let prev = *st;
        if need.writes() || prev.layout != need.layout {
            // Wait for the last write and every read since; start from
            // UNDEFINED when the contents don't matter.
            let src_stages = prev.write_stages | prev.read_stages;
            let old = if need.discard {
                vk::ImageLayout::UNDEFINED
            } else {
                prev.layout
            };
            // A transition made for a reader is made for every later read in
            // the same layout, so those need no barrier of their own.
            let (dst_stages, dst_access) = if need.writes() {
                (need.stages, need.access)
            } else {
                readers_ahead(key, need, later)
            };
            *st = if need.writes() {
                State {
                    layout: need.layout,
                    write_stages: need.stages,
                    write_access: need.access & WRITES,
                    ..State::FRESH
                }
            } else {
                // The transition is the last "write": later uses order after
                // it, through the stages it was made for.
                State {
                    layout: need.layout,
                    write_stages: dst_stages,
                    write_access: vk::AccessFlags::empty(),
                    read_stages: dst_stages,
                    visible_stages: dst_stages,
                    visible_access: dst_access,
                }
            };
            // Nothing to wait for and nothing to transition (a buffer's
            // first write in its frame).
            if src_stages.is_empty() && old == need.layout {
                return None;
            }
            return Some(Barrier {
                key,
                old,
                new: need.layout,
                src_stages: if src_stages.is_empty() {
                    vk::PipelineStageFlags::TOP_OF_PIPE
                } else {
                    src_stages
                },
                src_access: prev.write_access,
                dst_stages,
                dst_access,
            });
        }
        // A read in the current layout: make the last write visible here,
        // unless a barrier already did.
        let covered =
            prev.visible_stages.contains(need.stages) && prev.visible_access.contains(need.access);
        let barrier = (!prev.write_stages.is_empty() && !covered).then(|| {
            let (dst_stages, dst_access) = readers_ahead(key, need, later);
            st.visible_stages |= dst_stages;
            st.visible_access |= dst_access;
            Barrier {
                key,
                old: prev.layout,
                new: prev.layout,
                src_stages: prev.write_stages,
                src_access: prev.write_access,
                dst_stages,
                dst_access,
            }
        });
        st.read_stages |= need.stages;
        barrier
    }
}

/// One pass's needs with each resource once: a second use of the same one
/// must want the same layout, and adds its stages and accesses.
fn merge<K: Copy + Eq>(pass: &[(K, Need)]) -> Vec<(K, Need)> {
    let mut out: Vec<(K, Need)> = Vec::with_capacity(pass.len());
    for &(key, n) in pass {
        match out.iter_mut().find(|(k, _)| *k == key) {
            Some((_, m)) => {
                assert_eq!(
                    m.layout, n.layout,
                    "one resource in two layouts in one pass"
                );
                m.stages |= n.stages;
                m.access |= n.access;
                m.discard &= n.discard;
            }
            None => out.push((key, n)),
        }
    }
    out
}

/// `need` plus every later read of `key` in the same layout, up to its next
/// write or change of layout: what one barrier can serve.
fn readers_ahead<K: Copy + Eq>(
    key: K,
    need: Need,
    later: &[Vec<(K, Need)>],
) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    let (mut stages, mut access) = (need.stages, need.access);
    for pass in later {
        if let Some(&(_, n)) = pass.iter().find(|(k, _)| *k == key) {
            if n.layout != need.layout || n.writes() {
                break;
            }
            stages |= n.stages;
            access |= n.access;
        }
    }
    (stages, access)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::ImageLayout as L;
    use vk::PipelineStageFlags as S;

    /// Plan one frame of `passes`, each a list of uses.
    fn frame(t: &mut Tracker<Res>, passes: &[&[(Res, Use)]]) -> Vec<Vec<Barrier<Res>>> {
        let needs: Vec<Vec<(Res, Need)>> = passes
            .iter()
            .map(|p| p.iter().flat_map(|&(r, u)| needs(r, u)).collect())
            .collect();
        t.plan(&needs)
    }

    const CLEAR_DEPTH: (Res, Use) = (
        Res::Depth,
        Use::Depth {
            load: Load::Clear,
            write: true,
            resolve: None,
        },
    );
    const AO_READS_DEPTH: (Res, Use) = (
        Res::Depth,
        Use::Sample {
            stage: Stage::Compute,
            layout: Sampled::DepthReadOnly,
        },
    );

    /// §21's rule, derived: the second frame's clear waits for the first
    /// frame's compute read. (Its depth writes are ordered before that read
    /// by the read's own barrier, so waiting for the read covers them.)
    #[test]
    fn a_clear_waits_for_last_frames_readers() {
        let mut t = Tracker::new();
        frame(&mut t, &[&[CLEAR_DEPTH], &[AO_READS_DEPTH]]);
        let second = frame(&mut t, &[&[CLEAR_DEPTH], &[AO_READS_DEPTH]]);
        let clear = second[0][0];
        assert_eq!(
            (clear.old, clear.new),
            (L::UNDEFINED, L::DEPTH_ATTACHMENT_OPTIMAL)
        );
        assert!(clear.src_stages.contains(S::COMPUTE_SHADER), "{clear:?}");
    }

    /// One transition serves every later read in its layout: the main
    /// pass's depth test and TAA's read ride on GTAO's barrier.
    #[test]
    fn later_reads_in_the_same_layout_ride_on_one_barrier() {
        let mut t = Tracker::new();
        let test_depth = (
            Res::Depth,
            Use::Depth {
                load: Load::Load,
                write: false,
                resolve: None,
            },
        );
        let taa_reads = (
            Res::Depth,
            Use::Sample {
                stage: Stage::Compute,
                layout: Sampled::DepthReadOnly,
            },
        );
        let b = frame(
            &mut t,
            &[
                &[CLEAR_DEPTH],
                &[AO_READS_DEPTH],
                &[test_depth],
                &[taa_reads],
            ],
        );
        assert_eq!(b[1].len(), 1);
        assert_eq!(b[1][0].new, L::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        assert_eq!(
            b[1][0].dst_stages,
            S::COMPUTE_SHADER | S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS
        );
        assert!(b[2].is_empty() && b[3].is_empty(), "{b:?}");
    }

    /// A write after reads waits for every reader, not only the last.
    #[test]
    fn a_write_waits_for_every_reader() {
        let mut t = Tracker::new();
        let write = (Res::Bloom, Use::Storage { discard: true });
        let read = |stage| {
            (
                Res::Bloom,
                Use::Sample {
                    stage,
                    layout: Sampled::General,
                },
            )
        };
        let b = frame(
            &mut t,
            &[
                &[write],
                &[read(Stage::Compute)],
                &[read(Stage::Fragment)],
                &[write],
            ],
        );
        // The two reads: one barrier, made for both.
        assert_eq!(b[1][0].dst_stages, S::COMPUTE_SHADER | S::FRAGMENT_SHADER);
        assert!(b[2].is_empty());
        let rewrite = b[3][0];
        assert!(
            rewrite
                .src_stages
                .contains(S::COMPUTE_SHADER | S::FRAGMENT_SHADER),
            "{rewrite:?}"
        );
        assert_eq!(rewrite.src_access, vk::AccessFlags::SHADER_WRITE);
    }

    /// A read-only depth attachment that is also sampled is one resource in
    /// one layout: one barrier covers both.
    #[test]
    fn a_tested_and_sampled_depth_merges() {
        let mut t = Tracker::new();
        let both: &[(Res, Use)] = &[
            (
                Res::Depth,
                Use::Depth {
                    load: Load::Load,
                    write: false,
                    resolve: None,
                },
            ),
            (
                Res::Depth,
                Use::Sample {
                    stage: Stage::Fragment,
                    layout: Sampled::DepthReadOnly,
                },
            ),
        ];
        let b = frame(&mut t, &[&[CLEAR_DEPTH], both]);
        assert_eq!(b[1].len(), 1);
        assert_eq!(
            b[1][0].dst_stages,
            S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS | S::FRAGMENT_SHADER
        );
    }

    #[test]
    #[should_panic(expected = "two layouts")]
    fn one_resource_in_two_layouts_in_one_pass_is_refused() {
        let mut t = Tracker::new();
        frame(&mut t, &[&[CLEAR_DEPTH, AO_READS_DEPTH]]);
    }

    /// A just-acquired swapchain image is first written in the stage the
    /// submit waits on the acquire at, so the transition chains after it.
    #[test]
    fn the_swapchain_chains_after_the_acquire() {
        let mut t = Tracker::new();
        t.set(Res::Swapchain, State::ACQUIRED);
        let post = (
            Res::Swapchain,
            Use::Color {
                load: Load::DontCare,
                resolve: None,
            },
        );
        let ui = (
            Res::Swapchain,
            Use::Color {
                load: Load::Load,
                resolve: None,
            },
        );
        let b = frame(&mut t, &[&[post], &[ui], &[(Res::Swapchain, Use::Present)]]);
        assert_eq!(b[0][0].src_stages, S::COLOR_ATTACHMENT_OUTPUT);
        assert_eq!(b[0][0].old, L::UNDEFINED);
        // The overlay blends over what the tonemap wrote.
        assert_eq!(b[1][0].src_access, vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        assert!(b[1][0]
            .dst_access
            .contains(vk::AccessFlags::COLOR_ATTACHMENT_READ));
        assert_eq!(b[2][0].new, L::PRESENT_SRC_KHR);
    }

    /// Recreated images start over: nothing to wait for, from UNDEFINED.
    #[test]
    fn a_reset_starts_from_nothing() {
        let mut t = Tracker::new();
        frame(&mut t, &[&[CLEAR_DEPTH], &[AO_READS_DEPTH]]);
        t.reset();
        let b = frame(&mut t, &[&[AO_READS_DEPTH]]);
        assert_eq!(
            (b[0][0].old, b[0][0].src_stages),
            (L::UNDEFINED, S::TOP_OF_PIPE)
        );
    }

    /// A resolve writes its target: a later read waits for the resolve.
    #[test]
    fn a_resolve_writes_its_target() {
        let mut t = Tracker::new();
        let main = (
            Res::Hdr,
            Use::Color {
                load: Load::Clear,
                resolve: Some(Res::HdrResolve),
            },
        );
        let taa = (
            Res::HdrResolve,
            Use::Sample {
                stage: Stage::Compute,
                layout: Sampled::ReadOnly,
            },
        );
        let b = frame(&mut t, &[&[main], &[taa]]);
        assert_eq!(b[0].len(), 2);
        assert_eq!(b[1][0].src_stages, S::COLOR_ATTACHMENT_OUTPUT);
        assert_eq!(b[1][0].new, L::SHADER_READ_ONLY_OPTIMAL);
    }

    /// A buffer's first write in its frame needs nothing; the read after it
    /// waits for it.
    #[test]
    fn a_buffer_read_waits_for_its_write() {
        let mut t = Tracker::new();
        let b = frame(
            &mut t,
            &[
                &[(Res::Clusters, Use::BufferWrite)],
                &[(
                    Res::Clusters,
                    Use::BufferRead {
                        stage: Stage::Fragment,
                    },
                )],
            ],
        );
        assert!(b[0].is_empty());
        assert_eq!(
            (b[1][0].src_stages, b[1][0].dst_stages),
            (S::COMPUTE_SHADER, S::FRAGMENT_SHADER)
        );
    }
}
