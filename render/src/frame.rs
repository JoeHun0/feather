//! The frame's passes, in order (§10): which ones run, and what each reads
//! and writes. gfx executes the list (`Renderer::draw_frame`) and derives
//! every barrier from these uses; the recorders are the app's, since that's
//! where the frame's data lives.
//!
//! A new pass is an entry here with its uses, and a recorder. Nobody writes
//! its barriers, nor the ones the passes after it need.

use feather_gfx::passes::{slot, Kind, Load, Pass, Record, Res, Sampled, Stage, Use};

/// What decides which passes run and which images they use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameOpts {
    /// The geometry is multisampled: later passes read its resolves.
    pub msaa: bool,
    /// TAA (§13) runs, and bloom, exposure and the tonemap read its output.
    pub taa: bool,
    /// The tonemap writes the LDR image, and FXAA the swapchain.
    pub fxaa: bool,
    /// Something casts: one rendering per cascade. Otherwise one layered
    /// rendering clears them all, so every receiver reads as lit.
    pub shadow_casters: bool,
}

/// The app's recorders, one per pass. `taa` and `aa` only run when their
/// pass does.
pub struct Recorders<'a> {
    pub shadow: Record<'a>,
    pub cluster: Record<'a>,
    pub prepass: Record<'a>,
    pub ao: Record<'a>,
    pub main: Record<'a>,
    pub taa: Record<'a>,
    pub bloom: Record<'a>,
    pub exposure: Record<'a>,
    pub post: Record<'a>,
    pub aa: Record<'a>,
    pub ui: Record<'a>,
}

/// The frame, in order.
pub fn frame_passes(o: FrameOpts, r: Recorders<'_>) -> Vec<Pass<'_>> {
    use Res::*;
    let sample = |stage, layout| Use::Sample { stage, layout };
    // The single-sample images later passes read: the resolves under MSAA.
    let (scene, depth) = if o.msaa {
        (HdrResolve, DepthResolve)
    } else {
        (Hdr, Depth)
    };
    // What bloom, exposure and the tonemap read: TAA's output when it runs.
    let image = if o.taa { TaaOut } else { scene };
    let pass = |name, kind, uses, timer: (u32, u32), record| Pass {
        name,
        kind,
        uses,
        timer: (Some(timer.0), Some(timer.1)),
        record: Some(record),
    };
    let graphics = Kind::Graphics { per_layer: false };
    let mut passes = vec![
        // The sun's cascades (§11). It also uploads the frame's buffers.
        pass(
            "shadow",
            Kind::Graphics {
                per_layer: o.shadow_casters,
            },
            vec![(
                Shadow,
                Use::Depth {
                    load: Load::Clear,
                    write: true,
                    resolve: None,
                },
            )],
            slot::SHADOW,
            r.shadow,
        ),
        // The light clusters (§12), after the shadow pass's upload.
        pass(
            "cluster",
            Kind::Compute,
            vec![(Clusters, Use::BufferWrite)],
            slot::CLUSTER,
            r.cluster,
        ),
        // The depth prepass (§10), on its own so GTAO can read it.
        pass(
            "prepass",
            graphics,
            vec![(
                Depth,
                Use::Depth {
                    load: Load::Clear,
                    write: true,
                    resolve: o.msaa.then_some(DepthResolve),
                },
            )],
            slot::PREPASS,
            r.prepass,
        ),
        // GTAO (§13). Its targets are transitioned even when it's off: the
        // main pass's descriptors name their layout.
        pass(
            "gtao",
            Kind::Compute,
            vec![
                (depth, sample(Stage::Compute, Sampled::DepthReadOnly)),
                (AoRaw, Use::Storage { discard: true }),
                (AoHalf, Use::Storage { discard: true }),
                (AoDepth, Use::Storage { discard: true }),
            ],
            slot::AO,
            r.ao,
        ),
        // The lit opaque pass and the sky, over the prepass's depth, which
        // it tests and never writes.
        pass(
            "main",
            graphics,
            vec![
                (
                    Hdr,
                    Use::Color {
                        load: Load::Clear,
                        resolve: o.msaa.then_some(HdrResolve),
                    },
                ),
                (
                    Depth,
                    Use::Depth {
                        load: Load::Load,
                        write: false,
                        resolve: None,
                    },
                ),
                (Shadow, sample(Stage::Fragment, Sampled::ReadOnly)),
                (AoHalf, sample(Stage::Fragment, Sampled::General)),
                (AoDepth, sample(Stage::Fragment, Sampled::General)),
                (
                    Clusters,
                    Use::BufferRead {
                        stage: Stage::Fragment,
                    },
                ),
            ],
            slot::MAIN,
            r.main,
        ),
    ];
    if o.taa {
        passes.push(pass(
            "taa",
            Kind::Compute,
            vec![
                (scene, sample(Stage::Compute, Sampled::ReadOnly)),
                (depth, sample(Stage::Compute, Sampled::DepthReadOnly)),
                (TaaPrev, sample(Stage::Compute, Sampled::ReadOnly)),
                (TaaOut, Use::Storage { discard: true }),
            ],
            slot::TAA,
            r.taa,
        ));
    }
    passes.extend([
        // Bloom (§13): its chain is transitioned even when it's off, since
        // the tonemap's descriptor names its layout.
        pass(
            "bloom",
            Kind::Compute,
            vec![
                (image, sample(Stage::Compute, Sampled::ReadOnly)),
                (Bloom, Use::Storage { discard: true }),
            ],
            slot::BLOOM,
            r.bloom,
        ),
        // Auto-exposure's metering (§13); it syncs its own buffers.
        pass(
            "exposure",
            Kind::Compute,
            vec![(image, sample(Stage::Compute, Sampled::ReadOnly))],
            slot::EXPOSURE,
            r.exposure,
        ),
    ]);
    // The tonemap covers every pixel, so what was there doesn't matter.
    let target = if o.fxaa { Ldr } else { Swapchain };
    let mut tonemap = pass(
        "tonemap",
        graphics,
        vec![
            (
                target,
                Use::Color {
                    load: Load::DontCare,
                    resolve: None,
                },
            ),
            (image, sample(Stage::Fragment, Sampled::ReadOnly)),
            (Bloom, sample(Stage::Fragment, Sampled::General)),
        ],
        slot::POST,
        r.post,
    );
    if o.fxaa {
        // FXAA ends the post chain's span.
        tonemap.timer.1 = None;
        passes.push(tonemap);
        let mut fxaa = pass(
            "fxaa",
            graphics,
            vec![
                (
                    Swapchain,
                    Use::Color {
                        load: Load::DontCare,
                        resolve: None,
                    },
                ),
                (Ldr, sample(Stage::Fragment, Sampled::ReadOnly)),
            ],
            slot::POST,
            r.aa,
        );
        fxaa.timer.0 = None;
        passes.push(fxaa);
    } else {
        passes.push(tonemap);
    }
    // The overlay (§19), blended over the finished frame.
    let mut ui = pass(
        "ui",
        graphics,
        vec![(
            Swapchain,
            Use::Color {
                load: Load::Load,
                resolve: None,
            },
        )],
        (0, 0),
        r.ui,
    );
    ui.timer = (None, None);
    passes.push(ui);
    passes
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;
    use feather_gfx::passes::{needs, Barrier, State, Tracker};

    fn noop<'a>() -> Record<'a> {
        Box::new(|_, _, _, _| {})
    }

    fn recorders<'a>() -> Recorders<'a> {
        Recorders {
            shadow: noop(),
            cluster: noop(),
            prepass: noop(),
            ao: noop(),
            main: noop(),
            taa: noop(),
            bloom: noop(),
            exposure: noop(),
            post: noop(),
            aa: noop(),
            ui: noop(),
        }
    }

    fn every_opts() -> Vec<FrameOpts> {
        let mut v = Vec::new();
        for bits in 0..16u32 {
            let b = |i: u32| bits & (1 << i) != 0;
            v.push(FrameOpts {
                msaa: b(0),
                taa: b(1),
                fxaa: b(2),
                shadow_casters: b(3),
            });
        }
        v
    }

    /// Plan one frame as gfx does: the swapchain freshly acquired, the
    /// clusters this frame's own, then the passes and the present.
    fn plan(t: &mut Tracker<Res>, o: FrameOpts) -> Vec<Vec<Barrier<Res>>> {
        t.set(Res::Swapchain, State::ACQUIRED);
        t.set(Res::Clusters, State::FRESH);
        let mut list: Vec<Vec<(Res, _)>> = frame_passes(o, recorders())
            .iter()
            .map(|p| p.uses.iter().flat_map(|&(r, u)| needs(r, u)).collect())
            .collect();
        list.push(needs(Res::Swapchain, Use::Present));
        t.plan(&list)
    }

    /// In a steady frame every transition waits for something: the frame
    /// before's last use of that image (§21's rule), or the acquire for the
    /// swapchain. Only a buffer's first write needs nothing, and emits
    /// nothing. Every option combination, as each changes the list.
    #[test]
    fn in_a_steady_frame_every_barrier_waits_for_a_previous_use() {
        for o in every_opts() {
            let mut t = Tracker::new();
            plan(&mut t, o);
            let second = plan(&mut t, o);
            for (i, pass) in second.iter().enumerate() {
                for b in pass {
                    assert_ne!(
                        b.src_stages,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        "{o:?}, pass {i}: {b:?}"
                    );
                }
            }
        }
    }

    /// Every pass's images are ones the frame has: no resolve at 1x, and
    /// nothing that reads the multisampled images directly under MSAA
    /// except as an attachment.
    #[test]
    fn single_sample_reads_take_the_resolves_under_msaa() {
        for o in every_opts() {
            for p in frame_passes(o, recorders()) {
                for &(r, u) in &p.uses {
                    let resolves = [Some(Res::HdrResolve), Some(Res::DepthResolve)];
                    if !o.msaa {
                        assert!(!resolves.contains(&Some(r)), "{o:?} {}: {r:?}", p.name);
                        if let Use::Color { resolve, .. } | Use::Depth { resolve, .. } = u {
                            assert_eq!(resolve, None, "{o:?} {}", p.name);
                        }
                    } else if matches!(r, Res::Hdr | Res::Depth) {
                        assert!(
                            matches!(u, Use::Color { .. } | Use::Depth { .. }),
                            "{o:?} {} samples the multisampled {r:?}",
                            p.name
                        );
                    }
                }
            }
        }
    }

    /// The tonemap reads what TAA wrote when TAA runs, and the swapchain
    /// or the LDR image as FXAA says.
    #[test]
    fn the_post_chain_reads_taa_and_writes_where_fxaa_says() {
        for o in every_opts() {
            let passes = frame_passes(o, recorders());
            let tonemap = passes.iter().find(|p| p.name == "tonemap").unwrap();
            let reads = if o.taa {
                Res::TaaOut
            } else if o.msaa {
                Res::HdrResolve
            } else {
                Res::Hdr
            };
            assert!(tonemap.uses.iter().any(|&(r, _)| r == reads), "{o:?}");
            let target = if o.fxaa { Res::Ldr } else { Res::Swapchain };
            assert!(
                matches!(tonemap.uses[0], (r, Use::Color { .. }) if r == target),
                "{o:?}"
            );
            assert_eq!(passes.iter().any(|p| p.name == "taa"), o.taa);
            assert_eq!(passes.iter().any(|p| p.name == "fxaa"), o.fxaa);
        }
    }
}
