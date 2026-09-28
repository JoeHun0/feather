//! Render: frame graph, passes, pipelines, the RenderFrame consumer.

mod bloom;
mod environment;
mod fxaa;
mod mesh;
mod sky;
mod tonemap;
mod ui;
pub use bloom::{BloomPass, BLOOM_STRENGTH};
pub use environment::Environment;
pub use fxaa::FxaaPass;
pub use mesh::{
    CascadeSetup, ClusterView, FrameStats, GpuLight, InstanceData, MeshId, MeshRenderer, MAX_LIGHTS,
};
pub use sky::SkyPass;
pub use tonemap::TonemapPass;
pub use ui::UiPass;
