//! Gfx: low-level Vulkan — instance, device, swapchain, queues, allocation.

mod renderer;
pub use renderer::{Buffer, Image, MappedBuffer, Renderer, FRAMES_IN_FLIGHT};
