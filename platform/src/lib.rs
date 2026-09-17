//! Platform: windowing, event loop, input collection, action mapping.

pub use winit;

/// Default attributes for the engine's main window.
pub fn window_attributes(title: &str) -> winit::window::WindowAttributes {
    winit::window::Window::default_attributes().with_title(title)
}
