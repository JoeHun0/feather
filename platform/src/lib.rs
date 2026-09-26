//! Platform: windowing, event loop, input collection, action mapping.

pub use winit;

/// Default window size, in logical pixels (so it scales with the desktop's
/// HiDPI factor). The window stays resizable; `--bench` overrides this with a
/// fixed physical size so timings are comparable.
pub const DEFAULT_WINDOW_SIZE: (u32, u32) = (640, 480);

/// Default attributes for the engine's main window.
pub fn window_attributes(title: &str) -> winit::window::WindowAttributes {
    let (w, h) = DEFAULT_WINDOW_SIZE;
    winit::window::Window::default_attributes()
        .with_title(title)
        .with_inner_size(winit::dpi::LogicalSize::new(w, h))
}
