//! Milestone 0: a window that presents a cleared swapchain, driven by a
//! multi-threaded bevy_ecs schedule that ticks once per frame.

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ExecutorKind;
use feather_gfx::Renderer;
use feather_platform::winit;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

// ---- Example ECS data (moves into feather-game once real systems exist) ----

#[derive(Component)]
struct Position(glam::Vec3);

#[derive(Component)]
struct Velocity(glam::Vec3);

#[derive(Resource, Default)]
struct FrameCount(u64);

// integrate (writes Position, reads Velocity) and tick (writes FrameCount) touch
// disjoint data, so the multi-threaded executor may run them on different threads
// in the same frame.
fn integrate(mut q: Query<(&mut Position, &Velocity)>) {
    let dt = 1.0 / 60.0;
    for (mut pos, vel) in &mut q {
        pos.0 += vel.0 * dt;
    }
}

fn tick(mut frame: ResMut<FrameCount>) {
    frame.0 += 1;
}

// ---- App ----

struct App {
    // Declared before `window`: fields drop top-to-bottom, so the renderer (and
    // its VkSurfaceKHR) is torn down before the window it targets.
    renderer: Option<Renderer>,
    window: Option<Window>,
    world: World,
    schedule: Schedule,
}

impl App {
    fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(FrameCount::default());
        for i in 0..1000u32 {
            world.spawn((
                Position(glam::Vec3::ZERO),
                Velocity(glam::Vec3::new(i as f32 * 0.001, 0.0, 0.0)),
            ));
        }

        let mut schedule = Schedule::default();
        schedule.set_executor_kind(ExecutorKind::MultiThreaded);
        schedule.add_systems((integrate, tick));

        Self {
            renderer: None,
            window: None,
            world,
            schedule,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = event_loop
            .create_window(feather_platform::window_attributes("feather — milestone 0"))
            .expect("create window");
        let size = window.inner_size();
        self.renderer =
            Some(Renderer::new(&window, size.width, size.height).expect("create renderer"));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                self.schedule.run(&mut self.world);
                if let Some(r) = &mut self.renderer {
                    r.draw_frame();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run app");
}
