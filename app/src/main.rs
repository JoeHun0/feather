//! Milestone: fly around a shaded cube. WASD moves, mouse looks, Space/Ctrl go
//! up/down, Esc quits. Camera is updated at render rate; the ECS schedule still
//! ticks each frame. Input is handled directly here for now — the action-mapping
//! layer is a later milestone.

use std::time::Instant;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ExecutorKind;
use feather_gfx::Renderer;
use feather_platform::winit;
use feather_render::CubePipeline;
use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

// ---- Example ECS data (moves into feather-game once real systems exist) ----

#[derive(Component)]
struct Position(glam::Vec3);
#[derive(Component)]
struct Velocity(glam::Vec3);
#[derive(Resource, Default)]
struct FrameCount(u64);

fn integrate(mut q: Query<(&mut Position, &Velocity)>) {
    let dt = 1.0 / 60.0;
    for (mut pos, vel) in &mut q {
        pos.0 += vel.0 * dt;
    }
}
fn tick(mut frame: ResMut<FrameCount>) {
    frame.0 += 1;
}

// ---- Fly camera + input (temporary; superseded by the input-mapping layer) ----

struct Camera {
    pos: Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp)
    }

    fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_to_rh(self.pos, self.forward(), Vec3::Y);
        let mut proj = Mat4::perspective_rh(60f32.to_radians(), aspect, 0.1, 100.0);
        proj.y_axis.y *= -1.0; // Vulkan clip space (Y down)
        proj * view
    }
}

#[derive(Default)]
struct Input {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    mouse_dx: f32,
    mouse_dy: f32,
}

struct App {
    pipeline: Option<CubePipeline>,
    renderer: Option<Renderer>,
    window: Option<Window>,
    world: World,
    schedule: Schedule,
    camera: Camera,
    input: Input,
    start: Instant,
    last_frame: Instant,
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

        let now = Instant::now();
        Self {
            pipeline: None,
            renderer: None,
            window: None,
            world,
            schedule,
            camera: Camera {
                pos: Vec3::new(0.0, 0.0, 3.0),
                yaw: -std::f32::consts::FRAC_PI_2, // look toward -Z
                pitch: 0.0,
            },
            input: Input::default(),
            start: now,
            last_frame: now,
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        if let Some(r) = &self.renderer {
            r.wait_idle();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = event_loop
            .create_window(feather_platform::window_attributes("feather — fly cam"))
            .expect("create window");
        // Lock the pointer for mouse-look (fall back to confined).
        window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
            .ok();
        window.set_cursor_visible(false);

        let size = window.inner_size();
        let renderer = Renderer::new(&window, size.width, size.height).expect("create renderer");
        let device = renderer.device();
        let pipeline = CubePipeline::new(&device, renderer.color_format());

        self.pipeline = Some(pipeline);
        self.renderer = Some(renderer);
        self.window = Some(window);
    }

    fn device_event(&mut self, _e: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.input.mouse_dx += dx as f32;
            self.input.mouse_dy += dy as f32;
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => self.input.forward = pressed,
                        KeyCode::KeyS => self.input.back = pressed,
                        KeyCode::KeyA => self.input.left = pressed,
                        KeyCode::KeyD => self.input.right = pressed,
                        KeyCode::Space => self.input.up = pressed,
                        KeyCode::ControlLeft => self.input.down = pressed,
                        KeyCode::Escape if pressed => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                // Look.
                let sens = 0.0025;
                self.camera.yaw += self.input.mouse_dx * sens;
                self.camera.pitch =
                    (self.camera.pitch - self.input.mouse_dy * sens).clamp(-1.54, 1.54);
                self.input.mouse_dx = 0.0;
                self.input.mouse_dy = 0.0;

                // Move.
                let fwd = self.camera.forward();
                let right = fwd.cross(Vec3::Y).normalize();
                let mut delta = Vec3::ZERO;
                if self.input.forward {
                    delta += fwd;
                }
                if self.input.back {
                    delta -= fwd;
                }
                if self.input.right {
                    delta += right;
                }
                if self.input.left {
                    delta -= right;
                }
                if self.input.up {
                    delta += Vec3::Y;
                }
                if self.input.down {
                    delta -= Vec3::Y;
                }
                if delta != Vec3::ZERO {
                    self.camera.pos += delta.normalize() * (3.0 * dt);
                }

                self.schedule.run(&mut self.world);

                let size = self.window.as_ref().unwrap().inner_size();
                let aspect = size.width as f32 / size.height.max(1) as f32;
                let model = Mat4::from_rotation_y((now - self.start).as_secs_f32() * 0.5);
                let mvp = self.camera.view_proj(aspect) * model;

                if let (Some(r), Some(p)) = (self.renderer.as_mut(), self.pipeline.as_ref()) {
                    r.draw_frame(|cmd, extent| p.draw(cmd, extent, mvp));
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
