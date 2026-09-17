//! Milestone 2: a drifting field of ~1000 cubes, each a real ECS entity, drawn
//! in a single instanced draw. The schedule integrates their positions; an
//! extract step turns (Position, Spin) into per-instance model matrices that the
//! renderer uploads to a storage buffer. WASD/mouse fly the camera; Esc quits.

use std::time::Instant;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ExecutorKind;
use feather_gfx::Renderer;
use feather_platform::winit;
use feather_render::CubeRenderer;
use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const GRID: i32 = 10; // GRID^3 cubes
const MAX_INSTANCES: u32 = 4096;
const BOUND: f32 = 9.0;

// ---- ECS data ----

#[derive(Component)]
struct Position(Vec3);
#[derive(Component)]
struct Velocity(Vec3);
#[derive(Component)]
struct Spin(f32);
#[derive(Resource, Default)]
struct FrameCount(u64);

fn integrate(mut q: Query<(&mut Position, &Velocity)>) {
    let dt = 1.0 / 60.0;
    for (mut p, v) in &mut q {
        p.0 += v.0 * dt;
        if p.0.x > BOUND {
            p.0.x -= 2.0 * BOUND;
        } else if p.0.x < -BOUND {
            p.0.x += 2.0 * BOUND;
        }
        if p.0.y > BOUND {
            p.0.y -= 2.0 * BOUND;
        } else if p.0.y < -BOUND {
            p.0.y += 2.0 * BOUND;
        }
        if p.0.z > BOUND {
            p.0.z -= 2.0 * BOUND;
        } else if p.0.z < -BOUND {
            p.0.z += 2.0 * BOUND;
        }
    }
}

fn tick(mut frame: ResMut<FrameCount>) {
    frame.0 += 1;
}

fn rand01(seed: u32) -> f32 {
    let mut h = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    h ^= h >> 15;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    (h & 0x00ff_ffff) as f32 / 0x0100_0000 as f32
}

// ---- Fly camera + input ----

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
        proj.y_axis.y *= -1.0;
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
    cube: Option<CubeRenderer>,
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

        let half = (GRID as f32 - 1.0) / 2.0;
        for i in 0..(GRID * GRID * GRID) {
            let (x, y, z) = (i % GRID, (i / GRID) % GRID, i / (GRID * GRID));
            let pos = Vec3::new(x as f32 - half, y as f32 - half, z as f32 - half) * 1.6;
            let u = i as u32;
            let vel = Vec3::new(
                rand01(u * 3) - 0.5,
                rand01(u * 3 + 1) - 0.5,
                rand01(u * 3 + 2) - 0.5,
            ) * 1.5;
            let spin = (rand01(u * 7 + 11) - 0.5) * 3.0;
            world.spawn((Position(pos), Velocity(vel), Spin(spin)));
        }

        let mut schedule = Schedule::default();
        schedule.set_executor_kind(ExecutorKind::MultiThreaded);
        schedule.add_systems((integrate, tick));

        let now = Instant::now();
        Self {
            cube: None,
            renderer: None,
            window: None,
            world,
            schedule,
            camera: Camera {
                pos: Vec3::new(0.0, 0.0, 22.0),
                yaw: -std::f32::consts::FRAC_PI_2,
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
            .create_window(feather_platform::window_attributes("feather — instanced cubes"))
            .expect("create window");
        window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
            .ok();
        window.set_cursor_visible(false);

        let size = window.inner_size();
        let renderer = Renderer::new(&window, size.width, size.height).expect("create renderer");
        let cube = CubeRenderer::new(&renderer, MAX_INSTANCES);

        self.cube = Some(cube);
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
                    self.camera.pos += delta.normalize() * (12.0 * dt);
                }

                // Simulate.
                self.schedule.run(&mut self.world);

                // Extract: (Position, Spin) -> per-instance model matrices.
                let t = (now - self.start).as_secs_f32();
                let mut models: Vec<Mat4> = Vec::with_capacity((GRID * GRID * GRID) as usize);
                let mut q = self.world.query::<(&Position, &Spin)>();
                for (p, s) in q.iter(&self.world) {
                    models.push(
                        Mat4::from_translation(p.0)
                            * Mat4::from_rotation_y(t * s.0)
                            * Mat4::from_scale(Vec3::splat(0.3)),
                    );
                }

                let size = self.window.as_ref().unwrap().inner_size();
                let aspect = size.width as f32 / size.height.max(1) as f32;
                let view_proj = self.camera.view_proj(aspect);

                if let (Some(r), Some(c)) = (self.renderer.as_mut(), self.cube.as_mut()) {
                    r.draw_frame(|cmd, extent, frame| {
                        c.draw(cmd, extent, frame, view_proj, &models)
                    });
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
