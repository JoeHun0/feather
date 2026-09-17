//! Milestone: bring up Vulkan and draw via dynamic rendering (VK 1.3).
//!
//! gfx owns the device/swapchain/frame mechanics and the per-frame render-target
//! setup (image transitions + begin/end rendering). It does NOT own pipelines —
//! `draw_frame` takes a callback so a higher layer (feather-render) records the
//! actual draw into the active rendering scope.
//!
//! Windowing-agnostic: `new` is generic over the raw-window-handle traits.

use std::error::Error;
use std::ffi::{c_char, c_void, CStr};

use ash::{vk, Device, Entry, Instance};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

const MAX_FRAMES_IN_FLIGHT: usize = 2;
const VALIDATION: bool = cfg!(debug_assertions);
const CLEAR_COLOR: [f32; 4] = [0.02, 0.02, 0.05, 1.0];

pub struct Renderer {
    _entry: Entry,
    instance: Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,

    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,

    physical_device: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    #[allow(dead_code)] // used at pool creation; kept for future queue-ownership work
    queue_family_index: u32,

    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    surface_format: vk::SurfaceFormatKHR,
    window_extent: vk::Extent2D,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available: Vec<vk::Semaphore>, // per frame-in-flight
    render_finished: Vec<vk::Semaphore>, // per swapchain image
    in_flight: Vec<vk::Fence>,           // per frame-in-flight
    current_frame: usize,

    allocator: Option<vk_mem::Allocator>,
}

impl Renderer {
    pub fn new<W: HasDisplayHandle + HasWindowHandle>(
        window: &W,
        width: u32,
        height: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::load()? };

        // ---- Instance ----
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"feather")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"feather")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let display_handle = window.display_handle()?.as_raw();
        let mut ext_names: Vec<*const c_char> =
            ash_window::enumerate_required_extensions(display_handle)?.to_vec();
        if VALIDATION {
            ext_names.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let layer_names = [c"VK_LAYER_KHRONOS_validation".as_ptr()];

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&ext_names);
        if VALIDATION {
            create_info = create_info.enabled_layer_names(&layer_names);
        }
        let instance = unsafe { entry.create_instance(&create_info, None)? };

        // ---- Debug messenger (debug builds only) ----
        let debug = if VALIDATION {
            let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            let m = unsafe { du.create_debug_utils_messenger(&info, None)? };
            Some((du, m))
        } else {
            None
        };

        // ---- Surface ----
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                display_handle,
                window.window_handle()?.as_raw(),
                None,
            )?
        };

        // ---- Physical device + a graphics-and-present queue family ----
        let (physical_device, queue_family_index) =
            pick_device(&instance, &surface_loader, surface)
                .ok_or("no GPU with a graphics+present queue and swapchain support")?;

        // ---- Logical device (enable dynamic rendering, core in 1.3 but opt-in) ----
        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let device_exts = [ash::khr::swapchain::NAME.as_ptr()];
        let mut features13 =
            vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let device_create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_exts)
            .push_next(&mut features13);
        let device = unsafe { instance.create_device(physical_device, &device_create, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // ---- Swapchain ----
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let hint = vk::Extent2D { width, height };
        let sc = create_swapchain_resources(
            &surface_loader,
            &swapchain_loader,
            &device,
            physical_device,
            surface,
            hint,
            vk::SwapchainKHR::null(),
        )?;

        // ---- Command pool + buffers ----
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let command_buffers = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32),
            )?
        };

        // ---- Sync objects ----
        let sem = vk::SemaphoreCreateInfo::default();
        let fence = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mut image_available = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available.push(unsafe { device.create_semaphore(&sem, None)? });
            in_flight.push(unsafe { device.create_fence(&fence, None)? });
        }

        // ---- VMA allocator (wired now; first buffers allocated next milestone) ----
        let allocator = {
            let info = vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
            unsafe { vk_mem::Allocator::new(info)? }
        };

        Ok(Self {
            _entry: entry,
            instance,
            debug,
            surface_loader,
            surface,
            physical_device,
            device,
            queue,
            queue_family_index,
            swapchain_loader,
            swapchain: sc.swapchain,
            images: sc.images,
            image_views: sc.image_views,
            surface_format: sc.format,
            window_extent: sc.extent,
            command_pool,
            command_buffers,
            image_available,
            render_finished: sc.render_finished,
            in_flight,
            current_frame: 0,
            allocator: Some(allocator),
        })
    }

    /// Cheap clone of the device handle, for building pipelines in higher layers.
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Swapchain color format — needed to build dynamic-rendering pipelines.
    pub fn color_format(&self) -> vk::Format {
        self.surface_format.format
    }

    /// Block until the GPU is idle. Call before tearing down GPU resources that
    /// live outside the renderer (e.g. pipelines) so their destroy calls don't
    /// race in-flight command buffers.
    pub fn wait_idle(&self) {
        unsafe { self.device.device_wait_idle().ok() };
    }

    /// Call on the window's resize event.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.window_extent = vk::Extent2D { width, height };
        self.recreate_swapchain();
    }

    /// Acquire -> begin dynamic rendering (clear) -> `record` the draw -> end ->
    /// present. `record` runs inside the active rendering scope.
    pub fn draw_frame(&mut self, record: impl FnOnce(vk::CommandBuffer, vk::Extent2D)) {
        if self.window_extent.width == 0 || self.window_extent.height == 0 {
            return; // minimized
        }
        let frame = self.current_frame;
        let dev = &self.device;

        unsafe {
            dev.wait_for_fences(&[self.in_flight[frame]], true, u64::MAX)
                .unwrap();
        }

        let image_index = match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available[frame],
                vk::Fence::null(),
            )
        } {
            Ok((idx, _suboptimal)) => idx,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain();
                return;
            }
            Err(e) => panic!("acquire_next_image: {e:?}"),
        };

        unsafe { dev.reset_fences(&[self.in_flight[frame]]).unwrap() };

        let cmd = self.command_buffers[frame];
        let image = self.images[image_index as usize];
        let view = self.image_views[image_index as usize];
        let extent = self.window_extent;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        unsafe {
            dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            dev.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();

            // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
            let to_color = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_color],
            );

            // ---- Dynamic rendering scope ----
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: CLEAR_COLOR,
                    },
                });
            let color_attachments = [color_attachment];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachments);

            dev.cmd_begin_rendering(cmd, &rendering_info);
            record(cmd, extent);
            dev.cmd_end_rendering(cmd);

            // COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR
            let to_present = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::empty());
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );

            dev.end_command_buffer(cmd).unwrap();

            let wait_sems = [self.image_available[frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_sems = [self.render_finished[image_index as usize]];
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmds)
                .signal_semaphores(&signal_sems);
            dev.queue_submit(self.queue, &[submit], self.in_flight[frame])
                .unwrap();

            let swapchains = [self.swapchain];
            let indices = [image_index];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_sems)
                .swapchains(&swapchains)
                .image_indices(&indices);
            match self.swapchain_loader.queue_present(self.queue, &present) {
                Ok(false) => {}
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate_swapchain(),
                Err(e) => panic!("queue_present: {e:?}"),
            }
        }

        self.current_frame = (frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn recreate_swapchain(&mut self) {
        if self.window_extent.width == 0 || self.window_extent.height == 0 {
            return;
        }
        unsafe { self.device.device_wait_idle().unwrap() };

        let old = self.swapchain;
        let sc = create_swapchain_resources(
            &self.surface_loader,
            &self.swapchain_loader,
            &self.device,
            self.physical_device,
            self.surface,
            self.window_extent,
            old,
        )
        .expect("recreate swapchain");

        unsafe {
            for &v in &self.image_views {
                self.device.destroy_image_view(v, None);
            }
            for &s in &self.render_finished {
                self.device.destroy_semaphore(s, None);
            }
            self.swapchain_loader.destroy_swapchain(old, None);
        }

        self.swapchain = sc.swapchain;
        self.images = sc.images;
        self.image_views = sc.image_views;
        self.surface_format = sc.format;
        self.window_extent = sc.extent;
        self.render_finished = sc.render_finished;
        self.current_frame = 0;
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.allocator.take(); // VMA before the device it was built on

            for &v in &self.image_views {
                self.device.destroy_image_view(v, None);
            }
            for &s in &self.render_finished {
                self.device.destroy_semaphore(s, None);
            }
            for &s in &self.image_available {
                self.device.destroy_semaphore(s, None);
            }
            for &f in &self.in_flight {
                self.device.destroy_fence(f, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some((du, m)) = self.debug.take() {
                du.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

struct SwapchainResources {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    render_finished: Vec<vk::Semaphore>,
}

fn create_swapchain_resources(
    surface_loader: &ash::khr::surface::Instance,
    swapchain_loader: &ash::khr::swapchain::Device,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    extent_hint: vk::Extent2D,
    old_swapchain: vk::SwapchainKHR,
) -> Result<SwapchainResources, Box<dyn Error>> {
    let caps = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
    };
    let formats =
        unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface)? };

    let format = formats
        .iter()
        .copied()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(formats[0]);

    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: extent_hint
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: extent_hint
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };

    let mut image_count = caps.min_image_count + 1;
    if caps.max_image_count > 0 && image_count > caps.max_image_count {
        image_count = caps.max_image_count;
    }

    let info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(old_swapchain);

    let swapchain = unsafe { swapchain_loader.create_swapchain(&info, None)? };
    let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

    let mut image_views = Vec::with_capacity(images.len());
    for &img in &images {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(img)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format.format)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        image_views.push(unsafe { device.create_image_view(&view_info, None)? });
    }

    let sem = vk::SemaphoreCreateInfo::default();
    let mut render_finished = Vec::with_capacity(images.len());
    for _ in 0..images.len() {
        render_finished.push(unsafe { device.create_semaphore(&sem, None)? });
    }

    Ok(SwapchainResources {
        swapchain,
        images,
        image_views,
        format,
        extent,
        render_finished,
    })
}

fn pick_device(
    instance: &Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Option<(vk::PhysicalDevice, u32)> {
    let devices = unsafe { instance.enumerate_physical_devices().ok()? };
    let mut fallback = None;
    for pd in devices {
        let props = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let family = props.iter().enumerate().find_map(|(i, qf)| {
            let i = i as u32;
            let graphics = qf.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let present = unsafe {
                surface_loader
                    .get_physical_device_surface_support(pd, i, surface)
                    .unwrap_or(false)
            };
            (graphics && present).then_some(i)
        });
        if let Some(family) = family {
            let dprops = unsafe { instance.get_physical_device_properties(pd) };
            if dprops.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                return Some((pd, family));
            }
            fallback.get_or_insert((pd, family));
        }
    }
    fallback
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut c_void,
) -> vk::Bool32 {
    let msg = CStr::from_ptr((*data).p_message);
    eprintln!("[vulkan {severity:?} {types:?}] {}", msg.to_string_lossy());
    vk::FALSE
}
