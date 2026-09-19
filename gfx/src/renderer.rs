//! Vulkan bring-up + dynamic-rendering frame loop, plus vk-mem-backed buffer
//! and image allocation.
//!
//! Allocation ownership: the VMA allocator is held as an `Arc`, so `Buffer` and
//! `Image` are RAII — they free themselves on drop via their `Arc` clone. As
//! long as every buffer/image lives in something that drops before the renderer
//! (true for app/render-held resources), the renderer's final `Arc` is released
//! before `destroy_device`.

use std::error::Error;
use std::ffi::{c_char, c_void, CStr};
use std::sync::Arc;

use ash::{vk, Device, Entry, Instance};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use vk_mem::Alloc; // brings create_buffer/create_image into scope

pub const FRAMES_IN_FLIGHT: usize = 2;
const VALIDATION: bool = cfg!(debug_assertions);
// Linear-space clear for the HDR target (tonemap encodes to sRGB on output).
const CLEAR_COLOR: [f32; 4] = [0.02, 0.02, 0.05, 1.0];
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
// Offscreen scene color. Geometry lights in linear space into this; the tonemap
// pass reads it and writes the sRGB swapchain. RGBA16F gives HDR headroom.
const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
// Single source of truth for the geometry pass's MSAA sample count (HDR + depth
// targets and the mesh/sky pipelines all read it via `Renderer::samples`).
// TYPE_1 = no MSAA. NOTE: bumping this alone does NOT enable MSAA — a
// multisampled HDR target can't be sampled directly by the tonemap pass, so a
// resolve attachment (multisample HDR -> single-sample resolve image) must be
// added first. This is the seam that makes that change small; see §26.
const MSAA_SAMPLES: vk::SampleCountFlags = vk::SampleCountFlags::TYPE_1;

/// A GPU buffer that frees itself (and its allocation) on drop.
pub struct Buffer {
    allocator: Arc<vk_mem::Allocator>,
    allocation: vk_mem::Allocation,
    pub handle: vk::Buffer,
    pub size: vk::DeviceSize,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { self.allocator.destroy_buffer(self.handle, &mut self.allocation) };
    }
}

/// A persistently-mapped host-visible buffer for per-frame data (SSBO/UBO).
pub struct MappedBuffer {
    allocator: Arc<vk_mem::Allocator>,
    allocation: vk_mem::Allocation,
    ptr: *mut u8,
    pub handle: vk::Buffer,
    pub size: vk::DeviceSize,
}

impl MappedBuffer {
    /// Copy `data` into the mapped buffer (clamped to its size). Host-coherent.
    pub fn write(&mut self, data: &[u8]) {
        let n = data.len().min(self.size as usize);
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, n) };
    }
}

impl Drop for MappedBuffer {
    fn drop(&mut self) {
        unsafe { self.allocator.destroy_buffer(self.handle, &mut self.allocation) };
    }
}

/// A GPU image + view that free themselves on drop.
pub struct Image {
    allocator: Arc<vk_mem::Allocator>,
    allocation: vk_mem::Allocation,
    device: Device,
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.allocator.destroy_image(self.handle, &mut self.allocation);
        }
    }
}

pub struct Renderer {
    _entry: Entry,
    instance: Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,

    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,

    physical_device: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    #[allow(dead_code)]
    queue_family_index: u32,

    // Held before device destruction; buffers/images clone this Arc.
    allocator: Option<Arc<vk_mem::Allocator>>,

    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    depth: Option<Image>,
    // Engine-owned offscreen HDR scene color (sized to the swapchain, recreated
    // on resize). Geometry renders here; the tonemap pass samples it.
    hdr: Option<Image>,
    hdr_sampler: vk::Sampler,
    surface_format: vk::SurfaceFormatKHR,
    window_extent: vk::Extent2D,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    current_frame: usize,
}

impl Renderer {
    pub fn new<W: HasDisplayHandle + HasWindowHandle>(
        window: &W,
        width: u32,
        height: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::load()? };

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

        let (physical_device, queue_family_index) =
            pick_device(&instance, &surface_loader, surface)
                .ok_or("no GPU with a graphics+present queue and swapchain support")?;

        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let device_exts = [ash::khr::swapchain::NAME.as_ptr()];
        let mut features13 =
            vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        // Bindless-lite: non-uniform indexing into a fixed-size sampled-image
        // array (material_id -> textures[]). Widely supported on modern GPUs.
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_sampled_image_array_non_uniform_indexing(true);
        let device_create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_exts)
            .push_next(&mut features13)
            .push_next(&mut features12);
        let device = unsafe { instance.create_device(physical_device, &device_create, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Allocator first — the depth image is allocated through it.
        let allocator = {
            let info = vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
            Arc::new(unsafe { vk_mem::Allocator::new(info)? })
        };

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
        let depth = create_depth(&allocator, &device, sc.extent);
        let hdr = create_hdr(&allocator, &device, sc.extent);
        let hdr_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };

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
                    .command_buffer_count(FRAMES_IN_FLIGHT as u32),
            )?
        };

        let sem = vk::SemaphoreCreateInfo::default();
        let fence = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mut image_available = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            image_available.push(unsafe { device.create_semaphore(&sem, None)? });
            in_flight.push(unsafe { device.create_fence(&fence, None)? });
        }

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
            allocator: Some(allocator),
            swapchain_loader,
            swapchain: sc.swapchain,
            images: sc.images,
            image_views: sc.image_views,
            depth: Some(depth),
            hdr: Some(hdr),
            hdr_sampler,
            surface_format: sc.format,
            window_extent: sc.extent,
            command_pool,
            command_buffers,
            image_available,
            render_finished: sc.render_finished,
            in_flight,
            current_frame: 0,
        })
    }

    pub fn device(&self) -> Device {
        self.device.clone()
    }

    pub fn color_format(&self) -> vk::Format {
        self.surface_format.format
    }

    pub fn depth_format(&self) -> vk::Format {
        DEPTH_FORMAT
    }

    /// Format of the offscreen HDR scene-color target (geometry render target).
    pub fn hdr_format(&self) -> vk::Format {
        HDR_FORMAT
    }

    /// Geometry-pass MSAA sample count. The HDR + depth targets and the mesh/sky
    /// pipelines must all agree on this; the tonemap pass to the swapchain stays
    /// single-sample regardless. One knob for future MSAA (see `MSAA_SAMPLES`).
    pub fn samples(&self) -> vk::SampleCountFlags {
        MSAA_SAMPLES
    }

    /// View of the current HDR target. Changes on resize, so consumers that hold
    /// a descriptor pointing at it must refresh when the handle changes.
    pub fn hdr_view(&self) -> vk::ImageView {
        self.hdr.as_ref().expect("hdr target alive").view
    }

    /// Shared sampler for reading the HDR target in the tonemap pass.
    pub fn hdr_sampler(&self) -> vk::Sampler {
        self.hdr_sampler
    }

    pub fn wait_idle(&self) {
        unsafe { self.device.device_wait_idle().ok() };
    }

    fn allocator(&self) -> &Arc<vk_mem::Allocator> {
        self.allocator.as_ref().expect("allocator alive")
    }

    /// Create a device-local buffer initialized from `data` via a staging copy.
    pub fn create_device_local_buffer(
        &self,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Buffer {
        let size = data.len() as vk::DeviceSize;
        let allocator = self.allocator();

        // Staging (host-visible, mappable).
        let staging_ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_ai = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        let (staging_buf, mut staging_alloc) =
            unsafe { allocator.create_buffer(&staging_ci, &staging_ai).expect("staging buffer") };
        unsafe {
            let ptr = allocator.map_memory(&mut staging_alloc).expect("map staging");
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            allocator.unmap_memory(&mut staging_alloc);
        }

        // Device-local target.
        let dev_ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let dev_ai = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        let (dev_buf, dev_alloc) =
            unsafe { allocator.create_buffer(&dev_ci, &dev_ai).expect("device buffer") };

        self.one_time_submit(|cmd| unsafe {
            let region = vk::BufferCopy::default().size(size);
            self.device.cmd_copy_buffer(cmd, staging_buf, dev_buf, &[region]);
        });

        unsafe { allocator.destroy_buffer(staging_buf, &mut staging_alloc) };

        Buffer {
            allocator: allocator.clone(),
            allocation: dev_alloc,
            handle: dev_buf,
            size,
        }
    }

    /// A persistently-mapped host-visible buffer, e.g. for per-frame SSBO data.
    pub fn create_host_visible_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> MappedBuffer {
        let allocator = self.allocator();
        let ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let ai = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };
        let (handle, allocation) =
            unsafe { allocator.create_buffer(&ci, &ai).expect("host-visible buffer") };
        let ptr = allocator.get_allocation_info(&allocation).mapped_data as *mut u8;
        MappedBuffer {
            allocator: allocator.clone(),
            allocation,
            ptr,
            handle,
            size,
        }
    }

    /// Upload RGBA8 `pixels` into a sampled 2D image (single mip) and leave it in
    /// `SHADER_READ_ONLY_OPTIMAL`. `srgb` picks `R8G8B8A8_SRGB` (base color, gets
    /// linearized on sample) vs `_UNORM`. No mip chain yet — a follow-up.
    pub fn create_texture(&self, pixels: &[u8], width: u32, height: u32, srgb: bool) -> Image {
        let allocator = self.allocator();
        let format = if srgb {
            vk::Format::R8G8B8A8_SRGB
        } else {
            vk::Format::R8G8B8A8_UNORM
        };
        let extent = vk::Extent3D {
            width: width.max(1),
            height: height.max(1),
            depth: 1,
        };

        let image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let ai = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        let (image, allocation) =
            unsafe { allocator.create_image(&image_ci, &ai).expect("texture image") };

        // Staging buffer holding the pixels.
        let staging_ci = vk::BufferCreateInfo::default()
            .size(pixels.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_ai = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        let (staging_buf, mut staging_alloc) = unsafe {
            allocator
                .create_buffer(&staging_ci, &staging_ai)
                .expect("texture staging")
        };
        unsafe {
            let ptr = allocator.map_memory(&mut staging_alloc).expect("map staging");
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr, pixels.len());
            allocator.unmap_memory(&mut staging_alloc);
        }

        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        self.one_time_submit(|cmd| unsafe {
            // UNDEFINED -> TRANSFER_DST_OPTIMAL
            let to_dst = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );

            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(extent);
            self.device.cmd_copy_buffer_to_image(
                cmd,
                staging_buf,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
            let to_read = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
        });

        unsafe { allocator.destroy_buffer(staging_buf, &mut staging_alloc) };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(range);
        let view = unsafe {
            self.device
                .create_image_view(&view_info, None)
                .expect("texture view")
        };

        Image {
            allocator: allocator.clone(),
            allocation,
            device: self.device.clone(),
            handle: image,
            view,
            format,
        }
    }

    fn one_time_submit(&self, record: impl FnOnce(vk::CommandBuffer)) {
        unsafe {
            let cmd = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("alloc one-time cmd")[0];
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
            record(cmd);
            self.device.end_command_buffer(cmd).unwrap();

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            self.device.queue_submit(self.queue, &[submit], fence).unwrap();
            self.device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cmd]);
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.window_extent = vk::Extent2D { width, height };
        self.recreate_swapchain();
    }

    /// Records and submits one frame: `geometry` draws into the linear HDR
    /// target, then `post` (the tonemap pass) samples it and writes the sRGB
    /// swapchain. Both receive `(cmd, extent, frame_in_flight)`.
    pub fn draw_frame(
        &mut self,
        geometry: impl FnOnce(vk::CommandBuffer, vk::Extent2D, usize),
        post: impl FnOnce(vk::CommandBuffer, vk::Extent2D, usize),
    ) {
        if self.window_extent.width == 0 || self.window_extent.height == 0 {
            return;
        }
        let frame = self.current_frame;

        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight[frame]], true, u64::MAX)
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
            Ok((idx, _)) => idx,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain();
                return;
            }
            Err(e) => panic!("acquire_next_image: {e:?}"),
        };

        unsafe {
            self.device
                .reset_fences(&[self.in_flight[frame]])
                .unwrap()
        };

        let cmd = self.command_buffers[frame];
        let image = self.images[image_index as usize];
        let swap_view = self.image_views[image_index as usize];
        let depth = self.depth.as_ref().unwrap();
        let hdr = self.hdr.as_ref().unwrap();
        let extent = self.window_extent;
        let color_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let depth_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let dev = &self.device;
        unsafe {
            dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            dev.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();

            // ---- Geometry pass: render into the linear HDR target. ----

            // HDR: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL (prior contents discarded).
            let to_hdr = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(hdr.handle)
                .subresource_range(color_range)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_hdr],
            );

            // Depth: UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL
            let to_depth = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(depth.handle)
                .subresource_range(depth_range)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_depth],
            );

            let hdr_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(hdr.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: CLEAR_COLOR },
                });
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });
            let hdr_attachments = [hdr_attachment];
            let geo_rendering = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(&hdr_attachments)
                .depth_attachment(&depth_attachment);

            dev.cmd_begin_rendering(cmd, &geo_rendering);
            geometry(cmd, extent, frame);
            dev.cmd_end_rendering(cmd);

            // ---- Tonemap pass: sample HDR, write the sRGB swapchain. ----

            // HDR: COLOR_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL (frag read).
            let hdr_to_read = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(hdr.handle)
                .subresource_range(color_range)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[hdr_to_read],
            );

            // Swapchain: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
            let to_color = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range)
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

            // The fullscreen tonemap covers every pixel, so the swapchain load
            // op is DONT_CARE (no clear needed).
            let swap_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(swap_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE);
            let swap_attachments = [swap_attachment];
            let post_rendering = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(&swap_attachments);

            dev.cmd_begin_rendering(cmd, &post_rendering);
            post(cmd, extent, frame);
            dev.cmd_end_rendering(cmd);

            // Swapchain: COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR
            let to_present = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range)
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

        self.current_frame = (frame + 1) % FRAMES_IN_FLIGHT;
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

        // Drops the old depth/HDR images (frees view+image) before reassigning.
        self.depth = Some(create_depth(self.allocator(), &self.device, sc.extent));
        self.hdr = Some(create_hdr(self.allocator(), &self.device, sc.extent));
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

            // Free VMA-backed resources while the device + allocator are alive.
            self.depth.take();
            self.hdr.take();
            self.device.destroy_sampler(self.hdr_sampler, None);

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

            // Now release the allocator (last Arc ref) before destroying the device.
            self.allocator.take();

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

fn create_depth(allocator: &Arc<vk_mem::Allocator>, device: &Device, extent: vk::Extent2D) -> Image {
    let image_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(DEPTH_FORMAT)
        .extent(vk::Extent3D {
            width: extent.width.max(1),
            height: extent.height.max(1),
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(MSAA_SAMPLES)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let ai = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::AutoPreferDevice,
        ..Default::default()
    };
    let (image, allocation) =
        unsafe { allocator.create_image(&image_ci, &ai).expect("depth image") };

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(DEPTH_FORMAT)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { device.create_image_view(&view_info, None).expect("depth view") };

    Image {
        allocator: allocator.clone(),
        allocation,
        device: device.clone(),
        handle: image,
        view,
        format: DEPTH_FORMAT,
    }
}

fn create_hdr(allocator: &Arc<vk_mem::Allocator>, device: &Device, extent: vk::Extent2D) -> Image {
    let image_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(HDR_FORMAT)
        .extent(vk::Extent3D {
            width: extent.width.max(1),
            height: extent.height.max(1),
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(MSAA_SAMPLES)
        .tiling(vk::ImageTiling::OPTIMAL)
        // Rendered into as a color attachment, then sampled by the tonemap pass.
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let ai = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::AutoPreferDevice,
        ..Default::default()
    };
    let (image, allocation) = unsafe { allocator.create_image(&image_ci, &ai).expect("hdr image") };

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(HDR_FORMAT)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { device.create_image_view(&view_info, None).expect("hdr view") };

    Image {
        allocator: allocator.clone(),
        allocation,
        device: device.clone(),
        handle: image,
        view,
        format: HDR_FORMAT,
    }
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
