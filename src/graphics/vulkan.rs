//! Vulkan rendering backend.
//!
//! The Vulkan calls follow the API of `ash` 0.37 and the structure of the
//! reference implementation shipped with ash (`examples/src/lib.rs`).

use std::ffi::CStr;
use std::os::raw::c_char;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use ash::vk::Handle;
use ash::{vk, Device, Entry, Instance};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::graphics::*;

mod frame;
mod resource;

pub use resource::*;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Descriptor binding reserved for the uniform block a shader declares.
pub const UNIFORM_BINDING: u32 = 0;
/// First descriptor binding available to the samplers a shader declares.
pub const IMAGE_BINDING_BASE: u32 = 1;

/// Depth format used for every depth attachment the backend creates. `D16_UNORM`
/// matches the `GL_UNSIGNED_SHORT` depth format of the OpenGL backend and the
/// two bytes per pixel `TextureFormat::Depth` reports.
pub const DEPTH_FORMAT: vk::Format = vk::Format::D16_UNORM;

/// Descriptor writes collected between `apply_bindings`/`apply_uniforms` and
/// `draw`, flushed into a descriptor set allocated for that draw call.
#[derive(Default)]
pub(crate) struct PendingDescriptors {
    /// Slice of the frame's uniform arena the pending draw call reads its
    /// uniform block from. All writes between two draws share it, so the last
    /// one wins, exactly like the immediate `glUniform*` calls a draw call makes
    /// on the OpenGL backend.
    pub uniform_offset: Option<vk::DeviceSize>,
    pub images: Vec<Option<vk::DescriptorImageInfo>>,
    /// Textures this pass samples. Vulkan forbids image barriers inside a
    /// dynamic rendering instance, so their layout transition is recorded once
    /// the pass has ended.
    pub sampled: Vec<usize>,
}

impl PendingDescriptors {
    pub fn clear(&mut self) {
        self.uniform_offset = None;
        self.images.clear();
        self.sampled.clear();
    }
}

/// Size a uniform arena starts with, in bytes.
const UNIFORM_ARENA_SIZE: vk::DeviceSize = 1 << 16;

/// Storage for the uniform blocks of one frame.
///
/// A frame's uniforms cannot all live in one buffer: the frame is recorded in
/// full before it is submitted, so the last write would be the only one the GPU
/// ever sees. Every draw call copies its block into its own slice instead and
/// names it with a dynamic offset when its descriptor set is bound.
pub struct UniformArena {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped: *mut u8,
    pub capacity: vk::DeviceSize,
    /// Where the next slice starts.
    pub offset: vk::DeviceSize,
    /// `VkPhysicalDeviceLimits::minUniformBufferOffsetAlignment`.
    pub alignment: vk::DeviceSize,
    /// Arenas the frame that owns this slot outgrew. Their command buffer may
    /// still read them, so they are released once that frame has finished.
    pub retired: Vec<(vk::Buffer, vk::DeviceMemory)>,
}

/// Per-frame host visible arena the stream buffers are written into.
///
/// A frame's command buffer is recorded in full before it is submitted, so a
/// buffer that is updated twice in one frame cannot keep a single copy: every
/// update gets its own slice and every draw binds the slice that was current
/// when it was recorded.
pub struct StreamArena {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped: *mut u8,
    pub capacity: vk::DeviceSize,
    pub offset: vk::DeviceSize,
    pub retired: Vec<(vk::Buffer, vk::DeviceMemory)>,
}

pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub surface_loader: Surface,
    pub swapchain_loader: Swapchain,
    pub debug_utils_loader: DebugUtils,
    pub debug_messenger: Option<vk::DebugUtilsMessengerEXT>,

    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// `VkPhysicalDeviceLimits::maxImageDimension2D`.
    pub max_texture_size: u32,
    /// `VkPhysicalDeviceLimits::minUniformBufferOffsetAlignment`.
    pub uniform_alignment: vk::DeviceSize,
    /// `VkPhysicalDeviceLimits::framebufferColorSampleCounts`; sample counts a
    /// colour attachment may use on this device.
    pub color_sample_counts: vk::SampleCountFlags,
    pub device: Device,
    pub queue_family_index: u32,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub video_queue_family: Option<u32>,
    pub video_queue: vk::Queue,

    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,

    /// `image_available` is handed out round-robin because
/// `acquire_next_image` has to name the wait semaphore before the image index
/// is known; `render_finished` is indexed by the acquired image so a semaphore
/// is never reused while the swapchain may still be presenting with it.
pub image_available: Vec<vk::Semaphore>,
    pub render_finished: Vec<vk::Semaphore>,
    pub retired_semaphores: Vec<vk::Semaphore>,
    pub in_flight: Vec<vk::Fence>,
    /// Cumulative time spent waiting on the per-frame fence in `begin_frame`
    /// (the non-offscreen path).  Diagnostics only.
    pub frame_wait_ns: u64,

    pub window_size: (u32, u32),
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_suboptimal: bool,
    pub surface_format: vk::SurfaceFormatKHR,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    /// Layout of each swap chain image. `UNDEFINED` right after acquisition,
    /// `COLOR_ATTACHMENT_OPTIMAL` while the frame draws into it, and
    /// `PRESENT_SRC_KHR` once it has been handed to the presentation engine.
    pub swapchain_image_layouts: Vec<vk::ImageLayout>,

    /// Persistent staging buffer and command buffer used by `read_pixels`.
    pub readback_buffer: vk::Buffer,
    pub readback_memory: vk::DeviceMemory,
    pub readback_mapped: *mut u8,
    pub readback_capacity: vk::DeviceSize,
    pub readback_cmd: vk::CommandBuffer,
    pub readback_fence: vk::Fence,

    /// Depth attachment for the default frame, sized with the swap chain.
    pub default_depth: Texture,
    /// 1x1 opaque black texture bound to image slots a draw leaves empty.
    pub dummy_texture: usize,

    pub shaders: Vec<ShaderModule>,
    pub pipelines: Vec<PipelineModule>,
    pub buffers: Vec<BufferModule>,
    pub textures: Vec<TextureModule>,
    pub passes: Vec<Option<RenderPassModule>>,
    pub free_passes: Vec<usize>,

    /// One pool per frame slot: a set stays referenced by the command
    /// buffer it was bound in, so only the slot whose fence has been waited on
    /// can have its pool reset.
    pub descriptor_pools: Vec<vk::DescriptorPool>,

    /// One uniform arena per frame slot: a slice a frame wrote is still read by
    /// the GPU while the next frame is recorded.
    pub uniform_arenas: Vec<UniformArena>,
    /// One stream arena per frame slot for vertex and index buffer updates.
    pub stream_arenas: Vec<StreamArena>,

    pub current_frame: usize,
    pub acquire_slot: usize,
    /// The `image_available` semaphore the current frame waits on.
    pub frame_wait: vk::Semaphore,
    /// Whether `frame_wait` has yet to be consumed by a queue submission.
    pub frame_wait_pending: bool,
    pub acquired_image: Option<usize>,
    pub recording: bool,
    pub pass_open: bool,
    pub frame_active: bool,
    /// Whether commands have been recorded into the frame command buffer since
    /// it was last submitted by `flush_recording`.
    pub cmd_dirty: bool,
    /// Whether `flush_recording` submitted the frame and the GPU has not been
    /// waited on yet.
    pub frame_submitted: bool,
    /// Offscreen video export: frame command buffers are indexed by
    /// `current_frame` (a `MAX_FRAMES_IN_FLIGHT` ring) and no swapchain image
    /// is acquired.  Used by the video exporter so several frames can be
    /// recorded, submitted and copied back without a host stall per frame.
    pub offscreen: bool,
    /// Samples per pixel of the open render target; decides which pipeline
    /// variant `draw` binds.
    pub cur_samples: u32,
    /// Colour format of the open render target; a pipeline only matches a
    /// target of the format it was compiled for.
    pub cur_color_format: vk::Format,
    /// Depth format of the open render target, if it has a depth attachment.
    pub cur_depth_format: Option<vk::Format>,
    /// Index of the render target currently open, if any.
    pub cur_pass: Option<usize>,
    /// Target the open rendering instance draws into, kept after the instance
    /// is closed so it can be reopened after an image layout transition.
    pub cur_target: Option<usize>,
    /// Pipeline state resolved from the bound pipeline and the `set_*` calls
    /// made after it. Vulkan bakes the state into the pipeline, so a change
    /// compiles a variant.
    pub cur_params: PipelineParams,

    pub cur_pipeline: Option<Pipeline>,
    pub cur_index_buffer: Option<usize>,
    pub cur_vertex_buffers: Vec<Option<usize>>,
    pub viewport: vk::Rect2D,
    pub scissor: vk::Rect2D,
    pub(crate) pending: PendingDescriptors,

    pub display: Option<*mut dyn crate::NativeDisplay>,
}

unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message: &str = if !callback_data.p_message.is_null() {
        CStr::from_ptr(callback_data.p_message).to_str().unwrap_or("")
    } else {
        ""
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            eprintln!(
                "[miniquad][vulkan] {:?}/{:?}: {}",
                message_severity, message_type, message
            );
        }
        _ => {
            println!(
                "[miniquad][vulkan] {:?}/{:?}: {}",
                message_severity, message_type, message
            );
        }
    }

    vk::FALSE
}

impl VulkanContext {
    pub fn new(
        display: RawDisplayHandle,
        window: RawWindowHandle,
        window_size: (u32, u32),
    ) -> VulkanContext {
        unsafe {
            let entry = Entry::load().expect("Failed to load the Vulkan loader");
            let app_name = CStr::from_bytes_with_nul_unchecked(b"miniquad\0");

            // Validation needs the Khronos layer to be installed, so it is opt-in
            // rather than always requested: the loader is told about
            // `VK_INSTANCE_LAYERS` on its own, this only decides whether the
            // messages get routed to stderr.
            let validation = std::env::var_os("MINIQUAD_VULKAN_VALIDATION").is_some();

            let layer_names: Vec<*const c_char> = if validation {
                [cstr("VK_LAYER_KHRONOS_validation")]
                    .iter()
                    .map(|name| name.as_ptr())
                    .collect()
            } else {
                Vec::new()
            };

            let mut extension_names =
                ash_window::enumerate_required_extensions(display).unwrap().to_vec();
            if validation {
                extension_names.push(DebugUtils::name().as_ptr());
            }

            let app_info = vk::ApplicationInfo::builder()
                .application_name(app_name)
                .application_version(0)
                .engine_name(app_name)
                .engine_version(0)
                // Dynamic rendering (`vkCmdBeginRendering`) is what lets a
                // single pipeline run against the swap chain, plain offscreen
                // targets and multisampled targets alike.
                .api_version(vk::make_api_version(0, 1, 3, 0));

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_layer_names(&layer_names)
                .enabled_extension_names(&extension_names);

            let instance: Instance = entry
                .create_instance(&create_info, None)
                .expect("Failed to create the Vulkan instance");

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));

            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_messenger = if validation {
                Some(
                    debug_utils_loader
                        .create_debug_utils_messenger(&debug_info, None)
                        .expect("Failed to create the debug messenger"),
                )
            } else {
                None
            };

            let surface = ash_window::create_surface(&entry, &instance, display, window, None)
                .expect("Failed to create a Vulkan surface for the window");

            let surface_loader = Surface::new(&entry, &instance);

            let physical_devices = instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate Vulkan physical devices");

            let (physical_device, queue_family_index) = physical_devices
                .iter()
                .find_map(|device| {
                    instance
                        .get_physical_device_queue_family_properties(*device)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let graphics_and_surface = info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                && surface_loader
                                    .get_physical_device_surface_support(
                                        *device,
                                        index as u32,
                                        surface,
                                    )
                                    .unwrap();
                            if graphics_and_surface {
                                Some((*device, index as u32))
                            } else {
                                None
                            }
                        })
                })
                .expect("No Vulkan device with graphics and presentation support was found");

            let available_extensions = instance
                .enumerate_device_extension_properties(physical_device)
                .expect("Failed to enumerate device extensions");
            let has_extension = |name: &str| {
                available_extensions.iter().any(|extension| {
                    CStr::from_ptr(extension.extension_name.as_ptr()).to_bytes() == name.as_bytes()
                })
            };
            let video_queue_supported = has_extension("VK_KHR_video_queue")
                && has_extension("VK_KHR_video_encode_queue")
                && has_extension("VK_KHR_video_encode_h264");
            let video_queue_family = if video_queue_supported {
                instance
                    .get_physical_device_queue_family_properties(physical_device)
                    .iter()
                    .position(|info| {
                        info.queue_flags
                            .contains(vk::QueueFlags::from_raw(0x0000_0040))
                    })
                    .map(|index| index as u32)
            } else {
                None
            };

            let mut device_extension_names = vec![Swapchain::name().as_ptr()];
            if video_queue_supported {
                device_extension_names.push(cstr("VK_KHR_video_queue").as_ptr());
                device_extension_names.push(cstr("VK_KHR_video_encode_queue").as_ptr());
                device_extension_names.push(cstr("VK_KHR_video_encode_h264").as_ptr());
            }
            let mut features = vk::PhysicalDeviceFeatures2::builder()
                .features(vk::PhysicalDeviceFeatures {
                    fill_mode_non_solid: 1,
                    ..Default::default()
                })
                .build();
            let mut dynamic_rendering = vk::PhysicalDeviceVulkan13Features::builder()
                .dynamic_rendering(true)
                .build();
            let priorities = [1.0f32];
            let mut queue_infos = vec![vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)
                .build()];
            if let Some(video_family) = video_queue_family {
                if video_family != queue_family_index {
                    queue_infos.push(
                        vk::DeviceQueueCreateInfo::builder()
                            .queue_family_index(video_family)
                            .queue_priorities(&priorities)
                            .build(),
                    );
                }
            }
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&device_extension_names)
                .push_next(&mut features)
                .push_next(&mut dynamic_rendering);

            let device: Device = instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create the Vulkan logical device");

            let graphics_queue = device.get_device_queue(queue_family_index, 0);
            let video_queue = video_queue_family
                .map(|family| device.get_device_queue(family, 0))
                .unwrap_or(vk::Queue::null());
            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            let properties = instance.get_physical_device_properties(physical_device);
            let max_texture_size = properties.limits.max_image_dimension2_d;
            let uniform_alignment = properties.limits.min_uniform_buffer_offset_alignment;
            let color_sample_counts = properties.limits.framebuffer_color_sample_counts;
            let swapchain_loader = Swapchain::new(&instance, &device);

            let command_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::builder()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(queue_family_index),
                    None,
                )
                .expect("Failed to create the command pool");

            let descriptor_pools: Vec<vk::DescriptorPool> = (0..MAX_FRAMES_IN_FLIGHT)
                .map(|_| create_descriptor_pool(&device))
                .collect();

            // Synchronisation objects are created before `device` is moved into
            // the context below. The semaphores are re-created together with the
            // swap chain, which decides how many are needed.
            let in_flight: Vec<vk::Fence> = (0..MAX_FRAMES_IN_FLIGHT)
                .map(|_| create_fence(&device))
                .collect();

            let readback_cmd = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_buffer_count(1)
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY),
                )
                .expect("Failed to allocate the readback command buffer")[0];
            let readback_fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("Failed to create the readback fence");

            let mut context = VulkanContext {
                entry,
                instance,
                surface_loader,
                swapchain_loader,
                debug_utils_loader,
                debug_messenger,

                surface,
                physical_device,
                memory_properties,
                max_texture_size,
                uniform_alignment,
                color_sample_counts,
                device,
                queue_family_index,
                graphics_queue,
                present_queue: graphics_queue,
                video_queue_family,
                video_queue,

                command_pool,
                command_buffers: Vec::new(),

                image_available: Vec::new(),
                render_finished: Vec::new(),
                retired_semaphores: Vec::new(),
                in_flight,
                frame_wait_ns: 0,

                window_size,
                swapchain: vk::SwapchainKHR::null(),
                swapchain_suboptimal: false,
                surface_format: vk::SurfaceFormatKHR::default(),
                swapchain_extent: vk::Extent2D::default(),
                swapchain_images: Vec::new(),
                swapchain_image_views: Vec::new(),
                swapchain_image_layouts: Vec::new(),
                readback_buffer: vk::Buffer::null(),
                readback_memory: vk::DeviceMemory::null(),
                readback_mapped: std::ptr::null_mut(),
                readback_capacity: 0,
                readback_cmd,
                readback_fence,
                default_depth: Texture::empty(),
                dummy_texture: 0,

                shaders: Vec::new(),
                pipelines: Vec::new(),
                buffers: Vec::new(),
                textures: Vec::new(),
                passes: Vec::new(),
                free_passes: Vec::new(),

                descriptor_pools,
                uniform_arenas: Vec::new(),
                stream_arenas: Vec::new(),

                current_frame: 0,
                acquire_slot: 0,
                frame_wait: vk::Semaphore::null(),
                frame_wait_pending: false,
                acquired_image: None,
                recording: false,
                pass_open: false,
                frame_active: false,
                cmd_dirty: false,
                frame_submitted: false,
                offscreen: false,
                cur_samples: 1,
                cur_color_format: vk::Format::UNDEFINED,
                cur_depth_format: None,
                cur_pass: None,
                cur_target: None,
                cur_params: PipelineParams::default(),

                cur_pipeline: None,
                cur_index_buffer: None,
                cur_vertex_buffers: Vec::new(),
                viewport: vk::Rect2D::default(),
                scissor: vk::Rect2D::default(),
                pending: PendingDescriptors::default(),

                display: None,
            };
            context.textures.push(empty_texture_module());
            context.dummy_texture = context.new_texture(
                Some(&[0, 0, 0, 255]),
                TextureParams {
                    width: 1,
                    height: 1,
                    format: TextureFormat::RGBA8,
                    ..Default::default()
                },
                TextureAccess::Static,
            );
            context.create_uniform_arenas();
            context.create_stream_arenas();
            context.create_swapchain(window_size);
            context
        }
    }

    fn create_stream_arenas(&mut self) {
        const STREAM_ARENA_SIZE: vk::DeviceSize = 1 << 22;
        unsafe {
            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                let (buffer, memory) = self.create_buffer(
                    STREAM_ARENA_SIZE,
                    vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                let mapped = self
                    .device
                    .map_memory(memory, 0, STREAM_ARENA_SIZE, vk::MemoryMapFlags::empty())
                    .expect("Failed to map the stream arena") as *mut u8;
                self.stream_arenas.push(StreamArena {
                    buffer,
                    memory,
                    mapped,
                    capacity: STREAM_ARENA_SIZE,
                    offset: 0,
                    retired: Vec::new(),
                });
            }
        }
    }

    /// Create the uniform arena of every frame slot.
    fn create_uniform_arenas(&mut self) {
        unsafe {
            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                let (buffer, memory) = self.create_buffer(
                    UNIFORM_ARENA_SIZE,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                let mapped = self
                    .device
                    .map_memory(memory, 0, UNIFORM_ARENA_SIZE, vk::MemoryMapFlags::empty())
                    .expect("Failed to map the uniform arena") as *mut u8;
                self.uniform_arenas.push(UniformArena {
                    buffer,
                    memory,
                    mapped,
                    capacity: UNIFORM_ARENA_SIZE,
                    offset: 0,
                    alignment: self.uniform_alignment,
                    retired: Vec::new(),
                });
            }
        }
    }

    pub fn features(&self) -> Features {
        Features {
            instancing: true,
            alpha_texture: false,
        }
    }

    pub fn cleanup(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            let passes = std::mem::take(&mut self.passes);
            for pass in passes.iter().flatten() {
                self.delete_texture(pass.color.texture as usize);
                if let Some(depth) = pass.depth {
                    self.delete_texture(depth.texture as usize);
                }
                if let Some(resolve) = pass.resolve {
                    self.delete_texture(resolve.texture as usize);
                }
            }
            for shader in self.shaders.drain(..) {
                self.device.destroy_shader_module(shader.vertex, None);
                self.device.destroy_shader_module(shader.fragment, None);
                self.device
                    .destroy_descriptor_set_layout(shader.descriptor_set_layout, None);
                self.device.destroy_pipeline_layout(shader.pipeline_layout, None);
            }
            for arena in self.uniform_arenas.drain(..) {
                self.device.destroy_buffer(arena.buffer, None);
                self.device.free_memory(arena.memory, None);
                for (buffer, memory) in arena.retired {
                    self.device.destroy_buffer(buffer, None);
                    self.device.free_memory(memory, None);
                }
            }
            for arena in self.stream_arenas.drain(..) {
                self.device.destroy_buffer(arena.buffer, None);
                self.device.free_memory(arena.memory, None);
                for (buffer, memory) in arena.retired {
                    self.device.destroy_buffer(buffer, None);
                    self.device.free_memory(memory, None);
                }
            }
            if self.readback_buffer != vk::Buffer::null() {
                self.device.unmap_memory(self.readback_memory);
                self.device.destroy_buffer(self.readback_buffer, None);
                self.device.free_memory(self.readback_memory, None);
                self.readback_buffer = vk::Buffer::null();
            }
            self.device.destroy_fence(self.readback_fence, None);
            for pipeline in self.pipelines.drain(..) {
                for variant in pipeline.variants {
                    self.device.destroy_pipeline(variant.pipeline, None);
                }
            }
            for buffer in self.buffers.drain(..) {
                self.device.destroy_buffer(buffer.buffer, None);
                self.device.free_memory(buffer.memory, None);
            }
            self.default_depth = Texture::empty();
            for index in 0..self.textures.len() {
                self.destroy_texture(index);
            }
            self.textures.clear();

            for pool in self.descriptor_pools.drain(..) {
                self.device.destroy_descriptor_pool(pool, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);

            for semaphore in self.image_available.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.render_finished.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.retired_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for fence in self.in_flight.drain(..) {
                self.device.destroy_fence(fence, None);
            }

            for view in self.swapchain_image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            if let Some(messenger) = self.debug_messenger {
                self.debug_utils_loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }

    /// (Re)create the swap chain and everything derived from it.
    fn create_swapchain(&mut self, window_size: (u32, u32)) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.window_size = window_size;

            if !self.command_buffers.is_empty() {
                self.device
                    .free_command_buffers(self.command_pool, &self.command_buffers);
                self.command_buffers.clear();
            }
            for view in self.swapchain_image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            self.retired_semaphores.extend(self.image_available.drain(..));
            self.retired_semaphores.extend(self.render_finished.drain(..));
            if self.default_depth.texture != 0 {
                let depth = self.default_depth.texture as usize;
                self.default_depth = Texture::empty();
                self.delete_texture(depth);
            }

            let surface_capabilities = self
                .surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .unwrap();

            let formats = self
                .surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)
                .unwrap();
            // The swap chain format is what the default-frame-buffer pipelines
            // get compiled against, so any 8-bit UNORM format works. Prefer one
            // over an `_SRGB` variant, which would apply a gamma transfer the
            // offscreen colour targets do not.
            self.surface_format = *formats
                .iter()
                .find(|format| {
                    format.format == vk::Format::R8G8B8A8_UNORM
                        || format.format == vk::Format::B8G8R8A8_UNORM
                })
                .unwrap_or(&formats[0]);

            let present_modes = self
                .surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|mode| *mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let (width, height) = window_size;
            self.swapchain_extent = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D { width, height },
                _ => surface_capabilities.current_extent,
            };

            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };

            let mut image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && image_count > surface_capabilities.max_image_count
            {
                image_count = surface_capabilities.max_image_count;
            }

            let old_swapchain = self.swapchain;
            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(self.surface)
                .old_swapchain(old_swapchain)
                .min_image_count(image_count)
                .image_color_space(self.surface_format.color_space)
                .image_format(self.surface_format.format)
                .image_extent(self.swapchain_extent)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            self.swapchain = self
                .swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create the swapchain");
            if old_swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader.destroy_swapchain(old_swapchain, None);
            }
            self.swapchain_images = self
                .swapchain_loader
                .get_swapchain_images(self.swapchain)
                .unwrap();
            self.swapchain_image_layouts =
                vec![vk::ImageLayout::UNDEFINED; self.swapchain_images.len()];

            self.image_available = (0..self.swapchain_images.len())
                .map(|_| create_semaphore(&self.device))
                .collect();
            self.render_finished = (0..self.swapchain_images.len())
                .map(|_| create_semaphore(&self.device))
                .collect();
            self.acquire_slot = 0;

            self.swapchain_image_views = self
                .swapchain_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    self.device.create_image_view(&create_view_info, None).unwrap()
                })
                .collect();

            self.command_buffers = self
                .swapchain_images
                .iter()
                .map(|_| {
                    let allocate_info = vk::CommandBufferAllocateInfo::builder()
                        .command_buffer_count(1)
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY);
                    self.device.allocate_command_buffers(&allocate_info).unwrap()[0]
                })
                .collect();

            // A depth attachment matching the swap chain, for `Camera2D::depth_enabled`.
            self.default_depth = Texture {
                texture: self.new_texture(
                    None,
                    TextureParams {
                        width: self.swapchain_extent.width,
                        height: self.swapchain_extent.height,
                        format: TextureFormat::Depth,
                        ..Default::default()
                    },
                    TextureAccess::RenderTarget,
                ) as GLuint,
                width: self.swapchain_extent.width,
                height: self.swapchain_extent.height,
                format: TextureFormat::Depth,
                wrap: TextureWrap::Clamp,
                filter: FilterMode::Linear,
                mipmap_filter: MipmapFilter::None,
            };
        }
    }

    pub fn texture_image_raw(&self, index: usize) -> u64 {
        unsafe { std::mem::transmute(self.textures[index].image) }
    }

    pub fn texture_layout(&self, index: usize) -> vk::ImageLayout {
        self.textures[index].layout
    }

    pub fn copy_nv12_from_textures(
        &mut self,
        y_index: usize,
        uv_index: usize,
        nv12_raw: u64,
    ) -> vk::ImageLayout {
        self.flush_recording();
        unsafe {
            let nv12 = vk::Image::from_raw(nv12_raw);
            let command_buffer = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0];
            self.device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
            let y = self.textures[y_index].image;
            let uv = self.textures[uv_index].image;
            let y_layout = self.textures[y_index].layout;
            let uv_layout = self.textures[uv_index].layout;
            let color = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let to_src = |image, layout| {
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(layout)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(color)
                    .build()
            };
            let to_dst = vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(nv12)
                .subresource_range(color)
                .build();
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_src(y, y_layout), to_src(uv, uv_layout), to_dst],
            );
            let width = self.textures[y_index].width;
            let height = self.textures[y_index].height;
            let copies = [vk::ImageCopy {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offset: vk::Offset3D::default(),
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_0,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offset: vk::Offset3D::default(),
                extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            }];
            self.device.cmd_copy_image(
                command_buffer,
                y,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                nv12,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copies,
            );
            let copies = [vk::ImageCopy {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offset: vk::Offset3D::default(),
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_1,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offset: vk::Offset3D::default(),
                extent: vk::Extent3D {
                    width: width / 2,
                    height: height / 2,
                    depth: 1,
                },
            }];
            self.device.cmd_copy_image(
                command_buffer,
                uv,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                nv12,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copies,
            );
            let back_src = |image, layout| {
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(color)
                    .build()
            };
            let back_dst = vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(nv12)
                .subresource_range(color)
                .build();
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[back_src(y, y_layout), back_src(uv, uv_layout), back_dst],
            );
            self.device.end_command_buffer(command_buffer).unwrap();
            let command_buffers = [command_buffer];
            let submit = vk::SubmitInfo::builder().command_buffers(&command_buffers);
            self.device
                .queue_submit(self.graphics_queue, &[submit.build()], vk::Fence::null())
                .unwrap();
            self.device.queue_wait_idle(self.graphics_queue).unwrap();
            self.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
        }
        self.release_frame_resources();
        vk::ImageLayout::GENERAL
    }

    pub fn set_window_size(&mut self, width: u32, height: u32) {
        self.window_size = (width, height);
    }

    pub(crate) fn recreate_swapchain(&mut self) {
        let size = self.window_size;
        self.create_swapchain(size);
    }

    pub(crate) fn allocate_descriptor_set(
        &mut self,
        layout: vk::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        unsafe {
            let info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(self.descriptor_pools[self.current_frame])
                .set_layouts(std::slice::from_ref(&layout));
            self.device
                .allocate_descriptor_sets(&info)
                .expect("Failed to allocate a descriptor set")[0]
        }
    }

    /// Release every descriptor set the frame's pool handed out.
    ///
    /// Descriptor sets stay referenced by the command buffers they are bound
    /// in, so this may only run once the frame fence has been waited on.
    pub(crate) fn free_allocated_sets(&mut self) {
        unsafe {
            self.device
                .reset_descriptor_pool(
                    self.descriptor_pools[self.current_frame],
                    vk::DescriptorPoolResetFlags::empty(),
                )
                .expect("Failed to reset the descriptor pool");
        }
    }
}

pub(crate) fn cstr(name: &str) -> &'static CStr {
    let mut bytes = name.as_bytes().to_vec();
    bytes.push(0);
    CStr::from_bytes_with_nul(bytes.leak())
        .expect("layer and extension names must not contain interior nul bytes")
}

pub(crate) fn create_semaphore(device: &Device) -> vk::Semaphore {
    unsafe {
        device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap()
    }
}

pub(crate) fn create_fence(device: &Device) -> vk::Fence {
    unsafe {
        device
            .create_fence(
                &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
            .unwrap()
    }
}

pub(crate) fn create_descriptor_pool(device: &Device) -> vk::DescriptorPool {
    // A set is allocated per draw call and the pool is reset once per frame, so
    // the budget has to cover the busiest frame a scene can produce.
    const MAX_SETS_PER_FRAME: u32 = 1 << 16;
    let sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: MAX_SETS_PER_FRAME,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: MAX_SETS_PER_FRAME * MAX_SHADERSTAGE_IMAGES as u32,
        },
    ];
    unsafe {
        let info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(MAX_SETS_PER_FRAME)
            .pool_sizes(&sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        device.create_descriptor_pool(&info, None).unwrap()
    }
}


/// Index of a memory type satisfying `type_bits` and `properties`.
pub(crate) fn find_memory_type_index(
    properties: vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for index in 0..properties.memory_type_count {
        if type_bits & (1 << index) != 0
            && properties.memory_types[index as usize].property_flags.contains(required)
        {
            return Some(index);
        }
    }
    None
}
