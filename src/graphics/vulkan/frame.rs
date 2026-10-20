use std::ptr;use ash::vk;use ash::vk::Handle;use crate::graphics::*;
use super::resource::vk_index_type;
use super::{VulkanContext, DEPTH_FORMAT, IMAGE_BINDING_BASE, MAX_FRAMES_IN_FLIGHT, UNIFORM_BINDING};

fn source_scope(layout: vk::ImageLayout) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match layout {
        vk::ImageLayout::UNDEFINED => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
        ),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
    }
}

impl VulkanContext {
    pub(crate) fn record_submit<F>(&mut self, f: F)
    where
        F: FnOnce(&VulkanContext, vk::CommandBuffer),
    {
        self.flush_recording();
        unsafe {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);
            let command_buffer = self
                .device
                .allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate a command buffer")[0];

            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin the command buffer");

            f(self, command_buffer);

            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end the command buffer");

            let command_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(&command_buffers);
            self.device
                .queue_submit(self.graphics_queue, &[submit_info.build()], vk::Fence::null())
                .expect("Failed to submit the command buffer");
            self.device
                .queue_wait_idle(self.graphics_queue)
                .expect("Failed to wait for the command buffer");

            self.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
        }
        self.release_frame_resources();
    }

    /// Submit everything recorded so far and start a new command buffer.
    ///
    /// Some callers read a render target back in the middle of a frame (video
    /// export renders every frame without yielding to the event loop). Their
    /// copy commands are submitted on their own, so the draws they depend on
    /// have to reach the queue first.
    pub(crate) fn flush_recording(&mut self) -> bool {
        if !self.recording || !self.frame_active || !self.cmd_dirty {
            return false;
        }
        let image_index = self.acquired_image.unwrap();
        self.close_pass();
        let command_buffer = self.command_buffers[image_index];
        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end the command buffer");
            let command_buffers = [command_buffer];
            let wait_semaphores = [self.frame_wait];
            let wait_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = if self.frame_wait_pending {
                vk::SubmitInfo::builder()
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&wait_mask)
                    .command_buffers(&command_buffers)
                    .build()
            } else {
                vk::SubmitInfo::builder()
                    .command_buffers(&command_buffers)
                    .build()
            };
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], vk::Fence::null())
                .expect("Failed to submit the recorded frame");
        }
        self.frame_wait_pending = false;
        self.cmd_dirty = false;
        self.frame_submitted = true;
        true
    }

    /// Release the per-frame pools and start recording again, after the GPU has
    /// finished the submission `flush_recording` made.
    pub(crate) fn release_frame_resources(&mut self) {
        if !self.frame_submitted {
            return;
        }
        let image_index = self.acquired_image.unwrap();
        unsafe {
            self.free_allocated_sets();
            let slot = self.current_frame;
            for (buffer, memory) in self.uniform_arenas[slot].retired.drain(..) {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            self.uniform_arenas[slot].offset = 0;
            for (buffer, memory) in self.stream_arenas[slot].retired.drain(..) {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            self.stream_arenas[slot].offset = 0;

            let command_buffer = self.command_buffers[image_index];
            self.device
                .reset_command_buffer(
                    command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Failed to reset the command buffer");
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin the command buffer");
        }
        self.frame_submitted = false;
    }

    fn begin_frame(&mut self) {
        unsafe {
            let fence = self.in_flight[self.current_frame];
            let wait_start = std::time::Instant::now();
            self.device
                .wait_for_fences(&[fence], true, std::u64::MAX)
                .expect("Failed to wait for the frame fence");
            self.frame_wait_ns = self
                .frame_wait_ns
                .saturating_add(wait_start.elapsed().as_nanos() as u64);
            self.free_allocated_sets();

            // The frame that used this slot has finished, so the arenas it
            // outgrew are idle and the slot starts over with an empty one.
            let slot = self.current_frame;
            for (buffer, memory) in self.uniform_arenas[slot].retired.drain(..) {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            self.uniform_arenas[slot].offset = 0;
            for (buffer, memory) in self.stream_arenas[slot].retired.drain(..) {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            self.stream_arenas[slot].offset = 0;

            self.device
                .reset_fences(&[fence])
                .expect("Failed to reset the frame fence");
            let (image_index, slot, suboptimal) = loop {
                let slot = self.acquire_slot;
                self.frame_wait = self.image_available[slot];
                match self.swapchain_loader.acquire_next_image(
                    self.swapchain,
                    std::u64::MAX,
                    self.image_available[slot],
                    vk::Fence::null(),
                ) {
                    Ok((image_index, suboptimal)) => break (image_index, slot, suboptimal),
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        self.recreate_swapchain();
                    }
                    Err(error) => panic!("Failed to acquire a swapchain image: {:?}", error),
                }
            };
            let image_index = image_index as usize;
            self.swapchain_suboptimal = suboptimal;

            self.acquire_slot = (slot + 1) % self.image_available.len();
            self.acquired_image = Some(image_index);
            // The presentation engine owned the image until now, so whatever it
            // holds is undefined.
            self.swapchain_image_layouts[image_index] = vk::ImageLayout::UNDEFINED;

            let command_buffer = self.command_buffers[image_index];
            self.device
                .reset_command_buffer(
                    command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Failed to reset the command buffer");
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin the command buffer");

            self.frame_wait_pending = true;
            self.recording = true;
            self.frame_active = true;
            self.cmd_dirty = false;
        }

        self.cur_pipeline = None;
        self.cur_index_buffer = None;
        self.cur_vertex_buffers.clear();
        self.pending.clear();
    }

    /// Switch this context to offscreen video export.
    ///
    /// In offscreen mode a frame is recorded into a command-buffer ring indexed
    /// by `current_frame`; no swap-chain image is acquired and nothing is
    /// presented.  The ring is what lets the exporter keep several frames in
    /// flight instead of blocking on a readback fence every frame.
    ///
    /// Must be called once, before the first `begin_offscreen_frame`, while the
    /// device is idle.
    pub fn enable_offscreen_video(&mut self) {
        if self.offscreen {
            return;
        }
        let command_buffers = unsafe {
            self.device.device_wait_idle().unwrap();
            if self.pass_open && self.acquired_image.is_some() {
                self.close_pass();
            }
            for command_buffer in &self.command_buffers {
                let _ = self.device.reset_command_buffer(
                    *command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                );
            }
            let info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);
            self.device.allocate_command_buffers(&info).unwrap()
        };
        unsafe {
            if !self.command_buffers.is_empty() {
                self.device
                    .free_command_buffers(self.command_pool, &self.command_buffers);
            }
        }
        self.command_buffers = command_buffers;
        // An incomplete normal frame can leave a per-frame fence reset without
        // a matching submission.  The device is idle here, so replace the ring
        // with fresh signaled fences before the offscreen ring starts using
        // them.
        let old_fences = std::mem::take(&mut self.in_flight);
        for fence in old_fences {
            unsafe { self.device.destroy_fence(fence, None) };
        }
        self.in_flight = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| super::create_fence(&self.device))
            .collect();
        self.offscreen = true;
        self.current_frame = 0;
        self.acquired_image = None;
        self.frame_active = false;
        self.recording = false;
        self.pass_open = false;
        self.cmd_dirty = false;
        self.frame_submitted = false;
    }

    /// Start recording one offscreen frame.  The returned slot is the index in
    /// the `MAX_FRAMES_IN_FLIGHT` ring.
    ///
    /// The previous use of this slot is waited on here; that is the only
    /// per-frame stall, and it is `MAX_FRAMES_IN_FLIGHT - 1` frames behind the
    /// frame currently being recorded.
    pub fn begin_offscreen_frame(&mut self) -> usize {
        assert!(self.offscreen, "enable_offscreen_video() was not called");
        assert!(
            !self.frame_active,
            "the previous offscreen frame was not finished"
        );
        let slot = self.current_frame;
        unsafe {
            let fence = self.in_flight[slot];
            match self.device.wait_for_fences(&[fence], true, 10_000_000_000) {
                Ok(()) => {}
                Err(e) => panic!("offscreen frame fence wait failed/timeout: {:?}", e),
            }
            self.free_allocated_sets();
            for (buffer, memory) in self.uniform_arenas[slot].retired.drain(..) {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            self.uniform_arenas[slot].offset = 0;
            for (buffer, memory) in self.stream_arenas[slot].retired.drain(..) {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            self.stream_arenas[slot].offset = 0;
            self.device
                .reset_fences(&[fence])
                .expect("Failed to reset the offscreen frame fence");
            let command_buffer = self.command_buffers[slot];
            self.device
                .reset_command_buffer(
                    command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Failed to reset the offscreen command buffer");
            self.device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("Failed to begin the offscreen command buffer");
        }
        self.acquired_image = Some(slot);
        self.frame_wait_pending = false;
        self.recording = true;
        self.frame_active = true;
        self.cmd_dirty = false;
        self.frame_submitted = false;
        slot
    }

    /// Record the GPU Y/UV planes into an app-owned NV12 image and copy that
    /// image into an app-owned host-visible staging buffer.
    ///
    /// The commands go into the current offscreen frame's command buffer, so
    /// they are submitted together with the scene at `finish_offscreen_frame`.
    /// The staging buffer contains tightly packed NV12 (Y plane then
    /// interleaved UV plane), which is exactly what ffmpeg's `nv12` rawvideo
    /// input expects.
    pub fn record_nv12_copy_to_staging(
        &mut self,
        y_index: usize,
        uv_index: usize,
        nv12_image_raw: u64,
        staging_buffer_raw: u64,
        width: u32,
        height: u32,
    ) {
        assert!(
            self.offscreen && self.frame_active,
            "record_nv12_copy_to_staging() requires an active offscreen frame"
        );
        // Transfer commands cannot be recorded inside a dynamic rendering
        // instance.
        self.close_pass();
        let command_buffer = self.command_buffers[self.acquired_image.unwrap()];
        unsafe {
            let nv12 = vk::Image::from_raw(nv12_image_raw);
            let staging = vk::Buffer::from_raw(staging_buffer_raw);
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
            let to_transfer_src = |image, layout| {
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
            let back_to_attachment = |image, layout| {
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
            let nv12_barrier = |old, new, src, dst| {
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(src)
                    .dst_access_mask(dst)
                    .old_layout(old)
                    .new_layout(new)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(nv12)
                    .subresource_range(color)
                    .build()
            };

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    to_transfer_src(y, y_layout),
                    to_transfer_src(uv, uv_layout),
                    nv12_barrier(
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                        vk::AccessFlags::TRANSFER_WRITE,
                    ),
                ],
            );

            let y_copy = vk::ImageCopy::builder()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_0,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .build();
            self.device.cmd_copy_image(
                command_buffer,
                y,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                nv12,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[y_copy],
            );
            let uv_copy = vk::ImageCopy::builder()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_1,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .extent(vk::Extent3D {
                    width: width / 2,
                    height: height / 2,
                    depth: 1,
                })
                .build();
            self.device.cmd_copy_image(
                command_buffer,
                uv,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                nv12,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[uv_copy],
            );

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[nv12_barrier(
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                )],
            );

            let nv12_regions = [
                vk::BufferImageCopy::builder()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::PLANE_0,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .build(),
                vk::BufferImageCopy::builder()
                    .buffer_offset(width as u64 * height as u64)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::PLANE_1,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: width / 2,
                        height: height / 2,
                        depth: 1,
                    })
                    .build(),
            ];
            self.device.cmd_copy_image_to_buffer(
                command_buffer,
                nv12,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging,
                &nv12_regions,
            );

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    nv12_barrier(
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::ImageLayout::GENERAL,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                    ),
                    back_to_attachment(y, y_layout),
                    back_to_attachment(uv, uv_layout),
                ],
            );
        }
    }

    /// Copy the Y/UV render targets straight into an NV12 staging buffer.
    ///
    /// The old exporter first copied both planes into an intermediate NV12
    /// image and then copied that image to the buffer.  The intermediate copy
    /// is unnecessary: Y is R8, UV is RG8, so two `cmd_copy_image_to_buffer`
    /// calls already produce exactly the NV12 layout ffmpeg expects.
    pub fn record_yuv_copy_to_staging(
        &mut self,
        y_index: usize,
        uv_index: usize,
        staging_buffer_raw: u64,
        width: u32,
        height: u32,
    ) {
        assert!(
            self.offscreen && self.frame_active,
            "record_yuv_copy_to_staging() requires an active offscreen frame"
        );
        self.close_pass();
        let command_buffer = self.command_buffers[self.acquired_image.unwrap()];
        unsafe {
            let staging = vk::Buffer::from_raw(staging_buffer_raw);
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
            let to_transfer_src = |image, layout| {
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
            let back_to_attachment = |image, layout| {
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
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer_src(y, y_layout), to_transfer_src(uv, uv_layout)],
            );
            let y_region = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D { width, height, depth: 1 })
                .build();
            let uv_region = vk::BufferImageCopy::builder()
                .buffer_offset(width as u64 * height as u64)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: width / 2,
                    height: height / 2,
                    depth: 1,
                })
                .build();
            self.device.cmd_copy_image_to_buffer(
                command_buffer,
                y,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging,
                &[y_region],
            );
            self.device.cmd_copy_image_to_buffer(
                command_buffer,
                uv,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging,
                &[uv_region],
            );
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[back_to_attachment(y, y_layout), back_to_attachment(uv, uv_layout)],
            );
        }
    }

    /// Close and submit the offscreen frame.  The GPU is not waited on here;
    /// the fence is consumed by `begin_offscreen_frame` when this ring slot is
    /// reused.
    pub fn finish_offscreen_frame(&mut self) {
        if !self.frame_active {
            return;
        }
        self.close_pass();
        let slot = self.acquired_image.take().unwrap();
        let command_buffer = self.command_buffers[slot];
        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end the offscreen command buffer");
            let command_buffers = [command_buffer];
            let submit = vk::SubmitInfo::builder().command_buffers(&command_buffers);
            self.device
                .queue_submit(
                    self.graphics_queue,
                    &[submit.build()],
                    self.in_flight[slot],
                )
                .expect("Failed to submit the offscreen frame");
        }
        self.current_frame = (slot + 1) % MAX_FRAMES_IN_FLIGHT;
        self.recording = false;
        self.frame_active = false;
        self.pass_open = false;
        self.cmd_dirty = false;
        self.frame_submitted = true;
    }

    /// Wait for every offscreen frame still in flight (used at the end of an
    /// export so the last staging buffers can be drained).
    pub fn wait_offscreen_idle(&mut self) {
        if !self.offscreen {
            return;
        }
        let fences = self.in_flight.clone();
        unsafe {
            self.device
                .wait_for_fences(&fences, true, u64::MAX)
                .expect("Failed to wait for offscreen frames");
        }
    }

    pub fn begin_pass(&mut self, pass: Option<RenderPass>, action: PassAction) {
        if self.offscreen && pass.is_none() {
            // The exporter renders into its own render targets; the swapchain
            // image is never acquired in offscreen mode, so a default pass
            // would transition an image that is not ours.
            return;
        }
        if !self.frame_active {
            self.begin_frame();
        }
        let target = pass.map(|pass| pass.0);
        // MiniQuad's passes only rebind the target: everything drawn between two
        // `begin_pass` calls on the same target belongs to one pass. Keeping the
        // rendering instance open is also what orders the draws of a frame —
        // nothing orders the contents of two separate instances.
        if self.pass_open && self.cur_pass == target && matches!(action, PassAction::Nothing) {
            return;
        }
        if self.pass_open {
            self.close_pass();
        }
        self.transition_sampled_textures();

        let (color_view, resolve_view, depth_view, samples, width, height, color_format, depth_format) = match pass {
            None => {
                self.transition_swapchain_image();
                (
                    self.swapchain_image_views[self.acquired_image.unwrap()],
                    vk::ImageView::null(),
                    self.textures[self.default_depth.texture as usize].view,
                    1,
                    self.swapchain_extent.width,
                    self.swapchain_extent.height,
                    self.surface_format.format,
                    DEPTH_FORMAT,
                )
            }
            Some(pass) => {
                self.transition_attachments(pass.0);
                let module = self.passes[pass.0]
                    .as_ref()
                    .expect("the render pass has been deleted");
                (
                    self.textures[module.color.texture as usize].view,
                    module
                        .resolve
                        .map_or(vk::ImageView::null(), |resolve| {
                            self.textures[resolve.texture as usize].view
                        }),
                    module.depth.map_or(vk::ImageView::null(), |depth| {
                        self.textures[depth.texture as usize].view
                    }),
                    module.samples,
                    module.color.width,
                    module.color.height,
                    self.textures[module.color.texture as usize].format,
                    module
                        .depth
                        .map_or(DEPTH_FORMAT, |depth| self.textures[depth.texture as usize].format),
                )
            }
        };
        self.cur_samples = samples;
        self.cur_color_format = color_format;
        self.cur_depth_format = if depth_view == vk::ImageView::null() {
            None
        } else {
            Some(depth_format)
        };
        self.cur_target = target;

        let mut clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let mut clear_depth = vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        };
        let load_op = match action {
            PassAction::Nothing => vk::AttachmentLoadOp::LOAD,
            PassAction::Clear {
                color: clear_color,
                depth,
                stencil,
            } => {
                if let Some((r, g, b, a)) = clear_color {
                    clear.color = vk::ClearColorValue {
                        float32: [r, g, b, a],
                    };
                }
                if let Some(v) = depth {
                    clear_depth.depth = v;
                }
                if let Some(v) = stencil {
                    clear_depth.stencil = v as u32;
                }
                vk::AttachmentLoadOp::CLEAR
            }
        };

        let color_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        let has_resolve = resolve_view != vk::ImageView::null();
        let color_attachments = [
            vk::RenderingAttachmentInfo::builder()
                .image_view(color_view)
                .image_layout(color_layout)
                .resolve_mode(if has_resolve {
                    vk::ResolveModeFlags::AVERAGE
                } else {
                    vk::ResolveModeFlags::NONE
                })
                .resolve_image_view(resolve_view)
                .resolve_image_layout(if has_resolve {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                })
                .load_op(load_op)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear)
                .build(),
        ];
        let has_depth = depth_view != vk::ImageView::null();
        let depth_attachments = [
            vk::RenderingAttachmentInfo::builder()
                .image_view(depth_view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(load_op)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: clear_depth,
                })
                .build(),
        ];

        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let rendering_info = vk::RenderingInfo {
            render_area: area,
            layer_count: 1,
            color_attachment_count: 1,
            p_color_attachments: color_attachments.as_ptr(),
            p_depth_attachment: if has_depth {
                depth_attachments.as_ptr()
            } else {
                std::ptr::null()
            },
            p_stencil_attachment: std::ptr::null(),
            ..Default::default()
        };

        unsafe {
            self.device.cmd_begin_rendering(
                self.command_buffers[self.acquired_image.unwrap()],
                &rendering_info,
            );
        }

        self.pass_open = true;
        self.cur_pass = pass.map(|pass| pass.0);
        self.viewport = area;
        self.scissor = area;
        self.cmd_dirty = true;
        self.pending.clear();
    }

    /// The OpenGL backend leaves the framebuffer bound here, so the rendering
    /// instance stays open until a `begin_pass` needs a different target.
    pub fn end_render_pass(&mut self) {}

    fn close_pass(&mut self) {
        if !self.pass_open {
            return;
        }
        unsafe {
            self.device
                .cmd_end_rendering(self.command_buffers[self.acquired_image.unwrap()]);
        }
        self.pass_open = false;
        self.cur_pass = None;
        self.cmd_dirty = true;
    }

    /// `vkCmdBeginRendering` has no `initialLayout` to transition from, so the
    /// acquired swap chain image has to be moved into attachment layout
    /// explicitly. Its previous contents are never needed: the first pass of
    /// every frame clears.
    fn transition_swapchain_image(&mut self) {
        let index = self.acquired_image.unwrap();
        let image = self.swapchain_images[index];
        // Only the first pass of a frame may come from `UNDEFINED`; later passes
        // load what the earlier ones wrote, so their contents have to survive.
        let old_layout = self.swapchain_image_layouts[index];
        if old_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            return;
        }
        let command_buffer = self.command_buffers[index];
        let barrier = vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::MEMORY_READ)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();
        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
        self.swapchain_image_layouts[index] = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    }

    /// Move a render target's colour and resolve images into attachment layout.
    ///
    /// The other half of `transition_sampled_textures`: a target may have been
    /// sampled (or written) by an earlier pass, and `vkCmdBeginRendering` does
    /// not transition on the application's behalf.
    fn transition_attachments(&mut self, pass: usize) {
        let command_buffer = self.command_buffers[self.acquired_image.unwrap()];
        let module = self.passes[pass]
            .as_ref()
            .expect("the render pass has been deleted");

        let mut indexes = vec![module.color.texture as usize];
        if let Some(resolve) = module.resolve {
            indexes.push(resolve.texture as usize);
        }

        for index in indexes {
            let module = &self.textures[index];
            if module.layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
                continue;
            }
            let (src_stage, src_access) = source_scope(module.layout);
            let barrier = vk::ImageMemoryBarrier::builder()
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(module.layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(module.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            unsafe {
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    src_stage,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
            self.textures[index].layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        }
    }

    pub fn clear(
        &mut self,
        color: Option<(f32, f32, f32, f32)>,
        depth: Option<f32>,
        stencil: Option<i32>,
    ) {
        self.begin_pass(
            None,
            PassAction::Clear {
                color,
                depth,
                stencil,
            },
        );
        self.end_render_pass();
    }

    pub fn apply_pipeline(&mut self, pipeline: &Pipeline) {
        self.cur_pipeline = Some(*pipeline);
        self.cur_params = self.pipelines[pipeline.0].params;
    }

    pub fn set_cull_face(&mut self, cull_face: CullFace) {
        self.cur_params.cull_face = cull_face;
    }

    pub fn set_color_write(&mut self, color_write: (bool, bool, bool, bool)) {
        self.cur_params.color_write = color_write;
    }

    pub fn set_blend(&mut self, color_blend: Option<BlendState>, alpha_blend: Option<BlendState>) {
        if color_blend.is_none() && alpha_blend.is_some() {
            panic!("AlphaBlend without ColorBlend");
        }
        self.cur_params.color_blend = color_blend;
        self.cur_params.alpha_blend = alpha_blend;
    }

    pub fn set_stencil(&mut self, stencil_test: Option<StencilState>) {
        self.cur_params.stencil_test = stencil_test;
    }

    pub fn apply_viewport(&mut self, x: i32, y: i32, w: i32, h: i32) {
        self.viewport = vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D {
                width: w.max(0) as u32,
                height: h.max(0) as u32,
            },
        };
    }

    pub fn apply_scissor_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        self.scissor = vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D {
                width: w.max(0) as u32,
                height: h.max(0) as u32,
            },
        };
    }

    pub fn apply_bindings(&mut self, bindings: &Bindings) {
        let pipeline = self
            .cur_pipeline
            .expect("apply_bindings without a bound pipeline");
        let shader = self.pipelines[pipeline.0].shader;
        let images = self.shaders[shader].images.len();

        if bindings.vertex_buffers.len() > self.cur_vertex_buffers.len() {
            self.cur_vertex_buffers.resize(bindings.vertex_buffers.len(), None);
        }
        for (slot, buffer) in bindings.vertex_buffers.iter().enumerate() {
            self.cur_vertex_buffers[slot] = Some(buffer.gl_buf as usize);
        }
        self.cur_index_buffer = Some(bindings.index_buffer.gl_buf as usize);

        let sampled: Vec<usize> = bindings
            .images
            .iter()
            .take(images)
            .filter(|texture| texture.texture != 0)
            .map(|texture| texture.texture as usize)
            .collect();
        self.pending.sampled.extend(sampled.iter().copied());
        let needs_transition = sampled
            .iter()
            .any(|&index| self.textures[index].layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        if needs_transition && self.pass_open {
            let target = self.cur_target;
            let viewport = self.viewport;
            let scissor = self.scissor;
            self.close_pass();
            self.begin_pass(target.map(RenderPass), PassAction::Nothing);
            self.viewport = viewport;
            self.scissor = scissor;
        }

        self.pending.images = vec![None; images];
        for (slot, texture) in bindings.images.iter().take(images).enumerate() {
            if texture.texture == 0 {
                continue;
            }
            let module = &self.textures[texture.texture as usize];
            self.pending.images[slot] = Some(vk::DescriptorImageInfo {
                sampler: module.sampler,
                image_view: module.sample_view.unwrap_or(module.view),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            });
        }
    }

    /// Move the textures the pass just sampled into `SHADER_READ_ONLY_OPTIMAL`.
    ///
    /// Recorded after `cmdEndRendering` because Vulkan rejects image barriers
    /// inside a dynamic rendering instance.
    fn transition_sampled_textures(&mut self) {
        let command_buffer = self.command_buffers[self.acquired_image.unwrap()];
        for index in std::mem::take(&mut self.pending.sampled) {
            let (image, from, aspect_mask) = {
                let module = &self.textures[index];
                let aspect_mask = if module.format == DEPTH_FORMAT {
                    vk::ImageAspectFlags::DEPTH
                } else {
                    vk::ImageAspectFlags::COLOR
                };
                (module.image, module.layout, aspect_mask)
            };
            if from == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
                continue;
            }
            let (src_stage, src_access) = source_scope(from);
            let barrier = vk::ImageMemoryBarrier::builder()
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(from)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            unsafe {
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    src_stage,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
            self.textures[index].layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        }
    }

    pub fn apply_uniforms_from_bytes(&mut self, uniform_ptr: *const u8, size: usize) {
        let pipeline = self
            .cur_pipeline
            .expect("apply_uniforms without a bound pipeline");
        let shader = self.pipelines[pipeline.0].shader;
        let (buffer_size, offsets) = {
            let module = &self.shaders[shader];
            (module.uniforms_size, module.uniform_offsets.clone())
        };
        if buffer_size == 0 {
            return;
        }

        // Everything written between two draws shares one slice, so the last
        // call before the draw is the one that ends up in it.
        let offset = match self.pending.uniform_offset {
            Some(offset) => offset,
            None => {
                let offset = self.allocate_uniforms(buffer_size);
                self.pending.uniform_offset = Some(offset);
                offset
            }
        };

        unsafe {
            let mapped = self.uniform_arenas[self.current_frame]
                .mapped
                .add(offset as usize);
            for (tight, std140, element, stride, count) in offsets.iter() {
                assert!(
                    *tight as usize + *count as usize * *element as usize <= size,
                    "Uniforms struct does not match shader uniforms layout"
                );
                for index in 0..*count {
                    ptr::copy_nonoverlapping(
                        uniform_ptr.add((*tight + index * *element) as usize),
                        mapped.add((*std140 + index * *stride) as usize),
                        *element as usize,
                    );
                }
            }
        }
    }

    /// Reserve a slice of the frame's uniform arena for one draw call.
    ///
    /// The arena is a bump allocator: slices are aligned to what the device
    /// requires of a dynamic uniform offset and are all released when the frame
    /// ends.
    fn allocate_uniforms(&mut self, size: vk::DeviceSize) -> vk::DeviceSize {
        let slot = self.current_frame;
        let alignment = self.uniform_arenas[slot].alignment;
        let offset = (self.uniform_arenas[slot].offset + alignment - 1) & !(alignment - 1);
        if offset + size > self.uniform_arenas[slot].capacity {
            self.grow_uniform_arena(slot, offset + size);
        }
        self.uniform_arenas[slot].offset = offset + size;
        offset
    }

    /// Enlarge a frame slot's arena, carrying the slices it handed out already
    /// over to the new one.
    fn grow_uniform_arena(&mut self, slot: usize, required: vk::DeviceSize) {
        unsafe {
            let mut capacity = self.uniform_arenas[slot].capacity;
            while capacity < required {
                capacity *= 2;
            }
            let used = self.uniform_arenas[slot].offset;
            let (buffer, memory) = self.create_buffer(
                capacity,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let mapped = self
                .device
                .map_memory(memory, 0, capacity, vk::MemoryMapFlags::empty())
                .expect("Failed to map the uniform arena") as *mut u8;

            let arena = &mut self.uniform_arenas[slot];
            ptr::copy_nonoverlapping(arena.mapped, mapped, used as usize);
            self.device.unmap_memory(arena.memory);
            arena.retired.push((arena.buffer, arena.memory));
            arena.buffer = buffer;
            arena.memory = memory;
            arena.mapped = mapped;
            arena.capacity = capacity;
        }
    }

    pub fn draw(&mut self, base_element: i32, num_elements: i32, num_instances: i32) {
        let pipeline = self.cur_pipeline.expect("Drawing without a bound pipeline");

        let index_buffer = self.cur_index_buffer.expect("Unset index buffer");

        let shader_index = self.pipelines[pipeline.0].shader;
        let vk_pipeline = self.pipeline_for_target(
            pipeline,
            self.cur_color_format,
            self.cur_samples,
            self.cur_depth_format,
            self.cur_params,
        );
        let shader_layout = self.shaders[shader_index].descriptor_set_layout;
        let uniforms_size = self.shaders[shader_index].uniforms_size;
        let images = self.shaders[shader_index].images.len();

        let uniform_offset = match self.pending.uniform_offset {
            Some(offset) => Some(offset),
            None if uniforms_size > 0 => {
                let offset = self.allocate_uniforms(uniforms_size);
                unsafe {
                    ptr::write_bytes(
                        self.uniform_arenas[self.current_frame]
                            .mapped
                            .add(offset as usize),
                        0,
                        uniforms_size as usize,
                    );
                }
                Some(offset)
            }
            None => None,
        };

        // Flush the pending descriptor writes into a set for this draw.
        let descriptor_set = self.allocate_descriptor_set(shader_layout);

        unsafe {
            let shader_module = &self.shaders[shader_index];
            let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

            let buffer_infos: Vec<vk::DescriptorBufferInfo> = if uniforms_size > 0 {
                vec![vk::DescriptorBufferInfo {
                    buffer: self.uniform_arenas[self.current_frame].buffer,
                    offset: 0,
                    range: uniforms_size,
                }]
            } else {
                Vec::new()
            };
            if !buffer_infos.is_empty() {
                writes.push(vk::WriteDescriptorSet {
                    dst_set: descriptor_set,
                    dst_binding: UNIFORM_BINDING,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    p_buffer_info: buffer_infos.as_ptr(),
                    ..Default::default()
                });
            }

            let dummy = &self.textures[self.dummy_texture];
            let dummy_info = vk::DescriptorImageInfo {
                sampler: dummy.sampler,
                image_view: dummy.sample_view.unwrap_or(dummy.view),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            let mut infos: Vec<vk::DescriptorImageInfo> = Vec::with_capacity(images);
            for slot in 0..images {
                infos.push(self.pending.images[slot].unwrap_or(dummy_info));
            }
            for (slot, _) in infos.iter().enumerate() {
                writes.push(vk::WriteDescriptorSet {
                    dst_set: descriptor_set,
                    dst_binding: IMAGE_BINDING_BASE + slot as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: infos.as_ptr().add(slot),
                    ..Default::default()
                });
            }

            if !writes.is_empty() {
                self.device.update_descriptor_sets(&writes, &[]);
            }

            let cmd = self.command_buffers[self.acquired_image.unwrap()];
            let index_module = &self.buffers[index_buffer];
            let index_type = index_module.index_type.expect("Unset index buffer type");

            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, vk_pipeline);
            // OpenGL clip space has +Y up, Vulkan has +Y down. A viewport whose origin
            // sits at its bottom edge with a negative height maps the former
            // onto the latter without touching the projection matrix, which is how the
            // swapchain has to be drawn: it is presented with its first row on top.
            // A render target instead keeps its first row at the bottom of the picture
            // the way an OpenGL framebuffer does, so offscreen passes draw with the
            // plain Vulkan viewport and sample their texture the OpenGL way.
            let screen = self.cur_target.is_none();
            let height = self.viewport.extent.height as f32;
            let viewport = vk::Viewport {
                x: self.viewport.offset.x as f32,
                y: if screen {
                    self.viewport.offset.y as f32 + height
                } else {
                    self.viewport.offset.y as f32
                },
                width: self.viewport.extent.width as f32,
                height: if screen { -height } else { height },
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: self.scissor.offset,
                extent: self.scissor.extent,
            };
            let scissors = [scissor];
            let viewports = [viewport];
            self.device.cmd_set_viewport(cmd, 0, &viewports);
            self.device.cmd_set_scissor(cmd, 0, &scissors);
            let buffers = [descriptor_set];
            let dynamic_offsets = uniform_offset.map_or(Vec::new(), |offset| vec![offset as u32]);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                shader_module.pipeline_layout,
                0,
                &buffers,
                &dynamic_offsets,
            );

            let index_size = index_type.size() as i32;
            self.device.cmd_bind_index_buffer(
                cmd,
                index_module.current_buffer,
                index_module.current_offset,
                vk_index_type(index_type),
            );
            let mut vertex_buffers = Vec::with_capacity(self.cur_vertex_buffers.len());
            let mut offsets = Vec::with_capacity(self.cur_vertex_buffers.len());
            for slot in &self.cur_vertex_buffers {
                let module = &self.buffers[slot.expect("Unbound vertex buffer")];
                vertex_buffers.push(module.current_buffer);
                offsets.push(module.current_offset);
            }
            self.device
                .cmd_bind_vertex_buffers(cmd, 0, &vertex_buffers, &offsets);

            self.device.cmd_draw_indexed(
                cmd,
                num_elements as u32,
                num_instances as u32,
                (index_size * base_element) as u32,
                0,
                0,
            );
            self.cmd_dirty = true;

            self.pending.uniform_offset = None;
        }
    }

    pub fn commit_frame(&mut self) {}

    pub fn present(&mut self) {
        if !self.frame_active {
            return;
        }
        self.close_pass();

        let image_index = self.acquired_image.take().unwrap();
        let command_buffer = self.command_buffers[image_index];
        let wait = self.frame_wait;
        let signal = self.render_finished[image_index];
        let fence = self.in_flight[self.current_frame];
        let wait_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let swapchain_image = self.swapchain_images[image_index];
        let swapchain_layout = self.swapchain_image_layouts[image_index];
        let mut recreate = false;

        unsafe {
            // Hand the rendered image over to the presentation engine.
            let present_barrier = vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .old_layout(swapchain_layout)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(swapchain_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[present_barrier],
            );

            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end the command buffer");

            let wait_semaphores = [wait];
            let command_buffers = [command_buffer];
            let signal_semaphores = [signal];
            let submit_info = if self.frame_wait_pending {
                vk::SubmitInfo::builder()
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&wait_mask)
                    .command_buffers(&command_buffers)
                    .signal_semaphores(&signal_semaphores)
                    .build()
            } else {
                vk::SubmitInfo::builder()
                    .command_buffers(&command_buffers)
                    .signal_semaphores(&signal_semaphores)
                    .build()
            };
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], fence)
                .expect("Failed to submit the frame");
            self.frame_wait_pending = false;

            let image_indices = [image_index as u32];
            let swapchains = [self.swapchain];
            let wait_for_present = [signal];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&wait_for_present)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            match self
                .swapchain_loader
                .queue_present(self.present_queue, &present_info.build())
            {
                Ok(_) => {
                    self.swapchain_image_layouts[image_index] =
                        vk::ImageLayout::PRESENT_SRC_KHR;
                    if self.swapchain_suboptimal {
                        self.swapchain_suboptimal = false;
                        recreate = true;
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    recreate = true;
                }
                Err(error) => panic!("Failed to present the frame: {:?}", error),
            }

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        self.recording = false;
        self.frame_active = false;
        // Rebuilding the swap chain frees the command buffers and image views this
        // frame used, so it waits until the frame is closed. The textures it creates
        // transition their layout on their own, which would otherwise flush a frame
        // `present` has already taken the image from.
        if recreate {
            self.recreate_swapchain();
        }
    }
}