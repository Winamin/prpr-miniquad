use ash::{vk, Device};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{IMAGE_BINDING_BASE, MAX_FRAMES_IN_FLIGHT, UNIFORM_BINDING};
use crate::graphics::MAX_SHADERSTAGE_IMAGES;

pub(crate) struct RecordedDraw {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_layout: vk::DescriptorSetLayout,
    pub uniform: Option<vk::DescriptorBufferInfo>,
    pub uniform_offset: Option<u32>,
    pub images: Vec<vk::DescriptorImageInfo>,
    pub viewport: vk::Viewport,
    pub scissor: vk::Rect2D,
    pub index_buffer: vk::Buffer,
    pub index_offset: vk::DeviceSize,
    pub index_type: vk::IndexType,
    pub vertex_buffers: Vec<vk::Buffer>,
    pub vertex_offsets: Vec<vk::DeviceSize>,
    pub first_index: u32,
    pub index_count: u32,
    pub instance_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderingContents {
    Inline,
    Secondary,
}

impl RenderingContents {
    fn flags(self) -> vk::RenderingFlags {
        match self {
            Self::Inline => vk::RenderingFlags::empty(),
            Self::Secondary => vk::RenderingFlags::CONTENTS_SECONDARY_COMMAND_BUFFERS,
        }
    }
}

enum PrimaryCommand {
    Begin {
        area: vk::Rect2D,
        color: vk::RenderingAttachmentInfo,
        depth: Option<vk::RenderingAttachmentInfo>,
        contents: RenderingContents,
    },
    End,
    Barrier {
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        image: vk::ImageMemoryBarrier,
    },
    Draw(usize),
}

struct RecordedBatch {
    draws: Vec<RecordedDraw>,
    commands: Vec<vk::CommandBuffer>,
    contents: RenderingContents,
    color: vk::Format,
    depth: Option<vk::Format>,
    samples: u32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct CommandRecordingStats {
    pub draw_calls: u64,
    pub batches: u64,
    pub command_buffers: u64,
    pub thread_mask: u64,
    pub threads: usize,
    pub worker_draw_calls: u64,
    pub inline_draw_calls: u64,
    pub descriptor_updates: u64,
    pub descriptor_cache_hits: u64,
}

// A descriptor set is immutable after its first update. Uniform contents may
// vary through dynamic offsets without allocating or updating the set again.
#[derive(Default)]
struct DescriptorCache {
    set: vk::DescriptorSet,
    layout: vk::DescriptorSetLayout,
    uniform: Option<(vk::Buffer, vk::DeviceSize, vk::DeviceSize)>,
    images: Vec<(vk::Sampler, vk::ImageView, vk::ImageLayout)>,
}

impl DescriptorCache {
    fn invalidate(&mut self) {
        self.set = vk::DescriptorSet::null();
    }

    fn matches(&self, draw: &RecordedDraw) -> bool {
        self.set != vk::DescriptorSet::null()
            && self.layout == draw.descriptor_layout
            && self.uniform
                == draw
                    .uniform
                    .map(|info| (info.buffer, info.offset, info.range))
            && self.images.iter().copied().eq(draw
                .images
                .iter()
                .map(|info| (info.sampler, info.image_view, info.image_layout)))
    }

    fn remember(&mut self, draw: &RecordedDraw, set: vk::DescriptorSet) {
        self.set = set;
        self.layout = draw.descriptor_layout;
        self.uniform = draw
            .uniform
            .map(|info| (info.buffer, info.offset, info.range));
        self.images.clear();
        self.images.extend(
            draw.images
                .iter()
                .map(|info| (info.sampler, info.image_view, info.image_layout)),
        );
    }
}

fn descriptor_writes<'a>(
    draw: &'a RecordedDraw,
    set: vk::DescriptorSet,
    inline: &'a mut [vk::WriteDescriptorSet],
    overflow: &'a mut Vec<vk::WriteDescriptorSet>,
) -> &'a [vk::WriteDescriptorSet] {
    let count = draw.images.len() + usize::from(draw.uniform.is_some());
    // The inline capacity covers common shaders, not a Vulkan sampler limit.
    let writes = if count <= inline.len() {
        &mut inline[..count]
    } else {
        overflow.resize(count, vk::WriteDescriptorSet::default());
        overflow.as_mut_slice()
    };
    let mut index = 0;
    if let Some(info) = &draw.uniform {
        writes[index] = vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(UNIFORM_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .buffer_info(std::slice::from_ref(info))
            .build();
        index += 1;
    }
    for (slot, info) in draw.images.iter().enumerate() {
        writes[index] = vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(IMAGE_BINDING_BASE + slot as u32)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(info))
            .build();
        index += 1;
    }
    writes
}

struct RecordingFrame {
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    used: usize,
    descriptor_pools: Vec<vk::DescriptorPool>,
    descriptor_pool: usize,
    descriptors_used: bool,
    cache_descriptors: bool,
    descriptor_cache: DescriptorCache,
    descriptor_updates: u64,
    descriptor_cache_hits: u64,
}

impl RecordingFrame {
    fn new(device: &Device, family: u32, cache_descriptors: bool) -> Self {
        let command_pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::builder()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )
                .expect("Failed to create a recording command pool")
        };
        Self {
            command_pool,
            command_buffers: Vec::new(),
            used: 0,
            descriptor_pools: Vec::new(),
            descriptor_pool: 0,
            descriptors_used: false,
            cache_descriptors,
            descriptor_cache: DescriptorCache::default(),
            descriptor_updates: 0,
            descriptor_cache_hits: 0,
        }
    }

    fn reset(&mut self, device: &Device) {
        if self.used == 0 && !self.descriptors_used {
            return;
        }
        unsafe {
            if self.used != 0 {
                device
                    .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                    .expect("Failed to reset a recording command pool");
            }
            if self.descriptors_used {
                for &pool in &self.descriptor_pools {
                    device
                        .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
                        .expect("Failed to reset a recording descriptor pool");
                }
            }
        }
        self.used = 0;
        self.descriptor_pool = 0;
        self.descriptors_used = false;
        self.descriptor_cache.invalidate();
    }

    fn command_buffer(&mut self, device: &Device) -> vk::CommandBuffer {
        if self.used == self.command_buffers.len() {
            let info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);
            let command_buffer = unsafe {
                device
                    .allocate_command_buffers(&info)
                    .expect("Failed to allocate a secondary command buffer")[0]
            };
            self.command_buffers.push(command_buffer);
        }
        let command_buffer = self.command_buffers[self.used];
        self.used += 1;
        command_buffer
    }

    fn descriptor_set(
        &mut self,
        device: &Device,
        layout: vk::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        loop {
            if self.descriptor_pool == self.descriptor_pools.len() {
                const SETS: u32 = 512; // The pool notes(max)
                let sizes = [
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                        descriptor_count: SETS,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: SETS * MAX_SHADERSTAGE_IMAGES as u32,
                    },
                ];
                let info = vk::DescriptorPoolCreateInfo::builder()
                    .max_sets(SETS)
                    .pool_sizes(&sizes);
                self.descriptor_pools.push(unsafe {
                    device
                        .create_descriptor_pool(&info, None)
                        .expect("Failed to create a recording descriptor pool")
                });
            }
            let info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(self.descriptor_pools[self.descriptor_pool])
                .set_layouts(std::slice::from_ref(&layout));
            match unsafe { device.allocate_descriptor_sets(&info) } {
                Ok(sets) => {
                    self.descriptors_used = true;
                    return sets[0];
                }
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)
                | Err(vk::Result::ERROR_FRAGMENTED_POOL) => self.descriptor_pool += 1,
                Err(error) => panic!("Failed to allocate a recording descriptor set: {:?}", error),
            }
        }
    }

    fn record(
        &mut self,
        device: &Device,
        draws: &[RecordedDraw],
        color: vk::Format,
        depth: Option<vk::Format>,
        samples: u32,
    ) -> vk::CommandBuffer {
        #[cfg(feature = "tracy")]
        let _span = crate::tracy::span(crate::tracy::span_location!("Vulkan::record_secondary"));
        let command_buffer = self.command_buffer(device);
        let formats = [color];
        let mut rendering = vk::CommandBufferInheritanceRenderingInfo::builder()
            .color_attachment_formats(&formats)
            .depth_attachment_format(depth.unwrap_or(vk::Format::UNDEFINED))
            .rasterization_samples(vk::SampleCountFlags::from_raw(samples));
        let inheritance = vk::CommandBufferInheritanceInfo::builder().push_next(&mut rendering);
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                    | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
            )
            .inheritance_info(&inheritance);
        unsafe {
            device
                .begin_command_buffer(command_buffer, &info)
                .expect("Failed to begin a secondary command buffer");
        }
        self.record_draws(device, command_buffer, draws);
        unsafe {
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to end a secondary command buffer");
        }
        command_buffer
    }

    fn record_draws(
        &mut self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        draws: &[RecordedDraw],
    ) {
        for draw in draws {
            let set = self.descriptors(device, draw);
            unsafe {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    draw.pipeline,
                );
                device.cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&draw.viewport));
                device.cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&draw.scissor));
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    draw.pipeline_layout,
                    0,
                    &[set],
                    draw.uniform_offset.as_slice(),
                );
                device.cmd_bind_index_buffer(
                    command_buffer,
                    draw.index_buffer,
                    draw.index_offset,
                    draw.index_type,
                );
                device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    &draw.vertex_buffers,
                    &draw.vertex_offsets,
                );
                device.cmd_draw_indexed(
                    command_buffer,
                    draw.index_count,
                    draw.instance_count,
                    draw.first_index,
                    0,
                    0,
                );
            }
        }
    }

    fn descriptors(&mut self, device: &Device, draw: &RecordedDraw) -> vk::DescriptorSet {
        if self.cache_descriptors && self.descriptor_cache.matches(draw) {
            self.descriptor_cache_hits += 1;
            return self.descriptor_cache.set;
        }
        let set = self.descriptor_set(device, draw.descriptor_layout);
        let mut inline = [vk::WriteDescriptorSet::default(); MAX_SHADERSTAGE_IMAGES + 1];
        let mut overflow = Vec::new();
        let writes = descriptor_writes(draw, set, &mut inline, &mut overflow);
        unsafe {
            device.update_descriptor_sets(writes, &[]);
        }
        self.descriptor_updates += 1;
        if self.cache_descriptors {
            self.descriptor_cache.remember(draw, set);
        }
        set
    }

    fn destroy(self, device: &Device) {
        unsafe {
            for pool in self.descriptor_pools {
                device.destroy_descriptor_pool(pool, None);
            }
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

struct RecordingWorker {
    frames: Vec<RecordingFrame>,
    results: Vec<RecordedChunk>,
    thread: Option<usize>,
}

struct RecordingJob {
    batch: usize,
    chunk: usize,
    draws: std::ops::Range<usize>,
}

struct RecordedChunk {
    batch: usize,
    chunk: usize,
    command: vk::CommandBuffer,
    draws: usize,
}

impl RecordingWorker {
    fn record_job(
        &mut self,
        device: &Device,
        slot: usize,
        job: &RecordingJob,
        batches: &[RecordedBatch],
    ) {
        #[cfg(feature = "tracy")]
        let _span = crate::tracy::span_with_callstack(crate::tracy::span_location!("Vulkan::record_job"));
        let batch = &batches[job.batch];
        let draws = &batch.draws[job.draws.clone()];
        let command =
            self.frames[slot].record(device, draws, batch.color, batch.depth, batch.samples);
        self.results.push(RecordedChunk {
            batch: job.batch,
            chunk: job.chunk,
            command,
            draws: draws.len(),
        });
    }
}

pub(crate) struct CommandRecorder {
    pool: Option<rayon::ThreadPool>,
    #[cfg(feature = "tracy")]
    tracy_registered: bool,
    workers: Vec<RecordingWorker>,
    pub draws: Vec<RecordedDraw>,
    batches: Vec<RecordedBatch>,
    jobs: Vec<RecordingJob>,
    batch_count: usize,
    primary: Vec<PrimaryCommand>,
    rendering_contents: Option<RenderingContents>,
    stats: CommandRecordingStats,
}

fn recording_chunk_size(draws: usize, workers: usize) -> usize {
    draws.div_ceil(workers * 4).max(16)
}

// A rendering scope can span several macroquad flushes. Its contents mode is
// fixed by vkCmdBeginRendering, even when a later flush has a different size.
fn plan_rendering_contents(
    commands: &mut [PrimaryCommand],
    batches: &mut [RecordedBatch],
    preferred: RenderingContents,
    mut active: Option<RenderingContents>,
) -> Option<RenderingContents> {
    for command in commands {
        match command {
            PrimaryCommand::Begin { contents, .. } => {
                *contents = preferred;
                active = Some(preferred);
            }
            PrimaryCommand::End => active = None,
            PrimaryCommand::Draw(id) => {
                batches[*id].contents = active.expect("Drawing outside a rendering scope");
            }
            PrimaryCommand::Barrier { .. } => {}
        }
    }
    active
}

impl CommandRecorder {
    pub fn new(device: &Device, family: u32) -> Self {
        let threads = std::env::var("PRPR_VULKAN_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map_or(1, |value| value.get())
                    .min(8)
            });
        let pool = if threads > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|id| format!("vulkan-record-{id}"))
                    .start_handler(|id| {
                        #[cfg(feature = "tracy")]
                        crate::tracy::register_thread(&format!("vulkan-record-{id}"));
                        #[cfg(not(feature = "tracy"))]
                        let _ = id;
                    })
                    .build()
                    .expect("Failed to create the Vulkan recording pool"),
            )
        } else {
            None
        };
        let cache_descriptors = std::env::var("PRPR_VULKAN_DESCRIPTOR_CACHE")
            .map_or(true, |value| {
                !matches!(value.as_str(), "0" | "false" | "off")
            });
        let workers = (0..if pool.is_some() { threads } else { 0 })
            .map(|_| RecordingWorker {
                frames: (0..MAX_FRAMES_IN_FLIGHT)
                    .map(|_| RecordingFrame::new(device, family, cache_descriptors))
                    .collect(),
                results: Vec::new(),
                thread: None,
            })
            .collect();
        Self {
            pool,
            #[cfg(feature = "tracy")]
            tracy_registered: false,
            workers,
            draws: Vec::new(),
            batches: Vec::new(),
            jobs: Vec::new(),
            batch_count: 0,
            primary: Vec::new(),
            rendering_contents: None,
            stats: CommandRecordingStats {
                threads,
                ..Default::default()
            },
        }
    }

    pub fn enabled(&self) -> bool {
        self.pool.is_some()
    }

    pub fn stats(&self) -> CommandRecordingStats {
        let mut stats = self.stats;
        for frame in self.workers.iter().flat_map(|worker| &worker.frames) {
            stats.descriptor_updates += frame.descriptor_updates;
            stats.descriptor_cache_hits += frame.descriptor_cache_hits;
        }
        stats
    }

    pub fn reset(&mut self, device: &Device, slot: usize) {
        for worker in &mut self.workers {
            worker.frames[slot].reset(device);
        }
    }

    pub fn begin(
        &mut self,
        area: vk::Rect2D,
        color: vk::RenderingAttachmentInfo,
        depth: Option<vk::RenderingAttachmentInfo>,
    ) {
        self.primary.push(PrimaryCommand::Begin {
            area,
            color,
            depth,
            contents: RenderingContents::Inline,
        });
    }

    pub fn end(&mut self) {
        self.primary.push(PrimaryCommand::End);
    }

    pub fn barrier(
        &mut self,
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        image: vk::ImageMemoryBarrier,
    ) {
        self.primary
            .push(PrimaryCommand::Barrier { src, dst, image });
    }

    pub fn batch(&mut self, color: vk::Format, depth: Option<vk::Format>, samples: u32) {
        if self.draws.is_empty() {
            return;
        }
        if self.batch_count == self.batches.len() {
            self.batches.push(RecordedBatch {
                draws: Vec::new(),
                commands: Vec::new(),
                contents: RenderingContents::Inline,
                color,
                depth,
                samples,
            });
        }
        let batch = &mut self.batches[self.batch_count];
        std::mem::swap(&mut batch.draws, &mut self.draws);
        batch.commands.clear();
        batch.color = color;
        batch.depth = depth;
        batch.samples = samples;
        self.primary.push(PrimaryCommand::Draw(self.batch_count));
        self.batch_count += 1;
    }

    pub fn execute(&mut self, device: &Device, primary: vk::CommandBuffer, slot: usize) {
        #[cfg(feature = "tracy")]
        let _span = crate::tracy::span_with_callstack(crate::tracy::span_location!("Vulkan::record_execute"));
        #[cfg(feature = "tracy")]
        {
            let connected = tracy_client::Client::is_connected();
            if connected && !self.tracy_registered {
                // On-demand capture may connect after worker startup. Register
                // every worker once, including workers with no recording jobs.
                if let Some(pool) = &self.pool {
                    pool.broadcast(|context| crate::tracy::register_thread(&format!("vulkan-record-{}", context.index())));
                }
                self.tracy_registered = true;
            } else if !connected {
                self.tracy_registered = false;
            }
        }
        if self.primary.is_empty() {
            return;
        }
        let job_count: usize = self.batches[..self.batch_count]
            .iter()
            .map(|batch| {
                batch
                    .draws
                    .len()
                    .div_ceil(recording_chunk_size(batch.draws.len(), self.workers.len()))
            })
            .sum();
        let preferred = if job_count <= 1 {
            RenderingContents::Inline
        } else {
            RenderingContents::Secondary
        };
        self.rendering_contents = plan_rendering_contents(
            &mut self.primary,
            &mut self.batches[..self.batch_count],
            preferred,
            self.rendering_contents,
        );
        self.jobs.clear();
        for (id, batch) in self.batches[..self.batch_count].iter_mut().enumerate() {
            if batch.contents == RenderingContents::Inline {
                continue;
            }
            let size = recording_chunk_size(batch.draws.len(), self.workers.len());
            batch
                .commands
                .resize(batch.draws.len().div_ceil(size), vk::CommandBuffer::null());
            for (chunk, start) in (0..batch.draws.len()).step_by(size).enumerate() {
                self.jobs.push(RecordingJob {
                    batch: id,
                    chunk,
                    draws: start..(start + size).min(batch.draws.len()),
                });
            }
        }
        if !self.jobs.is_empty() {
            let jobs = &self.jobs;
            let batches = &self.batches[..self.batch_count];
            let count = self.workers.len().min(jobs.len());
            let workers = &mut self.workers[..count];
            for worker in &mut *workers {
                worker.results.clear();
                worker.thread = None;
            }
            if count == 1 {
                for job in jobs {
                    workers[0].record_job(device, slot, job, batches);
                }
            } else {
                let next = AtomicUsize::new(0);
                self.pool.as_ref().unwrap().install(|| {
                    workers.par_iter_mut().with_max_len(1).for_each(|worker| {
                        worker.thread = rayon::current_thread_index();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some(job) = jobs.get(index) else {
                                break;
                            };
                            worker.record_job(device, slot, job, batches);
                        }
                    })
                });
            }
            // Jobs finish in any order. Each result owns one preassigned slot,
            // so submitting a batch in draw order needs no sorting or locking.
            for worker in workers {
                for result in &worker.results {
                    self.batches[result.batch].commands[result.chunk] = result.command;
                    self.stats.command_buffers += 1;
                    if let Some(thread) = worker.thread {
                        self.stats.thread_mask |= 1u64 << thread.min(63);
                        self.stats.worker_draw_calls += result.draws as u64;
                    }
                }
            }
        }
        for command in self.primary.drain(..) {
            unsafe {
                match command {
                    PrimaryCommand::Begin {
                        area,
                        color,
                        depth,
                        contents,
                    } => {
                        let info = vk::RenderingInfo::builder()
                            .flags(contents.flags())
                            .render_area(area)
                            .layer_count(1)
                            .color_attachments(std::slice::from_ref(&color));
                        let info = if let Some(depth) = &depth {
                            info.depth_attachment(depth)
                        } else {
                            info
                        };
                        device.cmd_begin_rendering(primary, &info);
                    }
                    PrimaryCommand::End => device.cmd_end_rendering(primary),
                    PrimaryCommand::Barrier { src, dst, image } => device.cmd_pipeline_barrier(
                        primary,
                        src,
                        dst,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[image],
                    ),
                    PrimaryCommand::Draw(id) => {
                        let batch = &self.batches[id];
                        match batch.contents {
                            RenderingContents::Inline => {
                                self.workers[0].frames[slot].record_draws(
                                    device,
                                    primary,
                                    &batch.draws,
                                );
                                self.stats.inline_draw_calls += batch.draws.len() as u64;
                            }
                            RenderingContents::Secondary => {
                                device.cmd_execute_commands(primary, &batch.commands)
                            }
                        }
                    }
                }
            }
        }
        for batch in &mut self.batches[..self.batch_count] {
            self.stats.draw_calls += batch.draws.len() as u64;
            self.stats.batches += 1;
            batch.draws.clear();
        }
        self.batch_count = 0;
    }

    pub fn destroy(&mut self, device: &Device) {
        for worker in self.workers.drain(..) {
            for frame in worker.frames {
                frame.destroy(device);
            }
        }
        self.pool.take();
        self.draws.clear();
        self.batches.clear();
        self.jobs.clear();
        self.batch_count = 0;
        self.primary.clear();
        self.rendering_contents = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn draw() -> RecordedDraw {
        RecordedDraw {
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_layout: vk::DescriptorSetLayout::from_raw(1),
            uniform: Some(vk::DescriptorBufferInfo {
                buffer: vk::Buffer::from_raw(2),
                offset: 0,
                range: 32,
            }),
            uniform_offset: Some(64),
            images: vec![vk::DescriptorImageInfo {
                sampler: vk::Sampler::from_raw(3),
                image_view: vk::ImageView::from_raw(4),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }],
            viewport: vk::Viewport::default(),
            scissor: vk::Rect2D::default(),
            index_buffer: vk::Buffer::null(),
            index_offset: 0,
            index_type: vk::IndexType::UINT16,
            vertex_buffers: Vec::new(),
            vertex_offsets: Vec::new(),
            first_index: 0,
            index_count: 6,
            instance_count: 1,
        }
    }

    #[test]
    fn dynamic_offsets_reuse_immutable_descriptors() {
        let mut draw = draw();
        let mut cache = DescriptorCache::default();
        cache.remember(&draw, vk::DescriptorSet::from_raw(5));
        draw.uniform_offset = Some(256);
        draw.pipeline = vk::Pipeline::from_raw(6);
        assert!(cache.matches(&draw));

        cache.invalidate();
        assert!(
            !cache.matches(&draw),
            "A reset descriptor pool invalidates every cached set"
        );
    }

    #[test]
    fn descriptor_binding_changes_cannot_reuse_a_set() {
        let original = draw();
        let mut cache = DescriptorCache::default();
        cache.remember(&original, vk::DescriptorSet::from_raw(5));
        let mut changed = draw();
        changed.images[0].sampler = vk::Sampler::from_raw(7);
        assert!(!cache.matches(&changed));
        changed = draw();
        changed.images[0].image_view = vk::ImageView::from_raw(8);
        assert!(!cache.matches(&changed));
        changed = draw();
        changed.images[0].image_layout = vk::ImageLayout::GENERAL;
        assert!(!cache.matches(&changed));
        changed = draw();
        changed.uniform.as_mut().unwrap().range = 64;
        assert!(!cache.matches(&changed));
        changed = draw();
        changed.uniform.as_mut().unwrap().buffer = vk::Buffer::from_raw(9);
        assert!(!cache.matches(&changed));
        changed = draw();
        changed.descriptor_layout = vk::DescriptorSetLayout::from_raw(10);
        assert!(!cache.matches(&changed));
    }

    #[test]
    fn descriptor_writes_support_shaders_beyond_the_inline_capacity() {
        for image_count in [MAX_SHADERSTAGE_IMAGES, MAX_SHADERSTAGE_IMAGES + 1, 24] {
            let mut draw = draw();
            draw.images.resize(image_count, draw.images[0]);
            let set = vk::DescriptorSet::from_raw(5);
            let mut inline = [vk::WriteDescriptorSet::default(); MAX_SHADERSTAGE_IMAGES + 1];
            let mut overflow = Vec::new();
            let writes = descriptor_writes(&draw, set, &mut inline, &mut overflow);
            assert_eq!(writes.len(), image_count + 1);
            assert_eq!(writes[0].dst_binding, UNIFORM_BINDING);
            for (slot, write) in writes[1..].iter().enumerate() {
                assert_eq!(write.dst_set, set);
                assert_eq!(write.dst_binding, IMAGE_BINDING_BASE + slot as u32);
                assert_eq!(write.descriptor_count, 1);
                assert_eq!(write.p_image_info, &draw.images[slot] as *const _);
            }
            assert_eq!(overflow.is_empty(), image_count <= MAX_SHADERSTAGE_IMAGES);
        }
    }

    fn begin() -> PrimaryCommand {
        PrimaryCommand::Begin {
            area: vk::Rect2D::default(),
            color: vk::RenderingAttachmentInfo::default(),
            depth: None,
            contents: RenderingContents::Inline,
        }
    }

    fn batch() -> RecordedBatch {
        RecordedBatch {
            draws: Vec::new(),
            commands: Vec::new(),
            contents: RenderingContents::Inline,
            color: vk::Format::R8G8B8A8_UNORM,
            depth: None,
            samples: 1,
        }
    }

    #[test]
    fn inline_scope_stays_inline_when_a_later_flush_grows() {
        let mut batches = [batch()];
        let active = plan_rendering_contents(
            &mut [begin(), PrimaryCommand::Draw(0)],
            &mut batches,
            RenderingContents::Inline,
            None,
        );
        assert_eq!(active, Some(RenderingContents::Inline));

        let active = plan_rendering_contents(
            &mut [PrimaryCommand::Draw(0)],
            &mut batches,
            RenderingContents::Secondary,
            active,
        );
        assert_eq!(active, Some(RenderingContents::Inline));
        assert_eq!(batches[0].contents, RenderingContents::Inline);
    }

    #[test]
    fn secondary_scope_stays_secondary_when_a_later_flush_shrinks() {
        let mut batches = [batch()];
        let mut commands = [begin(), PrimaryCommand::Draw(0)];
        let active = plan_rendering_contents(
            &mut commands,
            &mut batches,
            RenderingContents::Secondary,
            None,
        );
        let PrimaryCommand::Begin { contents, .. } = &commands[0] else {
            unreachable!();
        };
        assert_eq!(
            contents.flags(),
            vk::RenderingFlags::CONTENTS_SECONDARY_COMMAND_BUFFERS
        );

        let active = plan_rendering_contents(
            &mut [PrimaryCommand::Draw(0), PrimaryCommand::End],
            &mut batches,
            RenderingContents::Inline,
            active,
        );
        assert_eq!(batches[0].contents, RenderingContents::Secondary);
        assert_eq!(active, None);
    }

    #[test]
    fn a_new_scope_can_change_mode_after_end() {
        let mut batches = [batch(), batch()];
        let mut commands = [
            PrimaryCommand::Draw(0),
            PrimaryCommand::End,
            PrimaryCommand::Barrier {
                src: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst: vk::PipelineStageFlags::FRAGMENT_SHADER,
                image: vk::ImageMemoryBarrier::default(),
            },
            begin(),
            PrimaryCommand::Draw(1),
        ];
        let active = plan_rendering_contents(
            &mut commands,
            &mut batches,
            RenderingContents::Secondary,
            Some(RenderingContents::Inline),
        );
        assert_eq!(batches[0].contents, RenderingContents::Inline);
        assert_eq!(batches[1].contents, RenderingContents::Secondary);
        assert_eq!(active, Some(RenderingContents::Secondary));
    }

    #[test]
    fn a_clear_only_flush_still_fixes_the_scope_mode() {
        let active =
            plan_rendering_contents(&mut [begin()], &mut [], RenderingContents::Inline, None);
        let mut batches = [batch()];
        let active = plan_rendering_contents(
            &mut [PrimaryCommand::Draw(0), PrimaryCommand::End],
            &mut batches,
            RenderingContents::Secondary,
            active,
        );
        assert_eq!(batches[0].contents, RenderingContents::Inline);
        assert_eq!(active, None);
        assert!(RenderingContents::Inline.flags().is_empty());
    }
}
