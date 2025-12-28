//! Vulkan example demonstrating GPU compute and MSAA support
//! 
//! This example showcases the Vulkan backend capabilities including:
//! - GPU compute shaders
//! - Multisample anti-aliasing (MSAA)
//! - Modern Vulkan rendering pipeline

use miniquad::*;

#[cfg(feature = "vulkan")]
use miniquad::graphics::vulkan::*;

struct VulkanExample {
    pipeline: Pipeline,
    compute_pipeline: Pipeline,
    bindings: Bindings,
    compute_bindings: Bindings,
    time: f32,
}

#[cfg(feature = "vulkan")]
impl VulkanExample {
    fn new(ctx: &mut Context) -> Self {
        // Vertex shader for simple quad rendering
        let vertex_shader = "
            #version 450
            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec3 color;
            
            layout(location = 0) out vec3 v_color;
            
            void main() {
                gl_Position = vec4(pos, 0.0, 1.0);
                v_color = color;
            }
        ";
        
        // Fragment shader with MSAA support
        let fragment_shader = "
            #version 450
            layout(location = 0) in vec3 v_color;
            layout(location = 0) out vec4 frag_color;
            
            void main() {
                frag_color = vec4(v_color, 1.0);
            }
        ";
        
        // Compute shader for GPU computation
        let compute_shader = "
            #version 450
            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
            
            layout(set = 0, binding = 0) buffer DataBuffer {
                float data[];
            };
            
            layout(set = 0, binding = 1) uniform TimeUniform {
                float time;
            };
            
            void main() {
                uint index = gl_GlobalInvocationID.x;
                data[index] = sin(time + float(index) * 0.1) * 0.5 + 0.5;
            }
        ";
        
        // Create graphics pipeline with MSAA
        let shader = Shader::new(ctx, vertex_shader, fragment_shader, ShaderMeta::default()).unwrap();
        let pipeline = Pipeline::new(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("pos", VertexFormat::Float2),
                VertexAttribute::new("color", VertexFormat::Float3),
            ],
            shader,
        ).unwrap();
        
        // Create compute pipeline
        let compute_shader = Shader::new_compute(ctx, compute_shader, ShaderMeta::default()).unwrap();
        let compute_pipeline = Pipeline::new_compute(ctx, compute_shader).unwrap();
        
        // Create vertex data for a colored quad
        let vertices: &[f32] = &[
            // Position     Color
            -0.5, -0.5,    1.0, 0.0, 0.0,  // Bottom left - Red
             0.5, -0.5,    0.0, 1.0, 0.0,  // Bottom right - Green
             0.5,  0.5,    0.0, 0.0, 1.0,  // Top right - Blue
            -0.5,  0.5,    1.0, 1.0, 0.0,  // Top left - Yellow
        ];
        
        let indices: &[u16] = &[0, 1, 2, 0, 2, 3];
        
        let vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, vertices);
        let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, indices);
        
        // Create compute buffer (1024 floats)
        let compute_data = vec![0.0f32; 1024];
        let compute_buffer = Buffer::stream(ctx, BufferType::StorageBuffer, 1024 * 4);
        
        // Create uniform buffer for time
        let time_uniform = Buffer::stream(ctx, BufferType::UniformBuffer, 4);
        
        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer],
            index_buffer,
            images: vec![],
        };
        
        let compute_bindings = Bindings {
            vertex_buffers: vec![compute_buffer, time_uniform],
            index_buffer: Buffer::default(),
            images: vec![],
        };
        
        Self {
            pipeline,
            compute_pipeline,
            bindings,
            compute_bindings,
            time: 0.0,
        }
    }
    
    fn update_compute(&mut self, ctx: &mut Context) {
        // Update time uniform
        self.time += 0.016; // ~60 FPS
        
        // Update uniform buffer
        let time_data = [self.time];
        ctx.buffer_update(&self.compute_bindings.vertex_buffers[1], &time_data);
        
        // Dispatch compute shader
        ctx.compute_pass(&self.compute_pipeline, &self.compute_bindings, 16, 1, 1);
    }
}

impl EventHandler for VulkanExample {
    fn update(&mut self, ctx: &mut Context) {
        #[cfg(feature = "vulkan")]
        self.update_compute(ctx);
    }
    
    fn draw(&mut self, ctx: &mut Context) {
        ctx.begin_default_pass(Default::default());
        
        // Draw the quad with MSAA
        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&self.bindings);
        ctx.draw(0, 6, 1);
        
        ctx.end_render_pass();
        
        // Display performance stats
        #[cfg(feature = "vulkan")]
        {
            if let Ok(stats) = ctx.get_vulkan_stats() {
                egui::Window::new("Vulkan Performance").show(ctx, |ui| {
                    ui.label(format!("MSAA Samples: {:?}", stats.msaa_samples));
                    ui.label(format!("MSAA Enabled: {}", stats.msaa_enabled));
                    ui.label(format!("Buffer Count: {}", stats.buffer_count));
                    ui.label(format!("Texture Count: {}", stats.texture_count));
                    ui.label(format!("Shader Count: {}", stats.shader_count));
                    ui.label(format!("Pipeline Count: {}", stats.pipeline_count));
                    ui.label(format!("Frame Time: {:.3} ms", stats.frame_time * 1000.0));
                });
            }
        }
    }
}

#[cfg(not(feature = "vulkan"))]
impl EventHandler for VulkanExample {
    fn update(&mut self, _ctx: &mut Context) {
        // No-op for non-Vulkan builds
    }
    
    fn draw(&mut self, ctx: &mut Context) {
        ctx.begin_default_pass(Default::default());
        ctx.clear(Some((1.0, 0.0, 0.0, 1.0)), None, None);
        ctx.end_render_pass();
        
        egui::Window::new("Vulkan Not Available").show(ctx, |ui| {
            ui.label("This example requires the 'vulkan' feature to be enabled.");
            ui.label("Run with: cargo run --features vulkan --example vulkan_example");
        });
    }
}

fn main() {
    let conf = Conf {
        window_title: "Vulkan Example - GPU Compute & MSAA".to_string(),
        window_width: 800,
        window_height: 600,
        sample_count: 4, // Enable 4x MSAA
        high_dpi: true,
        ..Default::default()
    };
    
    #[cfg(feature = "vulkan")]
    {
        miniquad::start(conf, |ctx| {
            let example = VulkanExample::new(ctx);
            UserData::owning(example, EventHandler::on_update, EventHandler::on_draw)
        });
    }
    
    #[cfg(not(feature = "vulkan"))]
    {
        miniquad::start(conf, |ctx| {
            let example = VulkanExample { };
            UserData::owning(example, EventHandler::on_update, EventHandler::on_draw)
        });
    }
}