//! Rendering backend abstraction layer
//! 
//! This module provides a common abstraction for different rendering backends
//! (OpenGL and Vulkan), allowing the same high-level API to work with either backend.

use crate::conf::RenderingBackend;
use crate::graphics::*;
use crate::native::NativeDisplay;

#[cfg(feature = "vulkan")]
use crate::graphics::vulkan::vk::VulkanContext;

/// Rendering backend abstraction
pub enum RenderingBackendContext {
    OpenGL(GraphicsContext),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanContext),
}

impl RenderingBackendContext {
    /// Create a new rendering backend context
    pub fn new(backend: RenderingBackend) -> Self {
        match backend {
            RenderingBackend::OpenGL => RenderingBackendContext::OpenGL(GraphicsContext::new(false)),
            #[cfg(feature = "vulkan")]
            RenderingBackend::Vulkan => RenderingBackendContext::Vulkan(VulkanContext::new()),
            #[cfg(not(feature = "vulkan"))]
            RenderingBackend::Vulkan => panic!("Vulkan backend is not available. Enable the 'vulkan' feature to use Vulkan."),
        }
    }

    /// Initialize the backend
    pub fn initialize(&mut self, display: &mut dyn NativeDisplay) -> Result<(), String> {
        match self {
            RenderingBackendContext::OpenGL(gl_ctx) => {
                // Update GL context after OpenGL functions are loaded
                let is_gles2 = unsafe { crate::native::gl::is_gl2() };
                *gl_ctx = GraphicsContext::new(is_gles2);
                Ok(())
            }
            #[cfg(feature = "vulkan")]
            RenderingBackendContext::Vulkan(vk_ctx) => {
                vk_ctx.initialize(display).map_err(|e| e.to_string())
            }
        }
    }

    /// Begin a render pass
    pub fn begin_render_pass(&mut self, clear_color: Option<(f32, f32, f32, f32)>) -> Result<(), String> {
        match self {
            RenderingBackendContext::OpenGL(gl_ctx) => {
                // OpenGL render pass handling
                Ok(())
            }
            #[cfg(feature = "vulkan")]
            RenderingBackendContext::Vulkan(vk_ctx) => {
                let color = clear_color.unwrap_or((0.0, 0.0, 0.0, 1.0));
                vk_ctx.begin_render_pass(color).map_err(|e| e.to_string())
            }
        }
    }

    /// End a render pass
    pub fn end_render_pass(&mut self) -> Result<(), String> {
        match self {
            RenderingBackendContext::OpenGL(gl_ctx) => {
                // OpenGL render pass handling
                Ok(())
            }
            #[cfg(feature = "vulkan")]
            RenderingBackendContext::Vulkan(vk_ctx) => {
                vk_ctx.end_render_pass().map_err(|e| e.to_string())
            }
        }
    }

    /// Present the current frame
    pub fn present(&mut self) -> Result<(), String> {
        match self {
            RenderingBackendContext::OpenGL(gl_ctx) => {
                // OpenGL present handling (usually done by swap buffers)
                Ok(())
            }
            #[cfg(feature = "vulkan")]
            RenderingBackendContext::Vulkan(vk_ctx) => {
                vk_ctx.present().map_err(|e| e.to_string())
            }
        }
    }

    /// Create a buffer
    pub fn create_buffer(&mut self, size: usize, usage: BufferType) -> Result<usize, String> {
        match self {
            RenderingBackendContext::OpenGL(gl_ctx) => {
                // OpenGL buffer creation
                Ok(0) // Placeholder
            }
            #[cfg(feature = "vulkan")]
            RenderingBackendContext::Vulkan(vk_ctx) => {
                use ash_037::vk;
                use gpu_allocator_022::MemoryLocation;
                let vk_usage = match usage {
                    BufferType::VertexBuffer => vk::BufferUsageFlags::VERTEX_BUFFER,
                    BufferType::IndexBuffer => vk::BufferUsageFlags::INDEX_BUFFER,
                    BufferType::IndexBuffer => vk::BufferUsageFlags::UNIFORM_BUFFER, // Temporary mapping
                };
                vk_ctx.create_buffer(size as vk::DeviceSize, vk_usage, MemoryLocation::CpuToGpu)
                    .map_err(|e| e.to_string())
            }
        }
    }

    /// Update buffer data
    pub fn update_buffer(&mut self, buffer_id: usize, data: &[u8]) -> Result<(), String> {
        match self {
            RenderingBackendContext::OpenGL(gl_ctx) => {
                // OpenGL buffer update
                Ok(())
            }
            #[cfg(feature = "vulkan")]
            RenderingBackendContext::Vulkan(vk_ctx) => {
                vk_ctx.update_buffer(buffer_id, data).map_err(|e| e.to_string())
            }
        }
    }

    /// Create a texture
    pub fn create_texture(&mut self, width: u32, height: u32, data: &[u8]) -> Result<usize, String> {
        match self {
            RenderingBackendContext::OpenGL(gl_ctx) => {
                // OpenGL texture creation
                Ok(0) // Placeholder
            }
            #[cfg(feature = "vulkan")]
            RenderingBackendContext::Vulkan(vk_ctx) => {
                vk_ctx.create_texture(width, height, data).map_err(|e| e.to_string())
            }
        }
    }

    /// Cleanup resources
    pub fn cleanup(&mut self) {
        match self {
            RenderingBackendContext::OpenGL(_) => {
                // OpenGL cleanup
            }
            #[cfg(feature = "vulkan")]
            RenderingBackendContext::Vulkan(vk_ctx) => {
                vk_ctx.cleanup();
            }
        }
    }

    /// Check if the backend is available
    pub fn is_available(backend: RenderingBackend) -> bool {
        match backend {
            RenderingBackend::OpenGL => true, // OpenGL is always available
            #[cfg(feature = "vulkan")]
            RenderingBackend::Vulkan => VulkanContext::is_available(),
            #[cfg(not(feature = "vulkan"))]
            RenderingBackend::Vulkan => false,
        }
    }

    /// Get the backend type
    pub fn backend_type(&self) -> RenderingBackend {
        match self {
            RenderingBackendContext::OpenGL(_) => RenderingBackend::OpenGL,
            #[cfg(feature = "vulkan")]
            RenderingBackendContext::Vulkan(_) => RenderingBackend::Vulkan,
        }
    }
}

/// Common graphics context that works with both OpenGL and Vulkan
pub struct GraphicsContextWrapper {
    backend: RenderingBackendContext,
}

impl GraphicsContextWrapper {
    /// Create a new graphics context with the specified backend
    pub fn new(backend: RenderingBackend) -> Result<Self, String> {
        if !RenderingBackendContext::is_available(backend) {
            return Err(format!("Rendering backend {:?} is not available", backend));
        }

        Ok(Self {
            backend: RenderingBackendContext::new(backend),
        })
    }

    /// Initialize the graphics context
    pub fn initialize(&mut self, display: &mut dyn NativeDisplay) -> Result<(), String> {
        self.backend.initialize(display)
    }

    /// Get the backend type
    pub fn backend_type(&self) -> RenderingBackend {
        self.backend.backend_type()
    }

    /// Get the underlying OpenGL context (if available)
    pub fn as_opengl(&mut self) -> Option<&mut GraphicsContext> {
        match &mut self.backend {
            RenderingBackendContext::OpenGL(gl_ctx) => Some(gl_ctx),
            #[cfg(feature = "vulkan")]
            _ => None,
        }
    }

    /// Get the underlying Vulkan context (if available)
    #[cfg(feature = "vulkan")]
    pub fn as_vulkan(&mut self) -> Option<&mut VulkanContext> {
        match &mut self.backend {
            RenderingBackendContext::Vulkan(vk_ctx) => Some(vk_ctx),
            _ => None,
        }
    }
}