//! Vulkan rendering backend implementation
//! 
//! This module provides a Vulkan-based rendering backend for miniquad,
//! offering modern graphics capabilities while maintaining the same
//! high-level API as the OpenGL backend.

#[cfg(feature = "vulkan")]
pub mod vk;
