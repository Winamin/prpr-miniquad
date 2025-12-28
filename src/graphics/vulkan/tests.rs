#[cfg(test)]
mod vulkan_tests {
    use super::*;
    
    #[test]
    #[cfg(feature = "vulkan")]
    fn test_vulkan_context_creation() {
        let context = VulkanContext::new();
        assert!(context.msaa_samples == vk::SampleCountFlags::TYPE_4);
    }
    
    #[test]
    #[cfg(feature = "vulkan")]
    fn test_msaa_sample_setting() {
        let mut context = VulkanContext::new();
        
        // Test valid sample counts
        assert!(context.set_msaa_samples(1).is_ok());
        assert!(context.msaa_samples == vk::SampleCountFlags::TYPE_1);
        
        assert!(context.set_msaa_samples(2).is_ok());
        assert!(context.msaa_samples == vk::SampleCountFlags::TYPE_2);
        
        assert!(context.set_msaa_samples(4).is_ok());
        assert!(context.msaa_samples == vk::SampleCountFlags::TYPE_4);
        
        assert!(context.set_msaa_samples(8).is_ok());
        assert!(context.msaa_samples == vk::SampleCountFlags::TYPE_8);
        
        // Test invalid sample count
        assert!(context.set_msaa_samples(3).is_err());
    }
    
    #[test]
    #[cfg(feature = "vulkan")]
    fn test_performance_stats() {
        let context = VulkanContext::new();
        let stats = context.get_performance_stats();
        
        assert_eq!(stats.buffer_count, 0);
        assert_eq!(stats.texture_count, 0);
        assert_eq!(stats.shader_count, 0);
        assert_eq!(stats.pipeline_count, 0);
        assert_eq!(stats.allocated_memory, 0);
        assert_eq!(stats.frame_time, 0.0);
        assert!(stats.msaa_enabled);
        assert!(stats.msaa_samples == vk::SampleCountFlags::TYPE_4);
    }
    
    #[test]
    #[cfg(not(feature = "vulkan"))]
    fn test_non_vulkan_stats() {
        let context = VulkanContext::new();
        let stats = context.get_performance_stats();
        
        assert!(!stats.msaa_enabled);
        assert!(stats.msaa_samples == vk::SampleCountFlags::TYPE_1);
    }
}