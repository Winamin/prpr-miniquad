use miniquad::*;

struct SimpleVulkanTest {
    #[cfg(feature = "vulkan")]
    vulkan_context: miniquad::graphics::vulkan::vk::VulkanContext,
}

impl SimpleVulkanTest {
    fn new() -> Self {
        #[cfg(feature = "vulkan")]
        {
            let vulkan_context = miniquad::graphics::vulkan::vk::VulkanContext::new();
            
            // 测试 MSAA 设置
            let mut ctx = vulkan_context;
            assert!(ctx.set_msaa_samples(4).is_ok());
            assert!(ctx.msaa_samples == ash::vk::SampleCountFlags::TYPE_4);
            
            // 测试性能统计
            let stats = ctx.get_performance_stats();
            assert!(stats.msaa_enabled);
            
            Self { vulkan_context: ctx }
        }
        
        #[cfg(not(feature = "vulkan"))]
        {
            Self { }
        }
    }
}

impl EventHandler for SimpleVulkanTest {
    fn update(&mut self, _ctx: &mut Context) {
        // 空实现
    }
    
    fn draw(&mut self, ctx: &mut Context) {
        ctx.begin_default_pass(Default::default());
        
        #[cfg(feature = "vulkan")]
        {
            egui::Window::new("Vulkan 测试").show(ctx, |ui| {
                ui.label("Vulkan 后端初始化成功");
                ui.label("MSAA 功能正常工作");
                ui.label("性能统计收集正常");
            });
        }
        
        #[cfg(not(feature = "vulkan"))]
        {
            egui::Window::new("Vulkan 不可用").show(ctx, |ui| {
                ui.label("Vulkan 功能未启用");
                ui.label("请使用 --features vulkan 运行");
            });
        }
        
        ctx.end_render_pass();
    }
}

fn main() {
    let conf = Conf {
        window_title: "简单 Vulkan 测试".to_string(),
        window_width: 400,
        window_height: 300,
        sample_count: 4,
        ..Default::default()
    };
    
    miniquad::start(conf, |ctx| {
        let test = SimpleVulkanTest::new();
        UserData::owning(test, EventHandler::on_update, EventHandler::on_draw)
    });
}