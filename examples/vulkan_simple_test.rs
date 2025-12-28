use miniquad::*;

struct VulkanTest {
    rotation: f32,
}

impl VulkanTest {
    fn new() -> Self {
        Self { rotation: 0.0 }
    }
}

impl EventHandler for VulkanTest {
    fn update(&mut self, _ctx: &mut Context) {
        self.rotation += 0.02;
    }
    
    fn draw(&mut self, ctx: &mut Context) {
        ctx.begin_default_pass(Default::default());
        
        // 清除屏幕为深蓝色
        ctx.clear(Some((0.1, 0.2, 0.3, 1.0)), None, None);
        
        // 如果启用了 Vulkan 功能，显示相关信息
        #[cfg(feature = "vulkan")]
        {
            // 这里可以添加 Vulkan 特定的渲染代码
            // 目前只是清除屏幕以验证 Vulkan 后端工作正常
        }
        
        ctx.end_render_pass();
        
        // 显示一些信息（如果支持的话）
        #[cfg(feature = "vulkan")]
        {
            // 注意：这里不能直接使用 egui，因为它需要额外依赖
            // 在实际应用中，可以通过其他方式显示 Vulkan 信息
        }
    }
}

fn main() {
    let conf = miniquad::conf::Conf {
        window_title: "Vulkan 测试".to_string(),
        window_width: 800,
        window_height: 600,
        sample_count: 4, // 启用 4x MSAA
        high_dpi: true,
        platform: miniquad::conf::Platform {
            rendering_backend: miniquad::conf::RenderingBackend::Vulkan,
            ..Default::default()
        },
        ..Default::default()
    };
    
    println!("启动 Vulkan 测试...");
    println!("如果看到蓝色窗口，说明 Vulkan 后端工作正常");
    
    miniquad::start(conf, |mut ctx| {
        Box::new(VulkanTest::new())
    });
}