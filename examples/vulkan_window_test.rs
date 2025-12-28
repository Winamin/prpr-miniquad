use miniquad::*;

#[cfg(feature = "vulkan")]
use miniquad::graphics::vulkan::*;

struct VulkanWindowTest {
    pipeline: Pipeline,
    bindings: Bindings,
    rotation: f32,
}

impl VulkanWindowTest {
    #[cfg(feature = "vulkan")]
    fn new(ctx: &mut Context) -> Self {
        // 顶点着色器
        let vertex_shader = "
            #version 450
            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec3 color;
            
            layout(location = 0) out vec3 v_color;
            
            uniform float u_time;
            
            void main() {
                float angle = u_time;
                mat2 rotation = mat2(
                    cos(angle), -sin(angle),
                    sin(angle), cos(angle)
                );
                vec2 rotated_pos = rotation * pos;
                gl_Position = vec4(rotated_pos, 0.0, 1.0);
                v_color = color;
            }
        ";
        
        // 片段着色器
        let fragment_shader = "
            #version 450
            layout(location = 0) in vec3 v_color;
            layout(location = 0) out vec4 frag_color;
            
            void main() {
                frag_color = vec4(v_color, 1.0);
            }
        ";
        
        // 创建着色器
        let shader = Shader::new(ctx, vertex_shader, fragment_shader, ShaderMeta::default())
            .expect("Failed to create shader");
        
        // 创建管线
        let pipeline = Pipeline::new(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("pos", VertexFormat::Float2),
                VertexAttribute::new("color", VertexFormat::Float3),
            ],
            shader,
        ).expect("Failed to create pipeline");
        
        // 创建三角形顶点数据
        let vertices: &[f32] = &[
            // 位置        颜色
            -0.5, -0.5,   1.0, 0.0, 0.0,  // 左下 - 红色
             0.5, -0.5,   0.0, 1.0, 0.0,  // 右下 - 绿色
             0.0,  0.5,   0.0, 0.0, 1.0,  // 顶部 - 蓝色
        ];
        
        // 创建顶点缓冲区
        let vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, vertices);
        
        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer],
            index_buffer: Buffer::default(),
            images: vec![],
        };
        
        Self {
            pipeline,
            bindings,
            rotation: 0.0,
        }
    }
    
    #[cfg(not(feature = "vulkan"))]
    fn new(_ctx: &mut Context) -> Self {
        Self {
            pipeline: Pipeline::default(),
            bindings: Bindings::default(),
            rotation: 0.0,
        }
    }
}

impl EventHandler for VulkanWindowTest {
    fn update(&mut self, _ctx: &mut Context) {
        // 更新旋转角度
        self.rotation += 0.02;
    }
    
    fn draw(&mut self, ctx: &mut Context) {
        // 清除屏幕为深蓝色
        ctx.begin_default_pass(PassAction::clear_color(0.1, 0.2, 0.3, 1.0));
        
        #[cfg(feature = "vulkan")]
        {
            // 应用管线和绑定
            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_bindings(&self.bindings);
            
            // 设置 uniform 变量
            let uniforms = &[self.rotation];
            ctx.apply_uniforms(uniforms);
            
            // 绘制三角形
            ctx.draw(0, 3, 1);
        }
        
        ctx.end_render_pass();
        
        // 显示渲染信息
        #[cfg(feature = "vulkan")]
        {
            egui::Window::new("Vulkan 渲染测试").show(ctx, |ui| {
                ui.label("Vulkan 后端正在运行");
                ui.label(format!("旋转角度: {:.2} 度", self.rotation * 180.0 / std::f32::consts::PI));
                ui.label("正在渲染一个旋转的彩色三角形");
                
                if let Ok(stats) = ctx.get_vulkan_stats() {
                    ui.separator();
                    ui.label("Vulkan 性能统计:");
                    ui.label(format!("MSAA 采样数: {:?}", stats.msaa_samples));
                    ui.label(format!("MSAA 启用: {}", stats.msaa_enabled));
                    ui.label(format!("缓冲区数量: {}", stats.buffer_count));
                    ui.label(format!("纹理数量: {}", stats.texture_count));
                    ui.label(format!("着色器数量: {}", stats.shader_count));
                    ui.label(format!("管线数量: {}", stats.pipeline_count));
                }
            });
        }
        
        #[cfg(not(feature = "vulkan"))]
        {
            egui::Window::new("Vulkan 不可用").show(ctx, |ui| {
                ui.label("Vulkan 功能未启用");
                ui.label("请使用以下命令运行:");
                ui.label("cargo run --features vulkan --example vulkan_window_test");
            });
        }
    }
}

fn main() {
    // 配置窗口
    let conf = Conf {
        window_title: "Vulkan 渲染窗口测试".to_string(),
        window_width: 800,
        window_height: 600,
        sample_count: 4, // 启用 4x MSAA
        high_dpi: true,
        fullscreen: false,
        platform: miniquad::conf::Platform {
            rendering_backend: miniquad::conf::RenderingBackend::Vulkan,
            ..Default::default()
        },
        ..Default::default()
    };
    
    // 启动应用
    miniquad::start(conf, |ctx| {
        let test = VulkanWindowTest::new(ctx);
        UserData::owning(test, EventHandler::on_update, EventHandler::on_draw)
    });
}