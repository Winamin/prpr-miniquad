use miniquad::*;

struct Stage {
    pipeline: Pipeline,
    bindings: Bindings,
}

impl Stage {
    fn new(ctx: &mut Context) -> Self {
        #[rustfmt::skip]
        let vertices: [f32; 30] = [
            // positions      // colors
            0.0,  0.5, 0.0,   1.0, 0.0, 0.0, // top
            -0.5, -0.5, 0.0,  0.0, 1.0, 0.0, // left
            0.5, -0.5, 0.0,   0.0, 0.0, 1.0, // right
        ];

        let vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &vertices);
        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer],
            index_buffer: Buffer::immutable(ctx, BufferType::IndexBuffer, &[]),
            images: vec![],
        };

        let shader = Shader::new(ctx, VERTEX_SHADER, FRAGMENT_SHADER, ShaderMeta { uniforms: vec![], images: vec![] }).unwrap();

        let pipeline = Pipeline::new(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("position", VertexFormat::Float3),
                VertexAttribute::new("color", VertexFormat::Float3),
            ],
            shader,
        );

        Self { pipeline, bindings }
    }
}

impl EventHandler for Stage {
    fn update(&mut self, _ctx: &mut Context) {}

    fn draw(&mut self, ctx: &mut Context) {
        ctx.begin_default_pass(PassAction::clear_color(0.1, 0.1, 0.1, 1.0));
        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&self.bindings);
        ctx.draw(0, 3);
        ctx.end_render_pass();
    }
}

fn main() {
    // Test with OpenGL 4.6 (maximum on Windows/Linux)
    // The system will try 4.6, then fall back to 4.1, 3.3, 3.2, and finally 2.1
    let conf = Conf {
        window_title: "OpenGL 4.6 Test".to_string(),
        window_width: 800,
        window_height: 600,
        window_resizable: true,
        platform: Platform {
            opengl_version: conf::GlVersion::gl_4_6(),
            ..Default::default()
        },
        ..Default::default()
    };

    miniquad::start(conf, |ctx| Box::new(Stage::new(ctx)));
}

// Simple vertex shader
const VERTEX_SHADER: &str = r#"#version 150
in vec3 position;
in vec3 color;
out vec3 v_color;

void main() {
    gl_Position = vec4(position, 1.0);
    v_color = color;
}
"#;

// Simple fragment shader
const FRAGMENT_SHADER: &str = r#"#version 150
in vec3 v_color;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_color, 1.0);
}
"#;