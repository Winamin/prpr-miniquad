use miniquad::*;

struct BasicTest {
}

impl BasicTest {
    fn new() -> Self {
        Self {}
    }
}

impl EventHandler for BasicTest {
    fn update(&mut self, _ctx: &mut Context) {
    }
    
    fn draw(&mut self, ctx: &mut Context) {
        ctx.begin_default_pass(Default::default());
        
        // 清除屏幕为蓝色
        ctx.clear(Some((0.0, 0.5, 1.0, 1.0)), None, None);
        
        ctx.end_render_pass();
    }
}

fn main() {
    let conf = miniquad::conf::Conf {
        window_title: "基本测试".to_string(),
        window_width: 400,
        window_height: 300,
        ..Default::default()
    };
    
    miniquad::start(conf, |mut ctx| {
        Box::new(BasicTest::new())
    });
}