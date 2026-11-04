//! Exercise asynchronous window submission, swapchain resize, synchronous
//! uploads/readback, and the transition to offscreen rendering.
//!
//! Build with `--release --features vulkan --example submission_verify`.
//! Run with a new output directory; compare the .rgba files from runs with
//! PRPR_VULKAN_SUBMIT_THREAD=0 and 1. The window closes after 192 frames.

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("Build with --features vulkan");
    std::process::exit(2);
}

#[cfg(feature = "vulkan")]
fn main() {
    use miniquad::*;
    use std::{fs, path::PathBuf, time::Instant};

    const SIZE: u32 = 8;
    const WINDOW_FRAMES: usize = 192;

    struct Stage {
        output: PathBuf,
        target: Texture,
        pass: RenderPass,
        upload: Texture,
        frame: usize,
        started: Instant,
    }

    impl Stage {
        fn verify_color(&self, ctx: &mut Context, frame: usize, color: [u8; 4]) {
            let mut pixels = vec![0; (SIZE * SIZE * 4) as usize];
            self.target.read_pixels(ctx, &mut pixels);
            for pixel in pixels.chunks_exact(4) {
                assert!(
                    pixel.iter().zip(color).all(|(&a, b)| a.abs_diff(b) <= 1),
                    "frame {}: expected {:?}, read {:?}",
                    frame,
                    color,
                    pixel
                );
            }
            fs::write(self.output.join(format!("frame-{frame:03}.rgba")), pixels).unwrap();
        }

        fn clear_target(&self, ctx: &mut Context, color: [u8; 4]) {
            ctx.begin_pass(
                self.pass,
                PassAction::clear_color(
                    color[0] as f32 / 255.,
                    color[1] as f32 / 255.,
                    color[2] as f32 / 255.,
                    1.,
                ),
            );
            ctx.end_render_pass();
        }
    }

    impl EventHandler for Stage {
        fn update(&mut self, _ctx: &mut Context) {}

        fn draw(&mut self, ctx: &mut Context) {
            assert!(
                self.started.elapsed().as_secs() < 20,
                "window verification timed out"
            );
            if self.frame == WINDOW_FRAMES {
                // This must retire the last queued host submission before the
                // exporter reclaims its command buffers and graphics queue.
                ctx.vulkan_context().unwrap().enable_offscreen_video();
                for i in 0..8 {
                    ctx.vulkan_context().unwrap().begin_offscreen_frame();
                    let color = [21 + i as u8 * 17, 62, 153, 255];
                    self.clear_target(ctx, color);
                    ctx.vulkan_context().unwrap().finish_offscreen_frame();
                    ctx.vulkan_context().unwrap().wait_offscreen_idle();
                    self.verify_color(ctx, WINDOW_FRAMES + i, color);
                }
                fs::write(
                    self.output.join("complete.txt"),
                    "192 window frames; 2 resizes; upload/delete/readback; 8 offscreen frames\n",
                )
                .unwrap();
                ctx.order_quit();
                return;
            }

            if self.frame == 48 || self.frame == 112 {
                let (width, height) = if self.frame == 48 {
                    (208, 152)
                } else {
                    (160, 120)
                };
                ctx.set_window_size(width, height);
            }
            // Transfer and deletion paths must first reclaim host queue access.
            if self.frame % 31 == 0 {
                self.upload.update(ctx, &[self.frame as u8, 71, 193, 255]);
            }
            if self.frame == 79 {
                self.upload.delete(ctx);
                self.upload = Texture::from_rgba8(ctx, 1, 1, &[42, 87, 129, 255]);
            }

            let color = [
                (self.frame * 13 % 251) as u8,
                (self.frame * 7 % 239) as u8,
                117,
                255,
            ];
            self.clear_target(ctx, color);
            if self.frame % 16 == 15 {
                self.verify_color(ctx, self.frame, color);
            }
            ctx.begin_default_pass(PassAction::clear_color(
                color[0] as f32 / 255.,
                color[1] as f32 / 255.,
                117. / 255.,
                1.,
            ));
            ctx.end_render_pass();
            ctx.commit_frame();
            self.frame += 1;
        }
    }

    let output = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .expect("Expected a new output directory"),
    );
    fs::create_dir(&output).expect("Output directory must not already exist");
    miniquad::start(
        conf::Conf {
            window_title: "Vulkan submission verification".into(),
            window_width: 160,
            window_height: 120,
            platform: conf::Platform {
                rendering_backend: conf::RenderingBackend::Vulkan,
                swap_interval: Some(0),
                ..Default::default()
            },
            ..Default::default()
        },
        move |mut ctx| {
            let target = Texture::new_render_texture(
                &mut ctx,
                TextureParams {
                    width: SIZE,
                    height: SIZE,
                    format: TextureFormat::RGBA8,
                    wrap: TextureWrap::Clamp,
                    filter: FilterMode::Nearest,
                    mipmap_filter: MipmapFilter::None,
                },
            );
            let pass = RenderPass::new(&mut ctx, target, None);
            let upload = Texture::from_rgba8(&mut ctx, 1, 1, &[42, 87, 129, 255]);
            Box::new(Stage {
                output,
                target,
                pass,
                upload,
                frame: 0,
                started: Instant::now(),
            })
        },
    );
}
