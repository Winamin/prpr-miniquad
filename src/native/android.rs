use crate::{
    event::{EventHandler, KeyCode, TouchPhase},
    native::egl::{self, LibEgl},
    native::NativeDisplay,
    Context,
};

use std::{cell::RefCell, sync::mpsc, thread};

pub use crate::gl::{self, *};

mod keycodes;

pub use ndk_sys;

pub mod ndk_utils;

#[no_mangle]
pub unsafe extern "C" fn JNI_OnLoad(
    vm: *mut ndk_sys::JavaVM,
    _: std::ffi::c_void,
) -> ndk_sys::jint {
    VM = vm as *mut _ as _;

    ndk_sys::JNI_VERSION_1_6 as _
}

extern "C" {
    fn quad_main();
}

/// Short recap on how miniquad on Android works
/// There is a MainActivity, a normal Java activity
/// It creates a View and pass a reference to a view to rust.
/// Rust spawn a thread that render things into this view as often as
/// possible.
/// Also MainActivty collects user input events and calls native rust functions.
///
/// This long explanation was to illustrate how we ended up with evets callback
/// and drawing in the different threads.
/// Message enum is used to send data from the callbacks to the drawing thread.
#[derive(Debug)]
enum Message {
    SurfaceChanged {
        window: *mut ndk_sys::ANativeWindow,
        width: i32,
        height: i32,
    },
    SurfaceCreated {
        window: *mut ndk_sys::ANativeWindow,
    },
    SurfaceDestroyed,
    Touch {
        phase: TouchPhase,
        touch_id: u64,
        x: f32,
        y: f32,
        time: u64,
    },
    Character {
        character: u32,
    },
    KeyDown {
        keycode: KeyCode,
    },
    KeyUp {
        keycode: KeyCode,
    },
    Pause,
    Resume,
    Destroy,
}
unsafe impl Send for Message {}

thread_local! {
    static MESSAGES_TX: RefCell<Option<mpsc::Sender<Message>>> = RefCell::new(None);
    static RENDER_THREAD: RefCell<Option<thread::JoinHandle<()>>> = RefCell::new(None);
}

fn send_message(message: Message) {
    MESSAGES_TX.with(|tx| {
        let mut tx = tx.borrow_mut();
        if let Some(tx) = tx.as_mut() { let _ = tx.send(message); }
    })
}

static mut ACTIVITY: ndk_sys::jobject = std::ptr::null_mut();
static mut VM: *mut ndk_sys::JavaVM = std::ptr::null_mut();

struct AndroidDisplay {
    screen_width: f32,
    screen_height: f32,
    fullscreen: bool,
}

impl NativeDisplay for AndroidDisplay {
    fn screen_size(&self) -> (f32, f32) {
        (self.screen_width as _, self.screen_height as _)
    }
    fn dpi_scale(&self) -> f32 {
        1.
    }
    fn high_dpi(&self) -> bool {
        true
    }
    fn order_quit(&mut self) {
        unsafe {
            let env = attach_jni_env();

            let get_method_id = (**env).GetMethodID.unwrap();
            let get_object_class = (**env).GetObjectClass.unwrap();
            let call_void_method = (**env).CallVoidMethod.unwrap();

            let mid = (get_method_id)(
                env,
                get_object_class(env, ACTIVITY),
                b"finish\0".as_ptr() as _,
                b"()V\0".as_ptr() as _,
            );
            (call_void_method)(env, ACTIVITY, mid);
        }
    }
    fn request_quit(&mut self) {}
    fn cancel_quit(&mut self) {}
    fn set_cursor_grab(&mut self, _grab: bool) {}
    fn show_mouse(&mut self, _shown: bool) {}
    fn set_mouse_cursor(&mut self, _cursor: crate::CursorIcon) {}
    fn set_window_size(&mut self, _new_width: u32, _new_height: u32) {}
    fn set_fullscreen(&mut self, fullscreen: bool) {
        unsafe {
            let env = attach_jni_env();
            set_full_screen(env, fullscreen);
        }
        self.fullscreen = fullscreen;
    }
    fn clipboard_get(&mut self) -> Option<String> {
        None
    }
    fn clipboard_set(&mut self, data: &str) {
        unsafe {
            let env = crate::native::attach_jni_env();
            let ctx = ndk_context::android_context().context();
            let class = (**env).GetObjectClass.unwrap()(env, ctx);
            let method = (**env).GetMethodID.unwrap()(
                env,
                class,
                b"copy\0".as_ptr() as _,
                b"(Ljava/lang/String;)V\0".as_ptr() as _,
            );
            let data = std::ffi::CString::new(data.to_owned()).unwrap();
            (**env).CallVoidMethod.unwrap()(
                env,
                ctx,
                method,
                (**env).NewStringUTF.unwrap()(env, data.as_ptr()),
            );
        }
    }
    fn show_keyboard(&mut self, show: bool) {
        unsafe {
            let env = attach_jni_env();
            ndk_utils::call_void_method!(env, ACTIVITY, "showKeyboard", "(Z)V", show as i32);
        }
    }
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub unsafe fn console_debug(msg: *const ::std::os::raw::c_char) {
    ndk_sys::__android_log_write(
        ndk_sys::android_LogPriority_ANDROID_LOG_DEBUG as _,
        b"SAPP\0".as_ptr() as _,
        msg,
    );
}

pub unsafe fn console_info(msg: *const ::std::os::raw::c_char) {
    ndk_sys::__android_log_write(
        ndk_sys::android_LogPriority_ANDROID_LOG_INFO as _,
        b"SAPP\0".as_ptr() as _,
        msg,
    );
}

pub unsafe fn console_warn(msg: *const ::std::os::raw::c_char) {
    ndk_sys::__android_log_write(
        ndk_sys::android_LogPriority_ANDROID_LOG_WARN as _,
        b"SAPP\0".as_ptr() as _,
        msg,
    );
}

pub unsafe fn console_error(msg: *const ::std::os::raw::c_char) {
    ndk_sys::__android_log_write(
        ndk_sys::android_LogPriority_ANDROID_LOG_ERROR as _,
        b"SAPP\0".as_ptr() as _,
        msg,
    );
}

// fn log_info(message: &str) {
//     use std::ffi::CString;

//     let msg = CString::new(message).unwrap_or_else(|_| panic!());

//     unsafe { console_info(msg.as_ptr()) };
// }

struct MainThreadState {
    libegl: LibEgl,
    context: Context,
    egl_display: egl::EGLDisplay,
    egl_config: egl::EGLConfig,
    egl_context: egl::EGLContext,
    surface: egl::EGLSurface,
    display: AndroidDisplay,
    window: *mut ndk_sys::ANativeWindow,
    event_handler: Box<dyn EventHandler>,
    quit: bool,
}

impl MainThreadState {
    unsafe fn destroy_surface(&mut self) {
        (self.libegl.eglMakeCurrent.unwrap())(
            self.egl_display,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        (self.libegl.eglDestroySurface.unwrap())(self.egl_display, self.surface);
        self.surface = std::ptr::null_mut();
    }

    unsafe fn update_surface(&mut self, window: *mut ndk_sys::ANativeWindow) {
        if !self.window.is_null() {
            ndk_sys::ANativeWindow_release(self.window);
        }
        self.window = window;
        if self.surface.is_null() == false {
            self.destroy_surface();
        }

        self.surface = (self.libegl.eglCreateWindowSurface.unwrap())(
            self.egl_display,
            self.egl_config,
            window as _,
            std::ptr::null_mut(),
        );

        assert!(!self.surface.is_null());

        let res = (self.libegl.eglMakeCurrent.unwrap())(
            self.egl_display,
            self.surface,
            self.surface,
            self.egl_context,
        );

        assert!(res != 0);
    }

    fn process_message(&mut self, msg: Message) {
        match msg {
            Message::SurfaceCreated { window } => unsafe {
                self.update_surface(window);
            },
            Message::SurfaceDestroyed => unsafe {
                self.destroy_surface();
            },
            Message::SurfaceChanged {
                window,
                width,
                height,
            } => {
                unsafe {
                    self.update_surface(window);
                }

                self.display.screen_width = width as _;
                self.display.screen_height = height as _;
                self.event_handler.resize_event(
                    self.context.with_display(&mut self.display),
                    width as _,
                    height as _,
                );
            }
            Message::Touch {
                phase,
                touch_id,
                x,
                y,
                time,
            } => {
                self.event_handler.touch_event(
                    self.context.with_display(&mut self.display),
                    phase,
                    touch_id,
                    x,
                    y,
                    time as f64 / 1000.,
                );
            }
            Message::Character { character } => {
                if let Some(character) = char::from_u32(character) {
                    self.event_handler.char_event(
                        self.context.with_display(&mut self.display),
                        character,
                        Default::default(),
                        false,
                    );
                }
            }
            Message::KeyDown { keycode } => {
                self.event_handler.key_down_event(
                    self.context.with_display(&mut self.display),
                    keycode,
                    Default::default(),
                    false,
                );
            }
            Message::KeyUp { keycode } => {
                self.event_handler.key_up_event(
                    self.context.with_display(&mut self.display),
                    keycode,
                    Default::default(),
                );
            }
            Message::Pause => self
                .event_handler
                .window_minimized_event(self.context.with_display(&mut self.display)),
            Message::Resume => {
                if self.display.fullscreen {
                    unsafe {
                        let env = attach_jni_env();
                        set_full_screen(env, true);
                    }
                }

                self.event_handler
                    .window_restored_event(self.context.with_display(&mut self.display))
            }
            Message::Destroy => {
                self.quit = true;
            }
        }
    }

    fn frame(&mut self) {
        self.event_handler
            .update(self.context.with_display(&mut self.display));

        if self.surface.is_null() == false {
            self.event_handler
                .draw(self.context.with_display(&mut self.display));

            unsafe {
                (self.libegl.eglSwapBuffers.unwrap())(self.egl_display, self.surface);
            }
        }
    }
}

/// Get the JNI Env by calling ndk's AttachCurrentThread
///
/// Safety note: This function is not exactly correct now, it should be fixed!
///
/// AttachCurrentThread should be called at least once for any given thread that
/// wants to use the JNI and DetachCurrentThread should be called only once, when
/// the thread stack is empty and the thread is about to stop
///
/// calling AttachCurrentThread from the same thread multiple time is very cheap
///
/// BUT! there is no DetachCurrentThread call right now, this code:
/// `thread::spawn(|| attach_jni_env());` will lead to internal jni crash :/
/// thread::spawn(|| { attach_jni_env(); loop {} }); is basically what miniquad
/// is doing. this is not correct, but works
/// TODO: the problem here -
/// TODO:   thread::spawn(|| { Attach(); .. Detach() }); will not work as well.
/// TODO: JNI will check that thread's stack is still alive and will crash.
///
/// TODO: Figure how to get into the thread destructor to correctly call Detach
/// TODO: (this should be a GH issue)
/// TODO: for reference - grep for "pthread_setspecific" in SDL2 sources, SDL fixed it!
pub unsafe fn attach_jni_env() -> *mut ndk_sys::JNIEnv {
    let mut env: *mut ndk_sys::JNIEnv = std::ptr::null_mut();
    let attach_current_thread = (**VM).AttachCurrentThread.unwrap();

    let res = attach_current_thread(VM, &mut env, std::ptr::null_mut());
    assert!(res == 0);

    env
}

pub unsafe fn run<F>(conf: crate::conf::Conf, f: F)
where
    F: 'static + FnOnce(&mut crate::Context) -> Box<dyn EventHandler>,
{
    {
        use std::ffi::CString;
        use std::panic;

        panic::set_hook(Box::new(|info| {
            let msg = CString::new(format!("{:?}", info)).unwrap_or_else(|_| {
                CString::new(format!("MALFORMED ERROR MESSAGE {:?}", info.location())).unwrap()
            });
            console_error(msg.as_ptr());
        }));
    }

    if conf.fullscreen {
        let env = attach_jni_env();
        set_full_screen(env, true);
    }

    // yeah, just adding Send to outer F will do it, but it will brake the API
    // in other backends
    struct SendHack<F>(F);
    unsafe impl<F> Send for SendHack<F> {}

    let f = SendHack(f);

    let (tx, rx) = mpsc::channel();

    MESSAGES_TX.with(move |messages_tx| *messages_tx.borrow_mut() = Some(tx));

    let render_thread = thread::spawn(move || {
        let mut libegl = LibEgl::try_load().expect("Cant load LibEGL");

        // skip all the messages until android will be able to actually open a window
        //
        // sometimes before launching an app android will show a permission dialog
        // it is important to create GL context only after a first SurfaceChanged
        let (window, screen_width, screen_height) = if conf.headless {
            (std::ptr::null_mut(), conf.window_width as f32, conf.window_height as f32)
        } else {
            let mut ready = None;
            while ready.is_none() {
                match rx.recv() {
                    Ok(Message::SurfaceChanged { window, width, height }) => {
                        ready = Some((window, width as f32, height as f32));
                    }
                    Ok(Message::SurfaceCreated { window }) => ndk_sys::ANativeWindow_release(window),
                    Ok(Message::Destroy) | Err(_) => return,
                    _ => {}
                }
            }
            ready.unwrap()
        };

        let surface_type = if conf.headless {
            egl::EGL_PBUFFER_BIT
        } else {
            egl::EGL_WINDOW_BIT | egl::EGL_PBUFFER_BIT
        };
        let (egl_context, egl_config, egl_display, _es_major) = crate::native::egl::create_egl_context_surface(
            &mut libegl,
            std::ptr::null_mut(), /* EGL_DEFAULT_DISPLAY */
            conf.platform.framebuffer_alpha,
            surface_type,
        )
        .expect("Cant create EGL context");

        assert!(!egl_display.is_null());
        assert!(!egl_config.is_null());

        crate::native::gl::load_gl_funcs(|proc| {
            let name = std::ffi::CString::new(proc).unwrap();
            libegl.eglGetProcAddress.expect("non-null function pointer")(name.as_ptr() as _)
        });

        let surface = if conf.headless {
            #[rustfmt::skip]
            let attribs = [
                egl::EGL_WIDTH, screen_width as u32,
                egl::EGL_HEIGHT, screen_height as u32,
                egl::EGL_NONE,
            ];
            (libegl.eglCreatePbufferSurface.unwrap())(
                egl_display,
                egl_config,
                attribs.as_ptr() as _,
            )
        } else {
            (libegl.eglCreateWindowSurface.unwrap())(
                egl_display,
                egl_config,
                window as _,
                std::ptr::null_mut(),
            )
        };

        if (libegl.eglMakeCurrent.unwrap())(egl_display, surface, surface, egl_context) == 0 {
            panic!();
        }

        let mut context = Context::new(
            crate::conf::RenderingBackend::OpenGL,
            gl::is_gl2(),
            None,
            (screen_width as _, screen_height as _),
        );

        let mut display = AndroidDisplay {
            screen_width,
            screen_height,
            fullscreen: conf.fullscreen,
        };
        let event_handler = f.0(context.with_display(&mut display));
        let mut s = MainThreadState {
            libegl,
            egl_display,
            egl_config,
            egl_context,
            surface,
            context,
            display,
            window,
            event_handler,
            quit: false,
        };

        while !s.quit {
            // process all the messages from the main thread
            while let Ok(msg) = rx.try_recv() {
                s.process_message(msg);
            }

            if s.quit { break; }
            s.frame();
            if s.surface.is_null() { thread::sleep(std::time::Duration::from_millis(16)); }

            thread::yield_now();
        }

        // SurfaceView may have already gone away. Keep EGL current while releasing GPU resources.
        if s.surface.is_null() {
            let attributes = [egl::EGL_WIDTH, 1, egl::EGL_HEIGHT, 1, egl::EGL_NONE];
            s.surface = (s.libegl.eglCreatePbufferSurface.unwrap())(s.egl_display, s.egl_config, attributes.as_ptr() as _);
            (s.libegl.eglMakeCurrent.unwrap())(s.egl_display, s.surface, s.surface, s.egl_context);
        }
        s.event_handler.destroy_event(s.context.with_display(&mut s.display));
        (s.libegl.eglMakeCurrent.unwrap())(
            s.egl_display,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        (s.libegl.eglDestroySurface.unwrap())(s.egl_display, s.surface);
        (s.libegl.eglDestroyContext.unwrap())(s.egl_display, s.egl_context);
        (s.libegl.eglTerminate.unwrap())(s.egl_display);
        if !s.window.is_null() { ndk_sys::ANativeWindow_release(s.window); }
        drop(s);
        // This is a native rendering thread; ART requires detaching before it exits.
        (**VM).DetachCurrentThread.unwrap()(VM);
    });
    RENDER_THREAD.with(|thread| *thread.borrow_mut() = Some(render_thread));
}

#[no_mangle]
extern "C" fn jni_on_load(vm: *mut std::ffi::c_void) {
    unsafe {
        VM = vm as _;
    }
}

unsafe fn create_native_window(surface: ndk_sys::jobject) -> *mut ndk_sys::ANativeWindow {
    let env = attach_jni_env();

    ndk_sys::ANativeWindow_fromSurface(env, surface)
}

#[no_mangle]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_initializeContext(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    activity: ndk_sys::jobject,
) {
    let env = attach_jni_env();
    let activity = (**env).NewGlobalRef.unwrap()(env, activity);
    ndk_context::initialize_android_context(std::mem::transmute(VM), activity as _);
}

#[no_mangle]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_releaseContext(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    ndk_context::release_android_context();
}

#[no_mangle]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_activityOnCreate(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    activity: ndk_sys::jobject,
) {
    let env = attach_jni_env();
    ACTIVITY = (**env).NewGlobalRef.unwrap()(env, activity);
    quad_main();
}

#[no_mangle]
unsafe extern "C" fn Java_quad_1native_QuadNative_activityOnResume(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    send_message(Message::Resume);
}

#[no_mangle]
unsafe extern "C" fn Java_quad_1native_QuadNative_activityOnPause(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    send_message(Message::Pause);
}

#[no_mangle]
unsafe extern "C" fn Java_quad_1native_QuadNative_activityOnDestroy(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    send_message(Message::Destroy);
    RENDER_THREAD.with(|thread| {
        if let Some(thread) = thread.borrow_mut().take() { let _ = thread.join(); }
    });
    MESSAGES_TX.with(|sender| *sender.borrow_mut() = None);
    let env = attach_jni_env();
    if !ACTIVITY.is_null() {
        (**env).DeleteGlobalRef.unwrap()(env, ACTIVITY);
        ACTIVITY = std::ptr::null_mut();
    }
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnSurfaceCreated(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    surface: ndk_sys::jobject,
) {
    let window = unsafe { create_native_window(surface) };
    send_message(Message::SurfaceCreated { window });
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnSurfaceDestroyed(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    send_message(Message::SurfaceDestroyed);
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnSurfaceChanged(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    surface: ndk_sys::jobject,
    width: ndk_sys::jint,
    height: ndk_sys::jint,
) {
    let window = unsafe { create_native_window(surface) };

    send_message(Message::SurfaceChanged {
        window,
        width: width as _,
        height: height as _,
    });
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnTouch(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    touch_id: ndk_sys::jint,
    action: ndk_sys::jint,
    x: ndk_sys::jfloat,
    y: ndk_sys::jfloat,
    time: ndk_sys::jlong,
) {
    let phase = match action {
        0 => TouchPhase::Moved,
        1 => TouchPhase::Ended,
        2 => TouchPhase::Started,
        3 => TouchPhase::Cancelled,
        x => panic!("Unsupported touch phase: {}", x),
    };

    send_message(Message::Touch {
        phase,
        touch_id: touch_id as _,
        x: x as f32,
        y: y as f32,
        time: time as u64,
    });
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnKeyDown(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    keycode: ndk_sys::jint,
) {
    let keycode = keycodes::translate_keycode(keycode as _);

    send_message(Message::KeyDown { keycode });
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnKeyUp(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    keycode: ndk_sys::jint,
) {
    let keycode = keycodes::translate_keycode(keycode as _);

    send_message(Message::KeyUp { keycode });
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnCharacter(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    character: ndk_sys::jint,
) {
    send_message(Message::Character {
        character: character as u32,
    });
}

unsafe fn set_full_screen(env: *mut ndk_sys::JNIEnv, fullscreen: bool) {
    ndk_utils::call_void_method!(env, ACTIVITY, "setFullScreen", "(Z)V", fullscreen as i32);
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct android_asset {
    pub content: *mut ::std::os::raw::c_char,
    pub content_length: ::std::os::raw::c_int,
}

// According to documentation, AAssetManager_fromJava is as available as an
// AAssetManager_open, which was used before
// For some reason it is missing fron ndk_sys binding
extern "C" {
    pub fn AAssetManager_fromJava(
        env: *mut ndk_sys::JNIEnv,
        assetManager: ndk_sys::jobject,
    ) -> *mut ndk_sys::AAssetManager;
}

pub(crate) unsafe fn load_asset(filepath: *const ::std::os::raw::c_char, out: *mut android_asset) {
    let mut env: *mut ndk_sys::JNIEnv = std::ptr::null_mut();
    let mut activity: ndk_sys::jobject = ACTIVITY;

    if !ndk_context::android_context().vm().is_null() {
        let context = ndk_context::android_context();
        let vm = context.vm() as *mut ndk_sys::JavaVM;
        let attach_current_thread = (**vm).AttachCurrentThread.unwrap();
        if attach_current_thread(vm, &mut env, std::ptr::null_mut()) != 0 {
            return;
        }
        activity = context.context() as ndk_sys::jobject;
    } else if !VM.is_null() {
        env = attach_jni_env();
    }

    if env.is_null() || activity.is_null() {
        return;
    }

    let get_method_id = (**env).GetMethodID.unwrap();
    let get_object_class = (**env).GetObjectClass.unwrap();
    let call_object_method = (**env).CallObjectMethod.unwrap();

    let mid = (get_method_id)(
        env,
        get_object_class(env, activity),
        b"getAssets\0".as_ptr() as _,
        b"()Landroid/content/res/AssetManager;\0".as_ptr() as _,
    );
    if mid.is_null() {
        return;
    }
    let asset_manager = (call_object_method)(env, activity, mid);
    if asset_manager.is_null() {
        return;
    }
    let mgr = AAssetManager_fromJava(env, asset_manager);
    if mgr.is_null() {
        return;
    }
    let asset = ndk_sys::AAssetManager_open(mgr, filepath, ndk_sys::AASSET_MODE_BUFFER as _);
    if asset.is_null() {
        return;
    }
    let length = ndk_sys::AAsset_getLength64(asset);
    let buffer = libc::malloc(length as _);
    if !buffer.is_null() && ndk_sys::AAsset_read(asset, buffer, length as _) == length as i32 {
        (*out).content_length = length as _;
        (*out).content = buffer as _;
    } else {
        libc::free(buffer);
    }
    ndk_sys::AAsset_close(asset);
}
