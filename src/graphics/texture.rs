use crate::{native::gl::*, native::*, Context};
use super::GraphicsBackend;

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub struct Texture {
    pub(crate) texture: GLuint,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub(crate) wrap: TextureWrap,
    pub(crate) filter: FilterMode,
    pub(crate) mipmap_filter: MipmapFilter,
}

impl Texture {
    pub fn empty() -> Texture {
        Texture {
            texture: 0,
            width: 0,
            height: 0,
            format: TextureFormat::RGBA8,
            wrap: TextureWrap::Clamp,
            filter: FilterMode::Linear,
            mipmap_filter: MipmapFilter::None,
        }
    }

    pub fn gl_internal_id(&self) -> GLuint {
        self.texture
    }

    pub unsafe fn from_raw_id(texture: GLuint, format: TextureFormat) -> Self {
        Self {
            texture,
            width: 0,
            height: 0,
            format,
            wrap: TextureWrap::Clamp,
            filter: FilterMode::Linear,
            mipmap_filter: MipmapFilter::None,
        }
    }

    /// Delete GPU texture
    pub fn delete(&self, ctx: &mut Context) {
        match ctx.backend() {
            GraphicsBackend::OpenGL(_) => unsafe {
                glDeleteTextures(1, &self.texture as *const _);
            },
            #[cfg(feature = "vulkan")]
            GraphicsBackend::Vulkan(_) => ctx.vulkan().delete_texture(self.texture as usize),
        }
    }
}

/// List of all the possible formats of input data when uploading to texture.
/// The list is built by intersection of texture formats supported by 3.3 core profile and webgl1.
#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum TextureFormat {
    RGB8,
    RGBA8,
    Depth,
    Alpha,
    LuminanceAlpha,
}

impl TextureFormat {
    /// Converts from TextureFormat to (internal_format, format, pixel_type)
    pub fn into_gl_params(self, alpha_texture: bool) -> (GLenum, GLenum, GLenum) {
        match self {
            TextureFormat::RGB8 => (GL_RGB, GL_RGB, GL_UNSIGNED_BYTE),
            TextureFormat::RGBA8 => (GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE),
            TextureFormat::Depth => (GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT),

            #[cfg(target_arch = "wasm32")]
            TextureFormat::Alpha => (GL_ALPHA, GL_ALPHA, GL_UNSIGNED_BYTE),
            #[cfg(not(target_arch = "wasm32"))]
            TextureFormat::Alpha if alpha_texture => (GL_ALPHA, GL_ALPHA, GL_UNSIGNED_BYTE),
            #[cfg(not(target_arch = "wasm32"))]
            TextureFormat::Alpha => (GL_R8, GL_RED, GL_UNSIGNED_BYTE), // texture updates will swizzle Red -> Alpha to match WASM

            #[cfg(target_arch = "wasm32")]
            TextureFormat::LuminanceAlpha => {
                (GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE)
            }
            #[cfg(not(target_arch = "wasm32"))]
            TextureFormat::LuminanceAlpha if alpha_texture => {
                (GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE)
            }
            #[cfg(not(target_arch = "wasm32"))]
            TextureFormat::LuminanceAlpha => (GL_RG, GL_RG, GL_UNSIGNED_BYTE), // texture updates will swizzle Green -> Alpha to match WASM
        }
    }

    /// Returns the size in bytes of texture with `dimensions`.
    pub fn size(self, width: u32, height: u32) -> u32 {
        let square = width * height;
        match self {
            TextureFormat::RGB8 => 3 * square,
            TextureFormat::RGBA8 => 4 * square,
            TextureFormat::Depth => 2 * square,
            TextureFormat::Alpha => 1 * square,
            TextureFormat::LuminanceAlpha => 2 * square,
        }
    }
}

impl Default for TextureParams {
    fn default() -> Self {
        TextureParams {
            format: TextureFormat::RGBA8,
            wrap: TextureWrap::Clamp,
            filter: FilterMode::Linear,
            mipmap_filter: MipmapFilter::None,
            width: 0,
            height: 0,
        }
    }
}

/// Sets the wrap parameter for texture.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum TextureWrap {
    /// Samples at coord x + 1 map to coord x.
    Repeat = GL_REPEAT as isize,
    /// Samples at coord x + 1 map to coord 1 - x.
    Mirror = GL_MIRRORED_REPEAT as isize,
    /// Samples at coord x + 1 map to coord 1.
    Clamp = GL_CLAMP_TO_EDGE as isize,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum FilterMode {
    Linear = GL_LINEAR as isize,
    Nearest = GL_NEAREST as isize,
}

/// How the levels of a mipmapped texture are filtered.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum MipmapFilter {
    /// The texture has a single level.
    None,
    Nearest,
    Linear,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TextureAccess {
    /// Used as read-only from GPU
    Static,
    /// Can be written to from GPU
    RenderTarget,
}

#[derive(Debug, Copy, Clone)]
pub struct TextureParams {
    pub format: TextureFormat,
    pub wrap: TextureWrap,
    pub filter: FilterMode,
    pub mipmap_filter: MipmapFilter,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    /// Shorthand for `new(ctx, TextureAccess::RenderTarget, params)`
    pub fn new_render_texture(ctx: &mut Context, params: TextureParams) -> Texture {
        Self::new(ctx, TextureAccess::RenderTarget, None, params)
    }

    pub fn new(
        ctx: &mut Context,
        access: TextureAccess,
        bytes: Option<&[u8]>,
        params: TextureParams,
    ) -> Texture {
        if let Some(bytes_data) = bytes {
            assert_eq!(
                params.format.size(params.width, params.height) as usize,
                bytes_data.len()
            );
        }

        match ctx.backend() {
            GraphicsBackend::OpenGL(_) => Texture::new_opengl(ctx, bytes, params),
            #[cfg(feature = "vulkan")]
            GraphicsBackend::Vulkan(_) => Texture {
                texture: ctx.vulkan().new_texture(bytes, params, access) as GLuint,
                width: params.width,
                height: params.height,
                format: params.format,
                wrap: params.wrap,
                filter: params.filter,
                mipmap_filter: params.mipmap_filter,
            },
        }
    }

    fn new_opengl(
        ctx: &mut Context,
        bytes: Option<&[u8]>,
        params: TextureParams,
    ) -> Texture {
        let alpha_texture = ctx.features().alpha_texture;
        let (internal_format, format, pixel_type) = params.format.into_gl_params(alpha_texture);

        ctx.opengl().cache.store_texture_binding(0);

        let mut texture: GLuint = 0;

        unsafe {
            glGenTextures(1, &mut texture as *mut _);
            ctx.opengl().cache.bind_texture(0, texture);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // miniquad always uses row alignment of 1

            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                internal_format as i32,
                params.width as i32,
                params.height as i32,
                0,
                format,
                pixel_type,
                match bytes {
                    Some(bytes) => bytes.as_ptr() as *const _,
                    Option::None => std::ptr::null(),
                },
            );

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, params.wrap as i32);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, params.wrap as i32);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, params.filter as i32);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, params.filter as i32);

            #[cfg(not(target_arch = "wasm32"))]
            match params.format {
                // on non-WASM alpha value is stored in red channel
                // swizzle red -> alpha, zero red
                TextureFormat::Alpha if !alpha_texture => {
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_RED as _);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_ZERO as _);
                }
                // on non-WASM luminance is stored in red channel, alpha is stored in green channel
                // keep red, swizzle green -> alpha, zero green
                TextureFormat::LuminanceAlpha if !alpha_texture => {
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_GREEN as _);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_ZERO as _);
                }
                _ => {}
            }
        }
        ctx.opengl().cache.restore_texture_binding(0);

        Texture {
            texture,
            width: params.width,
            height: params.height,
            format: params.format,
            wrap: params.wrap,
            filter: params.filter,
            mipmap_filter: params.mipmap_filter,
        }
    }

    /// Upload texture to GPU with given TextureParams
    pub fn from_data_and_format(ctx: &mut Context, bytes: &[u8], params: TextureParams) -> Texture {
        Self::new(ctx, TextureAccess::Static, Some(bytes), params)
    }

    /// Upload RGBA8 texture to GPU
    pub fn from_rgba8(ctx: &mut Context, width: u16, height: u16, bytes: &[u8]) -> Texture {
        assert_eq!(width as usize * height as usize * 4, bytes.len());

        Self::from_data_and_format(
            ctx,
            bytes,
            TextureParams {
                width: width as _,
                height: height as _,
                format: TextureFormat::RGBA8,
                wrap: TextureWrap::Clamp,
                filter: FilterMode::Linear,
                mipmap_filter: MipmapFilter::None,
            },
        )
    }

    pub fn set_filter(&self, ctx: &mut Context, filter: FilterMode) {
        match ctx.backend() {
            GraphicsBackend::OpenGL(_) => {
                ctx.opengl().cache.store_texture_binding(0);
                ctx.opengl().cache.bind_texture(0, self.texture);
                unsafe {
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter as i32);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter as i32);
                }
                ctx.opengl().cache.restore_texture_binding(0);
            }
            #[cfg(feature = "vulkan")]
            GraphicsBackend::Vulkan(_) => ctx.vulkan().set_texture_filter(self.texture as usize, filter),
        }
    }

    pub fn set_wrap(&self, ctx: &mut Context, wrap: TextureWrap) {
        match ctx.backend() {
            GraphicsBackend::OpenGL(_) => {
                ctx.opengl().cache.store_texture_binding(0);
                ctx.opengl().cache.bind_texture(0, self.texture);
                unsafe {
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap as i32);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap as i32);
                }
                ctx.opengl().cache.restore_texture_binding(0);
            }
            #[cfg(feature = "vulkan")]
            GraphicsBackend::Vulkan(_) => ctx.vulkan().set_texture_wrap(self.texture as usize, wrap),
        }
    }

    pub fn resize(&mut self, ctx: &mut Context, width: u32, height: u32, bytes: Option<&[u8]>) {
        match ctx.backend() {
            GraphicsBackend::OpenGL(_) => {
                self.resize_opengl(ctx, width, height, bytes);
            }
            #[cfg(feature = "vulkan")]
            GraphicsBackend::Vulkan(_) => {
                let format = self.format;
                ctx.vulkan()
                    .resize_texture(self.texture as usize, width, height, format, bytes);
                self.width = width;
                self.height = height;
            }
        }
    }

    fn resize_opengl(
        &mut self,
        ctx: &mut Context,
        width: u32,
        height: u32,
        bytes: Option<&[u8]>,
    ) {
        let alpha_texture = ctx.features().alpha_texture;
        ctx.opengl().cache.store_texture_binding(0);
        ctx.opengl().cache.bind_texture(0, self.texture);

        let (internal_format, format, pixel_type) =
            self.format.into_gl_params(alpha_texture);

        self.width = width;
        self.height = height;

        unsafe {
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // miniquad always uses row alignment of 1

            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                internal_format as i32,
                self.width as i32,
                self.height as i32,
                0,
                format,
                pixel_type,
                match bytes {
                    Some(bytes) => bytes.as_ptr() as *const _,
                    Option::None => std::ptr::null(),
                },
            );
        }

        ctx.opengl().cache.restore_texture_binding(0);
    }

    /// Update whole texture content
    /// bytes should be width * height * 4 size - non rgba8 textures are not supported yet anyway
    pub fn update(&self, ctx: &mut Context, bytes: &[u8]) {
        assert_eq!(self.size(self.width, self.height), bytes.len());

        self.update_texture_part(ctx, 0, 0, self.width as _, self.height as _, bytes);
    }

    pub fn update_texture_part(
        &self,
        ctx: &mut Context,
        x_offset: i32,
        y_offset: i32,
        width: i32,
        height: i32,
        bytes: &[u8],
    ) {
        assert_eq!(self.size(width as _, height as _), bytes.len());
        assert!(x_offset + width <= self.width as _);
        assert!(y_offset + height <= self.height as _);

        match ctx.backend() {
            GraphicsBackend::OpenGL(_) => {
                let alpha_texture = ctx.features().alpha_texture;
                ctx.opengl().cache.store_texture_binding(0);
                ctx.opengl().cache.bind_texture(0, self.texture);

                let (_, format, pixel_type) = self.format.into_gl_params(alpha_texture);

                unsafe {
                    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // miniquad always uses row alignment of 1

                    glTexSubImage2D(
                        GL_TEXTURE_2D,
                        0,
                        x_offset as _,
                        y_offset as _,
                        width as _,
                        height as _,
                        format,
                        pixel_type,
                        bytes.as_ptr() as *const _,
                    );
                }

                ctx.opengl().cache.restore_texture_binding(0);
            }
            #[cfg(feature = "vulkan")]
            GraphicsBackend::Vulkan(_) => ctx.vulkan().update_texture_part(
                self.texture as usize,
                x_offset,
                y_offset,
                width,
                height,
                bytes,
                self.format,
            ),
        }
    }

    /// Fill the mip chain of a texture created with a `MipmapFilter`.
    pub fn generate_mipmap(&self, ctx: &mut Context) {
        match ctx.backend() {
            GraphicsBackend::OpenGL(_) => unsafe {
                glBindTexture(GL_TEXTURE_2D, self.texture);
                glGenerateMipmap(GL_TEXTURE_2D);
            },
            #[cfg(feature = "vulkan")]
            GraphicsBackend::Vulkan(_) => {
                ctx.vulkan().generate_mipmap(self.texture as usize)
            }
        }
    }

    /// Largest edge length a texture may have.
    pub fn max_size(ctx: &mut Context) -> u32 {
        match ctx.backend() {
            GraphicsBackend::OpenGL(_) => unsafe {
                let mut size = 0;
                glGetIntegerv(GL_MAX_TEXTURE_SIZE, &mut size);
                size as u32
            },
            #[cfg(feature = "vulkan")]
            GraphicsBackend::Vulkan(_) => ctx.vulkan().max_texture_size,
        }
    }

    /// Read texture data into CPU memory
    pub fn read_pixels(&self, ctx: &mut Context, bytes: &mut [u8]) {
        match ctx.backend() {
            GraphicsBackend::OpenGL(_) => {
                if self.format == TextureFormat::Alpha
                    || self.format == TextureFormat::LuminanceAlpha
                {
                    unimplemented!(
                        "read_pixels is not implement for Alpha and LuminanceAlpha textures"
                    );
                }
                let (_, format, pixel_type) = self.format.into_gl_params(false);

                let mut fbo = 0;
                unsafe {
                    let mut binded_fbo: i32 = 0;
                    glGetIntegerv(gl::GL_DRAW_FRAMEBUFFER_BINDING, &mut binded_fbo);
                    glGenFramebuffers(1, &mut fbo);
                    glBindFramebuffer(gl::GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(
                        gl::GL_FRAMEBUFFER,
                        gl::GL_COLOR_ATTACHMENT0,
                        gl::GL_TEXTURE_2D,
                        self.texture,
                        0,
                    );

                    glReadPixels(
                        0,
                        0,
                        self.width as _,
                        self.height as _,
                        format,
                        pixel_type,
                        bytes.as_mut_ptr() as _,
                    );

                    glBindFramebuffer(gl::GL_FRAMEBUFFER, binded_fbo as _);
                    glDeleteFramebuffers(1, &mut fbo);
                }
            }
            #[cfg(feature = "vulkan")]
            GraphicsBackend::Vulkan(_) => ctx.vulkan().read_texture_pixels(
                self.texture as usize,
                self.width,
                self.height,
                self.format,
                bytes,
            ),
        }
    }

    #[inline]
    fn size(&self, width: u32, height: u32) -> usize {
        self.format.size(width, height) as usize
    }
}