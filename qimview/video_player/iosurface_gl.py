"""
macOS-only: zero-copy VideoToolbox → OpenGL texture interop via IOSurface.

VideoToolbox decoded frames (AV_PIX_FMT_VIDEOTOOLBOX) are NV12-encoded
CVPixelBuffers backed by IOSurface.  We can bind each plane directly as a
GL_TEXTURE_RECTANGLE texture using CGLTexImageIOSurface2D, avoiding any
CPU-side copy or colour-conversion.

Plane 0: Y  luma       — 8-bit  GL_LUMINANCE,       width   × height
Plane 1: CbCr chroma   — 8-bit  GL_LUMINANCE_ALPHA,  width/2 × height/2
"""

import ctypes
import sys

assert sys.platform == 'darwin', "iosurface_gl is macOS-only"

from OpenGL.GL import (
    glGenTextures, glBindTexture, glDeleteTextures,
    glTexParameteri, glGetError,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE,
    GL_NO_ERROR,
)

# GL_TEXTURE_RECTANGLE is 0x84F5 (ARB extension, core since GL 3.1)
GL_TEXTURE_RECTANGLE = 0x84F5

# Pixel format constants not always exported by PyOpenGL
_GL_UNSIGNED_BYTE   = 0x1401
_GL_UNSIGNED_SHORT  = 0x1403
_GL_LUMINANCE       = 0x1909
_GL_LUMINANCE_ALPHA = 0x190A
_GL_LUMINANCE_ALPHA_INTEGER_EXT = 0x8DCD  # GL_EXT_texture_integer
_GL_LUMINANCE_INTEGER_EXT = 0x8D90  # GL_EXT_texture_integer
_GL_RG = 0x8227  # GL_RG, for 10-bit P010 packed UV plane (GL_EXT_texture_integer)

_cgl = ctypes.CDLL('/System/Library/Frameworks/OpenGL.framework/OpenGL')
_cgl.CGLGetCurrentContext.restype  = ctypes.c_void_p
_cgl.CGLTexImageIOSurface2D.restype = ctypes.c_int
_cgl.CGLTexImageIOSurface2D.argtypes = [
    ctypes.c_void_p,  # CGLContextObj   ctx
    ctypes.c_uint,    # GLenum          target
    ctypes.c_uint,    # GLenum          internalformat
    ctypes.c_int,     # GLsizei         width
    ctypes.c_int,     # GLsizei         height
    ctypes.c_uint,    # GLenum          format
    ctypes.c_uint,    # GLenum          type
    ctypes.c_void_p,  # IOSurfaceRef    ioSurface
    ctypes.c_uint,    # GLuint          plane
]


def _set_rect_texture_params():
    """Set nearest-neighbour wrap/filter for current GL_TEXTURE_RECTANGLE binding."""
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)


def bind_iosurface_nv12(iosurface_handle: int, y_width: int, y_height: int, precision: int,
                        y_tex: int = 0, cbcr_tex: int = 0) -> tuple:
    """
    Bind the two NV12 planes of an IOSurface directly as GL_TEXTURE_RECTANGLE
    textures via CGLTexImageIOSurface2D (zero CPU copy).

    Parameters
    ----------
    iosurface_handle : integer IOSurface pointer returned by Frame.getIOSurface()
    y_width, y_height: luma plane dimensions
    y_tex, cbcr_tex  : existing GL texture IDs to reuse (0 = allocate new ones)

    Returns
    -------
    (y_tex_id, cbcr_tex_id) — GL texture IDs for luma and chroma planes
    """
    cgl_ctx = _cgl.CGLGetCurrentContext()
    if cgl_ctx is None:
        raise RuntimeError("No current CGL context — call bind_iosurface inside paintGL()")

    surface = ctypes.c_void_p(iosurface_handle)

    if y_tex == 0 or cbcr_tex == 0:
        ids = glGenTextures(2)
        y_tex, cbcr_tex = int(ids[0]), int(ids[1])

    # --- Plane 0: Y luma (8-bit single channel) ---
    glBindTexture(GL_TEXTURE_RECTANGLE, y_tex)
    err = _cgl.CGLTexImageIOSurface2D(
        cgl_ctx,
        GL_TEXTURE_RECTANGLE,
        _GL_LUMINANCE,          # internalformat
        y_width, y_height,
        _GL_LUMINANCE,          # format
        _GL_UNSIGNED_BYTE if precision <= 8 else _GL_UNSIGNED_SHORT,      # type
        surface,
        0,                      # plane 0
    )
    if err:
        raise RuntimeError(f"CGLTexImageIOSurface2D (Y plane) returned error {err}")
    _set_rect_texture_params()

    # Drain any pending GL error left by the Y-plane CGL call (e.g. when the IOSurface
    # is 10-bit P010 but we described it as 8-bit to CGLTexImageIOSurface2D).
    # Without this, PyOpenGL raises a RuntimeError on the next glBindTexture call.
    gl_err = glGetError()
    if gl_err != GL_NO_ERROR:
        print(f"bind_iosurface_nv12: GL error {gl_err:#x} after Y plane CGL bind (precision={precision})")

    # --- Plane 1: CbCr chroma (half resolution) ---
    # 8-bit NV12: GL_UNSIGNED_BYTE — 2 bytes per chroma pair (Cb, Cr)
    # 10-bit P010: GL_UNSIGNED_SHORT — 2×16-bit per chroma pair, values in top 10 bits
    glBindTexture(GL_TEXTURE_RECTANGLE, cbcr_tex)
    err = _cgl.CGLTexImageIOSurface2D(
        cgl_ctx,
        GL_TEXTURE_RECTANGLE,
        _GL_RG,    # internalformat: Cb in .r, Cr in .a
        y_width // 2, y_height // 2,
        _GL_RG, # if precision <= 8 else _GL_LUMINANCE_ALPHA_INTEGER_EXT,    # format
        _GL_UNSIGNED_BYTE if precision <= 8 else _GL_UNSIGNED_SHORT,
        surface,
        1,                      # plane 1
    )
    if err:
        raise RuntimeError(f"CGLTexImageIOSurface2D (CbCr plane) returned error {err}")
    _set_rect_texture_params()

    glBindTexture(GL_TEXTURE_RECTANGLE, 0)
    return y_tex, cbcr_tex
