
from types import ModuleType
import numpy as np
import OpenGL
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
from qimview.utils.qt_imports   import QtGui, QOpenGLTexture, QImage
from qimview.utils.viewer_image import ImageFormat, ViewerImage
import time
import ctypes

from qimview.video_player.video_player_config import VideoConfig

use_opengl_cpp = False
if use_opengl_cpp:
    # WIP: this part needs to be moved to the file header and improved
    import os
    os.add_dll_directory(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "../opengl_cpp/glew-2.2.0/glew-2.2.0/build/cmake/build/bin/Release"))
    import opengl_cpp


# Use PBO: https://www.songho.ca/opengl/gl_pbo.html#unpack
# 1. generate buffer with glGenBuffers()
# 2. bind with glBindBuffer()
# 3. copy the data with glBufferData() with GL_STREAM_DRAW
# or copy part with glBufferSubData
# 4. glDeleteBuffers() to delete it at the end
# example:
#
#
#

class GLTexture:
    """ Deal with OpenGL textures """
    def __init__(self, _gl: QtGui.QOpenGLFunctions | ModuleType | None):
        self._gl = _gl if _gl else gl
        self.texture : QOpenGLTexture | None = None
        self._use_buffers: bool = True
        self._buf_idx : int = 0
        self._nbuf = 2
        self._buffers : dict[str, np.ndarray | None] = { 'Y' : None, 'U': None, 'V': None, 'UV': None, 'RGB': None }
        self._texture : dict[str, int | None]        = { 'Y' : None, 'U': None, 'V': None, 'UV': None, 'RGB': None }
        self.interlaced_uv = False
        self._gl_types = {
            'int8'   : gl.GL_BYTE,
            'uint8'  : gl.GL_UNSIGNED_BYTE,
            'int16'  : gl.GL_SHORT,
            'uint16' : gl.GL_UNSIGNED_SHORT,
            'int32'  : gl.GL_INT,
            'uint32' : gl.GL_UNSIGNED_INT,
            'float32': gl.GL_FLOAT,
            'float64': gl.GL_DOUBLE
        }
        self._qt_gl_types = {
            'int8'   : QOpenGLTexture.PixelType.Int8,
            'uint8'  : QOpenGLTexture.PixelType.UInt8,
            'int16'  : QOpenGLTexture.PixelType.Int16,
            'uint16' : QOpenGLTexture.PixelType.UInt16,
            'int32'  : QOpenGLTexture.PixelType.Int32,
            'uint32' : QOpenGLTexture.PixelType.UInt32,
            'float32': QOpenGLTexture.PixelType.Float32,
        }
        self._qimage_types = {
            'int8'   : QImage.Format.Format_Grayscale8,
            'uint8'  : QImage.Format.Format_Grayscale8,
            'int16'  : QImage.Format.Format_Grayscale16,
            'uint16' : QImage.Format.Format_Grayscale16,
        }
        self._channels2format = {
                ImageFormat.CH_RGB : gl.GL_RGB,
                ImageFormat.CH_BGR : gl.GL_BGR,
                ImageFormat.CH_Y   : gl.GL_RED, # not sure about this one
                ImageFormat.CH_RGGB: gl.GL_RGBA, # we save 4 component data
                ImageFormat.CH_GRBG: gl.GL_RGBA,
                ImageFormat.CH_GBRG: gl.GL_RGBA,
                ImageFormat.CH_BGGR: gl.GL_RGBA
            }
        self._qt_channels2format = {
                ImageFormat.CH_RGB : QOpenGLTexture.PixelFormat.RGB,
                ImageFormat.CH_BGR : QOpenGLTexture.PixelFormat.BGR,
                ImageFormat.CH_Y   : QOpenGLTexture.PixelFormat.Red, # not sure about this one
                ImageFormat.CH_RGGB: QOpenGLTexture.PixelFormat.RGBA, # we save 4 component data
                ImageFormat.CH_GRBG: QOpenGLTexture.PixelFormat.RGBA,
                ImageFormat.CH_GBRG: QOpenGLTexture.PixelFormat.RGBA,
                ImageFormat.CH_BGGR: QOpenGLTexture.PixelFormat.RGBA
            }
        self.width  : int = 0
        self.height : int = 0
        # Print log on average time for texSubImage
        self.timing : dict[str,float] = {}
        self.counter : dict[str,int] = {}
        self._log_timings : bool = False

        self._buffer_size = np.zeros(1, dtype=np.int32)

        self._buffersubdata_total_time = {}
        self._buffersubdata_count      = {}

        # Use glMapBuffer and the memcopy, this version is slower 
        self._use_glmap = False

        # Use own binding of some opengl functions, can be faster but required glew library
        self._use_opengl_cpp = use_opengl_cpp
        if self._use_opengl_cpp:
            self._glew_initialized = False

    @property
    def textureY(self):
        return self._texture['Y']

    @textureY.setter
    def textureY(self, value):
        self._texture['Y'] = value

    @property
    def textureU(self):
        return self._texture['U']

    @textureU.setter
    def textureU(self, value):
        self._texture['U'] = value

    @property
    def textureV(self):
        return self._texture['V']

    @textureV.setter
    def textureV(self, value):
        self._texture['V'] = value

    @property
    def textureUV(self):
        return self._texture['UV']

    @textureUV.setter
    def textureUV(self, value):
        self._texture['UV'] = value

    @property
    def textureRGB(self):
        return self._texture['RGB']

    @textureRGB.setter
    def textureRGB(self, value):
        self._texture['RGB'] = value

    def _internal_format(self, image):
        # Not sure what is the right parameter for internal format of 2D texture based
        # on the input data type uint8, uint16, ...
        # need to test with different DXR images
        internal_format = gl.GL_RGB
        if len(image.data.shape) == 3:
            match image.data.shape[2]:
                # 3 channels
                case 3:
                    match image.precision:
                        case 8:  internal_format = gl.GL_RGB
                        case 10: internal_format = gl.GL_RGB10
                        case 12: internal_format = gl.GL_RGB12
                        case _: print("Image precision not available for opengl texture")
                # 4 channels, probably Bayer image
                case 4:
                    match image.precision:
                        case 8:  internal_format = gl.GL_RGBA
                        case 12: internal_format = gl.GL_RGBA12
                        case 16: internal_format = gl.GL_RGBA16
                        case _:  print("Image precision not available for opengl texture")
                                # internal_format = gl.GL_RGBA32F
                case _:  print("Image number of channels not available for opengl texture")
        return internal_format

    def create_texture_qt(self, image):
        """ Create a texture using QOpenGLTexture """
        self.texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        self.texture.create()
        # new school
        self.texture.bind()
        # def setData (sourceFormat, sourceType, data[, options=None])
        self.texture.setData(image)
        self.texture.setMinMagFilters(QOpenGLTexture.Filter.Linear, QOpenGLTexture.Filter.Linear)
        self.texture.setWrapMode(QOpenGLTexture.CoordinateDirection.DirectionS, QOpenGLTexture.WrapMode.Repeat)
        self.texture.setWrapMode(QOpenGLTexture.CoordinateDirection.DirectionT, QOpenGLTexture.WrapMode.Repeat)

    def set_default_parameters(self):
        """ Default texture parameters for active texture unit using OpenGL functions """
        # self._gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
        # Setting max level >0 slows down on non GPU graphics
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, VideoConfig.mipmap_max_level)
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        # self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 10)
        # self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        # self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST_MIPMAP_NEAREST)
        # self._gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        # Antialiasing?
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP,
                                 gl.GL_TRUE if VideoConfig.mipmap_max_level>0 else gl.GL_FALSE)
    
    def bind(self, texture):
        self._gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        # Need to call this function or it does not work
        self.set_default_parameters()

    def new_texture(self):
        # if texture is not None: gl.glDeleteTextures(np.array([texture]))
        # Issue on macos, so try to sort out the 2 versions with exception catch
        try:
            id = gl.glGenTextures(1)
        except Exception as e:
            texture = np.array([0], dtype=np.uint32)
            self._gl.glGenTextures(1, texture)
            id = texture[0]
        if self._use_opengl_cpp and not self._glew_initialized:
            opengl_cpp.InitGlew()
            self._glew_initialized = True
        return id

    def free_texture(self, texture: int | None):
        if texture is not None:
            self._gl.glDeleteTextures(1, np.array([texture]))
        
    def free_all_textures(self):
        for k in self._texture:
            if self._texture[k] is not None:
                self.free_texture(self._texture[k])
                self._texture[k] = None

    def new_buffers(self, n: int) -> np.ndarray:
        """
          Returns n opengl buffers
        """
        return gl.glGenBuffers(n)

    def check_buffers(self):
        """ Create all the PBO buffers if they don't already exist
        """
        if self._use_buffers and self._buffers['Y'] is None:
            self._buffers['Y'] = self.new_buffers(self._nbuf)
            if self.interlaced_uv:
                self._buffers['UV'] = self.new_buffers(self._nbuf)
            else:
                self._buffers['U']  = self.new_buffers(self._nbuf)
                self._buffers['V']  = self.new_buffers(self._nbuf)

    def free_buffers(self):
        if self._use_buffers:
            for name in self._buffers:
                if self._buffers[name] is not None:
                    gl.glDeleteBuffers(2, self._buffers[name])
                    self._buffers[name] = None
                self._buf_idx = 0

    def reset(self):
        self.free_buffers()
        # Issue?
        self.free_all_textures()

    def texSubImage(self, tex, w, h_start, h_end, LUM, gl_type, data, name='unamed'):
        if self._log_timings:
            start_time = time.perf_counter()
        self._gl.glBindTexture(  gl.GL_TEXTURE_2D, tex)
        self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, h_start, w, h_end-h_start, LUM, gl_type, data[h_start:h_end,:])
        if self._log_timings:
            if name not in self.timing:
                self.timing[name], self.counter[name] = 0, 0
            self.timing[name] += time.perf_counter() - start_time
            self.counter[name] += 1
            if self.counter[name] == 30:
                print(f" {name}: textSubImage() av {self.counter[name]} = {(self.timing[name]/self.counter[name])*1000:0.1f} ms")
                self.timing[name], self.counter[name] = 0, 0

    def bufferTexSubImage(self, buf, buf_idx, tex, w, h_start, h_end, LUM, gl_type, data, name='unamed'):
        # print(f"bufferTexSubImage {name}")
        gl = self._gl
        if self._log_timings:
            start_time = time.perf_counter()

        buf0 = buf_idx     % self._nbuf
        buf1 = (buf_idx+1) % self._nbuf

        # Copy image Y to buffer Y
        # print(f"{buf=}")
        target = gl.GL_PIXEL_UNPACK_BUFFER

        if self._use_opengl_cpp:
            if data.dtype.itemsize == 1:
                glBufferDataFunc    = opengl_cpp.glBufferData_u8
                glBufferSubDataFunc = opengl_cpp.glBufferSubData_u8
            elif data.dtype.itemsize == 2:
                glBufferDataFunc    = opengl_cpp.glBufferData_u16
                glBufferSubDataFunc = opengl_cpp.glBufferSubData_u16
            else:
                assert False, "Data type is not compatible"
        else:
            glBufferDataFunc    = gl.glBufferData
            glBufferSubDataFunc = gl.glBufferSubData

        d = data[h_start:h_end,:]
        gl.glBindBuffer(target, buf[buf0])
        gl.glGetBufferParameteriv(target,gl.GL_BUFFER_SIZE, self._buffer_size)
        if self._buffer_size[0] < d.nbytes:
            # t = time.perf_counter()
            glBufferDataFunc(target, d.nbytes, d, gl.GL_STREAM_DRAW)
            # print(f" glBufferData() {name} took {(time.perf_counter()-t)*1000:0.2f} ms")

        gl.glBindTexture(  gl.GL_TEXTURE_2D, tex)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, h_start, w, h_end-h_start, LUM, gl_type, None)

        gl.glBindBuffer(target, buf[buf1])
        gl.glGetBufferParameteriv(target,gl.GL_BUFFER_SIZE, self._buffer_size)
        if self._buffer_size[0] < d.nbytes:
            # print(f"Different buffer 1 sizes {buffer_size[0]} {d.nbytes}")
            # t = time.perf_counter()
            # opengl_cpp.InitGlew()
            glBufferDataFunc(target, d.nbytes, d, gl.GL_STREAM_DRAW)
            # print(f" glBufferData() {name} took {(time.perf_counter()-t)*1000:0.2f} ms")
            self._buffersubdata_total_time[name] = 0
            self._buffersubdata_count[name]      = 0
            # gl.glBufferData(target, d.nbytes, d, gl.GL_STREAM_DRAW)
        else:
            t = time.perf_counter()
            if self._use_glmap:
                vp = gl.glMapBuffer(target, gl.GL_WRITE_ONLY)
                vp_array = ctypes.cast(vp, ctypes.POINTER(ctypes.c_void_p) )
                ctypes.memmove(vp_array, d.ctypes.data_as(ctypes.c_void_p),  d.nbytes)
                gl.glUnmapBuffer(target)
            else:
                glBufferSubDataFunc(target, 0, d.nbytes, d)
            self._buffersubdata_total_time[name] += time.perf_counter()-t
            self._buffersubdata_count[name]      += 1
            if self._log_timings:
                if self._buffersubdata_count[name] % 30 == 0:
                    print(f" glBufferSubData() {name} took {(self._buffersubdata_total_time[name]/self._buffersubdata_count[name])*1000:0.2f} ms")

        # Release PBOs
        gl.glBindBuffer(target, 0)

        if self._log_timings:
            if name not in self.timing:
                self.timing[name], self.counter[name] = 0, 0
            self.timing[name] += time.perf_counter() - start_time
            self.counter[name] += 1
            if self.counter[name] == 30:
                print(f" {name}: bufferTexSubImage() av {self.counter[name]} = {(self.timing[name]/self.counter[name])*1000:0.1f} ms")
                self.timing[name], self.counter[name] = 0, 0

    def create_texture_qt_gl(self, image: ViewerImage):
        """ Creates a QOpenGLTexture from a ViewerImage

        Args:
            image (ViewerImage): _description_
        """
        height, width = image.data.shape[:2]
        qt_gl_type = self._qt_gl_types[image.data.dtype.name]
        if image.channels == ImageFormat.CH_YUV420:
            LUM = QOpenGLTexture.PixelFormat.Luminance
            match image.data.dtype:
                case np.uint8:  format = QOpenGLTexture.PixelFormat.Red_Integer
                case np.uint16: format = QOpenGLTexture.PixelFormat.Red_Integer
                case _:
                    print(f"create_texture_gl(): Image format not supported {image.data.dtype}")
                    format = QOpenGLTexture.PixelFormat.Red
            self.interlaced_uv = image.uv is not None
            if self.interlaced_uv:
                RG = gl.GL_RG
                match image.data.dtype:
                    case np.uint8:  format_uv = QOpenGLTexture.PixelFormat.RG_Integer
                    case np.uint16: format_uv = QOpenGLTexture.PixelFormat.RG_Integer
                    case _:
                        print(f"create_texture_gl(): Image format not supported {image.data.dtype}")
                        format_uv = QOpenGLTexture.PixelFormat.RG
            w, h = width, height
            w2, h2 = int(w/2), int(h/2)
            if (self.width,self.height) != (width,height) or self.textureY is None:
                self._texture['Y'] = self.new_texture()
                # Create QImage
                image = QtGui.QImage(image.data, w, h, )
                self.bind(self._texture['Y'])
                self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, w, h, 0, LUM, gl_type, image.data)
                self.width, self.height = width, height
                if self.interlaced_uv:
                    # UV
                    self._texture['UV'] = self.new_texture()
                    self.bind(self.textureUV)
                    self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format_uv, w2, h2, 0, RG, gl_type, image.uv)
                else:
                    # U
                    self._texture['U'] = self.new_texture()
                    self.bind(self._texture['U'])
                    self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, w2, h2, 0, LUM, gl_type, image.u)
                    # V
                    self._texture['V'] = self.new_texture()
                    self.bind(self._texture['V'])
                    self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, w2, h2, 0,LUM, gl_type, image.v)

    def resize_event(self):
        self.free_buffers()

    def create_texture_gl(self, image: ViewerImage, h_min: int = 0, h_max: int = -1, use_PBO:bool = False):
        """ Create an OpenGL texture from a ViewerImage using OpenGL functions """
        # Copy to temporary textures
        height, width = image.data.shape[:2]
        if h_max == -1:
            h_max = height
        gl_type = self._gl_types[image.data.dtype.name]
        if image.channels == ImageFormat.CH_YUV420:
            # TODO: check if this condition is sufficient
            LUM = gl.GL_LUMINANCE
            match image.data.dtype:
                case np.uint8:  format = gl.GL_R8
                # TODO: check if SHORT or UNSIGNED_SHORT
                case np.uint16: format = gl.GL_R16
                case _:
                    print(f"create_texture_gl(): Image format not supported {image.data.dtype}")
                    format = gl.GL_R8
            self.interlaced_uv = image.uv is not None
            if self.interlaced_uv:
                RG = gl.GL_RG
                match image.data.dtype:
                    case np.uint8:  format_uv = gl.GL_RG8
                    # TODO: check if SHORT or UNSIGNED_SHORT
                    case np.uint16: format_uv = gl.GL_RG16
                    case _:
                        print(f"create_texture_gl(): Image format not supported {image.data.dtype}")
                        format_uv = gl.GL_R8
            
            w, h = width, height
            w2, h2 = w>>1, h>>1
            h2_min = h_min>>1
            h2_max = h_max>>1
            if use_PBO:
                self.check_buffers()

            if (self.width,self.height) != (width,height) or self._texture['Y'] is None:
                self._texture['Y'] = self.new_texture()
                self.bind(self._texture['Y'])
                self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, w, h, 0, LUM, gl_type, image.data)
                self.width, self.height = width, height
                if self.interlaced_uv:
                    # UV
                    self._texture['UV'] = self.new_texture()
                    self.bind(self._texture['UV'])
                    self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format_uv, w2, h2, 0, RG, gl_type, image.uv)
                else:
                    # U
                    self._texture['U'] = self.new_texture()
                    self.bind(self._texture['U'])
                    self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, w2, h2, 0, LUM, gl_type, image.u)
                    # V
                    self._texture['V'] = self.new_texture()
                    self.bind(self._texture['V'])
                    self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, w2, h2, 0,LUM, gl_type, image.v)
            else:
                if self._use_buffers and use_PBO:
                    self.bufferTexSubImage(self._buffers['Y'], self._buf_idx,  self._texture['Y'], 
                        w, h_min, h_max,LUM, gl_type, image.data, 'BY')
                else:
                    self.texSubImage(self._texture['Y'], w, h_min, h_max ,LUM, gl_type, image.data, 'Y')
                # self._gl.glBindTexture(  gl.GL_TEXTURE_2D, self.textureY)
                # # Split in two
                # self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h2, LUM, gl_type, image.data)
                # self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, h2, w, h2, LUM, gl_type, image.data[h2:,:])
                if self.interlaced_uv:
                    assert self.textureUV is not None, "textureUV should not be None"
                    if self._use_buffers and use_PBO:
                        self.bufferTexSubImage(self._buffers['UV'], self._buf_idx,  self._texture['UV'], 
                            w2, h2_min, h2_max,RG, gl_type, image.uv, 'BUV')
                    else:
                        self._gl.glBindTexture(  gl.GL_TEXTURE_2D, self.textureUV)
                        self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w2, h2, RG, gl_type, image.uv)
                else:
                    assert self._texture['U'] is not None and self._texture['V'] is not None, \
                            "textureV and textureV should not be None"
                    if self._use_buffers and use_PBO:
                        self.bufferTexSubImage(self._buffers['U'], self._buf_idx,  self._texture['U'], 
                            w2, h2_min, h2_max,LUM, gl_type, image.u, 'BU')
                        self.bufferTexSubImage(self._buffers['V'], self._buf_idx,  self._texture['V'], 
                            w2, h2_min, h2_max,LUM, gl_type, image.v, 'BV')
                    else:
                        self.texSubImage(self._texture['U'], w2, h2_min, h2_max, LUM, gl_type, image.u,'U')
                        self.texSubImage(self._texture['V'], w2, h2_min, h2_max, LUM, gl_type, image.v,'V')
                self._buf_idx += 1
        else:
            # Texture pixel format
            pix_fmt = self._channels2format[image.channels]

            if (self.width,self.height) != (width,height):
                try:
                    self.free_texture(self._texture['RGB'])
                    self._texture['RGB'] = gl.glGenTextures(1)
                    self.bind(self._texture['RGB'])
                    # _gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureID)
                    format = self._internal_format(image)
                    self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, width, height, 0, pix_fmt, gl_type, image.data)
                    self.width, self.height = width, height
                except Exception as e:
                    print(f"setTexture failed shape={image.data.shape}: {e}")
                    return False
            else:
                try:
                    self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, width, height, pix_fmt, gl_type, image.data)
                    # _gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
                except Exception as e:
                    print(f"setTexture glTexSubImage2D() failed shape={image.data.shape} {self.height, self.width}: {e}")
                    return False
