
from types import ModuleType
import numpy as np
import OpenGL
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
from qimview.utils.qt_imports   import QtGui, QOpenGLTexture
from qimview.utils.viewer_image import ImageFormat, ViewerImage
import time

class GLTexture:
    """ Deal with OpenGL textures """
    def __init__(self, _gl: QtGui.QOpenGLFunctions | ModuleType | None):
        self._gl = _gl if _gl else gl
        self.texture : QOpenGLTexture | None = None
        self._current_texture_idx : int = 0
        self._textureY  = [None, None]
        self._textureU  = [None, None]
        self._textureV  = [None, None]
        self.textureUV = None
        self.textureID = None
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
        self._channels2format = {
                ImageFormat.CH_RGB : gl.GL_RGB,
                ImageFormat.CH_BGR : gl.GL_BGR,
                ImageFormat.CH_Y   : gl.GL_RED, # not sure about this one
                ImageFormat.CH_RGGB: gl.GL_RGBA, # we save 4 component data
                ImageFormat.CH_GRBG: gl.GL_RGBA,
                ImageFormat.CH_GBRG: gl.GL_RGBA,
                ImageFormat.CH_BGGR: gl.GL_RGBA
            }
        self.width  : int = 0
        self.height : int = 0
        # Print log on average time for texSubImage
        self.timing : dict[str,float] = {}
        self.counter : dict[str,int] = {}

    @property
    def textureY(self):
        return self._textureY[self._current_texture_idx]

    @property
    def textureU(self):
        return self._textureU[self._current_texture_idx]

    @property
    def textureV(self):
        return self._textureV[self._current_texture_idx]

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
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        # self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 10)
        # self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        # self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST_MIPMAP_NEAREST)
        # self._gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        # Antialiasing?
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP, gl.GL_TRUE)
    
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
        return id
    
    def texSubImage(self, textureY, w, h, LUM, gl_type, data, name='unamed'):
        start_time = time.perf_counter()
        self._gl.glBindTexture(  gl.GL_TEXTURE_2D, textureY)
        self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h, LUM, gl_type, data)
        if name not in self.timing:
            self.timing[name], self.counter[name] = 0, 0
        self.timing[name] += time.perf_counter() - start_time
        self.counter[name] += 1
        if self.counter[name] == 30:
            print(f" {name}: textSubImage() av {self.counter[name]} = {(self.timing[name]/self.counter[name])*1000:0.1f} ms")
            self.timing[name], self.counter[name] = 0, 0

    def create_texture_gl(self, image: ViewerImage):
        """ Create an OpenGL texture from a ViewerImage using OpenGL functions """
        # Copy to temporary textures
        texture_idx = (self._current_texture_idx+1)%2
        height, width = image.data.shape[:2]
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
            w2, h2 = int(w/2), int(h/2)
            if (self.width,self.height) != (width,height) or self._textureY[texture_idx] is None:
                self._textureY[texture_idx] = self.new_texture()
                self.bind(self._textureY[texture_idx])
                self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, w, h, 0, LUM, gl_type, image.data)
                self.width, self.height = width, height
                if self.interlaced_uv:
                    # UV
                    self.textureUV = self.new_texture()
                    self.bind(self.textureUV)
                    self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format_uv, w2, h2, 0, RG, gl_type, image.uv)
                else:
                    # U
                    self._textureU[texture_idx] = self.new_texture()
                    self.bind(self._textureU[texture_idx])
                    self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, w2, h2, 0, LUM, gl_type, image.u)
                    # V
                    self._textureV[texture_idx] = self.new_texture()
                    self.bind(self._textureV[texture_idx])
                    self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, w2, h2, 0,LUM, gl_type, image.v)
            else:
                self.texSubImage(self._textureY[texture_idx], w,h,LUM, gl_type, image.data, 'Y')
                # self._gl.glBindTexture(  gl.GL_TEXTURE_2D, self.textureY)
                # # Split in two
                # self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h2, LUM, gl_type, image.data)
                # self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, h2, w, h2, LUM, gl_type, image.data[h2:,:])
                if self.interlaced_uv:
                    assert self.textureUV is not None, "textureUV should not be None"
                    self._gl.glBindTexture(  gl.GL_TEXTURE_2D, self.textureUV)
                    self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w2, h2, RG, gl_type, image.uv)
                else:
                    assert self._textureU[texture_idx] is not None and self._textureV[texture_idx] is not None, \
                            "textureV and textureV should not be None"
                    self.texSubImage(self._textureU[texture_idx], w2, h2, LUM, gl_type, image.u,'U')
                    self.texSubImage(self._textureV[texture_idx], w2, h2, LUM, gl_type, image.v,'V')
            self._current_texture_idx = texture_idx
        else:
            # Texture pixel format
            pix_fmt = self._channels2format[image.channels]

            if (self.width,self.height) != (width,height):
                try:
                    if self.textureID is not None: gl.glDeleteTextures(np.array([self.textureID]))
                    self.textureID = gl.glGenTextures(1)
                    self.bind(self.textureID)
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
