
from types import ModuleType
import numpy as np
import OpenGL
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
from qimview.utils.qt_imports   import QtGui, QOpenGLTexture
from qimview.utils.viewer_image import ImageFormat, ViewerImage


class GLTexture:
    """ Deal with OpenGL textures """
    def __init__(self, _gl: QtGui.QOpenGLFunctions | ModuleType | None):
        self._gl = _gl if _gl else gl
        self.texture : QOpenGLTexture | None = None
        self.textureY  = None
        self.textureU  = None
        self.textureV  = None
        self.textureID = None
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
        self.tex_width  : int = 0
        self.tex_height : int = 0

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

    def set_default_parameters(self, texture):
        """ Default texture parameters using OpenGL functions """
        self._gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
        self._gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        # self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 10)
        # self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        # self._gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST_MIPMAP_NEAREST)
        # self._gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

    def new_texture(self, texture):
        if self.texture is not None: gl.glDeleteTextures(np.array([self.texture]))
        return gl.glGenTextures(1)

    def create_texture_gl(self, image: ViewerImage):
        """ Create an OpenGL texture from a ViewerImage using OpenGL functions """
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
            w, h = width, height
            w2, h2 = int(w/2), int(h/2)
            if (self.tex_width,self.tex_height) != (width,height) or self.textureY is None or \
                  self.textureU is None or self.textureV is None:
                self.textureY = self.new_texture(self.textureY)
                self.set_default_parameters(self.textureY)
                self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, w, h, 0, LUM, gl_type, image.data)
                self.tex_width, self.tex_height = width, height
                # U
                self.textureU = self.new_texture(self.textureU)
                self.set_default_parameters(self.textureU)
                self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, w2, h2, 0, LUM, gl_type, image.u)
                # V
                self.textureV = self.new_texture(self.textureY)
                self.set_default_parameters(self.textureV)
                self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, w2, h2, 0,LUM, gl_type, image.v)
            else:
                self._gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureY)
                self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h, LUM, gl_type, image.data)
                self._gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureU)
                self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w2, h2, LUM, gl_type, image.u)
                self._gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureV)
                self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w2, h2, LUM, gl_type, image.v)
        else:
            # Texture pixel format
            pix_fmt = self._channels2format[image.channels]

            if (self.tex_width,self.tex_height) != (width,height):
                try:
                    if self.textureID is not None: gl.glDeleteTextures(np.array([self.textureID]))
                    self.textureID = gl.glGenTextures(1)
                    self.set_default_parameters(self.textureID)
                    # _gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureID)
                    format = self._internal_format(image)
                    self._gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, width, height, 0, pix_fmt, gl_type, image.data)
                    self.tex_width, self.tex_height = width, height
                except Exception as e:
                    print(f"setTexture failed shape={image.data.shape}: {e}")
                    return False
            else:
                try:
                    self._gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, width, height, pix_fmt, gl_type, image.data)
                    # _gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
                except Exception as e:
                    print(f"setTexture glTexSubImage2D() failed shape={image.data.shape}: {e}")
                    return False
