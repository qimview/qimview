#
#
# started from https://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
#
#

import OpenGL.GL as gl
from OpenGL.GL import shaders
import argparse
import sys
import numpy as np

from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLShader,
                              QOpenGLShaderProgram, QOpenGLTexture)

from qimview.utils.qt_imports  import QtWidgets, QtGui
from .image_viewer             import trace_method, get_time
from .gl_image_viewer_base     import GLImageViewerBase
from qimview.utils.viewer_image import ImageFormat


class GLImageViewerShaders(GLImageViewerBase):
    # vertex shader program
    vertexShader = """
        #version 330 core

        attribute vec3 vert;
        attribute vec2 uV;
        uniform mat4 mvMatrix;
        uniform mat4 pMatrix;
        out vec2 UV;

        void main() {
          gl_Position = pMatrix * mvMatrix * vec4(vert, 1.0);
          UV = uV;
        }
        """
    # fragment shader program
    fragmentShader_RGB = """
        #version 330 core
    
        in vec2 UV;
        uniform sampler2D backgroundTexture;
        uniform int channels; // channel representation
        uniform float white_level;
        uniform float black_level;
        uniform float g_r_coeff;
        uniform float g_b_coeff;
        uniform float max_value; // maximal value based on image precision
        uniform float max_type;  // maximal value based on image type (uint8, etc...)
        uniform float gamma;
        out vec3 colour;
    
        void main() {
          colour = texture(backgroundTexture, UV).rgb;

          // black level
          colour.rgb = colour.rgb/max_value*max_type;
          colour.rgb = max((colour.rgb-vec3(black_level).rgb),0);

          // white balance
          colour.r = colour.r*g_r_coeff;
          colour.b = colour.b*g_b_coeff;

          // rescale to white level as saturation level
          colour.rgb = colour.rgb/(white_level-black_level);
          
          // apply gamma
          colour.rgb = pow(colour.rgb, vec3(1.0/gamma).rgb);

        }
    """

    fragmentShader_RAW = """
        #version 330 core
        
        in vec2 UV;
        uniform sampler2D backgroundTexture;
        uniform int channels; // channel representation
        uniform float white_level;
        uniform float black_level;
        uniform float g_r_coeff;
        uniform float g_b_coeff;
        uniform float max_value; // maximal value based on image precision
        uniform float max_type;  // maximal value based on image type (uint8, etc...)
        uniform float gamma;
        out vec3 colour;

        void main() {

           const int CH_RGGB = 4; // phase 0, bayer 2
           const int CH_GRBG = 5; // phase 1, bayer 3 (Boilers)
           const int CH_GBRG = 6; // phase 2, bayer 0
           const int CH_BGGR = 7; // phase 3, bayer 1 (Coconuts)

          vec4 bayer = texture(backgroundTexture, UV);
          // transform bayer data to RGB
          int r,gr,gb,b;
          switch (channels) {
            case 4:   r = 0; gr = 1; gb = 2; b = 3;  break; // CH_RGGB = 4 phase 0, bayer 2
            case 5:   r = 1; gr = 0; gb = 3; b = 2;  break; // CH_GRBG = 5 phase 1, bayer 3 (Boilers)
            case 6:   r = 2; gr = 3; gb = 0; b = 1;  break; // CH_GBRG = 6 phase 2, bayer 0
            case 7:   r = 3; gr = 2; gb = 1; b = 0;  break; // CH_BGGR = 7 phase 3, bayer 1 (Coconuts)
            default:        r = 0; gr = 1; gb = 2; b = 3;  break; // this should not happen
          }

          // first retreive black point to get the coefficients right ...
          // 5% of dynamics?
          
          // bayer 2 rgb
          colour.r   = bayer[r];
          colour.g = (bayer[gr]+bayer[gb])/2.0;
          colour.b = bayer[b];

          // black level
          colour.rgb = colour.rgb/max_value*max_type;
          colour.rgb = max((colour.rgb-vec3(black_level).rgb),0);
          
          // white balance
          colour.r = colour.r*g_r_coeff;
          colour.b = colour.b*g_b_coeff;

          // rescale to white level as saturation level
          colour.rgb = colour.rgb/(white_level-black_level);
          
          // apply gamma
          colour.rgb = pow(colour.rgb, vec3(1.0/gamma).rgb);
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAutoFillBackground(False)
        self.textureID = None
        self.tex_width, self.tex_height = 0, 0
        self.opengl_debug = False
        self.pMatrix  = np.identity(4, dtype=np.float32)
        self.mvMatrix = np.identity(4, dtype=np.float32)
        self.program_RGB = None
        self.program_RAW = None
        self.program = None
        self.vertexBuffer = None

    def set_shaders(self):
        if self.program_RGB is None:
            vs = shaders.compileShader(self.vertexShader, gl.GL_VERTEX_SHADER)
            fs = shaders.compileShader(self.fragmentShader_RGB, gl.GL_FRAGMENT_SHADER)
            try:
                self.program_RGB = shaders.compileProgram(vs, fs, validate=False)
                print("\n***** self.program_RGB = {} *****\n".format(self.program_RGB))
            except Exception as e:
                print('failed RGB shaders.compileProgram() {}'.format(e))
            shaders.glDeleteShader(vs)
            shaders.glDeleteShader(fs)

        if self.program_RAW is None:
            vs = shaders.compileShader(self.vertexShader, gl.GL_VERTEX_SHADER)
            fs = shaders.compileShader(self.fragmentShader_RAW, gl.GL_FRAGMENT_SHADER)
            try:
                self.program_RAW = shaders.compileProgram(vs, fs, validate=False)
                print("\n***** self.program_RAW = {} *****\n".format(self.program_RAW))
            except Exception as e:
                print('failed RAW shaders.compileProgram() {}'.format(e))
            shaders.glDeleteShader(vs)
            shaders.glDeleteShader(fs)

    def setVerticesBufferData(self):
        try:
            x0, x1, y0, y1 = self.image_centered_position()
            # print(" x0, x1, y0, y1 {} {} {} {}".format(x0, x1, y0, y1))
        except Exception as e:
            print(" Failed image_centered_position() {}".format(e))
            x0, x1, y0, y1 = 0, 100, 0, 100
            # set background vertices
        backgroundVertices = [
            x0, y1, 0.0,
            x0, y0, 0.0,
            x1, y1, 0.0,
            x1, y1, 0.0,
            x0, y0, 0.0,
            x1, y0, 0.0]
        vertexData = np.array(backgroundVertices, np.float32)

        if self.vertexBuffer is not None:
            self.vertexBuffer.destroy()
        self.vertexBuffer = QOpenGLBuffer()
        self.vertexBuffer.create()
        self.vertexBuffer.bind()
        self.vertexBuffer.allocate(vertexData, 4 * len(vertexData))

    def setBufferData(self):
        # set background UV
        backgroundUV = [
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0]
        uvData = np.array(backgroundUV, np.float32)

        self.uvBuffer = QOpenGLBuffer()
        self.uvBuffer.create()
        self.uvBuffer.bind()
        self.uvBuffer.allocate(uvData, 4 * len(uvData))

    def setTexture(self):
        texture_ok = super(GLImageViewerShaders, self).setTexture()
        self.setVerticesBufferData()
        return texture_ok

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport.
        """
        # print(f"resizeGL {width}x{height}")
        if self.trace_calls:
            t = trace_method(self.tab)
        self._width = width*self.devicePixelRatio()
        self._height = height*self.devicePixelRatio()
        self.setVerticesBufferData()
        self.update()

    def initializeGL(self):
        """
        Initialize OpenGL, VBOs, upload data on the GPU, etc.
        """
        self.start_timing()

        time1 = get_time()
        self.set_shaders()
        self.add_time('set_shaders', time1)

        self.setVerticesBufferData()
        self.setBufferData()
        self.print_timing()

    def viewer_update(self):
        self.update()

    def paintGL(self):
        self.paintAll()

    def myPaintGL(self):
        """Paint the scene.
        """
        if self.textureID is None or not self.isValid():
            print("paintGL() not ready")
            return
        self.opengl_error()
        self.start_timing()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        if self._image.data.shape[2] == 4:
            self.program = self.program_RAW
        else:
            # TODO: check for other types: scalar ...
            self.program = self.program_RGB

        # Obtain uniforms and attributes
        self.aVert              = shaders.glGetAttribLocation(self.program, "vert")
        self.aUV                = shaders.glGetAttribLocation(self.program, "uV")
        self.uPMatrix           = shaders.glGetUniformLocation(self.program, 'pMatrix')
        self.uMVMatrix          = shaders.glGetUniformLocation(self.program, "mvMatrix")
        self.uBackgroundTexture = shaders.glGetUniformLocation(self.program, "backgroundTexture")
        self.channels_location    = shaders.glGetUniformLocation(self.program, "channels")
        self.black_level_location = shaders.glGetUniformLocation(self.program, "black_level")
        self.white_level_location = shaders.glGetUniformLocation(self.program, "white_level")
        self.g_r_coeff_location   = shaders.glGetUniformLocation(self.program, "g_r_coeff")
        self.g_b_coeff_location   = shaders.glGetUniformLocation(self.program, "g_b_coeff")
        self.max_value_location   = shaders.glGetUniformLocation(self.program, "max_value")
        self.max_type_location    = shaders.glGetUniformLocation(self.program, "max_type")
        self.gamma_location       = shaders.glGetUniformLocation(self.program, "gamma")

        # use shader program
        self.print_log("self.program = {}".format(self.program))
        shaders.glUseProgram(self.program)

        # set uniforms
        gl.glUniformMatrix4fv(self.uPMatrix, 1, gl.GL_FALSE, self.pMatrix)
        gl.glUniformMatrix4fv(self.uMVMatrix, 1, gl.GL_FALSE, self.mvMatrix)
        gl.glUniform1i(self.uBackgroundTexture, 0)

        gl.glUniform1i( self.channels_location, self._image.channels)

        # set color transformation parameters
        self.print_log("levels {} {}".format(self.filter_params.black_level.value,
                                             self.filter_params.white_level.value))
        gl.glUniform1f( self.black_level_location, self.filter_params.black_level.float)
        gl.glUniform1f( self.white_level_location, self.filter_params.white_level.float)

        # white balance coefficients
        gl.glUniform1f(self.g_r_coeff_location, self.filter_params.g_r.float)
        gl.glUniform1f(self.g_b_coeff_location, self.filter_params.g_b.float)

        # Should work for unsigned types for the moment
        gl.glUniform1f( self.max_value_location, (1 << self._image.precision)-1)
        gl.glUniform1f( self.max_type_location,  np.iinfo(self._image.data.dtype).max)

        gl.glUniform1f( self.gamma_location,       self.filter_params.gamma.float)

        # enable attribute arrays
        gl.glEnableVertexAttribArray(self.aVert)
        gl.glEnableVertexAttribArray(self.aUV)

        # set vertex and UV buffers
        # vert_buffers = VertexBuffers()
        # vert_buffers.vert_pos_buffer = vert_pos_buffer
        # vert_buffers.normal_buffer = normal_buffer
        # vert_buffers.tex_coord_buffer = tex_coord_buffer
        # vert_buffers.amount_of_vertices = int(len(index_array) / 3)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertexBuffer.bufferId())
        gl.glVertexAttribPointer(self.aVert, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.uvBuffer.bufferId())
        gl.glVertexAttribPointer(self.aUV, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # bind background texture
        # gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureID)
        gl.glEnable(gl.GL_TEXTURE_2D)

        # draw
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

        # disable attribute arrays
        gl.glDisableVertexAttribArray(self.aVert)
        gl.glDisableVertexAttribArray(self.aUV)
        gl.glDisable(gl.GL_TEXTURE_2D)

        shaders.glUseProgram(0)

        self.print_timing(force=True)

    def updateTransforms(self) -> float:
        if self.trace_calls:
            t = trace_method(self.tab)
        if self.display_timing:
            start_time = get_time()
        self.makeCurrent()
        w = self._width
        h = self._height
        dx, dy = self.new_translation()
        # Deduce new scale from mouse vertical displacement
        scale = self.new_scale(-self.mouse_zoom_displ.y(), self.tex_height)
        # update the window size
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        translation_unit = min(w, h)/2
        gl.glScale(scale, scale, scale)
        gl.glTranslate(dx/translation_unit, dy/translation_unit, 0)
        # the window corner OpenGL coordinates are (-+1, -+1)
        gl.glOrtho(0, w, 0, h, -1, 1)
        self.pMatrix = np.array(gl.glGetFloatv(gl.GL_PROJECTION_MATRIX), dtype=np.float32).flatten()

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        self.mvMatrix = np.array(gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX), dtype=np.float32).flatten()
        if self.display_timing:
            self.print_log('updateTransforms time {:0.1f} ms'.format((get_time()-start_time)*1000))
        return scale


if __name__ == '__main__':
    from qimview.image_readers import gb_image_reader
    # import numpy for generating random data points
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_image', help='input image')
    args = parser.parse_args()
    _params = vars(args)

    # define a Qt window with an OpenGL widget inside it
    # class TestWindow(QtGui.QMainWindow):
    class TestWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            self.widget = GLImageViewerShaders(self)
            self.show()
        def load(self):
            im = gb_image_reader.read(_params['input_image'])
            self.widget.set_image(im)
            self.widget.image_name = _params['input_image']
            # put the window at the screen position (100, 100)
            self.setGeometry(0, 0, self.widget._width, self.widget._height)
            self.setCentralWidget(self.widget)

    # create the Qt App and window
    app = QtWidgets.QApplication(sys.argv)
    window = TestWindow()
    window.load()
    window.show()
    app.exec_()