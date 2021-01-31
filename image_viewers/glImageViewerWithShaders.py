#
#
# started from https://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
#
#

from utils.qt_imports import *

import OpenGL.GL as gl
import argparse
import sys
from OpenGL.GL import shaders
import numpy as np
from image_viewers.ImageViewer import ImageViewer, ReadImage, trace_method, get_time

import pygame


class glImageViewerWithShaders(QOpenGLWidget, ImageViewer):
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
    fragmentShader = """
        #version 330 core
    
        in vec2 UV;
        uniform sampler2D backgroundTexture;
        uniform float white_level;
        uniform float black_level;
        uniform float gamma;
        out vec3 colour;
    
        void main() {
          colour = texture(backgroundTexture, UV).rgb;
          //colour.x = (colour.x-black_level)/(white_level-black_level);
          //colour.y = (colour.y-black_level)/(white_level-black_level);
          //colour.z = (colour.z-black_level)/(white_level-black_level);
          colour.rgb = (colour.rgb-black_level)/(white_level-black_level);
          colour.rgb = pow(colour.rgb, vec3(1.0/gamma));
        }
    """

    def __init__(self, parent=None):
        # QGLWidget.__init__(self, parent)
        QOpenGLWidget.__init__(self, parent)
        ImageViewer.__init__(self, parent)

        # self.setUpdateBehavior(QOpenGLWidget.PartialUpdate)

        _format = QtGui.QSurfaceFormat()
        print('profile is {}'.format(_format.profile()))
        print('version is {}'.format(_format.version()))
        # _format.setDepthBufferSize(24)
        # _format.setVersion(4,0)
        # _format.setProfile(PySide2.QtGui.QSurfaceFormat.CoreProfile)
        _format.setProfile(QtGui.QSurfaceFormat.CompatibilityProfile)
        self.setFormat(_format)

        self.setAutoFillBackground(False)
        self.textureID = None
        self.tex_width, self.tex_height = 0, 0
        self.opengl_debug = False
        self.synchronize_viewer = None
        self.pMatrix  = np.identity(4, dtype=np.float32)
        self.mvMatrix = np.identity(4, dtype=np.float32)
        pygame.font.init()
        self.program = None
        self.current_text = None

    def __del__(self):
        # self.makeCurrent()
        if self.textureID is not None:
            gl.glDeleteTextures(np.array([self.textureID]))

    def set_image(self, image):
        changed = super(glImageViewerWithShaders, self).set_image(image)
        if changed and self.textureID is not None:
            self.makeCurrent()
            self.initializeGL()
            self.update()

    def initializeGL(self):
        """
        Initialize OpenGL, VBOs, upload data on the GPU, etc.
        """
        if self.display_timing:
            start_time = get_time()

        # if self.program is not None:
        #     self.program = None
        # create shader program
        if self.program is None:
            self.vs = shaders.compileShader(self.vertexShader, gl.GL_VERTEX_SHADER)
            self.fs = shaders.compileShader(self.fragmentShader, gl.GL_FRAGMENT_SHADER)
            try:
                self.program = shaders.compileProgram(self.vs, self.fs, validate=False)
                print("\n***** self.program = {} *****\n".format(self.program))
            except Exception as e:
                print('failed shaders.compileProgram() {}'.format(e))
            shaders.glDeleteShader(self.vs)
            shaders.glDeleteShader(self.fs)

        # shaders.glUseProgram(self.program)

        # obtain uniforms and attributes
        self.aVert              = shaders.glGetAttribLocation(self.program, "vert")
        self.aUV                = shaders.glGetAttribLocation(self.program, "uV")
        self.uPMatrix           = shaders.glGetUniformLocation(self.program, 'pMatrix')
        self.uMVMatrix          = shaders.glGetUniformLocation(self.program, "mvMatrix")
        self.uBackgroundTexture = shaders.glGetUniformLocation(self.program, "backgroundTexture")
        self.black_level_location = shaders.glGetUniformLocation(self.program, "black_level")
        self.white_level_location = shaders.glGetUniformLocation(self.program, "white_level")
        self.gamma_location       = shaders.glGetUniformLocation(self.program, "gamma")

        if self.display_timing:
            print('initiliazeGL shaders time {:0.1f} ms'.format((get_time()-start_time)*1000))

        # set background vertices
        backgroundVertices = [
            0, 1, 0.0,
            0, 0, 0.0,
            1.0, 1, 0.0,
            1.0, 1, 0.0,
            0, 0, 0.0,
            1.0, 0, 0.0]

        self.vertexBuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertexBuffer)
        vertexData = np.array(backgroundVertices, np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 4 * len(vertexData), vertexData, gl.GL_STATIC_DRAW)

        # set background UV
        backgroundUV = [
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0]

        self.uvBuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.uvBuffer)
        uvData = np.array(backgroundUV, np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 4 * len(uvData), uvData, gl.GL_STATIC_DRAW)

        # background color
        # gl.glClearColor(0, 0, 0, 0)
        # Replace texture only if required
        img_height, img_width = self.cv_image.shape[0:2]
        if (self.tex_width,self.tex_height) != (img_width,img_height):
            if self.textureID is not None:
                gl.glDeleteTextures(np.array([self.textureID]))
            self.textureID = gl.glGenTextures(1)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureID)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, img_width, img_height,
                         0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, self.cv_image)
            self.tex_width, self.tex_height = img_width, img_height
        else:
            try:
                gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, img_width, img_height,
                             gl.GL_BGR, gl.GL_UNSIGNED_BYTE, self.cv_image)
            except Exception as e:
                print("glTexSubImage2D failed shape={}: {}".format(self.cv_image.shape, e))
        if self.display_timing:
            self.print_log('initiliazeGL time {:0.1f} ms'.format((get_time()-start_time)*1000))

    def my_drawText(self, position, textString, size=64, color=(255, 255, 255, 127)):
        if self.display_timing:
            start_time = get_time()
        new_text = (textString, size, color)

        # Set font
        # available_fonts = pygame.font.get_fonts()
        # font_name = available_fonts[self.font_number]
        # print("font_number {}".format(self.font_number))
        # self.font_number = (self.font_number + 1) % len(available_fonts)
        # font = pygame.font.Font(font_name, size)

        if self.current_text is None or new_text != self.current_text:
            font = pygame.font.Font(None, size)
            font.set_bold(False)
            # font = pygame.font.SysFont(available_fonts[0], 24)
            # Draw text
            self.textSurface = font.render(textString, True, color, (0, 0, 0, 0))
            self.textData = pygame.image.tostring(self.textSurface, "RGBA", True)
            self.current_text = new_text


        # Set viewport and projection matrix
        shaders.glUseProgram(0)
        gl.glViewport(0, 0, self._width, self._height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, 1, 0, 1, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        # gl.glRasterPos3d(*position)
        gl.glRasterPos3f(position[0], position[1], position[2])
        gl.glDrawPixels(self.textSurface.get_width(), self.textSurface.get_height(), gl.GL_RGBA,
                        gl.GL_UNSIGNED_BYTE,
                        self.textData)
        if self.display_timing:
            print('my_drawText time {:0.1f} ms'.format((get_time()-start_time)*1000))

    def opengl_error(self, force=False):
        if self.opengl_debug or force:
            status = gl.glGetError()
            if status != gl.GL_NO_ERROR:
                print(self.tab[0]+'gl error %s' % status)

    def paintAll(self):
        self.update()

    def paintGL(self):
        # gl = PySide2.QtWidgets.QOpenGLContext.currentContext().functions()

        painter = QtGui.QPainter()

        self.opengl_error()
        if self.trace_calls:
            t = trace_method(self.tab)
        self.makeCurrent()

        painter.begin(self)

        # gl.glEnable(gl.GL_DEPTH_TEST)
        # draw_text = True
        # if draw_text:
        #     # gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        #     # shaders.glUseProgram(0)
        #     # gl.glViewport(0, 0, self.width, self.height)
        #     # gl.glMatrixMode(gl.GL_PROJECTION)
        #     # gl.glLoadIdentity()
        #     # gl.glOrtho(0, 1, 0, 1, -1, 1)
        #     # gl.glMatrixMode(gl.GL_MODELVIEW)
        #     # gl.glLoadIdentity()
        #     color = QtGui.QColor(255, 50, 50, 255) if self.is_active() else QtGui.QColor(50, 50, 255, 255)
        #     painter.setPen(color)
        #     painter.setFont(QtGui.QFont('Decorative', 16))
        #     print("Trying to draw text")
        #     painter.drawText(10, self.height-10, self.image_name)
        #     self.opengl_error()

        painter.beginNativePainting()

        # gl.glEnable(gl.GL_BLEND)
        # gl.glBlendColor(1,1,1,0.5)
        # gl.glBlendFunc(gl.GL_CONSTANT_ALPHA	, gl.GL_CONSTANT_ALPHA	)

        try:
            self.updateViewPort()
            self.mypaintGL()
            # self.updateGL()
        except Exception as e:
            self.print_log(" failed paintGL {}".format(e))
        # self.opengl_error()

        # self.opengl_error(force=True)
        # self.opengl_error(force=True)
        # rectangle = QtCore.QRectF(10,10,20,20)
        # painter.drawRect(rectangle)
        #status = gl.glGetError()
        # self.opengl_error(force=True)


        painter.endNativePainting()

        # gl.glDisable(gl.GL_DEPTH_TEST)
        draw_text = False
        if draw_text:
            # gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())
            # shaders.glUseProgram(0)
            # gl.glViewport(0, 0, self.width, self.height)
            # gl.glMatrixMode(gl.GL_PROJECTION)
            # gl.glLoadIdentity()
            # gl.glOrtho(0, 1, 0, 1, -1, 1)
            # gl.glMatrixMode(gl.GL_MODELVIEW)
            # gl.glLoadIdentity()
            color = QtGui.QColor(255, 50, 50, 255) if self.is_active() else QtGui.QColor(50, 50, 255, 255)
            painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
            pen = painter.pen()
            pen.setColor(color)
            painter.setPen(pen)
            # painter.setPen(color)
            font = QtGui.QFont('Decorative', 16)
            painter.setFont(font)
            print("Trying to draw text")
            painter.drawText(10, self._height - 10, self.image_name)
            self.opengl_error()

        # shaders.glUseProgram(0)
        # if self.is_active():
        #     gl.glColor3f(1, 0.2, 0.2)
        # else:
        #     gl.glColor3f(1., 1., 1.)
        # self.renderText(20, 20, self.image_name)

        # painter.beginNativePainting()

        # # self.makeCurrent()
        # painter.endNativePainting()

        painter.end()
        # self.makeCurrent()
        # self.updateViewPort()
        # self.updateGL()
        self.opengl_error()

    # def paintEvent(self, event):
    #     self.makeCurrent()
    #     if self.trace_calls:
    #         t = trace_method(self.tab)
    #     self.paintAll()
    #     self.opengl_error()

    def mypaintGL(self):
        """Paint the scene.
        """
        self.opengl_error()
        if self.display_timing:
            start_time = get_time()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # use shader program
        print("self.program = {}".format(self.program))
        shaders.glUseProgram(self.program)

        # set uniforms
        gl.glUniformMatrix4fv(self.uPMatrix, 1, gl.GL_FALSE, self.pMatrix)
        gl.glUniformMatrix4fv(self.uMVMatrix, 1, gl.GL_FALSE, self.mvMatrix)
        gl.glUniform1i(self.uBackgroundTexture, 0)

        # set color transformation parameters
        gl.glUniform1f( self.black_level_location, self.filter_params.black_level.float)
        gl.glUniform1f( self.white_level_location, self.filter_params.white_level.float)
        gl.glUniform1f( self.gamma_location,       self.filter_params.gamma)

        # enable attribute arrays
        gl.glEnableVertexAttribArray(self.aVert)
        gl.glEnableVertexAttribArray(self.aUV)

        # set vertex and UV buffers
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertexBuffer)
        gl.glVertexAttribPointer(self.aVert, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.uvBuffer)
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

        # color = (255, 50, 50, 128) if self.is_active() else (255, 255, 255, 128)
        # self.my_drawText(position=[0.02, 0.02, 0], textString=self.image_name, color=color)
        shaders.glUseProgram(0)

        if self.display_timing:
            print('paintGL time {:0.1f} ms'.format((get_time()-start_time)*1000))

    def updateViewPort(self):

        if self.tex_height == 0:
            return
        # gl.glEnable(gl.GL_DEPTH_TEST)

        self.opengl_error()
        if self.display_timing:
            start_time = get_time()
        # keep image proportions
        w = self._width
        h = self._height
        print('self width height {} {}'.format(self._width, self._height))
        print('self width() height() {} {}'.format(self.width(), self.height()))
        dx, dy = self.new_translation()
        # scale = max(self.mouse_zx, self.mouse_zy)
        scale = self.new_scale(self.mouse_zy, self.tex_height)
        image_ratio = float(self.tex_width)/float(self.tex_height)
        if h*image_ratio < w:
            view_width  = int(h*image_ratio+0.5)
            view_height = h
            start_x = int((w - view_width) / 2 + 0.5)
            start_y = 0
        else:
            view_width  = w
            view_height = int(w/image_ratio+0.5)
            start_x = 0
            start_y = int((h - view_height) / 2 + 0.5)
        print("image_ratio = {}".format(image_ratio))
        print("view_width = {} view_height {}".format(view_width,view_height))
        gl.glViewport(start_x, start_y, view_width, view_height)
        # update the window size
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        translation_unit = min(w, h)/2
        gl.glScale(scale, scale, scale)
        gl.glTranslate(dx/translation_unit, dy/translation_unit, 0)
        # the window corner OpenGL coordinates are (-+1, -+1)
        gl.glOrtho(0, 1, 0, 1, -1, 1)
        self.pMatrix = np.array(gl.glGetFloatv(gl.GL_PROJECTION_MATRIX), dtype=np.float32).flatten()

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        self.mvMatrix = np.array(gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX), dtype=np.float32).flatten()

        if self.display_timing:
            print('resizeGL time {:0.1f} ms'.format((get_time()-start_time)*1000))

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport.
        """
        self.opengl_error()
        self._width = width
        self._height = height
        self.updateViewPort()
        #self.update()

    def resizeEvent(self, event):
        print(event)
        print("event.width {}".format(event.size().width()))
        print("width = {}".format(self.width()))
        self.resizeGL(self.width(), self.height())

    def mousePressEvent(self, event):
        self.mouse_press_event(event)

    def mouseMoveEvent(self, event):
        self.mouse_move_event(event)

    def mouseReleaseEvent(self, event):
        self.mouse_release_event(event)

    def mouseDoubleClickEvent(self, event):
        self.mouse_double_click_event(event)

    def wheelEvent(self, event):
        self.mouse_wheel_event(event)


if __name__ == '__main__':
    # import numpy for generating random data points
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input_image', help='input image')
    args = parser.parse_args()
    _params = vars(args)

    # define a Qt window with an OpenGL widget inside it
    # class TestWindow(QtGui.QMainWindow):
    class TestWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            self.widget = glImageViewerWithShaders()
            im = ReadImage(_params['input_image'])
            self.widget.set_image(im)
            # put the window at the screen position (100, 100)
            self.setGeometry(0, 0, self.widget._width, self.widget._height)
            self.setCentralWidget(self.widget)
            self.show()

    # create the Qt App and window
    app = QtWidgets.QApplication(sys.argv)
    window = TestWindow()
    window.show()
    app.exec_()