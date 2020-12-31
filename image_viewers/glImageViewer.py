#
#
# started from https://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
#
# check also https://doc.qt.io/archives/4.6/opengl-overpainting.html
#

from Qt import QtWidgets

import OpenGL
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl

import argparse
import sys
from image_viewers.ImageViewer import ReadImage, trace_method
from image_viewers.qglwImageViewerBase import qglImageViewerBase


class glImageViewer(qglImageViewerBase):

    def __init__(self, parent=None):
        qglImageViewerBase.__init__(self, parent)
        self.setAutoFillBackground(False)
        # self.setFormat()
        self.textureID  = None
        self.tex_width, self.tex_height = 0, 0
        self.opengl_debug = True
        # pygame.font.init()
        self.trace_calls  = True

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc.
        """
        # self.setTexture()
        pass

    def paintGL(self):
        """Paint the scene.
        """
        if self.trace_calls:
            t = trace_method(self.tab)
        self.start_timing()
        if self.textureID is None:
            return
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_DECAL)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureID)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBegin(gl.GL_QUADS)

        x0, x1, y0, y1 = self.image_centered_position()
        # print("{} {} {} {}".format(x0,x1,y0,y1))

        gl.glTexCoord2i(0, 0)
        gl.glVertex2i(x0, y1)

        gl.glTexCoord2i(0, 1)
        gl.glVertex2i(x0, y0)

        gl.glTexCoord2i(1, 1)
        gl.glVertex2i(x1, y0)

        gl.glTexCoord2i(1, 0)
        gl.glVertex2i(x1, y1)

        gl.glEnd()

        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE)

        self.gl_draw_cursor()

        self.print_timing(add_total=True)
        self.opengl_error()


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
            self.widget = glImageViewer()
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