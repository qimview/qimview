#
#
# started from https://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
#
# check also https://doc.qt.io/archives/4.6/opengl-overpainting.html
#

from ..utils.qt_imports    import QtWidgets
from .image_viewer         import trace_method
from .gl_image_viewer_base import GLImageViewerBase

import OpenGL
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl

import argparse
import sys


class GLImageViewer(GLImageViewerBase):

    def __init__(self, parent=None, event_recorder=None):
        self.event_recorder = event_recorder
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.opengl_debug = True
        self.trace_calls  = False

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc.
        """
        # self.setTexture()
        pass

    def viewer_update(self):
        self.update()

    def paintGL(self):
        self.paintAll()

    def myPaintGL(self):
        """Paint the scene.
        """
        if self.trace_calls:
            t = trace_method(self.tab)
        self.start_timing()
        if self.texture is None:
            print("GLImageViewer paintGL texture not set")
            return
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_DECAL)
        # TODO: fix textureID will only work for RGB textures, not YUV
        assert self.texture.textureRGB is not None, "RGB texture not initialized"
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.textureRGB)
        # gl.glGenerateMipmap (gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBegin(gl.GL_QUADS)

        x0, x1, y0, y1 = self.image_centered_position()
        x0 = int(x0)
        x1 = int(x1)
        y0 = int(y0)
        y1 = int(y1)
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

        self.print_timing(add_total=True)
        self.opengl_error()

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport.
        """
        # print("ResizeGL")
        if self.trace_calls:
            t = trace_method(self.tab)
        # size give for opengl are in pixels, qt uses device independent size otherwise
        print(f"self.devicePixelRatio() {self.devicePixelRatio()}")
        self._width = width*self.devicePixelRatio()
        self._height = height*self.devicePixelRatio()
        # print("width height ratios {} {}".format(self._width/self.width(), self._height/self.height()))
        self.viewer_update()

    def event(self, evt):
        if self.event_recorder is not None:
            self.event_recorder.store_event(self, evt)
        return super().event(evt)

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
            self.widget = GLImageViewer(self)
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
    app.exec()
