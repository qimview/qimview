#
#
# started from https://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
#
# check also https://doc.qt.io/archives/4.6/opengl-overpainting.html
#

from qimview.utils.qt_imports   import QtWidgets, QOpenGLWidget, QtCore, QtGui

import OpenGL
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
import OpenGL.GLU as glu
import traceback
import numpy as np
from typing import Optional, Tuple

from qimview.utils.viewer_image import ImageFormat, ViewerImage
from qimview.image_viewers.image_viewer import ImageViewer, trace_method


class GLImageViewerBase(ImageViewer, QOpenGLWidget, ):

    def __init__(self, parent : Optional[QtWidgets.QWidget] = None):
        ImageViewer.__init__(self, self)
        QOpenGLWidget.__init__(self, parent)
        self.setAutoFillBackground(False)

        _format = QtGui.QSurfaceFormat()
        print('profile is {}'.format(_format.profile()))
        print('version is {}'.format(_format.version()))
        # _format.setDepthBufferSize(24)
        # _format.setVersion(4,0)
        # _format.setProfile(PySide2.QtGui.QSurfaceFormat.CoreProfile)
        _format.setProfile(QtGui.QSurfaceFormat.CompatibilityProfile)
        self.setFormat(_format)

        # self.setFormat()
        self.textureID  = None
        self.tex_width, self.tex_height = 0, 0
        self.opengl_debug = True
        self.current_text = None
        self.cursor_imx_ratio = 0.5
        self.cursor_imy_ratio = 0.5
        self.trace_calls = False
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

    # def __del__(self):
    #     if self.textureID is not None:
    #         gl.glDeleteTextures(np.array([self.textureID]))

    def set_image(self, image):
        if self.trace_calls:
            t = trace_method(self.tab)
        changed = super(GLImageViewerBase, self).set_image(image)

        if self._image is None:
            return
        img_width = self._image.data.shape[1]
        if img_width % 4 != 0:
            print("Image is resized to a multiple of 4 dimension in X")
            img_width = ((img_width >> 4) << 4)
            im = np.ascontiguousarray(self._image.data[:,:img_width, :])
            self._image = ViewerImage(im, precision = self._image.precision, downscale = 1,
                                        channels = self._image.channels)
            print(self._image.data.shape)

        if changed:  # and self.textureID is not None:
            print(f"self.textureID {self.textureID}")
            if self.setTexture():
                self.show()
                self.update()
            else:
                print("setTexture() return False")

    def synchronize_data(self, other_viewer):
        super(GLImageViewerBase, self).synchronize_data(other_viewer)
        other_viewer.cursor_imx_ratio = self.cursor_imx_ratio
        other_viewer.cursor_imy_ratio = self.cursor_imy_ratio

    def opengl_error(self, force=False):
        if self.opengl_debug or force:
            status = gl.glGetError()
            if status != gl.GL_NO_ERROR:
                print(self.tab[0]+'gl error %s' % status)

    def setTexture(self) -> bool:
        """
        :return: set opengl texture based on input numpy array image
        """
        # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP_SGIS, gl.GL_TRUE)

        if self.trace_calls:
            t = trace_method(self.tab)
        self.start_timing()
        self.makeCurrent()

        # Replace texture only if required
        if self._image is None:
            print("self._image is None")
            return False

        img_height, img_width = self._image.data.shape[:2]

        # default type
        gl_types = {
            'int8'   : gl.GL_BYTE,
            'uint8'  : gl.GL_UNSIGNED_BYTE,
            'int16'  : gl.GL_SHORT,
            'uint16' : gl.GL_UNSIGNED_SHORT,
            'int32'  : gl.GL_INT,
            'uint32' : gl.GL_UNSIGNED_INT,
            'float32': gl.GL_FLOAT,
            'float64': gl.GL_DOUBLE
        }
        gl_type = gl_types[self._image.data.dtype.name]

        # It seems that the dimension in X should be even

        # Not sure what is the right parameter for internal format of 2D texture based
        # on the input data type uint8, uint16, ...
        # need to test with different DXR images
        internal_format = gl.GL_RGB
        if self._image.data.shape[2] == 3:
            if self._image.precision == 8:
                internal_format = gl.GL_RGB
            if self._image.precision == 10:
                internal_format = gl.GL_RGB10
            if self._image.precision == 12:
                internal_format = gl.GL_RGB12
        if self._image.data.shape[2] == 4:
            if self._image.precision == 8:
                internal_format = gl.GL_RGBA
            else:
                if self._image.precision <= 12:
                    internal_format = gl.GL_RGBA12
                else:
                    if self._image.precision <= 16:
                        internal_format = gl.GL_RGBA16
                    else:
                        internal_format = gl.GL_RGBA32F

        channels2format = {
            ImageFormat.CH_RGB  : gl.GL_RGB,
            ImageFormat.CH_BGR  : gl.GL_BGR,
            ImageFormat.CH_Y    : gl.GL_RED, # not sure about this one
            ImageFormat.CH_RGGB : gl.GL_RGBA, # we save 4 component data
            ImageFormat.CH_GRBG : gl.GL_RGBA,
            ImageFormat.CH_GBRG : gl.GL_RGBA,
            ImageFormat.CH_BGGR : gl.GL_RGBA
        }
        texture_pixel_format = channels2format[self._image.channels]

        if (self.tex_width,self.tex_height) != (img_width,img_height):
            try:
                if self.textureID is not None:
                    gl.glDeleteTextures(np.array([self.textureID]))
                self.textureID = gl.glGenTextures(1)
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureID)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
                # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 10)
                # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
                # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST_MIPMAP_NEAREST)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0,
                                internal_format,
                                img_width, img_height,
                            0, texture_pixel_format, gl_type, self._image.data)
                # gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
                self.tex_width, self.tex_height = img_width, img_height
            except Exception as e:
                print("setTexture failed shape={}: {}".format(self._image.data.shape, e))
                return False
        else:
            try:
                gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, img_width, img_height,
                             texture_pixel_format, gl_type, self._image.data)
                # gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            except Exception as e:
                print("setTexture failed shape={}: {}".format(self._image.data.shape, e))
                return False

        self.print_timing(add_total=True)
        self.opengl_error()
        return True

    # @abstract_method
    def viewer_update(self):
        self.update()

    # To be defined in children
    def myPaintGL(self):  pass 

    def paintAll(self):
        if self.trace_calls:
            t = trace_method(self.tab)
        if self.textureID is None or not self.isValid() or not self.isVisible():
            print("paintGL()** not ready {} {}".format(self.textureID, self.isValid()))
            return
        # No need for makeCurrent() since it is called from PaintGL() only
        # self.makeCurrent()
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.beginNativePainting()
        im_pos = None
        scale  = None
        try:
            self.updateViewPort()
            scale = self.updateTransforms()
            self.myPaintGL()
            if self.show_cursor:
                im_pos = self.gl_draw_cursor()
        except Exception as e:
            self.print_log(" failed paintGL {}".format(e))
            traceback.print_exc()
        painter.endNativePainting()
        # status = gl.glGetError()

        draw_text = True
        if draw_text:
            # adapt scale depending on the ratio image / viewport
            scale *= self._width/self._image.data.shape[1]
            self.display_text(painter, self.display_message(im_pos, scale))

        # draw histogram
        if self.show_histogram:
            current_image = self._image.data
            rect = QtCore.QRect(0, 0, self.width(), self.height())
            histograms = self.compute_histogram_Cpp(current_image, show_timings=self.display_timing)
            self.display_histogram(histograms, 1,  painter, rect, show_timings=self.display_timing)

        painter.end()
        # Seems required here, at least on linux
        # self.update()

    def set_cursor_image_position(self, cursor_x, cursor_y):
        """
        Sets the image position from the cursor in proportion of the image dimension
        :return:
        """
        self.updateTransforms()
        ratio = self.screen().devicePixelRatio()
        self.print_log("ratio {}".format(ratio))
        pos_x = cursor_x * ratio
        pos_y = (self.height() - cursor_y) * ratio
        self.print_log("pos {} {}".format(pos_x, pos_y))
        x0, x1, y0, y1 = self.image_centered_position()

        gl_posX, gl_posY = self.get_mouse_gl_coordinates(pos_x, pos_y)
        self.cursor_imx_ratio = (gl_posX - x0) / (x1 - x0)
        self.cursor_imy_ratio = 1 - (gl_posY - y0) / (y1 - y0)
        self.print_log("cursor ratio {} {}".format(self.cursor_imx_ratio, self.cursor_imy_ratio))


    def gl_draw_cursor(self) -> Optional[Tuple[int, int]]:
        x0, x1, y0, y1 = self.image_centered_position()

        im_x = int(self.cursor_imx_ratio*self.tex_width)
        im_y = int(self.cursor_imy_ratio*self.tex_height)

        glpos_from_im_x = (im_x+0.5)*(x1-x0)/self.tex_width + x0
        glpos_from_im_y = (self.tex_height - (im_y+0.5))*(y1-y0)/self.tex_height+y0

        # get image coordinates
        length = 20 # /self.current_scale
        width = 4
        gl.glLineWidth(width)
        gl.glColor3f(0.0, 1.0, 1.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(glpos_from_im_x-length, glpos_from_im_y, -0.001)
        gl.glVertex3f(glpos_from_im_x+length, glpos_from_im_y, -0.001)
        gl.glVertex3f(glpos_from_im_x, glpos_from_im_y-length, -0.001)
        gl.glVertex3f(glpos_from_im_x, glpos_from_im_y+length, -0.001)
        gl.glEnd()
        if im_x>=0 and im_x<self.tex_width and im_y>=0 and im_y<=self.tex_height:
            return (im_x, im_y) 
        return None


    def image_centered_position(self):
        w = self._width
        h = self._height
        self.print_log(f'self width height {self._width} {self._height} tex {self.tex_width} {self.tex_height}')
        if self.tex_width == 0 or self.tex_height == 0:
            return 0, w, 0, h
        # self.print_log(' {}x{}'.format(w, h))
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

        x0 = start_x
        x1 = start_x+view_width
        y0 = start_y
        y1 = start_y+view_height
        return x0, x1, y0, y1

    def updateViewPort(self):
        if self.trace_calls:
            t = trace_method(self.tab)
        self.start_timing()
        # keep image proportions
        w = self._width
        h = self._height
        # print(f" w, h {w, h}")
        try:
            gl.glViewport(0,0,w,h) # keep everything in viewport
        except Exception as e:
            self.print_log(" failed glViewport {}".format(e))
        self.print_timing(add_total=True)

    def updateTransforms(self) -> float:
        if self.trace_calls:
            t = trace_method(self.tab)
        self.start_timing()
        self.makeCurrent()
        w = self._width
        h = self._height
        dx, dy = self.new_translation()
        scale = self.new_scale(self._mouse_events._mouse_zy, self.tex_height)
        print(f"updateTransforms scale {scale}")
        try:
            # print("current context ", QtOpenGL.QGLContext.currentContext())
            # gl = QtOpenGL.QGLContext.currentContext().functions()
            # update the window size
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            translation_unit = min(w, h)/2
            # self.print_log("scale {}".format(scale))
            gl.glScale(scale, scale, scale)
            gl.glTranslate(dx/translation_unit, dy/translation_unit, 0)
            # the window corner OpenGL coordinates are (-+1, -+1)
            gl.glOrtho(0, w, 0, h, -1, 1)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
        except Exception as e:
            self.print_log(" setting gl matrices failed {}".format(e))
        self.print_timing(add_total=True)
        self.opengl_error()
        return scale

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

    def get_mouse_gl_coordinates(self, x, y):
        modelview = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
        projection = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        posX, posY, posZ = glu.gluUnProject(x, y, 0, modelview, projection, viewport)
        return posX, posY

    def mousePressEvent(self, event):
        self._mouse_events.mouse_press_event(event)

    def mouseMoveEvent(self, event):
        if self.show_cursor:
            self.set_cursor_image_position(event.x(), event.y())
        self._mouse_events.mouse_move_event(event)

    def mouseReleaseEvent(self, event):
        self._mouse_events.mouse_release_event(event)

    def mouseDoubleClickEvent(self, event):
        self._mouse_events.mouse_double_click_event(event)

    def wheelEvent(self, event):
        self._mouse_events.mouse_wheel_event(event)

    def keyPressEvent(self, event):
        # TODO: Fix the correct parameters for selecting image zoom/pan
        x0, x1, y0, y1 = self.image_centered_position()
        print(f"image centered position {x1-x0} x {y1-y0}")
        self.key_press_event(event, wsize=QtCore.QSize(x1-x0, y1-y0))

    def resizeEvent(self, event):
        """Called upon window resizing: reinitialize the viewport.
        """
        if self.trace_calls:
            t = trace_method(self.tab)
        self.print_log(f"resize {event.size()}  self {self.width()} {self.height()}")
        self.evt_width = event.size().width()
        self.evt_height = event.size().height()
        QOpenGLWidget.resizeEvent(self, event)
        self.print_log(f"resize {event.size()}  self {self.width()} {self.height()}")

