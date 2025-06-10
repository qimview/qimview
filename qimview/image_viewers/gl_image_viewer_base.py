#
#
# started from https://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
#
# check also https://doc.qt.io/archives/4.6/opengl-overpainting.html
#

import traceback
from typing import Optional, Tuple
import numpy as np
import OpenGL
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl

from .gltexture import GLTexture
from qimview.utils.qt_imports   import QtWidgets, QOpenGLWidget, QtCore, QtGui
from qimview.utils.viewer_image import ViewerImage
from qimview.image_viewers.image_viewer import ImageViewer, trace_method, OverlapMode
import glm

class GLImageViewerBase(ImageViewer, QOpenGLWidget, ):

    def __init__(self, parent : Optional[QtWidgets.QWidget] = None):
        ImageViewer.__init__(self, self)
        QOpenGLWidget.__init__(self, parent)
        self.setAutoFillBackground(False)

        _format = QtGui.QSurfaceFormat()
        #_format.setDepthBufferSize(24)
        #_format.setVersion(3,3)
        #_format.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        _format.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
        _format.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
        self.setFormat(_format)

        print(f'profile is {_format.profile()}')
        print(f'version is {_format.version()}')

        self.texture     : GLTexture | None = None
        # YUV texture of reference image
        self.texture_ref : GLTexture | None = None
        self.opengl_debug = True
        self.current_text = None
        self.cursor_imx_ratio = 0.5
        self.cursor_imy_ratio = 0.5
        self.trace_calls = False
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        # default type
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

        # Projection matrix
        self.pMatrix  : np.ndarray | None = None
        # Model view matrix
        self.mvMatrix : np.ndarray | None = None

        # Projection matrix
        self.pMatrix_glm  : glm.mat4  = glm.mat4()
        # Model view matrix
        self.mvMatrix_glm : glm.mat4  = glm.mat4()

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

        if changed:
            if self.setTexture():
                # self.show()
                # self.update()
                pass
            else:
                print("setTexture() return False")

    def set_image_fast(self,
                       image,
                       image_ref=None,
                       texture_ref = None,
                       use_crop:bool=True,
                       use_PBO: bool=False):
        """_summary_

        Args:
            image (_type_): _description_
            image_ref (_type_, optional): _description_. Defaults to None.
            texture_ref (_type_, optional): _description_. Defaults to None.
            use_crop (bool, optional): _description_. Defaults to True.
            use_PBO (bool, optional): When PBO (Pixel Buffer Object) is used, the displayed image is
            delayed until the next image display. Defaults to False.
        """
        self._image     = image
        self._image_ref = image_ref
        self.image_id += 1
        res = self.setTexture(use_crop, texture_ref, use_PBO=use_PBO)
        if not res: print("setTexture() returned False")

    def synchronize_data(self, other_viewer):
        super(GLImageViewerBase, self).synchronize_data(other_viewer)
        other_viewer.cursor_imx_ratio = self.cursor_imx_ratio
        other_viewer.cursor_imy_ratio = self.cursor_imy_ratio

    def opengl_error(self, force=False):
        if self.opengl_debug or force:
            status = gl.glGetError()
            if status != gl.GL_NO_ERROR:
                print(self.tab[0]+'gl error %s' % status)

    def setTexture(self, use_crop:bool=True, texture_ref : GLTexture = None, use_PBO:bool=False) -> bool:
        """ set opengl texture based on input numpy array image

        Args:
            use_crop (bool, optional): _description_. Defaults to True.
            texture_ref (GLTexture, optional): Texture of compared image, if not set, will be computed. Defaults to None.
            use_PBO (bool, optional): When PBO (Pixel Buffer Object) is used, the displayed image is delayed until the next image display. Defaults to False.
        """
        # print(f"{use_crop=}")

        # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP_SGIS, gl.GL_TRUE)

        if self.trace_calls:
            t = trace_method(self.tab)
        if self._display_timing: self.start_timing()
        self._makeCurrent()
        # _gl = QtGui.QOpenGLContext.currentContext().functions()
        _gl = gl

        # Replace texture only if required
        if self._image is None:
            print("self._image is None")
            return False

        # Set Y, U and V
        if self.texture is None:
            self.texture = GLTexture(_gl)
            h_min = 0
            h_max = self._image.data.shape[0]
        else:
            if use_crop:
                # Compute Y range of displayed texture
                self.updateTransforms()
                ratio = self.screen().devicePixelRatio()
                _, _, y0, y1 = self.image_centered_position()
                _, gl_posY0 = self.get_gl_coordinates(0, 0)
                _, gl_posY1 = self.get_gl_coordinates(0, self.height() * ratio)
                h0 = 1 - (gl_posY0 - y0) / (y1 - y0)
                h1 = 1 - (gl_posY1 - y0) / (y1 - y0)
                h0 = int(h0*self.texture.height+0.5)
                h1 = int(h1*self.texture.height+0.5)
                h0=min(self.texture.height,max(0,h0))
                h1=min(self.texture.height,max(0,h1))
                h_min = min(h0,h1)
                h_max = max(h0,h1)
            else:
                h_min = 0
                h_max = self._image.data.shape[0]
            # print(f"{h_min=} {h_max=}")

        self.print_log("cursor ratio {} {}".format(self.cursor_imx_ratio, self.cursor_imy_ratio))

        self.texture.create_texture_gl(self._image, h_min, h_max, use_PBO=use_PBO)
        # Set image_ref if available and compatible
        if self._image_ref and self._image_ref.channels == self._image.channels:
            if texture_ref:
                self.texture_ref = texture_ref
            else:
                if self.texture_ref is None:
                    self.texture_ref = GLTexture(_gl)
                self.texture_ref.create_texture_gl(self._image_ref, h_min, h_max, use_PBO=use_PBO)

        if self._display_timing: self.print_timing(add_total=True)
        self.opengl_error()
        # self.doneCurrent()
        return True

    # @abstract_method
    def viewer_update(self):
        self.update()

    # To be defined in children
    def myPaintGL(self):  pass

    def paintAll(self, make_current: bool = True):
        """ Should be called from PaintGL() exclusively """
        # print("GLIVB paintAll")
        if not self.isValid():
            print(f"paintAll() opengl widget not yet valid")
            return
        if self.trace_calls:
            t = trace_method(self.tab)
        if self._image is None:
            return
        if self.texture is None or not self.isValid() or not self.isVisible():
            print(f"paintGL()** not ready {self.texture} isValid = {self.isValid()} isVisible {self.isVisible()}")
            return
        # No need for makeCurrent() since it is called from PaintGL() only ?
        if make_current: self._makeCurrent()
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.beginNativePainting()
        im_pos = None
        scale  = 1
        try:
            scale = self.updateTransforms()
            self.myPaintGL()

            # Keep openGL drawing working by setting the projection matrix and viewport
            self.updateViewPort()
            # Still need this part for drawing lines (cursor, etc...)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadMatrixf(self.pMatrix.data)
            if self.show_cursor:
                im_pos = self.gl_draw_cursor()
            if self._show_overlap:
                self.gl_draw_overlap_separation()

            for _,cb in self.gl_paint_callbacks.items():
                cb(self)

        except Exception as e:
            self.print_log(" failed paintGL {}".format(e))
            traceback.print_exc()
        painter.endNativePainting()
        # status = _gl.glGetError()

        # adapt scale depending on the ratio image / viewport
        if self._show_text:
            scale *= self._width/self._image.data.shape[1]
            self.display_text(painter, self.display_message(im_pos, scale))

        # draw histogram
        if self.show_histogram:
            current_image = self._image.data
            rect = QtCore.QRect(0, 0, self.width(), self.height())
            histograms = self.compute_histogram_Cpp(current_image, show_timings=self._display_timing)
            self.display_histogram(histograms, 1,  painter, rect, show_timings=self._display_timing)

        for _,cb in self.gl_nonnative_paint_callbacks.items():
            cb(self, painter)

        painter.end()
        # self.context().swapBuffers(self.context().surface())
        # Seems required here, at least on linux
        # self.update()
        # doneCurrent() fails on windows
        # self.doneCurrent()

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

        gl_posX, gl_posY = self.get_gl_coordinates(pos_x, pos_y)
        self.cursor_imx_ratio = (gl_posX - x0) / (x1 - x0)
        self.cursor_imy_ratio = 1 - (gl_posY - y0) / (y1 - y0)
        self.print_log("cursor ratio {} {}".format(self.cursor_imx_ratio, self.cursor_imy_ratio))


    def gl_draw_cursor(self) -> Optional[Tuple[int, int]]:
        # _gl = QtGui.QOpenGLContext.currentContext().functions()
        _gl = gl

        x0, x1, y0, y1 = self.image_centered_position()

        im_x = int(self.cursor_imx_ratio*self.texture.width)
        im_y = int(self.cursor_imy_ratio*self.texture.height)

        glpos_from_im_x = (im_x+0.5)*(x1-x0)/self.texture.width + x0
        glpos_from_im_y = (self.texture.height - (im_y+0.5))*(y1-y0)/self.texture.height+y0

        # get image coordinates
        length = 20 # /self.current_scale
        width = 4
        _gl.glLineWidth(width)
        gl.glColor3f(0.0, 1.0, 1.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(glpos_from_im_x-length, glpos_from_im_y, -0.001)
        gl.glVertex3f(glpos_from_im_x+length, glpos_from_im_y, -0.001)
        gl.glVertex3f(glpos_from_im_x, glpos_from_im_y-length, -0.001)
        gl.glVertex3f(glpos_from_im_x, glpos_from_im_y+length, -0.001)
        gl.glEnd()
        if im_x>=0 and im_x<self.texture.width and im_y>=0 and im_y<=self.texture.height:
            return (im_x, im_y)
        return None

    def gl_draw_overlap_separation(self):
        # _gl = QtGui.QOpenGLContext.currentContext().functions()
        _gl = gl

        x0, x1, y0, y1 = self.image_centered_position()

        im_x = int(self.cursor_imx_ratio*self.texture.width)
        im_y = int(self.cursor_imy_ratio*self.texture.height)

        glpos_from_im_x = (im_x+0.5)*(x1-x0)/self.texture.width + x0
        glpos_from_im_y = (self.texture.height - (im_y+0.5))*(y1-y0)/self.texture.height+y0

        match self._overlap_mode:
            case OverlapMode.Horizontal:
                width = 2
                _gl.glLineWidth(width)
                _gl.glColor3f(0.0, 1.0, 1.0)
                _gl.glBegin(gl.GL_LINES)
                _gl.glVertex3f(glpos_from_im_x, y0, -0.001)
                _gl.glVertex3f(glpos_from_im_x, y1, -0.001)
                _gl.glEnd()
            case OverlapMode.Vertical:
                width = 2
                _gl.glLineWidth(width)
                _gl.glColor3f(0.0, 1.0, 1.0)
                _gl.glBegin(gl.GL_LINES)
                _gl.glVertex3f(x0, glpos_from_im_y, -0.001)
                _gl.glVertex3f(x1, glpos_from_im_y, -0.001)
                _gl.glEnd()
            case OverlapMode.Rectangle:
                w = 0.05*(x1-x0)
                h = 0.2*(y1-y0)
                px = glpos_from_im_x
                py = glpos_from_im_y
                dx = w*(1.01)
                dy = 0.0
                _gl.glLineWidth(1)
                _gl.glColor3f(1.0, 1.0, 0.0)
                _gl.glBegin(gl.GL_LINES)
                _gl.glVertex3f(px,py, -0.001)
                _gl.glVertex3f(px+dx, py+dy, -0.001)
                _gl.glEnd()
                width = 2
                _gl.glLineWidth(width)
                _gl.glColor3f(0.0, 1.0, 1.0)
                _gl.glBegin(gl.GL_LINE_LOOP)
                _gl.glVertex3f(px+dx,   py+dy,   -0.001)
                _gl.glVertex3f(px+dx+w, py+dy,   -0.001)
                _gl.glVertex3f(px+dx+w, py+dy-h, -0.001)
                _gl.glVertex3f(px+dx,   py+dy-h, -0.001)
                _gl.glEnd()

    def image_centered_position(self):
        w = self._width
        h = self._height
        self.print_log(f'self width height {self._width} {self._height} tex {self.texture.width} {self.texture.height}')
        if self.texture.width == 0 or self.texture.height == 0:
            return 0, w, 0, h
        # self.print_log(' {}x{}'.format(w, h))
        image_ratio = float(self.texture.width)/float(self.texture.height)
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
        if self._display_timing: self.start_timing()
        # keep image proportions
        w = self._width
        h = self._height
        # print(f" w, h {w, h}")
        try:
            gl.glViewport(0,0,w,h) # keep everything in viewport
        except Exception as e:
            self.print_log(" failed glViewport {}".format(e))
        if self._display_timing: self.print_timing(add_total=True)

    def updateTransforms(self) -> float:
        if self.trace_calls:
            t = trace_method(self.tab)
        if self._display_timing: self.start_timing()
        # _gl = QtGui.QOpenGLContext.currentContext().functions()
        _gl = gl
        w = self._width
        h = self._height
        dx, dy = self.new_translation()
        scale = self.new_scale(-self.mouse_zoom_displ.y(), self.texture.height)

        # update the window size
        translation_unit = min(w, h)/2
        # use_glm = False
        m = glm.mat4()
        # the window corner OpenGL coordinates are (-+1, -+1)
        m = m*glm.transpose(glm.ortho(0., w, 0., h, -1., 1.))
        m = m*glm.transpose(glm.translate(glm.vec3(dx/translation_unit,dy/translation_unit,0)))
        m = m*glm.transpose(glm.scale(glm.vec3(scale,scale,scale)))
        self.pMatrix_glm = m
        self.mvMatrix_glm = glm.mat4()

        # For use in shaders
        self.mvMatrix = np.array(self.mvMatrix_glm, dtype=np.float32).flatten()
        self.pMatrix  = np.array(self.pMatrix_glm,  dtype=np.float32).flatten()

        if self._display_timing: self.print_timing(add_total=True)
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

    # def event(self, evt):
    #     print(f"event {evt.type()}")
    #     return super().event(evt)

    def get_gl_coordinates(self, x, y):
        viewport_glm = glm.vec4(0,0,int(self._width+0.5), int(self._height+0.5))
        pos_glm = glm.unProject(glm.vec3(x,y,0),
                                glm.transpose(self.mvMatrix_glm),
                                glm.transpose(self.pMatrix_glm),
                                viewport_glm)
        return pos_glm[0], pos_glm[1]

    # def paintEvent(self, event):
    #     # print("GLIVB paintEvent")
    #     if self._image is not None:
    #         self.paintAll()

    # def paintGL(self):
    #     self.paintAll()

    def mousePressEvent(self, event):
        self._mouse_events.mouse_press_event(event)

    def mouseMoveEvent(self, event):
        if self.show_cursor or self._show_overlap:
            self.set_cursor_image_position(event.x(), event.y())
        self._mouse_events.mouse_move_event(event)

    def mouseReleaseEvent(self, event):
        self._mouse_events.mouse_release_event(event)

    def mouseDoubleClickEvent(self, event):
        self._mouse_events.mouse_double_click_event(event)

    def wheelEvent(self, event):
        self._mouse_events.mouse_wheel_event(event)

    def keyPressEvent(self, event):
        x0, x1, y0, y1 = self.image_centered_position()
        self.key_press_event(event, wsize=QtCore.QSize(x1-x0, y1-y0))

    def _makeCurrent(self):
        # print(f"{self.isValid()=}")
        self.makeCurrent()
        # print(f" {self.context()=}")
        # if self.context():
        #     print(f"{self.context().isValid()=}")

    def resetTextureBuffers(self):
        self._makeCurrent()
        if self.isValid():
            if self.texture:
                self.texture.free_buffers()
            if self.texture_ref:
                self.texture_ref.free_buffers()

    def resizeEvent(self, event):
        """Called upon window resizing: reinitialize the viewport.
        """
        if self.trace_calls:
            t = trace_method(self.tab)
        self.print_log(f"resize {event.size()}  self {self.width()} {self.height()}")
        self.evt_width = event.size().width()
        self.evt_height = event.size().height()
        self._makeCurrent()
        if self.isValid():
            if self.texture:
                self.texture.resize_event()
            if self.texture_ref:
                self.texture_ref.resize_event()
        QOpenGLWidget.resizeEvent(self, event)
        self.setTexture()
        self.print_log(f"resize {event.size()}  self {self.width()} {self.height()}")

