Module qimview.image_viewers.gl_image_viewer_base
=================================================

Classes
-------

`GLImageViewerBase(parent=None)`
:   QOpenGLWidget(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * PySide6.QtOpenGLWidgets.QOpenGLWidget
    * PySide6.QtWidgets.QWidget
    * PySide6.QtCore.QObject
    * PySide6.QtGui.QPaintDevice
    * Shiboken.Object
    * qimview.image_viewers.image_viewer.ImageViewer

    ### Descendants

    * qimview.image_viewers.gl_image_viewer.GLImageViewer
    * qimview.image_viewers.gl_image_viewer_shaders.GLImageViewerShaders

    ### Class variables

    `staticMetaObject`
    :

    ### Methods

    `get_mouse_gl_coordinates(self, x, y)`
    :

    `gl_draw_cursor(self) ‑> Optional[Tuple[int, int]]`
    :

    `image_centered_position(self)`
    :

    `keyPressEvent(self, event)`
    :   keyPressEvent(self, event: PySide6.QtGui.QKeyEvent) -> None

    `mouseDoubleClickEvent(self, event)`
    :   mouseDoubleClickEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `mouseMoveEvent(self, event)`
    :   mouseMoveEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `mousePressEvent(self, event)`
    :   mousePressEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `mouseReleaseEvent(self, event)`
    :   mouseReleaseEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `myPaintGL(self)`
    :

    `opengl_error(self, force=False)`
    :

    `paintAll(self)`
    :

    `resizeEvent(self, event)`
    :   Called upon window resizing: reinitialize the viewport.

    `resizeGL(self, width, height)`
    :   Called upon window resizing: reinitialize the viewport.

    `setTexture(self)`
    :   :return: set opengl texture based on input numpy array image

    `set_cursor_image_position(self, cursor_x, cursor_y)`
    :   Sets the image position from the cursor in proportion of the image dimension
        :return:

    `set_image(self, image)`
    :

    `synchronize_data(self, other_viewer)`
    :

    `updateTransforms(self) ‑> float`
    :

    `updateViewPort(self)`
    :

    `viewer_update(self)`
    :

    `wheelEvent(self, event)`
    :   wheelEvent(self, event: PySide6.QtGui.QWheelEvent) -> None