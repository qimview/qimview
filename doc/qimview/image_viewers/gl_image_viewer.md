Module qimview.image_viewers.gl_image_viewer
============================================

Classes
-------

`GLImageViewer(parent=None, event_recorder=None)`
:   QOpenGLWidget(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * qimview.image_viewers.gl_image_viewer_base.GLImageViewerBase
    * PySide6.QtOpenGLWidgets.QOpenGLWidget
    * PySide6.QtWidgets.QWidget
    * PySide6.QtCore.QObject
    * PySide6.QtGui.QPaintDevice
    * Shiboken.Object
    * qimview.image_viewers.image_viewer.ImageViewer

    ### Class variables

    `staticMetaObject`
    :

    ### Methods

    `event(self, evt)`
    :   event(self, e: PySide6.QtCore.QEvent) -> bool

    `initializeGL(self)`
    :   Initialize OpenGL, VBOs, upload data on the GPU, etc.

    `myPaintGL(self)`
    :   Paint the scene.

    `paintGL(self)`
    :   paintGL(self) -> None

    `viewer_update(self)`
    :