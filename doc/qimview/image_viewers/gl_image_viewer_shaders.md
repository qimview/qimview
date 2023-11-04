Module qimview.image_viewers.gl_image_viewer_shaders
====================================================

Classes
-------

`GLImageViewerShaders(parent=None)`
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

    `fragmentShader_RAW`
    :

    `fragmentShader_RGB`
    :

    `staticMetaObject`
    :

    `vertexShader`
    :

    ### Methods

    `initializeGL(self)`
    :   Initialize OpenGL, VBOs, upload data on the GPU, etc.

    `myPaintGL(self)`
    :   Paint the scene.

    `paintGL(self)`
    :   paintGL(self) -> None

    `setBufferData(self)`
    :

    `setVerticesBufferData(self)`
    :

    `set_shaders(self)`
    :

    `updateTransforms(self) ‑> float`
    :

    `viewer_update(self)`
    :