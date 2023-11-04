Module qimview.image_viewers.qt_image_viewer
============================================

Classes
-------

`QTImageViewer(parent=None, event_recorder=None)`
:   QWidget(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * PySide6.QtWidgets.QWidget
    * PySide6.QtCore.QObject
    * PySide6.QtGui.QPaintDevice
    * Shiboken.Object
    * qimview.image_viewers.image_viewer.ImageViewer

    ### Class variables

    `staticMetaObject`
    :

    ### Methods

    `apply_filters(self, current_image)`
    :

    `apply_translation(self, crop)`
    :   :param crop:
        :return: the new crop

    `apply_zoom(self, crop)`
    :

    `check_translation(self)`
    :   This method computes the translation really applied based on the current requested translation
        :return:

    `draw_cursor(self, cropped_image_shape, crop_xmin, crop_ymin, rect, painter) ‑> Optional[Tuple[int, int]]`
    :   :param cropped_image_shape: dimensions of current crop
        :param crop_xmin: left pixel of current crop
        :param crop_ymin: top pixel of current crop
        :param rect: displayed image area
        :param painter:
        :return:
            tuple: (posx, posy) image pixel position of the cursor, if None cursor is out of image

    `draw_overlay_separation(self, cropped_image_shape, rect, painter)`
    :

    `event(self, evt)`
    :   event(self, event: PySide6.QtCore.QEvent) -> bool

    `get_difference_image(self, verbose=True)`
    :

    `keyPressEvent(self, event)`
    :   keyPressEvent(self, event: PySide6.QtGui.QKeyEvent) -> None

    `keyReleaseEvent(self, evt)`
    :   keyReleaseEvent(self, event: PySide6.QtGui.QKeyEvent) -> None

    `mouseDoubleClickEvent(self, event)`
    :   mouseDoubleClickEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `mouseMoveEvent(self, event)`
    :   mouseMoveEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `mousePressEvent(self, event)`
    :   mousePressEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `mouseReleaseEvent(self, event)`
    :   mouseReleaseEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `paintEvent(self, event)`
    :   paintEvent(self, event: PySide6.QtGui.QPaintEvent) -> None

    `paint_image(self)`
    :

    `resizeEvent(self, event)`
    :   Called upon window resizing: reinitialize the viewport.

    `set_image(self, image)`
    :

    `show(self)`
    :   show(self) -> None

    `update_crop(self)`
    :

    `update_crop_new(self)`
    :

    `viewer_update(self)`
    :

    `wheelEvent(self, event)`
    :   wheelEvent(self, event: PySide6.QtGui.QWheelEvent) -> None