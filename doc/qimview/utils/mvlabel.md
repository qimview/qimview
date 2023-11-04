Module qimview.utils.mvlabel
============================

Classes
-------

`MVLabel(text, parent=None)`
:   This Class is a standard QLabel with the simple and double click mouse events
    created for MultiView class
    
    __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    __init__(self, text: str, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * PySide6.QtWidgets.QLabel
    * PySide6.QtWidgets.QFrame
    * PySide6.QtWidgets.QWidget
    * PySide6.QtCore.QObject
    * PySide6.QtGui.QPaintDevice
    * Shiboken.Object

    ### Class variables

    `staticMetaObject`
    :

    ### Methods

    `mouseDoubleClickEvent(self, event)`
    :   mouseDoubleClickEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `mousePressEvent(self, event)`
    :   mousePressEvent(self, ev: PySide6.QtGui.QMouseEvent) -> None

    `mouseReleaseEvent(self, event)`
    :   mouseReleaseEvent(self, ev: PySide6.QtGui.QMouseEvent) -> None

    `performSingleClickAction(self)`
    :