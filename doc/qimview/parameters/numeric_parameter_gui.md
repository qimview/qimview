Module qimview.parameters.numeric_parameter_gui
===============================================

Classes
-------

`NumericParameterGui(name, param, callback, layout=None, parent_name='')`
:   For the moment, it can only be a slider with associated text
    
    __init__(self, orientation: PySide6.QtCore.Qt.Orientation, parent: Optional[PySide6.QtWidgets.QWidget] = None) -> None
    __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = None) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * PySide6.QtWidgets.QSlider
    * PySide6.QtWidgets.QAbstractSlider
    * PySide6.QtWidgets.QWidget
    * PySide6.QtCore.QObject
    * PySide6.QtGui.QPaintDevice
    * Shiboken.Object

    ### Class variables

    `staticMetaObject`
    :

    ### Methods

    `add_to_layout(self, layout)`
    :

    `changed(self, callback=None)`
    :

    `create(self, moved_callback=False)`
    :   create(self, arg__1: int = 0, initializeWindow: bool = True, destroyOldWindow: bool = True) -> None

    `event(self, evt)`
    :   event(self, event: PySide6.QtCore.QEvent) -> bool

    `mouseDoubleClickEvent(self, evt)`
    :   mouseDoubleClickEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `register_event_player(self, event_player)`
    :

    `reset(self)`
    :

    `set_event_recorder(self, evtrec)`
    :