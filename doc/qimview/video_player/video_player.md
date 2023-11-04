Module qimview.video_player.video_player
========================================

Classes
-------

`VideoPlayer(open_button=False)`
:   QWidget(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * PySide6.QtWidgets.QWidget
    * PySide6.QtCore.QObject
    * PySide6.QtGui.QPaintDevice
    * Shiboken.Object

    ### Class variables

    `staticMetaObject`
    :

    ### Methods

    `duration_changed(self, duration)`
    :

    `handle_errors(self)`
    :

    `init_ui(self)`
    :

    `mediastate_changed(self, state)`
    :

    `open_file(self)`
    :

    `position_changed(self, position)`
    :

    `set_play_position(self)`
    :

    `set_position(self, position)`
    :

    `set_synchronize(self, viewer)`
    :

    `set_video(self, filename)`
    :

    `synchronize_set_play_position(self, event_viewer)`
    :

    `synchronize_toggle_play(self, event_viewer)`
    :

    `toggle_play_video(self)`
    :

`myVideoWidget()`
:   QVideoWidget(self, parent: Optional[PySide6.QtWidgets.QWidget] = None) -> None
    
    __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = None) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * PySide6.QtMultimediaWidgets.QVideoWidget
    * PySide6.QtWidgets.QWidget
    * PySide6.QtCore.QObject
    * PySide6.QtGui.QPaintDevice
    * Shiboken.Object

    ### Class variables

    `staticMetaObject`
    :

    ### Methods

    `new_scale(self, mouse_zy, height)`
    :

    `resizeEvent(self, event)`
    :   Called upon window resizing: reinitialize the viewport.

    `wheelEvent(self, event)`
    :   wheelEvent(self, event: PySide6.QtGui.QWheelEvent) -> None