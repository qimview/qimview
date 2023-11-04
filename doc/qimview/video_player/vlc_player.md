Module qimview.video_player.vlc_player
======================================

Classes
-------

`VLCPlayer()`
:   A simple Media Player using VLC and Qt
        
    
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

    `OpenFile(self, filename=None)`
    :   Open a media file in a MediaPlayer

    `PlayPause(self)`
    :   Toggle play/pause status

    `Stop(self)`
    :   Stop player

    `createUI(self)`
    :   Set up the user interface, signals & slots

    `setVolume(self, Volume)`
    :   Set the volume

    `set_play_position(self)`
    :

    `set_synchronize(self, viewer)`
    :

    `set_video(self, filename=None)`
    :

    `synchronize_set_play_position(self, event_viewer)`
    :

    `synchronize_toggle_play(self, event_viewer)`
    :

    `updateUI(self)`
    :   updates the user interface

`myVideoWidget()`
:   QFrame(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * PySide6.QtWidgets.QFrame
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

    `set_media_player(self, media_player)`
    :

    `wheelEvent(self, event)`
    :   wheelEvent(self, event: PySide6.QtGui.QWheelEvent) -> None