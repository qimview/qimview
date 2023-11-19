from qimview.utils.qt_imports import QtGui, QtCore, QtWidgets
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .image_viewer import ImageViewer
QtKeys  = QtCore.Qt.Key
QtMouse = QtCore.Qt.MouseButton


class ImageViewerEvents:
    """ Implement events for ImageViewer
    """
    def __init__(self, viewer: 'ImageViewer'):
        self._viewer : 'ImageViewer' = viewer
        self._wsize : QtCore.QSize = viewer.widget.size()

    @staticmethod
    def get_key_seq(event : QtGui.QKeyEvent) -> QtGui.QKeySequence:
        """ Return a key sequence from a keyboard event
        """
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        QtMod = QtCore.Qt.KeyboardModifier
        # Python > 3.7 key order is maintained
        mod_str = { 
            QtMod.ShiftModifier   : 'Shift',
            QtMod.ControlModifier : 'Ctrl',
            QtMod.AltModifier     : 'Alt',
            QtMod.MetaModifier    : 'Meta',
        }
        # Compact line to create the contatenated string like 'Ctrl+Alt+'
        mod_seq_str = "".join([ f"{mod_str[m]}+" for  m in mod_str.keys() if modifiers & m])
        key_str = QtGui.QKeySequence(event.key()).toString()
        return QtGui.QKeySequence(mod_seq_str + key_str)
    
    def helpDialog(self) -> bool :
        """ Open Help Dialog with links to wiki help
        """
        import qimview
        mb = QtWidgets.QMessageBox(self._viewer.widget)
        mb.setWindowTitle(f"qimview {qimview.__version__}: ImageViewer help")
        mb.setTextFormat(QtCore.Qt.TextFormat.RichText)
        mb.setText(
                "<a href='https://github.com/qimview/qimview/wiki'>qimview</a><br>"
                "<a href='https://github.com/qimview/qimview/wiki/3.-Image-Viewers'>Image Viewer</a>")
        mb.exec()
        return True
    
    def toggleFullScreen(self) -> bool: 
        return self._viewer._fullscreen.toggle_fullscreen(self._viewer._widget)

    def exitFullScreen(self) -> bool: 
        return self._viewer._fullscreen.exit_fullscreen(self._viewer._widget)
    
    def updateAndAccept(self) -> bool:
        self._viewer.viewer_update()
        self._viewer.synchronize()
        return True
    
    def zoomUpperLeft(self) -> bool:
        self._viewer.current_dx =  self._wsize.width()/4
        self._viewer.current_dy = -self._wsize.height()/4
        self._viewer.current_scale = 2
        return self.updateAndAccept()

    def zoomLowerLeft(self)    -> bool:
        self._viewer.current_dx = self._wsize.width() / 4
        self._viewer.current_dy = self._wsize.height() / 4
        self._viewer.current_scale = 2
        return self.updateAndAccept()

    def zoomUpperRight(self)    -> bool:
        self._viewer.current_dx = -self._wsize.width()/4
        self._viewer.current_dy = -self._wsize.height()/4
        self._viewer.current_scale = 2
        return self.updateAndAccept()

    def zoomLowerRight(self)    -> bool:
        self._viewer.current_dx = -self._wsize.width() / 4
        self._viewer.current_dy =  self._wsize.height() / 4
        self._viewer.current_scale = 2
        return self.updateAndAccept()

    def unZoom(self)    -> bool:
        self._viewer.current_dx = 0
        self._viewer.current_dy = 0
        self._viewer.current_scale = 1
        return self.updateAndAccept()
    
    def toggleAntialiasing(self)->bool: 
        self._viewer.antialiasing   = not self._viewer.antialiasing
        return self.updateAndAccept()
    
    def toggleHistogram(self) ->bool:
        self._viewer.show_histogram = not self._viewer.show_histogram
        return self.updateAndAccept()
    
    def toggleOverlay(self) ->bool:
        self._viewer.show_overlay   = not self._viewer.show_overlay
        return self.updateAndAccept()
    
    def toggleCursor(self) ->bool:
        self._viewer.show_cursor    = not self._viewer.show_cursor
        return self.updateAndAccept()
    
    def toggleStats(self) ->bool:
        self._viewer.show_stats     = not self._viewer.show_stats
        return self.updateAndAccept()

    def toggleDifferences(self) ->bool: 
        self._viewer.show_image_differences = not self._viewer.show_image_differences
        return self.updateAndAccept()
    
    def toggleIntensityLine(self) ->bool: 
        self._viewer.show_intensity_line = not self._viewer.show_intensity_line
        return self.updateAndAccept()

    def key_press_event(self, event : QtGui.QKeyEvent, wsize : QtCore.QSize):
        self._wsize = wsize
        # print(f"ImageViewerEvents: key_press_event {event.key()}")
        if type(event) == QtGui.QKeyEvent:
            keys_callback = {
                'A'     : self.toggleAntialiasing,
                'C'     : self.toggleCursor,
                'D'     : self.toggleDifferences,
                'H'     : self.toggleHistogram,
                'I'     : self.toggleIntensityLine,
                'O'     : self.toggleOverlay,
                'S'     : self.toggleStats,
                'F1'    : self.helpDialog,
                'F11'   : self.toggleFullScreen,
                'Esc'   : self.exitFullScreen,
                'Alt+A' : self.zoomUpperLeft,
                'Alt+B' : self.zoomUpperRight,
                'Alt+C' : self.zoomLowerLeft,
                'Alt+D' : self.zoomLowerRight,
                'Alt+F' : self.unZoom,
            }

            key_seq : str = ImageViewerEvents.get_key_seq(event).toString()
            # print(f"key sequence = {key_seq}")
            if key_seq in keys_callback:
                event.setAccepted(keys_callback[key_seq]())
            else:
                event.ignore()
        else:
            event.ignore()
