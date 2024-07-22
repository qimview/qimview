from qimview.utils.qt_imports import QtGui, QtCore, QtWidgets
from typing import TYPE_CHECKING, List, Tuple, Callable
from qimview.utils.tab_dialog import TabDialog
if TYPE_CHECKING:
    from .image_viewer import (ImageViewer, OverlapMode)
QtKeys  = QtCore.Qt.Key
QtMouse = QtCore.Qt.MouseButton


class ImageViewerKeyEvents:

    plugins_key_events : dict[str,Callable] = {}

    """ Implement events for ImageViewer
    """
    def __init__(self, viewer: 'ImageViewer'):
        self._viewer : 'ImageViewer' = viewer
        # Set arbitrary size for initialization
        self._wsize : QtCore.QSize = QtCore.QSize(100,100)

        # Set key events callbacks
        self.keys_callback = {
                'A'        : self.toggleAntialiasing,
                'C'        : self.toggleCursor,
                'D'        : self.toggleDifferences,
                'H'        : self.toggleHistogram,
                'I'        : self.toggleIntensityLine,
                'O'        : self.toggleOverlap,
                'Alt+O'    : self.toggleOverlapMode,
                'S'        : self.toggleStats,
                'T'        : self.toggleText,
                'F1'       : self.helpDialog,
                'F11'      : self.toggleFullScreen,
                'Esc'      : self.exitFullScreen,
                'Alt+A'    : self.zoomUpperLeft,
                'Alt+B'    : self.zoomUpperRight,
                'Alt+C'    : self.zoomLowerLeft,
                'Alt+D'    : self.zoomLowerRight,
                'Alt+F'    : self.unZoom,
                'Ctrl+P'   : self.ImagePath2Clipboard,
                'Ctrl+B'   : self.copy2Clipboard,
                'Ctrl+S'   : self.saveImage,
                'Shift+P'  : self.syncPos,
        }
        for plg in self.plugins_key_events:
            self.keys_callback.update(
                {plg: lambda: self.plugins_key_events[plg][self]}
            )

        self._help_tabs  : List[Tuple[str,str]] = []
        self._help_links : str = ''

    def _get_markdown_help(self) -> str:
        res = ''
        res += '|Keys  |Action  |  \n'
        res += '|:-----|:------:|  \n'
        # TODO create html table
        for k,v in self.keys_callback.items():
            res += f'|{k}|{v.__doc__}|  \n'
        res += '  \n'
        return res
    
    def add_help_tab(self, title: str, text: str) -> None:
        """ Additional help to display
        Args:  help (str): help string in markdown format
        """
        self._help_tabs.append((title, text))

    def add_help_links(self, help_links: str) -> None:
        """ Additional help links to display
        Args:  help_links (str): help string in markdown format
        """
        self._help_links = help_links

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
        """ Open help dialog with links to wiki pages """
        import qimview
        help_win = TabDialog(self._viewer.widget, f"qimview {qimview.__version__}: ImageViewer help")
        # mb.setTextFormat(QtCore.Qt.TextFormat.RichText)
        help_win.add_markdown_tab("ImageViewer keys",
                                  self._get_markdown_help()
                                  )
        for (title,text) in self._help_tabs:
            help_win.add_markdown_tab(title, text)

        help_win.add_markdown_tab("Links",
                "### Links \n" +
                "[github: qimview](https://github.com/qimview/qimview/wiki)  \n" +
                "[wiki help: Image Viewer](https://github.com/qimview/qimview/wiki/3.-Image-Viewers)  \n" +
                self._help_links
                )
        help_win.show()
        return True
    
    def toggleFullScreen(self) -> bool:
        """ toggle fullscreen mode """
        return self._viewer._fullscreen.toggle_fullscreen(self._viewer._widget)

    def exitFullScreen(self) -> bool: 
        """ exit fullscreen mode """
        return self._viewer._fullscreen.exit_fullscreen(self._viewer._widget)
    
    def updateAndAccept(self) -> bool:
        self._viewer.viewer_update()
        self._viewer.synchronize()
        return True
    
    def syncPos(self) -> bool:
        """ Toggle position synchronization from other viewer on/off """
        self._viewer.synchronize_pos = not self._viewer.synchronize_pos
        return self.updateAndAccept()
 
    def zoomUpperLeft(self) -> bool:
        """ Display upper left quarter of the image """
        self._viewer.current_dx = int(self._wsize.width()/4+0.5)
        self._viewer.current_dy = -int(self._wsize.height()/4+0.5)
        self._viewer.current_scale = 2
        return self.updateAndAccept()

    def zoomLowerLeft(self)    -> bool:
        """ Display lower left quarter of the image """
        self._viewer.current_dx = int(self._wsize.width()/4+0.5)
        self._viewer.current_dy = int(self._wsize.height()/4+0.5)
        self._viewer.current_scale = 2
        return self.updateAndAccept()

    def zoomUpperRight(self)    -> bool:
        """ Display upper right quarter of the image """
        self._viewer.current_dx = -int(self._wsize.width()/4+0.5)
        self._viewer.current_dy = -int(self._wsize.height()/4+0.5)
        self._viewer.current_scale = 2
        return self.updateAndAccept()

    def zoomLowerRight(self)    -> bool:
        """ Display lower right quarter of the image """
        self._viewer.current_dx = -int(self._wsize.width()/4+0.5)
        self._viewer.current_dy =  int(self._wsize.height()/4+0.5)
        self._viewer.current_scale = 2
        return self.updateAndAccept()

    def unZoom(self)    -> bool:
        """ Display full image """
        self._viewer.current_dx = 0
        self._viewer.current_dy = 0
        self._viewer.current_scale = 1
        return self.updateAndAccept()
    
    def toggleAntialiasing(self)->bool: 
        """ Toggle anti-aliasing """
        self._viewer.antialiasing   = not self._viewer.antialiasing
        return self.updateAndAccept()
    
    def toggleHistogram(self) ->bool:
        """ Toggle histogram display """
        self._viewer.show_histogram = not self._viewer.show_histogram
        return self.updateAndAccept()
    
    def toggleOverlap(self) ->bool:
        """ Toggle overlap display """
        self._viewer._show_overlap = not self._viewer._show_overlap
        return self.updateAndAccept()
    
    def toggleOverlapMode(self) ->bool:
        """ Toggle overlap mode (Horizontal/Vertical) """
        from .image_viewer import OverlapMode

        self._viewer._overlap_mode = OverlapMode(( self._viewer._overlap_mode+1)%len(OverlapMode))
        return self.updateAndAccept()
    
    def toggleCursor(self) ->bool:
        """ Toggle cursor display """
        self._viewer.show_cursor = not self._viewer.show_cursor
        return self.updateAndAccept()
    
    def toggleStats(self) ->bool:
        """ Toggle stats display """
        self._viewer.show_stats = not self._viewer.show_stats
        return self.updateAndAccept()

    def toggleText(self) ->bool:
        """ Toggle text/info display """
        self._viewer.show_text = not self._viewer.show_text
        return self.updateAndAccept()

    def toggleDifferences(self) ->bool: 
        """ Toggle image difference with reference """
        self._viewer._show_image_differences = not self._viewer._show_image_differences
        return self.updateAndAccept()
    
    def toggleIntensityLine(self) ->bool: 
        """ Toggle horizontal intensity line display """
        self._viewer.show_intensity_line = not self._viewer.show_intensity_line
        return self.updateAndAccept()
    
    def ImagePath2Clipboard(self)->bool:
        """ Copy image full path to clipboard """
        clipboard = QtWidgets.QApplication.clipboard()
        im = self._viewer.get_image()
        if im and im.filename:
            clipboard.setText(im.filename)
            return True
        return False

    def copy2Clipboard(self)->bool:
        """ Copy current image to clipboard """
        clipboard = QtWidgets.QApplication.clipboard()
        self._viewer.set_clipboard(clipboard, True)
        self._viewer.widget.repaint()
        self._viewer.set_clipboard(None, False)
        print("Image saved to clipboard")
        return True

    def saveImage(self)->bool:
        """ Save current image to disk and clipboard """
        clipboard = QtWidgets.QApplication.clipboard()
        self._viewer.set_clipboard(clipboard, True)
        self._viewer.widget.repaint()
        self._viewer.set_clipboard(None, False)
        # Ask for input file
        filename = QtWidgets.QFileDialog.getSaveFileName(self._viewer._widget, "ImageViewer: Save current image", filter="Images (*.png *.xpm *.jpg)")
        print(f"filename {filename}")
        im = clipboard.image()
        if im.isNull():
            print("Image not found")
            return False
        else:
            return im.save(filename[0])

    def key_press_event(self, event : QtGui.QKeyEvent, wsize : QtCore.QSize):
        self._wsize = wsize
        # print(f"ImageViewerEvents: key_press_event {event.key()}")
        if type(event) == QtGui.QKeyEvent:

            key_seq : str = ImageViewerKeyEvents.get_key_seq(event).toString()
            # print(f"key sequence = {key_seq}")
            if key_seq in self.keys_callback:
                event.setAccepted(self.keys_callback[key_seq]())
            else:
                event.ignore()
        else:
            event.ignore()

def imageviewer_add_plugins():
    import configparser, sys, os
    config = configparser.ConfigParser()
    res = config.read([os.path.expanduser('~/.qimview.cfg')])
    if res:
        try:
            plugins = config['IMAGEVIEWER']['Plugins'].split(',')
        except Exception as e:
            print(f" ----- No imageviewer plugin in config file : {e}")
            return
        for plg in plugins:
            try:
                format_cfg = config[f'IMAGEVIEWER.{plg.upper()}']
                folder, module, keyevent = [format_cfg[s] for s in ('Folder','Module','KeyEvent')]
                folder = os.path.expanduser(folder)
                print(f' {plg} {folder, keyevent}')
                # TODO: avoid sys.path.append?
                sys.path.append(folder)
                import importlib
                keyevent_module = importlib.import_module(module)
                ImageViewerKeyEvents.plugins_key_events.update({keyevent:keyevent_module.Plugin.run})
            except Exception as e:
                print(f" ----- Failed to add support for {plg}: {e}")

imageviewer_add_plugins()

