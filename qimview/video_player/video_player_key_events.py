from qimview.utils.qt_imports import QtGui, QtCore, QtWidgets
from typing import TYPE_CHECKING, List, Tuple
from qimview.utils.tab_dialog import TabDialog
if TYPE_CHECKING:
    from .video_player_base import (VideoPlayerBase)
QtKeys  = QtCore.Qt.Key
QtMouse = QtCore.Qt.MouseButton


class VideoPlayerKeyEvents:
    """ Implement events for VideoPlayer
    """
    def __init__(self, player: 'VideoPlayerBase'):
        self._player : 'VideoPlayerBase' = player
        # Set arbitrary size for initialization
        self._wsize : QtCore.QSize = QtCore.QSize(100,100)

        # Set key events callbacks
        self.keys_callback = {
                'F'           : self.toggleImageFilters,
                'Space'       : self.togglePlayPause,
                'Right'       : self.nextFrame,
                'Left'        : self.prevFrame,
                'Shift+Right' : self.nextTenFrame,
                'Shift+Left'  : self.prevTenFrame,
                'PgUp'        : self.nextSecond,
                'PgDown'      : self.prevSecond,
                'F1'          : self.helpDialog,
                'F10'         : self.toggleFullScreen,
                'Esc'         : self.exitFullScreen,
        }

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

    def markdown_help(self) -> str:
        return self._get_markdown_help()

    def help_links(self) -> str:
        return self._help_links

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
        help_win = TabDialog(self._player.widget, f"qimview {qimview.__version__}: VideoPlayer help")
        # mb.setTextFormat(QtCore.Qt.TextFormat.RichText)
        help_win.add_markdown_tab("VideoPlayer keys",
                                  self._get_markdown_help()
                                  )
        for (title,text) in self._help_tabs:
            help_win.add_markdown_tab(title, text)

        help_win.add_markdown_tab("Links",
                "### Links \n" +
                "[github: qimview](https://github.com/qimview/qimview/wiki)  \n" +
                self._help_links
                )
        help_win.show()
        return True
    
    def toggleImageFilters(self) -> bool:
        """ toggle image filters """
        self._player.filters_widget.setVisible(not self._player.filters_widget.isVisible())
        return True

    def togglePlayPause(self) -> bool:
        """ toggle play/pause mode """
        print("Play/Pause")
        self._player.play_pause()
        return True
    
    def nextFrame(self) -> bool:
        """ display next Frame """
        p = self._player
        p.play_position = p.play_position+p.frame_duration
        p.set_play_position()
        p.update_position(force=True)
        return True

    def prevFrame(self) -> bool:
        """ display previous Frame """
        p = self._player
        p.play_position = max(0,p.play_position-p.frame_duration)
        p.set_play_position()
        p.update_position(force=True)
        return True

    def nextTenFrame(self) -> bool:
        """ display Frame + 10 """
        p = self._player
        p.play_position = p.play_position+10*p.frame_duration
        p.set_play_position()
        p.update_position(force=True)
        return True

    def prevTenFrame(self) -> bool:
        """ display Frame - 10 """
        p = self._player
        p.play_position = max(0,p.play_position-10*p.frame_duration)
        p.set_play_position()
        p.update_position(force=True)
        return True

    def nextSecond(self) -> bool:
        """ display Frame at +1 second """
        p = self._player
        p.play_position = p.play_position+1
        p.set_play_position()
        p.update_position(force=True)
        return True

    def prevSecond(self) -> bool:
        """ display Frame at -1 second """
        p = self._player
        p.play_position = max(0,p.play_position-1)
        p.set_play_position()
        p.update_position(force=True)
        return True

    def toggleFullScreen(self) -> bool:
        """ toggle fullscreen mode """
        return self._player._fullscreen.toggle_fullscreen(self._player)

    def exitFullScreen(self) -> bool: 
        """ exit fullscreen mode """
        return self._player._fullscreen.exit_fullscreen(self._player)
    
    def updateAndAccept(self) -> bool:
        # self._player.player_update()
        # self._player.synchronize()
        return True
    
    def key_press_event(self, event : QtGui.QKeyEvent):
        print(f"VideoPlayerEvents: key_press_event {event.key()}")
        if type(event) == QtGui.QKeyEvent:

            key_seq : str = VideoPlayerKeyEvents.get_key_seq(event).toString()
            print(f"key sequence = {key_seq}")
            if key_seq in self.keys_callback:
                event.setAccepted(self.keys_callback[key_seq]())
            else:
                event.ignore()
        else:
            event.ignore()
