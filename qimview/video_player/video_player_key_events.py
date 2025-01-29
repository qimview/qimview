from qimview.utils.qt_imports import QtGui, QtCore, QtWidgets
from typing import TYPE_CHECKING, List, Tuple
from qimview.utils.tab_dialog import TabDialog
if TYPE_CHECKING:
    from .video_player_base import (VideoPlayerBase)
QtKeys  = QtCore.Qt.Key
QtMouse = QtCore.Qt.MouseButton

from qimview.parameters.numeric_parameter              import NumericParameter
from qimview.parameters.numeric_parameter_gui          import NumericParameterGui


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
                'Shift+F'     : self.syncFilters,
                'Shift+S'     : self.setTimeShifts,
                'Alt+Shift+S' : self.setTimeShiftsDialog,
        }

        self._help_tabs  : List[Tuple[str,str]] = []
        self._help_links : str = ''
        self._debug : bool = False

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
        if self._debug:
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
    
    def syncFilters(self) -> bool:
        """ Toggle image filters synchronization from other viewer on/off """
        self._player.synchronize_filters = not self._player.synchronize_filters
        return self.updateAndAccept()

    def setTimeShifts(self) -> bool:
        """ Use current players time to set time shifts with respect to first video player """
        self._player.setTimeShifts()
        return self.updateAndAccept()

    def setTimeShiftsDialog(self) -> bool:
        """ Use dialog window to set time shifts with respect to first video player """
        class CustomDialog(QtWidgets.QDialog):
            def __init__(self,mess:str, time_shift:float):
                super().__init__()

                self.setWindowTitle("Video time shifts")

                QBtn = (
                    QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                )

                self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
                self.buttonBox.accepted.connect(self.accept)
                self.buttonBox.rejected.connect(self.reject)

                self._ver_layout = QtWidgets.QVBoxLayout()
                self._hor_layout = QtWidgets.QHBoxLayout()
                message = QtWidgets.QLabel(mess)
                self._ver_layout.addWidget(message)
                self._ver_layout.addLayout(self._hor_layout)
                self._ver_layout.addWidget(self.buttonBox)
                self.setLayout(self._ver_layout)

                self._add_shift_slider(self._hor_layout)
                self._shift.float = time_shift
                self._shift_gui.updateGui()
            
            def _add_shift_slider(self, hor_layout):
                # Position slider
                self._shift = NumericParameter(_range=[-3000,3000])
                self._shift.float_scale = 1000
                self._shift_gui = NumericParameterGui(name="sec:", param=self._shift)
                self._shift_gui.decimals = 3
                # self._shift_gui.set_pressed_callback(self.pause)
                # self._shift_gui.set_released_callback(self.reset_play)
                self._shift_gui.set_valuechanged_callback(self.slider_value_changed)
                self._shift_gui.create()
                self._shift_gui.add_to_layout(hor_layout,5)
            
            def slider_value_changed(self):
                print(f'{self._shift.float=}')
                
            def getShift(self) -> float:
                return self._shift.float

        # self._player.setTimeShi
        time_shifts = self._player.getTimeShifts()
        if len(time_shifts)>0:
            dlg = CustomDialog(f"Set time shifts video[n] - video[0]", time_shifts[0])
            if dlg.exec():
                print("Success!")
                time_shifts[0]=dlg.getShift()
            else:
                print("Cancel!") 
        return self.updateAndAccept()

    def updateAndAccept(self) -> bool:
        # self._player.player_update()
        # self._player.synchronize()
        return True
    
    def key_press_event(self, event : QtGui.QKeyEvent):
        if self._debug:
            print(f"VideoPlayerEvents: key_press_event {event.key()}")
        if type(event) == QtGui.QKeyEvent:

            key_seq : str = VideoPlayerKeyEvents.get_key_seq(event).toString()
            if self._debug:
                print(f"key sequence = {key_seq}")
            if key_seq in self.keys_callback:
                event.setAccepted(self.keys_callback[key_seq]())
            else:
                event.ignore()
        else:
            event.ignore()
