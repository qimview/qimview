
from qimview.utils.qt_imports import QtGui, QtCore, QtWidgets
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from .multi_view import MultiView
QtKeys  = QtCore.Qt.Key
QtMouse = QtCore.Qt.MouseButton


class MultiViewKeyEvents:
    """ Implement events for MultiView
    """
    def __init__(self, multiview: 'MultiView'):
        self._multiview : 'MultiView' = multiview
        # Set key events callbacks
        self.keys_callback = {
                'F1'             : self.helpDialog,
                'F5'             : self.reloadImages,
                'F10'            : self.toggleFullScreen,
                'Esc'            : self.exitFullScreen,
                'Up'             : self.upCallBack,
                'Down'           : self.downCallBack,
                'Left'           : self.selectPreviousImage,
                'Right'          : self.selectNextImage,
                'G'              : self.changeGridLayout,
        }
        # Use regular expression with 1 group that will be the argument of the callback
        # currently limited to function that take only one integer parameter to return a callback
        self.keys_callback_re = {
                'Alt\+([0-9])'  : self.setActiveViewerImage,
                'Ctrl\+([0-9])' : self.setReferenceImage,
                '([1-9])'       : self.setNumberOfViewers,
        }

        self._help_links : str = "[wiki help: Multi-image Viewer](https://github.com/qimview/qimview/wiki/4.-Multi%E2%80%90image-viewer)  \n"

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

    def _get_markdown_help(self) -> str:
        res = ''
        res += '|Keys  |Action  |  \n'
        res += '|:-----|:------:|  \n'
        # TODO create html table
        for k,v in self.keys_callback.items():
            res += f'|{k}|{v.__doc__}|  \n'
        for k,v in self.keys_callback_re.items():
            res += f'|{k}|{v.__doc__}|  \n'
        res += '  \n'
        return res
    
    def markdown_help(self) -> str:
        return self._get_markdown_help()

    def help_links(self) -> str:
        return self._help_links

    def helpDialog(self) -> bool :
        """ Open help dialog with links to wiki pages """
        import qimview
        mb = QtWidgets.QMessageBox(self._multiview)
        mb.setWindowTitle(f"qimview {qimview.__version__}: MultiView help")
        mb.setTextFormat(QtCore.Qt.TextFormat.MarkdownText)
        mb.setText(
                f"# qimview {qimview.__version__}  \n" +
                self._get_markdown_help() +
                "### Links \n" +
                self._help_links +
                "[github: qimview](https://github.com/qimview/qimview/wiki)  \n" +
                "[wiki help: Image Viewer](https://github.com/qimview/qimview/wiki/3.-Image-Viewers)  \n"
                )
        mb.exec()
        return True

    def reloadImages(self) -> bool:
        """ Reload images from disk """
        self._multiview.update_image(reload=True)
        return True
    
    def toggleFullScreen(self) -> bool:
        """ Toggle fullscreen """
        return self._multiview._fullscreen.toggle_fullscreen(self._multiview)
    
    def exitFullScreen(self) -> bool:
        """ Exit fullscreen """
        return self._multiview._fullscreen.exit_fullscreen (self._multiview)

    def upCallBack(self) -> bool:
        """ Call user-defined callback """
        if self._multiview.key_up_callback is not None:
            self._multiview.key_up_callback()
            return True
        return False 
    
    def downCallBack(self) -> bool:
        """ Call user-defined callback """
        if self._multiview.key_down_callback is not None:
            self._multiview.key_down_callback()
            return True
        return False 
    
    def setNumberOfViewers(self, n:int) -> Callable:
        """ Set the number N of displayed viewers """
        def func() -> bool:
            mv = self._multiview
            mv.set_number_of_viewers(n)
            if mv._active_viewer not in mv.image_viewers:
                mv._active_viewer = mv.image_viewers[n-1]
            mv.viewer_grid_layout.update()
            mv.update_image(mv._active_viewer.image_name)
            mv.setFocus()
            return True
        return func

    def setActiveViewerImage(self, n:int) -> Callable:
        """ Set the image N displayed by the active viewer """
        def func() -> bool:
            mv = self._multiview
            if mv.image_list[n] is None: return False
            if mv.output_label_current_image != mv.image_list[n]:
                mv.update_image(mv.image_list[n])
                mv.setFocus()
                return True
            return False
        return func

    def setReferenceImage(self, n:int) -> Callable:
        """ Set the reference image N """
        def func() -> bool:
            mv = self._multiview
            if mv.image_list[n] is None: return False
            if mv.output_label_current_image != mv.image_list[n]:
                mv.set_reference_label(mv.image_list[n], update_viewers=True)
                mv.update_image()
                return True
            return False
        return func

    def selectPreviousImage(self) -> bool:
        """ Display previous image on active viewer """
        try:
            mv = self._multiview
            current_pos = mv.image_list.index(mv.output_label_current_image)
            nb_images = len(mv.image_list)
            mv.update_image(mv.image_list[(current_pos+nb_images-1)%nb_images])
            return True
        except ValueError:
            return False

    def selectNextImage(self) -> bool:
        """ Display next image on active viewer """
        try:
            mv = self._multiview
            current_pos = mv.image_list.index(mv.output_label_current_image)
            nb_images = len(mv.image_list)
            mv.update_image(mv.image_list[(current_pos+1)%nb_images])
            return True
        except ValueError:
            return False

    def changeGridLayout(self) -> bool:
        """ Change grid layout """
        mv = self._multiview
        mv.max_columns = int ((mv.max_columns + 1) % mv.nb_viewers_used + 1)
        mv.set_number_of_viewers(mv.nb_viewers_used, max_columns=mv.max_columns)
        mv.update_image(reload=True)
        mv.setFocus()
        return True

    def key_press_event(self, event : QtGui.QKeyEvent):
        if type(event) == QtGui.QKeyEvent:
            key_seq : str = MultiViewKeyEvents.get_key_seq(event).toString()
            # print(f"key_seq {key_seq}")
            if key_seq in self.keys_callback:
                event.setAccepted(self.keys_callback[key_seq]())
            else:
                import re
                # try with regular expressions
                for str in self.keys_callback_re:
                    e = re.compile(str)
                    m = e.match(key_seq)
                    if m and len(m.groups()) == 1:
                        val = int(m.group(1))
                        event.setAccepted(self.keys_callback_re[str](val)())
                        return
                event.ignore()
        else:
            event.ignore()
