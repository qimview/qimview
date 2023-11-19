from qimview.utils.qt_imports import  QtWidgets
from typing import Optional

class FullScreenHelper:
    def __init__(self):
        # Variables used for fullscreen mode
        self._replacing_widget  : Optional[QtWidgets.QWidget] = None
        self._before_max_parent : Optional[QtWidgets.QWidget] = None

    def find_in_layout(self, layout: QtWidgets.QLayout, widget:QtWidgets.QWidget) -> Optional[QtWidgets.QLayout]:
        """ Search Recursivement in Layouts for the current widget
        Args:    layout (QtWidgets.QLayout): input layout for search
        Returns: layout containing the current widget or None if not found
        """
        if layout.indexOf(widget) != -1: return layout
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() == widget: return layout
            if (l := item.layout()) and (found:=self.find_in_layout(l, widget)): return l
        return None

    def enter_fullscreen(self, widget:QtWidgets.QWidget) -> bool:
        # if before_max_parent is not None, we are already in fullscreen mode
        if self._before_max_parent is not None: return False
        parent : Optional[QtWidgets.QWidget] = widget.parent()
        if parent is not None and (playout := parent.layout()) is not None:
            if self.find_in_layout(playout, widget):
                self._before_max_parent = parent
                self._replacing_widget = QtWidgets.QWidget(self._before_max_parent)
                playout.replaceWidget(widget, self._replacing_widget)
                # We need to go up from the parent widget to the main window to get its geometry
                # so that the fullscreen is display on the same monitor
                toplevel_parent : Optional[QtWidgets.QWidget] = widget.parentWidget()
                while toplevel_parent.parentWidget(): toplevel_parent = toplevel_parent.parentWidget()
                widget.setParent(None)
                if toplevel_parent: widget.setGeometry(toplevel_parent.geometry())
                widget.showFullScreen()
                return True
        return False

    def exit_fullscreen(self, widget:QtWidgets.QWidget) -> bool:
        # if before_max_parent is None, we are already not in fullscreen mode
        if self._before_max_parent is None or self._replacing_widget is None: return False
        parent = self._before_max_parent
        widget.setParent(parent)
        parent.layout().replaceWidget(self._replacing_widget, widget)
        self._replacing_widget   = None
        self._before_max_parent = None
        # self.resize(self.before_max_size)
        parent.update()
        widget.show()
        # Reset active main window 
        widget.activateWindow()
        widget.setFocus()
        return True
