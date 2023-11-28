""" 
    Deal with MultiView mouse events
"""

from typing import NewType, TypeVar, TYPE_CHECKING
from qimview.utils.qt_imports import QtGui, QtCore
if TYPE_CHECKING:
    from qimview.image_viewers.multi_view import MultiView
from .mouse_events import MouseEvents, MouseMotionActions


# T is a type that inherits from QWidget

class MultiViewMouseEvents(MouseEvents['MultiView']):
    """ Implement mouse events for MultiView """

    def __init__(self, multiview: 'MultiView'):
        super().__init__(multiview)

        self._mouse_callback.update({
            'Left Pressed'  : self.action_activate,
            'Left+DblClick' : self.toggle_show_single_viewer,
        })


    def action_activate(self,  event: QtGui.QMouseEvent) -> bool:
        """ Set viewer active """
        for v in self._widget.image_viewers:
            if v.geometry().contains(event.pos()):
                # Set the current viewer active before processing the double click event
                self._widget.on_active(v)
                self._widget.update_image()
                return True
        return False

    def toggle_show_single_viewer(self, event: QtGui.QMouseEvent)->bool:
        """ Show only the selected viewer or all viewers """
        # Need to find the viewer that has been double clicked
        for v in self._widget.image_viewers:
            if v.geometry().contains(event.pos()):
                # Set the current viewer active before processing the double click event
                self._widget.on_active(v)
                print("set_active_viewer")
                self._widget._show_active_only = not self._widget._show_active_only
                if not self._widget._active_viewer:
                    return False
                # Set the current image
                self._widget.output_label_reference_image = v.image_name
                # Update the image to show/hide viewers
                self._widget.update_image()
                return True
        return False

    def mouse_move_unpressed(self, event: QtGui.QMoveEvent)->None:
        """ Actions while moving the mouse without pressing any button """
        event.ignore()
