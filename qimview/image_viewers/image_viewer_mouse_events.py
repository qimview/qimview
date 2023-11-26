from qimview.utils.qt_imports import QtGui, QtCore, QtWidgets
from typing import TYPE_CHECKING, Optional
from enum import Enum, auto
#from .image_viewer import ImageViewer, MouseAction
QtKeys  = QtCore.Qt.Key
QtMouse = QtCore.Qt.MouseButton


class MouseAction(Enum):
    """ Different processed events from mouse

    Args:
        Enum (_type_): _description_
    """
    Pan      = auto()
    Zoom     = auto()
    Other    = auto()
    NoAction = auto()


class MousePressActions:
    def __init__(self) -> None:
        self._press_pos : Optional[QtCore.QPoint] = None
        pass
    def press(self, event : QtGui.QMouseEvent) -> None:
        self._press_pos = event.pos()
        pass
    def move(self, event: QtGui.QMoveEvent) -> None:
        displacement : QtCore.QPoint = event.pos() - self._press_pos
        pass
    def release(self, event: QtGui.QMouseEvent) -> None:
        pass

class ImageViewerMouseEvents:
    """ Implement events for ImageViewer
    """
    def __init__(self, viewer: 'ImageViewer'):
        self._viewer : 'ImageViewer' = viewer
        self._press_pos = None # Last mouse position before mouse click
        # Current mouse event
        self._mouse_action     : MouseAction = MouseAction.NoAction
        self._mouse_displ      : QtCore.QPoint = QtCore.QPoint(0,0)
        self._mouse_pos        : QtCore.QPoint = QtCore.QPoint(0,0)
        self._mouse_zoom_displ : QtCore.QPoint = QtCore.QPoint(0,0)

        # Set key events callbacks
        # Each event will be associate with a unique string

        # Modifier + Key + Mouse Event + Position
        #  - modifier:
        #    QMouseEvent.modifiers(): Qt::KeyboardModifiers
        #    QFlags<KeyboardModifier>
        #      Qt::NoModifier          empty string
        #      Qt::ShiftModifier       Shft
        #      Qt::ControlModifier     Ctrl
        #      Qt::AltModifier         Alt
        #      Qt::MetaModifier        Meta
        #      Qt::KeypadModifier      Keypad: unused
        #      Qt::GroupSwitchModifier unused
        # - Mouse events
        #      MouseButtonPress
        #      MouseMove
        #      MouseButtonRelease
        #      MouseButtonDblClick
        #      Wheel event is separated on Qt but processes here
        # - Position
        #      Inside one of the available positions

        # wheel	zoom image or unzoom depending on the direction
        # move + left button	zoom
        # Alt + move + left button	pan
        # doubleClick on histogram	Switch histogram display size factor (x1,x2,x3)
        self.mouse_callback = {
        }

    def mouse_press_event(self, event):
        self._press_pos = event.pos()
        if event.buttons() & QtMouse.RightButton:
            event.accept()
            return
        # Else set current viewer active
        self._viewer.activate()
        self._viewer.viewer_update()
        event.accept()

    def _get_mouse_action(self,event: QtGui.QMouseEvent) -> MouseAction:
        is_alt = event.modifiers() == QtCore.Qt.KeyboardModifier.AltModifier
        left_button = event.buttons() & QtMouse.LeftButton
        if left_button:
            if is_alt: 
                return MouseAction.Pan
            else:
                return MouseAction.Zoom
        return MouseAction.Other

    def _pan_update(self, event : QtGui.QMouseEvent):
        self._mouse_displ = event.pos() - self._press_pos

    def _pan_end(self, event):
        self._viewer.current_dx, self._viewer.current_dy = self._viewer.check_translation()
        self._mouse_displ = QtCore.QPoint(0,0)

    def _zoom_update(self, event):
        self._mouse_zoom_displ = event.pos() - self._press_pos

    def _zoom_end(self, event):
        if self._viewer._image is not None:
            self._viewer.current_scale = self._viewer.new_scale(-self._mouse_zoom_displ.y(), self._viewer._image.data.shape[0])
        self._mouse_zoom_displ = QtCore.QPoint(0,0)

    def mouse_move_event(self, event: QtGui.QMoveEvent):
        self._mouse_pos = event.pos()
        # We save the event type in a member variable to be able to process the release event
        self._mouse_action = self._get_mouse_action(event)
        event_cb = {
            MouseAction.Pan  : self._pan_update,
            MouseAction.Zoom : self._zoom_update,
        }
        if self._mouse_action in event_cb:
            event_cb[self._mouse_action](event)
            self._viewer.viewer_update()
            self._viewer.synchronize()
            event.accept()
        else:
            if self._viewer.show_overlay:
                self._viewer.viewer_update()
                event.accept()
            elif self._viewer.show_cursor:
                self._viewer.viewer_update()
                self._viewer.synchronize()
                event.accept()

    def mouse_release_event(self, event):
        event_cb = {
            MouseAction.Pan  : self._pan_end,
            MouseAction.Zoom : self._zoom_end,
        }
        if self._mouse_action in event_cb:
            event_cb[self._mouse_action](event)
            event.accept()
        self._viewer.synchronize()
        self._mouse_action = MouseAction.NoAction

    def mouse_double_click_event(self, event):
        self._viewer.print_log("double click ")
        # Check if double click is on histogram, if so, toggle histogram size
        if self._viewer._histo_rect and self._viewer._histo_rect.contains(event.x(), event.y()):
            # scale loops from 1 to 3 
            self._viewer._histo_scale = (self._viewer._histo_scale % 3) + 1 
            self._viewer.viewer_update()
            event.accept()
        else:
            event.setAccepted(False)

    def mouse_wheel_event(self,event):
        # Zoom by applying a factor to the distances to the sides
        if hasattr(event, 'delta'):
            delta = event.delta()
        else:
            delta = event.angleDelta().y()
        # print("delta = {}".format(delta))
        coeff = delta/5
        # coeff = 20 if delta > 0 else -20
        if self._viewer._image:
            self._viewer.current_scale = self._viewer.new_scale(coeff, self._viewer._image.data.shape[0])
            self._viewer.viewer_update()
            self._viewer.synchronize()
