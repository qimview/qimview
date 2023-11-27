""" 
    Deal with ImageViewer mouse events
"""

from typing import Optional
from qimview.utils.qt_imports import QtGui, QtCore
QtKeys  = QtCore.Qt.Key
QtMouse = QtCore.Qt.MouseButton

def add2repr(res:str, elt:str) -> str:
    """ Add a substring to a string representing an event """
    if res=='':
        return elt
    return res+'+'+elt

class MouseMotionActions:
    """ Base class to deal with Mouse Motion with button (Press + Move + Release) events """
    def __init__(self, viewer: 'ImageViewer') -> None:
        self._press_pos : Optional[QtCore.QPoint] = None
        self._delta     : QtCore.QPoint = QtCore.QPoint(0,0)
        self._viewer    : 'ImageViewer'   = viewer

    def press(self, event : QtGui.QMouseEvent) -> None:
        """ Press event """
        self._press_pos = event.pos()

    def move(self, event: QtGui.QMoveEvent) -> None:
        """ Move event """
        self._delta = event.pos() - self._press_pos

    def release(self, event: QtGui.QMouseEvent) -> None:
        """ Release event """
        self._delta = QtCore.QPoint(0,0)


class MousePanActions(MouseMotionActions):
    """ Image panning """
    def press(self, event : QtGui.QMouseEvent) -> None:
        """ Press event """
        super().press(event)
        event.setAccepted(True)
    def move(self, event: QtGui.QMoveEvent) -> None:
        """ Move event """
        super().move(event)
        self._viewer.mouse_displ = self._delta
    def release(self, event: QtGui.QMouseEvent) -> None:
        """ Release event """
        self._viewer.current_dx, self._viewer.current_dy = self._viewer.check_translation()
        super().release(event)
        self._viewer.mouse_displ = self._delta

class MouseZoomActions(MouseMotionActions):
    """ Image Zooming """
    def press(self, event : QtGui.QMouseEvent) -> None:
        """ Press event """
        super().press(event)
        event.setAccepted(True)
    def move(self, event: QtGui.QMoveEvent) -> None:
        """ Move event """
        super().move(event)
        self._viewer.mouse_zoom_displ = self._delta
    def release(self, event: QtGui.QMouseEvent) -> None:
        """ Release event """
        if self._viewer.get_image():
            self._viewer.current_scale = self._viewer.new_scale(
                        -self._delta.y(),
                        self._viewer.get_image().data.shape[0]
                        )
        super().release(event)
        self._viewer.mouse_zoom_displ = self._delta

class ImageViewerMouseEvents:
    """ Implement events for ImageViewer

        Create a string representation of each event to associate a corresponding callback
        Modifier + Key + Mouse Event + Position
         - modifier:
           QMouseEvent.modifiers(): Qt::KeyboardModifiers
           QFlags<KeyboardModifier>
             Qt::NoModifier          empty string
             Qt::ShiftModifier       Shft
             Qt::ControlModifier     Ctrl
             Qt::AltModifier         Alt
             Qt::MetaModifier        Meta
             Qt::KeypadModifier      Keypad: unused
             Qt::GroupSwitchModifier unused
        - Mouse events
             MouseButtonPress        Press Left/Right/Middle
             MouseMove               Move
             MouseButtonRelease      Release
             MouseButtonDblClick     DblClick
             Wheel                   Wheel
               event is separated on Qt but processes here
        - Position
             Inside one of the available positions
    """
    def __init__(self, viewer: 'ImageViewer'):
        self._viewer : 'ImageViewer' = viewer
        self._press_pos = None # Last mouse position before mouse click

        # Instance of motion action
        self._current_motion    : Optional[MouseMotionActions] = None

        # Set key events callbacks
        # Each event will be associate with a unique string


        # wheel	zoom image or unzoom depending on the direction
        # move + left button	zoom
        # Alt + move + left button	pan
        # doubleClick on histogram	Switch histogram display size factor (x1,x2,x3)
        self._mouse_callback = {
            'Left+DblClick on histogram' : self.toogle_histo_size,
            'Wheel'                      : self.wheel_zoom,
        }

        self._motion_classes = {
            'Left Motion'     : MouseZoomActions,
            'Alt+Left Motion' : MousePanActions,
        }

    def _get_markdown_help(self) -> str:
        res = ''
        res += '|Mouse    |Action  |  \n'
        res += '|:--------|:------:|  \n'
        # TODO create html table
        for k,v in self._mouse_callback.items():
            res += f'|{k}|{v.__doc__}|  \n'
        for k,v in self._motion_classes.items():
            res += f'|{k}|{v.__doc__}|  \n'
        res += '  \n'
        return res

    def markdown_help(self) -> str:
        """ return events documentation in markdown format """
        return self._get_markdown_help()

    def modifiers2str(self, modifiers: QtCore.Qt.KeyboardModifiers ) -> str:
        """ Converts the modifiers to a string representation """
        res = ''
        qt_mod = QtCore.Qt.KeyboardModifiers
        mod_str = {
            qt_mod.ShiftModifier   : 'Shft',
            qt_mod.ControlModifier : 'Ctrl',
            qt_mod.AltModifier     : 'Alt',
            qt_mod.MetaModifier    : 'Meta',
        }
        for m,m_str in mod_str.items():
            if modifiers & m:
                res = add2repr(res, m_str)
        return res

    def buttons2str(self, buttons: QtGui.Qt.MouseButtons ) -> str:
        """ Converts the mouse buttons to a string representation """
        res = ''
        qt_but = QtGui.Qt.MouseButton
        button_str = {
            qt_but.LeftButton   : 'Left',
            qt_but.RightButton  : 'Right',
            qt_but.MiddleButton : 'Middle',
        }
        for b,b_str in button_str.items():
            if buttons & b:
                res = add2repr(res, b_str)
        return res

    def process_event(self, event_repr: str, event : QtCore.QEvent) -> bool:
        """ Process the event base on its representation as string and on the dict of callbacks """
        if event_repr in self._mouse_callback:
            return self._mouse_callback[event_repr](event)
        return False

    def start_motion(self, event_repr: str, event : QtCore.QEvent) -> bool:
        """ Instantiate motion actions """
        print(f"{event_repr}")
        if event_repr in self._motion_classes and self._current_motion is None:
            self._current_motion = self._motion_classes[event_repr](self._viewer)
            self._current_motion.press(event)
            print(f"started motion {event_repr}")
            return True
        return False

    def mouse_press_event(self, event : QtGui.QMouseEvent):
        """ Build str that represents the event and call process_event() """
        event_repr : str = ''
        # 1. Get Modifiers
        event_repr = add2repr(event_repr, self.modifiers2str(event.modifiers()))
        # 2. Get Buttons
        event_repr = add2repr(event_repr, self.buttons2str(event.buttons()))
        motion_event = event_repr + ' Motion'
        press_event  = event_repr + ' Pressed'
        print(f'ImageViewer press_event = {press_event}')
        self._press_pos = event.pos()
        processed = self.process_event(press_event,  event)
        started   = self.start_motion (motion_event, event)
        # Propagate event to parent anyway?
        print(f"accepted = {processed}")
        event.setAccepted(processed) # or started)

    def mouse_release_event(self, event: QtGui.QMouseEvent) -> None:
        """ Build str that represents the event and call process_event() """
        if self._current_motion:
            self._current_motion.release(event)
            self._current_motion = None
            event.accept()
        event.ignore()

    def mouse_move_event(self, event: QtGui.QMoveEvent):
        """ Build str that represents the event and call process_event() """
        self._viewer.mouse_pos = event.pos()
        # We save the event type in a member variable to be able to process the release event
        if self._current_motion:
            self._current_motion.move(event)
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

    def toogle_histo_size(self, _)->bool:
        """ Switch histogram scale from 1 to 3 """
        self._viewer._histo_scale = (self._viewer._histo_scale % 3) + 1 
        self._viewer.viewer_update()
        return True

    def mouse_double_click_event(self, event: QtGui.QMouseEvent):
        """ Deal with double-click event """
        event_repr : str = ''
        # 1. Get Modifiers
        event_repr = add2repr(event_repr, self.modifiers2str(event.modifiers()))
        # 2. Get Buttons
        event_repr = add2repr(event_repr, self.buttons2str(event.buttons()))
        event_repr = add2repr(event_repr, 'DblClick')
        # Check if double click is on histogram, if so, toggle histogram size
        if self._viewer._histo_rect and self._viewer._histo_rect.contains(event.x(), event.y()):
            event_repr += ' on histogram'
        print(f"{event_repr}")
        processed = self.process_event(event_repr,  event)
        event.setAccepted(processed)

    def mouse_wheel_event(self,event: QtGui.QWheelEvent) -> None:
        """ Build str that represents the wheek event and call process_event() """
        # Can add button states
        # 1. Get Modifiers
        event_repr = add2repr('', self.modifiers2str(event.modifiers()))
        event_repr = add2repr(event_repr, 'Wheel')
        print(f"{event_repr}")
        processed = self.process_event(event_repr,  event)
        event.setAccepted(processed)

    def wheel_zoom(self, event) -> bool:
        """ Zoom in/out based on wheel angle """
        # Zoom by applying a factor to the distances to the sides
        delta = event.angleDelta().y()
        # print("delta = {}".format(delta))
        coeff = delta/5
        # coeff = 20 if delta > 0 else -20
        if self._viewer.get_image():
            self._viewer.current_scale = self._viewer.new_scale(coeff, self._viewer.get_image().data.shape[0])
            self._viewer.viewer_update()
            self._viewer.synchronize()
            return True
        return False
