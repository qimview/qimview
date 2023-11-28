""" 
    Base class for mouse events
"""

from abc import abstractmethod
from typing import Optional, Dict, Callable, Generic, TypeVar, Type
from qimview.utils.qt_imports import QtGui, QtCore, QtWidgets

def add2repr(res:str, elt:str) -> str:
    """ Add a substring to a string representing an event """
    if res=='':
        return elt
    return res+'+'+elt

# T is a type that inherits from QWidget
T = TypeVar('T')

class MouseMotionActions(Generic[T]):
    """ Base class to deal with Mouse Motion with button (Press + Move + Release) events """
    def __init__(self, widget: T) -> None:
        self._press_pos : Optional[QtCore.QPoint] = None
        self._delta     : QtCore.QPoint = QtCore.QPoint(0,0)
        self._widget    : T = widget

    def press(self, event : QtGui.QMouseEvent) -> None:
        """ Press event """
        self._press_pos = event.pos()

    def move(self, event: QtGui.QMoveEvent) -> None:
        """ Move event """
        if self._press_pos:
            self._delta = event.pos() - self._press_pos

    def release(self, event: QtGui.QMouseEvent) -> None:
        """ Release event """
        self._delta = QtCore.QPoint(0,0)

class MouseEvents(Generic[T]):
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
    def __init__(self, widget: T):
        self._widget : T = widget
        # Instance of motion action
        self._current_motion    : Optional[MouseMotionActions] = None

        # Set key events callbacks
        # Each event will be associate with a unique string
        self._mouse_callback : Dict[str, Callable             ] = {}
        self._motion_classes : Dict[str, Type[MouseMotionActions[T]]] = {}

        # Regions to process events differently
        # A function which returns a bool is given to check if the 
        # current mouse position is inside the region
        self._regions : Dict[str, Callable[[QtCore.QPoint], bool]] = {}

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

    def start_motion(self, event_repr: str, event : QtGui.QMouseEvent) -> bool:
        """ Instantiate motion actions """
        print(f"{event_repr}")
        if event_repr in self._motion_classes and self._current_motion is None:
            self._current_motion = self._motion_classes[event_repr](self._widget)
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
        # self._viewer.mouse_pos = event.pos()
        # We save the event type in a member variable to be able to process the release event
        if self._current_motion:
            self._current_motion.move(event)
        else:
            self.mouse_move_unpressed(event)

    @abstractmethod
    def mouse_move_unpressed(self, event: QtGui.QMoveEvent)->None:
        """ Actions while moving the mouse without pressing any button """

    def check_regions(self, pos: QtCore.QPoint )->str:
        """ Return region information for event representation string """
        res = ''
        for name,call in self._regions.items():
            if call(pos):
                res += f' on {name}'
        return res

    def mouse_double_click_event(self, event: QtGui.QMouseEvent):
        """ Deal with double-click event """
        event_repr : str = ''
        # 1. Get Modifiers
        event_repr = add2repr(event_repr, self.modifiers2str(event.modifiers()))
        # 2. Get Buttons
        event_repr = add2repr(event_repr, self.buttons2str(event.buttons()))
        event_repr = add2repr(event_repr, 'DblClick')
        # Check if double click is on any defined region
        event_repr += self.check_regions(event.pos())
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
