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
    def __init__(self, multiview: 'MultiView') -> None:
        self._press_pos : Optional[QtCore.QPoint] = None
        self._delta     : QtCore.QPoint = QtCore.QPoint(0,0)
        self._multiview : 'MultiView'   = multiview

    def press(self, event : QtGui.QMouseEvent) -> None:
        """ Press event """
        self._press_pos = event.pos()

    def move(self, event: QtGui.QMoveEvent) -> None:
        """ Move event """
        self._delta = event.pos() - self._press_pos

    def release(self, event: QtGui.QMouseEvent) -> None:
        """ Release event """
        self._delta = QtCore.QPoint(0,0)



class MultiViewMouseEvents:
    """ Implement mouse events for MultiView """

    def __init__(self, multiview: 'MultiView'):
        self._multiview : 'MultiView' = multiview

        self._mouse_callback = {
            'Left Pressed'  : self.action_activate,
            'Left+DblClick' : self.toggle_show_single_viewer,
        }

    def _get_markdown_help(self) -> str:
        res = ''
        res += '|Mouse    |Action  |  \n'
        res += '|:--------|:------:|  \n'
        # TODO create html table
        for k,v in self._mouse_callback.items():
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

    def action_activate(self,  event: QtGui.QMouseEvent) -> bool:
        """ Set viewer active """
        for v in self._multiview.image_viewers:
            if v.geometry().contains(event.pos()):
                # Set the current viewer active before processing the double click event
                self._multiview.on_active(v)
                self._multiview.update_image()
                return True
        return False

    def toggle_show_single_viewer(self, event: QtGui.QMouseEvent)->bool:
        """ Show only the selected viewer or all viewers """
        # Need to find the viewer that has been double clicked
        for v in self._multiview.image_viewers:
            if v.geometry().contains(event.pos()):
                # Set the current viewer active before processing the double click event
                self._multiview.on_active(v)
                print("set_active_viewer")
                self._multiview._show_active_only = not self._multiview._show_active_only
                if not self._multiview._active_viewer:
                    return False
                # Set the current image
                self._multiview.output_label_reference_image = v.image_name
                # Update the image to show/hide viewers
                self._multiview.update_image()
                return True
        return False

    def mouse_press_event(self, event : QtGui.QMouseEvent):
        """ Build str that represents the event and call process_event() """
        event_repr : str = ''
        # 1. Get Modifiers
        event_repr = add2repr(event_repr, self.modifiers2str(event.modifiers()))
        # 2. Get Buttons
        event_repr = add2repr(event_repr, self.buttons2str(event.buttons()))
        press_event  = event_repr + ' Pressed'
        print(f'MultiView press_event = {press_event}')
        self._press_pos = event.pos()
        processed = self.process_event(press_event,  event)
        event.setAccepted(processed)

    def mouse_double_click_event(self, event: QtGui.QMouseEvent):
        """ Deal with double-click event """
        event_repr : str = ''
        # 1. Get Modifiers
        event_repr = add2repr(event_repr, self.modifiers2str(event.modifiers()))
        # 2. Get Buttons
        event_repr = add2repr(event_repr, self.buttons2str(event.buttons()))
        event_repr = add2repr(event_repr, 'DblClick')
        print(f"{event_repr}")
        processed = self.process_event(event_repr,  event)
        event.setAccepted(processed)

