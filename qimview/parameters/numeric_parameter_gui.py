"""
    GUI for NumericParameter instances
"""

from typing import Callable
from qimview.utils.qt_imports import QtWidgets, QtCore


class NumericParameterGui(QtWidgets.QSlider):
    """
    For the moment, it can only be a slider with associated text
    """
    def __init__(self, name, param, callback = None, layout=None, parent_name=""):
        QtWidgets.QSlider.__init__(self, QtCore.Qt.Horizontal)
        self.name = name
        self.param = param
        self._valuechanged_callback : Callable | None = callback
        self._pressed_callback      : Callable | None = None
        self._moved_callback        : Callable | None = None
        self._released_callback     : Callable | None = None
        self.event_recorder = None
        self.parent_name = parent_name
        self.widget_name = f"slider_{self.parent_name}_{self.name}"
        self.created = False
        self.decimals = 2
        if layout is not None:
            self.create()
            self.add_to_layout(layout)

    def set_event_recorder(self, evtrec):
        self.event_recorder = evtrec
        if self.event_recorder is not None:
            self.event_recorder.register_widget(id(self), self.widget_name)

    def register_event_player(self, event_player):
        event_player.register_widget(self.widget_name, self)

    def set_valuechanged_callback(self,cb:Callable):
        self._valuechanged_callback = cb

    def set_pressed_callback(self,cb:Callable):
        self._pressed_callback = cb

    def set_moved_callback(self,cb:Callable):
        self._moved_callback = cb

    def set_released_callback(self,cb:Callable):
        self._released_callback = cb

    def create(self, moved_callback=False):
        self.label = QtWidgets.QLabel("")
        self.setRange(self.param.range[0], self.param.range[1])
        self.setValue(self.param.value)
        self.changed()
        if self._moved_callback: 
            self.sliderMoved.connect(lambda: self.changed(self._moved_callback))
        if self._valuechanged_callback:
            self.valueChanged.connect(lambda: self.changed(self._valuechanged_callback))
        if self._pressed_callback:
            self.sliderPressed.connect(self._pressed_callback)
        if self._released_callback:
            self.sliderReleased.connect(self._released_callback)
        self.created = True

    def add_to_layout(self, layout):
        if not self.created:
            self.create()
        layout.addWidget(self.label)
        layout.addWidget(self)

    def reset(self):
        self.setValue(self.param.default_value)

    def updateSlider(self):
        self.setValue(self.param.int)

    def updateText(self):
        self.label.setText(f"{self.name} {self.param.float:0.{self.decimals}f}")

    def updateGui(self):
        self.updateSlider()
        self.updateText()

    def changed(self, callback=None):
        self.param.int = int(self.value())
        self.updateText()
        if callback is not None:
            callback()

    def mouseDoubleClickEvent(self, evt):
        self.reset()

    def event(self, evt):
        if self.event_recorder is not None:
            if evt.spontaneous():
                self.event_recorder.store_event(self, evt)
        return QtWidgets.QSlider.event(self, evt)
