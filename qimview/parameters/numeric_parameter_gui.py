"""
    GUI for NumericParameter instances
"""

from typing import Callable
from qimview.utils.qt_imports import QtWidgets, QtCore
from qimview.parameters.numeric_parameter import NumericParameter

class NumericParameterGui(QtWidgets.QSlider):
    """
    For the moment, it can only be a slider with associated text
    """
    def __init__(self, name, param, callback = None, layout=None, parent_name=""):
        QtWidgets.QSlider.__init__(self, QtCore.Qt.Horizontal)
        self.name = name
        self.param : NumericParameter = param
        self._valuechanged_callback : Callable | None = callback
        self._pressed_callback      : Callable | None = None
        self._moved_callback        : Callable | None = None
        self._released_callback     : Callable | None = None
        self.event_recorder = None
        self.parent_name = parent_name
        self.widget_name = f"slider_{self.parent_name}_{self.name}"
        self.created = False
        self.decimals = 2
        self._text_format : Callable | None = None
        # This flag also updating only the label position on during slider tracking
        self._tracking_textonly : bool = False
        if layout is not None:
            self.create()
            self.updateText()
            self.add_to_layout(layout)

    @property
    def tracking_textonly(self) -> bool:
        return self._tracking_textonly

    @tracking_textonly.setter
    def tracking_textonly(self, val:bool) -> None:
        self._tracking_textonly = val

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

    def create(self):
        self.label = QtWidgets.QLabel(f"{self.name}")
        self.setRange(self.param.range[0], self.param.range[1])
        self.setValue(self.param.value)
        self.changed()
        if self._moved_callback:
            # Move callback is used only if tracking is off, so set it to off here
            self.setTracking(False)
            self.sliderMoved.connect(self._moved_callback)
        if self._valuechanged_callback:
            self.valueChanged.connect(lambda: self.changed(self._valuechanged_callback))
        if self._pressed_callback:
            self.sliderPressed.connect(self._pressed_callback)
        if self._released_callback:
            self.sliderReleased.connect(self._released_callback)
        self.created = True

    def set_tooltip(self, mess):
        self.label.setToolTip(mess)

    def add_to_layout(self, layout, stretch=0):
        if not self.created:
            self.create()
        layout.addWidget(self.label)
        if stretch!=0:
            layout.addWidget(self, stretch)
        else:
            layout.addWidget(self)

    def reset(self):
        self.setValue(self.param.default_value)

    def updateSlider(self):
        self.setValue(self.param.int)

    def setTextFormat(self, _format: Callable):
        self._text_format = _format

    def updateText(self):
        if self._text_format:
            text = self._text_format(self.param)
        else:
            text = f"{self.param.float:0.{self.decimals}f}"
        self.label.setText(f"{self.name} {text}")

    def updateGui(self):
        self.updateSlider()
        self.updateText()

    def changed(self, callback=None):
        self.param.int = self.value()
        self.updateText()
        if callback is None or (self._tracking_textonly and self.isSliderDown()):
            return
        callback()

    def mouseDoubleClickEvent(self, evt):
        self.reset()

    def event(self, evt):
        if self.event_recorder is not None:
            if evt.spontaneous():
                self.event_recorder.store_event(self, evt)
        return QtWidgets.QSlider.event(self, evt)
