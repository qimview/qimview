from Qt import QtWidgets, QtCore, QtGui
from image_viewers.ImageFilterParameters import ImageFilterParameters
import types


class NumericParameterGui(QtWidgets.QSlider):
    """
    For the moment, it can only be a slider with associated text
    """
    def __init__(self, name, param, callback, layout=None, parent_name=""):
        QtWidgets.QSlider.__init__(self, QtCore.Qt.Horizontal)
        self.name = name
        self.param = param
        self.callback = callback
        self.event_recorder = None
        self.parent_name = parent_name
        self.widget_name = f"slider_{self.parent_name}_{self.name}"
        if layout is not None:
            self.create()
            self.add_to_layout(layout)

    def set_event_recorder(self, evtrec):
        self.event_recorder = evtrec
        if self.event_recorder is not None:
            self.event_recorder.register_widget(id(self), self.widget_name)

    def register_event_player(self, event_player):
        event_player.register_widget(self.widget_name, self)

    def create(self):
        self.label = QtWidgets.QLabel("")
        self.setRange(self.param.range[0], self.param.range[1])
        self.setValue(self.param.value)
        self.changed()
        self.valueChanged.connect(lambda: self.changed(self.callback))

    def add_to_layout(self, layout):
        layout.addWidget(self.label)
        layout.addWidget(self)

    def reset(self):
        self.setValue(self.param.default_value)

    def changed(self, callback=None):
        self.param.int = int(self.value())
        self.label.setText(f"{self.name} {self.param.float:0.2f}")
        if callback is not None:
            callback()

    def mouseDoubleClickEvent(self, evt):
        self.reset()

    def event(self, evt):
        if self.event_recorder is not None:
            if evt.spontaneous():
                self.event_recorder.store_event(self, evt)
        return QtWidgets.QSlider.event(self, evt)


class ImageFilterParametersGui:
    def __init__(self, parameters, name=""):
        """
        :param parameters: instance of ImageFilterParameters
        """
        self.params    = parameters
        self.bl_gui    = None
        self.wl_gui    = None
        self.gamma_gui = None
        self.g_r_gui   = None
        self.g_b_gui   = None
        self.event_recorder = None
        self.name           = name

    def set_event_recorder(self, evtrec):
        self.event_recorder = evtrec

    def add_blackpoint(self, layout, callback):
        self.bl_gui = NumericParameterGui("Black", self.params.black_level, callback, layout, self.name)
        self.bl_gui.set_event_recorder(self.event_recorder)

    def add_whitepoint(self, layout, callback):
        self.wl_gui = NumericParameterGui("White", self.params.white_level, callback, layout, self.name)
        self.wl_gui.set_event_recorder(self.event_recorder)

    def add_gamma(self, layout, callback):
        self.gamma_gui = NumericParameterGui("Gamma", self.params.gamma, callback, layout, self.name)
        self.gamma_gui.set_event_recorder(self.event_recorder)

    def add_g_r(self, layout, callback):
        self.g_r_gui = NumericParameterGui("G/R", self.params.g_r, callback, layout, self.name)
        self.g_r_gui.set_event_recorder(self.event_recorder)

    def add_g_b(self, layout, callback):
        self.g_b_gui = NumericParameterGui("G/B", self.params.g_b, callback, layout, self.name)
        self.g_b_gui.set_event_recorder(self.event_recorder)

    def register_event_player(self, event_player):
        for v in vars(self):
            if 'gui' in v:
                self.__dict__[v].register_event_player(event_player)

    def reset_all(self):
        for v in vars(self):
            if 'gui' in v:
                self.__dict__[v].reset()

