
from qimview.utils.qt_imports import *
from qimview.parameters.numeric_parameter_gui import NumericParameterGui

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
        self.saturation_gui    = None
        self.imdiff_factor_gui = None
        self.event_recorder    = None
        self.name              = name

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

    def add_saturation(self, layout, callback):
        self.saturation_gui = NumericParameterGui("Sat", self.params.saturation, callback, layout, self.name)
        self.saturation_gui.set_tooltip("Image Saturation")
        self.saturation_gui.set_event_recorder(self.event_recorder)

    def add_imdiff_factor(self, layout, callback):
        self.imdiff_factor_gui = NumericParameterGui("Imdiff x", self.params.imdiff_factor, callback, layout, self.name)
        self.imdiff_factor_gui.set_tooltip("Image differences display factor")
        self.imdiff_factor_gui.set_event_recorder(self.event_recorder)

    def register_event_player(self, event_player):
        for v in vars(self):
            if 'gui' in v and self.__dict__[v]:
                self.__dict__[v].register_event_player(event_player)

    def reset_all(self):
        for v in vars(self):
            if 'gui' in v:
                self.__dict__[v].reset()

    def updateGui(self):
        # Call updateGui for all members that contain this method
        for v in vars(self):
          if ug := getattr(self.__dict__[v],'updateGui',False):
              if callable(ug):
                  ug()

