from Qt import QtWidgets, QtCore, QtGui
from image_viewers.ImageFilterParameters import ImageFilterParameters


class ImageFilterParametersGui:
    def __init__(self, parameters):
        """
        :param parameters: instance of ImageFilterParameters
        """
        self.params = parameters

    def add_blackpoint(self, layout, callback):
        """
        Black point adjustment
        :param layout:
        :param callback:
        :return:
        """
        black_level = self.params.black_level
        self.blackpoint_label = QtWidgets.QLabel("")
        layout.addWidget(self.blackpoint_label)
        self.blackpoint_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blackpoint_slider.setRange(black_level.range[0], black_level.range[1])
        self.blackpoint_slider.setValue(black_level.default_value)
        self.blackpoint_changed()
        self.blackpoint_slider.valueChanged.connect(lambda: self.blackpoint_changed(callback))
        layout.addWidget(self.blackpoint_slider)

    def blackpoint_reset(self):
        self.blackpoint_slider.setValue(self.params.black_level.default_value)

    def blackpoint_changed(self, callback=None):
        black_level = self.params.black_level
        black_level.value = int(self.blackpoint_slider.value())
        self.blackpoint_label.setText("Black {}".format(black_level.value))
        if callback is not None:
            callback()

    def add_whitepoint(self, layout, callback):
        """
        White point adjustment
        """
        white_level = self.params.white_level
        self.whitepoint_label = QtWidgets.QLabel("")
        layout.addWidget(self.whitepoint_label)
        self.whitepoint_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.whitepoint_slider.setRange(white_level.range[0], white_level.range[1])
        self.whitepoint_slider.setValue(white_level.default_value)
        self.whitepoint_changed()
        self.whitepoint_slider.valueChanged.connect(lambda: self.whitepoint_changed(callback))
        layout.addWidget(self.whitepoint_slider)

    def whitepoint_reset(self):
        self.whitepoint_slider.setValue(self.params.white_level.default_value)

    def whitepoint_changed(self, callback=None):
        white_level = self.params.white_level
        white_level.value = int(self.whitepoint_slider.value())
        self.whitepoint_label.setText("White {}".format(white_level.value))
        if callback is not None:
            callback()

    def add_gamma(self, layout, callback):
        """
        Gamma adjustment
        :param layout:
        :param callback:
        :return:
        """
        gamma = self.params.gamma
        self.gamma_label = QtWidgets.QLabel("Gamma 1.00")
        layout.addWidget(self.gamma_label)
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setRange(gamma.range[0], gamma.range[1])
        self.gamma_slider.setValue(gamma.default_value)
        self.gamma_slider.valueChanged.connect(lambda: self.gamma_changed(callback))
        layout.addWidget(self.gamma_slider)

    def gamma_reset(self):
        self.gamma_slider.setValue(self.params.gamma.default_value)

    def gamma_changed(self, callback):
        gamma = self.params.gamma
        gamma.value = int(self.gamma_slider.value())
        self.gamma_label.setText("Gamma  {:0.2f}".format(gamma.get_float()))
        callback()
