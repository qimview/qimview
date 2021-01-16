
from Qt import QtWidgets, QtCore, QtGui
from image_viewers.glImageViewer import glImageViewer
from image_viewers.pyQtGraphImageViewer import pyQtGraphImageViewer
#import glImageViewerWithShaders
from image_viewers.glImageViewerWithShaders_qglw import glImageViewerWithShaders_qglw
from image_viewers.qtImageViewer import qtImageViewer

from enum import Enum

from image_viewers.ImageFilterParameters import ImageFilterParameters
from image_viewers.ImageFilterParametersGui import ImageFilterParametersGui


class ViewerType(Enum):
    QT_VIEWER = 1
    OPENGL_VIEWER = 2
    OPENGL_SHADERS_VIEWER = 3
    PYQTGRAPH_VIEWER = 4


class MultiView(QtWidgets.Widget):

    def __init__(self, parent=None, viewer_mode=ViewerType.QT_VIEWER, nb_viewers=1):
        """
        :param parent:
        :param viewer_mode:
        :param nb_viewers_used:
        """
        QtWidgets.QWidget.__init__(self, parent)

        self.use_pyqtgraph = viewer_mode == ViewerType.PYQTGRAPH_VIEWER
        self.use_opengl = viewer_mode in [ViewerType.OPENGL_SHADERS_VIEWER, ViewerType.OPENGL_VIEWER]

        self.nb_viewers_used = nb_viewers
        self.image_viewers = []
        self.image_viewer_class = {
            ViewerType.QT_VIEWER:             qtImageViewer,
            ViewerType.OPENGL_VIEWER:         glImageViewer,
            ViewerType.OPENGL_SHADERS_VIEWER: glImageViewerWithShaders_qglw,
            ViewerType.PYQTGRAPH_VIEWER:      pyQtGraphImageViewer
        }[viewer_mode]

        self.viewer_mode = viewer_mode

        self.filter_params = ImageFilterParameters()
        self.filter_params_gui = ImageFilterParametersGui(self.filter_params)


    def layout_parameters(self, parameters_layout):
        # Add color difference slider
        self.display_profiles = QtWidgets.QCheckBox("Profiles")
        self.display_profiles.stateChanged.connect(self.toggle_display_profiles)
        self.display_profiles.setChecked(False)
        parameters_layout.addWidget(self.display_profiles)
        self.keep_zoom = QtWidgets.QCheckBox("Keep zoom")
        self.keep_zoom.setChecked(False)
        parameters_layout.addWidget(self.keep_zoom)

        # Reset button
        self.reset_button = QtWidgets.QPushButton("reset")
        parameters_layout.addWidget(self.reset_button)
        self.reset_button.clicked.connect(self.reset_intensities)

        # Add color difference slider
        self.diff_color_label = QtWidgets.QLabel("Color diff. factor")
        parameters_layout.addWidget(self.diff_color_label)
        self.diff_color_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.diff_color_slider.setRange(1, 10)
        self.diff_color_slider.setValue(3)
        parameters_layout.addWidget(self.diff_color_slider)

        # Add saturation slider
        self.saturation_default = 50
        self.saturation_label = QtWidgets.QLabel("Saturation")
        parameters_layout.addWidget(self.saturation_label)
        self.saturation_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.saturation_slider.setRange(1, 150)
        self.saturation_slider.setValue(self.saturation_default)
        self.saturation_slider.setToolTip("{}".format(self.saturation_default))
        self.saturation_slider.valueChanged.connect(self.update_image_intensity_event)
        parameters_layout.addWidget(self.saturation_slider)
        # --- Black point adjustment
        self.filter_params_gui.add_blackpoint(parameters_layout, self.update_image_intensity_event)
        # --- white point adjustment
        self.filter_params_gui.add_whitepoint(parameters_layout, self.update_image_intensity_event)
        # --- Gamma adjustment
        self.filter_params_gui.add_gamma(parameters_layout, self.update_image_intensity_event)

    def layout_parameters_2(self, parameters2_layout):
        # --- G_R adjustment
        self.filter_params_gui.add_g_r(parameters2_layout, self.update_image_intensity_event)
        # --- G_B adjustment
        self.filter_params_gui.add_g_b(parameters2_layout, self.update_image_intensity_event)

    def update_layout(self):
        print("update_layout")
        vertical_layout = QtWidgets.QVBoxLayout()
        self.layout_buttons(vertical_layout)

        # First line of parameter control
        parameters_layout = QtWidgets.QHBoxLayout()
        self.layout_parameters(parameters_layout)
        vertical_layout.addLayout(parameters_layout, 1)

        # Second line of parameter control
        parameters2_layout = QtWidgets.QHBoxLayout()
        self.layout_parameters_2(parameters2_layout)
        vertical_layout.addLayout(parameters2_layout, 1)

        self.viewer_grid_layout = QtWidgets.QGridLayout()
        self.update_viewer_layout()
        vertical_layout.addLayout(self.viewer_grid_layout, 1)

        self.figures_widget = QtWidgets.QWidget()
        self.figures_layout = QtWidgets.QHBoxLayout()
        self.figures_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.figures_layout.addWidget(self.value_in_range_canvas)
        self.figures_widget.setLayout(self.figures_layout)

        vertical_layout.addWidget(self.figures_widget)
        self.toggle_display_profiles()
        self.setLayout(vertical_layout)
        print("update_layout done")
