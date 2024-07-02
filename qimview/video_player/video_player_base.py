from typing import Union, Generator, List, Optional, Iterator, NewType

from qimview.utils.qt_imports                          import QtWidgets, QtCore, QtGui
from qimview.image_viewers.qt_image_viewer             import QTImageViewer
from qimview.image_viewers.gl_image_viewer_shaders     import GLImageViewerShaders
from qimview.parameters.numeric_parameter              import NumericParameter
from qimview.parameters.numeric_parameter_gui          import NumericParameterGui
from qimview.image_viewers.image_filter_parameters     import ImageFilterParameters
from qimview.image_viewers.image_filter_parameters_gui import ImageFilterParametersGui
from qimview.image_viewers.image_viewer                import ImageViewer

# Class that derives from ImageViewer
ImageViewerClass = NewType('ImageViewerClass', ImageViewer)

class VideoPlayerBase(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._vertical_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._vertical_layout)
        # self.viewer_class = QTImageViewer
        self.viewer_class = GLImageViewerShaders
        self.widget: Union[GLImageViewerShaders, QTImageViewer]
        self.widget = self.viewer_class() # event_recorder = self.event_recorder)
        self.widget.set_synchronization_callback(self.on_synchronize)
        # don't show the histogram
        self.widget.show_histogram = False
        self.widget._show_text = False
        self.setGeometry(0, 0, self.widget.width(), self.widget.height())

        self._filters_widget = self._add_filters()
        hor_layout = QtWidgets.QHBoxLayout()
        self._add_play_pause_button(       hor_layout)
        self._add_playback_speed_slider(   hor_layout)
        self._add_playback_position_slider(hor_layout)
        self._vertical_layout.addWidget(self._filters_widget)
        self._vertical_layout.addWidget(self.widget, stretch=1)
        self._vertical_layout.addLayout(hor_layout)
    
    def _add_filters(self) -> QtWidgets.QWidget:
        self.filter_params = ImageFilterParameters()
        self.filter_params_gui = ImageFilterParametersGui(self.filter_params, name="TestViewer")

        filters_widget = QtWidgets.QWidget()
        filters_layout = QtWidgets.QHBoxLayout()
        filters_widget.setLayout(filters_layout)
        # Add color difference slider
        self.filter_params_gui.add_imdiff_factor(filters_layout, self.update_image_intensity_event)

        self.filter_params_gui.add_blackpoint(filters_layout, self.update_image_intensity_event)
        # white point adjustment
        self.filter_params_gui.add_whitepoint(filters_layout, self.update_image_intensity_event)
        # Gamma adjustment
        self.filter_params_gui.add_gamma(filters_layout, self.update_image_intensity_event)
        # G_R adjustment
        self.filter_params_gui.add_g_r(filters_layout, self.update_image_intensity_event)
        # G_B adjustment
        self.filter_params_gui.add_g_b(filters_layout, self.update_image_intensity_event)

        return filters_widget

    def update_image_intensity_event(self):
        self.widget.filter_params.copy_from(self.filter_params)
        # print(f"parameters {self.filter_params}")
        self.on_synchronize(self.widget)
        self.widget.viewer_update()

    def _add_play_pause_button(self, hor_layout):
        self._button_play_pause = QtWidgets.QPushButton()
        self._icon_play = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        self._icon_pause = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause)
        self._button_play_pause.setIcon(self._icon_play)
        hor_layout.addWidget(self._button_play_pause)

    def _add_playback_speed_slider(self, hor_layout):
        # Playback speed slider
        self.playback_speed = NumericParameter()
        self.playback_speed.float_scale = 100
        self.playback_speed_gui = NumericParameterGui(name="x", param=self.playback_speed)
        self.playback_speed_gui.decimals = 1
        self.playback_speed_gui.set_pressed_callback(self.pause)
        self.playback_speed_gui.set_released_callback(self.reset_play)
        self.playback_speed_gui.set_valuechanged_callback(self.speed_value_changed)
        self.playback_speed_gui.create()
        self.playback_speed_gui.setTextFormat(lambda p: f"{pow(2,p.float):0.2f}")
        self.playback_speed_gui.setRange(-300, 300)
        self.playback_speed_gui.update()
        self.playback_speed_gui.updateText()
        self.playback_speed_gui.add_to_layout(hor_layout,1)
        self.playback_speed_gui.setSingleStep(1)
        self.playback_speed_gui.setPageStep(10)
        self.playback_speed_gui.setTickInterval(10)
        self.playback_speed_gui.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)

    def _add_playback_position_slider(self, hor_layout):
        # Position slider
        self._play_position = NumericParameter()
        self._play_position.float_scale = 1000
        self.play_position_gui = NumericParameterGui(name="sec:", param=self._play_position)
        self.play_position_gui.decimals = 3
        self.play_position_gui.set_pressed_callback(self.pause)
        self.play_position_gui.set_released_callback(self.reset_play)
        self.play_position_gui.set_valuechanged_callback(self.slider_value_changed)
        self.play_position_gui.create()
        self.play_position_gui.add_to_layout(hor_layout,5)

    def on_synchronize(self, viewer : ImageViewerClass) -> None:
        pass

    def pause(self):
        pass # to override

    def reset_play(self):
        pass # to override

    def slider_value_changed(self):
        pass # to override

    def speed_value_changed(self):
        pass # to override

    def play_pause(self):
        pass # to override

    def set_play_position(self, recursive=True, fromSlider=False):
        pass # to override

    def update_position(self, precision=0.02, recursive=True, force=False) -> bool:
        pass # to override

    @property
    def vertical_layout(self):
        return self._vertical_layout

    @property
    def filters_widget(self):
        return self._filters_widget

    @property
    def frame_duration(self) -> float:
        return 0.1 # to override

    @property
    def play_position(self) -> float:
        return self.play_position_gui.param.float
    
    @play_position.setter
    def play_position(self, p:float):
        self.play_position_gui.param.float = p
