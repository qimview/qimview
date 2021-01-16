
from Qt import QtWidgets, QtCore, QtGui
from image_viewers.glImageViewer import glImageViewer
from image_viewers.pyQtGraphImageViewer import pyQtGraphImageViewer
#import glImageViewerWithShaders
from image_viewers.glImageViewerWithShaders_qglw import glImageViewerWithShaders_qglw
from image_viewers.qtImageViewer import qtImageViewer

from enum import Enum

from image_viewers.ImageFilterParameters import ImageFilterParameters
from image_viewers.ImageFilterParametersGui import ImageFilterParametersGui

from utils.utils import get_time, read_image
from utils.ViewerImage import *
from utils import MyQLabel
import types
import math

class ViewerType(Enum):
    QT_VIEWER = 1
    OPENGL_VIEWER = 2
    OPENGL_SHADERS_VIEWER = 3
    PYQTGRAPH_VIEWER = 4


class MultiView(QtWidgets.QWidget):

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
        self.bold_font = QtGui.QFont()

        self.verbosity = 0

        self.verbosity_LIGHT = 1
        self.verbosity_TIMING = 1 << 2
        self.verbosity_TIMING_DETAILED = 1 << 3
        self.verbosity_TRACE = 1 << 4
        self.verbosity_DEBUG = 1 << 5
        self.set_verbosity(self.verbosity_TIMING_DETAILED)
        self.set_verbosity(self.verbosity_TRACE)

        self.current_image_filename = None

        self.filter_params = ImageFilterParameters()
        self.filter_params_gui = ImageFilterParametersGui(self.filter_params)

        self.viewer_layouts = ['1', '2', '3', '2+2', '3+2', '3+3', '4+3', '4+4', '3+3+3']
        self.default_viewer_layout = '1'

    def set_verbosity(self, flag, enable=True):
        """
        :param v: verbosity flags
        :param b: boolean to enable or disable flag
        :return:
        """
        if enable:
            self.verbosity = self.verbosity | flag
        else:
            self.verbosity = self.verbosity & ~flag

    def check_verbosity(self, flag):
        return self.verbosity & flag

    def show_timing(self):
        return self.check_verbosity(self.verbosity_TIMING) or self.check_verbosity(self.verbosity_TIMING_DETAILED)

    def show_timing_detailed(self):
        return self.check_verbosity(self.verbosity_TIMING_DETAILED)

    def show_trace(self):
        return self.check_verbosity(self.verbosity_TRACE)

    def make_mouse_press(self, image_name):
        def mouse_press(obj, event):
            print('mouse_press')
            obj.update_image(image_name)

        return types.MethodType(mouse_press, self)

    def mouse_release(self, event):
        self.update_image(self.output_label_reference_image)

    def make_mouse_double_click(self, image_name):
        def mouse_double_click(obj, event):
            '''
            Sets the double clicked label as the reference image
            :param obj:
            :param event:
            '''
            print('mouse_double_click {}'.format(image_name))
            obj.output_label_reference_image = image_name
            obj.output_label_current_image = obj.output_label_reference_image
            obj.update_image()

        return types.MethodType(mouse_double_click, self)

    def update_image_intensity_event(self):
        self.update_image_parameters()

    def reset_intensities(self):
        self.filter_params_gui.reset_all()

    def update_image_parameters(self):
        '''
        Uses the variable self.output_label_current_image
        :return:
        '''
        print('update_image_parameters')
        update_start = get_time()

        for n in range(self.nb_viewers_used):
            self.image_viewers[n].filter_params.copy_from(self.filter_params)
            self.image_viewers[n].paintAll()

        if self.show_timing():
            time_spent = get_time() - update_start
            print(" Update image took {0:0.3f} sec.".format(time_spent))

    def CreateImageDisplay(self, image_list):
        # choose image to display
        self.image_list = image_list
        print("CreateImageDisplay {}".format(image_list))
        self.label = dict()
        for image_name in self.image_list:
            # possibility to disable an image using the string 'none', especially useful for input image
            if image_name != 'none':
                self.label[image_name] = MyQLabel.MyQLabel(image_name)
                self.label[image_name].setFrameShape(QtWidgets.QFrame.Panel)
                self.label[image_name].setFrameShadow(QtWidgets.QFrame.Sunken)
                self.label[image_name].setLineWidth(3)
                self.label[image_name].setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
                self.label[image_name].setFixedHeight(40)
                self.label[image_name].mousePressEvent = self.make_mouse_press(image_name)
                self.label[image_name].mouseReleaseEvent = self.mouse_release
                self.label[image_name].mouseDoubleClickEvent = self.make_mouse_double_click(image_name)

        # the crop area can be changed using the mouse wheel
        self.output_label_crop = (0., 0., 1., 1.)

        # set viewers
        for n in range(self.nb_viewers_used):
            self.image_viewers.append(self.image_viewer_class())

        self.output_label_current_rowid = ""
        self.output_label_current_image = ""
        self.output_label_reference_image = ""
        self.output_images = dict()
        self.output_image_label = dict()

    def layout_buttons(self, vertical_layout):
        self.button_widget = QtWidgets.QWidget(self)
        button_layout = QtWidgets.QGridLayout()
        button_layout.setHorizontalSpacing(0)
        button_layout.setVerticalSpacing(0)
        # button_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        max_grid_columns = 5
        idx = 0
        for image_name in self.image_list:
            # possibility to disable an image using the string 'none', especially useful for input image
            if image_name != 'none':
                button_layout.addWidget(self.label[image_name], idx // max_grid_columns, idx % max_grid_columns)
                idx += 1
        vertical_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        # vertical_layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.button_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.button_widget.setLayout(button_layout)
        vertical_layout.addWidget(self.button_widget, 0, QtCore.Qt.AlignTop)

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
        self.update_viewer_layout('1')
        vertical_layout.addLayout(self.viewer_grid_layout, 1)

        self.figures_widget = QtWidgets.QWidget()
        self.figures_layout = QtWidgets.QHBoxLayout()
        self.figures_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        # for the moment ignore this
        # self.figures_layout.addWidget(self.value_in_range_canvas)
        # self.figures_widget.setLayout(self.figures_layout)

        vertical_layout.addWidget(self.figures_widget)
        self.toggle_display_profiles()
        self.setLayout(vertical_layout)
        print("update_layout done")

    def toggle_display_profiles(self):
        self.figures_widget.setVisible(self.display_profiles.isChecked())
        self.update_image()

    def update_image(self, image_name=None, image_filename=None):
        '''
        Uses the variable self.output_label_current_image
        :return:
        '''
        print('update_image {} current: {}'.format(image_name, self.output_label_current_image))
        update_image_start = get_time()
        if image_name is not None:
            self.output_label_current_image = image_name
        if self.output_label_current_image == "" and image_filename is None:
            if self.current_image_filename is not None:
                image_filename = self.current_image_filename
            else:
                return

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        # allow to switch between images by pressing Alt+'image position' (Alt+0, Alt+1, etc)
        # Control key enable display of difference image
        show_diff = modifiers & QtCore.Qt.ControlModifier and self.output_label_current_image != ""
        try:
            current_image = self.get_output_image(self.output_label_current_image, image_filename)
        except Exception as e:
            print("Error: failed to get image {}: {}".format(self.output_label_current_image, e))
            return

        if image_filename is None:
            if self.output_label_current_image != "":
                current_filename = self.output_image_label[self.output_label_current_image]
            else:
                print("Error, we should not get here ...")
                current_filename = self.current_image_filename
        else:
            current_filename = image_filename
            self.current_image_filename = current_filename

        if self.show_timing_detailed():
            time_spent = get_time() - update_image_start
            print(" Update statusbar message took {0:0.3f} sec.".format(time_spent))

        self.statusBar().showMessage("Image: {0}".format(current_filename))
        for im_name in self.image_list:
            # possibility to disable an image using the string 'none', especially useful for input image
            if im_name != 'none':
                is_bold = im_name == self.output_label_current_image
                self.bold_font.setBold(is_bold)
                self.bold_font.setPointSize(8)
                self.label[im_name].setFont(self.bold_font)
                self.label[im_name].setWordWrap(True)
            # self.label[im_name].setMaximumWidth(160)

        # convert PIL image to a PIL.ImageQt object
        if show_diff and self.output_label_reference_image != self.output_label_current_image:
            # don't save the difference
            if self.verbosity > 1:
                print(">> Not saving difference")
            diff_image = self.difference_image(self.output_label_reference_image, self.output_label_current_image)
            current_image = ViewerImage(diff_image, precision=current_image.precision,
                                        downscale=current_image.downscale,
                                        channels=current_image.channels)

        # find first active window
        first_active_window = 0
        for n in range(self.nb_viewers_used):
            if self.image_viewers[n].is_active():
                first_active_window = n
                break
        print("first_active_window = {}".format(first_active_window))

        if self.save_image_clipboard:
            print("set save image to clipboard")
            self.image_viewers[first_active_window].set_clipboard(self.clip, True)
        print("first_active_window {}".format(first_active_window))
        self.image_viewers[first_active_window].set_active(True)
        self.image_viewers[first_active_window].set_image_name(self.output_label_current_image)
        self.image_viewers[first_active_window].set_image(current_image)
        self.image_viewers[first_active_window].show()
        if self.save_image_clipboard:
            print("end save image to clipboard")
            self.image_viewers[first_active_window].set_clipboard(None, False)

        if self.nb_viewers_used >= 2:
            prev_n = first_active_window
            for n in range(1, self.nb_viewers_used):
                n1 = (first_active_window + n) % self.nb_viewers_used
                viewer = self.image_viewers[n1]
                viewer.set_active(False)
                if viewer.get_image() is None:
                    viewer.set_image_name(self.output_label_current_image)
                    viewer.set_image(current_image)
                else:
                    # try to update corresponding images in row
                    try:
                        viewer_image = self.get_output_image(viewer.get_image_name())
                    except Exception as e:
                        print("Error: failed to get image {}: {}".format(viewer.get_image_name(), e))
                    else:
                        viewer.set_image(viewer_image)

                viewer.show()
                self.image_viewers[prev_n].set_synchronize(viewer)
                prev_n = n1
            # Create a synchronization loop
            if prev_n != first_active_window:
                self.image_viewers[prev_n].set_synchronize(self.image_viewers[first_active_window])

        # self.image_scroll_area.adjustSize()
        if self.show_timing():
            time_spent = get_time() - update_image_start
            print(" Update image took {0:0.3f} sec.".format(time_spent))

    def update_viewer_layout(self, layout_name='1'):
        print("*** update_viewer_layout()")
        # # Action from menu ...
        # menu = self.option_viewer_layout
        # for action in menu.actions():
        #     if action.text() == self.current_viewer_layout:
        #         action.setChecked(False)
        # for action in menu.actions():
        #     if action.isChecked():
        #         self.current_viewer_layout = action.text()

        self.current_viewer_layout = layout_name
        # 1. remove current viewers from grid layout
        # self.viewer_grid_layout.hide()
        for v in self.image_viewers:
            v.hide()
            self.viewer_grid_layout.removeWidget(v)

        self.nb_viewers_used = eval(self.current_viewer_layout)
        col_length = int(math.sqrt(self.nb_viewers_used))
        row_length = int(math.ceil(self.nb_viewers_used / col_length))
        print('col_length = {} row_length = {}'.format(col_length, row_length))
        prev_image_viewers = self.image_viewers
        self.image_viewers = []
        for n in range(self.nb_viewers_used):
            if n < len(prev_image_viewers):
                self.image_viewers.append(prev_image_viewers[n])
            else:
                self.image_viewers.append(self.image_viewer_class())
        for n in range(self.nb_viewers_used):
            self.viewer_grid_layout.addWidget(self.image_viewers[n], int(n / float(row_length)), n % row_length)
            self.image_viewers[n].hide()

        # for n in range(row_length):
        # 	self.viewer_grid_layout.setColumnStretch(n, 1)
        # for n in range(col_length):
        # 	self.viewer_grid_layout.setRowStretch(n, 1)

        for n in range(self.nb_viewers_used):
            print("Viewer {} size {}".format(n, (self.image_viewers[n].width(), self.image_viewers[n].height())))

    def update_viewer_layout_callback(self):
        self.update_viewer_layout()
        self.viewer_grid_layout.update()
        self.update_image()

