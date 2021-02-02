
from ..utils.qt_imports import *
from ..utils.utils import get_time
from ..utils.image_reader import ImageReader
from ..utils.image_cache import ImageCache
from ..utils.ViewerImage import *
from ..utils import MyQLabel

from .glImageViewer import glImageViewer
from .pyQtGraphImageViewer import pyQtGraphImageViewer
from .glImageViewerWithShaders_qglw import glImageViewerWithShaders_qglw
from .qtImageViewer import qtImageViewer
from .ImageFilterParameters import ImageFilterParameters
from .ImageFilterParametersGui import ImageFilterParametersGui

from enum import Enum

import types
import math
import cv2

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
        self.allocated_image_viewers = []  # keep allocated image viewers here
        self.image_viewers = []
        self.image_viewer_class = {
            ViewerType.QT_VIEWER:             qtImageViewer,
            ViewerType.OPENGL_VIEWER:         glImageViewer,
            ViewerType.OPENGL_SHADERS_VIEWER: glImageViewerWithShaders_qglw,
            ViewerType.PYQTGRAPH_VIEWER:      pyQtGraphImageViewer
        }[viewer_mode]

        # Create viewer instances
        for n in range(self.nb_viewers_used):
            viewer = self.image_viewer_class()
            self.allocated_image_viewers.append(viewer)
            self.image_viewers.append(viewer)

        self.viewer_mode = viewer_mode
        self.bold_font = QtGui.QFont()

        self.verbosity_LIGHT = 1
        self.verbosity_TIMING = 1 << 2
        self.verbosity_TIMING_DETAILED = 1 << 3
        self.verbosity_TRACE = 1 << 4
        self.verbosity_DEBUG = 1 << 5
        # self.set_verbosity(self.verbosity_TIMING_DETAILED)
        # self.set_verbosity(self.verbosity_TRACE)

        self.verbosity = self.verbosity_LIGHT

        self.current_image_filename = None
        self.save_image_clipboard = False

        self.filter_params = ImageFilterParameters()
        self.filter_params_gui = ImageFilterParametersGui(self.filter_params)

        self.raw_bayer = {'Read': None, 'Bayer0': CH_GBRG, 'Bayer1': CH_BGGR, 'Bayer2': CH_RGGB, 'Bayer3': CH_GRBG}
        self.default_raw_bayer = 'Read'
        self.current_raw_bayer = self.default_raw_bayer

        self.viewer_layouts = ['1', '2', '3', '2+2', '3+2', '3+3', '4+3', '4+4', '3+3+3']
        self.default_viewer_layout = '1'

        # save images of last visited row
        self.cache = ImageCache()
        self.image_dict = { }
        self.read_size = 'full'
        self.image1 = dict()
        self.image2 = dict()
        self.button_layout = None
        self.message_cb = None
        self.replacing_widget = self.before_max_parent = None

        if 'ClickFocus' in QtCore.Qt.FocusPolicy.__dict__:
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        else:
            self.setFocusPolicy(QtCore.Qt.ClickFocus)

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

    def print_log(self, mess):
        if self.verbosity & self.verbosity_LIGHT:
            print(mess)

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

    def set_read_size(self, read_size):
        self.read_size = read_size
        # reset cache
        self.cache.reset()

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

    def set_images(self, images, set_viewers=False):
        self.print_log(f"MultiView.set_images() {images}")
        if images.keys() == self.image_dict.keys():
            self.image_dict = images
        else:
            self.image_dict = images
            self.update_image_buttons()

    def set_viewer_images(self):
        """
        Set viewer images based on self.image_dict.keys()
        :return:
        """
        # if set_viewers, we force the viewer layout and images based on the list
        self.nb_viewers_used = eval(self.current_viewer_layout)
        # be sure to have enough image viewers allocated
        while self.nb_viewers_used > len(self.allocated_image_viewers):
            viewer = self.image_viewer_class()
            self.allocated_image_viewers.append(viewer)
        self.image_viewers = self.allocated_image_viewers[:self.nb_viewers_used]
        image_names = list(self.image_dict.keys())
        for n in range(self.nb_viewers_used):
            if n < len(image_names):
                self.image_viewers[n].set_image_name(image_names[n])
            else:
                self.image_viewers[n].set_image_name(image_names[len(image_names)-1])

    def set_reference_label(self, ref):
        try:
            if ref is not None:
                self.output_label_reference_image = ref
                reference_image = self.get_output_image(self.output_label_reference_image)
                for n in range(self.nb_viewers_used):
                    viewer = self.image_viewers[n]
                    # set reference image
                    viewer.set_image_ref(reference_image)
        except Exception as e:
            print(f' Failed to set reference label {e}')

    def update_image_buttons(self):
        # choose image to display
        self.clear_buttons()
        self.image_list = list(self.image_dict.keys())
        self.print_log("MultiView.update_image_buttons() {}".format(self.image_list))
        self.label = dict()
        for image_name in self.image_list:
            # possibility to disable an image using the string 'none', especially useful for input image
            if image_name != 'none':
                self.label[image_name] = MyQLabel.MyQLabel(image_name, self)
                self.label[image_name].setFrameShape(QtWidgets.QFrame.Panel)
                self.label[image_name].setFrameShadow(QtWidgets.QFrame.Sunken)
                # self.label[image_name].setLineWidth(3)
                self.label[image_name].setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
                # self.label[image_name].setFixedHeight(40)
                self.label[image_name].mousePressEvent = self.make_mouse_press(image_name)
                self.label[image_name].mouseReleaseEvent = self.mouse_release
                self.label[image_name].mouseDoubleClickEvent = self.make_mouse_double_click(image_name)
        self.create_buttons()

        # the crop area can be changed using the mouse wheel
        self.output_label_crop = (0., 0., 1., 1.)

        if len(self.image_list)>0:
            self.output_label_current_image = self.image_list[0]
            self.set_reference_label(self.image_list[0])
        else:
            self.output_label_current_image = ''
            self.output_label_reference_image = ''
        self.output_image_label = dict()

    def clear_buttons(self):
        if self.button_layout is not None:
            # start clearing the layout
            # for i in range(self.button_layout.count()): self.button_layout.itemAt(i).widget().close()
            self.print_log(f"MultiView.clear_buttons() {self.image_list}")
            for image_name in reversed(self.image_list):
                if image_name in self.label:
                    self.button_layout.removeWidget(self.label[image_name])
                    self.label[image_name].close()

    def create_buttons(self):
        if self.button_layout is not None:
            max_grid_columns = 10
            idx = 0
            for image_name in self.image_list:
                # possibility to disable an image using the string 'none', especially useful for input image
                if image_name != 'none':
                    self.button_layout.addWidget(self.label[image_name], idx // max_grid_columns, idx % max_grid_columns)
                    idx += 1

    def layout_buttons(self, vertical_layout):
        self.button_widget = QtWidgets.QWidget(self)
        self.button_layout = QtWidgets.QGridLayout()
        self.button_layout.setHorizontalSpacing(0)
        self.button_layout.setVerticalSpacing(0)
        # button_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.create_buttons()
        vertical_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        # vertical_layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.button_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.button_widget.setLayout(self.button_layout)
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

        # self.saturation_default = 50
        # self.saturation_label = QtWidgets.QLabel("Saturation")
        # parameters_layout.addWidget(self.saturation_label)
        # self.saturation_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # self.saturation_slider.setRange(1, 150)
        # self.saturation_slider.setValue(self.saturation_default)
        # self.saturation_slider.setToolTip("{}".format(self.saturation_default))
        # self.saturation_slider.valueChanged.connect(self.update_image_intensity_event)
        # Add saturation slider
        # parameters_layout.addWidget(self.saturation_slider)

        # --- Saturation adjustment
        self.filter_params_gui.add_saturation(parameters_layout, self.update_image_intensity_event)
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
        self.print_log("update_layout")
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

    def get_output_image(self, im_string_id):
        """
        Search for the image with given label in the current row
        if not in cache reads it and add it to the cache
        :param im_string_id: string that identifies the image to display
        :return:
        """
        # print(f"get_output_image({im_string_id}) ")
        start = get_time()

        # Read both clean and original images and save them in dict as QPixmaps
        image_filename = self.image_dict[im_string_id]
        image_transform = pyQtGraphImageViewer.numpy2imageitem if self.use_pyqtgraph else None
        self.print_log(f"MultiView.get_output_image() image_filename:{image_filename}")

        image_data, _ = self.cache.get_image(image_filename, self.read_size, verbose=self.show_timing_detailed(),
                                             use_RGB=not self.use_opengl, image_transform=image_transform)

        if image_data is not None:
            self.output_image_label[im_string_id] = image_filename
            output_image = image_data
        else:
            print(f"failed to get image {im_string_id}: {image_filename}")
            return None

        if im_string_id in self.image1 and im_string_id in self.image2:
            # Add difference image, now fast to process no need to have it in cache
            output_image = self.difference_image(self.image1[im_string_id],
                                                                     self.image2[im_string_id])
            self.output_image_label[im_string_id] = "127+ ({0} - {1})".format(self.image2[im_string_id],
                                                                              self.image1[im_string_id])

        if self.show_timing_detailed():
            print(" get_output_image took {0:0.3f} sec.".format(get_time() - start))

        # force image bayer information if selected from menu
        res = output_image
        set_bayer = self.raw_bayer[self.current_raw_bayer]
        if res.channels in [CH_BGGR, CH_GBRG, CH_GRBG, CH_RGGB] and set_bayer is not None:
            print(f"Setting bayer {set_bayer}")
            res.channels = set_bayer

        return res

    def set_message_callback(self, message_cb):
        self.message_cb = message_cb

    def setMessage(self, mess):
        if self.message_cb is not None:
            self.message_cb(mess)

    def cache_read_images(self, image_filenames):
        """
        Search for the image with given label in the current row
        if not in cache reads it and add it to the cache
        :param im_string_id: string that identifies the image to display
        :return:
        """
        # print(f"cache_read_images({image_filenames}) ")
        # Read both clean and original images and save them in dict as QPixmaps
        image_transform = pyQtGraphImageViewer.numpy2imageitem if self.use_pyqtgraph else None
        self.cache.add_images(image_filenames, self.read_size, verbose=False, use_RGB=not self.use_opengl,
                             image_transform=image_transform)

    def update_label_fonts(self):
        # Update selected image label, we could do it later too
        for im_name in self.image_list:
            # possibility to disable an image using the string 'none', especially useful for input image
            if im_name != 'none':
                is_bold      = im_name == self.output_label_current_image
                is_underline = im_name == self.output_label_reference_image
                is_bold |= is_underline
                self.bold_font.setBold(is_bold)
                self.bold_font.setUnderline(is_underline)
                self.bold_font.setPointSize(8)
                self.label[im_name].setFont(self.bold_font)
                self.label[im_name].setWordWrap(True)
            # self.label[im_name].setMaximumWidth(160)

    def update_image(self, image_name=None):
        """
        Uses the variable self.output_label_current_image
        :return:
        """
        self.print_log('update_image {} current: {}'.format(image_name, self.output_label_current_image))
        update_image_start = get_time()

        # Define the current selected image
        if image_name is not None:
            self.output_label_current_image = image_name
        if self.output_label_current_image == "":
            return

        if self.image_dict[self.output_label_current_image] is None:
            print(" No image filename for current image")
            return

        self.update_label_fonts()

        # find first active window
        first_active_window = 0
        for n in range(self.nb_viewers_used):
            if self.image_viewers[n].is_active():
                first_active_window = n
                break

        # Read images in parallel to improve preformances
        # list all required image filenames
        # set all viewers image names (labels)
        image_filenames = [self.image_dict[self.output_label_current_image]]
        # define image associated to each used viewer and add it to the list of images to get
        for n in range(self.nb_viewers_used):
            viewer = self.image_viewers[n]
            # Set active only the first active window
            viewer.set_active(n == first_active_window)
            if viewer.get_image() is None:
                if n < len(self.image_list):
                    viewer.set_image_name(self.image_list[n])
                    image_filenames.append(self.image_dict[self.image_list[n]])
                else:
                    viewer.set_image_name(self.output_label_current_image)
            else:
                # get_image_name() should belong to image_dict
                if viewer.get_image_name() in self.image_dict:
                    image_filenames.append(self.image_dict[viewer.get_image_name()])
                else:
                    viewer.set_image_name(self.output_label_current_image)

        # remove duplicates
        image_filenames = list(set(image_filenames))
        # print(f"image filenames {image_filenames}")
        self.cache_read_images(image_filenames)

        try:
            current_image = self.get_output_image(self.output_label_current_image)
            if current_image is None:
                return
        except Exception as e:
            print("Error: failed to get image {}: {}".format(self.output_label_current_image, e))
            return

        current_filename = self.output_image_label[self.output_label_current_image]

        if self.show_timing_detailed():
            time_spent = get_time() - update_image_start

        self.setMessage("Image: {0}".format(current_filename))

        # allow to switch between images by pressing Alt+'image position' (Alt+0, Alt+1, etc)
        # Control key enable display of difference image
        # show_diff = self.show_image_differences and self.output_label_reference_image != self.output_label_current_image
        # if show_diff:
        #     # don't save the difference
        #     if self.verbosity > 1:
        #         print(">> Not saving difference")
        #     diff_image = self.difference_image(self.output_label_reference_image, self.output_label_current_image)
        #     current_image = ViewerImage(diff_image, precision=current_image.precision,
        #                                 downscale=current_image.downscale,
        #                                 channels=current_image.channels)

        current_viewer = self.image_viewers[first_active_window]
        if self.save_image_clipboard:
            print("set save image to clipboard")
            current_viewer.set_clipboard(self.clip, True)
        current_viewer.set_active(True)
        current_viewer.set_image_name(self.output_label_current_image)
        current_viewer.set_image(current_image)
        if self.save_image_clipboard:
            print("end save image to clipboard")
            current_viewer.set_clipboard(None, False)

        reference_image = self.get_output_image(self.output_label_reference_image)

        if self.nb_viewers_used >= 2:
            prev_n = first_active_window
            for n in range(1, self.nb_viewers_used):
                n1 = (first_active_window + n) % self.nb_viewers_used
                viewer = self.image_viewers[n1]
                # viewer image has already been defined
                # try to update corresponding images in row
                try:
                    viewer_image = self.get_output_image(viewer.get_image_name())
                except Exception as e:
                    print("Error: failed to get image {}: {}".format(viewer.get_image_name(), e))
                    viewer.set_image(current_image)
                else:
                    viewer.set_image(viewer_image)

                # set reference image
                viewer.set_image_ref(reference_image)

                self.image_viewers[prev_n].set_synchronize(viewer)
                prev_n = n1
            # Create a synchronization loop
            if prev_n != first_active_window:
                self.image_viewers[prev_n].set_synchronize(self.image_viewers[first_active_window])

        # Be sure to show the required viewers
        for n in range(self.nb_viewers_used):
            viewer = self.image_viewers[n]
            # print(f"show viewer {n}")
            # Note: calling show in any case seems to avoid double calls to paint event that update() triggers
            # viewer.show()
            if viewer.isHidden():
                print(f"show viewer {n}")
                viewer.show()
            else:
                print(f"update viewer {n}")
                viewer.update()


        # self.image_scroll_area.adjustSize()
        # if self.show_timing():
        print(f" Update image took {(get_time() - update_image_start)*1000:0.0f} ms")

    def difference_image(self, image1, image2):
        factor = int(self.diff_color_slider.value())
        # Fast OpenCV code
        start = get_time()
        # add difference image
        im1 = self.get_output_image(image1)
        im2 = self.get_output_image(image2)
        # positive diffs in unsigned 8 bits, OpenCV puts negative values to 0
        diff_plus = cv2.subtract(im1, im2)
        diff_minus = cv2.subtract(im2, im1)
        res = cv2.addWeighted(diff_plus, factor, diff_minus, -factor, 127)
        # print " difference_image OpenCV took {0} sec.".format(get_time() - start)
        # print "max diff = ", np.max(res-res2)
        return res

    def update_viewer_layout(self, layout_name='1'):
        self.print_log("*** update_viewer_layout()")
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
        # be sure to have enough image viewers allocated
        while self.nb_viewers_used > len(self.allocated_image_viewers):
            viewer = self.image_viewer_class()
            self.allocated_image_viewers.append(viewer)

        self.image_viewers = self.allocated_image_viewers[:self.nb_viewers_used]
        # self.image_viewers = []
        # for n in range(self.nb_viewers_used):
        #     self.image_viewers.append(self.allocated_image_viewers[n])

        for n in range(self.nb_viewers_used):
            self.viewer_grid_layout.addWidget(self.image_viewers[n], int(n / float(row_length)), n % row_length)
            self.image_viewers[n].hide()

        # for n in range(row_length):
        # 	self.viewer_grid_layout.setColumnStretch(n, 1)
        # for n in range(col_length):
        # 	self.viewer_grid_layout.setRowStretch(n, 1)

        # for n in range(self.nb_viewers_used):
        #     print("Viewer {} size {}".format(n, (self.image_viewers[n].width(), self.image_viewers[n].height())))

    def update_viewer_layout_callback(self):
        self.update_viewer_layout()
        self.viewer_grid_layout.update()
        self.update_image()

    def keyReleaseEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            # allow to switch between images by pressing Alt+'image position' (Alt+0, Alt+1, etc)
            if modifiers & (QtCore.Qt.AltModifier | QtCore.Qt.ControlModifier):
                event.accept()
            # else:
            #     try:
            #         # reset reference image
            #         if self.output_label_current_image != self.output_label_reference_image:
            #             self.update_image(self.output_label_reference_image)
            #     except Exception as e:
            #         print(" Error: {}".format(e))

    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            if self.show_trace():
                print("key is ", event.key())
                print("key down int is ", int(QtCore.Qt.Key_Down))
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if event.key() == QtCore.Qt.Key_F11:
                # Should be inside a layout
                if self.before_max_parent is None:
                    if self.parent() is not None and self.parent().layout() is not None and \
                            self.parent().layout().indexOf(self) != -1:
                        self.before_max_parent = self.parent()
                        self.replacing_widget = QtWidgets.QWidget(self.before_max_parent)

                        self.parent().layout().replaceWidget(self, self.replacing_widget)
                        self.setParent(None)
                        # Prevent user from closing the window
                        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
                        self.showMaximized()
                        event.accept()
                        return
                if self.before_max_parent is not None:
                    self.setParent(self.before_max_parent)
                    self.parent().layout().replaceWidget(self.replacing_widget, self)
                    self.replacing_widget = self.before_max_parent = None
                    # self.resize(self.before_max_size)
                    self.show()
                    self.parent().update()
                    self.setFocus()
                    event.accept()
                    return

            # allow to switch between images by pressing Alt+'image position' (Alt+0, Alt+1, etc)
            if modifiers & QtCore.Qt.AltModifier:
                for n in range(len(self.image_list)):
                    if self.image_list[n] is not None:
                        if event.key() == QtCore.Qt.Key_0 + n:
                            if self.output_label_current_image != self.image_list[n]:
                                # with Alt+Ctrl, change reference image
                                # if modifiers & QtCore.Qt.ControlModifier:
                                #     self.set_reference_label(self.image_list[n])
                                self.update_image(self.image_list[n])
                                self.setFocus()
                                return
                event.accept()
                return

            if event.modifiers() & QtCore.Qt.ControlModifier:
                # allow to switch between images by pressing Ctrl+'image position' (Ctrl+0, Ctrl+1, etc)
                for n in range(len(self.image_list)):
                    if self.image_list[n] != 'none':
                        if event.key() == QtCore.Qt.Key_0 + n:
                            if self.output_label_current_image != self.image_list[n]:
                                self.set_reference_label(self.image_list[n])
                                self.update_image()
                                event.accept()
                                return
                return
            # print(f"event.modifiers {event.modifiers()}")
            # if not event.modifiers():
            for n in range(len(self.viewer_layouts)):
                if event.key() == QtCore.Qt.Key_0 + (n+1):
                    self.update_viewer_layout(self.viewer_layouts[n])
                    self.viewer_grid_layout.update()
                    self.update_image()
                    self.setFocus()
                    event.accept()
                    return

        else:
            event.ignore()
