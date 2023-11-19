
from qimview.utils.qt_imports     import QtGui, QtWidgets, QtCore
from qimview.utils.utils          import get_time
from qimview.utils.viewer_image   import *
from qimview.utils.menu_selection import MenuSelection
from qimview.utils.mvlabel        import MVLabel
from qimview.cache                import ImageCache
from .fullscreen_helper           import FullScreenHelper
from qimview.image_viewers        import *
from enum                         import Enum, auto
import math

import types
from typing import List, TYPE_CHECKING, Optional, NewType
from qimview.image_viewers.image_viewer import ImageViewer
# Class that derives from ImageViewer
ImageViewerClass = NewType('ImageViewerClass', ImageViewer) 

class ViewerType(Enum):
    QT_VIEWER             = auto()
    OPENGL_VIEWER         = auto()
    OPENGL_SHADERS_VIEWER = auto()


class MultiView(QtWidgets.QWidget):

    def __init__(self, parent=None, viewer_mode: ViewerType =ViewerType.QT_VIEWER, nb_viewers: int =1) -> None:
        """
        :param parent:
        :param viewer_mode:
        :param nb_viewers_used:
        """
        QtWidgets.QWidget.__init__(self, parent)

        # --- Protected members
        # Active viewer index, -1 if none viewer is active
        self._active_viewer_index : int = -1
        # Show only active window
        self._show_active_only : bool = False

        # FullScreen helper features
        self._fullscreen      : FullScreenHelper = FullScreenHelper()

        # --- Public members
        self.use_opengl = viewer_mode in [ViewerType.OPENGL_SHADERS_VIEWER, ViewerType.OPENGL_VIEWER]

        self.nb_viewers_used : int = nb_viewers
        self.allocated_image_viewers = []  # keep allocated image viewers here
        self.image_viewer_classes = {
            ViewerType.QT_VIEWER:             QTImageViewer,
            ViewerType.OPENGL_VIEWER:         GLImageViewer,
            ViewerType.OPENGL_SHADERS_VIEWER: GLImageViewerShaders
        }
        self.image_viewer_class = self.image_viewer_classes[viewer_mode]

        self.image_viewers : List[ImageViewerClass] = []
        # Create viewer instances
        for n in range(self.nb_viewers_used):
            viewer = self.image_viewer_class()
            # viewer.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.PreventContextMenu)
            self.allocated_image_viewers.append(viewer)
            self.image_viewers.append(viewer)

        self.viewer_mode = viewer_mode
        self.bold_font = QtGui.QFont()

        self.verbosity_LIGHT = 1
        self.verbosity_TIMING = 1 << 2
        self.verbosity_TIMING_DETAILED = 1 << 3
        self.verbosity_TRACE = 1 << 4
        self.verbosity_DEBUG = 1 << 5
        self.verbosity = 0

        # self.set_verbosity(self.verbosity_LIGHT)
        # self.set_verbosity(self.verbosity_TIMING_DETAILED)
        # self.set_verbosity(self.verbosity_TRACE)

        self.current_image_filename = None
        self.save_image_clipboard = False

        self.filter_params = ImageFilterParameters()
        self.filter_params_gui = ImageFilterParametersGui(self.filter_params)

        self.raw_bayer = {'Read': None, 'Bayer0': ImageFormat.CH_GBRG, 'Bayer1': ImageFormat.CH_BGGR, 'Bayer2': ImageFormat.CH_RGGB, 'Bayer3': ImageFormat.CH_GRBG}
        self.default_raw_bayer = 'Read'
        self.current_raw_bayer = self.default_raw_bayer

        # Number of viewers currently displayed
        self.nb_viewers_used : int = 0

        # save images of last visited row
        self.cache = ImageCache()
        self.image_dict = { }
        self.read_size = 'full'
        self.image1 = dict()
        self.image2 = dict()
        self.button_layout = None
        self.message_cb = None

        if 'ClickFocus' in QtCore.Qt.FocusPolicy.__dict__:
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        else:
            self.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.key_up_callback = None
        self.key_down_callback = None
        self.output_image_label = dict()

        self.output_label_current_image   : str = ''
        self.output_label_reference_image : str = ''
        self.add_context_menu()
        
        # Parameter to set the number of columns in the viewer grid layout
        # if 0: computed automatically
        self.max_columns       : int = 0 

    def set_key_up_callback(self, c):
        self.key_up_callback = c

    def set_key_down_callback(self, c):
        self.key_down_callback = c

    def add_context_menu(self):
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self._context_menu = QtWidgets.QMenu()
        self.viewer_modes = {}
        for v in ViewerType:
            self.viewer_modes[v.name] = v
        self._default_viewer_mode = ViewerType.QT_VIEWER.name
        self.viewer_mode_selection = MenuSelection("Viewer mode", 
            self._context_menu, self.viewer_modes, self._default_viewer_mode, self.update_viewer_mode)
        self._context_menu.addSeparator()
        action = self._context_menu.addAction("Reset viewers")
        action.triggered.connect(self.reset_viewers)

    def reset_viewers(self):
        for v in self.image_viewers:
            v.hide()
            self.viewer_grid_layout.removeWidget(v)
        self.allocated_image_viewers.clear()
        self.image_viewers.clear()
        # Create viewer instances
        for n in range(self.nb_viewers_used):
            viewer = self.image_viewer_class()
            # viewer.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
            self.allocated_image_viewers.append(viewer)
            self.image_viewers.append(viewer)
        self.set_number_of_viewers(self.nb_viewers_used)
        self.viewer_grid_layout.update()
        self.update_image()

    def update_viewer_mode(self):
        viewer_mode = self.viewer_mode_selection.get_selection_value()
        self.image_viewer_class = self.image_viewer_classes[viewer_mode]

    def show_context_menu(self, pos):
        # allow to switch between images by pressing Alt+'image position' (Alt+0, Alt+1, etc)
        self._context_menu.show()
        self._context_menu.popup( self.mapToGlobal(pos) )

    def set_cache_memory_bar(self, progress_bar):
        self.cache.set_memory_bar(progress_bar)

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
        self.print_log('update_image_parameters')
        update_start = get_time()

        for n in range(self.nb_viewers_used):
            self.image_viewers[n].filter_params.copy_from(self.filter_params)
            self.image_viewers[n].update()

        if self.show_timing():
            time_spent = get_time() - update_start
            self.print_log(" Update image took {0:0.3f} sec.".format(time_spent))

    def set_images(self, images, set_viewers=False):
        self.print_log(f"MultiView.set_images() {images}")
        if images.keys() == self.image_dict.keys():
            self.image_dict = images
            self.update_reference()
        else:
            self.image_dict = images
            self.update_image_buttons()

    def set_viewer_images(self):
        """
        Set viewer images based on self.image_dict.keys()
        :return:
        """
        # if set_viewers, we force the viewer layout and images based on the list
        # be sure to have enough image viewers allocated
        while self.nb_viewers_used > len(self.allocated_image_viewers):
            viewer = self.image_viewer_class()
            # viewer.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
            self.allocated_image_viewers.append(viewer)
        self.image_viewers = self.allocated_image_viewers[:self.nb_viewers_used]
        image_names = list(self.image_dict.keys())
        for n in range(self.nb_viewers_used):
            if n < len(image_names):
                self.image_viewers[n].image_name = image_names[n]
            else:
                self.image_viewers[n].image_name = image_names[len(image_names)-1]

    def update_reference(self) -> None:
        reference_image = self.get_output_image(self.output_label_reference_image)
        for n in range(self.nb_viewers_used):
            viewer = self.image_viewers[n]
            # set reference image
            viewer.set_image_ref(reference_image)

    def set_reference_label(self, ref: str, update_viewers=False) -> None:
        try:
            if ref is not None:
                if ref!=self.output_label_reference_image:
                    self.output_label_reference_image = ref
                    if update_viewers:
                        self.update_reference()
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
                self.label[image_name] = MVLabel(image_name, self)
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
            self.set_reference_label(self.image_list[0], update_viewers=True)
        else:
            self.output_label_current_image = ''
            self.output_label_reference_image = ''

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
        # Add Profiles and keep zoom options
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
        self.filter_params_gui.add_imdiff_factor(parameters_layout, self.update_image_intensity_event)

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
        self.viewer_grid_layout.setHorizontalSpacing(1)
        self.viewer_grid_layout.setVerticalSpacing(1)
        self.set_number_of_viewers(1)
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

        image_filename = self.image_dict[im_string_id]
        image_transform = None
        self.print_log(f"MultiView.get_output_image() image_filename:{image_filename}")

        image_data, _ = self.cache.get_image(image_filename, self.read_size, verbose=self.show_timing_detailed(),
                                             use_RGB=not self.use_opengl, image_transform=image_transform)

        if image_data is not None:
            self.output_image_label[im_string_id] = image_filename
            output_image = image_data
        else:
            print(f"failed to get image {im_string_id}: {image_filename}")
            return None

        if self.show_timing_detailed():
            print(f" get_output_image took {int((get_time() - start)*1000+0.5)} ms".format)

        # force image bayer information if selected from menu
        res = output_image
        set_bayer = self.raw_bayer[self.current_raw_bayer]
        if res.channels in [ImageFormat.CH_BGGR, ImageFormat.CH_GBRG, ImageFormat.CH_GRBG, ImageFormat.CH_RGGB] and set_bayer is not None:
            print(f"Setting bayer {set_bayer}")
            res.channels = set_bayer

        return res

    def set_message_callback(self, message_cb):
        self.message_cb = message_cb

    def setMessage(self, mess):
        if self.message_cb is not None:
            self.message_cb(mess)

    def cache_read_images(self, image_filenames: List[str], reload: bool =False) -> None:
        """ Read the list of images into the cache, with option to reload them from disk

        Args:
            image_filenames (List[str]): list of image filenames
            reload (bool, optional): reload removes first the images from the ImageCache 
                before adding them. Defaults to False.
        """        
        # print(f"cache_read_images({image_filenames}) ")
        image_transform = None
        if reload:
            for f in image_filenames:
                self.cache.remove(f)
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

    def get_active_viewer_index(self) -> int:
        for n in range(self.nb_viewers_used):
            if self.image_viewers[n].is_active:
                return n
        return -1
    
    def on_active(self, viewer : ImageViewerClass) -> None:
        # Activation requested for a given viewer
        # Find viewer in viewer list
        pass

    def update_image(self, image_name=None, reload=False):
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
        first_active_window = self.get_active_viewer_index()
        # If not found, set to 0
        if first_active_window == 1: first_active_window = 0
        self.image_viewers[first_active_window].display_timing = self.show_timing()>0

        # Read images in parallel to improve preformances
        # list all required image filenames
        # set all viewers image names (labels)
        image_filenames = [self.image_dict[self.output_label_current_image]]
        # define image associated to each used viewer and add it to the list of images to get
        for n in range(self.nb_viewers_used):
            viewer : ImageViewer = self.image_viewers[n]
            # Set active only the first active window
            viewer.is_active = (n == first_active_window)
            if viewer.get_image() is None:
                if n < len(self.image_list):
                    viewer.image_name = self.image_list[n]
                    image_filenames.append(self.image_dict[self.image_list[n]])
                else:
                    viewer.image_name = self.output_label_current_image
            else:
                # image_name should belong to image_dict
                if viewer.image_name in self.image_dict:
                    image_filenames.append(self.image_dict[viewer.image_name])
                else:
                    viewer.image_name = self.output_label_current_image

        # remove duplicates
        image_filenames = list(set(image_filenames))
        # print(f"image filenames {image_filenames}")
        self.cache_read_images(image_filenames, reload=reload)

        try:
            current_image = self.get_output_image(self.output_label_current_image)
            if current_image is None:
                return
        except Exception as e:
            print("Error: failed to get image {}: {}".format(self.output_label_current_image, e))
            return

        # print(f"cur {self.output_label_current_image}")
        current_filename = self.output_image_label[self.output_label_current_image]

        if self.show_timing_detailed():
            time_spent = get_time() - update_image_start

        self.setMessage("Image: {0}".format(current_filename))

        current_viewer = self.image_viewers[first_active_window]
        if self.save_image_clipboard:
            print("set save image to clipboard")
            current_viewer.set_clipboard(self.clip, True)
        current_viewer.is_active = True
        current_viewer.image_name = self.output_label_current_image
        current_viewer.set_image(current_image)
        if self.save_image_clipboard:
            print("end save image to clipboard")
            current_viewer.set_clipboard(None, False)

        # print(f"ref {self.output_label_reference_image}")
        if self.output_label_reference_image==self.output_label_current_image:
            reference_image = current_image
        else:
            reference_image = self.get_output_image(self.output_label_reference_image)

        if self.nb_viewers_used >= 2:
            prev_n = first_active_window
            for n in range(1, self.nb_viewers_used):
                n1 = (first_active_window + n) % self.nb_viewers_used
                viewer = self.image_viewers[n1]
                # viewer image has already been defined
                # try to update corresponding images in row
                try:
                    viewer_image = self.get_output_image(viewer.image_name)
                except Exception as e:
                    print("Error: failed to get image {}: {}".format(viewer.image_name, e))
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
        if self._show_active_only:
            for n in range(self.nb_viewers_used):
                viewer = self.image_viewers[n]
                if not viewer.is_active:
                    viewer.hide()
                else:
                    viewer.show()
        else:
            for n in range(self.nb_viewers_used):
                viewer = self.image_viewers[n]
                # print(f"show viewer {n}")
                # Note: calling show in any case seems to avoid double calls to paint event that update() triggers
                # viewer.show()
                if viewer.isHidden():
                    # print(f"show viewer {n}")
                    viewer.show()
                else:
                    # print(f"update viewer {n}")
                    viewer.update()


        # self.image_scroll_area.adjustSize()
        # if self.show_timing():
        print(f" Update image took {(get_time() - update_image_start)*1000:0.0f} ms")

    def set_number_of_viewers(self, nb_viewers: int = 1, max_columns : int = 0) -> None:
        self.print_log("*** set_number_of_viewers()")

        # 1. remove current viewers from grid layout
        # self.viewer_grid_layout.hide()
        for v in self.image_viewers:
            v.hide()
            self.viewer_grid_layout.removeWidget(v)

        self.nb_viewers_used : int = nb_viewers
        print(f"max_columns = {max_columns}")
        if max_columns>0:
            row_length = min(self.nb_viewers_used, max_columns)
            col_length = int(math.ceil(self.nb_viewers_used / row_length))
        else:
            # Find best configuration to fill the space based on image size and widget size?
            col_length = int(math.sqrt(self.nb_viewers_used)+0.5)
            row_length = int(math.ceil(self.nb_viewers_used / col_length))
        self.print_log('col_length = {} row_length = {}'.format(col_length, row_length))
        # be sure to have enough image viewers allocated
        while self.nb_viewers_used > len(self.allocated_image_viewers):
            viewer = self.image_viewer_class()
            # viewer.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
            self.allocated_image_viewers.append(viewer)

        self.image_viewers = self.allocated_image_viewers[:self.nb_viewers_used]

        for n in range(self.nb_viewers_used):
            self.viewer_grid_layout.addWidget(self.image_viewers[n], int(n / float(row_length)), n % row_length)
            self.image_viewers[n].hide()

        # for n in range(self.nb_viewers_used):
        #     print("Viewer {} size {}".format(n, (self.image_viewers[n].width(), self.image_viewers[n].height())))

    def set_number_of_viewers_callback(self):
        self.set_number_of_viewers()
        self.viewer_grid_layout.update()
        self.update_image()

    def mouseDoubleClickEvent(self, event):
        self._show_active_only = not self._show_active_only
        # We need to set the current image 
        active_idx = self.get_active_viewer_index()
        print(f"active_idx {active_idx}")
        self.output_label_reference_image = self.image_viewers[active_idx].image_name
        print(f"output_label_reference_image {self.output_label_reference_image}")
        # Update the image to show/hide viewers
        self.update_image()
        event.accept()

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
        # Different functionalities
        def help():
            """ Open Help Dialog with links to wiki help
            """
            import qimview
            mb = QtWidgets.QMessageBox(self)
            mb.setWindowTitle(f"qimview {qimview.__version__}: MultiView help")
            mb.setTextFormat(QtCore.Qt.TextFormat.RichText)
            mb.setText(
                "<a href='https://github.com/qimview/qimview/wiki'>qimview</a><br>"
                "<a href='https://github.com/qimview/qimview/wiki/4.-Multi%E2%80%90image-viewer'>MultiImage Viewer</a><br>"
                "<a href='https://github.com/qimview/qimview/wiki/3.-Image-Viewers'>Image Viewer</a>")
            mb.exec()
            return True

        def reloadImages():    self.update_image(reload=True); return True
        def enterFullScreen(): return self._fullscreen.enter_fullscreen(self)
        def exitFullScreen():  return self._fullscreen.exit_fullscreen (self)

        if type(event) == QtGui.QKeyEvent:
            # print("key is ", event.key())
            self.print_log(f" QKeySequence() {QtGui.QKeySequence(event.key()).toString()}")
            # print( QtGui.QKeySequence(event.key()).toString())
            # print(f" capslock: {event.getModifierState('CapsLock')}")
            if self.show_trace():
                print("key is ", event.key())
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            QtKey = QtCore.Qt.Key
            keys_callback = {
                int(QtKey.Key_F1)    : help,
                int(QtKey.Key_F5)    : reloadImages,
                int(QtKey.Key_F10)   : enterFullScreen,
                int(QtKey.Key_Escape): exitFullScreen,
            }

            if event.key() in keys_callback:
                if keys_callback[event.key()]():
                    print(f"MultiView event.key() accept")
                    event.accept()
                else:
                    print(f"MultiView event.key() ignore")
                    event.ignore()
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
                                self.set_reference_label(self.image_list[n], update_viewers=True)
                                self.update_image()
                                event.accept()
                                return
                return
            # print(f"event.modifiers {event.modifiers()}")
            # if not event.modifiers():
            for n in range(1, 10):
                if event.key() == QtCore.Qt.Key_0 + n:
                    self.set_number_of_viewers(n)
                    self.viewer_grid_layout.update()
                    self.update_image()
                    self.setFocus()
                    event.accept()
                    return

            if event.key() == QtCore.Qt.Key_Up:
                if self.key_up_callback is not None:
                    self.key_up_callback()
                event.accept()
                return

            if event.key() == QtCore.Qt.Key_Down:
                if self.key_down_callback is not None:
                    self.key_down_callback()
                event.accept()
                return

            nb_images = len(self.image_list)
            if event.key() == QtCore.Qt.Key_Left:
                for n in range(nb_images):
                    if self.output_label_current_image == self.image_list[n]:
                        print(f"setting new image index {(n+nb_images-1)%nb_images}")
                        self.update_image(self.image_list[(n+nb_images-1)%nb_images])
                        event.accept()
                        return

            if event.key() == QtCore.Qt.Key_Right:
                for n in range(nb_images):
                    if self.output_label_current_image == self.image_list[n]:
                        print(f"setting new image index {(n+nb_images+1)%nb_images}")
                        self.update_image(self.image_list[(n+1)%nb_images])
                        event.accept()
                        return
                
            # G: display number of columns
            if event.key() == QtCore.Qt.Key_G:
                self.max_columns = int ((self.max_columns + 1) % self.nb_viewers_used + 1)
                self.set_number_of_viewers(self.nb_viewers_used, max_columns=self.max_columns)
                self.update_image(reload=True)
                self.setFocus()
                event.accept()
                return

        else:
            event.ignore()
