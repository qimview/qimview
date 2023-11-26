#
#
#

from qimview.image_viewers.image_filter_parameters import ImageFilterParameters
from qimview.utils.utils        import get_time
from qimview.utils.qt_imports   import QtGui, QtCore, QtWidgets
from .fullscreen_helper         import FullScreenHelper
from .image_viewer_key_events   import ImageViewerKeyEvents
from .image_viewer_mouse_events import ImageViewerMouseEvents
QtKeys  = QtCore.Qt.Key
QtMouse = QtCore.Qt.MouseButton

import cv2
import traceback
import inspect
import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple, Callable
# if TYPE_CHECKING:
from qimview.utils.viewer_image import ViewerImage, ImageFormat
from abc import abstractmethod
from dataclasses import dataclass

try:
    import qimview_cpp
except Exception as e:
    has_cppbind = False
    print("Failed to load qimview_cpp: {}".format(e))
else:
    has_cppbind = True
print("Do we have cpp binding ? {}".format(has_cppbind))



# copied from https://stackoverflow.com/questions/17065086/how-to-get-the-caller-class-name-inside-a-function-of-another-class-in-python
def get_class_from_frame(fr):
  args, _, _, value_dict = inspect.getargvalues(fr)
  # we check the first parameter for the frame function is
  # named 'self'
  if len(args) and args[0] == 'self':
    # in that case, 'self' will be referenced in value_dict
    instance = value_dict.get('self', None)
    if instance:
      # return its class name
      try:
          # return getattr(instance, '__class__', None)
          return getattr(instance, '__class__', None).__name__
      except:
          return None
  # return None otherwise
  return None


def get_function_name():
    return traceback.extract_stack(None, 2)[0][2]

class trace_method():
    def __init__(self, tab):
        self.tab = tab
        method = traceback.extract_stack(None, 2)[0][2]
        print(self.tab[0] + method)
        self.tab[0] += '  '

    def __del__(self):
        self.tab[0] = self.tab[0][:-2]


#  @dataclass(slots=True)
class ImageViewer:
    # Use slots to avoid new member creation
    # __slots__ = (
    #     '_widget',
    #     '_width',
    #     '_height',
    #     '_image_name',
    #     '_active',
    #     '_on_active',
    #     '_display_timing',
    #     '_verbose',
    #     '_image',
    #     '_image_ref',
    #     '_histo_ref',
    #     '_histo_rect',
    #     '_histo_scale',
    #     '_fullscreen',
    #     '_on_synchronize',
    #     '_save_image_clipboard',
    #     '_clipboard',
    #     '_key_events',
    #     '_mouse_events',
    #     'data',
    #     'lastPos',
    #     'current_dx',
    #     'current_dy',
    #     'current_scale',
    #     'tab',
    #     'trace_calls',
    #     'filter_params',
    #     'start_time',
    #     'timings',
    #     'replacing_widget',
    #     'before_max_parent',
    #     'show_histogram',
    #     'show_cursor',
    #     'show_overlay',
    #     'show_stats',
    #     'show_image_differences',
    #     'show_intensity_line',
    #     'antialiasing',
    #     'image_id',
    #     'image_ref_id',
    #     'evt_width',
    #     'evt_height',
    #     )

    def __init__(self, widget : QtWidgets.QWidget):
        self._widget          : QtWidgets.QWidget = widget

        # --- Protected members
        self._width           : int = 500
        self._height          : int = 500
        self._image_name      : str = ""
        # If the window is active, its appearance (text color or more) will be different
        self._active          : bool = False
        self._on_active       : Optional[Callable] = None
        self._display_timing  : bool = False
        self._verbose         : bool = False
        self._image           : Optional[ViewerImage] = None
        self._image_ref       : Optional[ViewerImage] = None
        # Rectangle in which the histogram is displayed
        self._histo_rect      : Optional[QtCore.QRect] = None
        # Histogram displayed scale
        self._histo_scale     : int = 1

        # FullScreen helper features
        self._fullscreen      : FullScreenHelper = FullScreenHelper()

        # Synchronization callback
        self._on_synchronize  : Optional[Callable]= None

        # Clipboard
        self._save_image_clipboard : bool                       = False
        self._clipboard            : Optional[QtGui.QClipboard] = None
        # Key event class
        self._key_events   : ImageViewerKeyEvents   = ImageViewerKeyEvents(self)
        # Mouse event class
        self._mouse_events : ImageViewerMouseEvents = ImageViewerMouseEvents(self)

        # --- Public members
        self.data = None
        self.lastPos = None # Last mouse position before mouse click
        self.current_dx = self.current_dy = 0
        self.current_scale = 1
        self.tab = ["--"]
        self.trace_calls           : bool                       = False
        self.filter_params         : ImageFilterParameters      = ImageFilterParameters()

        self.start_time = dict()
        self.timings = dict()
        self.replacing_widget = None
        self.before_max_parent = None
        self.show_histogram         : bool = True
        self.show_cursor            : bool = False
        self.show_overlay           : bool = False
        self.show_stats             : bool = False
        self.show_image_differences : bool = False
        self.show_intensity_line    : bool = False
        self.antialiasing           : bool = True
        # We track an image counter, changed by set_image, to help reducing same calculations
        self.image_id       = -1
        self.image_ref_id   = -1
        # Widget dimensions to be defined in child resize event
        self.evt_width : int
        self.evt_height : int

    # === Properties 

    # --- widget
    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._widget

    # --- display_timing
    @property
    def display_timing(self):
        return self._display_timing

    @display_timing.setter
    def display_timing(self, v):
        self._display_timing = v

    # --- verbose
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, v):
        self._verbose = v

    # --- is_active
    @property
    def is_active(self) -> bool:
        return self._active

    @is_active.setter
    def is_active(self, active: bool =True):
        # Do something only if the active flag changes
        if active != self._active:
            self._active = active

    def activate(self):
        """
            Activate this window and call possible callback if it was not already active
        """
        if self.is_active == False:
            # If a callback is set, rely on it
            self._active = True
            if self._on_active:
                self._on_active(self)

    # --- image_name
    @property
    def image_name(self) -> str:
        return self._image_name

    @image_name.setter
    def image_name(self, v : str):
        self._image_name = v

    # === Public methods
    def add_help_tab(self, title:str,  help:str) -> None:
        """ Add markdown formatted text to generated help in help dialog """
        self._key_events.add_help_tab(title, help)

    def add_help_links(self, help:str) -> None:
        """ Add markdown formatted text to generated help in help dialog """
        self._key_events.add_help_links(help)

    def set_activation_callback(self, cb : Optional[Callable]):
        self._on_active = cb

    def set_synchronization_callback(self, cb : Optional[Callable]):
        self._on_synchronize = cb

    def get_image(self):
        return self._image

    def set_image(self, image : Optional[ViewerImage]):
        is_different = (self._image is None) or (self._image is not image)
        if image is not None:
            self.print_log('set_image({}): is_different = {}'.format(image.data.shape, is_different))
        if is_different:
            self._image = image
            self.image_id += 1
        return is_different

    def set_image_ref(self, image_ref : Optional[ViewerImage] = None):
        is_different = (self._image_ref is None) or (self._image_ref is not image_ref)
        if is_different:
            self._image_ref = image_ref
            self.image_ref_id += 1

    def set_clipboard(self, clipboard : Optional[QtGui.QClipboard], save_image: bool):
        self._clipboard            = clipboard
        self._save_image_clipboard = save_image

    def print_log(self, mess, force=False):
        if self.verbose or force:
            caller_name = inspect.stack()[1][3]
            print("{}{}: {}".format(self.tab[0], caller_name, mess))

    def start_timing(self, title=None):
        if not self.display_timing: return
        if title is None:
            # it seems that inspect is slow
            caller_name = inspect.stack()[1][3]
            class_name = get_class_from_frame(inspect.stack()[1][0])
            if class_name is not None:
                caller_name = "{}.{}".format(class_name, caller_name)
        else:
            caller_name = title
        self.start_time[caller_name] = get_time()
        self.timings[caller_name] = ''

    def add_time(self, mess, current_start, force=False, title=None):
        if not self.display_timing: return
        if self.display_timing or force:
            if title is None:
                caller_name = inspect.stack()[1][3]
                class_name = get_class_from_frame(inspect.stack()[1][0])
                if class_name is not None:
                    caller_name = "{}.{}".format(class_name, caller_name)
            else:
                caller_name = title
            if caller_name in self.start_time:
                total_start = self.start_time[caller_name]
                ctime = get_time()
                mess = "{} {:0.1f} ms, total {:0.1f} ms".format(mess, (ctime -current_start)*1000, (ctime-total_start)*1000)
                self.timings[caller_name] += "{}{}: {}\n".format(self.tab[0], caller_name, mess)

    def print_timing(self, add_total=False, force=False, title=None):
        if not self.display_timing: return
        if title is None:
            caller_name = inspect.stack()[1][3]
            class_name = get_class_from_frame(inspect.stack()[1][0])
            if class_name is not None:
                caller_name = "{}.{}".format(class_name, caller_name)
        else:
            caller_name = title
        if add_total:
            self.add_time("total", self.start_time[caller_name], force)
        if self.timings[caller_name] != '':
            print(self.timings[caller_name])

    # Note: 'ImageViewer' is a forward reference to ImageViewer class
    def synchronize_data(self, dest_viewer: 'ImageViewer') -> None:
        """ Synchronize: copy parameters to another viewer
        """
        dest_viewer.current_scale = self.current_scale
        dest_viewer.current_dx = self.current_dx
        dest_viewer.current_dy = self.current_dy
        dest_viewer._mouse_events._mouse_dx   = self._mouse_events._mouse_dx
        dest_viewer._mouse_events._mouse_dy   = self._mouse_events._mouse_dy
        dest_viewer._mouse_events._mouse_zx   = self._mouse_events._mouse_zx
        dest_viewer._mouse_events._mouse_zy   = self._mouse_events._mouse_zy
        dest_viewer._mouse_events._mouse_x    = self._mouse_events._mouse_x
        dest_viewer._mouse_events._mouse_y    = self._mouse_events._mouse_y

        dest_viewer.show_histogram      = self.show_histogram
        dest_viewer.show_cursor         = self.show_cursor
        dest_viewer.show_intensity_line = self.show_intensity_line
        dest_viewer._histo_scale        = self._histo_scale

    def synchronize(self):
        """ Calls synchronization callback if available
        """
        start_time = get_time() if self.display_timing else None
        if self._on_synchronize: 
            self._on_synchronize(self)
        if start_time:
            print('       End sync --- {:0.1f} ms'.format((get_time()-start_time)*1000))

    def new_scale(self, mouse_zy, height):
        return max(1, self.current_scale * (1 + mouse_zy * 5.0 / self._height))
        # return max(1, self.current_scale  + mouse_zy * 5.0 / height)

    def new_translation(self):
        dx = self.current_dx + self._mouse_events._mouse_dx/self.current_scale
        dy = self.current_dy + self._mouse_events._mouse_dy/self.current_scale
        return dx, dy

    def check_translation(self):
        return self.new_translation()

    @abstractmethod
    def viewer_update(self):
        pass

    def key_press_event(self, event, wsize):
        self._key_events.key_press_event(event, wsize)

    def display_message(self, im_pos: Optional[Tuple[int,int]], scale = None) -> str:
        text : str = self.image_name
        if self.show_cursor and im_pos and self._image:
            text +=  f"\n {self._image.data.shape} {self._image.data.dtype} prec:{self._image.precision}"
            if scale is not None:
                text += f"\n x{scale:0.2f}"
            im_x, im_y = im_pos
            values = self._image.data[im_y, im_x]
            text += f"\n pos {im_x:4}, {im_y:4} \n rgb {values}"

        if self.show_overlay:
            text += "\n ref | im " 
        if self.show_image_differences:
            text += "\n im - ref" 
        return text

    def display_text(self, painter: QtGui.QPainter, text: str) -> None:
        self.start_timing()
        color = QtGui.QColor(255, 50, 50, 255) if self.is_active else QtGui.QColor(50, 50, 255, 255)
        painter.setPen(color)
        font = QtGui.QFont('Decorative', 12)
        # font.setBold(True)
        painter.setFont(font)
        painter.setBackground(QtGui.QColor(250, 250, 250, int(0.75*255)))
        painter.setBackgroundMode(QtGui.Qt.BGMode.OpaqueMode)
        text_options = \
            QtCore.Qt.AlignmentFlag.AlignTop  | \
            QtCore.Qt.AlignmentFlag.AlignLeft | \
            QtCore.Qt.TextFlag.TextWordWrap
        area_width = 400
        area_height = 200
        # boundingRect is interesting but slow to be called at each display
        # bounding_rect = painter.boundingRect(0, 0, area_width, area_height, text_options, self.display_message)
        margin_x = 8
        margin_y = 5
        painter.drawText(
            margin_x, 
            # self.evt_height-margin_y-bounding_rect.height(), area_width, area_height,
            margin_y, area_width, area_height,
            text_options,
            text
            )
        self.print_timing()

    def compute_histogram(self, current_image, show_timings=False):
        # print(f"compute_histogram show_timings {show_timings}")
        h_start = get_time() if show_timings else None
        # Compute steps based on input image resolution
        im_w, im_h = current_image.shape[1], current_image.shape[0]
        target_w = 800
        target_h = 600
        hist_x_step = max(1, int(im_w/target_w+0.5))
        hist_y_step = max(1, int(im_h/target_h+0.5))
        input_image = current_image
        # print(f"current_image {current_image.shape} _image {self._image.shape}")
        # input_image = self._image
        resized_im = input_image[::hist_y_step, ::hist_x_step, :]
        resized_im = input_image
        if self.verbose:
            print(f"qtImageViewer.compute_histograph() steps are {hist_x_step, hist_y_step} "
                f"shape {current_image.shape} --> {resized_im.shape}")
        resized_time = get_time()-h_start if h_start else None

        calc_hist_time = 0

        # First compute all histograms
        start_hist = get_time() if show_timings else None
        hist_all = np.empty((3, 256), dtype=np.float32)
        # print(f"{resized_im[::100,::100,:]}")
        for channel, im_ch in enumerate(cv2.split(resized_im)):
            # hist = cv2.calcHist(resized_im[:, :, channel], [0], None, [256], [0, 256])
            hist = cv2.calcHist([im_ch], [0], None, [256], [0, 256])
            # print(f"max diff {np.max(np.abs(hist-hist2))}")
            hist_all[channel, :] = hist[:, 0]

        hist_all = hist_all / np.max(hist_all)
        if start_hist: 
            end_hist = get_time()
            calc_hist_time += end_hist-start_hist
            gauss_start = get_time()
        else:
            gauss_start = None # Help syntax parser
        hist_all = cv2.GaussianBlur(hist_all, (7, 1), sigmaX=1.5, sigmaY=0.2)
        if gauss_start and h_start and resized_time:
            gauss_time = get_time() - gauss_start
            print(f"compute_histogram took {(get_time()-h_start)*1000:0.1f} msec. ", end="")
            print(f"from which calchist:{calc_hist_time*1000:0.1f}, "
              f"resizing:{resized_time*1000:0.1f}, "
              f"gauss:{gauss_time*1000:0.1f}")

        return hist_all

    def compute_histogram_Cpp(self, current_image, show_timings=False):
        # print(f"compute_histogram show_timings {show_timings}")
        h_start = get_time() if show_timings else None 
        # Compute steps based on input image resolution
        im_w, im_h = current_image.shape[1], current_image.shape[0]
        target_w = 800
        target_h = 600
        hist_x_step = max(1, int(im_w/target_w+0.5))
        hist_y_step = max(1, int(im_h/target_h+0.5))
        output_histogram = np.empty((3,256), dtype=np.uint32)
        qimview_cpp.compute_histogram(current_image, output_histogram, int(hist_x_step), int(hist_y_step))
        t1 = get_time() if show_timings else None
        hist_all = output_histogram.astype(np.float32)
        hist_all = hist_all / np.max(hist_all)
        hist_all = cv2.GaussianBlur(hist_all, (7, 1), sigmaX=1.5, sigmaY=0.2)
        if h_start and t1: 
            print(f"qimview_cpp.compute_histogram took {(get_time()-h_start)*1000:0.1f} ms, "
                                f"{(get_time()-t1)*1000:0.1f} ms")
        return hist_all

    def display_histogram(self, hist_all, id, painter, im_rect, show_timings=False):
        """
        :param painter:
        :param rect: displayed image area
        :return:
        """
        if hist_all is None:
            return
        histo_timings = show_timings
        #if histo_timings:
        h_start = get_time()
        # Histogram: keep constant width/height ratio
        display_ratio : float = 2.0
        # print(f'im_rect = {im_rect}')
        w, h = self.evt_width, self.evt_height
        width   : int = int( min(w/4*self._histo_scale, h/3*self._histo_scale))
        height  : int = int( width/display_ratio)
        start_x : int = w - width*id - 10
        start_y : int = h - 10
        margin  : int = 3

        rect_start = get_time() if histo_timings else None 
        rect = QtCore.QRect(start_x-margin, start_y-margin-height, width+2*margin, height+2*margin)
        self._histo_rect = rect
        # painter.fillRect(rect, QtGui.QBrush(QtGui.QColor(255, 255, 255, 128+64)))
        # Transparent light grey
        painter.fillRect(rect, QtGui.QColor(205, 205, 205, 128+32))
        rect_time = get_time()-rect_start if rect_start else None

        # print(f"current_image {current_image.shape} _image {self._image.shape}")
        # input_image = self._image
        path_time = 0

        pen = QtGui.QPen()
        pen.setWidth(2)

        qcolors = {
            0: QtGui.QColor(255, 50, 50, 255),
            1: QtGui.QColor(50, 255, 50, 255),
            2: QtGui.QColor(50, 50, 255, 255)
        }

        step_x = float(width) / 256
        step = 2
        x_range = np.array(range(0, 256, step))
        x_pos = start_x + x_range*step_x

        for channel in range(3):
            pen.setColor(qcolors[channel])
            painter.setPen(pen)
            # painter.setBrush(color)
            # print(f"histogram painting 1 took {get_time() - h_start} sec.")

            # print(f"histogram painting 2 took {get_time() - h_start} sec.")

            start_path = get_time() if histo_timings else None

            # apply a small Gaussian filtering to histogram curve
            path = QtGui.QPainterPath()

            y_pos = start_y - hist_all[channel, x_range]*height
            # polygon = QtGui.QPolygonF([QtCore.QPointF(x_pos[n], y_pos[n]) for n in range(len(x_range))])
            # path.addPolygon(polygon)
            path.moveTo(x_pos[0], y_pos[0])
            for n in range(1,len(x_range)):
                path.lineTo(x_pos[n], y_pos[n])
            painter.drawPath(path)
            path_time += get_time()-start_path if start_path else 0

        if rect_time: 
            print(f"display_histogram took {(get_time()-h_start)*1000:0.1f} msec. ", end='')
            print(f"from which path:{int(path_time*1000)}, rect:{int(rect_time*1000)}")


    def display_intensity_line(self, 
                               painter: QtGui.QPainter, 
                               im_rect: QtCore.QRect, 
                               line: np.ndarray,
                               channels : ImageFormat,
                                ) -> None:
        #if histo_timings:
        h_start = get_time()
        # print(f'im_rect = {im_rect}')
        w, h = self.evt_width, self.evt_height
        width    : int = im_rect.width()
        height   : int = int( h/5)
        start_x  : int = im_rect.x()
        margin_y : int = 2
        start_y  : int = h-margin_y

        rect = QtCore.QRect(start_x, start_y-height, width, height)
        painter.fillRect(rect, QtGui.QColor(205, 205, 205, 128+32))

        pen = QtGui.QPen()
        pen.setWidth(1)

        # Adapt for Bayer, Y, etc ...
        qcolors = {
            'R' : QtGui.QColor(240,  30,  30, 255),
            'G' : QtGui.QColor( 30, 240,  30, 255),
            'Gr': QtGui.QColor(130, 240,  30, 255),
            'Gb': QtGui.QColor( 30, 240, 130, 255),
            'B' : QtGui.QColor( 30,  30, 240, 255),
            'Y' : QtGui.QColor( 30,  30,  30, 255),
        }
        colors = {
            ImageFormat.CH_RGB  : ['R','G','B'],
            ImageFormat.CH_BGR  : ['B','G','R'],
            ImageFormat.CH_RGGB : ['R','Gr','Gb','B'],
            ImageFormat.CH_GRBG : ['Gr','R','B','Gb'],
            ImageFormat.CH_GBRG : ['Gb','B','R','Gr'],
            ImageFormat.CH_BGGR : ['B','Gb','Gr','R'],
        }[channels]
        assert line.shape[1] == len(colors), f"Error: Mismatch between imageformat and number of channels"

        nb_values = line.shape[0]
        step_x = float(width) / nb_values
        x_range = np.array(range(0, nb_values))
        x_pos = start_x + (x_range+0.5)*step_x

        max_val = np.max(line)
        line = line.astype(np.float32)
        in_margin = 2
        in_start_y = start_y - in_margin
        in_height  = height - 2*in_margin
        for channel in range(len(colors)):
            pen.setColor(qcolors[colors[channel]])
            painter.setPen(pen)
            # apply a small Gaussian filtering to histogram curve
            path = QtGui.QPainterPath()
            y_pos = (in_start_y - line[:,channel]*(in_height/max_val)+0.5).astype(np.uint32)
            path.moveTo(x_pos[0], y_pos[0])
            for n in range(1,len(x_range)):
                path.lineTo(x_pos[n], y_pos[n])
            painter.drawPath(path)


