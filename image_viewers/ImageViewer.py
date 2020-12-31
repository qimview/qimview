#
#
#

from Qt import QtGui, QtCore, QtOpenGL, QtWidgets
import cv2
import sys
import time
import traceback
import abc
import inspect
from image_viewers.ImageFilterParameters import ImageFilterParameters


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


def get_time():
    is_windows = sys.platform.startswith('win')
    if is_windows:
        if hasattr(time, 'clock'):
            return time.clock()
        else:
            return time.perf_counter()
    else:
        return time.time()


def ReadImage(filename):
    print('trying to open', filename)
    try:
        cv_image = cv2.imread(filename, cv2.IMREAD_COLOR)
    except IOError as ex:
        print('IOError: failed to open texture file {}'.format(ex))
        return -1
    print('opened file: ')
    return cv_image


class trace_method():
    def __init__(self, tab):
        self.tab = tab
        method = traceback.extract_stack(None, 2)[0][2]
        print(self.tab[0] + method)
        self.tab[0] += '  '

    def __del__(self):
        self.tab[0] = self.tab[0][:-2]


class ImageViewer:

    def __init__(self, parent=None):
        # super(ImageViewer, self).__init__(parent)
        self.data = None
        self._width = 500
        self._height = 500
        self.lastPos = None # Last mouse position before mouse click
        self.mouse_dx = 0
        self.mouse_dy = 0
        self.current_dx = 0
        self.current_dy = 0
        self.mouse_zx = 0
        self.mouse_zy = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.current_scale = 1
        self.cv_image = None
        self.synchronize_viewer = None
        self.tab = ["--"]
        self.display_timing = False
        self.trace_calls  = False
        self.image_name = ""
        self.active_window = False
        self.filter_params = ImageFilterParameters()
        self.save_image_clipboard = False
        self.clipboard = None
        self.setMouseTracking(True)
        self.verbose = False
        self.start_time = dict()
        self.timings = dict()

    def set_image(self, image):
        is_different = (self.cv_image is None) or (self.cv_image is not image)
        print('set_image({}): is_different = {}'.format(image.shape, is_different))
        if is_different:
            self.cv_image = image
        return is_different

    def set_clipboard(self, clipboard, save_image):
        self.clipboard = clipboard
        self.save_image_clipboard = save_image

    def print_log(self, mess, force=False):
        if self.verbose or force:
            caller_name = inspect.stack()[1][3]
            print("{}{}: {}".format(self.tab[0], caller_name, mess))

    def start_timing(self):
        caller_name = inspect.stack()[1][3]
        class_name = get_class_from_frame(inspect.stack()[1][0])
        if class_name is not None:
            caller_name = "{}.{}".format(class_name, caller_name)
        self.start_time[caller_name] = get_time()
        self.timings[caller_name] = ''

    def add_time(self, mess, current_start, force=False):
        if self.display_timing or force:
            caller_name = inspect.stack()[1][3]
            class_name = get_class_from_frame(inspect.stack()[1][0])
            if class_name is not None:
                caller_name = "{}.{}".format(class_name, caller_name)
            total_start = self.start_time[caller_name]
            ctime = get_time()
            mess = "{} {:0.1f} ms, total {:0.1f} ms".format(mess, (ctime -current_start)*1000, (ctime-total_start)*1000)
            self.timings[caller_name] += "{}{}: {}\n".format(self.tab[0], caller_name, mess)

    def print_timing(self, add_total=False, force=False):
        caller_name = inspect.stack()[1][3]
        class_name = get_class_from_frame(inspect.stack()[1][0])
        if class_name is not None:
            caller_name = "{}.{}".format(class_name, caller_name)
        if add_total:
            self.add_time("total", self.start_time[caller_name], force)
        if self.timings[caller_name] != '':
            print(self.timings[caller_name])

    def set_intensity_levels(self, black, white):
        self.filter_params.black_level_int = black
        self.filter_params.white_level_int = white
        self.filter_params.black_level = black/255.0
        self.filter_params.white_level = white/255.0

    def set_white_balance_scales(self, g_r, g_b):
        self.filter_params.g_r_coeff = g_r
        self.filter_params.g_b_coeff = g_b

    def set_gamma(self, gamma):
        self.filter_params.gamma = gamma

    def set_synchronize(self, viewer):
        self.synchronize_viewer = viewer

    def synchronize_data(self, other_viewer):
        other_viewer.current_scale = self.current_scale
        other_viewer.current_dx = self.current_dx
        other_viewer.current_dy = self.current_dy
        other_viewer.mouse_dx = self.mouse_dx
        other_viewer.mouse_dy = self.mouse_dy
        other_viewer.mouse_zx = self.mouse_zx
        other_viewer.mouse_zy = self.mouse_zy
        other_viewer.mouse_x = self.mouse_x
        other_viewer.mouse_y = self.mouse_y

    def synchronize(self, event_viewer):
        """
        This method needs to be overloaded with call to self.synchronize_viewer.synchronize()
        :param event_viewer: the viewer that started the synchronization
        :return:
        """
        if self==event_viewer:
            if self.display_timing:
                start_time = get_time()
                print("[ --- Start sync")
        if self.synchronize_viewer is not None and self.synchronize_viewer is not event_viewer:
            self.synchronize_data(self.synchronize_viewer)
            self.synchronize_viewer.paintAll()
            self.synchronize_viewer.synchronize(event_viewer)
        if self==event_viewer:
            if self.display_timing:
                print('       End sync --- {:0.1f} ms'.format((get_time()-start_time)*1000))

    def set_active(self, active=True):
        self.active_window = active

    def is_active(self):
        return self.active_window

    def set_image_name(self, image_name=''):
        # print('caption {}'.format(text))
        self.image_name = image_name

    def get_image_name(self):
        return self.image_name

    def get_image(self):
        return self.cv_image

    def new_scale(self, mouse_zy, height):
        return max(1, self.current_scale * (1 + mouse_zy * 5.0 / self._height))
        # return max(1, self.current_scale  + mouse_zy * 5.0 / height)

    def new_translation(self):
        dx = self.current_dx + self.mouse_dx/self.current_scale
        dy = self.current_dy + self.mouse_dy/self.current_scale
        return dx, dy

    def check_translation(self):
        return self.new_translation()

    @abc.abstractmethod
    def paintAll(self):
        pass

    def mouse_press_event(self, event):
        self.lastPos = event.pos()

    def mouse_move_event(self, event):
        self.mouse_x = event.x()
        self.mouse_y = event.y()
        if event.buttons() & QtCore.Qt.LeftButton:
            self.mouse_dx = event.x() - self.lastPos.x()
            self.mouse_dy = - (event.y() - self.lastPos.y())
            self.paintAll()
            self.synchronize(self)
        else:
            if event.buttons() & QtCore.Qt.RightButton:
                # right button zoom
                self.mouse_zx = event.x() - self.lastPos.x()
                self.mouse_zy = - (event.y() - self.lastPos.y())
                self.paintAll()
                self.synchronize(self)
            else:
                modifiers = QtWidgets.QApplication.keyboardModifiers()
                if modifiers & QtCore.Qt.AltModifier:
                    self.paintAll()
                    self.synchronize(self)

    def mouse_release_event(self, event):
        if event.button() & QtCore.Qt.LeftButton:
            self.current_dx, self.current_dy = self.check_translation()
            self.mouse_dy = 0
            self.mouse_dx = 0
        if event.button() & QtCore.Qt.RightButton:
            if self.cv_image is not None:
                self.current_scale = self.new_scale(self.mouse_zy, self.cv_image.shape[0])
            self.mouse_zy = 0
            self.mouse_zx = 0
        self.synchronize(self)

    def mouse_double_click_event(self, event):
        self.print_log("double click ")
        self.set_active()
        self.paintAll()
        if self.synchronize_viewer is not None:
            v = self.synchronize_viewer
            while v != self:
                v.set_active(False)
                v.paintAll()
                if v.synchronize_viewer is not None:
                    v = v.synchronize_viewer

    def mouse_wheel_event(self,event):
        # Zoom by applying a factor to the distances to the sides
        if hasattr(event, 'delta'):
            delta = event.delta()
        else:
            delta = event.angleDelta().y()
        # print("delta = {}".format(delta))
        coeff = delta/5
        # coeff = 20 if delta > 0 else -20
        self.current_scale = self.new_scale(coeff, self.cv_image.shape[0])
        self.paintAll()
        self.synchronize(self)
