#
#
#

from qimview.image_viewers.ImageFilterParameters import ImageFilterParameters
from qimview.utils.utils import get_time
from qimview.utils.qt_imports import QtGui, QtCore, QtWidgets

import cv2
import traceback
import abc
import inspect


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
        self.mouse_dx = self.mouse_dy = 0
        self.mouse_zx = 0
        self.mouse_zy = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.current_dx = self.current_dy = 0
        self.current_scale = 1
        self.cv_image = None
        self.cv_image_ref = None
        self.synchronize_viewer = None
        self.tab = ["--"]
        self.trace_calls  = False
        self.image_name = ""
        self.active_window = False
        self.filter_params = ImageFilterParameters()
        self.save_image_clipboard = False
        self.clipboard = None
        self._display_timing = False
        self._verbose = False
        self.start_time = dict()
        self.timings = dict()
        self.replacing_widget = None
        self.before_max_parent = None
        self.show_histogram = True
        self.show_cursor    = False
        self.show_overlay   = False
        self.show_image_differences = False
        self.antialiasing = True
        # We track an image counter, changed by set_image, to help reducing same calculations
        self.image_id       = -1
        self.image_ref_id   = -1

    @property
    def display_timing(self):
        return self._display_timing

    @display_timing.setter
    def display_timing(self, v):
        self._display_timing = v

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, v):
        self._verbose = v

    def set_image(self, image):
        is_different = (self.cv_image is None) or (self.cv_image is not image)
        if image is not None:
            self.print_log('set_image({}): is_different = {}'.format(image.data.shape, is_different))
        if is_different:
            self.cv_image = image
            self.image_id += 1
        return is_different

    def set_image_ref(self, image_ref=None):
        is_different = (self.cv_image_ref is None) or (self.cv_image_ref is not image_ref)
        if is_different:
            self.cv_image_ref = image_ref
            self.image_ref_id += 1

    def set_clipboard(self, clipboard, save_image):
        self.clipboard = clipboard
        self.save_image_clipboard = save_image

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

        other_viewer.show_histogram = self.show_histogram
        other_viewer.show_cursor    = self.show_cursor

    def synchronize(self, event_viewer):
        """
        This method needs to be overloaded with call to self.synchronize_viewer.synchronize()
        :param event_viewer: the viewer that started the synchronization
        :return:
        """
        if self==event_viewer:
            if self.display_timing:
                start_time = get_time()
                if self.display_timing:
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
        if event.buttons() & QtCore.Qt.RightButton:
            event.accept()

    def mouse_move_event(self, event):
        self.mouse_x = event.x()
        self.mouse_y = event.y()
        if self.show_overlay:
            self.paintAll()
        if event.buttons() & QtCore.Qt.LeftButton:
            self.mouse_dx = event.x() - self.lastPos.x()
            self.mouse_dy = - (event.y() - self.lastPos.y())
            self.paintAll()
            self.synchronize(self)
            event.accept()
        else:
            if event.buttons() & QtCore.Qt.RightButton:
                # right button zoom
                self.mouse_zx = event.x() - self.lastPos.x()
                self.mouse_zy = - (event.y() - self.lastPos.y())
                self.paintAll()
                self.synchronize(self)
                event.accept()
            else:
                modifiers = QtWidgets.QApplication.keyboardModifiers()
                if self.show_cursor:
                    self.paintAll()
                    self.synchronize(self)

    def mouse_release_event(self, event):
        if event.button() & QtCore.Qt.LeftButton:
            self.current_dx, self.current_dy = self.check_translation()
            self.mouse_dy = 0
            self.mouse_dx = 0
            event.accept()
        if event.button() & QtCore.Qt.RightButton:
            if self.cv_image is not None:
                self.current_scale = self.new_scale(self.mouse_zy, self.cv_image.data.shape[0])
            self.mouse_zy = 0
            self.mouse_zx = 0
            event.accept()
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
        self.current_scale = self.new_scale(coeff, self.cv_image.data.shape[0])
        self.paintAll()
        self.synchronize(self)

    def key_press_event(self, event, wsize):
        if type(event) == QtGui.QKeyEvent:
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
            key_list = []

            # # select upper left crop
            # key_list.append(QtCore.Qt.Key_A)
            # if event.key() == QtCore.Qt.Key_A:
            #     self.current_dx = wsize.width()/4
            #     self.current_dy = -wsize.height()/4
            #     self.current_scale = 2

            # select upper left crop
            key_list.append(QtCore.Qt.Key_B)
            if event.key() == QtCore.Qt.Key_B:
                self.current_dx = -wsize.width() / 4
                self.current_dy = -wsize.height() / 4
                self.current_scale = 2

            # # select lower left crop
            # key_list.append(QtCore.Qt.Key_C)
            # if event.key() == QtCore.Qt.Key_C:
            #     self.current_dx = wsize.width() / 4
            #     self.current_dy = wsize.height() / 4
            #     self.current_scale = 2

            # # select lower right crop
            # key_list.append(QtCore.Qt.Key_D)
            # if event.key() == QtCore.Qt.Key_D:
            #     self.current_dx = -wsize.width() / 4
            #     self.current_dy = wsize.height() / 4
            #     self.current_scale = 2

            # select full crop
            key_list.append(QtCore.Qt.Key_F)
            if event.key() == QtCore.Qt.Key_F:
                self.output_crop = (0., 0., 1., 1.)
                self.current_dx = 0
                self.current_dy = 0
                self.current_scale = 1

            # toggle antialiasing
            key_list.append(QtCore.Qt.Key_A)
            if event.key() == QtCore.Qt.Key_A:
                self.antialiasing = not self.antialiasing
                print(f"antialiasing {self.antialiasing}")

            # toggle histograph
            key_list.append(QtCore.Qt.Key_H)
            if event.key() == QtCore.Qt.Key_H:
                self.show_histogram = not self.show_histogram

            # toggle overlay
            key_list.append(QtCore.Qt.Key_O)
            if event.key() == QtCore.Qt.Key_O:
                self.show_overlay = not self.show_overlay

            # toggle cursor
            key_list.append(QtCore.Qt.Key_C)
            if event.key() == QtCore.Qt.Key_C:
                self.show_cursor = not self.show_cursor

            # toggle cursor
            key_list.append(QtCore.Qt.Key_D)
            if event.key() == QtCore.Qt.Key_D:
                self.show_image_differences = not self.show_image_differences

            if event.key() in key_list:
                self.paintAll()
                self.synchronize(self)
                event.accept()
                return
            event.ignore()
        else:
            event.ignore()
