#
#
# started from https://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
#
# check also https://doc.qt.io/archives/4.6/opengl-overpainting.html
#

from Qt import QtGui, QtCore, QtWidgets
from PIL import Image
from PIL.ImageQt import ImageQt

import argparse
import sys
from image_viewers.ImageViewer import ImageViewer, trace_method, get_time

import cv2
from utils.ViewerImage import *
import json

from tests_utils.event_recorder import EventRecorder
from tests_utils.event_player   import EventPlayer
from tests_utils.qtdump import *

try:
    import cppimport.import_hook
    from CppBind import wrap_numpy as wrap_numpy
except Exception as e:
    has_cppbind = False
    print("Failed to load wrap_numpy: {}".format(e))
else:
    has_cppbind = True

print("Do we have cpp binding ? {}".format(has_cppbind))

class qtImageViewer(QtWidgets.QWidget, ImageViewer ):

    def __init__(self, parent=None, event_recorder=None):
        self.event_recorder = event_recorder
        QtWidgets.QWidget.__init__(self, parent)
        ImageViewer.__init__(self, parent)
        self.anti_aliasing = True
        size_policy = QtWidgets.QSizePolicy()
        size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Ignored)
        size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Ignored)
        self.setSizePolicy(size_policy)
        # self.setAlignment(QtCore.Qt.AlignCenter )
        self.output_crop = (0., 0., 1., 1.)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        # self.display_timing = True
        # self.verbose = True
        # self.trace_calls = True

    #def __del__(self):
    #    pass

    def set_image(self, image, active=False):
        super(qtImageViewer, self).set_image(image)
        # no need for paintAll() show() is called in QTableWindow
        # self.paintAll()

    def apply_zoom(self, crop):
        # Apply zoom
        coeff = 1.0/self.new_scale(self.mouse_zy, self.cv_image.shape[0])
        # zoom from the center of the image
        center = np.array([0.5, 0.5, 0.5, 0.5])
        new_crop = center + (crop - center) * coeff

        # print("new crop zoom 1 {}".format(new_crop))

        # allow crop increase based on the available space
        label_width = self.size().width()
        label_height = self.size().height()
        im_width = self.cv_image.shape[1] * coeff
        im_height = self.cv_image.shape[0] * coeff

        ratio_width = float(label_width) / im_width
        ratio_height = float(label_height) / im_height

        ratio = min(ratio_width, ratio_height)

        if ratio_width<ratio_height:
            # margin to increase height
            margin_pixels = label_height/ratio - im_height
            margin_height = margin_pixels/self.cv_image.shape[0]
            new_crop[1] -= margin_height/2
            new_crop[3] += margin_height/2
        else:
            # margin to increase width
            margin_pixels = label_width/ratio - im_width
            margin_width = margin_pixels/self.cv_image.shape[1]
            new_crop[0] -= margin_width/2
            new_crop[2] += margin_width/2
        # print("new crop zoom 2 {}".format(new_crop))

        return new_crop

    def apply_translation(self, crop):
        """
        :param crop:
        :return: the new crop
        """
        # Apply translation
        diff_x, diff_y = self.new_translation()
        diff_y = - diff_y
        # print(" new translation {} {}".format(diff_x, diff_y))
        # apply the maximal allowed translation
        tr_x = float(diff_x) / self.size().width()
        tr_y = float(diff_y) / self.size().height()
        tr_x = np.clip(tr_x, crop[2]-1, crop[0])
        tr_y = np.clip(tr_y, crop[3]-1, crop[1])
        # normalized position relative to the full image
        return crop - np.array([tr_x, tr_y, tr_x,  tr_y])

    def check_translation(self):
        """
        This method computes the translation really applied based on the current requested translation
        :return:
        """
        # Apply zoom
        crop = self.apply_zoom(np.array(self.output_crop))

        # Compute the translation that is really applied after applying the constraints
        diff_x, diff_y = self.new_translation()
        diff_y = - diff_y
        # print(" new translation {} {}".format(diff_x, diff_y))
        # apply the maximal allowed translation
        w, h = self.size().width(), self.size().height()
        diff_x = np.clip(diff_x, w*(crop[2]-1), w*(crop[0]))
        diff_y = - np.clip(diff_y, h*(crop[3]-1), h*(crop[1]))
        # normalized position relative to the full image
        return diff_x, diff_y

    def update_crop(self):
        # print("output_crop {}".format(self.output_crop))
        # Apply zoom
        new_crop = self.apply_zoom(np.array(self.output_crop))
        # Apply translation
        new_crop = self.apply_translation(new_crop)
        new_crop = np.clip(new_crop, 0, 1)
        # print("move new crop {}".format(new_crop))
        return new_crop

    def apply_filters(self, current_image):
        self.start_timing()
        # # if change saturation
        # saturation_int = int(self.saturation_slider.value())
        # if saturation_int != self.saturation_default:
        #     saturation_coeff = float(saturation_int) / self.saturation_default
        #     print(">> saturation coeff = ", saturation_coeff)
        #     self.saturation_slider.setToolTip("sat. coeff = {}".format(saturation_coeff))
        #     self.saturation_label.setText("Sat coeff {}".format(saturation_coeff))
        #     # saturation_start = get_time()
        #     imghsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        #     # (h, s, v) = cv2.split(imghsv)
        #     sat = imghsv[:, :, 1]
        #     non_saturated = sat < 255 / saturation_coeff
        #     sat[non_saturated] = sat[non_saturated] * saturation_coeff
        #     sat[np.logical_not(non_saturated)] = 255
        #     imghsv[:, :, 1] = sat
        #     # s = np.clip(s, 0, 255)
        #     # imghsv = cv2.merge([h, s, v])
        #     current_image = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)
        # # print " saturation took {0} sec.".format(get_time() - saturation_start)

        # Output RGB from input
        ch = self.cv_image.channels
        if has_cppbind and ch != CH_RGB:
            channels = current_image.channels
            black_level = self.black_level
            white_level = self.white_level
            g_r_coeff = self.g_r_coeff
            g_b_coeff = self.g_b_coeff
            max_value = ((1<<current_image.precision)-1)
            max_type = 1  # not used
            gamma = self.gamma  # not used

            rgb_image = np.empty((current_image.shape[0], current_image.shape[1], 3), dtype=np.uint8)
            time1 = get_time()
            ok = wrap_numpy.apply_filters_RAW(current_image, rgb_image, channels, black_level, white_level, g_r_coeff,
                                              g_b_coeff, max_value, max_type, gamma)
            self.add_time('apply_filters_RAW()',time1, force=True)
            if not ok:
                self.print_log("Failed running wrap_num.apply_filters_RAW ...", force=True)
            current_image = ViewerImage(rgb_image, precision=current_image.precision,
                                        downscale=current_image.downscale,
                                        channels=CH_RGB)
        else:
            # self.print_log("current channels {}".format(ch))
            if ch != CH_RGB:
                ch_pos = channel_position[current_image.channels]
                self.print_log("Converting to RGB")
                # convert Bayer to RGB
                rgb_image = np.empty((current_image.shape[0], current_image.shape[1], 3), dtype=current_image.dtype)
                rgb_image[:, :, 0] = current_image[:, :, ch_pos['r']]
                rgb_image[:, :, 1] = (current_image[:, :, ch_pos['gr']]+current_image[:, :, ch_pos['gb']])/2
                rgb_image[:, :, 2] = current_image[:, :, ch_pos['b']]
                self.print_log(" {} --> {}".format(current_image.shape, rgb_image.shape))
                current_image = ViewerImage(rgb_image, precision=current_image.precision,
                                            downscale=current_image.downscale,
                                            channels=CH_RGB)
                self.print_log(" {} ".format(current_image.shape))

            # Use cv2.convertScaleAbs(I,a,b) function for fast processing
            # res = sat(|I*a+b|)
            # if current_image is not in 8 bits, we need to rescale
            if self.black_level_int != self.blackpoint_default or self.white_level_int != self.whitepoint_default\
                    or current_image.precision!=8:
                min_val = self.black_level_int
                max_val = self.white_level_int
                # adjust levels to precision
                precision = current_image.precision
                min_val = min_val/255*((1<<precision)-1)
                max_val = max_val/255*((1<<precision)-1)
                # to rescale: add min_val and multiply by (max_val-min_val)/255
                if min_val != 0:
                    current_image = cv2.add(current_image, (-min_val, -min_val, -min_val, 0))
                current_image = cv2.convertScaleAbs(current_image, alpha=255. / float(max_val - min_val), beta=0)

            # if gamma changed
            if self.gamma != self.gamma_default:
                gamma_coeff = self.gamma
                # self.gamma_label.setText("Gamma  {}".format(gamma_coeff))
                invGamma = 1.0 / gamma_coeff
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                current_image = cv2.LUT(current_image, table)

        self.print_timing()
        return current_image

    def paintAll(self):
        self.update()

    def paint_image(self):
        draw_text = True
        if self.trace_calls: t = trace_method(self.tab)
        self.start_timing()

        time1 = get_time()
        c = self.update_crop()
        do_crop = (c[2] - c[0] != 1) or (c[3] - c[1] != 1)
        if do_crop:
            w = self.cv_image.shape[1]
            h = self.cv_image.shape[0]
            crop_xmin = int(np.round(c[0] * w))
            crop_xmax = int(np.round(c[2] * w))
            crop_ymin = int(np.round(c[1] * h))
            crop_ymax = int(np.round(c[3] * h))
            current_image = self.cv_image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        else:
            current_image = self.cv_image
        self.add_time('crop', time1)

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        display_cursor =  modifiers & QtCore.Qt.AltModifier
        if display_cursor:
            cropped_image = np.copy(current_image)

        # time1 = get_time()
        label_width = self.size().width()
        label_height = self.size().height()
        image_width = current_image.shape[1]
        image_height = current_image.shape[0]

        ratio_width = float(label_width) / image_width
        ratio_height = float(label_height) / image_height
        ratio = min(ratio_width, ratio_height)
        display_width = int(round(image_width * ratio))
        display_height = int(round(image_height * ratio))
        anti_aliasing = ratio < 1
        #self.print_log("ratio is {:0.2f}".format(ratio))
        use_opencv_resize = anti_aliasing
        opencv_interpolation = cv2.INTER_NEAREST
        resize_applied = False
        # self.print_time('several settings', time1, start_time)

        # self.print_log("use_opencv_resize {} channels {}".format(use_opencv_resize, current_image.channels))
        if use_opencv_resize and not resize_applied and current_image.channels == CH_RGB:
            time1 = get_time()
            resized_image = cv2.resize(current_image, (display_width, display_height),
                                       interpolation=opencv_interpolation)
            resized_image = ViewerImage(resized_image, precision=current_image.precision,
                                                        downscale=current_image.downscale,
                                                        channels=current_image.channels)
            current_image = resized_image
            resize_applied = True
            self.add_time('cv2.resize',time1)

        current_image = self.apply_filters(current_image)

        # TODO: Resize before apply_filters, but it is tricky for Bayer, should maintain channels and precision,
        #  and ViewerImage information
        # if ratio<1 we want anti aliasing and we want to resize as soon as possible to reduce computation time
        # opencv_interpolation = cv2.INTER_AREA
        #if use_opencv_resize and not resize_applied:
        # try to resize anyway with opencv since qt resizing seems too slow
        if not resize_applied:
            time1 = get_time()
            current_image = cv2.resize(current_image, (display_width, display_height),
                                       interpolation=opencv_interpolation)
            resize_applied = True
            self.add_time('cv2.resize',time1)

        # Conversion from numpy array to QImage
        # version 1: goes through PIL image
        # version 2: use QImage constructor directly, faster
        # time1 = get_time()

        numpy2QImage_version = 2
        if numpy2QImage_version == 1:
            current_image_pil = Image.fromarray(current_image)
            self.image_qt = ImageQt(current_image_pil)
            self.qimage = QtGui.QImage(self.image_qt)  # cast PIL.ImageQt object to QImage object -that's the trick!!!
        else:
            if not current_image.flags['C_CONTIGUOUS']:
                current_image = np.require(current_image, np.uint8, 'C')
            self.qimage = QtGui.QImage(current_image.data, current_image.shape[1], current_image.shape[0],
                                       current_image.strides[0], QtGui.QImage.Format_RGB888)

        # self.add_time('QtGui.QPixmap',time1)

        if self.save_image_clipboard:
            self.print_log("exporting to clipboard")
            self.clipboard.setImage(self.qimage, mode=QtGui.QClipboard.Clipboard)

        self.in_paint_all = False


        if not resize_applied:
            time1 = get_time()
            if anti_aliasing:
                qimage = self.qimage.scaled(display_width, display_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            else:
                qimage = self.qimage.scaled(display_width, display_height, QtCore.Qt.KeepAspectRatio)
            self.add_time('qimage.scaled', time1)
            resize_applied = True
        else:
            qimage = self.qimage

        painter = QtGui.QPainter()
        painter.begin(self)

        rect = QtCore.QRect(qimage.rect())
        devRect = QtCore.QRect(0, 0, self.evt_width, self.evt_height)
        rect.moveCenter(devRect.center())

        time1 = get_time()
        painter.drawImage(rect.topLeft(), qimage)
        self.add_time('painter.drawImage',time1)

        # Draw cursor
        if display_cursor:
            if self.display_timing: time1 = self.get_time()
            # get image position
            (height, width, channels) = cropped_image.shape
            im_x = int((self.mouse_x -rect.x())/rect.width()*width)
            im_y = int((self.mouse_y -rect.y())/rect.height()*height)

            pos_from_im_x = int((im_x+0.5)*rect.width()/width +rect.x())
            pos_from_im_y = int((im_y+0.5)*rect.height()/height+rect.y())

            # ratio = self.screen().devicePixelRatio()
            # print("ratio = {}".format(ratio))
            pos_x = pos_from_im_x  # *ratio
            pos_y = pos_from_im_y  # *ratio
            length_percent = 0.04
            # use percentage of the displayed image dimensions
            length = int(max(self.width(),self.height())*length_percent)
            pen_width = 4
            color = QtGui.QColor(0, 255, 255, 200)
            pen = QtGui.QPen()
            pen.setColor(color)
            pen.setWidth(pen_width)
            painter.setPen(pen)
            painter.drawLine(pos_x-length, pos_y, pos_x+length, pos_y)
            painter.drawLine(pos_x, pos_y-length, pos_x, pos_y+length)

            # Update text
            if im_x>=0 and im_x<cropped_image.shape[1] and im_y>=0 and im_y<cropped_image.shape[0]:
                values = cropped_image[im_y, im_x, :]
                self.display_message = " pos {:4}, {:4} \n rgb {} \n {}".format(
                    im_x, im_y, values, self.cv_image.shape)
            else:
                self.display_message = "Out of image"
                # {}  {} {} mouse {} rect {}".format((im_x, im_y),cropped_image.shape,
                #                                     self.cv_image.shape, (self.mouse_x, self.mouse_y), rect)
            self.add_time('drawCursor',time1)
        else:
            self.display_message = self.image_name

        time1 = get_time()
        color = QtGui.QColor(255, 50, 50, 255) if self.is_active() else QtGui.QColor(50, 50, 255, 255)
        painter.setPen(color)
        painter.setFont(QtGui.QFont('Decorative', 16))
        painter.drawText(10, self.evt_height-200, 400, 200,
                         QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft | QtCore.Qt.TextWordWrap,
                         self.display_message)
        self.add_time('painter.drawText',time1)
        painter.end()
        self.print_timing()

    def paintEvent(self, event):
        if self.trace_calls:
            t = trace_method(self.tab)
        self.paint_image()

    def resizeEvent(self, event):
        """Called upon window resizing: reinitialize the viewport.
        """
        if self.trace_calls:
            t = trace_method(self.tab)
        print("resize {} {}  self {} {}".format(event.size().width(), event.size().height(),
              self.width(), self.height()))
        self.evt_width = event.size().width()
        self.evt_height = event.size().height()

    def mousePressEvent(self, event):
        self.mouse_press_event(event)

    def mouseMoveEvent(self, event):
        self.mouse_move_event(event)

    def mouseReleaseEvent(self, event):
        self.mouse_release_event(event)

    def mouseDoubleClickEvent(self, event):
        self.mouse_double_click_event(event)

    def wheelEvent(self, event):
        self.mouse_wheel_event(event)

    def event(self, evt):
        if self.event_recorder is not None:
            self.event_recorder.store_event(self, evt)
        return QtWidgets.QWidget.event(self, evt)

    def keyReleaseEvent(self, evt):
        print(f"evt {evt.type()}")


if __name__ == '__main__':
    # import numpy for generating random data points
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input_image', help='input image')
    parser.add_argument('-p', '--play', help='events json file', default=None)
    parser.add_argument('-r', '--record', help='record events in given json file', default=None)
    args = parser.parse_args()
    _params = vars(args)
    print(f"record {_params['record']}")

    # define a Qt window with an OpenGL widget inside it
    # class TestWindow(QtGui.QMainWindow):
    class TestWindow(QtWidgets.QMainWindow):
        def __init__(self, events):
            super(TestWindow, self).__init__()

            record_file = _params['record']
            if record_file is not None:
                self.event_recorder = EventRecorder(filename=record_file)
                self.event_recorder.register_widget(id(self), "TestWindow")
            else:
                self.event_recorder = None

            self.widget = qtImageViewer(event_recorder = self.event_recorder)
            if record_file is not None:
                self.event_recorder.register_widget(id(self.widget), "widget")

            from utils.utils import read_image
            im = read_image(_params['input_image'])
            self.widget.set_image(im)
            # put the window at the screen position (100, 100)
            self.setGeometry(0, 0, self.widget.width(), self.widget.height())
            self.setCentralWidget(self.widget)
            self.show()
            if events != None:
                event_list = events['events']
                # self.event_recorder.register_widget(self.id(), "TestWindow")
                # self.event_recorder.register_widget(self.widget.id(), "widget")
                event_player = EventPlayer()
                event_player.register_widget("TestWindow", self)
                event_player.register_widget("widget", self.widget)
                event_player.play_events(event_list=event_list)

        def event(self, evt):
            if self.event_recorder is not None:
                if evt.type() in RESIZE_EVENTS:
                    self.event_recorder.store_event(self, evt)
            return QtWidgets.QMainWindow.event(self, evt)

        def closeEvent(self, event):
            # Save event list
            if self.event_recorder is not None:
                self.event_recorder.save_screen(self)
                self.event_recorder.save_events()
            event.accept()


    # create the Qt App and window
    app = QtWidgets.QApplication(sys.argv)
    if _params['play'] is not None:
        with open(_params['play']) as json_file:
            events = json.load(json_file)
    else:
        events = None

    window = TestWindow(events)
    window.show()
    app.exec_()