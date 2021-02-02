#
#
# started from https://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
#
# check also https://doc.qt.io/archives/4.6/opengl-overpainting.html
#

from ..utils.qt_imports import *
from ..utils.ViewerImage import *
from ..tests_utils.qtdump import *
try:
    import cppimport.import_hook
    from ..CppBind import wrap_numpy
except Exception as e:
    has_cppbind = False
    print("Failed to load wrap_numpy: {}".format(e))
else:
    has_cppbind = True
print("Do we have cpp binding ? {}".format(has_cppbind))

from .ImageViewer import ImageViewer, trace_method, get_time
from .ImageFilterParameters import ImageFilterParameters

import cv2
import json
import numpy as np
import copy
from scipy.ndimage import gaussian_filter1d


# the opengl version is a bit slow for the moment, due to the texture generation
use_opengl = False
base_widget = QOpenGLWidget if use_opengl else QtWidgets.QWidget

class qtImageViewer(base_widget, ImageViewer ):

    def __init__(self, parent=None, event_recorder=None):
        self.event_recorder = event_recorder
        base_widget.__init__(self, parent)
        ImageViewer.__init__(self, parent)
        self.anti_aliasing = True
        size_policy = QtWidgets.QSizePolicy()
        size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Ignored)
        size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Ignored)
        self.setSizePolicy(size_policy)
        # self.setAlignment(QtCore.Qt.AlignCenter )
        self.output_crop = (0., 0., 1., 1.)

        if 'ClickFocus' in QtCore.Qt.FocusPolicy.__dict__:
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        else:
            self.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.paint_cache      = None
        self.paint_diff_cache = None
        self.diff_image       = None

        self.display_timing = False
        if base_widget is QOpenGLWidget:
            self.setAutoFillBackground(True)
        # self.verbose = True
        # self.trace_calls = True

    #def __del__(self):
    #    pass

    def set_image(self, image):
        super(qtImageViewer, self).set_image(image)

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
        # Apply zoom
        new_crop = self.apply_zoom(np.array(self.output_crop))
        # Apply translation
        new_crop = self.apply_translation(new_crop)
        new_crop = np.clip(new_crop, 0, 1)
        # print("move new crop {}".format(new_crop))
        # print(f"output_crop {self.output_crop} new crop {new_crop}")
        return new_crop

    def apply_filters(self, current_image):
        self.print_log(f"current_image.shape {current_image.shape}")
        # return current_image

        self.start_timing(title='apply_filters()')
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
        if has_cppbind:
            channels = current_image.channels
            black_level = self.filter_params.black_level.float
            white_level = self.filter_params.white_level.float
            g_r_coeff = self.filter_params.g_r.float
            g_b_coeff = self.filter_params.g_b.float
            saturation = self.filter_params.saturation.float
            max_value = ((1<<current_image.precision)-1)
            max_type = 1  # not used
            gamma = self.filter_params.gamma.float  # not used

            rgb_image = np.empty((current_image.shape[0], current_image.shape[1], 3), dtype=np.uint8)
            time1 = get_time()
            ok = False
            if ch in CH_RAWFORMATS or ch in CH_RGBFORMATS:
                cases = {
                    'uint8':  { 'func': wrap_numpy.apply_filters_u8_u8  , 'name': 'apply_filters_u8_u8'},
                    'uint16': { 'func': wrap_numpy.apply_filters_u16_u8, 'name': 'apply_filters_u16_u8'},
                    'uint32': { 'func': wrap_numpy.apply_filters_u32_u8, 'name': 'apply_filters_u32_u8'},
                    'int16': { 'func': wrap_numpy.apply_filters_s16_u8, 'name': 'apply_filters_s16_u8'},
                    'int32': { 'func': wrap_numpy.apply_filters_s32_u8, 'name': 'apply_filters_s32_u8'}
                }
                if current_image.dtype.name in cases:
                    func = cases[current_image.dtype.name]['func']
                    name = cases[current_image.dtype.name]['name']
                    self.print_log(f"wrap_numpy.{name}(current_image, rgb_image, channels, "
                          f"black_level={black_level}, white_level={white_level}, "
                          f"g_r_coeff={g_r_coeff}, g_b_coeff={g_b_coeff}, "
                          f"max_value={max_value}, max_type={max_type}, gamma={gamma})")
                    ok = func(current_image, rgb_image, channels, black_level, white_level, g_r_coeff,
                                g_b_coeff, max_value, max_type, gamma, saturation)
                    self.add_time(f'{name}()',time1, force=True, title='apply_filters()')
                else:
                    print(f"apply_filters() not available for {current_image.dtype} data type !")
            else:
                cases = {
                    'uint8':  { 'func': wrap_numpy.apply_filters_scalar_u8_u8 , 'name': 'apply_filters_scalar_u8_u8'},
                    'uint16': { 'func': wrap_numpy.apply_filters_scalar_u16_u8, 'name': 'apply_filters_scalar_u16_u8'},
                    'uint32': { 'func': wrap_numpy.apply_filters_scalar_u32_u8, 'name': 'apply_filters_scalar_u32_u8'},
                    'float64': { 'func': wrap_numpy.apply_filters_scalar_f64_u8, 'name': 'apply_filters_scalar_f64_u8'},
                }
                if current_image.dtype.name.startswith('float'):
                    max_value = 1.0
                if current_image.dtype.name in cases:
                    func = cases[current_image.dtype.name]['func']
                    name = cases[current_image.dtype.name]['name']
                    self.print_log(f"wrap_numpy.{name}(current_image, rgb_image, "
                          f"black_level={black_level}, white_level={white_level}, "
                          f"max_value={max_value}, max_type={max_type}, gamma={gamma})")
                    ok = func(current_image, rgb_image, black_level, white_level, max_value, max_type, gamma)
                    self.add_time(f'{name}()', time1, force=True, title='apply_filters()')
                else:
                    print(f"apply_filters_scalar() not available for {current_image.dtype} data type !")
            if not ok:
                self.print_log("Failed running wrap_num.apply_filters_u16_u8 ...", force=True)
        else:
            # self.print_log("current channels {}".format(ch))
            if ch in CH_RAWFORMATS:
                ch_pos = channel_position[current_image.channels]
                self.print_log("Converting to RGB")
                # convert Bayer to RGB
                rgb_image = np.empty((current_image.shape[0], current_image.shape[1], 3), dtype=current_image.dtype)
                rgb_image[:, :, 0] = current_image[:, :, ch_pos['r']]
                rgb_image[:, :, 1] = (current_image[:, :, ch_pos['gr']]+current_image[:, :, ch_pos['gb']])/2
                rgb_image[:, :, 2] = current_image[:, :, ch_pos['b']]
            else:
                if ch == CH_Y:
                    # Transform to RGB is it a good idea?
                    rgb_image = np.empty((current_image.shape[0], current_image.shape[1], 3), dtype=current_image.dtype)
                    rgb_image[:, :, 0] = current_image
                    rgb_image[:, :, 1] = current_image
                    rgb_image[:, :, 2] = current_image
                else:
                    rgb_image = current_image

            # Use cv2.convertScaleAbs(I,a,b) function for fast processing
            # res = sat(|I*a+b|)
            # if current_image is not in 8 bits, we need to rescale
            min_val = self.filter_params.black_level.float
            max_val = self.filter_params.white_level.float

            if min_val != 0 or max_val != 1 or current_image.precision!=8:
                min_val = self.filter_params.black_level.float
                max_val = self.filter_params.white_level.float
                # adjust levels to precision
                precision = current_image.precision
                min_val = min_val*((1 << precision)-1)
                max_val = max_val*((1 << precision)-1)
                if rgb_image.dtype == np.uint32:
                    # Formula a bit complicated, we need to be careful with unsigned processing
                    rgb_image =np.clip(((np.clip(rgb_image, min_val, None) - min_val)*(255/(max_val-min_val)))+0.5,
                                       None, 255).astype(np.uint8)
                else:
                    # to rescale: add min_val and multiply by (max_val-min_val)/255
                    if min_val != 0:
                        rgb_image = cv2.add(rgb_image, (-min_val, -min_val, -min_val, 0))
                    rgb_image = cv2.convertScaleAbs(rgb_image, alpha=255. / float(max_val - min_val), beta=0)

            # # if gamma changed
            # if self.filter_params.gamma.value != self.filter_params.gamma.default_value and work_image.dtype == np.uint8:
            #     gamma_coeff = self.filter_params.gamma.float
            #     # self.gamma_label.setText("Gamma  {}".format(gamma_coeff))
            #     invGamma = 1.0 / gamma_coeff
            #     table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            #     work_image = cv2.LUT(work_image, table)

        self.print_timing(title='apply_filters()')
        return rgb_image

    def paintAll(self):
        #if self.cv_image is not None:
        #    self.paint_image()
        if base_widget is QOpenGLWidget:
            self.paint_image()
            self.repaint()
        else:
            self.update()

    def draw_overlay_separation(self, cropped_image_shape, rect, painter):
        (height, width) = cropped_image_shape[:2]
        im_x = int((self.mouse_x - rect.x())/rect.width()*width)
        im_x = max(0, min(width - 1, im_x))
        # im_y = int((self.mouse_y - rect.y())/rect.height()*height)
        # Set position at the beginning of the pixel
        pos_from_im_x = int(im_x*rect.width()/width + rect.x())
        # pos_from_im_y = int((im_y+0.5)*rect.height()/height+ rect.y())
        pen_width = 4
        color = QtGui.QColor(255, 255, 0 , 128)
        pen = QtGui.QPen()
        pen.setColor(color)
        pen.setWidth(pen_width)
        painter.setPen(pen)
        painter.drawLine(pos_from_im_x, rect.y(), pos_from_im_x, rect.y() + rect.height())

    def draw_cursor(self, cropped_image_shape, crop_xmin, crop_ymin, rect, painter):
        """
        :param cropped_image_shape: dimensions of current crop
        :param crop_xmin: left pixel of current crop
        :param crop_ymin: top pixel of current crop
        :param rect: displayed image area
        :param painter:
        :return:
        """
        # Draw cursor
        if self.display_timing: self.start_timing()
        # get image position
        (height, width) = cropped_image_shape[:2]
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
        if im_x>=0 and im_x<cropped_image_shape[1] and im_y>=0 and im_y<cropped_image_shape[0]:
            # values = cropped_image[im_y, im_x]
            im_x += crop_xmin
            im_y += crop_ymin
            values = self.cv_image[im_y, im_x]
            display_message = f"pos {im_x:4}, {im_y:4} \n rgb {values} \n " \
                              f"{self.cv_image.shape} {self.cv_image.dtype} {self.cv_image.precision} bits"
        else:
            display_message = "Out of image"
            # {}  {} {} mouse {} rect {}".format((im_x, im_y),cropped_image.shape,
            #                                     self.cv_image.shape, (self.mouse_x, self.mouse_y), rect)
        if self.display_timing: self.print_timing()
        return display_message

    def display_text(self, painter):
        self.start_timing()
        color = QtGui.QColor(255, 50, 50, 255) if self.is_active() else QtGui.QColor(50, 50, 255, 255)
        painter.setPen(color)
        painter.setFont(QtGui.QFont('Decorative', 16))
        text_options = QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft | QtCore.Qt.TextWordWrap
        area_width = 400
        area_height = 200
        bounding_rect = painter.boundingRect(0, 0, area_width, area_height, text_options, self.display_message)
        margin_x = 8
        margin_y = 5
        painter.drawText(margin_x, self.evt_height-margin_y-bounding_rect.height(), area_width, area_height,
                         text_options,
                         self.display_message)
        self.print_timing()

    def compute_histogram(self, current_image, show_timings=False):
        h_start = get_time()
        histo_timings = True
        # Compute steps based on input image resolution
        im_w, im_h = current_image.shape[1], current_image.shape[0]
        target_w = 800
        target_h = 600
        hist_x_step = max(1, int(im_w/target_w+0.5))
        hist_y_step = max(1, int(im_h/target_h+0.5))
        input_image = current_image
        # print(f"current_image {current_image.shape} cv_image {self.cv_image.shape}")
        # input_image = self.cv_image
        resized_im = input_image[::hist_y_step, ::hist_x_step, :]
        print(f"qtImageViewer.compute_histograph() steps are {hist_x_step, hist_y_step} "
              f"shape {current_image.shape} --> {resized_im.shape}")
        if histo_timings: resized_time = get_time()-h_start

        calc_hist_time = 0

        # First compute all histograms
        if histo_timings: start_hist = get_time()
        hist_all = np.empty((3, 256), dtype=np.float32)
        # print(f"{resized_im[::100,::100,:]}")
        for channel, im_ch in enumerate(cv2.split(resized_im)):
            # hist = cv2.calcHist(resized_im[:, :, channel], [0], None, [256], [0, 256])
            hist = cv2.calcHist([im_ch], [0], None, [256], [0, 256])
            # print(f"max diff {np.max(np.abs(hist-hist2))}")
            hist_all[channel, :] = hist[:, 0]
        hist_all = hist_all / np.max(hist_all)
        # print(f"{hist_all[:,::10]}")
        if histo_timings: end_hist = get_time()
        if histo_timings: calc_hist_time += end_hist-start_hist
        # print(f"histogram painting 1.1 took {get_time() - h_start} sec.")

        if histo_timings: gauss_start = get_time()
        hist_all = cv2.GaussianBlur(hist_all, (7, 1), sigmaX=1.5, sigmaY=0.2)
        if histo_timings: gauss_time = get_time() - gauss_start

        # print(f"compute_histogram took {(get_time()-h_start)*1000:0.1f} msec. ")
        if histo_timings: print(f"from which calchist:{calc_hist_time*1000:0.1f}, "
              f"resizing:{resized_time*1000:0.1f}, "
              f"gauss:{gauss_time*1000:0.1f}")
        return hist_all


    def display_histogram(self, hist_all, painter, rect, show_timings=False):
        """
        :param painter:
        :param rect: displayed image area
        :return:
        """
        histo_timings = show_timings
        #if histo_timings:
        h_start = get_time()
        width = int(rect.width()/4)
        height = int(rect.height()/6)
        start_x = self.evt_width - width - 10
        start_y = self.evt_height - 10
        margin = 3

        if histo_timings: rect_start = get_time()
        rect = QtCore.QRect(start_x-margin, start_y-margin-height, width+2*margin, height+2*margin)
        painter.fillRect(rect, QtGui.QBrush(QtGui.QColor(255, 255, 255, 128+64)))
        if histo_timings: rect_time = get_time()-rect_start

        # print(f"current_image {current_image.shape} cv_image {self.cv_image.shape}")
        # input_image = self.cv_image
        path_time = 0

        pen = QtGui.QPen()
        pen.setWidth(2)

        qcolors = {
            0: QtGui.QColor(255, 50, 50, 255),
            1: QtGui.QColor(50, 255, 50, 255),
            2: QtGui.QColor(50, 50, 255, 255)
        }

        step_x = float(width) / 256

        for channel in range(3):
            pen.setColor(qcolors[channel])
            painter.setPen(pen)
            # painter.setBrush(color)
            # print(f"histogram painting 1 took {get_time() - h_start} sec.")

            # print(f"histogram painting 2 took {get_time() - h_start} sec.")

            if histo_timings: start_path = get_time()

            # apply a small Gaussian filtering to histogram curve
            path = QtGui.QPainterPath()

            step = 2
            x_range = np.array(range(0, 256, step))
            x_pos = start_x + x_range*step_x
            y_pos = start_y - hist_all[channel, x_range]*height
            polygon = QtGui.QPolygonF([QtCore.QPointF(x_pos[n], y_pos[n]) for n in range(len(x_range))])
            path.addPolygon(polygon)
            painter.drawPath(path)
            if histo_timings: path_time += get_time()-start_path

        # print(f"display_histogram took {(get_time()-h_start)*1000:0.1f} msec. ")
        if histo_timings: print(f"from which path:{int(path_time*1000)}, rect:{int(rect_time*1000)}")

    def get_difference_image(self):
        if self.paint_diff_cache is not None:
            use_cache = self.paint_diff_cache['imid'] == self.image_id and \
                        self.paint_diff_cache['imrefid'] == self.image_ref_id
        else:
            use_cache = False

        if not use_cache:
            im1 = self.cv_image
            im2 = self.cv_image_ref
            # TODO: get factor from parameters ...
            # factor = int(self.diff_color_slider.value())
            factor = 3
            # Fast OpenCV code
            start = get_time()
            # positive diffs in unsigned 8 bits, OpenCV puts negative values to 0
            diff_plus = cv2.subtract(im1, im2)
            diff_minus = cv2.subtract(im2, im1)
            res = cv2.addWeighted(diff_plus, factor, diff_minus, -factor, 127)
            print(f" qtImageViewer.difference_image()  took {int((get_time() - start)*1000)} ms")
            # print "max diff = ", np.max(res-res2)
            res = ViewerImage(res, precision=im1.precision, downscale=im1.downscale, channels=im1.channels)
            self.paint_diff_cache = {'imid': self.image_id, 'imrefid': self.image_ref_id}
            self.diff_image = res

        return self.diff_image

    def paint_image(self):
        if self.trace_calls: t = trace_method(self.tab)
        self.start_timing()
        time0 = time1 = get_time()

        label_width = self.size().width()
        label_height = self.size().height()

        show_diff = self.show_image_differences and self.cv_image is not self.cv_image_ref and \
                    self.cv_image_ref is not None and self.cv_image.shape == self.cv_image_ref.shape

        c = self.update_crop()
        # check paint_cache
        if self.paint_cache is not None:
            use_cache = self.paint_cache['imid'] == self.image_id and \
                        np.array_equal(self.paint_cache['crop'],c) and \
                        self.paint_cache['labelw'] == label_width and \
                        self.paint_cache['labelh'] == label_height and \
                        self.paint_cache['filterp'].is_equal(self.filter_params) and \
                        (self.paint_cache['showhist'] == self.show_histogram or not self.show_histogram) and \
                        self.paint_cache['show_diff'] == show_diff and \
                        not self.show_overlay
        else:
            use_cache = False

        # if show_diff, compute the image difference (put it in cache??)
        if show_diff:
            # don't save the difference
            current_image = self.get_difference_image()
        else:
            current_image = self.cv_image

        # could_use_cache = use_cache
        # if could_use_cache:
        #     print(" Could use cache here ... !!!")
        # use_cache = False

        do_crop = (c[2] - c[0] != 1) or (c[3] - c[1] != 1)
        h, w  = current_image.shape[:2]
        if do_crop:
            crop_xmin = int(np.round(c[0] * w))
            crop_xmax = int(np.round(c[2] * w))
            crop_ymin = int(np.round(c[1] * h))
            crop_ymax = int(np.round(c[3] * h))
            current_image = current_image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        else:
            crop_xmin = crop_ymin = 0
            crop_xmax = w
            crop_ymax = h

        cropped_image_shape = current_image.shape
        self.add_time('crop', time1)

        # time1 = get_time()
        image_height, image_width  = current_image.shape[:2]
        ratio_width = float(label_width) / image_width
        ratio_height = float(label_height) / image_height
        ratio = min(ratio_width, ratio_height)
        display_width = int(round(image_width * ratio))
        display_height = int(round(image_height * ratio))

        if self.show_overlay and self.cv_image_ref is not self.cv_image and \
            self.cv_image.shape == self.cv_image_ref.shape:
            # to create the overlay rapidly, we will mix the two images based on the current cursor position
            # 1. convert cursor position to image position
            (height, width) = cropped_image_shape[:2]
            # compute rect
            rect = QtCore.QRect(0, 0, display_width, display_height)
            devRect = QtCore.QRect(0, 0, self.evt_width, self.evt_height)
            rect.moveCenter(devRect.center())
            im_x = int((self.mouse_x - rect.x()) / rect.width() * width)
            im_x = max(0,min(width-1, im_x))
            # im_y = int((self.mouse_y - rect.y()) / rect.height() * height)
            # We need to have a copy here .. slow, better option???
            current_image = np.copy(current_image)
            current_image[:, :im_x] = self.cv_image_ref[crop_ymin:crop_ymax, crop_xmin:(crop_xmin+im_x)]
            # re-instantiate ViewerImage
            current_image = ViewerImage(current_image, precision=self.cv_image.precision,
                                        downscale=self.cv_image.downscale,
                                        channels=self.cv_image.channels)

        resize_applied = False
        if not use_cache:
            anti_aliasing = ratio < 1
            #self.print_log("ratio is {:0.2f}".format(ratio))
            use_opencv_resize = anti_aliasing
            # enable this as optional?
            # opencv_downscale_interpolation = cv2.INTER_AREA
            # opencv_upscale_interpolation   = cv2.INTER_LINEAR
            opencv_fast_interpolation = cv2.INTER_NEAREST
            opencv_downscale_interpolation = opencv_fast_interpolation
            opencv_upscale_interpolation   = opencv_fast_interpolation
            # self.print_time('several settings', time1, start_time)

            # self.print_log("use_opencv_resize {} channels {}".format(use_opencv_resize, current_image.channels))
            # if ratio<1 we want anti aliasing and we want to resize as soon as possible to reduce computation time
            if use_opencv_resize and not resize_applied and current_image.channels == CH_RGB:
                time1 = get_time()
                start_0 = get_time()
                prev_shape = current_image.shape
                resized_image = cv2.resize(current_image, (display_width, display_height),
                                        interpolation=opencv_downscale_interpolation)
                print(f' === qtImageViewer: paint_image() OpenCV resize from {prev_shape} to '
                    f'{(display_height, display_width)} --> {int((get_time()-start_0)*1000)} ms')
                resized_image = ViewerImage(resized_image, precision=current_image.precision,
                                                            downscale=current_image.downscale,
                                                            channels=current_image.channels)
                current_image = resized_image
                resize_applied = True
                self.add_time('cv2.resize',time1)

            current_image = self.apply_filters(current_image)

            # Compute the histogram here, with the smallest image!!!
            if self.show_histogram:
                histograms = self.compute_histogram(current_image, show_timings=self.display_timing)
            else:
                histograms = None

            # try to resize anyway with opencv since qt resizing seems too slow
            if not resize_applied and base_widget is not QOpenGLWidget:
                time1 = get_time()
                start_0 = get_time()
                prev_shape = current_image.shape
                current_image = cv2.resize(current_image, (display_width, display_height),
                                           interpolation=opencv_upscale_interpolation)
                print(f' === qtImageViewer: paint_image() OpenCV resize from {prev_shape} to '
                      f'{(display_height, display_width)} --> {int((get_time()-start_0)*1000)} ms')
                self.add_time('cv2.resize',time1)

            # no need for more resizing
            resize_applied = True

            # Conversion from numpy array to QImage
            # version 1: goes through PIL image
            # version 2: use QImage constructor directly, faster
            # time1 = get_time()

        else:
            resize_applied = True
            current_image = self.paint_cache['current_image']
            histograms = self.paint_cache['histograms']

        # if could_use_cache:
        #     print(f" ======= current_image equal ? {np.array_equal(self.paint_cache['current_image'],current_image)}")

        if not use_cache and not self.show_overlay:
            # cache_time = get_time()
            fp = ImageFilterParameters()
            fp.copy_from(self.filter_params)
            self.paint_cache = {
                'imid': self.image_id,
                'imrefid': self.image_ref_id,
                'crop': c, 'labelw': label_width, 'labelh': label_height,
                'filterp': fp, 'showhist': self.show_histogram,
                'histograms': histograms, 'current_image': current_image,
                'show_diff' : show_diff}
            # print(f"create cache data took {int((get_time() - cache_time) * 1000)} ms")

        if not current_image.flags['C_CONTIGUOUS']:
            current_image = np.require(current_image, np.uint8, 'C')
        qimage = QtGui.QImage(current_image.data, current_image.shape[1], current_image.shape[0],
                                    current_image.strides[0], QtGui.QImage.Format_RGB888)
        # self.add_time('QtGui.QPixmap',time1)

        assert resize_applied, "Image resized should be applied at this point"
        # if not resize_applied:
        #     printf("*** We should never get here ***")
        #     time1 = get_time()
        #     if anti_aliasing:
        #         qimage = qimage.scaled(display_width, display_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        #     else:
        #         qimage = qimage.scaled(display_width, display_height, QtCore.Qt.KeepAspectRatio)
        #     self.add_time('qimage.scaled', time1)
        #     resize_applied = True

        if self.save_image_clipboard:
            self.print_log("exporting to clipboard")
            self.clipboard.setImage(qimage, mode=QtGui.QClipboard.Clipboard)

        painter = QtGui.QPainter()

        painter.begin(self)
        if base_widget is QOpenGLWidget:
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # TODO: check that this condition is not needed
        if base_widget is QOpenGLWidget:
            rect = QtCore.QRect(0,0, display_width, display_height)
        else:
            rect = QtCore.QRect(qimage.rect())
        devRect = QtCore.QRect(0, 0, self.evt_width, self.evt_height)
        rect.moveCenter(devRect.center())

        time1 = get_time()
        if base_widget is QOpenGLWidget:
            painter.drawImage(rect, qimage)
        else:
            painter.drawImage(rect.topLeft(), qimage)
        self.add_time('painter.drawImage',time1)

        if self.show_overlay:
            self.draw_overlay_separation(cropped_image_shape, rect, painter)

        # Draw cursor
        if self.show_cursor:
            self.display_message = self.draw_cursor(cropped_image_shape, crop_xmin, crop_ymin, rect, painter)
        else:
            self.display_message = self.image_name
        if self.show_overlay:
            self.display_message = " REF vs " + self.display_message

        self.display_text(painter)

        # draw histogram
        if self.show_histogram:
            self.display_histogram(histograms, painter, rect, show_timings=self.display_timing)

        painter.end()
        self.print_timing()

        print(f" paint_image took {int((get_time()-time0)*1000)} ms")

    def show(self):
        if base_widget==QOpenGLWidget:
            self.update()
        base_widget.show(self)

    def paintEvent(self, event):
        # print(f" qtImageViewer.paintEvent() {self.image_name}")
        if self.trace_calls:
            t = trace_method(self.tab)
        # try:
        if self.cv_image is not None:
            self.paint_image()
        # except Exception as e:
        #     print(f"Failed paint_image() {e}")

    def resizeEvent(self, event):
        """Called upon window resizing: reinitialize the viewport.
        """
        if self.trace_calls:
            t = trace_method(self.tab)
        self.print_log(f"resize {event.size()}  self {self.width()} {self.height()}")
        self.evt_width = event.size().width()
        self.evt_height = event.size().height()
        base_widget.resizeEvent(self, event)
        self.print_log(f"resize {event.size()}  self {self.width()} {self.height()}")

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
        return base_widget.event(self, evt)

    def keyPressEvent(self, event):
        self.key_press_event(event, wsize=self.size())

    def keyReleaseEvent(self, evt):
        self.print_log(f"evt {evt.type()}")
