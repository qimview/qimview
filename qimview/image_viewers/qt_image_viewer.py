#
#
# started from https://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
#
# check also https://doc.qt.io/archives/4.6/opengl-overpainting.html
#

from qimview.utils.qt_imports import *
from qimview.utils.viewer_image import *
from qimview.utils.utils import clip_value
from qimview.utils.utils import get_time
from qimview.tests_utils.qtdump import *
# Renaming manually since syntax checker has issues with cv2
import cv2
# from cv2 import resize          as opencv_resize
# from cv2 import add             as opencv_add
# from cv2 import subtract        as opencv_subtract
# from cv2 import addWeighted     as opencv_addWeighted
# from cv2 import convertScaleAbs as opencv_convertScaleAbs
# from cv2 import INTER_NEAREST   as opencv_INTER_NEAREST
# from cv2 import INTER_AREA      as opencv_INTER_AREA
import numpy as np
from typing import Tuple, Optional

try:
    import qimview_cpp
except Exception as e:
    has_cppbind = False
    print("Failed to load qimview_cpp: {}".format(e))
else:
    has_cppbind = True
print("Do we have cpp binding ? {}".format(has_cppbind))

from qimview.image_viewers import ImageFilterParameters
from .image_viewer import (ImageViewer, trace_method,)


# the opengl version is a bit slow for the moment, due to the texture generation
# BaseWidget = QOpenGLWidget 
BaseWidget = QtWidgets.QWidget

class QTImageViewer(ImageViewer, BaseWidget):

    def __init__(self, parent : Optional[QtWidgets.QWidget] = None, event_recorder=None):
        BaseWidget.__init__(self, parent)
        ImageViewer.__init__(self, self)
        self.event_recorder = event_recorder
        self.setMouseTracking(True)
        self.anti_aliasing = True
        size_policy = QtWidgets.QSizePolicy()
        size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Ignored)
        size_policy.setVerticalPolicy  (QtWidgets.QSizePolicy.Policy.Ignored)
        self.setSizePolicy(size_policy)
        # self.setAlignment(QtCore.Qt.AlignCenter )
        self.output_crop = np.array([0., 0., 1., 1.], dtype=np.float32)
        self.zoom_center = np.array([0.5, 0.5, 0.5, 0.5])
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        self.paint_cache      = None
        self.paint_diff_cache = None
        self.diff_image       = None

        # self.display_timing = False
        if BaseWidget is QOpenGLWidget:
            self.setAutoFillBackground(True)

        # TODO: how can I set the background color to black without impacting display speed?
        # p = self.palette()
        # p.setColor(self.backgroundRole(), QtCore.Qt.black) 
        # self.setPalette(p)
        # self.setAutoFillBackground(True)

        self.verbose = False
        # self.trace_calls = True

    #def __del__(self):
    #    pass

    def apply_zoom(self, crop):
        if self._image is None: return crop
        (height, width) = self._image.data.shape[:2]
        # print(f"height, width = {height, width}")
        # Apply zoom
        coeff = 1.0/self.new_scale(self._mouse_events._mouse_zy, height)
        # zoom from the center of the image
        center = self.zoom_center
        new_crop = center + (crop - center) * coeff

        # print("new crop zoom 1 {}".format(new_crop))

        # allow crop increase based on the available space
        label_width = self.width()
        # print(f"label_width {label_width}")
        label_height = self.height()

        new_width = width * coeff
        new_height = height * coeff

        ratio_width = float(label_width) / new_width
        ratio_height = float(label_height) / new_height

        # print(f" ratio_width {ratio_width} ratio_height {ratio_height}")
        ratio = min(ratio_width, ratio_height)

        if ratio_width<ratio_height:
            # margin to increase height
            margin_pixels = label_height/ratio - new_height
            margin_height = margin_pixels/height
            new_crop[1] -= margin_height/2
            new_crop[3] += margin_height/2
        else:
            # margin to increase width
            margin_pixels = label_width/ratio - new_width
            margin_width = margin_pixels/width
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
        tr_x = float(diff_x) / self.width()
        tr_y = float(diff_y) / self.height()
        tr_x = clip_value(tr_x, crop[2]-1, crop[0])
        tr_y = clip_value(tr_y, crop[3]-1, crop[1])
        # normalized position relative to the full image
        crop[0] -= tr_x
        crop[1] -= tr_y
        crop[2] -= tr_x
        crop[3] -= tr_y

    def check_translation(self):
        """
        This method computes the translation really applied based on the current requested translation
        :return:
        """
        # Apply zoom
        crop = self.apply_zoom(self.output_crop)

        # Compute the translation that is really applied after applying the constraints
        diff_x, diff_y = self.new_translation()
        diff_y = - diff_y
        # print(" new translation {} {}".format(diff_x, diff_y))
        # apply the maximal allowed translation
        w, h = self.width(), self.height()
        diff_x = clip_value(diff_x, w*(crop[2]-1), w*(crop[0]))
        diff_y = - clip_value(diff_y, h*(crop[3]-1), h*(crop[1]))
        # normalized position relative to the full image
        return diff_x, diff_y

    def update_crop(self):
        # Apply zoom
        new_crop = self.apply_zoom(self.output_crop)
        # print(f"update_crop {self.output_crop} --> {new_crop}")
        # Apply translation
        self.apply_translation(new_crop)
        new_crop = np.clip(new_crop, 0, 1)
        # print("move new crop {}".format(new_crop))
        # print(f"output_crop {self.output_crop} new crop {new_crop}")
        return new_crop

    def update_crop_new(self):
        # 1. transform crop to display coordinates
        
        # Apply zoom
        new_crop = self.apply_zoom(self.output_crop)
        # print(f"update_crop {self.output_crop} --> {new_crop}")
        # Apply translation
        self.apply_translation(new_crop)
        new_crop = np.clip(new_crop, 0, 1)
        # print("move new crop {}".format(new_crop))
        # print(f"output_crop {self.output_crop} new crop {new_crop}")
        return new_crop

    def apply_filters(self, current_image: ViewerImage) -> np.ndarray:
        self.print_log(f"current_image.data.shape {current_image.data.shape}")
        # return current_image

        self.start_timing(title='apply_filters()')

        # Output RGB from input
        if self._image is None: return current_image
        ch = self._image.channels
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

            rgb_image = np.empty((current_image.data.shape[0], current_image.data.shape[1], 3), dtype=np.uint8)
            time1 = get_time()
            ok = False
            if ch in ImageFormat.CH_RAWFORMATS() or ch in ImageFormat.CH_RGBFORMATS():
                cases = {
                    'uint8':  { 'func': qimview_cpp.apply_filters_u8_u8  , 'name': 'apply_filters_u8_u8'},
                    'uint16': { 'func': qimview_cpp.apply_filters_u16_u8, 'name': 'apply_filters_u16_u8'},
                    'uint32': { 'func': qimview_cpp.apply_filters_u32_u8, 'name': 'apply_filters_u32_u8'},
                    'int16': { 'func': qimview_cpp.apply_filters_s16_u8, 'name': 'apply_filters_s16_u8'},
                    'int32': { 'func': qimview_cpp.apply_filters_s32_u8, 'name': 'apply_filters_s32_u8'}
                }
                if current_image.data.dtype.name in cases:
                    func = cases[current_image.data.dtype.name]['func']
                    name = cases[current_image.data.dtype.name]['name']
                    self.print_log(f"qimview_cpp.{name}(current_image, rgb_image, channels, "
                          f"black_level={black_level}, white_level={white_level}, "
                          f"g_r_coeff={g_r_coeff}, g_b_coeff={g_b_coeff}, "
                          f"max_value={max_value}, max_type={max_type}, gamma={gamma})")
                    ok = func(current_image.data, rgb_image, channels, black_level, white_level, g_r_coeff,
                                g_b_coeff, max_value, max_type, gamma, saturation)
                    self.add_time(f'{name}()',time1, force=True, title='apply_filters()')
                else:
                    print(f"apply_filters() not available for {current_image.data.dtype} data type !")
            else:
                cases = {
                    'uint8': { 'func': qimview_cpp.apply_filters_scalar_u8_u8, 'name': 'apply_filters_scalar_u8_u8'},
                    'uint16': { 'func': qimview_cpp.apply_filters_scalar_u16_u8, 'name': 'apply_filters_scalar_u16_u8'},
                    'int16': { 'func': qimview_cpp.apply_filters_scalar_s16_u8, 'name': 'apply_filters_scalar_s16_u8'},
                    'uint32': { 'func': qimview_cpp.apply_filters_scalar_u32_u8, 'name': 'apply_filters_scalar_u32_u8'},
                    'float64': { 'func': qimview_cpp.apply_filters_scalar_f64_u8, 'name': 'apply_filters_scalar_f64_u8'},
                }
                if current_image.data.dtype.name.startswith('float'):
                    max_value = 1.0
                if current_image.data.dtype.name in cases:
                    func = cases[current_image.data.dtype.name]['func']
                    name = cases[current_image.data.dtype.name]['name']
                    self.print_log(f"qimview_cpp.{name}(current_image, rgb_image, "
                          f"black_level={black_level}, white_level={white_level}, "
                          f"max_value={max_value}, max_type={max_type}, gamma={gamma})")
                    ok = func(current_image.data, rgb_image, black_level, white_level, max_value, max_type, gamma)
                    self.add_time(f'{name}()', time1, force=True, title='apply_filters()')
                else:
                    print(f"apply_filters_scalar() not available for {current_image.data.dtype} data type !")
            if not ok:
                self.print_log("Failed running wrap_num.apply_filters_u16_u8 ...", force=True)
        else:
            # self.print_log("current channels {}".format(ch))
            if ch in ImageFormat.CH_RAWFORMATS():
                channel_pos = channel_position[current_image.channels]
                self.print_log("Converting to RGB")
                # convert Bayer to RGB
                rgb_image = np.empty((current_image.data.shape[0], current_image.data.shape[1], 3), 
                                        dtype=current_image.data.dtype)
                rgb_image[:, :, 0] = current_image.data[:, :, channel_pos['r']]
                rgb_image[:, :, 1] = (current_image.data[:, :, channel_pos['gr']]+current_image.data[:, :, channel_pos['gb']])/2
                rgb_image[:, :, 2] = current_image.data[:, :, channel_pos['b']]
            else:
                if ch == ImageFormat.CH_Y:
                    # Transform to RGB is it a good idea?
                    rgb_image = np.empty((current_image.data.shape[0], current_image.data.shape[1], 3), 
                                            dtype=current_image.data.dtype)
                    rgb_image[:, :, 0] = current_image.data
                    rgb_image[:, :, 1] = current_image.data
                    rgb_image[:, :, 2] = current_image.data
                else:
                    rgb_image = current_image.data

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

    def viewer_update(self):
        if BaseWidget is QOpenGLWidget:
            self.paint_image()
            self.repaint()
        else:
            self.update()

    def draw_overlay_separation(self, cropped_image_shape, rect, painter):
        (height, width) = cropped_image_shape[:2]
        im_x = int((self._mouse_events._mouse_x - rect.x())/rect.width()*width)
        im_x = max(0, min(width - 1, im_x))
        # im_y = int((self._mouse_events._mouse_y - rect.y())/rect.height()*height)
        # Set position at the beginning of the pixel
        pos_from_im_x = int(im_x*rect.width()/width + rect.x())
        # pos_from_im_y = int((im_y+0.5)*rect.height()/height+ rect.y())
        pen_width = 2
        color = QtGui.QColor(255, 255, 0 , 128)
        pen = QtGui.QPen()
        pen.setColor(color)
        pen.setWidth(pen_width)
        painter.setPen(pen)
        painter.drawLine(pos_from_im_x, rect.y(), pos_from_im_x, rect.y() + rect.height())

    def draw_cursor(self, cropped_image_shape, crop_xmin, crop_ymin, rect, painter, full=False) -> Optional[Tuple[int, int]]:
        """
        :param cropped_image_shape: dimensions of current crop
        :param crop_xmin: left pixel of current crop
        :param crop_ymin: top pixel of current crop
        :param rect: displayed image area
        :param painter:
        :return:
            tuple: (posx, posy) image pixel position of the cursor, if None cursor is out of image
        """
        # Draw cursor
        if self.display_timing: self.start_timing()
        # get image position
        (height, width) = cropped_image_shape[:2]
        im_x = int((self._mouse_events._mouse_x -rect.x())/rect.width()*width)
        im_y = int((self._mouse_events._mouse_y -rect.y())/rect.height()*height)

        pos_from_im_x = int((im_x+0.5)*rect.width()/width +rect.x())
        pos_from_im_y = int((im_y+0.5)*rect.height()/height+rect.y())

        # ratio = self.screen().devicePixelRatio()
        # print("ratio = {}".format(ratio))
        pos_x = pos_from_im_x  # *ratio
        pos_y = pos_from_im_y  # *ratio
        length_percent = 0.04
        # use percentage of the displayed image dimensions
        length = int(max(self.width(),self.height())*length_percent)
        pen_width = 2 if full else 3
        color = QtGui.QColor(0, 255, 255, 200)
        pen = QtGui.QPen()
        pen.setColor(color)
        pen.setWidth(pen_width)
        painter.setPen(pen)
        if not full:
            painter.drawLine(pos_x-length, pos_y, pos_x+length, pos_y)
            painter.drawLine(pos_x, pos_y-length, pos_x, pos_y+length)
        else:
            painter.drawLine(rect.x(), pos_y, rect.x()+rect.width(), pos_y)
            painter.drawLine(pos_x, rect.y(), pos_x, rect.y()+rect.height())

        # Update text
        if im_x>=0 and im_x<cropped_image_shape[1] and im_y>=0 and im_y<cropped_image_shape[0]:
            # values = cropped_image[im_y, im_x]
            im_x += crop_xmin
            im_y += crop_ymin
            im_pos = (im_x, im_y)
        else:
            im_pos = None
        if self.display_timing: self.print_timing()
        return im_pos

    def get_difference_image(self, verbose=True) -> Optional[ViewerImage]:

        factor = self.filter_params.imdiff_factor.float
        if self.paint_diff_cache is not None:
            use_cache = self.paint_diff_cache['imid'] == self.image_id and \
                        self.paint_diff_cache['imrefid'] == self.image_ref_id and \
                        self.paint_diff_cache['factor'] == factor
        else:
            use_cache = False

        if self._image is None or self._image_ref is None: return None
        if not use_cache:
            im1 = self._image.data
            im2 = self._image_ref.data
            # TODO: get factor from parameters ...
            # factor = int(self.diff_color_slider.value())
            print(f'factor = {factor}')
            print(f' im1.dtype {im1.dtype} im2.dtype {im2.dtype}')
            # Fast OpenCV code
            start = get_time()
            # positive diffs in unsigned 8 bits, OpenCV puts negative values to 0
            try:
                if im1.dtype.name == 'uint8' and im2.dtype.name == 'uint8':
                    diff_plus = cv2.subtract(im1, im2)
                    diff_minus = cv2.subtract(im2, im1)
                    res = cv2.addWeighted(diff_plus, factor, diff_minus, -factor, 127)
                    if verbose:
                        print(f" qtImageViewer.difference_image()  took {int((get_time() - start)*1000)} ms")
                        vmin = np.min(res)
                        vmax = np.max(res)
                        print(f"min-max diff = {vmin} - {vmax}")
                        histo,_ = np.histogram(res, bins=int(vmax-vmin+0.5), range=(vmin, vmax))
                        sum = 0
                        for v in range(vmin,vmax):
                            if v!=127:
                                nb = histo[v-vmin]
                                if nb >0:
                                    print(f"{v-127}:{nb} ",end='')
                                    sum += nb
                        print('')
                        print(f'nb pixel diff  {sum}')
                    res = ViewerImage(res,  precision=self._image.precision, 
                                            downscale=self._image.downscale,
                                            channels=self._image.channels)
                    self.paint_diff_cache = {  'imid': self.image_id, 'imrefid': self.image_ref_id, 
                        'factor': self.filter_params.imdiff_factor.float
                    }
                    self.diff_image = res
                else:
                    d = (im1.astype(np.float32)-im2.astype(np.float32))*factor
                    d[d<-127] = -127
                    d[d>128] = 128
                    d = (d+127).astype(np.uint8)*255
                    res = ViewerImage(d,  precision=8, 
                                            downscale=self._image.downscale,
                                            channels=self._image.channels)
                    self.paint_diff_cache = {  'imid': self.image_id, 'imrefid': self.image_ref_id, 
                        'factor': self.filter_params.imdiff_factor.float
                    }
                    self.diff_image = res
            except Exception as e:
                print(f"Error {e}")
                res = (im1!=im2).astype(np.uint8)*255
                res = ViewerImage(res,  precision=8, 
                                        downscale=self._image.downscale,
                                        channels=ImageFormat.CH_Y)
                self.diff_image = res

        return self.diff_image

    def paint_image(self):
        # print(f"paint_image display_timing {self.display_timing}")
        if self.trace_calls: t = trace_method(self.tab)
        self.start_timing()
        time0 = time1 = get_time()

        label_width = self.size().width()
        label_height = self.size().height()

        show_diff = self.show_image_differences and self._image is not self._image_ref and \
                    self._image is not None and \
                    self._image_ref is not None and self._image.data.shape == self._image_ref.data.shape

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
                        self.paint_cache['antialiasing'] == self.antialiasing and \
                        not self.show_overlay
        else:
            use_cache = False

        # if show_diff, compute the image difference (put it in cache??)
        if show_diff:
            # Cache does not work well with differences
            use_cache = False
            # don't save the difference
            current_image = self.get_difference_image()
        else:
            current_image = self._image

        if current_image is None: return
        precision  = current_image.precision
        downscale  = current_image.downscale
        channels   = current_image.channels

        # TODO: get data based on the display ratio?
        image_data = current_image.data

        # could_use_cache = use_cache
        # if could_use_cache:
        #     print(" Could use cache here ... !!!")
        # use_cache = False

        do_crop = (c[2] - c[0] != 1) or (c[3] - c[1] != 1)
        h, w  = image_data.shape[:2]
        if do_crop:
            crop_xmin = int(np.round(c[0] * w))
            crop_xmax = int(np.round(c[2] * w))
            crop_ymin = int(np.round(c[1] * h))
            crop_ymax = int(np.round(c[3] * h))
            image_data = image_data[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        else:
            crop_xmin = crop_ymin = 0
            crop_xmax = w
            crop_ymax = h

        cropped_image_shape = image_data.shape
        self.add_time('crop', time1)

        # time1 = get_time()
        image_height, image_width  = image_data.shape[:2]
        ratio_width = float(label_width) / image_width
        ratio_height = float(label_height) / image_height
        ratio = min(ratio_width, ratio_height)
        display_width = int(round(image_width * ratio))
        display_height = int(round(image_height * ratio))

        if self.show_overlay and self._image_ref is not self._image and self._image_ref and self._image and \
            self._image.data.shape == self._image_ref.data.shape:
            # to create the overlay rapidly, we will mix the two images based on the current cursor position
            # 1. convert cursor position to image position
            (height, width) = cropped_image_shape[:2]
            # compute rect
            rect = QtCore.QRect(0, 0, display_width, display_height)
            devRect = QtCore.QRect(0, 0, self.evt_width, self.evt_height)
            rect.moveCenter(devRect.center())
            im_x = int((self._mouse_events._mouse_x - rect.x()) / rect.width() * width)
            im_x = max(0,min(width-1, im_x))
            # im_y = int((self._mouse_events._mouse_y - rect.y()) / rect.height() * height)
            # We need to have a copy here .. slow, better option???
            image_data = np.copy(image_data)
            image_data[:, :im_x] = self._image_ref.data[crop_ymin:crop_ymax, crop_xmin:(crop_xmin+im_x)]

        resize_applied = False
        if not use_cache:
            anti_aliasing = ratio < 1
            #self.print_log("ratio is {:0.2f}".format(ratio))
            use_opencv_resize : bool = anti_aliasing
            # enable this as optional?
            # opencv_downscale_interpolation = cv2.fast_interpolation
            cv2.fast_interpolation = cv2.INTER_NEAREST
            if self.antialiasing:
                opencv_downscale_interpolation = cv2.INTER_AREA
            else:
                opencv_downscale_interpolation = cv2.INTER_NEAREST
            # opencv_upscale_interpolation   = cv2.INTER_LINEAR
            opencv_upscale_interpolation   = cv2.fast_interpolation
            # self.print_time('several settings', time1, start_time)

            # self.print_log("use_opencv_resize {} channels {}".format(use_opencv_resize, current_image.channels))
            # if ratio<1 we want anti aliasing and we want to resize as soon as possible to reduce computation time
            if use_opencv_resize and not resize_applied and channels == ImageFormat.CH_RGB:

                prev_shape = image_data.shape
                initial_type = image_data.dtype
                if image_data.dtype != np.uint8:
                    print(f"image_data type {type(image_data)} {image_data.dtype}")
                    image_data = image_data.astype(np.float32)

                # if ratio is >2, start with integer downsize which is much faster
                # we could add this condition opencv_downscale_interpolation==cv2.INTER_AREA
                if ratio<=0.5:
                    if image_data.shape[0]%2!=0 or image_data.shape[1]%2 !=0:
                        # clip image to multiple of 2 dimension
                        image_data = image_data[:2*(image_data.shape[0]//2),:2*(image_data.shape[1]//2)]
                    start_0 = get_time()
                    resized_image = cv2.resize(image_data, (image_width>>1, image_height>>1),
                                            interpolation=opencv_downscale_interpolation)
                    if self.display_timing:
                        print(f' === qtImageViewer: ratio {ratio:0.2f} paint_image() OpenCV resize from '
                            f'{current_image.data.shape} to '
                            f'{resized_image.shape} --> {int((get_time()-start_0)*1000)} ms')
                    image_data = resized_image
                    if ratio<=0.25:
                        if image_data.shape[0]%2!=0 or image_data.shape[1]%2 !=0:
                            # clip image to multiple of 2 dimension
                            image_data = image_data[:2*(image_data.shape[0]//2),:2*(image_data.shape[1]//2)]
                        start_0 = get_time()
                        resized_image = cv2.resize(image_data, (image_width>>2, image_height>>2),
                                                interpolation=opencv_downscale_interpolation)
                        if self.display_timing:
                            print(f' === qtImageViewer: ratio {ratio:0.2f} paint_image() OpenCV resize from '
                                f'{current_image.data.shape} to '
                                f'{resized_image.shape} --> {int((get_time()-start_0)*1000)} ms')
                        image_data = resized_image

                time1 = get_time()
                start_0 = get_time()
                resized_image = cv2.resize(image_data, (display_width, display_height),
                                        interpolation=opencv_downscale_interpolation)
                if self.display_timing:
                    print(f' === qtImageViewer: paint_image() OpenCV resize from {image_data.shape} to '
                        f'{resized_image.shape} --> {int((get_time()-start_0)*1000)} ms')

                image_data = resized_image.astype(initial_type)
                resize_applied = True
                self.add_time('cv2.resize',time1)

            current_image = ViewerImage(image_data,  precision=precision, downscale=downscale, channels=channels)
            if self.show_stats and self._image:
                # Output RGB from input
                ch = self._image.channels
                data_shape = current_image.data.shape
                if len(data_shape)==2:
                    print(f"input average {np.average(current_image.data)}")
                if len(data_shape)==3:
                    for c in range(data_shape[2]):
                        print(f"input average ch {c} {np.average(current_image.data[:,:,c])}")
            current_image = self.apply_filters(current_image)

            # Compute the histogram here, with the smallest image!!!
            if self.show_histogram:
                # previous version only python with its modules
                # histograms  = self.compute_histogram    (current_image, show_timings=self.display_timing)
                # new version with bound C++ code and openMP: much faster
                histograms = self.compute_histogram_Cpp(current_image, show_timings=self.display_timing)
            else:
                histograms = None

            # try to resize anyway with opencv since qt resizing seems too slow
            if not resize_applied and BaseWidget is not QOpenGLWidget:
                time1 = get_time()
                start_0 = get_time()
                prev_shape = current_image.shape
                current_image = cv2.resize(current_image, (display_width, display_height),
                                           interpolation=opencv_upscale_interpolation)
                if self.display_timing:
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
            if self.paint_cache:
                current_image = self.paint_cache['current_image']
                histograms = self.paint_cache['histograms']
            # histograms2 = self.paint_cache['histograms2']

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
                'histograms': histograms, 
                # 'histograms2': histograms2, 
                'current_image': current_image,
                'show_diff' : show_diff,
                'antialiasing': self.antialiasing
                }
            # print(f"create cache data took {int((get_time() - cache_time) * 1000)} ms")

        if not current_image.flags['C_CONTIGUOUS']:
            current_image = np.require(current_image, np.uint8, 'C')
        qimage = QtGui.QImage(current_image.data, current_image.shape[1], current_image.shape[0],
                                    current_image.strides[0], QtGui.QImage.Format_RGB888)
        # self.add_time('QtGui.QPixmap',time1)

        assert resize_applied, "Image resized should be applied at this point"

        if self._save_image_clipboard and self._clipboard:
            self.print_log("exporting to clipboard")
            self._clipboard.setImage(qimage, mode=QtGui.QClipboard.Mode.Clipboard)

        painter : QtGui.QPainter = QtGui.QPainter()

        painter.begin(self)
        if BaseWidget is QOpenGLWidget:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # TODO: check that this condition is not needed
        if BaseWidget is QOpenGLWidget:
            rect = QtCore.QRect(0,0, display_width, display_height)
        else:
            rect = QtCore.QRect(qimage.rect())
        devRect = QtCore.QRect(0, 0, self.evt_width, self.evt_height)
        rect.moveCenter(devRect.center())

        time1 = get_time()
        if BaseWidget is QOpenGLWidget:
            painter.drawImage(rect, qimage)
        else:
            painter.drawImage(rect.topLeft(), qimage)
        self.add_time('painter.drawImage',time1)

        if self.show_overlay:
            self.draw_overlay_separation(cropped_image_shape, rect, painter)

        # Draw cursor
        im_pos = None
        if self.show_cursor:
            im_pos = self.draw_cursor(cropped_image_shape, 
                                      crop_xmin, 
                                      crop_ymin, 
                                      rect, 
                                      painter, 
                                      full = self.show_intensity_line,
                                      )

        if self.show_intensity_line and self._image:
            (height, width) = cropped_image_shape[:2]
            im_y = int((self._mouse_events._mouse_y -rect.y())/rect.height()*height)
            im_y += crop_ymin
            im_shape = self._image.data.shape
            # Horizontal display
            if im_y>=0 and im_y<im_shape[0] and crop_xmin>=0 and crop_xmin+cropped_image_shape[1]<=im_shape[1]:
                line = self._image.data[im_y, crop_xmin:crop_xmin+cropped_image_shape[1]]
                self.display_intensity_line(
                    painter, 
                    rect, 
                    line,
                    channels = self._image.channels,
                    )

        self.display_text(painter, self.display_message(im_pos, ratio*self.devicePixelRatio()))

        # draw histogram
        if self.show_histogram:
            self.display_histogram(histograms, 1,  painter, rect, show_timings=self.display_timing)
            # self.display_histogram(histograms2, 2, painter, rect, show_timings=self.display_timing)

        painter.end()
        self.print_timing()

        if self.display_timing:
            print(f" paint_image took {int((get_time()-time0)*1000)} ms")

    def show(self):
        if BaseWidget==QOpenGLWidget:
            self.update()
        BaseWidget.show(self)

    def paintEvent(self, event):
        # print(f" qtImageViewer.paintEvent() {self.image_name}")
        if self.trace_calls:
            t = trace_method(self.tab)
        # try:
        if self._image is not None:
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
        BaseWidget.resizeEvent(self, event)
        self.print_log(f"resize {event.size()}  self {self.width()} {self.height()}")

    def mousePressEvent(self, event):
        self._mouse_events.mouse_press_event(event)

    def mouseMoveEvent(self, event):
        self._mouse_events.mouse_move_event(event)

    def mouseReleaseEvent(self, event):
        self._mouse_events.mouse_release_event(event)

    def mouseDoubleClickEvent(self, event):
        # We need to set the current viewer active before processing the double click event
        self.is_active = True
        self._mouse_events.mouse_double_click_event(event)

    def wheelEvent(self, event):
        self._mouse_events.mouse_wheel_event(event)

    def event(self, evt):
        if self.event_recorder is not None:
            self.event_recorder.store_event(self, evt)
        return BaseWidget.event(self, evt)

    def keyPressEvent(self, event):
        self.key_press_event(event, wsize=self.size())

    def keyReleaseEvent(self, evt):
        self.print_log(f"evt {evt.type()}")
