
# from https://numpy.org/doc/stable/user/basics.subclassing.html

from __future__ import annotations
from typing import Tuple, Optional
from enum import Enum, IntEnum
import numpy as np
import cv2
from .utils import get_time

class ImageFormat(IntEnum):
    CH_RGB  = 1 
    """RGB colour image"""
    CH_BGR  = 2 
    "RBG with inverted order"
    CH_Y    = 3 
    "One channel greyscale image"
    CH_RGGB = 4 
    "phase 0, bayer 2"
    CH_GRBG = 5 
    "phase 1, bayer 3" 
    CH_GBRG = 6 
    "phase 2, bayer 0"
    CH_BGGR = 7 
    "phase 3, bayer 1"
    CH_YUV420 = 8
    "YUV 420 format"

    @staticmethod
    def CH_RAWFORMATS() -> Tuple[ImageFormat, ...]:
        return (ImageFormat.CH_RGGB, ImageFormat.CH_GRBG, ImageFormat.CH_GBRG, ImageFormat.CH_BGGR,)
    
    @staticmethod
    def CH_RGBFORMATS() -> Tuple[ImageFormat, ...]:
        return (ImageFormat.CH_RGB, ImageFormat.CH_BGR,)

    @staticmethod
    def CH_SCALARFORMATS() -> Tuple[ImageFormat, ...]:
        return (ImageFormat.CH_Y,)

channel_position = {
    ImageFormat.CH_RGGB: {'r' :0, 'gr':1, 'gb':2, 'b' :3},
    ImageFormat.CH_GRBG: {'gr':0, 'r' :1, 'b' :2, 'gb':3},
    ImageFormat.CH_GBRG: {'gb':0, 'b' :1, 'r' :2, 'gr':3},
    ImageFormat.CH_BGGR: {'b' :0, 'gb':1, 'gr':2, 'r' :3},
}

class ViewerImage:
    """
    Own image class that inherits from np.ndarray
    """

    def __init__(self, 
                 input_array : np.ndarray, 
                 precision : int =8, 
                 downscale=1, 
                 channels: ImageFormat = ImageFormat.CH_RGB,
                 ):
        """
        :param input_array:
        :param precision: Integer image precision (number of bits)
        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        self._data = input_array
        # add the new attribute to the created instance
        self.precision : int           = precision
        self.downscale : int           = downscale
        self.channels  : ImageFormat   = channels
        self.filename  : Optional[str] = None
        self.data_reduced_2 = None
        self.data_reduced_4 = None
        # For YUV format, _data contains Y and _u and _v contain U and V
        self._u : Optional[np.ndarray] = None
        self._v : Optional[np.ndarray] = None


    @property
    def y(self) -> Optional[np.ndarray] :
        return self._data

    @y.setter
    def y(self, d : np.ndarray):
        self._data = d

    @property
    def u(self) -> Optional[np.ndarray] :
        return self._u

    @u.setter
    def u(self, d : np.ndarray):
        self._u = d

    @property
    def v(self) -> Optional[np.ndarray] :
        return self._v

    @v.setter
    def v(self, d : np.ndarray):
        self._v = d

    @property
    def data(self) -> np.ndarray :
        return self._data

    @data.setter
    def data(self, d : np.ndarray):
        self._data = d

    def reduce_half(self, input_data, interpolation=cv2.INTER_AREA, display_timing=False):
        image_height, image_width  = input_data.shape[:2]
        start_0 = get_time()
        if image_height % 2 != 0 or image_width % 2 != 0:
            # clip image to multiple of 2 dimension
            input_2 = input_data[:2*(image_height//2),:2*(image_width[1]//2)]
        else:
            input_2 = input_data
        data_2 = cv2.resize(input_2, (image_width>>1, image_height>>1), interpolation=interpolation)
        if display_timing:
            print(  f' === ViewerImage.reduce_half(): OpenCV resize from {input_data.shape} to '
                    f'{data_2.shape} --> {int((get_time()-start_0)*1000)} ms')
        return data_2

    # Seems interesting to have this method, but it is not so obvious
    def get_data_for_ratio(self, ratio, display_timing=False):
        prev_shape = self._data.shape
        image_height, image_width  = self._data.shape[:2]
        downscale_interpolation = cv2.INTER_AREA
        # if ratio is >2, start with integer downsize which is much faster
        # we could add this condition opencv_downscale_interpolation==cv2.INTER_AREA
        if ratio<=0.5:
            data_2 = self.reduce_half(self._data, interpolation=downscale_interpolation, display_timing=display_timing)
            if ratio<=0.25:
                data_4 = self.reduce_half(data_2, interpolation=downscale_interpolation, display_timing=display_timing)
                return data_4
            else:
                return data_2
        else:
            return self._data

    def set_filename(self, fn):
        self.filename = fn

    def __sizeof__(self):
        # approximative estimation
        size = self.data.nbytes
        for v in vars(self):
            # print(f" v {v} {self.__dict__[v].__sizeof__()}")
            size += self.__dict__[v].__sizeof__()
        return size
