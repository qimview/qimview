
# from https://numpy.org/doc/stable/user/basics.subclassing.html

import numpy as np

CH_RGB = 1
CH_BGR = 2
CH_Y   = 3
CH_RGGB = 4  # phase 0, bayer 2
CH_GRBG = 5  # phase 1, bayer 3 (Boilers)
CH_GBRG = 6  # phase 2, bayer 0
CH_BGGR = 7  # phase 3, bayer 1 (Coconuts)
CH_RAWFORMATS    = [CH_RGGB, CH_GRBG, CH_GBRG, CH_BGGR]
CH_RGBFORMATS    = [CH_RGB, CH_BGR]
CH_SCALARFORMATS = [CH_Y]

channel_position = {
    CH_RGGB: {'r' :0, 'gr':1, 'gb':2, 'b' :3},
    CH_GRBG: {'gr':0, 'r' :1, 'b' :2, 'gb':3},
    CH_GBRG: {'gb':0, 'b' :1, 'r' :2, 'gr':3},
    CH_BGGR: {'b' :0, 'gb':1, 'gr':2, 'r' :3}
}

class ViewerImage(np.ndarray):
    """
    Own image class that inherits from np.ndarray
    """

    def __new__(cls, input_array, precision=8, downscale=1, channels=CH_RGB):
        """
        :param input_array:
        :param precision: Integer image precision (number of bits)
        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.precision = precision
        obj.downscale = downscale
        obj.channels  = channels
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.precision = getattr(obj, 'precision', 8)
        self.downscale = getattr(obj, 'downscale', 1)
        self.channels = getattr(obj, 'channels', CH_RGB)
