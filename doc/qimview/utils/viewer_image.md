Module qimview.utils.viewer_image
=================================

Classes
-------

`ImageFormat(value, names=None, *, module=None, qualname=None, type=None, start=1)`
:   An enumeration.

    ### Ancestors (in MRO)

    * enum.IntEnum
    * builtins.int
    * enum.Enum

    ### Class variables

    `CH_BGGR`
    :

    `CH_BGR`
    :

    `CH_GBRG`
    :

    `CH_GRBG`
    :

    `CH_RGB`
    :

    `CH_RGGB`
    :

    `CH_Y`
    :

    ### Static methods

    `CH_RAWFORMATS() ‑> Tuple[qimview.utils.viewer_image.ImageFormat, ...]`
    :

    `CH_RGBFORMATS() ‑> Tuple[qimview.utils.viewer_image.ImageFormat, ...]`
    :

    `CH_SCALARFORMATS() ‑> Tuple[qimview.utils.viewer_image.ImageFormat, ...]`
    :

`ViewerImage(input_array: np.ndarray, precision: int = 8, downscale=1, channels: ImageFormat = ImageFormat.CH_RGB)`
:   Own image class that inherits from np.ndarray
    
    :param input_array:
    :param precision: Integer image precision (number of bits)

    ### Instance variables

    `data: numpy.ndarray`
    :

    ### Methods

    `get_data_for_ratio(self, ratio, display_timing=False)`
    :

    `reduce_half(self, input_data, interpolation=3, display_timing=False)`
    :

    `set_filename(self, fn)`
    :