from qimview.utils.viewer_image import *
from qimview.utils.utils import get_time
import os
from typing import Optional
import simplejpeg


def read_jpeg_simplejpeg(image_filename, image_buffer, read_size='full', use_RGB=True, verbose=False) -> Optional[ViewerImage]:
    # print(f"read_jpeg_simplejpeg use_RGB {use_RGB} running ...")
    if verbose:
        start_time = get_time()
    format = 'RGB' if use_RGB else 'BGR'
    try:
        if image_buffer is None:
            with open(image_filename, 'rb') as d:
                image_buffer = d.read()
    except IOError as e:
        print("read_jpeg_simplejpeg: I/O error({0}): {1}".format(e.errno, e.strerror))
    except Exception as e: #handle other exceptions such as attribute errors
        print(f"read_jpeg_simplejpeg: Unexpected error: {e}")
        return None

    try:
        im = simplejpeg.decode_jpeg(image_buffer, format)
    except Exception as e:
        print(f"read_jpeg_simplejpeg: Failed to decode jpeg {e}")
        return None

    if verbose:
        end_time = get_time()
        print(f" simplejpeg.decode_jpeg() {format} {os.path.basename(image_filename)} "
                f"took {int((end_time-start_time)*1000+0.5)} ms")
    viewer_image = ViewerImage(im, precision=8, downscale=1, channels=ImageFormat.CH_RGB if use_RGB else ImageFormat.CH_BGR)
    return viewer_image
