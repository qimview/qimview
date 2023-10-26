"""
    Interface to reader a jpeg using PyTurboJPEG module.

    Please note: you need to install the turbojpeg library on your system
    for the module to work, you can follow the installation instruction in https://github.com/lilohuang/PyTurboJPEG

"""

from qimview.utils.ViewerImage import *
from qimview.utils.utils import get_time
from turbojpeg import TurboJPEG, TJPF_RGB, TJPF_BGR, TJFLAG_FASTDCT

# Initialize once the TurboJPEG instance, as a global variable
jpeg = TurboJPEG()

def read_jpeg_turbojpeg(image_filename, image_buffer, read_size='full', use_RGB=True, verbose=False):
    try:
        if verbose:
            start = get_time()
        # using default library installation
        # decoding input.jpg to BGR array
        downscale = {'full': 1, '1/2': 2, '1/4': 4, '1/8': 8}[read_size]
        scale = (1, downscale)

        # do we need the fast DCT?
        flags = TJFLAG_FASTDCT
        pixel_format = TJPF_RGB if use_RGB else TJPF_BGR

        if image_buffer is None:
            with open(image_filename, 'rb') as d:
                image_buffer = d.read()
        # if verbose:
        #     im_header = jpeg.decode_header(image_buffer)
        #     print(f" header {im_header} {int(get_time() - start1) * 1000} ms")
        im = jpeg.decode(image_buffer, pixel_format=pixel_format, scaling_factor=scale, flags=flags)

        if verbose:
            print(f" turbojpeg read ...{image_filename[-15:]} took {get_time() - start:0.3f} sec.")
        viewer_image = ViewerImage(im, precision=8, downscale=downscale, channels=CH_RGB if use_RGB else CH_BGR)
        return viewer_image
    except Exception as e:
        print("read_jpeg: Failed to load image with turbojpeg {0}: {1}".format(image_filename, e))
        return None

