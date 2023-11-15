"""
    Interface to reader a jpeg using PyTurboJPEG module.

    Please note: you need to install the turbojpeg library on your system
    for the module to work, you can follow the installation instruction in https://github.com/lilohuang/PyTurboJPEG

"""

from qimview.utils.viewer_image import *
from qimview.utils.utils import get_time

try:
    from turbojpeg import TurboJPEG, TJPF_RGB, TJPF_BGR, TJFLAG_FASTDCT
except:
    has_turbojpeg = False
else:
    has_turbojpeg = True
import configparser
import os

config = configparser.ConfigParser()
config_file = os.path.expanduser('~/.qimview.cfg')
lib_path = None
if os.path.isfile(config_file): 
    config.read([config_file])
    try:
        lib_path = config['READER.TURBOJPEG']['LibPath']
    except Exception as e:
        pass

# Initialize once the TurboJPEG instance, as a global variable
try:
    if lib_path:
        gb_turbo_jpeg = TurboJPEG(lib_path)
    else:
        gb_turbo_jpeg = TurboJPEG()
except Exception as e:
    print(f"Failed to load TurboJPEG library")
    gb_turbo_jpeg = None

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
        #     im_header = gb_turbo_jpeg.decode_header(image_buffer)
        #     print(f" header {im_header} {int(get_time() - start1) * 1000} ms")
        im = gb_turbo_jpeg.decode(image_buffer, pixel_format=pixel_format, scaling_factor=scale, flags=flags)

        if verbose:
            print(f" turbojpeg read ...{image_filename[-15:]} took {get_time() - start:0.3f} sec.")
        viewer_image = ViewerImage(im, precision=8, downscale=downscale, channels=ImageFormat.CH_RGB if use_RGB else ImageFormat.CH_BGR)
        return viewer_image
    except Exception as e:
        print("read_jpeg: Failed to load image with turbojpeg {0}: {1}".format(image_filename, e))
        return None

