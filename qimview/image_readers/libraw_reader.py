import rawpy
import math
import numpy as np
from  qimview.utils.ViewerImage import ViewerImage, CH_RGGB

def read_libraw(image_filename, image_buffer, read_size='full', use_RGB=True, verbose=False):
    if image_buffer:
        raw = rawpy.imread(image_buffer)
    else:
        raw = rawpy.imread(image_filename)
    height, width = raw.raw_image.shape
    print(f"height, width {(height, width)}")
    # Transform Bayer image
    im1 = np.empty((height >> 1, width >> 1, 4), dtype=raw.raw_image.dtype)
    bayer_desc = raw.color_desc.decode('utf-8')
    bayer_pattern = raw.raw_pattern
    # not perfect, we may switch Gr and Gb
    bayer_desc = bayer_desc.replace('G', 'g', 1)
    # Detect the green channels?
    channel_pos = {'R': 0, 'g': 1, 'G': 2, 'B': 3}
    for i in range(2):
        for j in range(2):
            print("str(bayer_desc[bayer_pattern[i,j]]) {}".format(
                str(bayer_desc[bayer_pattern[i, j]])))
            pos = channel_pos[str(bayer_desc[bayer_pattern[i, j]])]
            im1[:, :, pos] = raw.raw_image[i::2, j::2]
    prec = math.ceil(math.log2(raw.white_level+1))
    viewer_image = ViewerImage(im1, precision=prec, downscale=1, channels=CH_RGGB)
    print("viewer_image channels {}".format(viewer_image.channels))
    return viewer_image


def libraw_supported_formats():
    # Need to find the complete list of supported formats
    return [".ARW", ".GPR"]

