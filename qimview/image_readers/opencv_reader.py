from qimview.utils.viewer_image import *
from qimview.utils.utils import get_time
import cv2
import numpy as np


def read_opencv(image_filename, image_buffer, read_size='full', use_RGB=True, verbose=False):
    if verbose:
        open_cv2_start = get_time()
    cv_size = {'full': cv2.IMREAD_COLOR,
                '1/2': cv2.IMREAD_REDUCED_COLOR_2,
                '1/4': cv2.IMREAD_REDUCED_COLOR_4,
                '1/8': cv2.IMREAD_REDUCED_COLOR_8,
                }
    flags = cv_size[read_size]
    if image_buffer is None:
        cv2_im = cv2.imread(image_filename, flags)
    else:
        bytes_as_np_array = np.frombuffer(image_buffer, dtype=np.uint8)
        cv2_im = cv2.imdecode(bytes_as_np_array, flags) 

    open_cv2_start2 = get_time()
    # transform default opencv BGR to RGB
    if use_RGB:
        im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

    if verbose:
        last_time = get_time()
        print(" cv2 imread {0} took {1:0.3f} ( {2:0.3f} + {3:0.3f} ) sec.".format(image_filename, last_time - open_cv2_start,
                                                                    open_cv2_start2 - open_cv2_start,
                                                                    last_time - open_cv2_start2),
                )
    downscale = {'full': 1, '1/2': 2, '1/4': 4, '1/8': 8}[read_size]
    viewer_image = ViewerImage(im, precision=8, downscale=downscale, channels=CH_RGB if use_RGB else CH_BGR)
    return viewer_image


def opencv_supported_formats():
    # according to the doc, opencv supports the following formats
    # Windows bitmaps - *.bmp, *.dib (always supported)
    # JPEG files - *.jpeg, *.jpg, *.jpe (see the Note section)
    # JPEG 2000 files - *.jp2 (see the Note section)
    # Portable Network Graphics - *.png (see the Note section)
    # WebP - *.webp (see the Note section)
    # Portable image format - *.pbm, *.pgm, *.ppm *.pxm, *.pnm (always supported)
    # PFM files - *.pfm (see the Note section)
    # Sun rasters - *.sr, *.ras (always supported)
    # TIFF files - *.tiff, *.tif (see the Note section)
    # OpenEXR Image files - *.exr (see the Note section)
    # Radiance HDR - *.hdr, *.pic (always supported)
    # Raster and Vector geospatial data supported by GDAL (see the Note section)    
    return [ '.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.ppm', '.pxm', 
            '.pnm', '.pfm', '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic']

