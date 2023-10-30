from qimview.utils.ViewerImage import *
import os
from .libraw_reader import libraw_supported_formats, read_libraw
from .turbojpeg_reader import read_jpeg_turbojpeg, gb_turbo_jpeg
from .simplejpeg_reader import read_jpeg_simplejpeg
from .opencv_reader import read_opencv, opencv_supported_formats
from typing import Optional, TYPE_CHECKING

# Avoid circular imports
# Imports specific for type checking
if TYPE_CHECKING:
    from qimview.cache import FileCache

def read_jpeg(image_filename, image_buffer, read_size='full', use_RGB=True, verbose=False):
    # in terms of speed, it seems that turbojpeg is faster than simplejpeg which is faster than cv2
    # so turbojpeg > simplejpeg > cv2
    verbose = True
    im1 = im2 = im3 = None
    if gb_turbo_jpeg:
        im1 = read_jpeg_turbojpeg(image_filename, image_buffer, read_size=read_size, use_RGB=use_RGB, verbose=verbose)
    if im1 is not None: return im1
    if read_size == 'full':
        im2 = read_jpeg_simplejpeg(image_filename, image_buffer, read_size, use_RGB, verbose)
    if im2 is not None: return im2
    im3 = read_opencv(image_filename, image_buffer, read_size, use_RGB, verbose)
    return im3


class ImageReader:
    def __init__(self):
        # set default plugins
        self._plugins = {
            ".JPG":read_jpeg,
            ".JPEG":read_jpeg,
        }
        # add libraw supported extensions
        for ext in libraw_supported_formats():
            if ext.upper() not in self._plugins:
                self._plugins[ext.upper()] = read_libraw
        # add all opencv supposedly supported extensions
        for ext in opencv_supported_formats():
            if ext.upper() not in self._plugins:
                self._plugins[ext.upper()] = read_opencv
        self.file_cache : Optional[FileCache] = None

    def extensions(self):
        return list(self._plugins.keys())

    def set_file_cache(self, file_cache):
        self.file_cache = file_cache

    def set_plugin(self, extensions, callback):
        """ Set support to a image format based on list of extensions and callback

        Args:
            extensions ([type]): [description]
            callback (function): callback has signature (image_filename, image_buffer, read_size='full', use_RGB=True, verbose=False) 
            and returns a ViewerImage object
        """
        for ext in extensions:
            self._plugins[ext.upper()] = callback

    def read(self, filename, buffer=None, read_size='full', use_RGB=True, verbose=False, check_filecache_size=True):
        extension = os.path.splitext(filename)[1].upper()
        if extension not in self._plugins:
            print(  f"ERROR: ImageRead.read({filename}) extension not supported, "
                    f"supported extensions are {self._plugins.keys()}")
            return None

        try:
            if buffer is None and self.file_cache is not None:
                # try to get the buffer from the file cache
                buffer, fromcache = self.file_cache.get_file(filename, check_size=check_filecache_size)
                print(f" got buffer from cache? {fromcache}")
            res = self._plugins[extension](filename, buffer, read_size, use_RGB, verbose)
            return res
        except Exception as e:
            print(f"Exception while reading image {filename}: {e}")
            return None

# unique instance of ImageReader for the application
gb_image_reader = ImageReader()


def reader_add_plugins():
    print("reader_add_plugins()")
    import configparser, sys
    config = configparser.ConfigParser()
    # config.read_file(open('default.cfg'))
    res = config.read([os.path.expanduser('~/.qimview.cfg')])
    # if read fails, found is probably not found
    if res:
        # Add new image format support
        try:
            formats = config['READERS']['Formats'].split(',')
        except Exception as e:
            print(f" ----- No reader plugin in config file: {e}")
        for fmt in  formats:
            try:
                format_cfg = config[f'READER.{fmt.upper()}']
                folder, module, ext = format_cfg['Folder'], format_cfg['Module'], format_cfg['Extensions'].split(',')
                print(f' {fmt} {folder, ext}')
                # TODO: change this code, avoid sys.path.append()
                sys.path.append(folder)
                import importlib
                fmt_reader = importlib.import_module(f"{module}")
                gb_image_reader.set_plugin(ext, fmt_reader.read)
            except Exception as e:
                print(f" ----- Failed to add support for {fmt}: {e}")

reader_add_plugins()
