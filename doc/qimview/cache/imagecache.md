Module qimview.cache.imagecache
===============================

Classes
-------

`ImageCache()`
:   Save output bytes from read() function into a cache indexed by the filename
        inherits from BaseCache, with
            id as string: input filename
            ViewerImage: image object
            mtime: modification time as float from osp.getmtime(filename)
        If a file is in the cache but its modification time on disk is more recent,
        we can enable an automatic reload
    
    Args:
        BaseCache (_type_): _description_

    ### Ancestors (in MRO)

    * qimview.cache.basecache.BaseCache
    * typing.Generic

    ### Methods

    `add_image(self, filename, read_size='full', verbose=False, use_RGB=True, image_transform=None, progress_callback=None)`
    :   :param filename:
        :param show_timing:
        :return: pair image_data, boolean (True is coming from cache)

    `add_images(self, filenames, read_size='full', verbose=False, use_RGB=True, image_transform=None)`
    :

    `add_result(self, r)`
    :

    `get_image(self, filename, read_size='full', verbose=False, use_RGB=True, image_transform=None, check_size=True)`
    :   :param filename:
        :param show_timing:
        :return: pair image_data, boolean (True is coming from cache)

    `has_image(self, filename)`
    :

    `image_added(self, filename)`
    :