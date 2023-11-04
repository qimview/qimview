Module qimview.cache.filecache
==============================

Classes
-------

`FileCache()`
:   Save output bytes from read() function into a cache indexed by the filename
        inherits from BaseCache, with
            id as string: input filename
            bytes: output from binary read()
            mtime: modification time as float from osp.getmtime(filename)
        If a file is in the cache but its modification time on disk is more recent,
        we can enable an automatic reload
    
    Args:
        BaseCache (_type_): _description_

    ### Ancestors (in MRO)

    * qimview.cache.basecache.BaseCache
    * typing.Generic

    ### Methods

    `add_files(self, filenames)`
    :

    `add_result(self, r)`
    :

    `file_added(self, filename)`
    :

    `get_file(self, filename: str, check_size: bool = True) ‑> Tuple[Optional[bytes], bool]`
    :   _summary_
        
        Args:
            filename (str): file to read, is supposed to exist on disk, if not an exception will be raised
                by Python standard methods
            check_size (bool, optional): _description_. Defaults to True.
        
        Returns:
            Tuple[Optional[bytes], bool]: first elt is the data read 
                second is a boolean saying if it comes or not from the cache

    `has_file(self, filename)`
    :

    `on_finished(self)`
    :

    `show_progress(self, val)`
    :

    `thread_add_files(self, filenames, progress_callback=None)`
    :   :param filename:
        :param show_timing:
        :return: pair image_data, boolean (True is coming from cache)