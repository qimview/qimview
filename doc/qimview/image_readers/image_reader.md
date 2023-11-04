Module qimview.image_readers.image_reader
=========================================

Functions
---------

    
`read_jpeg(image_filename, image_buffer, read_size='full', use_RGB=True, verbose=False)`
:   

    
`reader_add_plugins()`
:   

Classes
-------

`ImageReader()`
:   

    ### Methods

    `extensions(self)`
    :

    `read(self, filename, buffer=None, read_size='full', use_RGB=True, verbose=False, check_filecache_size=True)`
    :

    `set_file_cache(self, file_cache)`
    :

    `set_plugin(self, extensions, callback)`
    :   Set support to a image format based on list of extensions and callback
        
        Args:
            extensions ([type]): [description]
            callback (function): callback has signature (image_filename, image_buffer, read_size='full', use_RGB=True, verbose=False) 
            and returns a ViewerImage object