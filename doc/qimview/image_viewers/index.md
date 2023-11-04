Module qimview.image_viewers
============================

Sub-modules
-----------
* qimview.image_viewers.gl_image_viewer
* qimview.image_viewers.gl_image_viewer_base
* qimview.image_viewers.gl_image_viewer_shaders
* qimview.image_viewers.image_filter_parameters
* qimview.image_viewers.image_filter_parameters_gui
* qimview.image_viewers.image_viewer
* qimview.image_viewers.multi_view
* qimview.image_viewers.qt_image_viewer

Classes
-------

`GLImageViewer(parent=None, event_recorder=None)`
:   QOpenGLWidget(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * qimview.image_viewers.gl_image_viewer_base.GLImageViewerBase
    * PySide6.QtOpenGLWidgets.QOpenGLWidget
    * PySide6.QtWidgets.QWidget
    * PySide6.QtCore.QObject
    * PySide6.QtGui.QPaintDevice
    * Shiboken.Object
    * qimview.image_viewers.image_viewer.ImageViewer

    ### Class variables

    `staticMetaObject`
    :

    ### Methods

    `event(self, evt)`
    :   event(self, e: PySide6.QtCore.QEvent) -> bool

    `initializeGL(self)`
    :   Initialize OpenGL, VBOs, upload data on the GPU, etc.

    `myPaintGL(self)`
    :   Paint the scene.

    `paintGL(self)`
    :   paintGL(self) -> None

    `viewer_update(self)`
    :

`GLImageViewerShaders(parent=None)`
:   QOpenGLWidget(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * qimview.image_viewers.gl_image_viewer_base.GLImageViewerBase
    * PySide6.QtOpenGLWidgets.QOpenGLWidget
    * PySide6.QtWidgets.QWidget
    * PySide6.QtCore.QObject
    * PySide6.QtGui.QPaintDevice
    * Shiboken.Object
    * qimview.image_viewers.image_viewer.ImageViewer

    ### Class variables

    `fragmentShader_RAW`
    :

    `fragmentShader_RGB`
    :

    `staticMetaObject`
    :

    `vertexShader`
    :

    ### Methods

    `initializeGL(self)`
    :   Initialize OpenGL, VBOs, upload data on the GPU, etc.

    `myPaintGL(self)`
    :   Paint the scene.

    `paintGL(self)`
    :   paintGL(self) -> None

    `setBufferData(self)`
    :

    `setVerticesBufferData(self)`
    :

    `set_shaders(self)`
    :

    `updateTransforms(self) ‑> float`
    :

    `viewer_update(self)`
    :

`ImageFilterParameters()`
:   

    ### Methods

    `copy_from(self, p)`
    :

    `is_equal(self, other)`
    :

`ImageFilterParametersGui(parameters, name='')`
:   :param parameters: instance of ImageFilterParameters

    ### Methods

    `add_blackpoint(self, layout, callback)`
    :

    `add_g_b(self, layout, callback)`
    :

    `add_g_r(self, layout, callback)`
    :

    `add_gamma(self, layout, callback)`
    :

    `add_imdiff_factor(self, layout, callback)`
    :

    `add_saturation(self, layout, callback)`
    :

    `add_whitepoint(self, layout, callback)`
    :

    `register_event_player(self, event_player)`
    :

    `reset_all(self)`
    :

    `set_event_recorder(self, evtrec)`
    :

`MultiView(parent=None, viewer_mode=ViewerType.QT_VIEWER, nb_viewers=1)`
:   QWidget(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    :param parent:
    :param viewer_mode:
    :param nb_viewers_used:

    ### Ancestors (in MRO)

    * PySide6.QtWidgets.QWidget
    * PySide6.QtCore.QObject
    * PySide6.QtGui.QPaintDevice
    * Shiboken.Object

    ### Class variables

    `staticMetaObject`
    :

    ### Methods

    `add_context_menu(self)`
    :

    `cache_read_images(self, image_filenames: List[str], reload: bool = False) ‑> None`
    :   Read the list of images into the cache, with option to reload them from disk
        
        Args:
            image_filenames (List[str]): list of image filenames
            reload (bool, optional): reload removes first the images from the ImageCache 
                before adding them. Defaults to False.

    `check_verbosity(self, flag)`
    :

    `clear_buttons(self)`
    :

    `create_buttons(self)`
    :

    `find_in_layout(self, layout: PySide6.QtWidgets.QLayout) ‑> Optional[PySide6.QtWidgets.QLayout]`
    :   Search Recursivement in Layouts for the current widget
        
        Args:
            layout (QtWidgets.QLayout): input layout for search
        
        Returns:
            layout containing the current widget or None if not found

    `get_output_image(self, im_string_id)`
    :   Search for the image with given label in the current row
        if not in cache reads it and add it to the cache
        :param im_string_id: string that identifies the image to display
        :return:

    `keyPressEvent(self, event)`
    :   keyPressEvent(self, event: PySide6.QtGui.QKeyEvent) -> None

    `keyReleaseEvent(self, event)`
    :   keyReleaseEvent(self, event: PySide6.QtGui.QKeyEvent) -> None

    `layout_buttons(self, vertical_layout)`
    :

    `layout_parameters(self, parameters_layout)`
    :

    `layout_parameters_2(self, parameters2_layout)`
    :

    `make_mouse_double_click(self, image_name)`
    :

    `make_mouse_press(self, image_name)`
    :

    `mouse_release(self, event)`
    :

    `print_log(self, mess)`
    :

    `reset_intensities(self)`
    :

    `reset_viewers(self)`
    :

    `setMessage(self, mess)`
    :

    `set_cache_memory_bar(self, progress_bar)`
    :

    `set_images(self, images, set_viewers=False)`
    :

    `set_key_down_callback(self, c)`
    :

    `set_key_up_callback(self, c)`
    :

    `set_message_callback(self, message_cb)`
    :

    `set_read_size(self, read_size)`
    :

    `set_reference_label(self, ref: str, update_viewers=False) ‑> None`
    :

    `set_verbosity(self, flag, enable=True)`
    :   :param v: verbosity flags
        :param b: boolean to enable or disable flag
        :return:

    `set_viewer_images(self)`
    :   Set viewer images based on self.image_dict.keys()
        :return:

    `show_context_menu(self, pos)`
    :

    `show_timing(self)`
    :

    `show_timing_detailed(self)`
    :

    `show_trace(self)`
    :

    `toggle_display_profiles(self)`
    :

    `toggle_fullscreen(self, event)`
    :

    `update_image(self, image_name=None, reload=False)`
    :   Uses the variable self.output_label_current_image
        :return:

    `update_image_buttons(self)`
    :

    `update_image_intensity_event(self)`
    :

    `update_image_parameters(self)`
    :   Uses the variable self.output_label_current_image
        :return:

    `update_label_fonts(self)`
    :

    `update_layout(self)`
    :

    `update_reference(self) ‑> None`
    :

    `update_viewer_layout(self, layout_name='1')`
    :

    `update_viewer_layout_callback(self)`
    :

    `update_viewer_mode(self)`
    :

`QTImageViewer(parent=None, event_recorder=None)`
:   QWidget(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = None, f: PySide6.QtCore.Qt.WindowType = Default(Qt.WindowFlags)) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * PySide6.QtWidgets.QWidget
    * PySide6.QtCore.QObject
    * PySide6.QtGui.QPaintDevice
    * Shiboken.Object
    * qimview.image_viewers.image_viewer.ImageViewer

    ### Class variables

    `staticMetaObject`
    :

    ### Methods

    `apply_filters(self, current_image)`
    :

    `apply_translation(self, crop)`
    :   :param crop:
        :return: the new crop

    `apply_zoom(self, crop)`
    :

    `check_translation(self)`
    :   This method computes the translation really applied based on the current requested translation
        :return:

    `draw_cursor(self, cropped_image_shape, crop_xmin, crop_ymin, rect, painter) ‑> Optional[Tuple[int, int]]`
    :   :param cropped_image_shape: dimensions of current crop
        :param crop_xmin: left pixel of current crop
        :param crop_ymin: top pixel of current crop
        :param rect: displayed image area
        :param painter:
        :return:
            tuple: (posx, posy) image pixel position of the cursor, if None cursor is out of image

    `draw_overlay_separation(self, cropped_image_shape, rect, painter)`
    :

    `event(self, evt)`
    :   event(self, event: PySide6.QtCore.QEvent) -> bool

    `get_difference_image(self, verbose=True)`
    :

    `keyPressEvent(self, event)`
    :   keyPressEvent(self, event: PySide6.QtGui.QKeyEvent) -> None

    `keyReleaseEvent(self, evt)`
    :   keyReleaseEvent(self, event: PySide6.QtGui.QKeyEvent) -> None

    `mouseDoubleClickEvent(self, event)`
    :   mouseDoubleClickEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `mouseMoveEvent(self, event)`
    :   mouseMoveEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `mousePressEvent(self, event)`
    :   mousePressEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `mouseReleaseEvent(self, event)`
    :   mouseReleaseEvent(self, event: PySide6.QtGui.QMouseEvent) -> None

    `paintEvent(self, event)`
    :   paintEvent(self, event: PySide6.QtGui.QPaintEvent) -> None

    `paint_image(self)`
    :

    `resizeEvent(self, event)`
    :   Called upon window resizing: reinitialize the viewport.

    `set_image(self, image)`
    :

    `show(self)`
    :   show(self) -> None

    `update_crop(self)`
    :

    `update_crop_new(self)`
    :

    `viewer_update(self)`
    :

    `wheelEvent(self, event)`
    :   wheelEvent(self, event: PySide6.QtGui.QWheelEvent) -> None

`ViewerType(value, names=None, *, module=None, qualname=None, type=None, start=1)`
:   An enumeration.

    ### Ancestors (in MRO)

    * enum.Enum

    ### Class variables

    `OPENGL_SHADERS_VIEWER`
    :

    `OPENGL_VIEWER`
    :

    `QT_VIEWER`
    :