Module qimview.image_viewers.image_viewer
=========================================

Functions
---------

    
`get_class_from_frame(fr)`
:   

    
`get_function_name()`
:   

Classes
-------

`ImageViewer(parent=None)`
:   

    ### Descendants

    * qimview.image_viewers.gl_image_viewer_base.GLImageViewerBase
    * qimview.image_viewers.qt_image_viewer.QTImageViewer

    ### Instance variables

    `display_timing`
    :

    `image_name: str`
    :

    `verbose`
    :

    ### Methods

    `add_time(self, mess, current_start, force=False, title=None)`
    :

    `check_translation(self)`
    :

    `compute_histogram(self, current_image, show_timings=False)`
    :

    `compute_histogram_Cpp(self, current_image, show_timings=False)`
    :

    `display_histogram(self, hist_all, id, painter, im_rect, show_timings=False)`
    :   :param painter:
        :param rect: displayed image area
        :return:

    `display_message(self, im_pos: Optional[Tuple[int, int]], scale=None) ‑> str`
    :

    `display_text(self, painter: PySide6.QtGui.QPainter, text: str) ‑> None`
    :

    `find_in_layout(self, layout: PySide6.QtWidgets.QLayout) ‑> Optional[PySide6.QtWidgets.QLayout]`
    :   Search Recursivement in Layouts for the current widget
        
        Args:
            layout (QtWidgets.QLayout): input layout for search
        
        Returns:
            layout containing the current widget or None if not found

    `get_image(self)`
    :

    `is_active(self)`
    :

    `key_press_event(self, event, wsize)`
    :

    `mouse_double_click_event(self, event)`
    :

    `mouse_move_event(self, event)`
    :

    `mouse_press_event(self, event)`
    :

    `mouse_release_event(self, event)`
    :

    `mouse_wheel_event(self, event)`
    :

    `new_scale(self, mouse_zy, height)`
    :

    `new_translation(self)`
    :

    `print_log(self, mess, force=False)`
    :

    `print_timing(self, add_total=False, force=False, title=None)`
    :

    `set_active(self, active=True)`
    :

    `set_clipboard(self, clipboard, save_image)`
    :

    `set_image(self, image)`
    :

    `set_image_ref(self, image_ref=None)`
    :

    `set_synchronize(self, viewer)`
    :

    `start_timing(self, title=None)`
    :

    `synchronize(self, event_viewer)`
    :   This method needs to be overloaded with call to self.synchronize_viewer.synchronize()
        :param event_viewer: the viewer that started the synchronization
        :return:

    `synchronize_data(self, other_viewer)`
    :

    `toggle_fullscreen(self, event)`
    :

    `viewer_update(self)`
    :

`trace_method(tab)`
: