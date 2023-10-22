"""
    Viewer for that takes numpy arrays as input
"""

import os
import glob
import numpy as np

from qimview.utils.qt_imports import QtWidgets
from qimview.image_viewers.MultiView import MultiView
from qimview.image_viewers.MultiView import ViewerType


# May need to use multiprocessing
# as in https://stackoverflow.com/questions/6142098/pyqt-is-it-possible-to-run-two-applications
# from multiprocessing import Queue, Process
# class MyApp(Process):

#    def __init__(self):
#        self.queue = Queue(1)
#        super(MyApp, self).__init__()

#    def run(self):
#        app = QApplication([])
#        ...
#        self.queue.put(return_value)

# app1 = MyApp()
# app1.start()
# app1.join()
# print("App 1 returned: " + app1.queue.get())

# app2 = MyApp()
# app2.start()
# app2.join()
# print("App 2 ret


def npViewer(images, vLayout = '0', vType = ViewerType.QT_VIEWER):
    """_summary_

    Args:
        images (list of numpy arrays): input images to display/compare
        vLayout:
        vType:

    Returns:
        qt application: the qt app of the viewer
    """    
    # parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('images', nargs='+', help='input images')
    # parser.add_argument('-v', '--viewer', type=str, choices={'gl', 'qt', 'shader'}, default='qt',
    #                     help="Viewer mode, qt: standard qt display, gl: use opengl,  shader: enable opengl with "
    #                          "shaders")
    # parser.add_argument('-l', '--layout', type=str, default='0', help='Set the layout (number of images in comparison on the window), if 0 try to use the number of input images')

    mv = MultiView(viewer_mode=vType)

    # table_win.setWindowTitle('Image Set Comparison ' + title_string)
    # table_win.set_default_report_file(default_report_file + '.json')
    # table_win.CreateImageDisplay(image_list)
    def get_name(path, maxlength=20):
        return os.path.splitext(os.path.basename(path))[0][-maxlength:]

    images_dict = {}
    for idx,im in enumerate(images):
        images_dict[f"{idx}_{get_name(im)}"] = im
    mv.set_images(images_dict)
    mv.update_layout()
    # table_win.resize(3000, 1800)

    mv.show()
    mv.resize(1000, 800)
    nb_inputs = len(images)
    if nb_inputs>=1 and nb_inputs<=9:
        mv.update_viewer_layout(f'{nb_inputs}')
        mv.viewer_grid_layout.update()
        mv.update_image()
        mv.setFocus()

