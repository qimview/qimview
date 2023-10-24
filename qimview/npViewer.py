"""
    Viewer for that takes numpy arrays as input
"""

import os
import os.path as osp
import glob
import numpy as np
from typing import List

from qimview.utils.qt_imports import QtWidgets, QApplication
from qimview.image_viewers.MultiView import MultiView
from qimview.image_viewers.MultiView import ViewerType


# May need to use multiprocessing
# as in https://stackoverflow.com/questions/6142098/pyqt-is-it-possible-to-run-two-applications

from multiprocessing import Queue, Process
from typing import List

class npViewer(Process):
    """
        The purpose of this class is to be able to visualiza and compare several
        numpy arrays from an interactive python session (debugger or ipython)
        using the MultiView class.

        For example, in an ipython session:
            import cv2
            import numpy as np
            b = (np.random.rand(100,100)*256).astype(np.uint8)
            images = [ cv2.GaussianBlur(b, (2*k+1,2*k+1), 0) for k in range(1,4)]
            npViewer(images).start()
        and being able to display the images on one side and continue the interactive session
        on the other side.


    Args:
        Process (_type_): _description_
    """
    def __init__(self, images: List[str]):
        self.queue = Queue(1)
        self.images = images
        super().__init__()

    def run(self):
        app = QApplication([])
        # file_list = glob.glob(osp.join(osp.dirname(__file__),'test_data/*.jpg'))
        # print(file_list)
        # mv = self.create(file_list)
        mv = self.create(self.images)
        mv.show()
        app.exec_()
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)
        self.queue.put(mv)

    def create(self, images: List[str], vLayout: str = '0', vType: ViewerType = ViewerType.QT_VIEWER) -> MultiView:
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

        # images can be a list of image filenames or a list of numpy arrays
        images_dict = {}
        self.temp_dir = None
        for idx,im in enumerate(images):
            if os.path.isfile(im):
                images_dict[f"{idx}_{get_name(im)}"] = im
            elif isinstance(im, np.ndarray):
                # For the moment, save numpy array as png
                # in a temporary directory
                import tempfile
                import cv2
                self.temp_dir = tempfile.mkdtemp()
                image_filename = osp.join(self.temp_dir, f"image_{idx}.png")
                cv2.imwrite(image_filename, im)
                print(f" Written image {image_filename} to disk")
                images_dict[f"{idx}_image"] = image_filename
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

        print("self.mv added")
        self.mv = mv
        return mv

if  __name__ == '__main__':

    # Example that can be run interactively (like in ipython)
    import cv2
    import numpy as np
    b = (np.random.rand(100,100)*256).astype(np.uint8)
    images = [ cv2.GaussianBlur(b, (2*k+1,2*k+1), 0) for k in range(1,4)]
    npViewer(images).start()

    # TODO: share memory? : https://stackoverflow.com/questions/14124588/shared-memory-in-multiprocessing
    # option 2: use qthreads, advantage: memory shared?
    # option 3: update images using the same temporary files + reload or auto-reload in viewer

    # watch file changes with QFileSystemWatcher, example in https://stackoverflow.com/questions/182197/how-do-i-watch-a-file-for-changes
    # in image cache, get/use file timestamp to decide if reload is needed 
