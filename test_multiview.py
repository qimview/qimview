#!/bin/python

import sys
import argparse
import os
import glob
import importlib
from pathlib import Path

def import_parents(current_file, level=1):
    global __package__
    file = Path(current_file).resolve()
    parent, top = file.parent, file.parents[level]
    
    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError: # already removed
        pass

    __package__ = '.'.join(parent.parts[len(top.parts):])
    importlib.import_module(__package__) # won't be needed after that

if __name__ == '__main__' and __package__ is None or __package__ == '':
    import_parents(__file__)

from .utils.qt_imports import QtWidgets
from .image_viewers.MultiView import MultiView
from .image_viewers.MultiView import ViewerType

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--images', nargs='+', help='input images')
    parser.add_argument('--viewer', type=str, choices={'gl', 'qt', 'shader', 'pyqtgraph'}, default='qt',
                        help="Viewer mode, qt: standard qt display, gl: use opengl,  shader: enable opengl with "
                             "shaders, pyqtgraph: experimental, use pyqtgraph module if installed")
    parser.add_argument('-l', '--layout', type=str, default='0', help='Set the layout (number of images in comparison on the window), if 0 try to use the number of input images')

    args = parser.parse_args()
    _params = vars(args)

    filenames = []
    for im in _params['images']:
      filenames.extend(glob.glob(im,recursive=False))

    app = QtWidgets.QApplication(sys.argv)

    mode = {
        'qt': ViewerType.QT_VIEWER,
        'gl': ViewerType.OPENGL_VIEWER,
        'shader': ViewerType.OPENGL_SHADERS_VIEWER,
        'pyqtgraph': ViewerType.PYQTGRAPH_VIEWER
    }[_params['viewer']]

    mv = MultiView(viewer_mode=mode)

    # table_win.setWindowTitle('Image Set Comparison ' + title_string)
    # table_win.set_default_report_file(default_report_file + '.json')
    # table_win.CreateImageDisplay(image_list)
    def get_name(path, maxlength=20):
        return os.path.splitext(os.path.basename(path))[0][-maxlength:]

    images_dict = {}
    for idx,im in enumerate(filenames):
        images_dict[f"{idx}_{get_name(im)}"] = im
    mv.set_images(images_dict)
    mv.update_layout()
    # table_win.resize(3000, 1800)

    mv.show()
    mv.resize(1000, 800)
    nb_inputs = len(filenames)
    if nb_inputs>=1 and nb_inputs<=9:
        mv.update_viewer_layout(f'{nb_inputs}')
        mv.viewer_grid_layout.update()
        mv.update_image()
        mv.setFocus()
    app.exec()