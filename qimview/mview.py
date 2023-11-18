#!/bin/python

import sys
import argparse
import os
import glob

from qimview.utils.qt_imports import QtWidgets, QtCore
from qimview.image_viewers    import MultiView, ViewerType

def get_filenames():
    filenames = []
    # Ask for input file
    selected_files =  QtWidgets.QFileDialog.getOpenFileNames(caption="miview: Select  one or various input images")
    filenames.extend(selected_files[0])
    return filenames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('images', nargs='*', help='input images')
    parser.add_argument('-v', '--viewer', type=str, choices={'gl', 'qt', 'shader'}, default='qt',
                        help="Viewer mode, qt: standard qt display, gl: use opengl,  shader: enable opengl with "
                             "shaders")
    parser.add_argument('-l', '--layout', type=str, default='0', help='Set the layout (number of images in comparison on the window), if 0 try to use the number of input images')

    args = parser.parse_args()
    _params = vars(args)

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationDisplayName('mview ' + ' '.join(sys.argv[1:]))

    filenames = []
    for im in _params['images']:
        filenames.extend(glob.glob(im,recursive=False))
    checked_filenames = []
    for f in filenames:
        if os.path.isfile(f): checked_filenames.append(f)
        else: print(f"Filename {f} not found")

    if len(checked_filenames) == 0:
        # Ask for input file(s)
        checked_filenames = get_filenames()

    mode = {
        'qt': ViewerType.QT_VIEWER,
        'gl': ViewerType.OPENGL_VIEWER,
        'shader': ViewerType.OPENGL_SHADERS_VIEWER
    }[_params['viewer']]

    # Create a simple widget and layout to put the multiview inside,
    # so that toggleFullscreen can work
    main_window = QtWidgets.QMainWindow()
    main_widget = QtWidgets.QWidget(main_window)
    main_window.setCentralWidget(main_widget)    
    multiview_layout = QtWidgets.QVBoxLayout()
    main_widget.setLayout(multiview_layout)
    mv = MultiView(parent=main_widget, viewer_mode=mode)
    multiview_layout.addWidget(mv, 1)

    # table_win.setWindowTitle('Image Set Comparison ' + title_string)
    # table_win.set_default_report_file(default_report_file + '.json')
    # table_win.CreateImageDisplay(image_list)
    def get_name(path, maxlength=20):
        return os.path.splitext(os.path.basename(path))[0][-maxlength:]

    images_dict = {}

    for idx,im in enumerate(checked_filenames):
        images_dict[f"{idx}_{get_name(im)}"] = im
    mv.set_images(images_dict)
    mv.update_layout()
    # table_win.resize(3000, 1800)

    nb_inputs = len(checked_filenames)
    mv.show()
    if nb_inputs>=1 and nb_inputs<=9:
        mv.set_number_of_viewers(nb_inputs)
        mv.viewer_grid_layout.update()
        mv.update_image()
        mv.setFocus()

    main_window.show()
    main_window.resize(1000, 800)

    app.exec()

if __name__ == '__main__':
    main()
