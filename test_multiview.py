from Qt import QtGui, QtCore, QtWidgets
from image_viewers.MultiView import MultiView
import sys
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--images', nargs='+', help='input images')

    args = parser.parse_args()
    _params = vars(args)

    app = QtWidgets.QApplication(sys.argv)
    mv = MultiView()

    # table_win.setWindowTitle('Image Set Comparison ' + title_string)
    # table_win.set_default_report_file(default_report_file + '.json')
    # table_win.CreateImageDisplay(image_list)
    def get_name(path, maxlength=10):
        return os.path.splitext(os.path.basename(path))[0][-maxlength:]

    images_dict = {}
    for im in _params['images']:
        images_dict[get_name(im)] = im
    mv.set_images(images_dict)
    mv.update_layout()
    # table_win.resize(3000, 1800)

    mv.show()
    mv.resize(1000, 800)
    app.exec_()