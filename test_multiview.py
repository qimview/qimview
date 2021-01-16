from Qt import QtGui, QtCore, QtWidgets
from image_viewers.MultiView import MultiView
import sys


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mv = MultiView()

    # table_win.setWindowTitle('Image Set Comparison ' + title_string)
    # table_win.set_default_report_file(default_report_file + '.json')
    # table_win.CreateImageDisplay(image_list)
    mv.CreateImageDisplay(['im1', 'im2'])
    mv.update_layout()
    # table_win.resize(3000, 1800)

    mv.show()
    app.exec_()