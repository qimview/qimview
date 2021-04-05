
#from Qt import QtWidgets, QtCore
# from Qt import QtGui, QtCore, QtWidgets

import Qt

if Qt.__binding__ == "PySide2":
    from PySide2 import QtGui, QtWidgets, QtCore, QtMultimedia, QtMultimediaWidgets
    from PySide2.QtOpenGL import QGLWidget
    from PySide2.QtWidgets import QOpenGLWidget as QOpenGLWidget
    from PySide2.QtCore import Signal, Slot
else:
    from PyQt5 import QtGui, QtWidgets, QtCore, QtMultimedia, QtMultimediaWidgets
    from PyQt5.QtOpenGL import QGLWidget
    from PyQt5.QtWidgets import QOpenGLWidget as QOpenGLWidget
    from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

QLabel = QtWidgets.QLabel
QApplication = QtWidgets.QApplication
QTimer = QtCore.QTimer
