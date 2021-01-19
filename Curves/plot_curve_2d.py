import sys
import Qt
import numpy as np
from scipy import signal

# from Qt.QtWidgets import *
# from Qt.QtCore import *
# from Qt.QtGui import *

# Switch easily between PyQt5 and PySide2
if Qt.IsPyQt5:
    print("Using PyQt5")
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5 import QChart as QtCharts

if Qt.IsPySide2:
    print("Using PySide2")
    from PySide2.QtWidgets import *
    from PySide2.QtCore import *
    from PySide2.QtGui import *
    from PySide2.QtCharts import QtCharts


class PlotCurves2D(QtCharts.QChartView):
    def __init__(self):
        super().__init__()

    def draw_curve(self, Y):
        series = QtCharts.QLineSeries(self)
        for idx,y in enumerate(Y):
            series.append(idx,y)
        chart = QtCharts.QChart()

        chart.addSeries(series)
        chart.createDefaultAxes()
        #chart.setAnimationOptions(Qself.SeriesAnimations)
        chart.setTitle("Line Chart Example")

        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)

        self.setChart(chart)
        self.setRubberBand(QtCharts.QChartView.HorizontalRubberBand)
        self.setRenderHint(QPainter.Antialiasing)

    def draw_example(self):
        Y = signal.windows.gaussian(51, std=7)
        self.draw_curve(Y)

