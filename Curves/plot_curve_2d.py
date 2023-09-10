import numpy as np
from scipy import signal
from PySide6 import QtCharts

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

