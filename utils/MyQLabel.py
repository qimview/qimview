from Qt import QtWidgets, QtCore


class MyQLabel(QtWidgets.QLabel):
	'''
	This Class is a standard QLabel with the simple and double click mouse events
	'''
	def __init__(self, text, parent = None):
		QtWidgets.QLabel.__init__(self, text, parent)
		self.message = ""

	def mousePressEvent(self, event):
		self.last = "Click"

	def mouseReleaseEvent(self, event):
		print("mouseReleaseEvent")
		if self.last == "Click":
			QtCore.QTimer.singleShot(QtWidgets.QApplication.instance().doubleClickInterval(),
			                         self.performSingleClickAction)
		else:
			# Perform double click action.
			self.message = "Double Click"
			self.update()

	def mouseDoubleClickEvent(self, event):
		self.last = "Double Click"

	def performSingleClickAction(self):
		if self.last == "Click":
			self.message = "Click"
			self.update()
