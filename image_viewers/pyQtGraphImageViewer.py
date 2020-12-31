try:
	import pyqtgraph as pg
except ImportError as e:
	print(" Failed to load pyqtgraph module, check that it is well installed")
	loaded_pyqtgraph = False
else:
	loaded_pyqtgraph = True

from image_viewers.ImageViewer import ImageViewer


class pyQtGraphImageViewer(pg.GraphicsLayoutWidget, ImageViewer):

	def __init__(self, parent=None):
		pg.GraphicsLayoutWidget.__init__(self, parent)
		ImageViewer.__init__(self, parent)
		self.viewbox = self.addViewBox(row=1, col=1)
		self.imv = pg.ImageItem()
		self.viewbox.addItem(self.imv)
		self.viewbox.setAspectLocked(True)
		pg.setConfigOptions(imageAxisOrder='row-major')

		# self.graphText = pg.TextItem(text='Image', color=(0, 255, 255))
		# self.graphText.setParentItem(self.viewbox)
		# self.graphText.setPos(0, 0)
		# self.graphText.setText('Image')
		# self.viewbox.addItem(self.graphText)
		# self.graphText = self.centralLayout.AddLabel(text='Image', color=(0, 255, 255))
		# self.graphText.setParentItem(self.viewbox)
		# self.graphText.setPos(0, 0)
		# self.graphText.setText('Image')
		# self.viewbox.addItem(self.graphText)

	@staticmethod
	def numpy2imageitem(im):
		return pg.ImageItem(im[::-1, :, :], autoLevels=False, levels=None)

	def set_image(self, image, active=False):
		super(pyQtGraphImageViewer, self).set_image(image)
		self.viewbox.removeItem(self.imv)
		self.viewbox.addItem(image)
		self.imv = image
		# 	if self.show_timing_detailed():
		# 		time_spent = get_time() - update_image_start
		# 		print(" After setImage took {0:0.3f} sec.".format(time_spent))
		self.paintAll()

	def paintAll(self):
		self.update()

	def mouseDoubleClickEvent(self, event):
		self.mouse_double_click_event(event)

