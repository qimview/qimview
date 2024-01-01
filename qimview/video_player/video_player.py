
# required to install codecs to work well on windows, I installed K-Lite standard codecs

from qimview.utils.qt_imports import QtGui, QtCore, QtWidgets, QtMultimedia, QtMultimediaWidgets
from qimview.parameters.numeric_parameter import  NumericParameter
from qimview.parameters.numeric_parameter_gui import NumericParameterGui

import sys

class myVideoWidget(QtMultimediaWidgets.QVideoWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_scale = 1

    def new_scale(self, mouse_zy, height):
        return max(1, self.current_scale * (1 + mouse_zy * 5.0 / height))
        # return max(1, self.current_scale  + mouse_zy * 5.0 / height)

    def wheelEvent(self, event):
        if hasattr(event, 'delta'):
            delta = event.delta()
        else:
            delta = event.angleDelta().y()
        coeff = delta/5
        rect = self.geometry()
        prev_scale = self.current_scale
        self.current_scale = self.new_scale(coeff, rect.height())
        print(f" geometry {rect.x()} {rect.y()} {rect.width()} {rect.height()}")
        print(f" current_scale {self.current_scale}")
        rect.setWidth (rect.width() *(self.current_scale/prev_scale))
        rect.setHeight(rect.height()*(self.current_scale/prev_scale))
        # self.setGeometry(rect)
        self.setFixedSize(rect.width(), rect.height())

    def resizeEvent(self, event):
        """Called upon window resizing: reinitialize the viewport.
        """
        # print("resize {} {}  self {} {}".format(event.size().width(), event.size().height(),
        #       self.width(), self.height()))
        # event.ignore()

class VideoPlayer(QtWidgets.QWidget):
    def __init__(self, parent, open_button=False):
        super().__init__(parent)

        self.setWindowTitle("QMediaPlayer")
        # self.setGeometry(350, 100, 700, 500)
        self.setWindowIcon(QtGui.QIcon('player.png'))

        p = self.palette()
        p.setColor(QtGui.QPalette.Window, QtCore.Qt.black)
        self.setPalette(p)

        self.synchronize_viewer = None
        self.add_open_button = open_button
        self.play_position = NumericParameter()
        self.play_position.float_scale = 1000

        self.init_ui()
        # self.show()

    def init_ui(self):

        # create media player objectexamples_qtvlc.py
        self.mediaPlayer = QtMultimedia.QMediaPlayer(self)
        # self.mediaPlayer.setNotifyInterval(20)

        # create videowidget object

        # using graphics view may seem a good idea but the video is playing with lags
        use_graphic_view = False
        if use_graphic_view:
            video_item = QtMultimediaWidgets.QGraphicsVideoItem()
            self.video_view  = QtWidgets.QGraphicsView()
        else:
            use_scroll = False
            videowidget = myVideoWidget()
            if use_scroll:
                video_scroll = QtWidgets.QScrollArea()
                video_scroll.setWidget(videowidget)
            else:
                video_scroll = videowidget

        # create open button
        if self.add_open_button:
            openBtn = QtWidgets.QPushButton('Open Video')
            openBtn.clicked.connect(self.open_file)

        # create button for playing
        self.playBtn = QtWidgets.QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(lambda: self.synchronize_toggle_play(self))

        # create slider
        self.play_position_gui = NumericParameterGui(name="play_position", param=self.play_position,
                                                     callback=lambda: self.synchronize_set_play_position(self))
        self.play_position_gui.decimals = 3


        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)
        # create label
        self.label = QtWidgets.QLabel()
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

        # create hbox layout
        hboxLayout = QtWidgets.QHBoxLayout()
        hboxLayout.setContentsMargins(0, 0, 0, 0)

        # set widgets to the hbox layout
        if self.add_open_button:
            hboxLayout.addWidget(openBtn)
        hboxLayout.addWidget(self.playBtn)
        hboxLayout.addWidget(self.slider)
        self.play_position_gui.add_to_layout(hboxLayout)

        # create vbox layout
        vboxLayout = QtWidgets.QVBoxLayout()

        if use_graphic_view:
            vboxLayout.addWidget(self.video_view)
            self.video_scene = QtWidgets.QGraphicsScene(0, 0, self.video_view.size().width(), self.video_view.size().height())
            self.video_view.setScene(self.video_scene)
            self.video_scene.addItem(video_item)
        else:
            vboxLayout.addWidget(video_scroll)

        vboxLayout.addLayout(hboxLayout)
        vboxLayout.addWidget(self.label)

        self.setLayout(vboxLayout)

        if use_graphic_view:
            self.mediaPlayer.setVideoOutput(video_item)
        else:
            self.mediaPlayer.setVideoOutput(videowidget)

        # media player signals

        self.mediaPlayer.playbackStateChanged.connect(self.mediastate_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)


    def set_synchronize(self, viewer):
        self.synchronize_viewer = viewer

    # def synchronize_data(self, other_viewer):
    #     other_viewer.current_scale = self.current_scale
    #     other_viewer.current_dx = self.current_dx
    #     other_viewer.current_dy = self.current_dy
    #     other_viewer.mouse_dx = self.mouse_dx
    #     other_viewer.mouse_dy = self.mouse_dy
    #     other_viewer.mouse_zx = self.mouse_zx
    #     other_viewer.mouse_zy = self.mouse_zy
    #     other_viewer.mouse_x = self.mouse_x
    #     other_viewer.mouse_y = self.mouse_y

    def synchronize_toggle_play(self, event_viewer):
        self.toggle_play_video()
        if self.synchronize_viewer is not None and self.synchronize_viewer is not event_viewer:
            self.synchronize_viewer.synchronize_toggle_play(event_viewer)

    def open_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video")

        if filename != '':
            self.mediaPlayer.setSource(QtCore.QUrl.fromLocalFile(filename))
            self.playBtn.setEnabled(True)

    def set_video(self, filename):
        if filename != '':
            self.mediaPlayer.setSource(QtCore.QUrl.fromLocalFile(filename))
            self.playBtn.setEnabled(True)

    def toggle_play_video(self):
        if self.mediaPlayer.playbackState() == QtMultimedia.QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediastate_changed(self, state):
        if self.mediaPlayer.playbackState() == QtMultimedia.QMediaPlayer.PlayingState:
            self.playBtn.setIcon( self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause) )
            metadatalist = self.mediaPlayer.metaData()
            print(f"metadatalist = {metadatalist}")
            print(f"metadatalist = {metadatalist}")
            for key in metadatalist:
                var_data = self.mediaPlayer.metaData(key)
                print(f"{key}: {var_data}")
        else:
            self.playBtn.setIcon( self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay) )

    def position_changed(self, position):
        self.slider.setValue(position)
        self.play_position_gui.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

        self.play_position.range = [0, duration]
        self.play_position_gui.setRange(0, duration)

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def synchronize_set_play_position(self, event_viewer):
        if self.mediaPlayer.playbackState() != QtMultimedia.QMediaPlayer.PlayingState:
            self.set_play_position()
            if self.synchronize_viewer is not None and self.synchronize_viewer is not event_viewer:
                self.synchronize_viewer.play_position.copy_from(self.play_position)
                self.synchronize_viewer.synchronize_set_play_position(event_viewer)

    def set_play_position(self):
        self.mediaPlayer.setPosition(self.play_position.int)

    def handle_errors(self):
        self.playBtn.setEnabled(False)
        self.label.setText("Error: " + self.mediaPlayer.errorString())


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = VideoPlayer(open_button=True)
    window.show()
    sys.exit(app.exec_())
