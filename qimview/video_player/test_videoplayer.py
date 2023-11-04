
from utils.qt_imports import QtWidgets
import sys
from video_player.video_player import VideoPlayer


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = VideoPlayer(app, open_button=True)
    sys.exit(app.exec_())
