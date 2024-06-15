
from utils.qt_imports import QtWidgets
import sys
from video_player.qt_video_player import QtVideoPlayer


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = QtVideoPlayerVideoPlayer(None, open_button=True)
    sys.exit(app.exec_())
