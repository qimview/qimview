
from PySide2.QtWidgets import QApplication
import sys
from video_player.video_player import VideoPlayer


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoPlayer(open_button=True)
    sys.exit(app.exec_())
