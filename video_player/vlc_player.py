# voir lien interessant https://github.com/geoffsalmon/vlc-python/blob/master/generated/vlc.py

#! /usr/bin/python

#
# Qt example for VLC Python bindings
# Copyright (C) 2009-2010 the VideoLAN team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
#

import sys
import os.path
import vlc
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QWidget, QFrame
from ..parameters.numeric_parameter import  NumericParameter
from ..parameters.numeric_parameter_gui import NumericParameterGui


unicode = str  # Python 3

# TODO: deal with macOS
# if sys.platform == "darwin":  # for MacOS
#     self.videoframe = QtGui.QMacCocoaViewContainer(0)

if sys.platform == "darwin":  # for MacOS
    BaseWidget = QtGui.QMacCocoaViewContainer
else:
    BaseWidget = QFrame


class myVideoWidget(BaseWidget):
    def __init__(self):
        if sys.platform == "darwin":  # for MacOS
            super().__init__(0)
        else:
            super().__init__()
        self.current_scale = 1
        self.media_player = None

    def new_scale(self, mouse_zy, height):
        return max(1, self.current_scale * (1 + mouse_zy * 5.0 / height))
        # return max(1, self.current_scale  + mouse_zy * 5.0 / height)

    def set_media_player(self, media_player):
        self.media_player = media_player

        # now if we set both inputs mouse and keyboard to false, we are able to catch event from the player!
        self.media_player.video_set_mouse_input(False)
        self.media_player.video_set_key_input(False)

        if sys.platform.startswith('linux'): # for Linux using the X Server
            media_player.set_xwindow(self.winId())
        elif sys.platform == "win32": # for Windows
            media_player.set_hwnd(self.winId())
        elif sys.platform == "darwin": # for MacOS
            media_player.set_nsobject(self.winId())

    def wheelEvent(self, event):
        if hasattr(event, 'delta'):
            delta = event.delta()
        else:
            delta = event.angleDelta().y()
        coeff = delta/5
        rect = self.geometry()
        prev_scale = self.current_scale
        self.current_scale = self.new_scale(coeff, rect.height())
        # print(f" geometry {rect.x()} {rect.y()} {rect.width()} {rect.height()}")
        # print(f" current_scale {self.current_scale}")
        rect.setWidth (rect.width() *(self.current_scale/prev_scale))
        rect.setHeight(rect.height()*(self.current_scale/prev_scale))
        # self.setGeometry(rect)
        # self.setFixedSize(rect.width(), rect.height())

        if self.media_player is not None:
            (video_w, video_h) = self.media_player.video_get_size()
            print(f"media size {video_w}x{video_h}")
            print(f" geometry  {rect.width()}x{rect.height()}")
            print(f" scale approx  {rect.width()/video_w:0.2f}x{rect.height()/video_h:0.2f}")

            geom = self.media_player.video_get_crop_geometry()
            print(f"geom: {geom}")
            print(f"video_get_scale: {self.media_player.video_get_scale()}")
            center_x = video_w/2
            center_y = video_h/2
            new_w = int(video_w/self.current_scale+0.5)
            new_h = int(video_h/self.current_scale+0.5)
            start_x = int((video_w-new_w)/2+0.5)
            start_y = int((video_h-new_h)/2+0.5)
            # not clear why it requires to multiply by 2 the top-left position
            self.media_player.video_set_crop_geometry(f"{new_w}x{new_h}+{start_x*2}+{start_y*2}")
            # else:
            #     self.mediaplayer.video_set_crop_geometry(None)
            # print(f"geom: {self.mediaplayer.video_get_crop_geometry()}")



    def resizeEvent(self, event):
        """Called upon window resizing: reinitialize the viewport.
        """
        # print("resize {} {}  self {} {}".format(event.size().width(), event.size().height(),
        #       self.width(), self.height()))
        # event.ignore()


class VLCPlayer(QWidget):
    """A simple Media Player using VLC and Qt
    """
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("Media Player")

        self.play_position = NumericParameter()
        self.play_position.range = [0, 1000]
        self.play_position.float_scale = 1000

        # creating a basic vlc instance
        self.instance = vlc.Instance()
        # creating an empty vlc media player
        self.mediaplayer = self.instance.media_player_new()
        self.synchronize_viewer = None

        self.createUI()
        self.isPaused = False

    def createUI(self):
        """Set up the user interface, signals & slots
        """
        # In this widget, the video will be drawn
        if sys.platform == "darwin": # for MacOS
            self.videoframe = QtGui.QMacCocoaViewContainer(0)
        else:
            self.videoframe = myVideoWidget()
        self.palette = self.videoframe.palette()
        self.palette.setColor (QtGui.QPalette.Window,
                               QtGui.QColor(0,0,0))
        self.videoframe.setPalette(self.palette)
        self.videoframe.setAutoFillBackground(True)

        # create slider
        self.play_position_gui = NumericParameterGui(name="play_position", param=self.play_position,
                                                     callback=lambda: self.synchronize_set_play_position(self))
        self.play_position_gui.decimals = 3

        # self.positionslider = QSlider(QtCore.Qt.Horizontal, self)
        # self.positionslider.setToolTip("Position")
        # self.positionslider.setMaximum(1000)
        # self.positionslider.sliderMoved.connect(self.setPosition)

        self.hbuttonbox = QtWidgets.QHBoxLayout()
        self.playbutton = QtWidgets.QPushButton("Play")
        self.hbuttonbox.addWidget(self.playbutton)
        self.playbutton.clicked.connect(lambda: self.synchronize_toggle_play(self))

        self.stopbutton = QtWidgets.QPushButton("Stop")
        self.hbuttonbox.addWidget(self.stopbutton)
        self.stopbutton.clicked.connect( self.Stop)

        self.hbuttonbox.addStretch(1)
        self.volumeslider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.volumeslider.setMaximum(100)
        self.volumeslider.setValue(self.mediaplayer.audio_get_volume())
        self.volumeslider.setToolTip("Volume")
        self.hbuttonbox.addWidget(self.volumeslider)
        self.volumeslider.valueChanged.connect( self.setVolume)

        self.vboxlayout = QtWidgets.QVBoxLayout()
        self.vboxlayout.addWidget(self.videoframe)
        self.hboxlayout = QtWidgets.QHBoxLayout()
        # self.play_position_gui.create(moved_callback=True)
        self.play_position_gui.add_to_layout(self.hboxlayout)
        self.vboxlayout.addLayout(self.hboxlayout)
        self.vboxlayout.addLayout(self.hbuttonbox)

        self.setLayout(self.vboxlayout)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.updateUI)

    def set_synchronize(self, viewer):
        self.synchronize_viewer = viewer

    def synchronize_toggle_play(self, event_viewer):
        self.PlayPause()
        if self.synchronize_viewer is not None and self.synchronize_viewer is not event_viewer:
            self.synchronize_viewer.synchronize_toggle_play(event_viewer)

    def PlayPause(self):
        """Toggle play/pause status
        """
        if self.mediaplayer.is_playing():
            self.mediaplayer.pause()
            self.playbutton.setText("Play")
            self.isPaused = True
            # geom = self.mediaplayer.video_get_crop_geometry()
            # if geom is None:
            #     self.mediaplayer.video_set_crop_geometry("520x520+10+10")
            # else:
            #     self.mediaplayer.video_set_crop_geometry(None)
            # print(f"geom: {self.mediaplayer.video_get_crop_geometry()}")
        else:
            if self.mediaplayer.play() == -1:
                self.OpenFile()
                return
            self.mediaplayer.play()
            self.playbutton.setText("Pause")
            self.timer.start()
            self.isPaused = False

    def Stop(self):
        """Stop player
        """
        self.mediaplayer.stop()
        self.playbutton.setText("Play")

    def set_video(self, filename=None):
        self.OpenFile(filename)

    def OpenFile(self, filename=None):
        """Open a media file in a MediaPlayer
        """
        print(f"filename is {filename}")
        if filename is None or not os.path.isfile(filename):
            filename = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", os.path.expanduser('~'))
            if not filename:
                return
            print(f"filename is {filename}")
            filename = filename[0]
            print(f"filename is {filename}")
        if not filename:
            return


        self.media = self.instance.media_new(filename)
        # put the media in the media player
        self.mediaplayer.set_media(self.media)

        # parse the metadata of the file
        self.media.parse()
        # set the title of the track as window title
        self.setWindowTitle(self.media.get_meta(0))

        # the media player has to be 'connected' to the QFrame
        # (otherwise a video would be displayed in it's own window)
        # this is platform specific!
        # you have to give the id of the QFrame (or similar object) to
        # vlc, different platforms have different functions for this

        self.videoframe.set_media_player(self.mediaplayer)

        self.PlayPause()

    def setVolume(self, Volume):
        """Set the volume
        """
        self.mediaplayer.audio_set_volume(Volume)

    def synchronize_set_play_position(self, event_viewer):
        if not self.mediaplayer.is_playing():
            print(f"synchronize_set_play_position {id(self)}")
            self.set_play_position()
            if self.synchronize_viewer is not None and self.synchronize_viewer is not event_viewer:
                self.synchronize_viewer.play_position.copy_from(self.play_position)
                self.synchronize_viewer.synchronize_set_play_position(event_viewer)

    def set_play_position(self):
        print(f"set_play_position {id(self)} {self.play_position.float}")
        self.mediaplayer.set_position(self.play_position.float)

    # def setPosition(self, position):
    #     """Set the position
    #     """
    #     # setting the position to where the slider was dragged
    #     self.mediaplayer.set_position(position / 1000.0)
    #     # the vlc MediaPlayer needs a float value between 0 and 1, Qt
    #     # uses integer variables, so you need a factor; the higher the
    #     # factor, the more precise are the results
    #     # (1000 should be enough)

    def updateUI(self):
        """updates the user interface"""
        # setting the slider to the desired position
        # self.positionslider.setValue(self.mediaplayer.get_position() * 1000)
        self.play_position.float = self.mediaplayer.get_position()
        self.play_position_gui.setValue(self.play_position.int)

        if not self.mediaplayer.is_playing():
            # no need to call this function if nothing is played
            self.timer.stop()
            if not self.isPaused:
                # after the video finished, the play button stills shows
                # "Pause", not the desired behavior of a media player
                # this will fix it
                self.Stop()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    player = VLCPlayer()
    player.show()
    player.resize(640, 480)
    if sys.argv[1:]:
        player.OpenFile(sys.argv[1])
    sys.exit(app.exec_())
