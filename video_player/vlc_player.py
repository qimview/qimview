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
from Qt import QtWidgets, QtCore, QtGui
from Qt.QtWidgets import QMainWindow, QWidget, QFrame, QSlider
from parameters.numeric_parameter import  NumericParameter
from parameters.numeric_parameter_gui import NumericParameterGui

unicode = str  # Python 3


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
            self.videoframe = QFrame()
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
        if sys.platform.startswith('linux'): # for Linux using the X Server
            self.mediaplayer.set_xwindow(self.videoframe.winId())
        elif sys.platform == "win32": # for Windows
            self.mediaplayer.set_hwnd(self.videoframe.winId())
        elif sys.platform == "darwin": # for MacOS
            self.mediaplayer.set_nsobject(self.videoframe.winId())
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
