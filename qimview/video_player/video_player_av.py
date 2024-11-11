import os
import re
import time
import numpy as np
from typing import Union, Generator, List, Optional, Iterator
if os.name == 'nt' and os.path.isdir("c:\\ffmpeg\\bin"):
    os.add_dll_directory("c:\\ffmpeg\\bin")
    #os.add_dll_directory("C:\\Users\\karl\\GIT\\vcpkg\\packages\\ffmpeg_x64-windows-release\\bin")
import av
from av import container
from  av.video.frame import VideoFrame as AVVideoFrame # type: ignore
from av.frame import Frame
from cv2 import cvtColor, COLOR_YUV2RGB_I420 # type: ignore

from qimview.utils.qt_imports                          import QtWidgets, QtCore, QtGui
from qimview.image_viewers.gl_image_viewer_shaders     import GLImageViewerShaders
from qimview.utils.viewer_image                        import ViewerImage, ImageFormat
from qimview.video_player.video_scheduler              import VideoScheduler
from qimview.video_player.video_player_base            import VideoPlayerBase, ImageViewerClass
from qimview.video_player.video_player_key_events      import VideoPlayerKeyEvents
from qimview.video_player.video_frame                  import VideoFrame

try:
    ffmpeg_path = os.path.join(os.environ.get('FFMPEG_ROOT', ''),'bin')
    if os.name == 'nt' and os.path.isdir(ffmpeg_path):
        os.add_dll_directory(ffmpeg_path)

    import decode_video_py as decode_lib
    has_decode_video_py = True
except Exception as e:
    print("Failed to load decode_video_py")
    has_decode_video_py = False

from qimview.video_player.video_frame_provider_cpp import VideoFrameProviderCpp
from qimview.video_player.video_frame_provider     import VideoFrameProvider

class AverageTime:
    def __init__(self):
        self._sum     : float = 0
        self._counter : int = 0
    def add_time(self,t):
        self._sum += t
        self._counter +=1
    def average(self):
        if self._counter>0:
            return self._sum/self._counter
        else:
            return -1


class VideoPlayerAV(VideoPlayerBase):

    def __init__(self, parent=None, 
                 use_decode_video_py : bool = has_decode_video_py, 
                 codec : str = '') -> None:
        super().__init__(parent)
        self.event_recorder = None
        self._use_decode_video_py : bool = use_decode_video_py
        self._codec : str = codec

        # Key event class
        self._key_events   : VideoPlayerKeyEvents   = VideoPlayerKeyEvents(self)

        self.widget.add_help_tab('VideoPlayer keys',  self._key_events.markdown_help())
        self.widget.add_help_links(self._key_events.help_links())

        # is show required here?
        self.show()
        self._im = None
        self._button_play_pause.clicked.connect(self.play_pause)
        self._container : container.InputContainer | None = None

        self._scheduler : VideoScheduler = VideoScheduler()
        self._start_video_time : float = 0
        self.loop_start_time  : float = 0
        self.loop_end_time    : float = -1 # -1 means end of video
        self._skipped : int = 0
        if use_decode_video_py:
            self._frame_provider : VideoFrameProvider | VideoFrameProviderCpp = VideoFrameProviderCpp()
        else:
            self._frame_provider : VideoFrameProvider | VideoFrameProviderCpp = VideoFrameProvider()
        self._displayed_pts : int = -1
        self._name : str = "video player"
        self._t1 : AverageTime = AverageTime()
        self._t2 : AverageTime = AverageTime()
        self._t3 : AverageTime = AverageTime()
        self._t4 : AverageTime = AverageTime()
        self._t5 : AverageTime = AverageTime()
        self._max_skip : int = 10

        self._filename            : str                 = "none"
        self._video_stream_number : int                 = 0
        self._basename            : str                 = "none"
        self._initialized         : bool                = False
        self._compare_players     : list[VideoPlayerAV] = []
        self._compare_timeshift   : list[float]         = [] # time shift for each comparing player

    @property
    def scheduler(self) -> VideoScheduler:
        return self._scheduler

    @property
    def frame_provider(self) -> VideoFrameProvider | VideoFrameProviderCpp:
        return self._frame_provider

    @property
    def frame_duration(self) -> float:
        if self.frame_provider:
            return self.frame_provider.frame_duration
        else:
            return 0.1

    def empty_compare(self):
        self._compare_players = []
        
    def compare(self, player):
        self._compare_players.append(player)
        self._compare_timeshift.append(0)
        print(f"{self._compare_players=} {self._compare_timeshift=}")

    def on_synchronize(self, viewer : ImageViewerClass) -> None:
        # Synchronize other viewer to calling viewer
        for v in self._compare_players:
            if v.widget != viewer:
                 viewer.synchronize_data(v.widget)
                 if self.synchronize_filters:
                     v.widget.filter_params.copy_from(viewer.filter_params)
                 v.widget.viewer_update()

    def set_name(self, n:str):
        self._name = n

    def set_video(self, filename):
        """ Set input video filename and optionally stream number

        Args:
            filename (string): filename or filename:stream_number
        """
        # Pause video if running
        self.pause()
        res = re.match('^(.*):(\d+)$', filename)
        if res:
            self._filename = res.group(1)
            self._video_stream_number = int(res.group(2))
        else:
            self._filename = filename
            self._video_stream_number = 0
        self._basename = os.path.basename(self._filename)

        self._initialized = False

    def set_pause(self):
        """ Method called from scheduler """
        self._frame_provider.playing = False

    def pause(self):
        """ Method called from video player """
        self._was_active = self._scheduler.is_running
        self._scheduler.pause()
    
    def set_play(self):
        """ Method called from scheduler """
        self._frame_provider.playing = True
        self._start_video_time = self._frame_provider.get_time()
        print(f"set_play _start_video_time = {self._start_video_time:0.3f}")

    def reset_play(self):
        if self._was_active:
            self._scheduler.play()

    def start_decode(self):
        if not self._scheduler.is_running:
            self._scheduler.pause()
        players = [self]
        players.extend(self._compare_players)
        self._scheduler.set_players(players)
        timeshifts = [0]
        timeshifts.extend(self._compare_timeshift)
        self._scheduler.set_timeshifts(timeshifts)
        self._scheduler.start_decode(self._frame_provider.get_time())
        
    def loop_clicked(self):
        self._scheduler._loop = self._button_loop.isChecked()

    def play_pause(self):
        # If the number of players has changes, restart decode
        if len(self.scheduler._players) != len(self._compare_players)+1:
            self._button_play_pause.setIcon(self._icon_pause)
            self.start_decode()
        else:
            if self.scheduler.is_running:
                self.pause()
                self._button_play_pause.setIcon(self._icon_play)
            else:
                self._button_play_pause.setIcon(self._icon_pause)
                self._scheduler.play()

    def slider_value_changed(self):
        self.set_play_position(fromSlider=True)

    def set_play_position(self, recursive=True, fromSlider=False):
        """ Sets the play position based on the member play_position

        Args:
            recursive (bool, optional): _description_. Defaults to True.
            fromSlider (bool, optional): _description_. Defaults to False.
        """
        # print(f"{self._name} set_play_position {fromSlider=} {self.play_position=}")
        # Here, instead of resetting the frame_buffer, we want to use existing frames
        # as much as possible
        if self._frame_provider.frame_buffer:
            self._frame_provider.frame_buffer.reset()
        self._frame_provider.set_time(self.play_position)
        self._start_video_time = self.play_position
        for idx, p in enumerate(self._compare_players):
            p.play_position = self.play_position + self._compare_timeshift[idx]
            p.set_play_position(recursive=False)
            p.update_position(recursive=False, force=fromSlider)
        self.display_frame(self._frame_provider.frame)
        
    def setTimeShifts(self) -> None:
        play_position = self.play_position
        for idx, p in enumerate(self._compare_players):
            self._compare_timeshift[idx] = p.play_position - play_position
            print(f"Setting time shift for player {idx} as {self._compare_timeshift[idx]}")

    def speed_value_changed(self):
        print(f"{self._name} New speed value {self.playback_speed.float}")
        self._scheduler.set_playback_speed(pow(2,self.playback_speed.float))

    def update_position(self, precision=0.1, recursive=True, force=False) -> bool:
        # print(f"{self._name} update_position({precision=}, {recursive=} {force=})")
        current_time = self._frame_provider.get_time()
        # print(f"{self._name} update_position({current_time=}, {self.play_position=})")
        if force or abs(self.play_position-current_time)>precision:
            # print(f"{self._name} update Gui {current_time=}")
            self.play_position = current_time
            # print(f"{self._name} {self.play_position=}")

            # Block signals to avoid calling changedValue signal
            self.play_position_gui.blockSignals(True)
            self.play_position_gui.updateGui()
            self.play_position_gui.blockSignals(False)
            if recursive:
                for p in self._compare_players:
                    p.update_position(recursive=False)
            return True
        return False

    def set_image_RGB(self, im : ViewerImage, im_name, force_new=False):
        if self._im is None or force_new or self._im.data.shape != im.data.shape:
            self._im = im
            self.widget.set_image_fast(self._im)
        else:
            if self._im.data.shape == im.data.shape:
                self._im._data = im.data
                self.widget.image_id += 1
        self._im.filename = f"{self._filename} : {self._frame_provider.get_frame_number()}"
        self.widget.image_name = im_name

    def set_image_YUV420(self, frame: AVVideoFrame, im_name: str, frame_str: str):
        video_frame = VideoFrame(frame)
        print(f" --- set_image_YUV420 for {self._name} with pos {frame.pts}")
        self._im = video_frame.toViewerImage()
        self._im.filename = self._filename + frame_str
        use_crop = self._scheduler.is_running
        if len(self._compare_players)>0:
            # Use image from _compare_player as a ref?
            print(f" *** comparing images ...{self._im.filename[-4:]} ...{self._compare_players[0]._im.filename[-4:]}")
            self.widget.set_image_fast(self._im, image_ref = self._compare_players[0]._im, use_crop=use_crop)
        else:
            self.widget.set_image_fast(self._im, use_crop=use_crop)
        self.widget.image_name = im_name + frame_str

    def set_image_data(self, np_array):
        self._im._data = np_array
        self.widget.image_id += 1

    def closeEvent(self, event):
        # Save event list
        if self.event_recorder is not None:
            self.event_recorder.save_screen(self.widget)
            self.event_recorder.save_events()
        event.accept()

    def set_synchronize(self, viewer):
        self.synchronize_viewer = viewer

    def display_frame_RGB(self, frame: AVVideoFrame):
        """ Convert YUV to RGB and display RGB frame """
        video_frame = VideoFrame(frame)
        im = video_frame.toViewerImage(rgb=True)
        im_name = f"frame_{frame.pts}"
        if im:
            if self._im is not None:
                self.set_image_data(im.data)
            else:
                self.set_image_RGB(im, im_name, force_new=self.viewer_class is GLImageViewerShaders)
        self.widget.viewer_update()

    def display_times(self):
        print(f"display_frame_RGB() 1. {self._t1.average()*1000:0.1f} ms")
        print(f"display_frame_RGB() 2. {self._t2.average()*1000:0.1f} ms")
        print(f"display_frame_RGB() 3. {self._t3.average()*1000:0.1f} ms")
        print(f"display_frame_RGB() 4. {self._t4.average()*1000:0.1f} ms")
        print(f"display_frame_RGB() 5. {self._t5.average()*1000:0.1f} ms")


    def display_frame_YUV420(self, frame: AVVideoFrame):
        """ Display YUV frame, for OpenGL with shaders """
        frame_str = ' :'+str(self._frame_provider.get_frame_number())
        self.set_image_YUV420(frame, self._basename, frame_str)
        if self.viewer_class == GLImageViewerShaders and self._im and self._im.crop is not None:
            # Apply crop on the right
            self.widget.set_crop(self._im.crop)
        else:
            self.widget.set_crop(np.array([0,0,1,1], dtype=np.float32))
        self.widget.viewer_update()

    def display_frame(self, frame=None):
        if frame is None:
            frame = self._frame_provider._frame
        if frame is None:
            return
        self.widget._custom_text =  f"\nFPS:     {self._frame_provider._framerate:0.3f}"
        self.widget._custom_text += f"\nduration:{self._frame_provider._duration:0.3f} sec."
        if self.viewer_class is GLImageViewerShaders:
            self.display_frame_YUV420(frame)
        else:
            self.display_frame_RGB(frame)

    def init_video_av(self):
        """ Initialize the container and frame generator """
        if not self._initialized:
            print("--- init_video_av() ")
        else:
            print("--- init_video_av() video already intialized")
            return
        
        if self.scheduler.is_running:
            self.scheduler.pause()
        print(f"filename = {self._filename}")
        if self._container is not None:
            self._frame_provider.frame_buffer.reset()
            # del self._frame_provider.frame_buffer
            # del self._frame_provider._container
            if not self._use_decode_video_py:
                self._container.close()
            del self._container
            self._container = None
        if self._use_decode_video_py:
            device_type = self._codec if self._codec != '' else None
            self._container = decode_lib.VideoDecoder()
            self._container.open(self._filename, device_type, self._video_stream_number, 
                                 num_threads=8 if device_type is None else 2)
        else:
            self._container = av.open(self._filename)
        self._frame_provider.set_input_container(self._container, self._video_stream_number)

        print(f"duration = {self._frame_provider._duration} seconds")
        slider_single_step = int(self._frame_provider._ticks_per_frame*
                                 self._frame_provider._time_base*
                                 self._play_position.float_scale+0.5)
        slider_page_step   = int(self._play_position.float_scale+0.5)
        self.play_position_gui.setSingleStep(slider_single_step)
        self.play_position_gui.setPageStep(slider_page_step)
        self._play_position.range = [0, int(self._frame_provider._end_time*
                                           self._play_position.float_scale)]
        print(f"range = {self._play_position.range}")
        self.play_position_gui.setRange(0, self._play_position.range[1])
        self.play_position_gui.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.play_position_gui.update()
        self.play_position_gui.changed()

        print(f"ticks_per_frame {self._frame_provider._ticks_per_frame}")
        self.yuv_array = np.empty((1), dtype=np.uint8)
        self._initialized = True

    def init_and_display(self):
        if not self._initialized:
            self.init_video_av()
            self._frame_provider.set_time(0)
            self.update_position()
            self.display_frame()
        else:
            print(" --- video already initialized")

    def keyPressEvent(self, event):
        print("Key pressed")
        self._key_events.key_press_event(event)

def main():
    import argparse
    # import ffmpeg
    # import pprint
    # import numpy for generating random data points
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_video', nargs='+', help='video[:stream_number]')
    parser.add_argument('--pyav', action='store_true', help='Use pyav instead of ffmpeg bound with pybind11')
    parser.add_argument('--codec', type=str, default='', help='Use codec (ex: cuda) hardware acceleration with ffmpeg bound library')
    args = parser.parse_args()
    # _params = vars(args)
    print(args)

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    # These 3 lines solve a flickering issue by allowing immediate repaint
    # of QOpenGLWidget objects (see https://forum.qt.io/topic/99824/qopenglwidget-immediate-opengl-repaint/3)
    format = QtGui.QSurfaceFormat.defaultFormat()
    format.setSwapInterval(0)
    QtGui.QSurfaceFormat.setDefaultFormat(format)

    app = QtWidgets.QApplication()
    app.setApplicationDisplayName(f'ffmpeg on imview: {args.input_video}')

    main_window = QtWidgets.QMainWindow()
    main_widget = QtWidgets.QWidget(main_window)
    main_window.setCentralWidget(main_widget)
    main_layout  = QtWidgets.QVBoxLayout()
    video_layout = QtWidgets.QHBoxLayout()

    main_widget.setLayout(main_layout)
    players = []
    for input in args.input_video:
        if args.pyav:
            player = VideoPlayerAV(main_widget, use_decode_video_py=False, codec=args.codec)
        else:
            player = VideoPlayerAV(main_widget, codec=args.codec)
        player.set_video(input)
        video_layout.addWidget(player, 1)
        players.append(player)
    main_layout.addLayout(video_layout)

    # button_pauseplay = QtWidgets.QPushButton(">||")
    # main_layout.addWidget(button_pauseplay)
    # button_pauseplay.clicked.connect(lambda: pause_play(player1, player2))

    # sch = VideoScheduler(10)
    # for p in players:
    #     sch.add_player(p)

    # button_scheduler = QtWidgets.QPushButton("Scheduler")
    # main_layout.addWidget(button_scheduler)
    # button_scheduler.clicked.connect(sch.start_decode)

    for p in players:
        p.show()

    main_window.setMinimumHeight(100)
    main_window.show()

    # First display compared videos
    for n in range(1,len(players)):
        p.show()
        p.set_name(f'player{idx}')
        p.init_and_display()
        # First video is compare to all others
        players[0].compare(p)
    players[0].show()
    players[0].set_name('player0')
    players[0].init_and_display()
    
    main_window.resize(main_window.width(), 800)

    app.exec()
    # process(args.input_video, width, height)

if __name__ == '__main__':
    main()
