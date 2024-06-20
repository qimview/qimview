import os
import time
from typing import Union, Generator, List, Optional, Iterator
import numpy as np
import os
if os.name == 'nt' and os.path.isdir("c:\\ffmpeg\\bin"):
    os.add_dll_directory("c:\\ffmpeg\\bin")
    #os.add_dll_directory("C:\\Users\\karl\\GIT\\vcpkg\\packages\\ffmpeg_x64-windows-release\\bin")
import av
from av import container, VideoFrame # type: ignore
from av.frame import Frame
from cv2 import cvtColor, COLOR_YUV2RGB_I420 # type: ignore

from qimview.utils.qt_imports                          import QtWidgets, QtCore, QtGui
from qimview.image_viewers.gl_image_viewer_shaders     import GLImageViewerShaders
from qimview.utils.viewer_image                        import ViewerImage, ImageFormat
from qimview.video_player.video_scheduler              import VideoScheduler
from qimview.video_player.video_frame_provider         import VideoFrameProvider
from qimview.video_player.video_player_base            import VideoPlayerBase, ImageViewerClass

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

    @staticmethod
    def useful_array(plane, crop=True, dtype=np.uint8):
        """
        Return the useful part of the VideoPlane as a single dimensional array.

        We are simply discarding any padding which was added for alignment.
        """
        total_line_size = int(abs(plane.line_size)/dtype().itemsize)
        arr = np.frombuffer(plane, dtype).reshape(-1, total_line_size)
        if crop:
            arr = arr[:,:plane.width]
            return np.ascontiguousarray(arr)
        else:
            return arr

    @staticmethod
    def to_ndarray_v1(frame, yuv_array: np.ndarray, crop=False) -> Optional[np.ndarray]:
        match frame.format.name:
            case 'yuv420p' | 'yuvj420p':
                dtype = np.uint8
            case 'yuv420p10le':
                dtype = np.uint16
            case _:
                dtype = None
        if dtype is not None:
            # assert frame.width % 2 == 0
            # assert frame.height % 2 == 0
            # assert frame.planes[0].line_size == 2*frame.planes[1].line_size
            # assert frame.planes[0].width     == 2*frame.planes[1].width
            # assert frame.planes[1].line_size == frame.planes[2].line_size
            # assert frame.planes[1].width     == frame.planes[2].width
            # width = frame.planes[0].line_size
            v0 = VideoPlayerAV.useful_array(frame.planes[0], crop=crop, dtype=dtype).ravel()
            v1 = VideoPlayerAV.useful_array(frame.planes[1], crop=crop, dtype=dtype).ravel()
            v2 = VideoPlayerAV.useful_array(frame.planes[2], crop=crop, dtype=dtype).ravel()
            total_size = v0.size+ v1.size + v2.size
            if yuv_array.size != total_size:
                output_array = np.empty((total_size,), dtype=dtype)
            else:
                output_array = yuv_array
            output_array[0:v0.size]                                   = v0
            output_array[v0.size:(v0.size+v1.size)]                   = v1
            output_array[(v0.size+v1.size):(v0.size+v1.size+v2.size)] = v2
            return output_array
            # if output_array.size == total_size:
            # else:
            #     # print(f"{v0.shape} {v1.shape} {v2.shape}")
            #     return np.hstack((v0, v1, v2)).reshape(-1, width)
        else:
            return None

    def to_yuv(frame) -> Optional[List[np.ndarray]]:
        match frame.format.name:
            case 'yuv420p' | 'yuvj420p':
                dtype = np.uint8
            case 'yuv420p10le':
                dtype = np.uint16
            case _:
                print(f"Unknow format {frame.format.name}")
                dtype = None
        if dtype is not None:
            y = VideoPlayerAV.useful_array(frame.planes[0], crop=False, dtype=dtype)
            u = VideoPlayerAV.useful_array(frame.planes[1], crop=False, dtype=dtype)
            v = VideoPlayerAV.useful_array(frame.planes[2], crop=False, dtype=dtype)
            return [y,u,v]
        else:
            return None


    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.event_recorder = None
        # is show required here?
        self.show()
        self._im = None
        self._button_play_pause.clicked.connect(self.play_pause)
        self._container : container.InputContainer | None = None

        self._scheduler : VideoScheduler = VideoScheduler()
        self._start_video_time : float = 0
        self._skipped : int = 0
        self._frame_provider : VideoFrameProvider = VideoFrameProvider()
        self._displayed_pts : int = -1
        self._timer : Optional[QtCore.QTimer] = None
        self._name : str = "video player"
        self._t1 : AverageTime = AverageTime()
        self._t2 : AverageTime = AverageTime()
        self._t3 : AverageTime = AverageTime()
        self._t4 : AverageTime = AverageTime()
        self._t5 : AverageTime = AverageTime()
        self._max_skip : int = 10

        self._filename : str = "none"
        self._basename : str = "none"
        self._initialized : bool = False
        self._compare_players : list[VideoPlayerAV] = []

    @property
    def frame_provider(self) -> Optional[VideoFrameProvider]:
        return self._frame_provider

    def compare(self, player):
        self._compare_players.append(player)

    def on_synchronize(self, viewer : ImageViewerClass) -> None:
        # Synchronize other viewer to calling viewer
        copy_filters = True
        for v in self._compare_players:
            if v.widget != viewer:
                 viewer.synchronize_data(v.widget)
                 if copy_filters:
                     v.widget.filter_params.copy_from(viewer.filter_params)
                 v.widget.viewer_update()

    def set_name(self, n:str):
        self._name = n

    def set_video(self, filename):
        self._filename = filename
        self._basename = os.path.basename(self._filename)
        self._initialized = False

    def set_pause(self):
        """ Method called from scheduler """
        self._frame_provider.playing = False

    def pause(self):
        """ Method called from video player """
        self._was_active = self._scheduler._timer.isActive()
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
        if self._scheduler._timer.isActive():
            self._scheduler.pause()
        players = [self]
        players.extend(self._compare_players)
        self._scheduler.set_players(players)
        self._scheduler.start_decode(self._frame_provider.get_time())

    def play_pause(self):
        if len(self._scheduler._players) == 0:
            self._button_play_pause.setIcon(self._icon_pause)
            self.start_decode()
        else:
            if self._scheduler._timer.isActive():
                self.pause()
                self._button_play_pause.setIcon(self._icon_play)
            else:
                self._button_play_pause.setIcon(self._icon_pause)
                self._scheduler.play()

    def slider_value_changed(self):
        self.set_play_position()

    def set_play_position(self, recursive=True):
        print(f"self.play_position {self.play_position.float}")
        if self._frame_provider.frame_buffer:
            self._frame_provider.frame_buffer.reset()
        self._frame_provider.set_time(self.play_position.float)
        self._start_video_time = self.play_position.float
        for p in self._compare_players:
            p.play_position = self.play_position
            p.set_play_position(recursive=False)
            p.update_position(recursive=False)
        self.display_frame(self._frame_provider.frame)

    def speed_value_changed(self):
        print(f"New speed value {self.playback_speed.float}")
        self._scheduler.set_playback_speed(pow(2,self.playback_speed.float))

    def update_position(self, precision=0.02, recursive=True) -> bool:
        current_time = self._frame_provider.get_time()
        if abs(self.play_position.float-current_time)>precision:
            self.play_position.float = current_time
            # Block signals to avoid calling changedValue signal
            self.play_position_gui.blockSignals(True)
            self.play_position_gui.updateGui()
            self.play_position_gui.blockSignals(False)
            if recursive:
                for p in self._compare_players:
                    p.update_position(recursive=False)
            return True
        return False

    def set_image(self, np_array, im_name, force_new=False):
        prec = 8

        if self._im is None or force_new:
            format = ImageFormat.CH_Y if len(np_array.shape) == 2 else ImageFormat.CH_RGB
            self._im = ViewerImage(np_array, channels = format, precision=prec)
            self.widget.set_image_fast(self._im)
        else:
            if self._im.data.shape == np_array.shape:
                self._im._data = np_array
                self.widget.image_id += 1
            else:
                format = ImageFormat.CH_Y if len(np_array.shape) == 2 else ImageFormat.CH_RGB
                self._im = ViewerImage(np_array, channels = format, precision=prec)
                self.widget.set_image_fast(self._im)
        self.widget.image_name = im_name

    def set_image_YUV420(self, y, u, v, im_name):
        prec = 8
        match y.dtype:
            case np.uint8:  prec=8
            case np.uint16: prec=10

        self._im = ViewerImage(y, channels = ImageFormat.CH_YUV420, precision=prec)
        self._im.u = u
        self._im.v = v
        if len(self._compare_players)>0:
            # Use image from _compare_player as a ref?
            self.widget.set_image_fast(self._im, image_ref = self._compare_players[0]._im)
        else:
            self.widget.set_image_fast(self._im)
        self.widget.image_name = im_name

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

    def display_frame_RGB(self, frame: av.VideoFrame):
        """ Convert YUV to RGB and display RGB frame """
        st = time.perf_counter()
        # a = frame.to_ndarray(format='rgb24')
        # to start with keep all pixels, then crop only at the end
        crop_yuv=True
        self.yuv_array = VideoPlayerAV.to_ndarray_v1(frame, self.yuv_array, crop=crop_yuv)
        # print(f"display_frame_RGB() 1. {(time.perf_counter()-st)*1000:0.1f} ms")
        self._t1.add_time(time.perf_counter()-st)
        if crop_yuv:
            a = cvtColor(self.yuv_array.reshape(-1,frame.planes[0].width), COLOR_YUV2RGB_I420)
            # print(f"display_frame_RGB() 2. {(time.perf_counter()-st)*1000:0.1f} ms")
            self._t2.add_time(time.perf_counter()-st)
            # a bit slow, try inplace with fastremap
            # a = a[:,:frame.width]
            self._t3.add_time(time.perf_counter()-st)
        else:
            a = cvtColor(self.yuv_array.reshape(-1,frame.planes[0].line_size), COLOR_YUV2RGB_I420)
            # print(f"display_frame_RGB() 2. {(time.perf_counter()-st)*1000:0.1f} ms")
            self._t2.add_time(time.perf_counter()-st)
            # a bit slow, try inplace with fastremap
            a = a[:,:frame.width]
            self._t3.add_time(time.perf_counter()-st)
        # print(f"display_frame_RGB() 2.1 {(time.perf_counter()-st)*1000:0.1f} ms")
        if not a.flags.contiguous:
            a = np.ascontiguousarray(a)
        self._t4.add_time(time.perf_counter()-st)
        # print(f"display_frame_RGB() 3. {(time.perf_counter()-st)*1000:0.1f} ms")
        if frame.pts==0:
            self.set_image(a, f"frame_{frame.pts}")
            # QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 5)
        else:
            if self.viewer_class is GLImageViewerShaders:
                # print("set_image GL")
                self.set_image(a, f"frame_{frame.pts}", force_new=True)
            else:
                # print("set_image_data")
                if self._im is not None:
                    self.set_image_data(a)
                else:
                    self.set_image(a, "frame_0")
        self.widget.viewer_update()
        self._t5.add_time(time.perf_counter()-st)
        # print(f"display_frame_RGB() {(time.perf_counter()-st)*1000:0.1f} ms")

    def display_times(self):
        print(f"display_frame_RGB() 1. {self._t1.average()*1000:0.1f} ms")
        print(f"display_frame_RGB() 2. {self._t2.average()*1000:0.1f} ms")
        print(f"display_frame_RGB() 3. {self._t3.average()*1000:0.1f} ms")
        print(f"display_frame_RGB() 4. {self._t4.average()*1000:0.1f} ms")
        print(f"display_frame_RGB() 5. {self._t5.average()*1000:0.1f} ms")


    def display_frame_YUV420(self, frame: av.VideoFrame):
        """ Display YUV frame, for OpenGL with shaders """
        # a = frame.to_ndarray(format='rgb24')
        # to start with keep all pixels, then crop only at the end
        # print("display YUV")
        y,u,v = VideoPlayerAV.to_yuv(frame)
        # print(f"y min {np.min(y)} y max {np.max(y)}")
        self.set_image_YUV420(y,u,v, f"{self._basename}: {self._frame_provider.get_frame_number()}")
        # update is not immediate
        # self.widget.viewer_update()
        pl = frame.planes[0]
        im_width = y.data.shape[1]
        # print(f"frame.planes[0].width = {pl.width} y.data.shape[1] = {im_width}")
        if im_width != pl.width and self.viewer_class == GLImageViewerShaders:
            # Apply crop on the right
            self.widget.set_crop(np.array([0,0,pl.width/im_width,1], dtype=np.float32))
        else:
            self.widget.set_crop(np.array([0,0,1,1], dtype=np.float32))
        self.widget.viewer_update()
        # self.widget.repaint()

    def display_frame(self, frame=None):
        if frame is None:
            frame = self._frame_provider._frame
        if frame is None:
            return
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
        
        if self._scheduler._timer.isActive():
            self._scheduler.pause()
        print(f"filename = {self._filename}")
        if self._container is not None:
            self._container.close()
        self._container = av.open(self._filename)
        self._frame_provider.set_input_container(self._container)

        print(f"duration = {self._frame_provider._duration} seconds")
        slider_single_step = int(self._frame_provider._ticks_per_frame*
                                 self._frame_provider._time_base*
                                 self.play_position.float_scale+0.5)
        slider_page_step   = int(self.play_position.float_scale+0.5)
        self.play_position_gui.setSingleStep(slider_single_step)
        self.play_position_gui.setPageStep(slider_page_step)
        self.play_position.range = [0, int(self._frame_provider._end_time*
                                           self.play_position.float_scale)]
        print(f"range = {self.play_position.range}")
        self.play_position_gui.setRange(0, self.play_position.range[1])
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
            print(" --- video alread initialized")

def main():
    import argparse
    # import ffmpeg
    # import pprint
    # import numpy for generating random data points
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_video', nargs='+', help='input image (if not specified, will open file dialog)')
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
    player1 = VideoPlayerAV(main_widget)
    video_layout.addWidget(player1, 1)
    if len(args.input_video) == 2:
        player2 = VideoPlayerAV(main_widget)
        video_layout.addWidget(player2, 1)
    else:
        player2 = None
    main_layout.addLayout(video_layout)

    # button_pauseplay = QtWidgets.QPushButton(">||")
    # main_layout.addWidget(button_pauseplay)
    # button_pauseplay.clicked.connect(lambda: pause_play(player1, player2))

    sch = VideoScheduler(10)
    sch.add_player(player1)
    if player2:
        sch.add_player(player2)

    # button_scheduler = QtWidgets.QPushButton("Scheduler")
    # main_layout.addWidget(button_scheduler)
    # button_scheduler.clicked.connect(sch.start_decode)

    player1.set_video(args.input_video[0])
    player1.show()

    if player2:
        player2.set_video(args.input_video[1])
        player2.show()

    main_window.show()

    player1.set_name('player1')
    player1.init_and_display()

    if player2:
        player2.set_name('player2')
        player2.init_and_display()
        player1.compare(player2)

    app.exec()
    # process(args.input_video, width, height)

if __name__ == '__main__':
    main()
