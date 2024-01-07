import os
import time
from typing import Union, Generator, List, Optional, Iterator
import numpy as np
import av
from av import container, VideoFrame # type: ignore
from av.frame import Frame
from cv2 import cvtColor, COLOR_YUV2RGB_I420 # type: ignore

from qimview.utils.qt_imports import QtWidgets, QtCore, QtGui
from qimview.image_viewers.qt_image_viewer import QTImageViewer
from qimview.image_viewers.gl_image_viewer_shaders import GLImageViewerShaders
from qimview.utils.viewer_image import ViewerImage, ImageFormat
from qimview.utils.thread_pool import ThreadPool
from qimview.video_player.video_scheduler     import VideoScheduler
from qimview.parameters.numeric_parameter     import NumericParameter
from qimview.parameters.numeric_parameter_gui import NumericParameterGui


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


class VideoPlayerAV(QtWidgets.QWidget):

    @staticmethod
    def useful_array_uint8(plane, crop=True):
        """
        Return the useful part of the VideoPlane as a single dimensional array.

        We are simply discarding any padding which was added for alignment.
        """
        total_line_size = abs(plane.line_size)
        arr = np.frombuffer(plane, np.uint8).reshape(-1, total_line_size)
        if crop:
            arr = arr[:,:plane.width]
        return np.ascontiguousarray(arr)

    @staticmethod
    def to_ndarray_v1(frame, yuv_array: np.ndarray, crop=False) -> Optional[np.ndarray]:
        if frame.format.name in ('yuv420p', 'yuvj420p'):
            # assert frame.width % 2 == 0
            # assert frame.height % 2 == 0
            # assert frame.planes[0].line_size == 2*frame.planes[1].line_size
            # assert frame.planes[0].width     == 2*frame.planes[1].width
            # assert frame.planes[1].line_size == frame.planes[2].line_size
            # assert frame.planes[1].width     == frame.planes[2].width
            width = frame.planes[0].line_size
            v0 = VideoPlayerAV.useful_array_uint8(frame.planes[0], crop=crop).ravel()
            v1 = VideoPlayerAV.useful_array_uint8(frame.planes[1], crop=crop).ravel()
            v2 = VideoPlayerAV.useful_array_uint8(frame.planes[2], crop=crop).ravel()
            total_size = v0.size+ v1.size + v2.size
            if yuv_array.size != total_size:
                output_array = np.empty((total_size,), dtype=np.uint8)
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
        if frame.format.name in ('yuv420p', 'yuvj420p'):
            y = VideoPlayerAV.useful_array_uint8(frame.planes[0], crop=False)
            u = VideoPlayerAV.useful_array_uint8(frame.planes[1], crop=False)
            v = VideoPlayerAV.useful_array_uint8(frame.planes[2], crop=False)
            return [y,u,v]
        else:
            return None

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.event_recorder = None
        self.main_widget = self
        vertical_layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(vertical_layout)
        # self.viewer_class = QTImageViewer
        self.viewer_class = GLImageViewerShaders
        self.widget: Union[GLImageViewerShaders, QTImageViewer]
        self.widget = self.viewer_class() # event_recorder = self.event_recorder)
        # don't show the histogram
        self.widget.show_histogram = False
        self.widget._show_text = False
        self.setGeometry(0, 0, self.widget.width(), self.widget.height())
        hor_layout = QtWidgets.QHBoxLayout()
        self._button_play_pause = QtWidgets.QPushButton()
        self._icon_play = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        self._icon_pause = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause)
        self._button_play_pause.setIcon(self._icon_play)
        hor_layout.addWidget(self._button_play_pause)
        vertical_layout.addWidget(self.widget)
        vertical_layout.addLayout(hor_layout)

        # create slider
        # TODO: move this variable to __init__()
        self.play_position = NumericParameter()
        self.play_position.float_scale = 1000
        self.play_position_gui = NumericParameterGui(name="sec:", param=self.play_position)
        self.play_position_gui.decimals = 3
        self.play_position_gui.set_pressed_callback(self.pause)
        self.play_position_gui.set_released_callback(self.reset_play)
        self.play_position_gui.set_valuechanged_callback(self.slider_value_changed)
        self.play_position_gui.create()
        self.play_position_gui.add_to_layout(hor_layout)

        # is show required here?
        self.show()
        self._im = None
        self._pause_time : float = 0
        self._pause = False
        self._button_play_pause.clicked.connect(self.play_pause)
        self._container : container.InputContainer | None = None
        self._video_stream = None

        self._scheduler : VideoScheduler = VideoScheduler()
        self._start_video_time : float = 0
        self._skipped : int = 0
        self._frame : Optional[av.VideoFrame] = None
        self._frame_generator : Optional[Iterator[Frame]] = None
        self._displayed_pts : int = -1
        self._timer : Optional[QtCore.QTimer] = None
        self._name : str = "video player"
        self._duration : float = 0
        self._end_time : float = 0
        self._t1 : AverageTime = AverageTime()
        self._t2 : AverageTime = AverageTime()
        self._t3 : AverageTime = AverageTime()
        self._t4 : AverageTime = AverageTime()
        self._t5 : AverageTime = AverageTime()
        self._max_skip : int = 5

        self._filename : str = "none"
        self._basename : str = "none"

    def set_name(self, n:str):
        self._name = n

    def set_video(self, filename):
        self._filename = filename
        self._basename = os.path.basename(self._filename)

    def set_pause(self):
        self._pause = True
        self._pause_time = float(self._frame.pts * self._frame.time_base)
    
    def set_play(self):
        self.set_time(self._pause_time)
        self._start_video_time = self._pause_time
        self._pause = False

    def start_decode(self):
        if self._scheduler._timer.isActive():
            self._scheduler.pause()
        self._scheduler.set_players([self])
        self._scheduler.start_decode()

    def pause(self):
        self._was_active = self._scheduler._timer.isActive()
        self._scheduler.pause()
    
    def reset_play(self):
        if self._was_active:
            self._scheduler.play()

    def play_pause(self):
        if len(self._scheduler._players) == 0:
            self._button_play_pause.setIcon(self._icon_pause)
            self.start_decode()
        else:
            if self._scheduler._timer.isActive():
                self._was_active = self._scheduler._timer.isActive()
                self._scheduler.pause()
                self._button_play_pause.setIcon(self._icon_play)
            else:
                self._button_play_pause.setIcon(self._icon_pause)
                self._scheduler.play()

    def slider_value_changed(self):
        self.set_play_position()

    def set_play_position(self):
        print(f"self.play_position {self.play_position.float}")
        self.set_time(self.play_position.float)
        self._pause_time = self.play_position.float
        self.display_frame(self._frame)

    def update_position(self, precision=0.02):
        current_time = float(self._frame.pts * self._frame.time_base)
        if abs(self.play_position.float-current_time)>precision:
            self.play_position.float = current_time
            # Block signals to avoid calling changedValue signal
            self.play_position_gui.blockSignals(True)
            self.play_position_gui.updateGui()
            self.play_position_gui.blockSignals(False)

    def set_image(self, np_array, im_name, force_new=False):
        if self._im is None or force_new:
            format = ImageFormat.CH_Y if len(np_array.shape) == 2 else ImageFormat.CH_RGB
            self._im = ViewerImage(np_array, channels = format)
            self.widget.set_image_fast(self._im)
        else:
            if self._im.data.shape == np_array.shape:
                self._im._data = np_array
                self.widget.image_id += 1
            else:
                format = ImageFormat.CH_Y if len(np_array.shape) == 2 else ImageFormat.CH_RGB
                self._im = ViewerImage(np_array, channels = format)
                self.widget.set_image_fast(self._im)
        self.widget.image_name = im_name

    def set_image_YUV420(self, y, u, v, im_name):
        self._im = ViewerImage(y, channels = ImageFormat.CH_YUV420)
        self._im.u = u
        self._im.v = v
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

    def get_frame_number(self) -> int:
        if self._frame:
            return int(self._frame.pts * float(self._frame.time_base) * float(self._video_stream.average_rate))
        else:
            return -1

    def set_time(self, time_pos : float):
        """ set time position in seconds """
        print(f"set_time {time_pos} , _end_time={self._end_time}")
        if self._container is None:
            print("Video not initialized")
        else:
            framerate = self._video_stream.average_rate # get the frame rate
            frame_num = int(time_pos*framerate+0.5)
            time_base = float(self._video_stream.time_base) # get the time base

            current_frame_time = self._frame.pts * float(time_base)
            sec_frame   = int(self._frame.pts * float(time_base) * framerate)
            initial_pos = float(self._frame.pts * time_base)
            # print(f"{sec_frame} -> {frame_num}")
            if frame_num == sec_frame:
                print(f"Current frame is at the requested position {frame_num} {sec_frame}")
                return
            # if we look for a frame slightly after, don't use seek()
            try:
                if frame_num<sec_frame or frame_num > sec_frame + 10:
                    # seek to that nearest timestamp
                    self._container.seek(int(time_pos*1000000), whence='time', backward=True)
                    # It seems that the frame generator is not automatically updated
                    # if we are at the end of the video, we don't want to call next()
                    if current_frame_time>=self._end_time:
                        self._frame_generator = self._container.decode(video=0)
                    # get the next available frame
                    frame = next(self._frame_generator)
                    # get the proper key frame number of that timestamp
                    sec_frame = int(frame.pts * float(time_base) * framerate)
                    initial_pos = float(frame.pts * time_base)
                # print(f"missing {int((time_pos-sec_frame)/float(1000000*time_base))} ")
                for _ in range(sec_frame, frame_num):
                    frame = next(self._frame_generator)
            except StopIteration:
                print(f"Reached end of video stream")
                # Reset valid generator
                self._frame_generator = self._container.decode(video=0)
            else:
                sec_frame = int(frame.pts * time_base * framerate)
                print(f"Frame at {initial_pos:0.3f}->{float(frame.pts * time_base):0.3f} request {time_pos}")
                self._frame = frame
                # Update pause time
                if not self._scheduler._timer.isActive():
                    self._pause_time = float(self._frame.pts * self._frame.time_base)

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
        self.set_image_YUV420(y,u,v, f"{self._basename}: {self.get_frame_number()}")
        # update is not immediate
        # self.widget.viewer_update()
        self.widget.viewer_update()
        # self.widget.repaint()

    def display_frame(self, frame=None):
        if frame is None:
            frame = self._frame
        if self.viewer_class is GLImageViewerShaders:
            self.display_frame_YUV420(frame)
        else:
            self.display_frame_RGB(frame)

    def init_video_av(self):
        """ Initialize the container and frame generator """
        if self._scheduler._timer.isActive():
            self._scheduler.pause()
        print(f"filename = {self._filename}")
        self._container = av.open(self._filename)
        self._container.streams.video[0].thread_type = "FRAME"
        self._container.streams.video[0].thread_count = 4
        self._video_stream = self._container.streams.video[0]
        framerate = float(self._video_stream.average_rate) # get the frame rate
        print(f"framerate {framerate} {self._video_stream.width}x{self._video_stream.height}")

        fps = self._video_stream.base_rate
        time_base = self._video_stream.time_base
        self._ticks_per_frame = int((1/fps) / time_base)
        self._duration = float(self._video_stream.duration * time_base)
        self._end_time = float((self._video_stream.duration-self._ticks_per_frame) * time_base)

        print(f"duration = {self._duration} seconds")
        slider_single_step = int(self._ticks_per_frame*time_base*self.play_position.float_scale+0.5)
        slider_page_step   = int(self.play_position.float_scale+0.5)
        self.play_position_gui.setSingleStep(slider_single_step)
        self.play_position_gui.setPageStep(slider_page_step)
        self.play_position.range = [0, int(self._end_time*self.play_position.float_scale)]
        print(f"range = {self.play_position.range}")
        self.play_position_gui.setRange(0, self.play_position.range[1])
        self.play_position_gui.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.play_position_gui.update()
        self.play_position_gui.changed()

        # fps = eval(probe['streams'][0]['r_frame_rate'])
        # print(f"fps={fps}")
        # width = int(video_stream['width'])
        # height = int(video_stream['height'])

        print(f"ticks_per_frame {self._ticks_per_frame}")

        self._frame_generator = self._container.decode(video=0)
        self.frame_number = -1
        self.yuv_array = np.empty((1), dtype=np.uint8)

    def get_next_frame(self, verbose=False) -> bool:
        t = time.perf_counter()
        if not self._frame_generator:
            return False
        try:
            self._frame = next(self._frame_generator)
            self.frame_number += 1
        except (StopIteration, av.EOFError) as e:
            print(f"Reached end of video stream: Exception {e}")
            # Reset valid generator
            if self._scheduler._timer.isActive():
                self.play_pause()
            self._frame_generator = self._container.decode(video=0)
            return False
        else:
            d = time.perf_counter()-t
            s = 'gen '
            s += 'key ' if self._frame.key_frame else ''
            s += f'{d:0.3f} ' if d>=0.001 else ''
            s += f'{self._frame.pict_type}'
            s += f' {self._frame.pts}'
            if verbose:
                if s != 'gen P': print(s)
                else: print(s)
            return True

    def init_and_display(self):
        self.init_video_av()
        if self.get_next_frame():
            self.set_time(0)
            self.update_position()
            self.display_frame()
        else:
            print("Failed to get next frame")

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

    app.exec()
    # process(args.input_video, width, height)

if __name__ == '__main__':
    main()
