from typing import Union, Generator, List, Optional
import time
import numpy as np
import av
from av import container, VideoFrame # type: ignore
from cv2 import cvtColor, COLOR_YUV2RGB_I420 # type: ignore
import fastremap

from qimview.utils.qt_imports import QtWidgets, QtCore
from qimview.image_viewers.qt_image_viewer import QTImageViewer
from qimview.image_viewers.gl_image_viewer_shaders import GLImageViewerShaders
from qimview.utils.viewer_image import ViewerImage, ImageFormat
from qimview.utils.thread_pool import ThreadPool

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
        button2 = QtWidgets.QPushButton("AV")
        button3 = QtWidgets.QPushButton("Thread Test")
        button_pauseplay = QtWidgets.QPushButton(">||")
        hor_layout.addWidget(button2)
        hor_layout.addWidget(button3)
        hor_layout.addWidget(button_pauseplay)
        vertical_layout.addWidget(self.widget)
        vertical_layout.addLayout(hor_layout)
        # is show required here?
        self.show()
        self._im = None
        self._pause_time : float = 0
        self._pause = False
        button2         .clicked.connect(self.start_video_av)
        button3         .clicked.connect(self.start_decode_with_thread)
        button_pauseplay.clicked.connect(self.pause_play)
        self._container = None
        self.filename   : str = ''

        self._start_clock_time : float = 0
        self._start_video_time : float = 0
        self._skipped : int = 0
        self._frame = None
        self._displayed_pts : int = -1
        self._timer : Optional[QtCore.QTimer] = None
        self._name : str = "video player"
        self._t1 : AverageTime = AverageTime()
        self._t2 : AverageTime = AverageTime()
        self._t3 : AverageTime = AverageTime()
        self._t4 : AverageTime = AverageTime()
        self._t5 : AverageTime = AverageTime()

    def set_name(self, n:str):
        self._name = n

    def set_video(self, filename):
        self.filename = filename

    def pause_play(self):
        self._pause = not self._pause
        print(f"self._pause = {self._pause}")
        if self._pause and self._timer:
            self._timer.stop()
            self._pause_time = float(self._frame.pts * self._frame.time_base)
        else:
            self.set_time(self._pause_time, self._container)
            self._start_video_time = self._pause_time
            self._start_clock_time = time.perf_counter()
            self._timer.start()

        if not self._pause:
            self.start_video_av(self._pause_time)

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
        format = ImageFormat.CH_YUV420
        self._im = ViewerImage(y, channels = format)
        self._im.u = u
        self._im.v = v
        self.widget.set_image(self._im)
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

    def set_time(self, time_pos : float, container):
        """ set time position in seconds """
        framerate = container.streams.video[0].average_rate # get the frame rate
        frame_num = int(time_pos*framerate)
        time_base = container.streams.video[0].time_base # get the time base
        container.seek(int(time_pos*1000000), whence='time', backward=True)  # seek to that nearest timestamp
        frame = next(container.decode(video=0)) # get the next available frame
        sec_frame = int(frame.pts * time_base * framerate)  # get the proper key frame number of that timestamp

        print(f"got frame at {float(frame.pts * time_base)}, expected {time_pos}")
        print(f"missing {int((time_pos-sec_frame)/float(1000000*time_base))} time_base {time_base}")
        for _ in range(sec_frame, frame_num):
           frame = next(container.decode(video=0))
        sec_frame = int(frame.pts * time_base * framerate)
        print(f"-> got frame at {float(frame.pts * time_base)}, expected {time_pos}")

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
        print("display YUV")
        y,u,v = VideoPlayerAV.to_yuv(frame)
        # print(f"y min {np.min(y)} y max {np.max(y)}")
        self.set_image_YUV420(y,u,v, f"frame_{frame.pts}")
        # update is not immediate
        # self.widget.viewer_update()
        self.widget.viewer_update()
        # self.widget.repaint()

    def display_frame(self, frame):
        if self.viewer_class is GLImageViewerShaders:
            self.display_frame_YUV420(frame)
        else:
            self.display_frame_RGB(frame)

    def sync_frame(self, frame):
        """ Display frame at the expected time, either skip frames or wait if time does not match """
        self.frame_number += 1
        time_base = float(frame.time_base)
        frame_time : float = float(frame.pts * time_base) - self._start_video_time
        time_spent : float = time.perf_counter() - self._start_clock_time
        # print(f"frame_time {frame_time} time_spent {time_spent}")
        if self._pause:
            self._pause_time = float(frame.pts * time_base)
            print(f"paused")
            return False
        if time_spent>frame_time and frame.pts%(3*self._ticks_per_frame) != 0:

            while time_spent>frame_time and frame.pts%(3*self._ticks_per_frame) != 0:
                frame = next(self.frame_generator)
                # print(frame.pts, end=' - ')
                self._skipped +=1
                frame_time = frame.pts * time_base - self._start_video_time
                time_spent = time.perf_counter() - self._start_clock_time
        if time_spent<frame_time:
            # wait a little bit
            while(time_spent<frame_time):
                time_spent = time.perf_counter() - self._start_clock_time
                # loop = QtCore.QEventLoop()
                # QtCore.QTimer.singleShot(4, loop.exit)
                # loop.exec()
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 4)

        self.display_frame(frame)

        ## Allow 1 ms
        if frame.pts%(6*self._ticks_per_frame)==0:
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(2, loop.exit)
            loop.exec()
            # QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 1)
        if frame.pts%(100*self._ticks_per_frame) == 0:
            print(f"frame {int(frame.pts/self._ticks_per_frame+0.5)}, skipped {self._skipped}")
            print(f"{float(frame.pts * time_base):0.2f} vs "
                    f"time spent {time.perf_counter() - self._start_clock_time + self._start_video_time:0.2f}")
        return True

    def init_video_av(self):
        """ Initialize the container and frame generator """
        if self._container is None:
            print(f"filename = {self.filename}")
            self._container : container.InputContainer = av.open(self.filename)
            self._container.streams.video[0].thread_type = "AUTO"
            self.video_stream = self._container.streams.video[0]
            framerate = float(self.video_stream.average_rate) # get the frame rate
            print(f"framerate {framerate}")
            # duration = probe['format']['duration']
            # fps = eval(probe['streams'][0]['r_frame_rate'])
            # print(f"fps={fps}")
            # width = int(video_stream['width'])
            # height = int(video_stream['height'])

            fps = self.video_stream.base_rate
            time_base = self.video_stream.time_base
            self._ticks_per_frame = int((1/fps) / time_base)
            print(f"ticks_per_frame {self._ticks_per_frame}")

            self.frame_generator : Generator = self._container.decode(video=0)
            self.frame_number = -1
            self.yuv_array = np.empty((1), dtype=np.uint8)

    def start_video_av(self, start_time: float =0):
        if self._container is None: self.init_video_av()
        self.set_time(start_time, self._container)
        self._start_video_time = start_time

        # Starting time in seconds
        self._start_clock_time = time.perf_counter()
        self._skipped = 0
        # expect format to be YUVJ420P, so values in 0-255
        n = 0
        for frame in self._container.decode(video=0):
            if n==0:
                print(f"frame format '{frame.format}'")
            ok = self.sync_frame(frame)
            n += 1
            if not ok: break

    def get_next_frame(self):
        t = time.perf_counter()
        self._frame = next(self.frame_generator)
        self.frame_number += 1
        d = time.perf_counter()-t
        s = 'gen '
        s += 'key ' if self._frame.key_frame else ''
        s += f'{d:0.3f} ' if d>=0.001 else ''
        s += f'{self._frame.pict_type}'
        s += f' {self._frame.pts}'
        if s != 'gen P': print(s)
        else: print(s)

    def check_next_frame(self, progress_callback=None):
        """ Check if we need to get a new frame based on the time spent """
        # print('==')
        # st = time.perf_counter()
        # if self._frame is None: self.get_next_frame()
        assert self._frame is not None, "No frame available"
        time_base = float(self._frame.time_base)
        if self._pause:
            self._pause_time = float(self._frame.pts * time_base)
            print(f"paused")
            return False
        next_frame_time : float = float((self._frame.pts+self._ticks_per_frame) * time_base) - self._start_video_time
        time_spent : float = time.perf_counter() - self._start_clock_time
        # print(f"{next_frame_time} {time_spent}")

        if time_spent>next_frame_time:
            iter = 0
            while time_spent>next_frame_time and iter<10:
                if iter>0:
                    self._skipped +=1
                    print(f" skipped {self._skipped},")
                self.get_next_frame()
                next_frame_time = float((self._frame.pts+self._ticks_per_frame) * time_base) - self._start_video_time
                time_spent = time.perf_counter() - self._start_clock_time
                iter +=1

    def _display_current_frame(self):
        print(f"*** {self._name} {time.perf_counter():0.4f}", end=' --')
        if self._frame.pts != self._displayed_pts:
            frame_time : float = float(self._frame.pts * float(self._frame.time_base)) - self._start_video_time
            time_spent : float = time.perf_counter() - self._start_clock_time
            if abs(time_spent-frame_time)>=self._timer.interval()*1000:
                print(f" frame {frame_time:0.3f} at time {time_spent:0.3f}")
            # print(f"{self.widget.context()}")
            self.display_frame(self._frame)
            self._displayed_pts = self._frame.pts
        print(f" done {self._name} {time.perf_counter():0.4f}")

    def _timer_cmds(self):
        # print(f" _timer_cmds() {self._timer_counter} - ")
        self._timer_counter += 1
        if self._timer_counter == 1000:
            self._timer.stop()
            print(f" skipped = {self._skipped}")

        self._thread_pool.start_worker()
        self._thread_pool.waitForDone()
        # print(f"time spent {(time.perf_counter()-self._start_clock_time)*1000:0.2f} ms")

    def start_decode_with_thread(self):
        # self.set_time(start_time, self._container)
        # self._start_video_time = start_time

        if self._container is None:
            print("init")
            self.init_video_av()
            # get the first frame which is slower
            self.get_next_frame()
            self._display_current_frame()

        # Starting time in seconds
        if self._start_clock_time == 0:
            self._start_clock_time = time.perf_counter()
            self._skipped = 0

        # have a timer within the thread, probably better?
        # expect format to be YUVJ420P, so values in 0-255
        self._displayed_pts : int = -1
        self._timer : QtCore.QTimer = QtCore.QTimer()
        self._timer_counter = 0

        self._thread_pool : ThreadPool = ThreadPool()
        self._thread_pool.set_worker(self.check_next_frame)
        self._thread_pool.set_autodelete(False)
        self._thread_pool.set_worker_callbacks(finished_cb=self._display_current_frame)

        # Set a 5 ms counter
        self._timer.timeout.connect(self._timer_cmds)
        self._timer.start(5)

class Scheduler:
    """ Create a timer to schedule frame extraction and display """
    def __init__(self, interval=5):
        self._players          : List[VideoPlayerAV]  = []
        self._interval         : int                  = interval # intervals in ms
        self._timer            : QtCore.QTimer        = QtCore.QTimer()
        self._timer_counter    : int                  = 0
        self._start_clock_time : float                = 0
        self._skipped          : List[int]            = [0, 0]
        self._displayed_pts    : List[int]            = [0, 0]
        self._current_player   : int                  = 0

    def add_player(self, p:VideoPlayerAV):
        print("add_player")
        self._players.append(p)

    def _init_player(self, played_idx: int = 0):
        print("_init_player")
        assert played_idx>=0 and played_idx<len(self._players), "Wrong player idx"
        p = self._players[played_idx]
        if p._container is None:
            p.init_video_av()
            # get the first frame which is slower
            p.get_next_frame()
            p._display_current_frame()

    def check_next_frame(self, player_idx: int = 0, progress_callback=None):
        """ Check if we need to get a new frame based on the time spent """
        self._current_player = (self._current_player + 1) % len(self._players)
        p : VideoPlayerAV = self._players[self._current_player]
        # print('==')
        # st = time.perf_counter()
        # if self._frame is None: self.get_next_frame()
        assert p._frame is not None, "No frame available"
        time_base = float(p._frame.time_base)
        if p._pause:
            p._pause_time = float(p._frame.pts * time_base)
            print(f"paused")
            return False
        next_frame_time : float = float((p._frame.pts+p._ticks_per_frame) * time_base) - p._start_video_time
        time_spent : float = time.perf_counter() - self._start_clock_time
        print(f"{next_frame_time} {time_spent}")

        if time_spent>next_frame_time:
            iter = 0
            while time_spent>next_frame_time and iter<10:
                if iter>0:
                    self._skipped[self._current_player] +=1
                    print(f" skipped {self._skipped[self._current_player]} / {p.frame_number},")
                p.get_next_frame()
                next_frame_time = float((p._frame.pts+p._ticks_per_frame) * time_base) - p._start_video_time
                time_spent = time.perf_counter() - self._start_clock_time
                iter +=1
            return True
        else:
            return False

    def _display_frame(self, player_idx: int = 0):
        p : VideoPlayerAV = self._players[self._current_player]
        if p._frame and p._frame.pts != self._displayed_pts[self._current_player]:
            # print(f"*** {p._name} {time.perf_counter():0.4f}", end=' --')
            frame_time : float = float(p._frame.pts * float(p._frame.time_base)) - p._start_video_time
            time_spent : float = time.perf_counter() - self._start_clock_time
            if abs(time_spent-frame_time)>=self._timer.interval()*1000:
                print(f" frame {frame_time:0.3f} at time {time_spent:0.3f}")
            p.display_frame(p._frame)
            # p.widget.context().swapBuffers()
            # QtCore.QTimer.singleShot(1, loop.exit)
            # loop.exec()
            # QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 1)

            # p.repaint()
            # print("<<===")
            if p.viewer_class is GLImageViewerShaders:
                loop = QtCore.QEventLoop()
                loop.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents, 1)
            # print("===>>")

            self._displayed_pts[self._current_player] = p._frame.pts

            # print(f" done {time.perf_counter():0.4f}")

    def _timer_cmds(self):
        print(f" _timer_cmds() {self._timer_counter} - ")
        self._timer_counter += 1
        if time.perf_counter() - self._start_clock_time >= 10:
            self._timer.stop()
            for idx in range(len(self._players)):
                p : VideoPlayerAV = self._players[idx]
                print(f" player {idx}: skipped = {self._skipped[idx]} {self._displayed_pts[idx]/p._ticks_per_frame}")
                p.display_times()

        # Without thread pool, it works quite well !
        if self.check_next_frame():
            self._display_frame()
        # self._thread_pool.start_worker()
        # self._thread_pool.waitForDone()
        # print(f"time spent {(time.perf_counter()-self._start_clock_time)*1000:0.2f} ms")

    def start_decode(self, player_idx: int = 0):
        print("start_decode")
        # self.set_time(start_time, self._container)
        # self._start_video_time = start_time
        print(f" nb videos {len(self._players)}")
        for idx in range(len(self._players)):
            self._init_player(idx)
            self._displayed_pts[idx] = -1
            self._skipped[idx] = 0
        # Starting time in seconds
        if self._start_clock_time == 0:
            self._start_clock_time = time.perf_counter()

        # have a timer within the thread, probably better?
        # expect format to be YUVJ420P, so values in 0-255

        self._timer : QtCore.QTimer = QtCore.QTimer()
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self._interval)
        self._timer_counter = 0

        # self._thread_pool : ThreadPool = ThreadPool()
        # self._thread_pool.set_worker(self.check_next_frame)
        # self._thread_pool.set_autodelete(False)
        # self._thread_pool.set_worker_callbacks(finished_cb=self._display_frame)

        # Set a 5 ms counter
        self._timer.timeout.connect(self._timer_cmds)
        self._timer.start()

# Experimental code, kind of working but not so good
def sync_frames(player1, player2, frame1, frame2):
    """ Display frame at the expected time, either skip frames or wait if time does not match """
    time_base1 = float(frame1.time_base)
    frame_time1 : float = float(frame1.pts * time_base1) - player1._start_video_time
    time_base2 = float(frame2.time_base)
    frame_time2 : float = float(frame2.pts * time_base2) - player2._start_video_time
    time_spent1 : float = time.perf_counter() - player1._start_clock_time
    time_spent2 : float = time.perf_counter() - player2._start_clock_time
    if player1._pause:
        player1._pause_time = float(frame1.pts * time_base1)
        print(f"paused")
        return False
    if time_spent1>frame_time1 and frame1.pts%(3*player1._ticks_per_frame) != 0:

        while time_spent1>frame_time1 and frame1.pts%(3*player1._ticks_per_frame) != 0:
            frame1 = next(player1.frame_generator)
            frame2 = next(player2.frame_generator)
            player1._skipped +=1
            frame_time1 = frame1.pts * time_base1 - player1._start_video_time
            time_spent1 = time.perf_counter() - player1._start_clock_time
    if time_spent1<frame_time1:
        # wait a little bit
        while(time_spent1<frame_time1):
            # loop = QtCore.QEventLoop()
            # QtCore.QTimer.singleShot(4, loop.exit)
            # loop.exec()
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 2)
            time_spent1 = time.perf_counter() - player1._start_clock_time

    # player1.widget.makeCurrent()
    player1.display_frame(frame1)
    QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 1)
    # player2.widget.makeCurrent()
    player2.display_frame(frame2)
    QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 1)

    ## Allow 1 ms
    if frame1.pts%(6*player1._ticks_per_frame)==0:
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(2, loop.exit)
        loop.exec()
        # QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 1)
    if frame1.pts%(100*player1._ticks_per_frame) == 0:
        print(f"frame {int(frame1.pts/player1._ticks_per_frame+0.5)}, skipped {player1._skipped}")
        print(f"{float(frame1.pts * time_base1):0.2f} vs "
                f"time spent {time.perf_counter() - player1._start_clock_time + player1._start_video_time:0.2f}")
    return True

def pause_play(player1, player2):
    player1._pause = not player1._pause
    player2._pause = player1._pause
    print(f"player1._pause = {player1._pause}")
    if not player1._pause:
        if player1._container is None:
            player1.init_video_av()
        if player2._container is None:
            player2.init_video_av()
        player1.set_time(player1._pause_time, player1._container)
        player1._start_video_time = player1._pause_time
        player2.set_time(player1._pause_time, player2._container)
        player2._start_video_time = player1._pause_time

        # Starting time in seconds
        player1._start_clock_time = time.perf_counter()
        player1._skipped = 0
        player2._start_clock_time = time.perf_counter()
        player2._skipped = 0
        for frame1,frame2 in zip(player1._container.decode(video=0), player2._container.decode(video=0)):
            ok = sync_frames(player1, player2, frame1, frame2)
            if not ok: break

def pause_play_v0(player1, player2):
    player1._pause = not player1._pause
    player2._pause = player1._pause
    print(f"player1._pause = {player1._pause}")
    if not player1._pause:
        if player1._container is None:
            player1.init_video_av()
        if player2._container is None:
            player2.init_video_av()
        player1.set_time(player1._pause_time, player1._container)
        player1._start_video_time = player1._pause_time
        player2.set_time(player1._pause_time, player2._container)
        player2._start_video_time = player1._pause_time

        # Starting time in seconds
        player1._start_clock_time = time.perf_counter()
        player1._skipped = 0
        player2._start_clock_time = time.perf_counter()
        player2._skipped = 0
        for frame1,frame2 in zip(player1._container.decode(video=0), player2._container.decode(video=0)):
            ok = player1.sync_frame(frame1)
            if not ok: break
            ok = player2.sync_frame(frame2)
            if not ok: break

def main():
    import argparse
    import ffmpeg
    import pprint
    # import numpy for generating random data points
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_video', nargs='+', help='input image (if not specified, will open file dialog)')
    args = parser.parse_args()
    # _params = vars(args)
    print(args)

    # probe = ffmpeg.probe(args.input_video[0])
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(probe)
    # video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    # width = int(video_stream['width'])
    # height = int(video_stream['height'])
    # print(f" width x height = {width}x{height}")

    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
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
    player2 = VideoPlayerAV(main_widget)
    video_layout.addWidget(player2, 1)
    main_layout.addLayout(video_layout)

    button_pauseplay = QtWidgets.QPushButton(">||")
    main_layout.addWidget(button_pauseplay)
    button_pauseplay.clicked.connect(lambda: pause_play(player1, player2))

    sch = Scheduler(10)
    sch.add_player(player1)
    if len(args.input_video) == 2:
        sch.add_player(player2)

    button_scheduler = QtWidgets.QPushButton("Scheduler")
    main_layout.addWidget(button_scheduler)
    button_scheduler.clicked.connect(sch.start_decode)

    player1.set_video(args.input_video[0])
    player1.show()

    if len(args.input_video) == 2:
        player2.set_video(args.input_video[1])
        player2.show()
    else:
        player2.hide()

    player1.set_name('player1')
    player1.init_video_av()
    player1.get_next_frame()

    if len(args.input_video) == 2:
        player2.set_name('player2')
        player2.init_video_av()
        player2.get_next_frame()

    main_window.show()
    app.exec()
    # process(args.input_video, width, height)

if __name__ == '__main__':
    main()
