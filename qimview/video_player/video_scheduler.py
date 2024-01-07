from typing import List, TYPE_CHECKING
import time
from qimview.utils.qt_imports import QtCore
if TYPE_CHECKING:
    from qimview.video_player.video_player_av import VideoPlayerAV

class VideoScheduler:
    """ Create a timer to schedule frame extraction and display """
    def __init__(self, interval=5):
        self._players          : List['VideoPlayerAV']  = []
        self._interval         : int                  = interval # intervals in ms
        self._timer            : QtCore.QTimer        = QtCore.QTimer()
        self._timer_counter    : int                  = 0
        self._start_clock_time : float                = 0
        self._skipped          : List[int]            = [0, 0]
        self._displayed_pts    : List[int]            = [0, 0]
        self._current_player   : int                  = 0

    def set_interval(self, interval: int):
        """ Set scheduler interval in ms """
        self._interval = interval

    def add_player(self, p:'VideoPlayerAV'):
        print("add_player")
        self._players.append(p)

    def set_players(self, lp: List['VideoPlayerAV']):
        self._players = lp

    def _init_player(self, played_idx: int = 0):
        print("_init_player")
        assert played_idx>=0 and played_idx<len(self._players), "Wrong player idx"
        p = self._players[played_idx]
        p.init_video_av()
        # get the first frame which is slower
        p.get_next_frame()
        p.display_frame(p._frame)

    def pause(self):
        """ pause all playing videos """
        if self._timer.isActive():
            self._timer.stop()
            for idx, p in enumerate(self._players):
                p.set_pause()
                p.update_position()
                print(f" player {idx}: "
                    f"skipped = {self._skipped[idx]} {self._displayed_pts[idx]/p._ticks_per_frame}")
                p.display_times()
        else:
            print("timer not active")

    def play(self):
        """ play videos """
        if not self._timer.isActive():
            for p in self._players:
                p.set_play()
            self._timer.start()
            self._start_clock_time = time.perf_counter()
        else:
            print("timer already active")

    def check_next_frame(self, player_idx: int = 0, progress_callback=None):
        """ Check if we need to get a new frame based on the time spent """
        self._current_player = (self._current_player + 1) % len(self._players)
        p : 'VideoPlayerAV' = self._players[self._current_player]
        assert p._frame is not None, "No frame available"
        time_base = float(p._frame.time_base)
        if p._pause:
            p._pause_time = float(p._frame.pts * time_base)
            print(f"paused")
            return False
        next_frame_time : float = float((p._frame.pts+p._ticks_per_frame) * time_base) - p._start_video_time
        time_spent : float = time.perf_counter() - self._start_clock_time
        # if time_spent >= 10:
        #     self.pause()
        # print(f"{next_frame_time} {time_spent}")

        if time_spent>next_frame_time:
            iter = 0
            ok = True
            while time_spent>next_frame_time and iter<p._max_skip and ok:
                if iter>0:
                    self._skipped[self._current_player] +=1
                    # print(f" skipped {self._skipped[self._current_player]} / {p.frame_number},")
                ok = p.get_next_frame()
                if ok:
                    next_frame_time = float((p._frame.pts+p._ticks_per_frame) * time_base) - p._start_video_time
                    time_spent = time.perf_counter() - self._start_clock_time
                    iter +=1
            if iter>1:
                print(f" skipped {iter-1} frames {self._skipped[self._current_player]} / {p.frame_number},")
            return True
        else:
            return False

    def _display_frame(self):
        p : 'VideoPlayerAV' = self._players[self._current_player]
        if p._frame and p._frame.pts != self._displayed_pts[self._current_player]:
            # print(f"*** {p._name} {time.perf_counter():0.4f}", end=' --')
            frame_time : float = float(p._frame.pts * float(p._frame.time_base)) - p._start_video_time
            time_spent : float = time.perf_counter() - self._start_clock_time
            if abs(time_spent-frame_time)>= 0.04: # self._timer.interval()/1000*2:
                print(f" frame {frame_time:0.3f} at time {time_spent:0.3f}")
            p.display_frame(p._frame)
            # QtCore.QTimer.singleShot(1, loop.exit)
            # loop.exec()
            # QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 1)

            # p.repaint()
            self._displayed_pts[self._current_player] = p._frame.pts
            if p._frame.key_frame:
                p.update_position()
            # print(f" done {time.perf_counter():0.4f}")

    def _timer_cmds(self):
        # print(f" _timer_cmds() {self._timer_counter} - ")
        self._timer_counter += 1

        # Without thread pool, it works quite well !
        try:
            if self.check_next_frame():
                self._display_frame()
        except StopIteration:
            print("End of video")
            self.pause()

        # self._thread_pool.start_worker()
        # self._thread_pool.waitForDone()
        # print(f"time spent {(time.perf_counter()-self._start_clock_time)*1000:0.2f} ms")

    def start_decode(self, start_time=0):
        print("start_decode")
        # self.set_time(start_time, self._container)
        print(f" nb videos {len(self._players)}")
        for idx, p  in enumerate(self._players):
            p._start_video_time = start_time
            self._init_player(idx)
            self._displayed_pts[idx] = -1
            self._skipped[idx] = 0
        # Starting time in seconds
        self._start_clock_time = time.perf_counter()

        # have a timer within the thread, probably better?
        # expect format to be YUVJ420P, so values in 0-255

        self._timer : QtCore.QTimer = QtCore.QTimer()
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self._interval)
        self._timer_counter = 0

        # Finally, no need for threads
        # self._thread_pool : ThreadPool = ThreadPool()
        # self._thread_pool.set_worker(self.check_next_frame)
        # self._thread_pool.set_autodelete(False)
        # self._thread_pool.set_worker_callbacks(finished_cb=self._display_frame)

        # Set a 5 ms counter
        self._timer.timeout.connect(self._timer_cmds)
        self._timer.start()
