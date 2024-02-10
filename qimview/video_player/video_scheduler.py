from typing import List, TYPE_CHECKING
import time
from qimview.utils.qt_imports import QtCore
from qimview.video_player.video_frame_buffer import EndOfVideo, TimeOut
if TYPE_CHECKING:
    from qimview.video_player.video_player_av import VideoPlayerAV

class VideoScheduler:
    """ Create a timer to schedule frame extraction and display 
        Can schedule 2 videos by playing each of them alternatively (TODO: remove this feature?)
    """
    def __init__(self, interval=5):
        self._players          : List['VideoPlayerAV']  = []
        self._interval         : int                  = interval # intervals in ms
        self._timer            : QtCore.QTimer        = QtCore.QTimer()
        self._timer_counter    : int                  = 0
        self._start_clock_time : float                = 0
        self._skipped          : List[int]            = [0, 0]
        self._displayed_pts    : List[int]            = [0, 0]
        self._current_player   : int                  = 0
        self._playback_speed   : float                = 1
        # _speed_ok: Check current frame queue for each video, True if queue size > 1
        # if all active videos are ok, speed might be increased
        # if any active video is not ok, speed will be decreased by 2%
        self._speed_ok         : List[bool]           = [True, True]

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
        p.init_and_display()

    def pause(self):
        """ pause all playing videos """
        if self._timer.isActive():
            self._timer.stop()
            for idx, p in enumerate(self._players):
                p.set_pause()
                p.update_position()
                print(f" player {idx}: "
                     f"skipped = {self._skipped[idx]} "
                     f"{self._displayed_pts[idx]/p._frame_provider._ticks_per_frame}")
                p.display_times()
            # Reset skipped counters
            self._skipped = [0,0]
        else:
            print("timer not active")

    def play(self):
        """ play videos """
        if not self._timer.isActive():
            for p in self._players:
                p.set_play()
            # Set _start_clock_time after starting the players to avoid rushing them
            self._start_clock_time = time.perf_counter()
            self._timer.start()
        else:
            print("timer already active")

    def set_playback_speed(self, speed: float):
        if speed>=1/8 and speed<=8:
            self._playback_speed = speed
        else:
            print("Playback speed not in range [1/4,4]")

    def get_time_spent(self) -> float:
        time_spent : float = time.perf_counter() - self._start_clock_time
        if self._playback_speed != 1:
            time_spent = time_spent*self._playback_speed
        return time_spent
    
    def reset_time(self, player: 'VideoPlayerAV', expected_time: float):
        # Not working well to set time during playing
        # player.set_time(expected_time, exact=False)
        # TODO: does it work for several players, or _start_clock_time should be per player?
        self._start_clock_time = time.perf_counter()
        player._start_video_time = player._frame_provider.get_time()

    def check_next_frame(self) -> bool:
        """ Check if we need to get a new frame based on the time spent """
        self._current_player = (self._current_player + 1) % len(self._players)
        p : 'VideoPlayerAV' = self._players[self._current_player]
        if p.frame_provider is None or not p.frame_provider.frame:
            self.pause()
            return False
        # assert p._frame is not None, "No frame available"
        if p._pause:
            print(f"paused")
            return False
        next_frame_time : float = p.frame_provider.get_time() \
                                  + p._frame_provider._frame_duration \
                                  - p._start_video_time
        time_spent = self.get_time_spent()
        # if time_spent >= 10:
        #     self.pause()
        # print(f"{next_frame_time} {time_spent}")

        if time_spent>next_frame_time:
            if time_spent-next_frame_time<1:
                iter = 0
                ok = True
                while time_spent>next_frame_time and iter<p._max_skip*self._playback_speed and ok:
                    if iter>0:
                        self._skipped[self._current_player] +=1
                        # print(f" skipped {self._skipped[self._current_player]} / {p.frame_number},")
                    try:
                        ok = p.frame_provider.get_next_frame()
                    except EndOfVideo as e:
                        if self._timer.isActive():
                            p.play_pause()
                        raise EndOfVideo from e
                    except TimeOut:
                        if self._timer.isActive():
                            p.play_pause()
                        print("Pausing video due to timeout")
                        ok = False
                    if ok:
                        next_frame_time = p.frame_provider.get_time() \
                                          + p.frame_provider._frame_duration \
                                          - p._start_video_time
                        time_spent = self.get_time_spent()
                        iter +=1
                # if iter>1:
                #     print(f" skipped {iter-1} frames {self._skipped[self._current_player]} / {p.frame_number},")
                if p.frame_provider.frame_buffer:
                    self._speed_ok[self._current_player] = p.frame_provider.frame_buffer.size()>1
                return True
            else:
                self.reset_time(p, next_frame_time)
                return False
        else:
            return False

    def _display_frame(self):
        p : 'VideoPlayerAV' = self._players[self._current_player]
        if p._frame_provider._frame and \
            p._frame_provider._frame.pts != self._displayed_pts[self._current_player]:
            # print(f"*** {p._name} {time.perf_counter():0.4f}", end=' --')
            frame_time : float = p._frame_provider.get_time() - p._start_video_time
            time_spent = self.get_time_spent()
            if abs(time_spent-frame_time)>= 0.2: # self._timer.interval()/1000*2:
                print(f" frame {frame_time:0.3f} at time {time_spent:0.3f}")
            p.display_frame()
            # QtCore.QTimer.singleShot(1, loop.exit)
            # loop.exec()
            # QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 1)

            # p.repaint()
            self._displayed_pts[self._current_player] = p._frame_provider._frame.pts
            diff = abs(p._frame_provider.get_time()-p.play_position_gui.param.float)
            if p._frame_provider._frame.key_frame or diff>1:
                p.update_position()

            # print(f" done {time.perf_counter():0.4f}")

    def _timer_cmds(self):
        # print(f" _timer_cmds() {self._timer_counter} - ")
        self._timer_counter += 1

        # Without thread pool, it works quite well !
        try:
            if self.check_next_frame():
                self._display_frame()
            slow_down = not self._speed_ok[self._current_player]
            if slow_down:
                # self._playback_speed *= 0.98
                p : 'VideoPlayerAV' = self._players[self._current_player]
                # since speed is in log2, 2**0.5 = 1.035 approx, so decreasing by 3.5% approx.
                print(f"p.playback_speed.float {p.playback_speed.float}")
                p.playback_speed.float  = p.playback_speed.float - 0.05 # = self._playback_speed
                print(f"  -> p.playback_speed.float {p.playback_speed.float}")
                p.playback_speed_gui.updateGui()
                p.speed_value_changed()
                print(f"    -> p.playback_speed.float {p.playback_speed.float}")
                # print(f"new playback speed {self._playback_speed}")
                # Reset speed_ok to True
                self._speed_ok[self._current_player] = True
        except EndOfVideo:
            print("End of video")
            self.pause()

        # self._thread_pool.start_worker()
        # self._thread_pool.waitForDone()
        # print(f"time spent {(time.perf_counter()-self._start_clock_time)*1000:0.2f} ms")

    def start_decode(self, start_time : float =0):
        print(f"start_decode({start_time:0.3f})")
        # self.set_time(start_time, self._container)
        print(f" nb videos {len(self._players)}")
        for idx, p  in enumerate(self._players):
            self._init_player(idx)
            p._start_video_time = start_time
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
        self.play()
        # self._timer.start()
