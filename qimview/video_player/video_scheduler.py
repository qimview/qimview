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
        self._interval         : int                  = interval  # intervals in ms
        self._use_period_timer : bool                 = False     # use a periodical timer
        self._minimal_period   : int                  = 2         # minimal time between single shot time calls
        self._is_running       : bool                 = False     # used for single shot timer
        self._timer            : QtCore.QTimer        = QtCore.QTimer()
        self._timer_counter    : int                  = 0
        self._start_clock_time : float                = 0
        self._skipped          : List[int]            = []
        self._displayed_pts    : List[int]            = []
        self._playback_speed   : float                = 1
        self._loop             : bool                 = True
        # _speed_ok: Check current frame queue for each video, True if queue size > 1
        # if all active videos are ok, speed might be increased
        # if any active video is not ok, speed will be decreased by 2%
        self._speed_ok         : List[bool]           = []
        self._fps_start        : float                = -1        # start time to compute displayed FPS
        self._fps_count        : int                  = 0         # count displayed frames

    @property
    def is_running(self) -> bool:
        return self._is_running

    def set_interval(self, interval: int):
        """ Set scheduler interval in ms """
        self._interval = interval

    def add_player(self, p:'VideoPlayerAV'):
        print("add_player")
        self._players.append(p)
        self._skipped.append(0)
        self._displayed_pts.append(0)
        self._speed_ok.append(True)

    def set_players(self, lp: List['VideoPlayerAV']):
        self._players = lp
        self._skipped       = [0]*len(self._players)
        self._displayed_pts = [0]*len(self._players)
        self._speed_ok      = [True]*len(self._players)

    def _init_player(self, played_idx: int = 0):
        print("_init_player")
        assert played_idx>=0 and played_idx<len(self._players), "Wrong player idx"
        p = self._players[played_idx]
        p.init_and_display()

    def pause(self):
        """ pause all playing videos """
        if self.is_running:
            if self._use_period_timer:
                assert self._timer.isActive(), "_timer shoud be active"
                self._timer.stop()
            self._is_running = False
            for idx, p in enumerate(self._players):
                p.set_pause()
                p.update_position()
                print(f" player {idx}: "
                    f"skipped = {self._skipped[idx]} "
                    f"{self._displayed_pts[idx]/p._frame_provider._ticks_per_frame}")
                p.display_times()
            # Reset skipped counters
            self._skipped = [0]*len(self._players)

    def play(self):
        """ play videos """
        if not self.is_running:
            self._is_running = True
            for p in self._players:
                p.set_play()
            # Set _start_clock_time after starting the players to avoid rushing them
            self._start_clock_time = time.perf_counter()
            if self._use_period_timer:
                assert not self._timer.isActive(), "Timer should not be active with _is_running False"
                self._timer.start()
            else:
                self._display_remaining_frames()

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

    def check_next_frame(self, player_index:int) -> bool:
        """ Check if we need to get a new frame based on the time spent
            If display is late, skip frames until time is ok or max skip is reached
            Only called with periodic timer
        """
        p : 'VideoPlayerAV' = self._players[player_index]
        if p.frame_provider is None or not p.frame_provider.frame:
            self.pause()
            return False
        # assert p._frame is not None, "No frame available"
        if not p._frame_provider.playing:
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
                        self._skipped[player_index] +=1
                        # print(f" skipped {self._skipped[player_index]} / {p.frame_number},")
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
                #     print(f" skipped {iter-1} frames {self._skipped[player_index]} / {p.frame_number},")
                if p.frame_provider.frame_buffer:
                    self._speed_ok[player_index] = p.frame_provider.frame_buffer.size()>1
                return True
            else:
                self.reset_time(p, next_frame_time)
                return False
        else:
            return False

    def _display_frame(self, player_index:int):
        p : 'VideoPlayerAV' = self._players[player_index]
        if p._frame_provider._frame and \
            p._frame_provider._frame.pts != self._displayed_pts[player_index]:
            # print(f"*** {p._name} {time.perf_counter():0.4f}", end=' --')
            frame_time : float = p._frame_provider.get_time() - p._start_video_time
            time_spent = self.get_time_spent()
            if self._use_period_timer and abs(time_spent-frame_time)>= 0.2: # self._timer.interval()/1000*2:
                print(f" frame {frame_time:0.3f} at time {time_spent:0.3f}")
            p.display_frame()
            # QtCore.QTimer.singleShot(1, loop.exit)
            # loop.exec()
            # QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 1)

            # p.repaint()
            self._displayed_pts[player_index] = p._frame_provider._frame.pts
            diff = abs(p._frame_provider.get_time()-p.play_position_gui.param.float)
            if p._frame_provider._frame.key_frame or diff>1:
                p.update_position()

            # print(f" done {time.perf_counter():0.4f}")
    def _display_next_frame(self) -> bool:
        """
            Get and display next frame of each video player 
        """
        try:
            ok = True
            for p in self._players:
                if p.frame_provider is not None:
                    ok = ok and p.frame_provider.get_next_frame()
                else:
                    ok = False
                if not ok: break
            if ok:
                # display player 0 at the end to allow multiple frames on the same display
                for n in range(1,len(self._players)):
                    self._display_frame(n)
                self._display_frame(0)
            return ok
        except EndOfVideo:
            print("End of video")
            if self._loop:
                for p in self._players:
                    if p.frame_provider is not None:
                        p.frame_provider.set_time(p.loop_start_time)
            else:
                self.pause()
            return False

    def _display_remaining_frames(self):
        """
            display next frames and call itself with a timer
        """
        try:
            start_display = time.perf_counter()
            if start_display>self._fps_start+1:
                if self._fps_start >0:
                    print(f" displayed FPS: {self._fps_count}")
                self._fps_count = 0
                self._fps_start = start_display
            if self._display_next_frame():
                self._fps_count += 1
            display_duration = time.perf_counter() - start_display
            assert self._players[0].frame_provider is not None
            frame_duration = (self._players[0].frame_provider.frame_duration)/self._playback_speed
            #print(f"{display_duration=} {frame_duration=}")
            slow_down = max(self._minimal_period,int((frame_duration-display_duration)*1000))
            if self.is_running:
                QtCore.QTimer.singleShot(slow_down, lambda : self._display_remaining_frames())
        except Exception as e:
            print(f"Exception: {e}")
            # duration = time.perf_counter()-self.start_time
            # # TODO: get frame number from Frame
            # fps = self.frame_count/duration
            # print(f"took {duration:0.3f} sec  {fps:0.2f} fps")

    def _timer_cmds(self):
        # print(f" _timer_cmds() {self._timer_counter} - ")
        self._timer_counter += 1

        # Without thread pool, it works quite well !
        try:
            ok = True
            for n in range(len(self._players)):
                ok = ok and self.check_next_frame(n)
                if not ok: break
            if ok:
                # display player 0 at the end to allow multiple frames on the same display
                for n in range(1,len(self._players)):
                    self._display_frame(n)
                self._display_frame(0)
            slow_down = not self._speed_ok[0]
            if slow_down:
                # self._playback_speed *= 0.98
                p : 'VideoPlayerAV' = self._players[0]
                # since speed is in log2, 2**0.5 = 1.035 approx, so decreasing by 3.5% approx.
                print(f"p.playback_speed.float {p.playback_speed.float}")
                p.playback_speed.float  = p.playback_speed.float - 0.05 # = self._playback_speed
                print(f"  -> p.playback_speed.float {p.playback_speed.float}")
                p.playback_speed_gui.updateGui()
                p.speed_value_changed()
                print(f"    -> p.playback_speed.float {p.playback_speed.float}")
                # print(f"new playback speed {self._playback_speed}")
                # Reset speed_ok to True
                self._speed_ok[0] = True
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

        if self._use_period_timer:
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
        else:
            self.play()
        # self._timer.start()
