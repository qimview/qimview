from typing import List, TYPE_CHECKING
import time
from qimview.utils.qt_imports import QtCore
from qimview.video_player.video_exceptions import EndOfVideo, TimeOut
if TYPE_CHECKING:
    from qimview.video_player.video_player_av import VideoPlayerAV

class VideoScheduler:

    _name = "VideoScheduler"

    """ Create a timer to schedule frame extraction and display 
        Can schedule 2 videos by playing each of them alternatively (TODO: remove this feature?)
    """
    def __init__(self, interval=5):
        self._players          : List['VideoPlayerAV'] = []
        self._timeshifts       : List[float]          = []
        self._interval         : int                  = interval  # intervals in ms
        self._minimal_period   : int                  = 2         # minimal time between single shot time calls
        self._is_running       : bool                 = False     # used for single shot timer
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
        self._total_fps_count  : int                  = 0         # count displayed and skipped frames
        self._displayed_fps    : int                  = 0         # displayed FPS
        self._skipped_fps      : int                  = 0         # skipped FPS
        self._allow_skipping   : bool                 = True      # skip displaying frame if rendered FPS is late
        self._max_skip         : int                  = 6         # maximum number of successive skips

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

    def set_timeshifts(self, timeshifts: List[float]) -> None:
        self._timeshifts = timeshifts

    def _init_player(self, played_idx: int = 0):
        print("_init_player")
        assert played_idx>=0 and played_idx<len(self._players), "Wrong player idx"
        p = self._players[played_idx]
        p.init_and_display()

    def pause(self):
        """ pause all playing videos """
        if self.is_running:
            self._is_running = False
            for idx, p in enumerate(self._players):
                p.set_pause()
                print(f" player {idx}: "
                    f"skipped = {self._skipped[idx]} "
                    f"{self._displayed_pts[idx]/p._frame_provider._ticks_per_frame}")
                # p.display_times()
            # Reset skipped counters
            self._players[0].update_position(force=True)
            self._skipped = [0]*len(self._players)

    def play(self):
        """ play videos """
        if not self.is_running:
            self._is_running = True
            for p in self._players:
                p.set_play()
            # Set _start_clock_time after starting the players to avoid rushing them
            self._start_clock_time = time.perf_counter()
            print(f"{self._name}.play() ", end='')
            for idx, p in enumerate(self._players):
                print(f" player{idx}, pos = {p.play_position}", end = '')
            print()

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

    def _display_frame(self, player_index:int, is_running: bool):
        """_summary_

        Args:
            player_index (int): _description_
            is_running (bool): True is video is playing
        """
        p : 'VideoPlayerAV' = self._players[player_index]
        if p._frame_provider._frame and \
            p._frame_provider._frame.pts != self._displayed_pts[player_index]:
            # print(f"*** {p._name} {time.perf_counter():0.4f}", end=' --')
            # frame_time : float = p._frame_provider.get_time() - p._start_video_time
            # time_spent = self.get_time_spent()
            p.display_frame(frame=None, is_running=is_running)
            self._displayed_pts[player_index] = p._frame_provider._frame.pts
            diff = abs(p._frame_provider.get_time()-p.play_position_gui.param.float)
            # print(f"{self._name}._display_frame({player_index}) {self._displayed_pts[player_index]=}")
            if p._frame_provider._frame.key_frame or diff>1:
                p.update_position()

            # print(f" done {time.perf_counter():0.4f}")

    def _display_next_frame(self) -> bool:
        """
            Get and display next frame of each video player 
        """
        # print(f"{self._name}._display_next_frame()")
        try:
            ok = True
            for p in self._players:
                if p.frame_provider is not None:
                    ok = ok and p.frame_provider.get_next_frame()
                    # print(f" {p._name}.frame_provider.get_next_frame() --> {ok}")
                else:
                    ok = False
            if ok:
                # display player 0 at the end to allow multiple frames on the same display
                # synchronize the use of PBO throught the parameter is running, for initial and compared videos
                for n in range(1,len(self._players)):
                    self._display_frame(n, self.is_running)
                self._display_frame(0, self.is_running)
            return ok
        except EndOfVideo:
            print("_display_next_frame() End of video")
            min_start_time = 0
            for ts in self._timeshifts:
                min_start_time = min(min_start_time, -ts)
            if self._loop:
                for idx, p in enumerate(self._players):
                    if p.frame_provider is not None:
                        p.frame_provider.set_time(max(min_start_time,p.loop_start_time)+self._timeshifts[idx])
            else:
                self.pause()
            return False

    def _skip_next_frame(self) -> bool:
        """
            Get and display next frame of each video player 
        """
        try:
            ok = True
            for p in self._players:
                if p.frame_provider is not None:
                    ok = ok and p.frame_provider.get_next_frame(timeout=2)
                else:
                    ok = False
            return ok
        except EndOfVideo:
            print("_skip_next_frame() End of video")
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
        # print(f"{self._name}._display_remaining_frames()")
        try:
            start_display = time.perf_counter()
            # Compute frame duration based on first video only
            total_frame_duration   = 0
            nb_skip = 0
            if start_display>self._fps_start+1:
                if self._fps_start >0:
                    # print(f" displayed FPS: {self._fps_count} {self._total_fps_count}")
                    self._displayed_fps = self._total_fps_count
                    self._skipped_fps = self._total_fps_count -self._fps_count
                self._fps_count = 0
                self._total_fps_count = 0
                self._fps_start = start_display
            if self._display_next_frame():
                self._fps_count += 1
                self._total_fps_count += 1
            assert self._players[0].frame_provider is not None
            frame_duration = (self._players[0].frame_provider.frame_duration)/self._playback_speed
            total_frame_duration   += frame_duration
            display_duration = time.perf_counter() - start_display
            if self._allow_skipping:
                while (
                        display_duration     > total_frame_duration and 
                        nb_skip              < self._max_skip       and 
                        total_frame_duration < 0.05                 and 
                        display_duration     < 0.05 
                       ):
                    # print(f"{total_frame_duration:0.3f} {display_duration:0.3f}", end='  ')
                    skip_ok = self._skip_next_frame()
                    if skip_ok:
                        total_frame_duration   += frame_duration
                        nb_skip += 1
                        self._total_fps_count += 1
                    display_duration = time.perf_counter() - start_display
                    # print(f"{total_frame_duration:0.3f} {display_duration:0.3f}")
                if nb_skip>0:
                    print('*'*nb_skip, end=';')
            slow_down_delay = 0
            slow_down = max(self._minimal_period,int((total_frame_duration-display_duration)*1000+0.5)-slow_down_delay)
            if self.is_running:
                QtCore.QTimer.singleShot(slow_down, self._display_remaining_frames)
        except Exception as e:
            print(f"Exception: {e}")
            # duration = time.perf_counter()-self.start_time
            # # TODO: get frame number from Frame
            # fps = self.frame_count/duration
            # print(f"took {duration:0.3f} sec  {fps:0.2f} fps")

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
        self.play()
