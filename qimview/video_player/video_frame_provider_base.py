from typing import Optional, TypeVar, Protocol
from abc import abstractmethod
import time
from qimview.video_player.video_frame_buffer_base import VideoFrameBufferBase
from qimview.video_player.video_exceptions import EndOfVideo
from qimview.utils.qt_imports import QtWidgets, QtCore
import math

FRAMETYPE   = TypeVar('FRAMETYPE')
DECODERTYPE = TypeVar('DECODERTYPE')

class VideoFrameProviderBase(Protocol[FRAMETYPE, DECODERTYPE]):
    _frame_buffer    : Optional[VideoFrameBufferBase]     = None
    _playing         : bool                               = False
    _frame           : Optional[FRAMETYPE]                = None
    # Tell is current _frame comes from saved frames, in which case
    # the decoder is not positioned at the frame number
    _from_saved      : bool                               = False
    _framerate       : float                              = 0
    _time_base       : float                              = 0
    _frame_duration  : float                              = 0
    _ticks_per_frame : int                                = 0
    _duration        : float                              = 0
    _end_time        : float                              = 0
    _name            : str                                = 'VideoFrameProviderBase'

    _container       : Optional[DECODERTYPE] = None
    _video_stream    = None # : Optional[streams.StreamContainer]  = None
    _debug          : bool = False
    _guessed_gop_size : int = -1

    def __protocol_init__(self):
        pass

    @property
    def name(self) -> str: return self._name
    @name.setter
    def name(self, n:str): self._name = n

    @property
    def stream(self):
        return self._video_stream

    @property
    def frame_buffer(self) -> Optional[VideoFrameBufferBase]:
        return self._frame_buffer

    @property
    def frame(self) -> Optional[FRAMETYPE]:
        return self._frame

    @property
    def playing(self) -> bool:
        return self._playing
    
    @playing.setter
    def playing(self, p:bool):
        self._playing = p
        # Stop frame buffer thread
        if self._frame_buffer:
            if not self._playing:
                self._frame_buffer.pause_frames()
            else:
                self._frame_buffer.start_thread()

    def get_time(self) -> float:
        if self._frame:
            return self._frame.pts * self._time_base
        else:
            return -1

    def get_frame_number(self) -> int:
        if self._frame:
            return int(self._frame.pts/self._ticks_per_frame + 0.5)
        else:
            return -1

    @abstractmethod
    def seek_position(self, time_pos: float) -> bool:
        pass

    @abstractmethod
    def get_video_streams(self):
        pass

    @abstractmethod
    def set_stream_threads(self, stream):
        pass

    @abstractmethod
    def CreateFrameBuffer(self, video_stream_number : int):
        pass

    @abstractmethod
    def logStreamInfo(self):
        pass

    @abstractmethod
    def copyStreamInfo(self):
        pass

    def set_input_container(self, container: DECODERTYPE, video_stream_number=0):
        self._container = container
        print(f"set_input_container() gop_size= {self._container.get_codec_ctx().get().gop_size}")
        assert self._container is not None, "No decoder" 
        
        video_streams = self.get_video_streams()
        nb_videos = len(video_streams)
        print(f"Found {nb_videos} videos")
        if nb_videos == 0:
            print(f"ERROR: Video not found !")
            return
        # self._container.streams.video[0].fast = True
        self._video_stream = video_streams[video_stream_number]
        self.set_stream_threads(self._video_stream)
        self.logStreamInfo()
        self.copyStreamInfo()
        self.CreateFrameBuffer(video_stream_number)
        self._frame = None

    def set_time(self, time_pos : float, exact: bool =True) -> bool:
        """ set time position in seconds 
            the video frame corresponding to the requested time_pos is set in _frame member
        """
        # print(f"set_time {time_pos} , _end_time={self._end_time}")
        # print(f"set_time() gop_size= {self._container.get_codec_ctx().get().gop_size} {self._framerate=}")
        if self.frame_buffer is None or not self.frame_buffer.decoderOk():
            print("Video not initialized")
            return False
        else:
            frame_num = int(time_pos*self._framerate+0.5)
            current_frame_time = self.get_time()
            cur_frame   = int(current_frame_time * self._framerate + 0.5)
            initial_pos = current_frame_time
            frame       = None
            # print(f"{sec_frame} -> {frame_num}")
            if frame_num == cur_frame:
                print(f"Current frame is at the requested position {frame_num} {cur_frame}")
                return True

            # Check if frame is in the frame buffer saved frames,
            # this allows fast moving within saved frames
            if self._debug:
                print(f"{len(self.frame_buffer._saved_frames)=}", end=": ")
                for f in self.frame_buffer._saved_frames:
                    fn = int(f.pts * self._time_base* self._framerate + 0.5)
                    print(f'{fn}-{f.pts}', end=', ')
                print('')
            for f in self.frame_buffer._saved_frames:
                fn = int(f.pts * self._time_base* self._framerate + 0.5)
                if frame_num == fn:
                    self._frame = f
                    self._from_saved = True
                    if self._debug:
                        print(f"Found frame from saved ones")
                    return True

            # if we look for a frame slightly after, don't use seek()
            try:
                wait_cursor = False
                if self._from_saved or frame_num<cur_frame or frame_num > cur_frame + 10:
                    # seek to that nearest timestamp
                    ok = self.seek_position(time_pos)
                    if not ok:
                        print(f"Failed to seek posiiton {time_pos}")
                        return False
                    # It seems that the frame generator is not automatically updated
                    if time_pos==0:
                        self.frame_buffer.reset()
                    # get the next available frame
                    frame = self.frame_buffer.get_frame(timeout=1, save=False)
                    self.update_gop_size(frame)
                    if frame is None:
                        print(f"Failed to get frame")
                        return False
                    # get the proper key frame number of that timestamp
                    found_frame_pos = int(frame.pts * self._time_base * self._framerate + 0.5)
                    initial_pos = float(frame.pts * self._time_base)
                    # self.playing = is_playing
                else:
                    found_frame_pos = cur_frame
                # print(f"missing {int((time_pos-found_frame_pos)/float(1000000*time_base))} ")
                # Loop over next frames to reach the exact requested position
                if not frame or exact or (abs(frame_num -found_frame_pos) >= abs(frame_num-cur_frame)):
                    if QtWidgets.QApplication.instance() is not None:
                        wait_cursor = frame_num-found_frame_pos>10
                        if wait_cursor:
                            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
                    print(f" {found_frame_pos=} {frame_num=}")
                    for idx in range(found_frame_pos, frame_num):
                        save = (frame_num-idx)<2
                        if self._debug:
                            if save:
                                print(f'{idx}*', end=', ')
                            else:
                                print(f'{idx}', end=', ')
                        frame = self.frame_buffer.get_frame(save=save)
                        self.update_gop_size(frame)
                    if self._debug:
                        print('')
                else:
                    if not exact:
                        print(f" {found_frame_pos=} {frame_num=} {cur_frame=} {abs(frame_num -found_frame_pos)=} {abs(frame_num-cur_frame)=}")

            except EndOfVideo: #  as e:
                print(f"set_time(): Reached end of video stream")
                # Reset valid generator
                self.frame_buffer.reset()
                # raise EndOfVideo() from e
            else:
                if self._debug:
                    print(f"Frame at {initial_pos:0.3f}->"
                        f"{float(frame.pts * self._time_base):0.3f} requested {time_pos}")
                self._from_saved = False
                self._frame = frame
            finally:
                if wait_cursor:
                    QtWidgets.QApplication.restoreOverrideCursor()
                return True

    def update_gop_size(self, frame):
        if frame is not None and frame.key_frame:
            pos = int( frame.pts * self._time_base * self._framerate + 0.5)
            # print(f"key frame at {pos}")
            if pos>0:
                if self._guessed_gop_size == -1:
                    self._guessed_gop_size = pos
                    print(f"{self._guessed_gop_size=}")
                else:
                    if pos % self._guessed_gop_size != 0:
                        self._guessed_gop_size = math.gcd(self._guessed_gop_size, pos)
                        print(f" --> {self._guessed_gop_size=}")

    def get_next_frame(self, timeout=6,  verbose=False) -> bool:
        """ Obtain the next frame, usually while video is playing, 
            can raise EndofVideo exception """
        t = time.perf_counter()
        if not self.frame_buffer:
            print("Video Frame Buffer not created")
            return False
        try:
            self._frame = self._frame_buffer.get_frame(timeout=timeout)
        except EndOfVideo as e:
            print(f"get_next_frame(): Reached end of video stream: Exception {e}")
            # Reset valid generator
            self._frame_buffer.reset()
            raise EndOfVideo() from e
        else:
            self.update_gop_size(self._frame)
            if verbose:
                d = time.perf_counter()-t
                s = 'gen '
                if self._frame is not None:
                    s += 'key ' if self._frame.key_frame else ''
                    s += 'I ' if self._frame.interlaced_frame else ''
                s += f'{d:0.3f} ' if d>=0.001 else ''
                # s += f'{self._frame.pict_type}'
                if self._frame is not None:
                    s += f' {self._frame.pts}'
                if s != 'gen P':
                    print(s)
            return True
