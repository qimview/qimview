
from typing import Optional
import time
from qimview.video_player.video_frame_buffer_cpp import VideoFrameBufferCpp
from qimview.video_player.video_exceptions import EndOfVideo

import os

ffmpeg_path = os.path.join(os.environ.get('FFMPEG_ROOT', ''),'bin')
if os.name == 'nt' and os.path.isdir(ffmpeg_path):
    os.add_dll_directory(ffmpeg_path)

import decode_video_py as decode_lib

class VideoFrameProviderCpp:
    def __init__(self):
        self._frame_buffer    : Optional[VideoFrameBufferCpp]      = None
        self._decoder         : decode_lib.VideoDecoder | None     = None
        self._playing         : bool                               = False
        self._frame           = None #: Optional[VideoFrame]               = None
        self._video_stream    = None #: Optional[streams.StreamContainer]  = None
        self._framerate       : float                              = 0
        self._time_base       : float                              = 0
        self._frame_duration  : float                              = 0
        self._ticks_per_frame : int                                = 0
        self._duration        : float                              = 0
        self._end_time        : float                              = 0
        self._name            : str                                = 'VideoFrameProviderCpp'

    @property
    def name(self) -> str: return self._name
    @name.setter
    def name(self, n:str): self._name = n

    @property
    def frame_duration(self) -> float:
        return self._frame_duration

    @property
    def frame_buffer(self) -> Optional[VideoFrameBufferCpp]:
        return self._frame_buffer
    
    @property
    def stream(self): # -> Optional[streams.StreamContainer]:
        return self._video_stream

    @property
    def frame(self): # -> Optional[VideoFrame]:
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

    def set_input_container(self, video_decoder: decode_lib.VideoDecoder, stream_number: int = 0 ):
        self._decoder = video_decoder

        assert self._decoder is not None, "No decoder" 
        # TODO: fix: crash when returned formatcontext is deleted
        nb_streams = self._decoder.getFormatContext().nb_streams
        print(f"{nb_streams=}")
        video_streams = []
        for i in range(nb_streams):
            if self._decoder.getStream(i).codecpar.codec_type == decode_lib.AVMediaType.AVMEDIA_TYPE_VIDEO:
                video_streams.append(self._decoder.getStream(i))
        nb_videos = len(video_streams)
        print(f"Found {nb_videos} videos")
        # self._container.streams.video[0].thread_type = "FRAME"
        # self._container.streams.video[0].thread_count = 8
        ## self._container.streams.video[0].fast = True
        assert nb_videos>0, "No video stream found"
        self._video_stream = video_streams[stream_number]
        # TODO: get width and height
        # print(f"Video dimensions w={self.stream.width} x h={self.stream.height}")

        st = self.stream
        # Assume all frame have the same time base as the video stream
        # fps = self.stream.base_rate, base_rate vs average_rate
        print(f"FPS  avg:{st.avg_frame_rate} ")
        # --- Initialize several constants/parameters for this video
        self._framerate       = float(st.avg_frame_rate) # get the frame rate
        self._time_base       = float(st.time_base)
        self._frame_duration  = float(1/self._framerate)
        self._ticks_per_frame = int(self._frame_duration / self._time_base)
        self._duration        = float(st.duration * self._time_base)
        self._end_time        = float(self._duration-self._frame_duration)

        self._frame_buffer = VideoFrameBufferCpp(self._decoder)
        self._frame = None

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

    def set_time(self, time_pos : float, exact: bool =True):
        """ set time position in seconds """
        print(f"set_time {time_pos} , _end_time={self._end_time}")
        if self._decoder is None or self.frame_buffer is None:
            print("Video not initialized")
        else:
            frame_num = int(time_pos*self._framerate+0.5)
            current_frame_time = self.get_time()
            sec_frame   = int(current_frame_time * self._framerate)
            initial_pos = current_frame_time
            frame       = None
            # print(f"{sec_frame} -> {frame_num}")
            if frame_num == sec_frame:
                print(f"Current frame is at the requested position {frame_num} {sec_frame}")
                return
            # if we look for a frame slightly after, don't use seek()
            try:
                if frame_num<sec_frame or frame_num > sec_frame + 10:
                    # seek to that nearest timestamp
                    # is_playing = self.playing
                    # self.playing = False
                    # print(f"seek {int(time_pos/self._time_base+0.5)}")
                    seek_ok = self._decoder.seek(int(time_pos/self._time_base+0.5))  #, backward=True)
                    # It seems that the frame generator is not automatically updated
                    if time_pos==0:
                        self.frame_buffer.reset()
                    # get the next available frame
                    frame = self.frame_buffer.get_frame(timeout=10)
                    # get the proper key frame number of that timestamp
                    sec_frame = int(frame.pts * self._time_base * self._framerate)
                    initial_pos = float(frame.pts * self._time_base)
                    # self.playing = is_playing
                # print(f"missing {int((time_pos-sec_frame)/float(1000000*time_base))} ")
                if not frame or exact:
                    for _ in range(sec_frame, frame_num):
                        frame = self.frame_buffer.get_frame()
            except StopIteration as e:
                print(f"set_time(): Reached end of video stream")
                # Reset valid generator
                self.frame_buffer.reset()
                raise EndOfVideo() from e
            else:
                sec_frame = int(frame.pts * self._time_base * self._framerate)
                print(f"Frame at {initial_pos:0.3f}->"
                      f"{float(frame.pts * self._time_base):0.3f} requested {time_pos}")
                self._frame = frame

    def get_next_frame(self, timeout=6, verbose=False) -> bool:
        """ Obtain the next frame, usually while video is playing, 
            can raise EndofVideo exception """
        t = time.perf_counter()
        if not self.frame_buffer:
            print("Video Frame Buffer not created")
            return False
        try:
            self._frame = self._frame_buffer.get_frame(timeout=timeout)
        except (StopIteration, EndOfVideo) as e:
            print(f"get_next_frame(): Reached end of video stream: Exception {e}")
            # Reset valid generator
            self._frame_buffer.reset()
            raise EndOfVideo() from e
        else:
            if verbose:
                d = time.perf_counter()-t
                s = 'gen '
                s += 'key ' if self._frame.key_frame else ''
                s += f'{d:0.3f} ' if d>=0.001 else ''
                s += f'{self._frame.pict_type}'
                s += f' {self._frame.pts}'
                if s != 'gen P': 
                    print(s)
            return True
