
from typing import Optional
from av import container, VideoFrame
from av.container import streams
from qimview.video_player.video_frame_buffer        import VideoFrameBuffer
from qimview.video_player.video_frame_provider_base import VideoFrameProviderBase

class VideoFrameProvider(VideoFrameProviderBase[VideoFrame,  container.InputContainer]):
    def __protocol_init__(self):
        super().__protocol_init__()
        self._name            : str                                = 'VideoFrameProvider'

    def get_video_streams(self):
        return self._container.streams.video
    
    def set_stream_threads(self, stream):
        stream.thread_type = "FRAME"
        stream.thread_count = 8

    def CreateFrameBuffer(self, video_stream_number: int):
        assert self._container is not None
        self._frame_buffer = VideoFrameBuffer(self._container, maxsize=10, stream_number = video_stream_number)

    def logStreamInfo(self):
        st = self.stream
        print(f"Video dimensions w={st.width} x h={st.height}")
        # Assume all frame have the same time base as the video stream
        # fps = self.stream.base_rate, base_rate vs average_rate
        print(f"FPS base:{st.base_rate} avg:{st.average_rate} guessed:{st.guessed_rate}")

    def copyStreamInfo(self):
        st = self.stream
        # --- Initialize several constants/parameters for this video
        self._framerate       = float(st.average_rate) # get the frame rate
        self._time_base       = float(st.time_base)
        self._frame_duration  = float(1/self._framerate)
        self._ticks_per_frame = int(self._frame_duration / self._time_base)
        if st.duration:
            self._duration    = float(st.duration * self._time_base)
            self._end_time    = float(self._duration-self._frame_duration)
        else:
            self._duration    = 0
            self._end_time    = 0

    @property
    def frame_duration(self) -> float:
        return self._frame_duration

    def seek_position(self, time_pos: float) -> bool:
        self._container.seek(int(time_pos*1000000), backward=True)
        return True
