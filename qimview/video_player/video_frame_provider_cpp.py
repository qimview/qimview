
import os
from qimview.video_player.video_frame_buffer_cpp import VideoFrameBufferCpp

ffmpeg_path = os.path.join(os.environ.get('FFMPEG_ROOT', ''),'bin')
if os.name == 'nt' and os.path.isdir(ffmpeg_path):
    os.add_dll_directory(ffmpeg_path)

import decode_video_py as decode_lib

from qimview.video_player.video_frame_provider_base import VideoFrameProviderBase

class VideoFrameProviderCpp(VideoFrameProviderBase[decode_lib.Frame, decode_lib.VideoDecoder]):
    def __init__(self):
        super().__init__()
        self._name            : str                                = 'VideoFrameProviderCpp'

    @property
    def frame_duration(self) -> float:
        return self._frame_duration

    def get_video_streams(self):
        nb_streams = self._container.getFormatContext().nb_streams
        print(f"{nb_streams=}")
        video_streams = []
        for i in range(nb_streams):
            if self._container.getStream(i).codecpar.codec_type == decode_lib.AVMediaType.AVMEDIA_TYPE_VIDEO:
                video_streams.append(self._container.getStream(i))
        return video_streams

    def set_stream_threads(self, stream):
        pass

    def CreateFrameBuffer(self, video_stream_number: int):
        assert self._container is not None
        self._frame_buffer = VideoFrameBufferCpp(self._container)

    def logStreamInfo(self):
        st = self.stream
        print(f"FPS  avg:{st.avg_frame_rate} ")

    def copyStreamInfo(self):
        st = self.stream
        # Assume all frame have the same time base as the video stream
        # fps = self.stream.base_rate, base_rate vs average_rate
        # --- Initialize several constants/parameters for this video
        self._framerate       = float(st.avg_frame_rate) # get the frame rate
        self._time_base       = float(st.time_base)
        self._frame_duration  = float(1/self._framerate)
        self._ticks_per_frame = int(self._frame_duration / self._time_base)
        self._duration        = float(st.duration * self._time_base)
        self._end_time        = float(self._duration-self._frame_duration)
        
    def seek_position(self, time_pos: float) -> bool:
        seek_ok = self._container.seek(int(time_pos/self._time_base+0.5))  #, backward=True)
        print(f"{self._name}.seek_position({time_pos=}) --> {seek_ok=}")
        return seek_ok
