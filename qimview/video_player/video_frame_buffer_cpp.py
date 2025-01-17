from typing import Optional, Iterator
import queue
import time
import threading

import os
from qimview.video_player.video_exceptions import EndOfVideo, TimeOut
from qimview.video_player.video_frame_buffer_base import VideoFrameBufferBase

ffmpeg_path = os.path.join(os.environ.get('FFMPEG_ROOT', ''),'bin')
if os.name == 'nt' and os.path.isdir(ffmpeg_path):
    os.add_dll_directory(ffmpeg_path)

import decode_video_py as decode_lib


class VideoFrameBufferCpp(VideoFrameBufferBase):
    """ This class uses a thread to store several successive videos frame in a queue
    that is available for the video player
    """
    def __init__(self, decoder: decode_lib.VideoDecoder, maxsize = 6):
        super().__protocol_init__(maxsize)
        self._decoder : decode_lib.VideoDecoder = decoder

    def decodeNextFrame(self):
        res = self._decoder.nextFrame(convert=True)
        if res !=0: return None
        f = self._decoder.getFrame()
        return f

    def resetDecoder(self) -> None:
        self._decoder.seek(0)

    def set_decoder(self, d):
        self.pause_frames()
        self._decoder = d

    def decoderOk(self) -> bool:
        return True

