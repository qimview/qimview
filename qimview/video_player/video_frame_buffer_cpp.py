from qimview.video_player.video_player_config import VideoConfig

import os
import sys
from qimview.video_player.video_frame_buffer_base import VideoFrameBufferBase

ffmpeg_path = os.path.join(os.environ.get('FFMPEG_ROOT', ''),'bin')
if os.name == 'nt' and os.path.isdir(ffmpeg_path):
    os.add_dll_directory(ffmpeg_path)

import decode_video_py as decode_lib

# On macOS, keep VideoToolbox frames on the GPU (convert=False) so that
# display_frame_YUV420() can bind them as IOSurface textures without a CPU copy.
_HW_ZERO_COPY = sys.platform == 'darwin'


class VideoFrameBufferCpp(VideoFrameBufferBase):
    """ This class uses a thread to store several successive videos frame in a queue
    that is available for the video player
    """
    def __init__(self, decoder: decode_lib.VideoDecoder, maxsize = VideoConfig.framebuffer_max_size):
        super().__protocol_init__(maxsize)
        self._decoder : decode_lib.VideoDecoder = decoder

    def decodeNextFrame(self):
        # On macOS: avoid GPU→CPU transfer so IOSurface interop can be used.
        # On other platforms (or when not using hw decode): convert to CPU memory.
        convert = not _HW_ZERO_COPY
        res = self._decoder.nextFrame(convert=convert)
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

