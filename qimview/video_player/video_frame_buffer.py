from typing import Optional, Iterator
import av
from av import container
from av.frame import Frame
from qimview.video_player.video_exceptions import EndOfVideo, TimeOut
from qimview.video_player.video_frame_buffer_base import VideoFrameBufferBase


class VideoFrameBuffer(VideoFrameBufferBase):
    """ This class uses a thread to store several successive videos frame in a queue
    that is available for the video player
    """
    def __init__(self, container: container.InputContainer, maxsize = 20, stream_number = 0):
        super().__protocol_init__(maxsize)
        self._stream_number : int = stream_number
        self._container : container.InputContainer = container
        self._frame_generator : Optional[Iterator[Frame]] = self._container.decode(video=self._stream_number)

    def decodeNextFrame(self):
        try:
            item = next(self._frame_generator)
        except (StopIteration, av.EOFError):
            return None
        return item

    def resetDecoder(self) -> None:
        self._frame_generator = self._container.decode(video=self._stream_number)

    def set_generator(self, g):
        self._frame_generator = g

    def decoderOk(self) -> bool:
        return self._frame_generator is not None

    def set_container(self, c):
        """ not used?

        Args:
            c (_type_): _description_
        """
        self.pause_frames()
        self._container = c
        self._frame_generator = self._container.decode(video=self._stream_number)

