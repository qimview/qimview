from typing import Optional, TypeVar, Protocol
from abc import abstractmethod
import queue
import time
import threading
import gc
from qimview.video_player.video_exceptions import EndOfVideo, TimeOut
from qimview.video_player.video_player_config import VideoConfig
from collections import deque

FRAMETYPE = TypeVar('FRAMETYPE')

class VideoFrameBufferBase(Protocol):
    """ 
        Base class for VideoFrameBuffer with either PyAV api or pybind11 bound ffmpeg api
        The decoding is managed in the inherited class
        This class uses a thread to store several successive videos frame in a queue
        that is available for the video player
    """

    _maxsize : int
    _queue : queue.Queue[FRAMETYPE]
    # Save frames before emptying the queue to faster manual display
    # The C++ decoder is currently limited to 8 frames in memory
    # TODO: improve this limitation and control it better
    _max_saved_frames : int = 8
    _saved_frames     : deque[FRAMETYPE] = deque(maxlen=_max_saved_frames)
    _running : bool = False
    _thread : Optional[threading.Thread] = None
    _end_of_video : bool = False
    _name : str = "VideoFrameBufferBase"

    def __protocol_init__(self, maxsize = VideoConfig.framebuffer_max_size):
        print(f" VideoFrameBuffer(maxsize = {maxsize})")
        self._maxsize : int = maxsize
        self._queue : queue.Queue[FRAMETYPE] = queue.Queue(maxsize=self._maxsize)
        # Save frames before emptying the queue to faster manual display
        self._saved_frames : deque[FRAMETYPE] = deque(maxlen=VideoFrameBufferBase._max_saved_frames)
        self._running : bool = False
        self._thread : Optional[threading.Thread] = None
        self._end_of_video : bool = False

    @property
    def running(self) -> bool:
        return self._running

    def add_saved_frame(self, frame: FRAMETYPE):
        # Add a frame only if not already saved
        # Warning: frames may point to the same C++ pointer, 
        # so we have to loop over _saved_frames which might point several times to the same C++ pointer
        for f in self._saved_frames:
            if f.pts == frame.pts:
                return
        # print(f"saving frame {frame.pts=}")
        self._saved_frames.append(frame)

    def reset_queue(self):
        if self._queue.qsize() > 0:
            print(f"VideoFrameBuffer.reset_queue() *** {self._queue.qsize()=} *** ")
            # If queue is empty, keep current saved frames
            if self._queue.qsize()>0:
                # print(f"saving {self._queue.qsize()} frames")
                for _ in range(self._queue.qsize()):
                    self.add_saved_frame(self._queue.get())
            self._queue = queue.Queue(maxsize=self._maxsize)

    @abstractmethod
    def decodeNextFrame(self) -> FRAMETYPE | None:
        return None

    @abstractmethod
    def resetDecoder(self) -> None:
        """ To implement in children classes """
        pass

    @abstractmethod
    def decoderOk(self) -> bool:
        """ To implement in children classes """
        return False

    def _worker(self):
        gc.collect()
        item  = None
        nb = 0
        total_time = 0
        while self._running:
            if item is None:
                # compute the item
                if self.decoderOk():
                    st = time.perf_counter()
                    item = self.decodeNextFrame()
                    if item is None:
                        self._running = False
                        self._end_of_video = True
                        self.resetDecoder()
                    else:
                        extract_time = time.perf_counter() - st
                        total_time += extract_time
                        nb += 1
                else:
                    print("Decoder not ok")
            if item is not None:
                try:
                    self._queue.put(item, timeout = 1/100)
                    # self._queue.put_nowait(item)
                    # print(f"{self._name}._worker() adding to queue: frame at {item.pts} / size {self.size()}")
                    # print(f"added item, qsize = {self._queue.qsize()}")
                    if nb%50 == 0:
                        print(f" {nb} Av extraction time: {total_time/nb:0.3f} queue: {self._queue.qsize()}")
                        total_time, nb = 0, 0
                except queue.Full:
                    # print("*", end="")
                    pass
                else:
                    item = None
            # if self._queue.qsize() != 0:
            #     time.sleep(1/1000)

    def set_max_size(self, m = 10):
        print(f" *** set_max_size {m}")
        self.pause_frames()
        self._queue = queue.Queue(maxsize=m)

    def reset(self):
        # print(f"{self._name}.reset()")
        self.pause_frames()
        self.reset_queue()
        # self.resetDecoder()
        self._end_of_video = False

    def size(self) -> int:
        return self._queue.qsize()

    def get_frame(self, timeout=0.5, save:bool = True) -> FRAMETYPE:
        """ Get the next video frame from the queue or from the decoder directly

        Args:
            timeout (int, optional): _description_. Defaults to 6.
            save (bool, optional): if true, save the frame if not thread is not running (not in playing mode)

        Raises:
            EndOfVideo: _description_
            TimeOut: maximum time in seconds to wait for the queue to return a frame
            EndOfVideo: _description_

        Returns:
            VideoFrame: _description_
        """
        # print(f"{self._name}.get_frame() ", end='; ')
        if self._running or self._queue.qsize()>0:
            try:
                res = self._queue.get(block=True, timeout=timeout)
                # print(f" {res.pts=} / {self.size()}")
            except queue.Empty as e:
                if self._end_of_video:
                    raise EndOfVideo() from e
                else:
                    print(f"Failed to get frame within {timeout} second")
                    raise TimeOut(f"Timeout of {timeout} seconds reached while getting a frame") from e
        elif not self._running and self._end_of_video:
            raise EndOfVideo()
        else:
            res = self.get_nothread(save=save)
        return res

    def get_nothread(self, save:bool = True) -> Optional[FRAMETYPE]:
        # print("VideoFrameBufferBase.get_nothread()")
        if self.decoderOk():
            res = self.decodeNextFrame()
            if res is None:
                print("get_nothread(): Failed to get next frame")
                self._running = False
                self.resetDecoder()
                return None # raise EndOfVideo()
            else:
                if save:
                    # Add to saved frames to speed-up moving around
                    self.add_saved_frame(res)
                return res
        else:
            return None

    def pause_frames(self):
        """
            Stop the thread that generates frames
        """        
        # print(f"VideoFrameBufferBase.pause_frames() {self._thread}")
        self._running = False
        if self._thread:
            self._thread.join()
        self._thread = None
        # self.resetDecoder()

    def start_thread(self):
        print(f"start_thread {self._thread} running {self._running}")
        if not self.decoderOk():
            print(" *** resetting frame generator ***")
            self.resetDecoder()
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._running = True
            self._thread.start()
            duration = 0
            # Empty saved frames to avoid using too much memory
            self._saved_frames.clear()
            # Fill the queue
            while self._queue.qsize()<self._maxsize and self._running:
                time.sleep(0.1)
                duration += 0.1
            print(f"queue size after {duration} sec {self._queue.qsize()}")
        else:
            print("Cannot start already running thread")
