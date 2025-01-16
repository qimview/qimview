from typing import Optional, TypeVar
from abc import abstractmethod
import queue
import time
import threading
import gc
from qimview.video_player.video_exceptions import EndOfVideo, TimeOut


class BufferDataThread:
    """ 
        This class will copy data to opengl mapped buffer in a thread
    """

    _maxsize : int
    _queue : queue.Queue
    # Save frames before emptying the queue to faster manual display
    _saved_frames = []
    _running : bool = False
    _thread : Optional[threading.Thread] = None
    _end_of_video : bool = False
    _name : str = "BufferDataThread"

    def __init__(self, maxsize = 2):
        print(f" {self._name}(maxsize = {maxsize})")
        self._maxsize : int = maxsize
        self._queue : queue.Queue = queue.Queue(maxsize=self._maxsize)
        self._running : bool = False
        self._thread : Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._running

    def reset_queue(self):
        self._queue = queue.Queue(maxsize=self._maxsize)

    def _worker(self):
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

    def get_frame(self, timeout=0.5) -> FRAMETYPE:
        """ Get the next video frame from the queue or from the decoder directly

        Args:
            timeout (int, optional): _description_. Defaults to 6.

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
            res = self.get_nothread()
        return res

    def get_nothread(self) -> Optional[FRAMETYPE]:
        # print("VideoFrameBufferBase.get_nothread()")
        if self.decoderOk():
            res = self.decodeNextFrame()
            if res is None:
                print("get_nothread(): Reached end of video")
                self._running = False
                self.resetDecoder()
                raise EndOfVideo()
            else:
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
        print(f" {self._name}: start_thread {self._thread} running {self._running}")
        # if not self.decoderOk():
        #     print(" *** resetting frame generator ***")
        #     self.resetDecoder()
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._running = True
            self._thread.start()
            duration = 0
            # Empty saved frames to avoid using too much memory
            self._saved_frames = []
            # Fill the queue
            while self._queue.qsize()<self._maxsize and self._running:
                time.sleep(0.1)
                duration += 0.1
            print(f"queue size after {duration} sec {self._queue.qsize()}")
        else:
            print("Cannot start already running thread")
