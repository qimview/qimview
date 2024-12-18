from typing import Optional, Iterator
import queue
import time
import threading
import av
from av import container, VideoFrame
from av.frame import Frame
import gc
from qimview.video_player.video_exceptions import EndOfVideo, TimeOut


class VideoFrameBuffer:
    """ This class uses a thread to store several successive videos frame in a queue
    that is available for the video player
    """
    def __init__(self, container: container.InputContainer, maxsize = 20, stream_number = 0):
        print(f" VideoFrameBuffer(maxsize = {maxsize})")
        self._maxsize : int = maxsize
        self._queue = queue.Queue(maxsize=self._maxsize)
        # Save frames before emptying the queue to faster manual display
        self._saved_frames = []
        self._running : bool = False
        self._container : container.InputContainer = container
        self._stream_number : int = stream_number
        self._frame_generator : Optional[Iterator[Frame]] = self._container.decode(video=self._stream_number)
        self._thread : Optional[threading.Thread] = None
        self._end_of_video : bool = False

    @property
    def running(self) -> bool:
        return self._running

    def reset_queue(self):
        # print("VideoFrameBuffer.reset_queue()")
        # If queue is empty, keep current saved frames
        if self._queue.qsize()>0:
            # print(f"saving {self._queue.qsize()} frames")
            self._saved_frames = [ self._queue.get() for _ in range(self._queue.qsize())]
        self._queue = queue.Queue(maxsize=self._maxsize)

    def _worker(self):
        gc.collect()
        item  = None
        nb = 0
        total_time = 0
        while self._running:
            if item is None:
                # compute the item
                if self._frame_generator is not None:
                    try:
                        st = time.perf_counter()
                        item = next(self._frame_generator)
                        extract_time = time.perf_counter() - st
                        total_time += extract_time
                        nb += 1
                    except (StopIteration, av.EOFError):
                        self._running = False
                        self._end_of_video = True
                        # Reset generator ?
                        self._frame_generator = self._container.decode(video=self._stream_number)
                        # raise StopIteration from e
                else:
                    print("Missing frame generator")
            if item is not None:
                try:
                    self._queue.put_nowait(item)
                    # print(f"added item, qsize = {self._queue.qsize()}")
                    if nb%30 == 0:
                        print(f" {nb} Av extraction time: {total_time/nb:0.3f} queue: {self._queue.qsize()}")
                        total_time = 0
                        nb = 0
                except queue.Full:
                    # print("*", end="")
                    pass
                else:
                    item = None
            if self._queue.qsize() != 0:
                time.sleep(1/2000)

    def set_generator(self, g):
        self._frame_generator = g

    def set_container(self, c):
        """ not used?

        Args:
            c (_type_): _description_
        """
        self.terminate()
        self._container = c
        self._frame_generator = self._container.decode(video=self._stream_number)

    def set_max_size(self, m = 10):
        print(f" *** set_max_size {m}")
        self.terminate()
        self._queue = queue.Queue(maxsize=m)

    def reset(self):
        # print("VideoFrameBuffer.reset()")
        self.terminate()
        self.reset_queue()
        self._frame_generator = self._container.decode(video=self._stream_number)
        self._end_of_video = False

    def size(self) -> int:
        return self._queue.qsize()

    def get_frame(self, timeout=6) -> VideoFrame | None:
        """ Get the next video frame from the queue or from the decoder directly

        Args:
            timeout (int, optional): _description_. Defaults to 6.

        Raises:
            EndOfVideo: _description_
            TimeOut: _description_
            EndOfVideo: _description_

        Returns:
            VideoFrame: _description_
        """
        if self._running or self._queue.qsize()>0:
            try:
                res = self._queue.get(block=True, timeout=timeout)
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

    def get_nothread(self) -> Optional[VideoFrame]:
        if self._frame_generator:
            try:
                res = next(self._frame_generator)
                return res
            except (StopIteration, av.EOFError) as e:
                print("get_nothread(): Reached end of video")
                self._running = False
                self._frame_generator = None
                raise EndOfVideo() from e
        else:
            return None

    def terminate(self):
        print(f"terminate {self._thread}")
        self._running = False
        if self._thread:
            self._thread.join()
        self._thread = None
        self._frame_generator = None

    def start_thread(self):
        print(f"start_thread {self._thread} running {self._running} generator {self._frame_generator}")
        if self._frame_generator is None:
            print(" *** resetting frame generator ***")
            self._frame_generator = self._container.decode(video=self._stream_number)
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._running = True
            self._thread.start()
            duration = 0
            # print(f"{self._queue.qsize()=}")
            # Empty saved frames to avoid using too much memory
            self._saved_frames = []
            while self._queue.qsize()<self._maxsize and self._running:
                # print(f"{self._queue.qsize()=}")
                time.sleep(0.1)
                duration += 0.1
            print(f"queue size after {duration} sec {self._queue.qsize()}")
        else:
            print("Cannot start already running thread")

