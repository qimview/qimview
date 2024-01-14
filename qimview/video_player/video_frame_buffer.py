from typing import Optional, Iterator
import queue
import time
import threading
from av import container, VideoFrame
from av.frame import Frame


class VideoFrameBuffer:
    def __init__(self, container: container.InputContainer, maxsize = 10):
        print(f" VideoFrameBuffer(maxsize = {maxsize})")
        self._maxsize : int = maxsize
        self._queue = queue.Queue(maxsize=self._maxsize)
        self._running : bool = False
        self._container : container.InputContainer = container
        self._frame_generator : Optional[Iterator[Frame]] = self._container.decode(video=0)
        self._thread : Optional[threading.Thread] = None

    def reset_queue(self):
        self._queue = queue.Queue(maxsize=self._maxsize)

    def _worker(self):
        item  = None
        while self._running:
            if item is None:
                # compute the item
                if self._frame_generator:
                    try:
                        item = next(self._frame_generator)
                    except StopIteration as e:
                        self._running = False
                        self._frame_generator = None
                        raise StopIteration from e
            if item is not None:
                try:
                    self._queue.put_nowait(item)
                    # print(f"added item, qsize = {self._queue.qsize()}")
                except queue.Full:
                    print("*", end="")
                    pass
                else:
                    item = None
            time.sleep(1/1000)

    def set_generator(self, g):
        self._frame_generator = g

    def set_container(self, c):
        self.terminate()
        self._container = c
        self._frame_generator = self._container.decode(video=0)

    def set_max_size(self, m = 10):
        print(f" *** set_max_size {m}")
        self.terminate()
        self._queue = queue.Queue(maxsize=m)

    def reset(self):
        self.terminate()
        self.reset_queue()
        self._frame_generator = self._container.decode(video=0)

    def get(self) -> VideoFrame:
        if self._running:
            res = self._queue.get()
        else:
            res = self.get_nothread()
        return res

    def get_nothread(self) -> Optional[VideoFrame]:
        if self._frame_generator:
            try:
                res = next(self._frame_generator)
                return res
            except StopIteration as e:
                self._running = False
                self._frame_generator = None
                raise StopIteration from e
        else:
            return None

    def terminate(self):
        self._running = False
        if self._thread:
            self._thread.join()
        self._thread = None

    def start_thread(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._running = True
            self._thread.start()
        else:
            print("Cannot start already running thread")

