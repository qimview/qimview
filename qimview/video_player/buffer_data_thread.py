from abc import abstractmethod
import queue


class BufferDataThread:
    """ 
        This class will copy data to opengl mapped buffer in a thread
    """

    _name : str = "BufferDataThread"

    def __init__(self, maxsize = 2):
        # TO IMPLEMENT
        pass

    @property
    def running(self) -> bool:
        return self._running

    def reset_queue(self):
        self._queue = queue.Queue(maxsize=self._maxsize)

    def _worker(self):
        # TO IMPLEMENT
        pass

    def reset(self):
        # TO IMPLEMENT
        pass

    def start_thread(self):
        # TO IMPLEMENT
        pass
