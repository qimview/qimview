# example from https://www.learnpyqt.com/tutorials/multithreading-pyqt-applications-qthreadpool/

from .qt_imports import *


class WorkerSignals(QtCore.QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        tuple (exctype, value, traceback.format_exc() )
    result
        object data returned from processing, anything
    progress
        int indicating % progress
    '''
    finished = Signal()
    error    = Signal(tuple)
    result   = Signal(object)
    progress = Signal(int)


class Worker(QtCore.QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.result_cb = None

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    def set_progress_callback(self, cb):
        if cb is not None:
            self.signals.progress.connect(cb)

    def set_finished_callback(self, cb):
        if cb is not None:
            self.signals.finished.connect(cb)

    def set_result_callback(self, cb):
        # self.signals.result.connect(cb)
        if cb is not None:
            self.result_cb = cb

    @Slot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """
        # Retrieve args/kwargs here; and fire processing using them
        try:
            # print("run")
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            print(f"Exception catched while processing Worker task: {e}")
            import sys
            import traceback
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # print(" ***** emit result signal ")
            if self.result_cb is not None:
                self.result_cb(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class ThreadPool(QtCore.QThreadPool):
    """ Can start several runnables in threads """
    def __init__(self):
        super().__init__()
        # print(f"Multithreading with maximum {self.maxThreadCount()} threads" )
        self._worker : Worker

    def set_worker(self, work_function, *args, **kwargs):
        """ Define a new worker to process work_function and its arguments """
        self._worker = Worker(work_function,  *args, **kwargs)  # Any other args, kwargs are passed to the run function

    def set_autodelete(self, d: bool):
        self._worker.setAutoDelete(d)

    def set_worker_callbacks(self, progress_cb=None, finished_cb=None, result_cb=None):
        """ Define several possible callbacks """
        self._worker.set_progress_callback(progress_cb)
        self._worker.set_finished_callback(finished_cb)
        self._worker.set_result_callback(result_cb)

    def start_worker(self):
        """ Starts the worker in a thread """
        self.start(self._worker)

