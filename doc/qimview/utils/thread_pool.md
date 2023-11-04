Module qimview.utils.thread_pool
================================

Classes
-------

`ThreadPool(*args, **kwargs)`
:   QThreadPool(self, parent: Optional[PySide6.QtCore.QObject] = None) -> None
    
    __init__(self, parent: Optional[PySide6.QtCore.QObject] = None) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * PySide6.QtCore.QThreadPool
    * PySide6.QtCore.QObject
    * Shiboken.Object

    ### Class variables

    `staticMetaObject`
    :

    ### Methods

    `set_worker(self, work_function, *args, **kwargs)`
    :

    `set_worker_callbacks(self, progress_cb=None, finished_cb=None, result_cb=None)`
    :

    `start_worker(self)`
    :

`Worker(fn, *args, **kwargs)`
:   Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    
    __init__(self) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * PySide6.QtCore.QRunnable
    * Shiboken.Object

    ### Methods

    `run(self)`
    :   Initialise the runner function with passed args, kwargs.

    `set_finished_callback(self, cb)`
    :

    `set_progress_callback(self, cb)`
    :

    `set_result_callback(self, cb)`
    :

`WorkerSignals(*args, **kwargs)`
:   Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        tuple (exctype, value, traceback.format_exc() )
    result
        object data returned from processing, anything
    progress
        int indicating % progress
    
    __init__(self, parent: Optional[PySide6.QtCore.QObject] = None) -> None
    
    Initialize self.  See help(type(self)) for accurate signature.

    ### Ancestors (in MRO)

    * PySide6.QtCore.QObject
    * Shiboken.Object

    ### Class variables

    `staticMetaObject`
    :

    ### Methods

    `error(...)`
    :

    `finished(...)`
    :

    `progress(...)`
    :

    `result(...)`
    :