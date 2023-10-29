
from qimview.utils.utils import get_time, deep_getsizeof
# from qimview.utils.qt_imports import *
from .basecache import BaseCache
import os
import psutil
from typing import Optional, Tuple

class FileCache(BaseCache): 
    def __init__(self):
        BaseCache.__init__(self, "FileCache")
        # let use 5% of total memory by default
        total_memory = psutil.virtual_memory().total / self.cache_unit
        self.max_cache_size = int(total_memory * 0.05)
        self.last_progress = 0

    def has_file(self, filename):
        # is it too slow
        filename = os.path.abspath(filename)
        return filename in self.cache_list

    def get_file(self, filename: str, check_size: bool = True) -> Tuple[Optional[bytes], bool]:
        """
        :param filename:
        :param show_timing:
        :return: pair file_data, boolean (True is coming from cache)
        """
        # print(f'get_file {filename}')
        start = get_time()
        # Get absolute normalized path
        filename = os.path.abspath(filename)
        # print(f"image cache get_image({filename})")
        file_data = self.search(filename)
        if file_data is not None:
            # print(f'get_file {filename} found end')
            return file_data, True
        else:
            try:
                # read file as binary data
                self.print_log(" FileCache::get_file() before read() {0:0.3f} sec.".format(get_time() - start))
                with open(filename, 'rb') as f:
                    file_data = f.read()
                self.print_log(" FileCache::get_file() after read() {0:0.3f} sec.".format(get_time() - start))
                self.append(filename, file_data, check_size=check_size)
                self.print_log("  FileCache::get_file() after append took {0:0.3f} sec.".format(get_time() - start))
            except Exception as e:
                print("Failed to load image {0}: {1}".format(filename, e))
                return None, False
            else:
                # print(f'get_file {filename} read end')
                return file_data, False

    def thread_add_files(self, filenames, progress_callback = None):
        """
        :param filename:
        :param show_timing:
        :return: pair image_data, boolean (True is coming from cache)
        """
        nb_files = len(filenames)
        nb_new_files = 0
        for n, f in enumerate(filenames):
            if nb_new_files>80:
                break
            if f is not None and not self.has_file(f):
                data, flag = self.get_file(f, check_size=False)
                # slow down to simulate network drive
                # sleep(.500)
                if not flag:
                    nb_new_files += 1
            if progress_callback is not None:
                progress_callback.emit(int(n*100/nb_files+0.5))
            # if n%10==0: 
            #     self.check_size_limit()

    def file_added(self, filename):
        pass
        # print(f" *** Added image {os.path.basename(filename)} to cache finished")

    def add_result(self, r):
        # print(f"adding {r} to results ")
        pass

    def show_progress(self, val):
        if val!=self.last_progress:
            print(f"add files done {val} %")
            self.check_size_limit()
            self.last_progress = val

    def on_finished(self):
        self.check_size_limit()

    def add_files(self, filenames):
        self.thread_pool.clear()
        start = get_time()
        self.add_results = []
        # print(f" start worker with image {f}")
        # This part may be causing issues
        use_threads = False
        if use_threads:
            self.thread_pool.set_worker(self.thread_add_files, filenames)
            self.thread_pool.set_worker_callbacks(progress_cb=self.show_progress, finished_cb=self.on_finished)
            self.thread_pool.start_worker()
        else:
            self.thread_add_files(filenames, progress_cb=self.show_progress)
        self.print_log(f" FileCache.add_files() {self.add_results} took {int((get_time()-start)*1000+0.5)} ms;")
