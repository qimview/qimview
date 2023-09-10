
from .utils import get_time, deep_getsizeof
from .image_reader import image_reader
from .ThreadPool import ThreadPool
from .qt_imports import *

from collections import deque
# import multiprocessing as mp
from time import sleep
import os
import psutil


class BaseCache:
    def __init__(self, name=""):
        self.cache = deque()
        self.cache_list = []
        self.cache_size = 0
        # Max size in Mb
        self.max_cache_size = 2000
        self.verbose = False
        self.cache_unit = 1024*1024 # Megabyte
        self.thread_pool = ThreadPool()
        self.memory_bar = None
        self._name = name
        self._check_size_mutex = QtCore.QMutex()

    def set_memory_bar(self, progress_bar):
        self.memory_bar = progress_bar
        self.memory_bar.setRange(0, self.max_cache_size)
        self.memory_bar.setFormat("%v Mb")

    def reset(self):
        self.cache = deque()
        self.cache_list = []
        self.cache_size = 0

    def set_max_cache_size(self, size):
        self.max_cache_size = size
        self.check_size_limit()
        if self.memory_bar is not None:
            self.memory_bar.setRange(0, self.max_cache_size)

    def print_log(self, message):
        if self.verbose:
            print(message)

    def search(self, id):
        if id in self.cache_list:
            pos = self.cache_list.index(id)
            # print(f"pos {pos} len(cache) {len(self.cache)}")
            try:
                res = self.cache[pos][1]
            except Exception as e:
                print(f" Error in getting cache data: {e}")
                self.print_log(f" *** Cache {self._name}: search() cache_list {len(self.cache_list)} cache {len(self.cache)}")
                res = None
            return res
        return None

    def append(self, id, value, extra=None, check_size=True):
        """
        :param id: cache element identifier
        :param value: cache value, typically numpy array of the image
        :param extra: additional data in the cache
        :return:
        """
        # update cache
        self.print_log(f"added size {deep_getsizeof([id, value, extra], set())}")
        self.cache.append([id, value, extra])
        self.cache_list.append(id)
        self.print_log(f" *** Cache {self._name}: append() cache_list {len(self.cache_list)} cache {len(self.cache)}")
        if check_size:
            self.check_size_limit()

    def check_size_limit(self):
        self._check_size_mutex.lock()
        self.print_log(" *** Cache: check_size_limit()")
        cache_size = deep_getsizeof(self.cache, set())
        while cache_size >= self.max_cache_size * self.cache_unit:
            self.cache.popleft()
            self.cache_list.pop(0)
            self.print_log(" *** Cache: pop ")
            cache_size = deep_getsizeof(self.cache, set())
        self.print_log(f" *** Cache::append() {self._name} {cache_size/self.cache_unit} Mb; size {len(self.cache)}")
        self.cache_size = cache_size
        if self.memory_bar is not None:
            new_progress_value = int(self.cache_size/self.cache_unit+0.5)
            if new_progress_value != self.memory_bar.value():
                self.memory_bar.setValue(new_progress_value)
        self._check_size_mutex.unlock()



class FileCache(BaseCache): 
    def __init__(self):
        BaseCache.__init__(self, "FileCache")
        # let use 5% of total memory by default
        total_memory = psutil.virtual_memory().total / self.cache_unit
        self.max_cache_size = total_memory * 0.05
        self.last_progress = 0

    def has_file(self, filename):
        # is it too slow
        filename = os.path.abspath(filename)
        return filename in self.cache_list

    def get_file(self, filename, check_size=True):
        """
        :param filename:
        :param show_timing:
        :return: pair file_data, boolean (True is coming from cache)
        """
        print(f'get_file {filename}')
        start = get_time()
        # Get absolute normalized path
        filename = os.path.abspath(filename)
        # print(f"image cache get_image({filename})")
        file_data = self.search(filename)
        if file_data is not None:
            print(f'get_file {filename} found end')
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
                print(f'get_file {filename} read end')
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
        self.thread_pool.set_worker(self.thread_add_files, filenames)
        self.thread_pool.set_worker_callbacks(progress_cb=self.show_progress, finished_cb=self.on_finished)
        self.thread_pool.start_worker()
        self.print_log(f" FileCache.add_files() {self.add_results} took {int((get_time()-start)*1000+0.5)} ms;")



class ImageCache(BaseCache): 
    def __init__(self):
        BaseCache.__init__(self, "ImageCache")
        # let use 25% of total memory
        total_memory = psutil.virtual_memory().total / self.cache_unit
        self.max_cache_size = total_memory * 0.25

    def has_image(self, filename):
        # is it too slow
        filename = os.path.abspath(filename)
        return filename in self.cache_list

    def get_image(self, filename, read_size='full', verbose=False, use_RGB=True, image_transform=None,
                  check_size=True):
        """
        :param filename:
        :param show_timing:
        :return: pair image_data, boolean (True is coming from cache)
        """
        start = get_time()
        # Get absolute normalized path
        filename = os.path.abspath(filename)
        # print(f"image cache get_image({filename})")
        image_data = self.search(filename)
        if image_data is not None:
            return image_data, True
        else:
            try:
                image_data = image_reader.read(filename, None, read_size, use_RGB=use_RGB, verbose=verbose)
                if image_transform is not None:
                    image_data = image_transform(image_data)
                self.append(filename, image_data, check_size=check_size)
                self.print_log(" get_image after read_image took {0:0.3f} sec.".format(get_time() - start))
            except Exception as e:
                print("Failed to load image {0}: {1}".format(filename, e))
                return None, False
            else:
                return image_data, False

    def add_image(self, filename, read_size='full', verbose=False, use_RGB=True, image_transform=None,
                    progress_callback = None):
        """
        :param filename:
        :param show_timing:
        :return: pair image_data, boolean (True is coming from cache)
        """
        data, flag = self.get_image(filename, read_size, verbose, use_RGB, image_transform, check_size=False)
        return data is not None

    def image_added(self, filename):
        pass
        # print(f" *** Added image {os.path.basename(filename)} to cache finished")

    def add_result(self, r):
        # print(f"adding {r} to results ")
        self.add_results.append(r)

    def add_images(self, filenames, read_size='full', verbose=False, use_RGB=True, image_transform=None):
        start = get_time()
        self.add_results = []
        num_workers = 0
        for f in filenames:
            if f is not None and not self.has_image(f):
                # print(f" start worker with image {f}")
                self.thread_pool.set_worker(self.add_image, f, read_size, verbose, use_RGB, image_transform)
                self.thread_pool.set_worker_callbacks(finished_cb=lambda: self.image_added(f),
                                                      result_cb=self.add_result)
                self.thread_pool.start_worker()
                num_workers += 1
        wait_time = 0
        while len(self.add_results) != num_workers and wait_time < 2:
            # wait 5 ms before checking again
            sleep(0.002)
            wait_time += 0.002
        self.print_log(f" ImageCache.add_images() wait_time {wait_time} took {int((get_time()-start)*1000+0.5)} ms;")
        if num_workers>0:
            self.thread_pool.clear()
            self.check_size_limit()
        self.print_log(f" ImageCache.add_images() {num_workers}, {self.add_results} took {int((get_time()-start)*1000+0.5)} ms;")
