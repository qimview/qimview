
from collections import deque
from utils.utils import get_time, read_image, deep_getsizeof
# import multiprocessing as mp
from time import sleep
import os
from utils.ThreadPool import ThreadPool


class ImageCache:
    def __init__(self):
        # save images of last visited row
        self.cache = deque()
        self.image_list = []
        self.cache_size = 0
        # Max size in Mb
        self.max_cache_size = 2000
        self.verbose = True
        self.cache_unit = 1024*1024 # Megabyte
        self.thread_pool = ThreadPool()

    def reset(self):
        self.cache = deque()
        self.image_list = []
        self.cache_size = 0

    def print_log(self, message):
        if self.verbose:
            print(message)

    def search(self, id):
        if id in self.image_list:
            pos = self.image_list.index(id)
            print(f"pos {pos} len(cache) {len(self.cache)}")
            try:
                res = self.cache[pos][1]
            except Exception as e:
                print(f" Error in getting cache data: {e}")
                self.print_log(f" *** Cache: search() image_list {len(self.image_list)} cache {len(self.cache)}")
                res = None
            return res
        return None

    def has_image(self, filename):
        # is it too slow
        filename = os.path.abspath(filename)
        return filename in self.image_list

    def check_size_limit(self):
        print(" *** Cache: check_size_limit()")
        cache_size = deep_getsizeof(self.cache, set())
        while cache_size >= self.max_cache_size * self.cache_unit:
            self.cache.popleft()
            self.image_list.pop(0)
            self.print_log(" *** Cache: pop ")
            cache_size = deep_getsizeof(self.cache, set())
        self.print_log(f" *** Cache::append() {cache_size/self.cache_unit} Mb; size {len(self.cache)}")
        self.cache_size = cache_size

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
        self.image_list.append(id)
        self.print_log(f" *** Cache: append() image_list {len(self.image_list)} cache {len(self.cache)}")
        if check_size:
            self.check_size_limit()

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
                image_data = read_image(filename, read_size, verbose=verbose, use_RGB=use_RGB)
                if image_transform is not None:
                    image_data = image_transform(image_data)
                self.append(filename, image_data, check_size=check_size)
                self.print_log(" get_image after read_image took {0:0.3f} sec.".format(get_time() - start))
            except Exception as e:
                print("Failed to load image {0}: {1}".format(filename, e))
                return None, False
            else:
                return image_data, False

    def add_image(self, filename, read_size='full', verbose=False, use_RGB=True, image_transform=None):
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
        if num_workers>0:
            self.thread_pool.clear()
            self.check_size_limit()
        print(f" ImageCache.add_images() {num_workers}, {self.add_results} took {int((get_time()-start)*1000+0.5)} ms;")
