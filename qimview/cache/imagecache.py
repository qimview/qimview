
from qimview.utils.utils import get_time
from qimview.image_readers import gb_image_reader
from .basecache import BaseCache
import os
import psutil
from typing import TYPE_CHECKING
from qimview.utils.ViewerImage import ViewerImage

class ImageCache(BaseCache[str, ViewerImage, float]): 
    """
        Save output bytes from read() function into a cache indexed by the filename
        inherits from BaseCache, with
            id as string: input filename
            ViewerImage: image object
            mtime: modification time as float from osp.getmtime(filename)
        If a file is in the cache but its modification time on disk is more recent,
        we can enable an automatic reload

    Args:
        BaseCache (_type_): _description_
    """    
    def __init__(self):
        BaseCache.__init__(self, "ImageCache")
        # let use 25% of total memory
        total_memory = psutil.virtual_memory().total / self.cache_unit
        self.max_cache_size = int(total_memory * 0.25)
        self.verbose : bool = False

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
        mtime = os.path.getmtime(filename)

        if image_data is not None:
            if image_data[2] >= mtime:
                return image_data[1], True
            else:
                # Remove outdated cache element
                self.cache.remove(image_data)
                self.cache_list.remove(filename)
        
        image = gb_image_reader.read(filename, None, read_size, use_RGB=use_RGB, verbose=verbose,
                                            check_filecache_size=check_size)
        if image is not None:
            if image_transform is not None:
                image = image_transform(image)
            self.append(filename, image, extra=mtime, check_size=check_size)
            self._print_log(" get_image after read_image took {0:0.3f} sec.".format(get_time() - start))
        else:
            print(f"Failed to load image {filename}")
            return None, False
        return image, False

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
            #sleep(0.002)
            self.thread_pool.waitForDone(2)
            wait_time += 0.002
        self._print_log(f" ImageCache.add_images() wait_time {wait_time} took {int((get_time()-start)*1000+0.5)} ms;")
        if num_workers>0:
            self.thread_pool.clear()
            # There could be unfinished thread here?
            # It seems that the progress bar must be updated outside the threads
            self.check_size_limit(update_progress=True)
            if gb_image_reader.file_cache is not None:
                gb_image_reader.file_cache.check_size_limit(update_progress=True)
        assert len(self.add_results) == num_workers, "Workers left"
        self._print_log(f" ImageCache.add_images() {num_workers}, {self.add_results} took {int((get_time()-start)*1000+0.5)} ms;")
