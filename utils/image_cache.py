
from collections import deque
from utils.utils import get_time, read_image, deep_getsizeof


class ImageCache:
    def __init__(self):
        # save images of last visited row
        self.cache = deque([])
        self.cache_size = 0
        # Max size in Mb
        self.max_cache_size = 2000
        self.verbose = True
        self.cache_unit = 1024*1024 # Megabyte

    def reset(self):
        self.cache = deque([])

    def print_log(self, message):
        if self.verbose:
            print(message)

    def search(self, id):
        for c in self.cache:
            if id == c[0]:
                return c[1]
        return None

    def append(self, id, value, extra=None):
        """
        :param id: cache element identifier
        :param value: cache value, typically numpy array of the image
        :param extra: additional data in the cache
        :return:
        """
        # update cache
        print("added size ", deep_getsizeof([id, value, extra], set()))
        self.cache.append([id, value, extra])
        cache_size = deep_getsizeof(self.cache, set())
        while cache_size >= self.max_cache_size * self.cache_unit:
            self.cache.popleft()
            print(" *** cache pop ")
            cache_size = deep_getsizeof(self.cache, set())
        self.print_log(f"Cache::append() {cache_size/self.cache_unit} Mb; size {len(self.cache)}")
        self.cache_size = cache_size
