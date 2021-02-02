
import sys
import time
import numpy as np
from collections import deque
import os


def get_time():
	is_windows = sys.platform.startswith('win')
	if is_windows:
		if hasattr(time, 'clock'):
			return time.clock()
		else:
			return time.perf_counter()
	else:
		return time.time()


def get_size(obj, seen=None):
	"""
	Recursively finds size of objects
	"""
	size = sys.getsizeof(obj)
	if seen is None:
		seen = set()
	obj_id = id(obj)
	if obj_id in seen:
		return 0
	# Important mark as seen *before* entering recursion to gracefully handle
	# self-referential objects
	seen.add(obj_id)
	if isinstance(obj, dict):
		size += sum([get_size(v, seen) for v in obj.values()])
		size += sum([get_size(k, seen) for k in obj.keys()])
	elif hasattr(obj, '__dict__'):
		size += get_size(obj.__dict__, seen)
	elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
		size += sum([get_size(i, seen) for i in obj])
	return size


# function found at https://code.tutsplus.com/tutorials/understand-how-much-memory-your-python-objects-use--cms-25609
def deep_getsizeof(o, ids):
	""" Find the memory footprint of a Python object
	This is a recursive function that drills down a Python object graph
	like a dictionary holding nested dictionaries with lists of lists
	and tuples and sets.

	The sys.getsizeof function does a shallow size of only. It counts each
	object inside a container as pointer only regardless of how big it
	really is.

	:param o: the object
	:param ids:
	:return:
	"""
	d = deep_getsizeof

	if id(o) in ids:
		return 0

	# print(f' deep_getsizeof() type {type(o)} getsizeof {sys.getsizeof(o)}')
	# deal with pyqtgraph ImageItem size
	try:
		image = o.image
		if isinstance(image, np.ndarray):
			r = image.nbytes
	except Exception as e:
		if isinstance(o, np.ndarray):
			r = o.nbytes
			# print(f' nbytes {o.nbytes}')
		else:
			r = sys.getsizeof(o)

	ids.add(id(o))
	if isinstance(o, str) or isinstance(0, str):
		return r
	if isinstance(o, dict):
		return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())

	if isinstance(o,  (tuple, list, set, deque)):
		# print "*"
		return r + sum(d(x, ids) for x in o)
	return r
