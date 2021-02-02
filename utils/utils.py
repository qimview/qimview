
from .ViewerImage import *

import sys
import time
import cv2
import numpy as np
from collections import deque
import rawpy
import math
import os

# check for module, needs python >= 3.4
# https://stackoverflow.com/questions/14050281/how-to-check-if-a-python-module-exists-without-importing-it
import importlib

has_simplejpeg = importlib.util.find_spec("simplejpeg") is not None
has_turbojpeg = importlib.util.find_spec("turbojpeg") is not None
print(f" ***  has_simplejpeg {has_simplejpeg}  has_turbojpeg {has_turbojpeg}   ***")

if has_simplejpeg:
	import simplejpeg
if has_turbojpeg:
	from turbojpeg import TurboJPEG, TJPF_RGB, TJPF_BGR, TJFLAG_FASTDCT

USE_LIBRAW = 1

RAW_FORMAT_EXTENSIONS = {
	".ARW": USE_LIBRAW,  # Sony RAW format
	".GPR": USE_LIBRAW,  # GoPro RAW, other ?
}

def get_time():
	is_windows = sys.platform.startswith('win')
	if is_windows:
		if hasattr(time, 'clock'):
			return time.clock()
		else:
			return time.perf_counter()
	else:
		return time.time()

def read_libraw(image_filename, read_size='full'):
	raw = rawpy.imread(image_filename)
	height, width = raw.raw_image.shape
	print(f"height, width {(height, width)}")
	# Transform Bayer image
	im1 = np.empty((height >> 1, width >> 1, 4), dtype=raw.raw_image.dtype)
	bayer_desc = raw.color_desc.decode('utf-8')
	bayer_pattern = raw.raw_pattern
	# not perfect, we may switch Gr and Gb
	bayer_desc = bayer_desc.replace('G','g',1)
	# Detect the green channels?
	channel_pos = {'R':0,'g':1,'G':2,'B':3}
	for i in range(2):
		for j in range(2):
			print("str(bayer_desc[bayer_pattern[i,j]]) {}".format(str(bayer_desc[bayer_pattern[i,j]])))
			pos = channel_pos[str(bayer_desc[bayer_pattern[i,j]])]
			im1[:, :, pos] = raw.raw_image[i::2, j::2]
	prec = math.ceil(math.log2(raw.white_level+1))
	viewer_image = ViewerImage(im1, precision=prec, downscale=1, channels=CH_RGGB)
	print("viewer_image channels {}".format(viewer_image.channels))
	return viewer_image


def read_jpeg_turbojpeg(image_filename, read_size='full', use_RGB=True, verbose=False):
	try:
		if verbose:
			start = get_time()
		# using default library installation
		jpeg = TurboJPEG()
		# decoding input.jpg to BGR array
		in_file = open(image_filename, 'rb')
		scale =  {'full': (1,1),
				   '1/2': (1,2),
				   '1/4': (1,4),
				   '1/8': (1,8),
			   }
		start1 = get_time()
		im_header = jpeg.decode_header(in_file.read())
		print(f" header {im_header} {int(get_time() - start1) * 1000} ms")
		in_file.seek(0)
		pixel_format = TJPF_RGB if use_RGB else TJPF_BGR
		flags = TJFLAG_FASTDCT
		im = jpeg.decode(in_file.read(), pixel_format=pixel_format, scaling_factor=scale[read_size], flags=flags)
		in_file.close()
		# if use_RGB:
		# 	start1 = get_time()
		# 	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		# 	print(f" COLOR_BGR2RGB took {int(get_time()-start1)*1000} ms")
		if verbose:
			print(f" turbojpeg read ...{image_filename[-15:]} took {get_time() - start:0.3f} sec.")
		return im
	except Exception as e:
		print("read_jpeg: Failed to load image with turbojpeg {0}: {1}".format(image_filename, e))
		return None


def read_jpeg_simplejpeg(image_filename, use_RGB=True, verbose=False):
	if verbose:
		start_time = get_time()
	format = 'RGB' if use_RGB else 'BGR'
	with open(image_filename, 'rb') as d:
		im = simplejpeg.decode_jpeg(d.read(), format)

	if verbose:
		end_time = get_time()
		print(f" simplejpeg.decode_jpeg() {format} {os.path.basename(image_filename)} "
			  f"took {int((end_time-start_time)*1000+0.5)} ms")
	return im


def read_image(image_filename, read_size='full', verbose=False, use_RGB=True):
	# verbose = True
	downscales = {'full': 1, '1/2': 2, '1/4': 4, '1/8': 8}
	if image_filename[-3:].upper() == 'JPG' or image_filename[-4:].upper() == 'JPEG':
		if has_turbojpeg and (not has_simplejpeg or read_size != 'full'):
			try:
				im = read_jpeg_turbojpeg(image_filename, read_size=read_size, verbose=verbose, use_RGB=use_RGB)
				if im is not None:
					viewer_image = ViewerImage(im, precision=8, downscale=downscales[read_size],
					                           channels=CH_RGB if use_RGB else CH_BGR)
					return viewer_image
			except Exception as e:
				print(f'Failed to read image with simplejpeg: {e}')
		if has_simplejpeg and read_size == 'full':
			try:
				im = read_jpeg_simplejpeg(image_filename, verbose=verbose, use_RGB=use_RGB)
				if im is not None:
						viewer_image = ViewerImage(im, precision=8, downscale=1, channels=CH_RGB if use_RGB else CH_BGR)
						return viewer_image
			except Exception as e:
				print(f'Failed to read image with simplejpeg: {e}')
	try:
		if RAW_FORMAT_EXTENSIONS.get(image_filename.upper()[-4:], -1) == USE_LIBRAW:
			# Use libraw to read the image
			res = read_libraw(image_filename)
			print("read_image channels {}".format(res.channels))
			return res
		if verbose:
			open_cv2_start = get_time()
		cv_size = {'full': cv2.IMREAD_COLOR,
				   '1/2': cv2.IMREAD_REDUCED_COLOR_2,
				   '1/4': cv2.IMREAD_REDUCED_COLOR_4,
				   '1/8': cv2.IMREAD_REDUCED_COLOR_8,
				   }
		cv2_im = cv2.imread(image_filename, cv_size[read_size])
		open_cv2_start2 = get_time()
		# transform default opencv BGR to RGB
		if use_RGB:
			im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
		else:
			im = cv2_im
		if verbose:
			last_time = get_time()
			print(" cv2 imread {0} took {1:0.3f} ( {2:0.3f} + {3:0.3f} ) sec.".format(image_filename, last_time - open_cv2_start,
																	   open_cv2_start2 - open_cv2_start,
																	   last_time - open_cv2_start2),
				  )
	except Exception as e:
		print("Failed to load image {0}: {1}".format(image_filename, e))
		return None

	viewer_image = ViewerImage(im, precision=8, downscale=downscales[read_size], channels=CH_RGB if use_RGB else CH_BGR)
	return viewer_image


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
