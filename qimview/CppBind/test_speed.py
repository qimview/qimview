#
# use %load file in ipython to exexute this file
#
import cppimport.import_hook
import wrap_numpy
import numpy as np

# if False:
# 	a = (np.random.rand(2000,2000,3)*100).astype(np.float64)
# 	b = a.copy()
# 	%time b += 1
# 	%time wrap_numpy.increment_3d_omp(b)
# 	%time wrap_numpy.increment_3d(b)


a = (np.random.rand(2000,2000,4)*4095).astype(np.uint16)

# try the function
channels = 4 # CH_RGGB
white_level = 1
black_level = 0
g_r_coeff = 1.5
g_b_coeff = 1
max_value = 4095
max_type = 1 # not used
gamma = 1 # not used

b = np.copy(a)
wrap_numpy.apply_filters_RAW(b, channels, black_level, white_level, g_r_coeff, g_b_coeff, max_value, max_type, gamma)


print(a[0,0,:])
print(a[0,0,:].astype(np.float64)/max_value*255)

print(b[0,0,:])
