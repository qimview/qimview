from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
import os
# import distutils
# distutils.log.set_verbosity(1)


# from distutils.command.build_ext import build_ext
# compiler/linker args depending on the platform: check how to set it up
# compiler_args = ['-std=c++11', '-fopenmp', '-O3']
# compiler_args = ['/openmp', '/O2', '/arch:AVX512']
# compiler_args = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7', '/openmp', '/Ox']

# copt =  {'msvc': ['/openmp', '/Ox', '/fp:fast','/favor:INTEL64','/Og']  ,
#      'mingw32' : ['-fopenmp','-O3','-ffast-math','-march=native']       }
# lopt =  {'mingw32' : ['-fopenmp'] }

# class build_ext_subclass( build_ext ):
#     def build_extensions(self):
#         c = self.compiler.compiler_type
#         if copt.has_key(c):
#            for e in self.extensions:
#                e.extra_compile_args = copt[ c ]
#         if lopt.has_key(c):
#             for e in self.extensions:
#                 e.extra_link_args = lopt[ c ]
#         build_ext.build_extensions(self)

setup_args = dict(
    ext_modules = [
        Pybind11Extension("qimview_cpp",
            ["qimview/cpp_bind/qimview_cpp.cpp"],
            depends = [ 
                'qimview/cpp_bind/image_histogram.hpp',
                'qimview/cpp_bind/image_resize.hpp',
                'qimview/cpp_bind/image_to_rgb.hpp',
                ],
            # Example: passing in the version to the compiled code
            # define_macros = [('VERSION_INFO', __version__)],
            include_dirs=['qimview/cpp_bind'],
            extra_compile_args=['-std=c++17', '-fopenmp','-O3']            
            ),
    ],
    # cmdclass = {'build_ext': build_ext_subclass } 
)
setup(**setup_args)
