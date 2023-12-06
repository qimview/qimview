from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
# import os
# import distutils
# distutils.log.set_verbosity(1)

# from distutils.command.build_ext import build_ext
# compiler/linker args depending on the platform: check how to set it up
# compiler_args = ['-std=c++11', '-fopenmp', '-O3']
# compiler_args = ['/openmp', '/O2', '/arch:AVX512']
# compiler_args = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7', '/openmp', '/Ox']

class build_ext_subclass( build_ext ):
    # win32, linux, darwin
    # msvc, unix
    copt =  {
        'win32' : ['/openmp',  '/O2', ],  # , '/fp:fast','/favor:INTEL64','/Og'],
        'linux' : ['-fopenmp', '-O3', '-std=c++17', ],
        'darwin': [            '-O3', '-std=c++17', ],
    }
    lopt =  {
        'darwin' : ['-L/usr/local/opt/libomp/lib'],
    }

    def build_extensions(self):
        # c = self.compiler.compiler_type
        # print(f"compiler_type {c}")
        import sys
        print(f" *** platform {sys.platform} {self.compiler.get_executable()}")
        c = sys.platform
        if c in self.copt:
           for e in self.extensions:
               e.extra_compile_args = self.copt[ c ]
        if c in self.lopt:
            for e in self.extensions:
                e.extra_link_args = self.lopt[ c ]
        build_ext.build_extensions(self)

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
            include_dirs=['qimview/cpp_bind','/usr/local/opt/libomp/include'],
            ),
    ],
    cmdclass = {'build_ext': build_ext_subclass } 
)
setup(**setup_args)
