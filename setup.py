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

#This should work pretty good
def compilerName():
    import re, sys
    import distutils.ccompiler
    comp = distutils.ccompiler.get_default_compiler()
    getnext = False

    for a in sys.argv[2:]:
        if getnext:
            comp = a
            getnext = False
            continue
        #separated by space
        if a == '--compiler'  or  re.search('^-[a-z]*c$', a):
            getnext = True
            continue
        #without space
        m = re.search('^--compiler=(.+)', a)
        if m is None:
            m = re.search('^-[a-z]*c(.+)', a)
        if m:
            comp = m.group(1)

    return comp


class build_ext_subclass( build_ext ):
    copt =  {
        'msvc':     ['/openmp',  '/O2', ],  # , '/fp:fast','/favor:INTEL64','/Og'],
        'mingw32' : ['-fopenmp', '-O3', '-std=c++17', '-march=native', ], # ,'-ffast-math'
        'unix':     ['-fopenmp', '-O3', '-std=c++17', ],
    }
    lopt =  {
        'mingw32' : ['-fopenmp'],
    }

    def build_extensions(self):
        c = self.compiler.compiler_type
        print(f"compiler_type {c}")
        print(f"Using compiler {compilerName()}")
        import sys
        print(f" *** platform {sys.platform}")
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
            extra_compile_args=['-std=c++17', '-O3'], # , '-fopenmp'
            extra_link_args=['-L/usr/local/opt/libomp/lib'],
            ),
    ],
    cmdclass = {'build_ext': build_ext_subclass } 
)
setup(**setup_args)
