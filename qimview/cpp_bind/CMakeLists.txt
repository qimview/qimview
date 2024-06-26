cmake_minimum_required(VERSION 3.21)

project(qimview_cpp LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development)

# find pybind11 path from python script
execute_process(
	COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_include(), end='')"
	OUTPUT_VARIABLE pybind11_path
)

set(pybind11_DIR "${pybind11_path}/../share/cmake/pybind11")
MESSAGE(" pybind11_DIR ${pybind11_DIR}")

find_package(pybind11 CONFIG)
find_package(OpenMP)

pybind11_add_module(qimview_cpp SHARED 
        qimview_cpp.cpp
        image_histogram.hpp
        image_resize.hpp
        image_to_rgb.hpp
)

MESSAGE("Python_Development.Embed_FOUND ${Python_Development.Embed_FOUND}")
MESSAGE("Python_INCLUDE_DIRS ${Python_INCLUDE_DIRS}")

target_link_libraries(
    qimview_cpp PRIVATE  pybind11::module 
)

MESSAGE("pybind11_INCLUDE_DIR = ${pybind11_INCLUDE_DIR}")
target_include_directories(qimview_cpp PUBLIC ${pybind11_INCLUDE_DIR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(qimview_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

install(TARGETS qimview_cpp DESTINATION .)

        #     ["qimview/cpp_bind/qimview_cpp.cpp"],
        #     depends = [ 
        #         'qimview/cpp_bind/image_histogram.hpp',
        #         'qimview/cpp_bind/image_resize.hpp',
        #         'qimview/cpp_bind/image_to_rgb.hpp',
        #         ],
        #     # Example: passing in the version to the compiled code
        #     # define_macros = [('VERSION_INFO', __version__)],
        #     include_dirs=['qimview/cpp_bind','/usr/local/opt/libomp/include'],
