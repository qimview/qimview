include(FetchContent)

cmake_minimum_required(VERSION 3.24)

project(qimview_cpp LANGUAGES CXX)

# Install pybind11, which is a header only library
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.1)
FetchContent_MakeAvailable(pybind11)

MESSAGE(" pybind11_SOURCE_DIR ${pybind11_SOURCE_DIR} pybind11_BINARY_DIR ${pybind11_BINARY_DIR}")

find_package(pybind11 CONFIG)
find_package(OpenMP)

pybind11_add_module(qimview_cpp SHARED 
        qimview_cpp.cpp
        image_histogram.hpp
        image_resize.hpp
        image_to_rgb.hpp
)


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
