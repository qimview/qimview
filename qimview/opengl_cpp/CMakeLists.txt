
include(FetchContent)
include(ExternalProject)

cmake_minimum_required(VERSION 3.22)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(opengl_functions LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development)

# Install pybind11, which is a header only library
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.1)
FetchContent_MakeAvailable(pybind11)

MESSAGE(" pybind11_SOURCE_DIR ${pybind11_SOURCE_DIR} pybind11_BINARY_DIR ${pybind11_BINARY_DIR}")

# set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
# set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
# set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# add_subdirectory(glfw-3.4)
#find_package(glfw3 3.3 REQUIRED)
# ExternalProject_Add(glew
#     GIT_REPOSITORY https://github.com/nigels-com/glew.git
#     GIT_TAG glew-2.2.0
#     SOURCE_SUBDIR build/cmake
#     BUILD_IN_SOURCE True
# )
# FetchContent_MakeAvailable(pybind11)

# set(GLEW_DIR "C:/Users/karl/GIT/qimview/qimview/opengl_cpp/glew-2.2.0/glew-2.2.0/build/cmake/installed/lib/cmake/glew")
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
#find_package(GLUT REQUIRED)

add_library(opengl_functions_lib opengl_functions.cpp opengl_functions.hpp)
# MESSAGE(" Python libs ${Python_Development_FOUND} ${Python_LIBRARIES}")

target_include_directories(opengl_functions_lib PUBLIC ${GLEW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})

target_link_libraries( opengl_functions_lib PUBLIC
        # glfw
        ${GLEW_LIBRARIES}
        ${OPENGL_LIBRARIES}
        pybind11::module
        ${Python_LIBRARIES}
)


# MESSAGE(" runtime dlls $<TARGET_RUNTIME_DLLS:opengl_functions_lib>")
# MESSAGE(" dir $<TARGET_FILE_DIR:opengl_functions_lib>")


# pybind11 method:
pybind11_add_module(opengl_cpp SHARED opengl_functions.pybind.cpp)

target_link_libraries(
    opengl_cpp PRIVATE  pybind11::module 
        opengl_functions_lib
)

install(TARGETS opengl_cpp DESTINATION .)
