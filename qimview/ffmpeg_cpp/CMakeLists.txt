
include(FetchContent)

cmake_minimum_required(VERSION 3.22)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(decode_video LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Install pybind11, which is a header only library
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.1)
FetchContent_MakeAvailable(pybind11)

MESSAGE(" pybind11_SOURCE_DIR ${pybind11_SOURCE_DIR} pybind11_BINARY_DIR ${pybind11_BINARY_DIR}")

find_package(FFMPEG COMPONENTS avcodec avdevice avfilter avformat avutil)

if(FFMPEG_FOUND)
#  FFMPEG_INCLUDE_DIRS  - Include directory necessary for using the required components headers.
#  FFMPEG_LIBRARIES     - Link these to use the required ffmpeg components.
#  FFMPEG_DEFINITIONS   - Compiler switches required for using the required ffmpeg components.
    message("FFMPEG_INCLUDE_DIRS = ${FFMPEG_INCLUDE_DIRS} ")
    message("FFMPEG_LIBRARIES = ${FFMPEG_LIBRARIES} ")
    message("FFMPEG_DEFINITIONS = ${FFMPEG_DEFINITIONS} ")

    include_directories(${FFMPEG_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "FFMPEG not found")
endif()

list(APPEND CMAKE_BUILD_RPATH ${FFMPEG_ROOT}/lib)

add_library(decode_video_lib decode_video.cpp decode_video.hpp)
# MESSAGE(" Python libs ${Python_Development_FOUND} ${Python_LIBRARIES}")

target_link_libraries( decode_video_lib PUBLIC
        FFMPEG::avutil
        FFMPEG::avcodec
        FFMPEG::avformat
        pybind11::module
#        ${Python_LIBRARIES}
)


add_executable(decode_video decode_video_main.cpp)

target_link_libraries(
        decode_video
        decode_video_lib
)

MESSAGE(" runtime dlls $<TARGET_RUNTIME_DLLS:decode_video>")
MESSAGE(" dir $<TARGET_FILE_DIR:decode_video>")

get_target_property( path FFMPEG::avutil IMPORTED_LOCATION)
MESSAGE(" ${path}")


#add_custom_command(TARGET decode_video POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        
#        $<TARGET_FILE_DIR:decode_video>)

# pybind11 method:
pybind11_add_module(decode_video_py SHARED decode_video.pybind.cpp)

target_link_libraries(
        decode_video_py PRIVATE  pybind11::module 
        decode_video_lib
)

install(TARGETS decode_video_py DESTINATION .)
