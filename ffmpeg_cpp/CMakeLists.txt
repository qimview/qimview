
cmake_minimum_required(VERSION 3.21)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(decode_video)

# set(CMAKE_C_STANDARD 99)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)


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

add_library(decode_video_lib decode_video.cpp decode_video.hpp)

target_link_libraries( decode_video_lib PUBLIC
        FFMPEG::avutil
        FFMPEG::avcodec
        FFMPEG::avformat
        pybind11::module
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
