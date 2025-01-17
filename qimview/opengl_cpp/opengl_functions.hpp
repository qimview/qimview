#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "opengl_functions.hpp"

#include <GL/glew.h>
#include <GL/gl.h>

namespace py = pybind11;

namespace GLcpp
{
    bool InitGlew();

    // binding for glBufferData with numpy array
    template<typename PixelType>
    void glBufferData(GLenum target, GLsizeiptr size, const py::array_t<PixelType>& data, GLenum usage)
    {
        py::buffer_info buf = data.request();
        ::glBufferData(target, size, buf.ptr, usage);
    }

    template<typename PixelType>
    void glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const py::array_t<PixelType>& data)
    {
        py::buffer_info buf = data.request();
        ::glBufferSubData(target, offset, size, buf.ptr);
    }

}
