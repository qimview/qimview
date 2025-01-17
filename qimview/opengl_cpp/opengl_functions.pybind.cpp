#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string> 
#include "opengl_functions.hpp"

namespace py = pybind11;


PYBIND11_MODULE(opengl_cpp, m) {
  m.doc() = "opengl_cpp plugin: binding of opengl functions with pybind11"; // optional module docstring

  m.def("InitGlew",        &GLcpp::InitGlew);
  m.def("glBufferData_u8",      &GLcpp::glBufferData<uint8_t>);
  m.def("glBufferData_u16",     &GLcpp::glBufferData<uint16_t>);
  m.def("glBufferSubData_u8",   &GLcpp::glBufferSubData<uint8_t>);
  m.def("glBufferSubData_u16",  &GLcpp::glBufferSubData<uint16_t>);
}
