// cppimport this line allows to use import directly
<%
import sys
if sys.platform == 'linux':
    cfg['compiler_args'] = ['-std=c++11', '-fopenmp', '-O3']
    cfg['linker_args'] = ['-fopenmp']
else:
    cfg['compiler_args'] = ['/openmp', '/O2', '/arch:AVX512']
# cfg['compiler_args'] = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7', '/openmp', '/Ox']
setup_pybind11(cfg)
%>


#include "image_resize.hpp"
#include "image_histogram.hpp"
#include "image_to_rgb.hpp"

namespace py = pybind11;

/**
 * Very convenient, with this code we can directly include C++ code for the viewer,
 * it will help improving the performance when we are not using OpenGL shader directly
 *
*/
PYBIND11_MODULE(wrap_numpy, m) {
    m.def("apply_filters_u16_u8", &apply_filters<uint16_t, uint8_t>,
    py::arg().noconvert(),
    py::arg().noconvert(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg()
     );
    m.def("apply_filters_u8_u8", &apply_filters_u8,
    py::arg().noconvert(),
    py::arg().noconvert(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg()
     );
    m.def("apply_filters_u32_u8", &apply_filters<uint32_t, uint8_t>,
    py::arg().noconvert(),
    py::arg().noconvert(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg()
     );
     m.def("apply_filters_s32_u8", &apply_filters<int32_t, uint8_t>,
    py::arg().noconvert(),
    py::arg().noconvert(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg()
     );
     m.def("apply_filters_s16_u8", &apply_filters<int16_t, uint8_t>,
    py::arg().noconvert(),
    py::arg().noconvert(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg()
     );
   m.def("apply_filters_scalar_u8_u8", &apply_filters_scalar<uint8_t, uint8_t>,
    py::arg().noconvert(),
    py::arg().noconvert(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg()
     );
    m.def("apply_filters_scalar_u16_u8", &apply_filters_scalar<uint16_t, uint8_t>,
    py::arg().noconvert(),
    py::arg().noconvert(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg()
     );
    m.def("apply_filters_scalar_u32_u8", &apply_filters_scalar<uint32_t, uint8_t>,
    py::arg().noconvert(),
    py::arg().noconvert(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg()
     );
    m.def("apply_filters_scalar_f64_u8", &apply_filters_scalar_float<double, uint8_t>,
    py::arg().noconvert(),
    py::arg().noconvert(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg()
     );
    m.def("compute_histogram", &compute_histogram,
    py::arg().noconvert(),
    py::arg().noconvert(),
    py::arg(),
    py::arg()
    );
    m.def("image_binning_2x2", &image_binning_2x2<uint8_t, uint8_t, uint16_t, 3>,
    py::arg().noconvert(),
    py::arg().noconvert()
    );
    m.def("image_binning_2x2_test1", &image_binning_2x2_test1<uint8_t, uint8_t, uint16_t, 3>,
    py::arg().noconvert(),
    py::arg().noconvert()
    );
    m.def("image_binning_2x2_test2", &image_binning_2x2_test2<uint8_t, uint8_t>,
    py::arg().noconvert(),
    py::arg().noconvert()
    );
    m.def("image_binning_2x2_test3", &image_binning_2x2_test3<uint8_t, uint8_t, uint16_t, 3>,
    py::arg().noconvert(),
    py::arg().noconvert()
    );
}
