// cppimport this line allows to use import directly
<%
cfg['compiler_args'] = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7', '/openmp']
setup_pybind11(cfg)
%>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

using namespace pybind11::literals;


PYBIND11_MODULE(wrap_numpy, m) {
    m.def("increment_3d_omp", 
	[](py::array_t<double> x) 
	{
		auto r = x.mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false
		#pragma omp parallel for 
		for (py::ssize_t i = 0; i < r.shape(0); i++)
			for (py::ssize_t j = 0; j < r.shape(1); j++)
				for (py::ssize_t k = 0; k < r.shape(2); k++)
					r(i, j, k) += 2.0;

	}, py::arg().noconvert());
    m.def("increment_3d", 
	[](py::array_t<double> x) 
	{
		auto r = x.mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false
		for (py::ssize_t i = 0; i < r.shape(0); i++)
			for (py::ssize_t j = 0; j < r.shape(1); j++)
				for (py::ssize_t k = 0; k < r.shape(2); k++)
					r(i, j, k) += 2.0;

	}, py::arg().noconvert());
}
