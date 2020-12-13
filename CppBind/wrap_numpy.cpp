// cppimport this line allows to use import directly
<%
cfg['compiler_args'] = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7', '/openmp']
setup_pybind11(cfg)
%>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

using namespace pybind11::literals;

template <class data_type>
void apply_filters(py::array_t<data_type> x)
{
    auto r = x.mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false
	#pragma omp parallel for
    for (py::ssize_t i = 0; i < r.shape(0); i++)
        for (py::ssize_t j = 0; j < r.shape(1); j++)
            for (py::ssize_t k = 0; k < r.shape(2); k++)
                r(i, j, k) += static_cast<data_type>(2);
}


#define CH_RGGB 4 // phase 0, bayer 2
#define CH_GRBG 5 // phase 1, bayer 3 (Boilers)
#define CH_GBRG 6 // phase 2, bayer 0
#define CH_BGGR 7 // phase 3, bayer 1 (Coconuts)
#define R 0
#define G 1
#define B 2

template <class input_type, class output_type>
bool apply_filters_RAW(
        py::array_t<input_type> in,
        py::array_t<output_type> out,
        int channels, // channel representation
        float black_level,
        float white_level,
        float g_r_coeff,
        float g_b_coeff,
        input_type max_value, // maximal value based on image precision
        output_type max_type,  // maximal value based on image type (uint8, etc...)
        float gamma
)
{
    auto input  = in.unchecked<3>(); // Will throw if ndim != 3
    auto output = out.mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false

	// assume shape(2) is 4
	if (input.shape(2) != 4)
	{
	    //std::cout << "Number of channels is not 4 ", r.shape(2) << std::endl;
	    return false;
	}

    // input has limited values possibilities from 0 to max_value, we can precompute all the results
    // it will save time especially if gamma is used
//    outputtype output_lut[4096*3];
    auto output_lut = new output_type[(max_value+1)*3];
    auto output_lut_pos = output_lut;

    for(int v=0; v<=max_value; v++, output_lut_pos += 3) {
        float r, g, b;

        // normalize to 1
        g = float(v)/max_value;

        // black level
        g  = (g<black_level)?0:(g-black_level);
        // rescale to white level as saturation level
        g  /= (white_level-black_level);

        // white balance
        r  = g*g_r_coeff;
        b  = g*g_b_coeff;

        if (gamma!=1) {
            float p = 1.0f/gamma;
            // apply gamma
            r  = powf(r, p);
            g  = powf(g, p);
            b  = powf(b, p);
        }

        // for the moment put result in first three components
        output_lut_pos[0] = static_cast<output_type>(std::min(1.f,r)*255.f);
        output_lut_pos[1] = static_cast<output_type>(std::min(1.f,g)*255.f);
        output_lut_pos[2] = static_cast<output_type>(std::min(1.f,b)*255.f);
    }

    // transform bayer input to RGB
    int r,gr,gb,b;
    switch (channels) {
    case 4:   r = 0; gr = 1; gb = 2; b = 3;  break; // CH_RGGB = 4 phase 0, bayer 2
    case 5:   r = 1; gr = 0; gb = 3; b = 2;  break; // CH_GRBG = 5 phase 1, bayer 3 (Boilers)
    case 6:   r = 2; gr = 3; gb = 0; b = 1;  break; // CH_GBRG = 6 phase 2, bayer 0
    case 7:   r = 3; gr = 2; gb = 1; b = 0;  break; // CH_BGGR = 7 phase 3, bayer 1 (Coconuts)
    default:  r = 0; gr = 1; gb = 2; b = 3;  break; // this should not happen
    }

	#pragma omp parallel for
    for (py::ssize_t i = 0; i < input.shape(0); i++)
    {
        for (py::ssize_t j = 0; j < input.shape(1); j++)
        {
            // first retreive black point to get the coefficients right ...
            // 5% of dynamics?
            // bayer 2 rgb
            // try to speed-up input data access
            auto input_ptr = &input(i, j, 0);
            input_type red   = input_ptr[r];
            input_type green = (input_ptr[gr]+input_ptr[gb]+1)>>1;
            input_type blue  = input_ptr[b];

            // for the moment put result in first three components
            auto output_ptr = &output(i, j, 0);
            *output_ptr++ = output_lut[red  *3];
            *output_ptr++ = output_lut[green*3+1];
            *output_ptr   = output_lut[blue *3+2];
       }
   }

   delete [] output_lut;
   return true;
}



/**
 * Very convenient, with this code we can directly include C++ code for the viewer,
 * it will help improving the performance when we are not using OpenGL shader directly
 *
*/
PYBIND11_MODULE(wrap_numpy, m) {
    m.def("apply_filters_float", &apply_filters<float>, py::arg().noconvert());
    m.def("apply_filters_uint16", &apply_filters<uint16_t>, py::arg().noconvert());
    m.def("apply_filters_RAW", &apply_filters_RAW<uint16_t, uint8_t>,
    py::arg().noconvert(),
    py::arg().noconvert(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg(),
    py::arg()
     );
}
