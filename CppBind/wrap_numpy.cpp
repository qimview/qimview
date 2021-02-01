// cppimport this line allows to use import directly
<%
import sys
if sys.platform == 'linux':
    cfg['compiler_args'] = ['-std=c++11', '-fopenmp', '-O3']
    cfg['linker_args'] = ['-fopenmp']
else:
    cfg['compiler_args'] = ['/openmp']
# cfg['compiler_args'] = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7', '/openmp', '/Ox']
setup_pybind11(cfg)
%>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <type_traits>

namespace py = pybind11;

using namespace pybind11::literals;

#define CH_RGB 1
#define CH_BGR 2
#define CH_RGGB 4 // phase 0, bayer 2
#define CH_GRBG 5 // phase 1, bayer 3 (Boilers)
#define CH_GBRG 6 // phase 2, bayer 0
#define CH_BGGR 7 // phase 3, bayer 1 (Coconuts)
#define R 0
#define G 1
#define B 2


// convert input scalar image to output RGB image applying image filters
template <class input_type, class output_type>
bool apply_filters_scalar(
        py::array_t<input_type> in,
        py::array_t<output_type> out,
        float black_level,
        float white_level,
        input_type max_value, // maximal value based on image precision
        output_type max_type,  // maximal value based on image type (uint8, etc...)
        float gamma
)
{
    auto input  = in.template unchecked<2>(); // Will throw if ndim != 2
    auto output = out.template mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false

    // input has limited values possibilities from 0 to max_value, we can precompute all the results
    // it will save time especially if gamma is used
//    outputtype output_lut[4096*3];
    // lookup table is interesting if the number of input values is clearly inferior to the number of pixels


    if (max_value < 2*input.size())
    {
        auto output_lut = new output_type[(max_value+1)];
        auto output_lut_pos = output_lut;

        for(int v=0; v<=max_value; v++, output_lut_pos++) {
            float y;
            // normalize to 1
            y = float(v)/max_value;
            // black level
            y  = (y<black_level)?0:(y-black_level);
            // rescale to white level as saturation level
            y  /= (white_level-black_level);
            if (gamma!=1) {
                float p = 1.0f/gamma;
                // apply gamma
                y  = powf(y, p);
            }
            // for the moment put result in first three components
            *output_lut_pos = static_cast<output_type>(std::min(1.f,y)*255.f);
        }

        #pragma omp parallel for
        for (py::ssize_t i = 0; i < input.shape(0); i++)
        {
            for (py::ssize_t j = 0; j < input.shape(1); j++)
            {
                input_type val   = std::min(max_value,(input_type)input(i, j));
                output_type val_out = output_lut[val];
                auto output_ptr = &output(i, j, 0);
                *output_ptr++ = val_out;
                *output_ptr++ = val_out;
                *output_ptr   = val_out;
           }
       }

       delete [] output_lut;
    }
    else {
        #pragma omp parallel for
        for (py::ssize_t i = 0; i < input.shape(0); i++)
        {
            for (py::ssize_t j = 0; j < input.shape(1); j++)
            {
                input_type v   = std::min(max_value,(input_type)input(i, j));
                float y;
                // normalize to 1
                y = float(v)/max_value;
                // black level
                y  = (y<black_level)?0:(y-black_level);
                // rescale to white level as saturation level
                y  /= (white_level-black_level);
                if (gamma!=1) {
                    float p = 1.0f/gamma;
                    // apply gamma
                    y  = powf(y, p);
                }
                // for the moment put result in first three components
                output_type val_out = static_cast<output_type>(std::min(1.f,y)*255.f);
                auto output_ptr = &output(i, j, 0);
                *output_ptr++ = val_out;
                *output_ptr++ = val_out;
                *output_ptr   = val_out;
           }
       }
    }
    return true;
}


template <class input_type, class output_type>
bool apply_filters_scalar_float(
        py::array_t<input_type> in,
        py::array_t<output_type> out,
        float black_level,
        float white_level,
        input_type max_value, // maximal value based on image precision
        output_type max_type,  // maximal value based on image type (uint8, etc...)
        float gamma
)
{
    auto input  = in.template unchecked<2>(); // Will throw if ndim != 2
    auto output = out.template mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false

    // float values: no lookup table

    #pragma omp parallel for
    for (py::ssize_t i = 0; i < input.shape(0); i++)
    {
        for (py::ssize_t j = 0; j < input.shape(1); j++)
        {
            input_type v   = std::min(max_value,(input_type)input(i, j));
            float y;
            // normalize to 1
            y = float(v)/max_value;
            // black level
            y  = (y<black_level)?0:(y-black_level);
            // rescale to white level as saturation level
            y  /= (white_level-black_level);
            if (gamma!=1) {
                float p = 1.0f/gamma;
                // apply gamma
                y  = powf(y, p);
            }
            // for the moment put result in first three components
            output_type val_out = static_cast<output_type>(std::min(1.f,y)*255.f);
            auto output_ptr = &output(i, j, 0);
            *output_ptr++ = val_out;
            *output_ptr++ = val_out;
            *output_ptr   = val_out;
        }
    }
    return true;
}


template <class input_type, class output_type>
bool apply_filters(
        py::array_t<input_type> in,
        py::array_t<output_type> out,
        int channels, // channel representation
        float black_level,
        float white_level,
        float g_r_coeff,
        float g_b_coeff,
        input_type max_value, // maximal value based on image precision
        output_type max_type,  // maximal value based on image type (uint8, etc...)
        float gamma,
        float saturation
)
{
    auto input  = in.template unchecked<3>(); // Will throw if ndim != 3
    auto output = out.template mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false

	// assume shape(2) is 4
	const int nb_channels = input.shape(2);
	if ((nb_channels != 4) && (nb_channels!=3))
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
    int r,gr,gb,b, g = -1;
    switch (channels) {
    case CH_RGB:   r = 0; g = 1; b = 2;  break;
    case CH_BGR:   r = 2; g = 1; b = 0;  break;
    case CH_RGGB:  r = 0; gr = 1; gb = 2; b = 3;  break; // CH_RGGB = 4 phase 0, bayer 2
    case CH_GRBG:  r = 1; gr = 0; gb = 3; b = 2;  break; // CH_GRBG = 5 phase 1, bayer 3 (Boilers)
    case CH_GBRG:  r = 2; gr = 3; gb = 0; b = 1;  break; // CH_GBRG = 6 phase 2, bayer 0
    case CH_BGGR:  r = 3; gr = 2; gb = 1; b = 0;  break; // CH_BGGR = 7 phase 3, bayer 1 (Coconuts)
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
            input_type red   = std::min(max_value,(input_type)input_ptr[r]);
            input_type blue  = std::min(max_value,(input_type)(input_ptr[b]));
            input_type green;
            if (g==-1) // raw data
                green = std::min(max_value,(input_type) ((input_ptr[gr]+input_ptr[gb]+1)>>1));
            else // 3 channels RGB or BGR data
                green  = std::min(max_value,(input_type)(input_ptr[g]));

            // for the moment put result in first three components
            auto output_ptr = &output(i, j, 0);
            if (saturation != 1.0f) {
                float r = output_lut[red  *3];
                float g = output_lut[green*3+1];
                float b = output_lut[blue *3+2];
                float mean = (r+b+g)/3;
                // get saturation vector
                r = r-mean;
                g = g-mean;
                b = b-mean;
                // by applying mean+(r,g,b)*coeff find the maximal possible coeff that maintain
                // the values in the RGB cube
                // Check the coefficient that reaches 255
                float val_max_pos = std::max(0.f, std::max(r,std::max(g,b)));
                float max_pos_coeff = 1.f;
                if (val_max_pos>0) max_pos_coeff = (255.f-mean)/val_max_pos;
                // Check the coefficient that reaches 0
                float val_max_neg = std::max(0.f, std::max(-r,std::max(-g,-b)));
                float max_neg_coeff = 1.f;
                if (val_max_neg>0) max_neg_coeff = mean/val_max_neg;
                // Combine both coeff
                float max_coeff = std::min(max_neg_coeff, max_pos_coeff);
                // max_coeff should be > 1
                if (max_coeff<1)
                    printf(" vibrancy: max_coeff<1 %f \n", max_coeff);

                // 1. saturation cannot go beyond max_coeff
                float sat = std::min(max_coeff, saturation);
                // 2. vibrancy = saturation * f(1/max_coeff):
                // 1/max_coeff is the proportion of color in the current pixel compared to the maximal
                float vibrancy =  sat;
                // the additional saturation is weighted by the distance to the maximal coefficient
                if (sat>1)
                    vibrancy = 1.f + (sat-1.f) * (1.0f-1.0f/max_coeff);

                *output_ptr++ = static_cast<output_type>(std::max(0.f, std::min(255.f, mean + r*vibrancy)) + 0.5);
                *output_ptr++ = static_cast<output_type>(std::max(0.f, std::min(255.f, mean + g*vibrancy)) + 0.5);
                *output_ptr   = static_cast<output_type>(std::max(0.f, std::min(255.f, mean + b*vibrancy)) + 0.5);
            } else {
                *output_ptr++ = output_lut[red  *3];
                *output_ptr++ = output_lut[green*3+1];
                *output_ptr   = output_lut[blue *3+2];
            }
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
    m.def("apply_filters_u8_u8", &apply_filters<uint8_t, uint8_t>,
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
}
