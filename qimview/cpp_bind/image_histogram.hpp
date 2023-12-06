#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cstdint>

namespace py = pybind11;

bool compute_histogram(
        py::array_t<uint8_t> in,
        py::array_t<uint32_t> out,
        int step_x,
        int step_y)
{
    // printf("compute_histogram in C++ step_x %d step_y %d \n", step_x, step_y);
    auto input  = in.template unchecked<3>(); // Will throw if ndim != 3
    auto output = out.template mutable_unchecked<2>(); // Will throw if ndim != 2 or flags.writeable is false

    if (output.shape(0)!=3) {
        printf("compute_histogram, shape(0) != 3 \n");
        return false;
    }
    if (output.shape(1)!=256) {
        printf("compute_histogram, shape(1) != 256 \n");
        return false;
    }

    uint32_t h[3][256] = {0};

    // We could in some cases add steps in X and Y to speed-up even more this processing

    // // this version is efficient but uses multithreading
    // int inc_x = 3*step_x;
    // #pragma omp parallel for
    // for(int c=0; c<3; c++) {
    //     uint32_t* hc = h[c];
    //     for (py::ssize_t i = 0; i < input.shape(0); i+=step_y)
    //     {
    //         auto input_ptr = &input(i, 0, c);
    //         for (py::ssize_t j = 0; j < input.shape(1); j+=step_x, input_ptr+=inc_x)
    //             hc[*input_ptr]++;
    //     }
    // }

    // int inc_x = 3*(step_x-1);
    // for (py::ssize_t i = 0; i < input.shape(0); i+=step_y)
    // {
    //     auto input_ptr = &input(i, 0, 0);
    //     for (py::ssize_t j = 0; j < input.shape(1); j+=step_x, input_ptr+=inc_x)
    //     {
    //         h[0][*input_ptr++]++;
    //         h[1][*input_ptr++]++;
    //         h[2][*input_ptr++]++;
    //     }
    // }


    // other option, split the image and compute separate histograms for each region
    // then merge the results
    // This value has not been tuned but it seems quite fast
    const int NH = 16;

    // can we use uint16 ?
    uint32_t nb_pixels = (uint32_t)(input.shape(0)*input.shape(1));
    int16_t split_size = nb_pixels/(1<<16);
    // printf("split_size is %d \n", (int)split_size);
    int inc_x = 3*(step_x-1);
    // NOTE: step_y is not used, steps in Y are set to 1

    if (split_size<NH) 
    {
        uint16_t hn[NH][3][256] = {0};

        #pragma omp parallel for
        for(int t=0; t<NH; t++)
        {
            auto lh = hn[t];
            for (py::ssize_t i = t; i < input.shape(0); i+=NH)
            {
                auto input_ptr = &input(i, 0, 0);
                for (py::ssize_t j = 0; j < input.shape(1); j+=step_x, input_ptr+=inc_x)
                {
                    lh[0][*input_ptr++]++;
                    lh[1][*input_ptr++]++;
                    lh[2][*input_ptr++]++;
                }
            }
        }

        for(int t=0; t<NH; t++)
            for(int c=0; c<3; c++)
                for(int n=0; n<256; n++)
                    h[c][n] += hn[t][c][n];
    } else {
        // We need to compute sums on uint32_t values
        uint32_t hn[NH][3][256] = {0};

        #pragma omp parallel for
        for(int t=0; t<NH; t++)
        {
            auto lh = hn[t];
            for (py::ssize_t i = t; i < input.shape(0); i+=NH)
            {
                auto input_ptr = &input(i, 0, 0);
                for (py::ssize_t j = 0; j < input.shape(1); j+=step_x, input_ptr+=inc_x)
                {
                    lh[0][*input_ptr++]++;
                    lh[1][*input_ptr++]++;
                    lh[2][*input_ptr++]++;
                }
            }
        }

        for(int t=0; t<NH; t++)
            for(int c=0; c<3; c++)
                for(int n=0; n<256; n++)
                    h[c][n] += hn[t][c][n];
    }

    // copy to output
    memcpy(&output(0, 0), h, sizeof(h));

    return true;
}
