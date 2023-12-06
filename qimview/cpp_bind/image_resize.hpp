#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <type_traits>

namespace py = pybind11;



template <class input_type, class output_type, class prec_type, uint8_t channels>
bool image_binning_2x2(
        py::array_t<input_type> in,
        py::array_t<output_type> out
)
{
    auto input  = in.template unchecked<3>(); // Will throw if ndim != 2
    auto output = out.template mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false

    // TODO: check that output has the correct size
    bool ok = (output.shape(0) == (input.shape(0)>>1))&& (output.shape(1) == (input.shape(1)>>1)) 
                && (output.shape(2) == input.shape(2)) && (output.shape(2) == channels);
    if (!ok) {
        printf("image_binning_2x2() output shape not valid (%d %d %d) != (%d %d %d)\n",
            (int) output.shape(0),  (int) output.shape(1), (int)  output.shape(2),
            (int) input.shape(0)>>1,(int) input.shape(1)>>1, (int) input.shape(2)
            );
        return false;
    }


    int16_t height = static_cast<int16_t>(input.shape(0));
    int16_t width  = static_cast<int16_t>(input.shape(1));

    for (int16_t i = 0; i < height; i+=2)
    {
        auto output_ptr = &output(i>>1, 0, 0);

        prec_type sum[3000][channels] = {0};

        auto input_ptr = &input(i,   0, 0);
        for (int16_t j = 0; j < (width>>1); j++)
        {
            for (uint8_t c = 0; c < channels; c++)
                sum[j][c] = input_ptr[c]+input_ptr[channels+c];
            input_ptr += 2*channels;
        }

        input_ptr = &input(i+1,   0, 0);
        for (int16_t j = 0; j < (width>>1); j++)
        {
            for (uint8_t c = 0; c < channels; c++) {
                sum[j][c] += input_ptr[c]+input_ptr[channels+c];
                *output_ptr++ = (sum[j][c]+2)>>2;
            }
            input_ptr += 2*channels;
        }
    }

    return true;
}


template <class input_type, class output_type, class prec_type, uint8_t channels>
bool image_binning_2x2_test1(
        py::array_t<input_type> in,
        py::array_t<output_type> out
)
{
    auto input  = in.template unchecked<3>(); // Will throw if ndim != 2
    auto output = out.template mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false

    // TODO: check that output has the correct size
    bool ok = (output.shape(0) == (input.shape(0)>>1))&& (output.shape(1) == (input.shape(1)>>1)) 
                && (output.shape(2) == input.shape(2)) && (output.shape(2) == channels);
    if (!ok) {
        printf("image_binning_2x2() output shape not valid (%d %d %d) != (%d %d %d)\n",
            (int) output.shape(0),  (int)output.shape(1),   (int) output.shape(2),
            (int) input.shape(0)>>1,(int) input.shape(1)>>1, (int) input.shape(2)
            );
        return false;
    }


    int16_t height = static_cast<int16_t>(input.shape(0));
    int16_t width  = static_cast<int16_t>(input.shape(1));
    int16_t height2 = height>>1;
    int16_t width2  = width>>1;

    #pragma omp parallel for 
    for (int_fast16_t i = 0; i < height2; i++)
    {
        auto output_ptr = &output(i, 0, 0);

        auto input_ptr1 = &input(2*i,   0, 0);
        auto input_ptr2 = input_ptr1 + channels;
        auto input_ptr3 = &input(2*i+1, 0, 0);
        auto input_ptr4 = input_ptr3 + channels;
        for (int_fast16_t j = 0; j < width2; j++)
        {
            for (uint_fast8_t c = 0; c < channels; c++)
            {
                prec_type sum = (prec_type) *input_ptr1++ + (prec_type)*input_ptr2++ + 
                                (prec_type)*input_ptr3++ + (prec_type)*input_ptr4++;
                *output_ptr++ = static_cast<output_type>((sum+2)>>2);
            }
            input_ptr1+=channels;
            input_ptr2+=channels;
            input_ptr3+=channels;
            input_ptr4+=channels;
        }

    }

    return true;
}


template <class input_type, class output_type>
bool image_binning_2x2_test2(
        py::array_t<input_type> in,
        py::array_t<output_type> out
)
{
    auto input  = in.template unchecked<3>(); // Will throw if ndim != 2
    auto output = out.template mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false

    // TODO: check that output has the correct size
    bool ok = (output.shape(0) == (input.shape(0)>>1))&& (output.shape(1) == (input.shape(1)>>1)) 
                && (output.shape(2) == input.shape(2));
    if (!ok) {
        printf("image_binning_2x2() output shape not valid (%d %d %d) != (%d %d %d)\n",
            (int) output.shape(0),  (int)output.shape(1),   (int) output.shape(2),
            (int) input.shape(0)>>1,(int) input.shape(1)>>1, (int) input.shape(2)
            );
        return false;
    }


    int16_t height = static_cast<int16_t>(input.shape(0));
    int16_t width  = static_cast<int16_t>(input.shape(1));
    int16_t height2 = height>>1;
    int16_t width2  = width>>1;
    // int16_t channels = input.shape(2);


    // #pragma omp parallel for
    auto output_ptr = &output(0, 0, 0);
    for (int_fast16_t i = 0; i < height2; i++)
    {
        auto input_ptr = &input(2*i, 0, 0);
        for (int_fast16_t j = 0; j < width2; j++)
        {
            *output_ptr++ = *input_ptr++;
            *output_ptr++ = *input_ptr++;
            *output_ptr++ = *input_ptr++;
            input_ptr += 3;
        }

    }

    return true;
}




template <class input_type, class output_type, class prec_type, uint8_t channels>
bool image_binning_2x2_test3(
        py::array_t<input_type> in,
        py::array_t<output_type> out
)
{
    auto input  = in.template unchecked<3>(); // Will throw if ndim != 2
    auto output = out.template mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false

    // TODO: check that output has the correct size
    bool ok = (output.shape(0) == (input.shape(0)>>1))&& (output.shape(1) == (input.shape(1)>>1)) 
                && (output.shape(2) == input.shape(2)) && (output.shape(2) == channels);
    if (!ok) {
        printf("image_binning_2x2() output shape not valid (%d %d %d) != (%d %d %d)\n",
            (int) output.shape(0),  (int)output.shape(1),   (int) output.shape(2),
            (int) input.shape(0)>>1,(int) input.shape(1)>>1, (int) input.shape(2)
            );
        return false;
    }


    int16_t height = static_cast<int16_t>(input.shape(0));
    int16_t width  = static_cast<int16_t>(input.shape(1));
    int16_t height2 = height>>1;
    // int16_t width2  = width>>1;

    prec_type* buf;

    #pragma omp parallel private(buf)
    {
        #define BUF_SIZE 128
        #define _MIN(a,b) ((a)<(b)?(a):(b))
        buf = (prec_type*) malloc(BUF_SIZE*channels*sizeof(prec_type));

        #pragma omp for 
        for (int_fast16_t i = 0; i < height2; i++)
        {
            // printf("i=%d \n", i);
            // split width into blocks

            // auto output_ptr = &output(i, 0, 0);

            auto input_ptr1 = &input(2*i,   0, 0);
            auto input_ptr2 = &input(2*i+1, 0, 0);
            int_fast16_t start = 0;
            int_fast16_t end   = _MIN(start+BUF_SIZE, width);
            while (start<width) 
            {
                // if (i==height2-1)
                //     printf(" %d -- %d \n", start, end);
                auto buf_ptr = buf;
                int_fast16_t size  = end-start;
                for (int_fast16_t j = 0; j < size; j++)
                {
                    *buf_ptr++ = (prec_type) *input_ptr1++ + (prec_type)*input_ptr2++;
                    *buf_ptr++ = (prec_type) *input_ptr1++ + (prec_type)*input_ptr2++;
                    *buf_ptr++ = (prec_type) *input_ptr1++ + (prec_type)*input_ptr2++;
                }

                //int_fast16_t size2 = size>>1;
                // buf_ptr = buf;
                // auto buf_ptr2 = buf + channels;
                // for (int_fast16_t j = 0; j < size2; j++, buf_ptr += 2*channels, but_ptr2 += 2*channels)
                //     for (uint_fast8_t c = 0; c < channels; c++)
                //         *output_ptr++ = static_cast<output_type>((*buf_ptr++ + *buf_ptr2++ + 2)>>2);

                start = end;
                end   = _MIN(start+BUF_SIZE, width);
            }
        }

        free(buf);
    }

    return true;
}
