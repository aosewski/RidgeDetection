

#include <helper_cuda.h>

#include <iostream>

#include "rd/utils/utilities.hpp"
#include "rd/utils/memory.h"

#include "rd/gpu/util/dev_memcpy.cuh"

#include "cub/test_util.h"


int main()
{
    
    const int width = 100;
    const int height = 150;

    float *h_in, *d_in;
    float *h_out;

    h_in = new float[width * height];
    h_out = new float[width * height];

    deviceInit(0);

    size_t pitch;
    checkCudaErrors(cudaMallocPitch(&d_in, &pitch, height * sizeof(float), width));

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            h_in[i * width + j] = i;
        }
    }

    rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
        d_in, h_in, width, height, pitch, width * sizeof(float));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::gpu::rdMemcpy2D<rd::ROW_MAJOR, rd::COL_MAJOR, cudaMemcpyDeviceToHost>(
        h_out, d_in, height, width, width * sizeof(float), pitch);
    checkCudaErrors(cudaDeviceSynchronize());

    rd::checkResult(h_in, h_out, width * height, true);

    deviceReset();

    return 0;
}
