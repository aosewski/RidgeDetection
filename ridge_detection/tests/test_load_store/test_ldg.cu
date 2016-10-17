
#include "cub/iterator/cache_modified_input_iterator.cuh"
#include "cub/test_util.h"

#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#include <iostream>

template <typename T>
__global__ void kernel(T *in, T *out, int size)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        cub::CacheModifiedInputIterator<cub::LOAD_LDG, T> itr(in);

        T value = *(itr + tid);
        value = (value << 2) - 1;
        out[tid] = value;
    }
}


int main()
{

    const int nElements = 1 << 16;
    int *d_in, *d_out;
    int *h_in, *h_out;

    h_in = new int[nElements];
    h_out = new int[nElements];

    checkCudaErrors(deviceInit(1));

    checkCudaErrors(cudaMalloc((void**)&d_in, nElements * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_out, nElements * sizeof(int)));

    for (int k = 0; k < nElements; ++k)
    {
        h_in[k] = k;
    }

    checkCudaErrors(cudaMemcpy(d_in, h_in, nElements * sizeof(int), cudaMemcpyHostToDevice));

    dim3 gridSize(1);
    const int blockThreads = 512;
    gridSize.x = (nElements + blockThreads - 1) / blockThreads;

    kernel<<<gridSize, blockThreads>>>(d_in, d_out, nElements);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_out, d_out, nElements * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    bool success = true;
    for (int k = 0; k < nElements; ++k)
    {
        int value = (h_in[k] << 2) - 1;
        if (h_out[k] != value)
        {
            success = false;
            std::cout << "ERROR!: is: " << h_out[k] << ", should be: " << value << "\n";
        }
    }

    if (success)
    {
        std::cout << "SUCCESS!\n";
    }

    delete[] h_in;
    delete[] h_out;

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    checkCudaErrors(deviceReset());

    return 0;
}