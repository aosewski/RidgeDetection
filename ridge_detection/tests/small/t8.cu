#include <iostream>
#include <assert.h>

__host__ __device__ __forceinline__ cudaError_t Debug(
    cudaError_t     error,
    const char*     filename,
    int             line)
{
    if (error)
    {
    #if (__CUDA_ARCH__ == 0)
        fprintf(stderr, "CUDA error %d [%s, %d]: %s\n",
            error, filename, line, cudaGetErrorString(error));
        fflush(stderr);
    #elif (__CUDA_ARCH__ >= 200)
        printf("CUDA error %d [block (%3d,%3d,%3d) thread (%3d,%3d,%3d), %s, %d]\n",
            error, blockIdx.z, blockIdx.y, blockIdx.x,
            threadIdx.z, threadIdx.y, threadIdx.x, filename, line);
    #endif
    }
    return error;
}

#ifndef devCheckCall
    #define devCheckCall(e) if ( Debug((e), __FILE__, __LINE__) ) { assert(0); }
#endif

#ifndef checkCudaErrors
    #define checkCudaErrors(e) if ( Debug((e), __FILE__, __LINE__) ) { cudaDeviceReset(); exit(1); }
#endif

//----------------------------------------------------------
//
//----------------------------------------------------------

__global__ void testKernelNoCDP(float *  in1, float * out1, int size, float value)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float inVal = in1[idx];
    out1[idx] = inVal * value + value;
}

// #if defined(__cplusplus) && defined(__CUDACC__) 
// __global__ void testKernelCDP(float *  in1, float * out1, int size, float value)
// {
//     #if (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))
//     const int blockThreads = 256;
//     dim3 gridSize(1);
//     gridSize.x = size / blockThreads;
//     testKernelNoCDP<<<gridSize, blockThreads>>>(in1, out1, size, value);
//     #endif
// }
// #endif


//----------------------------------------------------------
//
//----------------------------------------------------------


static constexpr int POINTS_NUM     = 1 << 16;

//----------------------------------------------------------
//
//----------------------------------------------------------


int main(void)
{

    float value = 0.47;

    float *h_data = new float[POINTS_NUM];
    float *h_out1 = new float[POINTS_NUM];
    float *d_in1, *d_out1;

    checkCudaErrors(cudaSetDevice(0));

    checkCudaErrors(cudaMalloc(&d_in1, POINTS_NUM * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_out1, POINTS_NUM * sizeof(float)));

    for (int k = 0; k < POINTS_NUM; ++k)
    {
        h_data[k] = k * 0.001f;
    }

    checkCudaErrors(cudaMemcpy(d_in1, h_data, POINTS_NUM * sizeof(float),
        cudaMemcpyHostToDevice));

    // #if defined(__cplusplus) && defined(__CUDACC__) 
    // #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))
    // for (int k = 0; k < 100; ++k)
    // {
    //     testKernelCDP<<<1, 1>>>(d_in1, d_out1, POINTS_NUM, value);
    //     checkCudaErrors(cudaGetLastError());
    // }
    // checkCudaErrors(cudaDeviceSynchronize());

    // checkCudaErrors(cudaMemcpy(h_out1, d_in1, POINTS_NUM * sizeof(float),
    //     cudaMemcpyDeviceToHost));

    // for (int k = 0; k < POINTS_NUM; ++k)
    // {
    //     if (int(h_out1[k]) != h_data[k] * value + value)
    //     {
    //         std::cout << "--- ERROR!---- h_out1["<<k<<"]: (" << int(h_out1[k])
    //             << ") != (" << h_data[k] * value + value << ")" << std::endl;
    //         break;
    //     }
    // }
    // #endif
    // #endif

    const int blockThreads = 256;
    dim3 gridSize(1);
    gridSize.x = POINTS_NUM / blockThreads;

    for (int k = 0; k < 100; ++k)
    {
        testKernelNoCDP<<<gridSize, blockThreads>>>(d_in1, d_out1, POINTS_NUM, value);
        checkCudaErrors(cudaGetLastError());
    }
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_out1, d_in1, POINTS_NUM * sizeof(float),
        cudaMemcpyDeviceToHost));

    for (int k = 0; k < POINTS_NUM; ++k)
    {
        if (int(h_out1[k]) != h_data[k] * value + value)
        {
            std::cout << "--- ERROR!---- h_out1["<<k<<"]: (" << int(h_out1[k])
                << ") != (" << h_data[k] * value + value << ")" << std::endl;
            break;
        }
    }

    checkCudaErrors(cudaDeviceReset());

    return 0;
}