#include <iostream>
#include <assert.h>

static constexpr int POINTS_NUM     = 1 << 20;
static constexpr int DIM            = 2;

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

/**
 * @brief Macros for error checking.     
 */
#ifndef devCheckCall
    #define devCheckCall(e) if ( Debug((e), __FILE__, __LINE__) ) { assert(0); }
#endif

#ifndef checkCudaErrors
    #define checkCudaErrors(e) if ( Debug((e), __FILE__, __LINE__) ) { cudaDeviceReset(); exit(1); }
#endif

/**
 * Without __forceinline__ cuda-memcheck catches errors when trying to print out some
 * builtin variables like threadIdx or blockIdx
 *
 */
__device__ /*__forceinline__*/ void foo(float val)
{
    float val2 = val + (threadIdx.x << 2) * 0.3f;
    printf("bid: %d tid: %d, val: %f, val2: %f\n",
        blockIdx.x, threadIdx.x, val, val2);
    // printf("val: %f, val2: %f\n",
    //     val, val2);
}

__global__ void kernelProxy2(
    float const * __restrict__  in,
    int                         pointNum,
    int                         tileId,
    int                         depth)
{
    if (depth == 3)
    {
        if (threadIdx.x == 0)
        {
            printf("[bid: %d, tid: %d] depth : %d, tileId: %d, pointNum: %d, offset: %d\n",
                blockIdx.x, threadIdx.x, depth, tileId, pointNum, pointNum * tileId);
        }
        foo(in[pointNum * tileId + threadIdx.x]);
        return;
    }

    if (threadIdx.x == 0 || threadIdx.x == 1)
    {
        int offset = POINTS_NUM >> depth;
            printf("bid: %d, tid: %d, depth: %d, offset %d, tileId: %d\n",
                blockIdx.x, threadIdx.x, depth, offset * tileId, tileId);

        cudaStream_t stream;
        devCheckCall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        kernelProxy2<<<1, 32, 0, stream>>>(
            in, offset, tileId * 2 + threadIdx.x, depth+1);
        devCheckCall(cudaPeekAtLastError());
        devCheckCall(cudaStreamDestroy(stream));
    }

}

__global__ void kernelProxy(
    float const * __restrict__  in,
    int                         depth)
{
    if (threadIdx.x == 0 || threadIdx.x == 1)
    {
        cudaStream_t stream;
        devCheckCall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        int offset = POINTS_NUM >> depth;
        kernelProxy2<<<1, 32, 0, stream>>>(
            in, offset, threadIdx.x, depth+1);

        devCheckCall(cudaPeekAtLastError());
        devCheckCall(cudaStreamDestroy(stream));
    }
}

int main(void)
{

    float *h_data = new float[POINTS_NUM * DIM];
    float *d_in;

    checkCudaErrors(cudaSetDevice(0));

    checkCudaErrors(cudaMalloc(&d_in, POINTS_NUM * DIM * sizeof(float)));

    for (int k = 0; k < POINTS_NUM; ++k)
    {
        for (int d = 0; d < DIM; ++d)
        {
            h_data[k * DIM + d] = k * 0.1f + d * 0.03f;
        }
    }

    checkCudaErrors(cudaMemcpy(d_in, h_data, POINTS_NUM * DIM * sizeof(float),
        cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 12));

    kernelProxy<<<1, 32>>>(d_in, 1);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaDeviceReset());

    return 0;
}