
#include <assert.h>
#include <cuda_profiler_api.h>

#include "rd/gpu/block/block_tile_load_store4.cuh"
#include "rd/utils/memory.h"
#include "tests/test_util.hpp"

#include "cub/thread/thread_store.cuh"


static constexpr int BLOCK_THREADS      = 128;
static constexpr int POINTS_PER_THREAD  = 4;
static constexpr int POINTS_NUM     = BLOCK_THREADS << 16;
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

#ifndef rdDebug
    #define rdDebug(e) Debug((e), __FILE__, __LINE__)
#endif



__global__ void kernel(
    float const * in,
    float *       out)
{
    __shared__ float smem[BLOCK_THREADS * POINTS_PER_THREAD * DIM];

    typedef float PointT[DIM];

    // PointT* smemPtr = reinterpret_cast<PointT*>(smem + threadIdx.x * DIM);
    // PointT const * inPtr = reinterpret_cast<const PointT*>(
    //     in + (blockIdx.x * BLOCK_THREADS + threadIdx.x) * DIM);

    in += blockIdx.x * BLOCK_THREADS * POINTS_PER_THREAD * DIM + threadIdx.x * DIM;

    #pragma unroll
    for (int p = 0; p < POINTS_PER_THREAD; ++p)
    {
        float * sptr = smem + p * BLOCK_THREADS * DIM + threadIdx.x * DIM;
        #pragma unroll
        for (int d = 0; d < DIM; ++d)
        {
            sptr[d] = in[d];
        }
        in += BLOCK_THREADS * DIM;
    }
    __syncthreads();

    typedef rd::gpu::BlockTileStorePolicy<
        BLOCK_THREADS,
        POINTS_PER_THREAD,
        cub::STORE_DEFAULT>
    BlockTileStorePolicyT;

    typedef rd::gpu::BlockTileStore<
        BlockTileStorePolicyT,
        DIM,
        rd::ROW_MAJOR,
        rd::gpu::IO_BACKEND_CUB,
        float,
        int>
    BlockTileStoreT;

    typedef float AliasedPoints[POINTS_PER_THREAD][DIM];

    BlockTileStoreT::storeTile2Row(
        out,
        *reinterpret_cast<AliasedPoints*>(smem + threadIdx.x * POINTS_PER_THREAD * DIM),
        int(blockIdx.x * BLOCK_THREADS * POINTS_PER_THREAD),
        1);
}


int main(void)
{
    float *h_data = new float[POINTS_NUM * DIM];
    float *d_in, *d_out;

    std::cout << "elements num: " << POINTS_NUM * DIM << std::endl;

    checkCudaErrors(cudaSetDevice(0));

    checkCudaErrors(cudaMalloc(&d_in, POINTS_NUM * DIM * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_out, POINTS_NUM * DIM * sizeof(float)));

    for (int k = 0; k < POINTS_NUM; ++k)
    {
        for (int d = 0; d < DIM; ++d)
        {
            h_data[k * DIM + d] = k * 0.1f + d * 0.03;
        }
    }

    checkCudaErrors(cudaMemcpy(d_in, h_data, POINTS_NUM * DIM * sizeof(float),
        cudaMemcpyHostToDevice));

    dim3 gridDim(1);
    gridDim.x = (POINTS_NUM / POINTS_PER_THREAD + BLOCK_THREADS - 1) / BLOCK_THREADS;

    #ifdef RD_PROFILE
    cudaProfilerStart();
    kernel<<<gridDim, BLOCK_THREADS>>>(d_in, d_out);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    #endif
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    int result = CompareDeviceResults(h_data, d_out, POINTS_NUM * DIM, true, true, false);

    std::cout << "\n\nTest... ";
    std::cout << ((result) ? "FAIL!" : "SUCCESS!") << std::endl;

    checkCudaErrors(cudaDeviceReset());
    return 0;
}