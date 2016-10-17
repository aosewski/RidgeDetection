#include "cub/thread/thread_load.cuh"
#include "cub/test_util.h"

#include "rd/gpu/util/dev_static_for.cuh"
#include "trove/block.h"

#include <helper_cuda.h>

#include <iostream>

static const int POINTS_PER_THREAD  = 8;
static const int DIM                = 2;
static const int SAMPLES_PER_THREAD = POINTS_PER_THREAD * DIM;


//-----------------------------------------------------------------------
//      UTILITY
//-----------------------------------------------------------------------

// /**
//  * Register modifier for pointer-types (for inlining PTX assembly)
//  */
// #if defined(_WIN64) || defined(__LP64__)
//     #define __CUB_LP64__ 1
//     // 64-bit register modifier for inlined asm
//     #define _CUB_ASM_PTR_ "l"
//     #define _CUB_ASM_PTR_SIZE_ "u64"
// #else
//     #define __CUB_LP64__ 0
//     // 32-bit register modifier for inlined asm
//     #define _CUB_ASM_PTR_ "r"
//     #define _CUB_ASM_PTR_SIZE_ "u32"
// #endif

// __device__ __forceinline__ unsigned int __isLocal(const void *ptr)
// {
//   // XXX WAR unused variable warning
//   (void) ptr;

//   unsigned int ret;

// #if __CUDA_ARCH__ >= 200
//   asm volatile ("{ \n\t"
//                 "    .reg .pred p; \n\t"
//                 "    isspacep.local p, %1; \n\t"
//                 "    selp.u32 %0, 1, 0, p;  \n\t"
//                 "} \n\t" : "=r"(ret) : _CUB_ASM_PTR_(ptr));
// #else
//   ret = 0;
// #endif

//   return ret;
// } 

//-----------------------------------------------------------------------
//      LOAD & STORE 
//-----------------------------------------------------------------------

template <int   BLOCK_THREADS>
__device__ __forceinline__ void loadTile1(
    double const *  in,
    double          (&samples)[POINTS_PER_THREAD][DIM])
{
    typedef ulonglong2 DeviceWord;
    typedef DeviceWord AliasedSamples[POINTS_PER_THREAD];

    AliasedSamples &aliasedSamples = reinterpret_cast<AliasedSamples&>(samples);
    DeviceWord const *inVecPtr     = reinterpret_cast<DeviceWord const *>(in);

    // if (blockIdx.x == 0 && threadIdx.x == 0 && __isLocal(aliasedSamples))
    // {
    //     printf("AliasedSamples is local!\n");
    // }

    //loading in striped fashion
    #pragma unroll
    for (int p = 0; p < POINTS_PER_THREAD; ++p)
    {
        // aliasedSamples[p] = *(inVecPtr + p * BLOCK_THREADS + threadIdx.x);
        aliasedSamples[p] = cub::ThreadLoad<cub::LOAD_LDG>(inVecPtr + p * BLOCK_THREADS + threadIdx.x);
    }
}

struct IterateDim
{
    template <typename D, typename P, typename TroveArray>
    static __device__ void impl(D dIdx, P pIdx, double (&smpl)[POINTS_PER_THREAD][DIM], TroveArray const &ary)
    {
        smpl[P::value][D::value] = trove::get<P::value * DIM + D::value>(ary);
    }
};

struct IteratePoints
{
    template <typename P, typename TroveArray>
    static __device__ void impl(P pIdx, double (&smpl)[POINTS_PER_THREAD][DIM], TroveArray const &ary)
    {
        rd::gpu::StaticFor<0, DIM, IterateDim>::impl(pIdx, smpl, ary);
    }
};

template <int   BLOCK_THREADS>
__device__ __forceinline__ void loadTile2(
    double const * __restrict__  in,
    double          (&samples)[POINTS_PER_THREAD][DIM])
{
    // typedef ulonglong2 DeviceWord;
    // DeviceWord vecPoints[POINTS_PER_THREAD];
    // DeviceWord const * __restrict__ inVecPtr     = reinterpret_cast<DeviceWord const *>(in);

    // // if (blockIdx.x == 0 && threadIdx.x == 0 && __isLocal(aliasedSamples))
    // // {
    // //     printf("AliasedSamples is local!\n");
    // // }

    // //loading in striped fashion
    // #pragma unroll
    // for (int p = 0; p < POINTS_PER_THREAD; ++p)
    // {
    //     // vecPoints[p] = *(inVecPtr + p * BLOCK_THREADS + threadIdx.x);
    //     vecPoints[p] = cub::ThreadLoad<cub::LOAD_LDG>(inVecPtr + p * BLOCK_THREADS + threadIdx.x);
    // }
    // //copy
    // #pragma unroll
    // for (int p = 0; p < POINTS_PER_THREAD; ++p)
    // {
    //     #pragma unroll
    //     for (int d = 0; d < DIM; ++d)
    //     {
    //         samples[p][d] = reinterpret_cast<double*>(vecPoints)[p * DIM + d];
    //     }
    // }
   
    typedef trove::array<double, SAMPLES_PER_THREAD> TroveArray;
    TroveArray tArray;
    tArray = trove::load_array_warp_contiguous<TroveArray::size>(in, threadIdx.x);

    rd::gpu::StaticFor<0, POINTS_PER_THREAD, IteratePoints>::impl(samples, tArray);
}

template <int   BLOCK_THREADS>
__device__ __forceinline__ void storeTile(
    double *        out,
    double          (&samples)[POINTS_PER_THREAD][DIM])
{
    typedef ulonglong2 DeviceWord;
    typedef DeviceWord AliasedSamples[POINTS_PER_THREAD];
    
    AliasedSamples &aliasedSamples = reinterpret_cast<AliasedSamples&>(samples);
    DeviceWord *outVecPtr          = reinterpret_cast<DeviceWord *>(out);

    // if (blockIdx.x == 0 && threadIdx.x == 0 && __isLocal(aliasedSamples))
    // {
    //     printf("AliasedSamples is local!\n");
    // }

    // writing in striped fashion
    #pragma unroll
    for (int p = 0; p < POINTS_PER_THREAD; ++p)
    {
        *(outVecPtr + p * BLOCK_THREADS + threadIdx.x) = aliasedSamples[p];
    }
}

template <int         BLOCK_THREADS>
__global__ void kernel1(double const *in, double *out)
{
    const int blockOffset = blockIdx.x * BLOCK_THREADS * SAMPLES_PER_THREAD;
    double samples[POINTS_PER_THREAD][DIM];

    //loading in striped fashion
    loadTile1<BLOCK_THREADS>(in + blockOffset, samples);

    // #pragma unroll
    // for (int p = 0; p < POINTS_PER_THREAD; ++p)
    // {
    //     #pragma unroll
    //     for (int d = 0; d < DIM; ++d)
    //     {
    //         samples[p][d] *= 0.5;
    //     }
    // }

    // writing in striped fashion
    storeTile<BLOCK_THREADS>(out + blockOffset, samples);
}

template <int         BLOCK_THREADS>
__global__ void kernel2(double const *in, double *out)
{
    const int blockOffset = blockIdx.x * BLOCK_THREADS * SAMPLES_PER_THREAD;
    double samples[POINTS_PER_THREAD][DIM];

    //loading in striped fashion
    loadTile2<BLOCK_THREADS>(in + blockOffset, samples);

    // #pragma unroll
    // for (int p = 0; p < POINTS_PER_THREAD; ++p)
    // {
    //     #pragma unroll
    //     for (int d = 0; d < DIM; ++d)
    //     {
    //         samples[p][d] *= 0.5;
    //     }
    // }

    // writing in striped fashion
    storeTile<BLOCK_THREADS>(out + blockOffset, samples);
}

//-----------------------------------------------------------------------
//      TEST 
//-----------------------------------------------------------------------


void test()
{
    const int nPoints = 1 << 18;
    const int nElements = nPoints * POINTS_PER_THREAD * DIM;

    std::cout << "nPoints: " << nPoints;
    std::cout << "\nnElements: " << nElements;
    std::cout << "\nmemSize: " << nElements * sizeof(double) / 1024.0 / 1024.0 << "(MB)\n";

    double *d_in, *d_out;
    double *h_in, *h_out;

    GpuTimer timer;

    h_in = new double[nElements];
    h_out = new double[nElements];

    checkCudaErrors(cudaMalloc((void**)&d_in, nElements * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_out, nElements * sizeof(double)));

    for (int k = 0; k < nElements; ++k)
    {
        h_in[k] = k;
    }

    checkCudaErrors(cudaMemcpy(d_in, h_in, nElements * sizeof(double), cudaMemcpyHostToDevice));

    dim3 gridSize(1);
    const int blockThreads = 64;
    const int iterations = 100;
    gridSize.x = (nPoints + blockThreads - 1) / blockThreads;

    float avgMilis, gigaRate, gigaBandwidth;
    bool success = true;

    // warm-up
    kernel1<blockThreads><<<gridSize, blockThreads>>>(d_in, d_out);

    timer.Start();
    for (int i = 0; i < iterations; ++i)
    {
        kernel1<blockThreads><<<gridSize, blockThreads>>>(d_in, d_out);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    avgMilis = timer.ElapsedMillis() / float(iterations);
    checkCudaErrors(cudaDeviceSynchronize());

    gigaRate = nElements * 2 / avgMilis / 1000.0 / 1000.0;
    gigaBandwidth = gigaRate * sizeof(double);

    std::cout << "[kernel1] avg millis: " << avgMilis << ", " << gigaBandwidth << " GB/s...";

    checkCudaErrors(cudaMemcpy(h_out, d_out, nElements * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    for (int k = 0; k < nElements; ++k)
    {
        // double value = h_in[k] * 0.5;
        double value = h_in[k];
        if (h_out[k] != value)
        {
            success = false;
            std::cout << "ERROR!: is: " << h_out[k] << ", should be: " << value << "\n";
        }
    }

    if (success)
    {
        std::cout << "\tSUCCESS!\n";
    }

    //----------------------------------------------------------------------------------------
    // warm-up
    kernel2<blockThreads><<<gridSize, blockThreads>>>(d_in, d_out);

    timer.Start();
    for (int i = 0; i < iterations; ++i)
    {
        kernel2<blockThreads><<<gridSize, blockThreads>>>(d_in, d_out);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    avgMilis = timer.ElapsedMillis() / float(iterations);
    checkCudaErrors(cudaDeviceSynchronize());

    gigaRate = nElements * 2 / avgMilis / 1000.0 / 1000.0;
    gigaBandwidth = gigaRate * sizeof(double);

    std::cout << "[kernel2] avg millis: " << avgMilis << ", " << gigaBandwidth << " GB/s...";

    checkCudaErrors(cudaMemcpy(h_out, d_out, nElements * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    success = true;
    for (int k = 0; k < nElements; ++k)
    {
        // double value = h_in[k] * 0.5;
        double value = h_in[k];
        if (h_out[k] != value)
        {
            success = false;
            std::cout << "ERROR!: is: " << h_out[k] << ", should be: " << value << "\n";
        }
    }

    if (success)
    {
        std::cout << "\tSUCCESS!\n";
    }

    delete[] h_in;
    delete[] h_out;

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
}

int main()
{
    checkCudaErrors(deviceInit());

    test();

    checkCudaErrors(cudaDeviceReset());

    return 0;
}