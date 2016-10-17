/**
 * @file test_dynamic_vector.cu
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is conducted under the supervision of prof. dr hab. inż. Marek
 * Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */

#include "cub/test_util.h"
#include "cub/util_debug.cuh"
#include "cub/util_device.cuh"

#include "rd/gpu/block/block_dynamic_vector.cuh"
#include "rd/gpu/util/dev_samples_set.cuh"

#include "rd/utils/rd_params.hpp"

#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#include <iostream>
#include <omp.h>
#include <vector>
#include <string>
#include <cmath>

static const int POINTS_PER_THREAD  = 4;
static const int DIM                = 2;
static const int SAMPLES_PER_THREAD = POINTS_PER_THREAD * DIM;

static const int INIT_VEC_SIZE      = 50000;


//-----------------------------------------------------------------------
//      LOAD & STORE 
//-----------------------------------------------------------------------

template <
    int BLOCK_THREADS, 
    typename VectorT,
    typename T>
__device__ __forceinline__ void loadFullTile(
    int             globalTilePointsOffset,
    T const *       in,
    VectorT &       dynVec)
{
    enum
    {
        TILE_SAMPLES = BLOCK_THREADS * SAMPLES_PER_THREAD,
    };

    #ifdef RD_DEBUG
        if (threadIdx.x == 0)
        {
            _CubLog("load ---full--- tile offset: %d\n", globalTilePointsOffset);
        }
    #endif

    in += globalTilePointsOffset * DIM + threadIdx.x * DIM;

    T samples[POINTS_PER_THREAD][DIM];
    T * dynBuffer = dynVec.begin();

    #pragma unroll
    for (int p = 0; p < POINTS_PER_THREAD; ++p)
    {
        #pragma unroll
        for (int d = 0; d < DIM; ++d)
        {
            samples[p][d] = in[p * BLOCK_THREADS * DIM + d];
        }
    }

    dynVec.resize(dynVec.size() + TILE_SAMPLES);
    dynBuffer = dynVec.begin();

    int x = dynVec.size() + threadIdx.x * DIM;
    #pragma unroll
    for (int p = 0; p < POINTS_PER_THREAD; ++p)
    {
        #pragma unrolle
        for (int d = 0; d < DIM; ++d)
        {
            dynBuffer[x + p * BLOCK_THREADS * DIM + d] = samples[p][d];
        }
    }
    dynVec.incrementItemsCnt(TILE_SAMPLES, true);
}

template <int BLOCK_THREADS, typename T>
__device__ __forceinline__ void storeFullTile(
    int     globalTileOffset,
    int     blockTileOffset,
    T *     out,
    T *     dynBuffer)
{

    // #ifdef RD_DEBUG
        // if (threadIdx.x == 0)
        // {
        //     _CubLog("store global --full-- tile offset: %d, block tile offset: %d\n", globalTileOffset, blockTileOffset);
        // }
    // #endif

    out += globalTileOffset * DIM + threadIdx.x * DIM;
    dynBuffer += blockTileOffset * DIM + threadIdx.x * DIM;

    #pragma unroll
    for (int p = 0; p < POINTS_PER_THREAD; ++p)
    {
        #pragma unroll
        for (int d = 0; d < DIM; ++d)
        {
            out[p * BLOCK_THREADS * DIM + d] = dynBuffer[p * BLOCK_THREADS * DIM + d];
        }
    }
}

template <
    int BLOCK_THREADS, 
    typename VectorT,
    typename T>
__device__ __forceinline__ void loadPartialTile(
    int             globalTilePointsOffset,
    int             validPoints,
    T const *       in,
    VectorT &       dynVec)
{
    enum
    {
        TILE_SAMPLES = BLOCK_THREADS * SAMPLES_PER_THREAD,
    };

    #ifdef RD_DEBUG
        if (threadIdx.x == 0)
        {
            _CubLog("load ---partial--- tile offset: %d, validPoints: %d\n", globalTilePointsOffset, validPoints);
        }
    #endif

    in += globalTilePointsOffset * DIM + threadIdx.x * DIM;

    T samples[POINTS_PER_THREAD][DIM];
    T * dynBuffer = dynVec.begin();

    #pragma unroll
    for (int p = 0; p < POINTS_PER_THREAD; ++p)
    {
        if (p * BLOCK_THREADS + threadIdx.x < validPoints)
        {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                samples[p][d] = in[p * BLOCK_THREADS * DIM + d];
            }
        }
    }

    dynVec.resize(dynVec.size() + validPoints * DIM);
    dynBuffer = dynVec.begin();

    int x = dynVec.size() + threadIdx.x * DIM;
    #pragma unroll
    for (int p = 0; p < POINTS_PER_THREAD; ++p)
    {
        if (p * BLOCK_THREADS + threadIdx.x < validPoints)
        {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                dynBuffer[x + p * BLOCK_THREADS * DIM + d] = samples[p][d];
            }
        }
    }
    dynVec.incrementItemsCnt(validPoints, true);
}

template <int BLOCK_THREADS, typename T>
__device__ __forceinline__ void storePartialTile(
    int     globalTileOffset,
    int     blockTileOffset,
    int     validPoints,
    T *     out,
    T *     dynBuffer)
{
    out += globalTileOffset * DIM + threadIdx.x * DIM;
    dynBuffer += blockTileOffset * DIM + threadIdx.x  * DIM;

    // #ifdef RD_DEBUG
        // if (threadIdx.x == 0)
        // {
        //     _CubLog("store global --partial-- tile offset: %d, block tile offset: %d, validPoints: %d \n", globalTileOffset, blockTileOffset, validPoints);
        // }
    // #endif

    #pragma unroll
    for (int p = 0; p < POINTS_PER_THREAD; ++p)
    {
        if (p * BLOCK_THREADS + threadIdx.x < validPoints)
        {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                out[p * BLOCK_THREADS * DIM + d] = dynBuffer[p * BLOCK_THREADS * DIM + d];
            }
        }
    }
}

template <int BLOCK_THREADS, typename T>
static __global__ void kernel(T const *in, T *out, int size)
{
    enum
    {
        TILE_POINTS = BLOCK_THREADS * POINTS_PER_THREAD,
    };

    const int tileCount = (size + TILE_POINTS - 1) / TILE_POINTS;

    typedef rd::gpu::BlockDynamicVector<
        BLOCK_THREADS, 
        SAMPLES_PER_THREAD, 
        T,
        8,
        float> VectorT;

    // VectorT dynVec(INIT_VEC_SIZE, 2);
    VectorT dynVec({0.01f, 0.025f, 0.05f, 0.1f, 0.15f, 0.25f, 0.66f, 1.0f}, (float)size * DIM);
    // VectorT dynVec({ (unsigned int)(0.1f * size * DIM),
    //                  (unsigned int)(0.25f * size * DIM),
    //                  (unsigned int)(0.66f * size * DIM), 
    //                  (unsigned int)(1.0f * size * DIM)});

    for (int t = blockIdx.x; t < tileCount; t += gridDim.x)
    {
        int globalTileOffset = t * TILE_POINTS;


        if (globalTileOffset + TILE_POINTS > size)
        {
            loadPartialTile<BLOCK_THREADS>(globalTileOffset, size - globalTileOffset, in, dynVec);
        }
        else
        {
            loadFullTile<BLOCK_THREADS>(globalTileOffset, in, dynVec);
        }
    }
    __syncthreads();
    // dynVec.print(">>> Tiles loaded.", 0);

    for (int t = blockIdx.x, k = 0; t < tileCount; t += gridDim.x, ++k)
    {
        int globalTileOffset = t * TILE_POINTS;
        int blockTileOffset = k * TILE_POINTS;

        if (globalTileOffset + TILE_POINTS > size)
        {
            storePartialTile<BLOCK_THREADS>(globalTileOffset, blockTileOffset, size - globalTileOffset, out, dynVec.begin());
        }
        else
        {
            storeFullTile<BLOCK_THREADS>(globalTileOffset, blockTileOffset, out, dynVec.begin());
        }
    }

    __syncthreads();
    dynVec.clear();
}

//-----------------------------------------------------------------------
//      TEST 
//-----------------------------------------------------------------------


template <typename T>
void test()
{
    rd::RDParams<T> rdp;
    rd::RDSpiralParams<T> rds;

    rdp.dim = DIM;
    // rdp.np = size_t(2 * 1e4);
    // rds.a = 35;
    // rds.b = 25;
    // rds.sigma = 8;
    
    rds.loadFromFile = true;
    // rds.file = "segment6D_50K.txt";
    // rds.file = "segment6D_20K_ones.txt";
    // rds.file = "spiral3D_20K.txt";
    rds.file = "spiral3D_1M.txt";
    // rds.file = "spiral3D_100K.txt";

    std::cout << "generate data.. " << std::endl;
    std::vector<std::string> samplesDir{"../../examples/data/nd_segments/", "../../examples/data/spirals/"};
    rd::gpu::Samples<T> d_samplesSet(rdp, rds, samplesDir, DIM);
    checkCudaErrors(cudaDeviceSynchronize());

    unsigned int nPoints = rdp.np;
    unsigned int nElements = nPoints * DIM;

    std::cout << "nPoints: " << nPoints;
    std::cout << "\nnElements: " << nElements;
    std::cout << "\nmemSize: " << nElements * sizeof(T) / 1024.0 / 1024.0 << "(MB)\n";


    T *d_in, *h_in;
    T *d_out, *h_out;

    GpuTimer timer;

    std::cout << "allocate and copy data... " << std::endl;

    h_in = new T[nElements];
    h_out = new T[nElements];

    checkCudaErrors(cudaMalloc((void**)&d_in, nElements * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_out, nElements * sizeof(T)));

    checkCudaErrors(cudaMemcpy(d_in, d_samplesSet.samples_, nElements * sizeof(T), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(h_in, d_samplesSet.samples_, nElements * sizeof(T), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemset(d_out, 0, nElements * sizeof(T)));
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 gridSize(1);
    const int blockThreads = 64;
    const int iterations = 100;
    // gridSize.x = 4;

    int deviceOrdinal;
    checkCudaErrors(cudaGetDevice(&deviceOrdinal));

    // Get SM count
    int smCount;
    checkCudaErrors(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceOrdinal));

    typedef void (*KernelPtrT)(T const *, T *, int);
    KernelPtrT kernelPtr = kernel<blockThreads>;

    // get SM occupancy
    int smOccupancy;
    checkCudaErrors(cub::MaxSmOccupancy(
        smOccupancy,
        kernelPtr,
        const_cast<int&>(blockThreads))
    );
    gridSize.x = smCount * smOccupancy;

    printf("smCount: %d, smOccupancy: %d\n", smCount, smOccupancy);

    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize,
         nElements * 0.5f * gridSize.x * sizeof(T)));

    size_t deviceHeapSize = 0;
    checkCudaErrors(cudaDeviceGetLimit(&deviceHeapSize, cudaLimitMallocHeapSize));

    std::cout << "-- Device malloc heap size: " << deviceHeapSize / 1024 / 1024 << "(MB)\n";

    float avgMilis, gigaRate, gigaBandwidth;
    bool success = true;

    // check correctnes iteration
    printf("invoke correctness check, kernel<<<%d, %d>>>\n", gridSize.x, blockThreads);
    kernel<blockThreads><<<gridSize, blockThreads>>>(d_in, d_out, nPoints);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_out, d_out, nElements * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    #pragma omp parallel for schedule(static)
    for (unsigned int k = 0; k < nElements; ++k)
    {
        T value = h_in[k];
        if (h_out[k] != value)
        {
            success = false;
            printf("ERROR! is h_out[%d]: %f ---> should be: %f\n", k, h_out[k], value);
        }
    }

    if (success)
    {
        std::cout << "\tSUCCESS!\n";
    }
    else
    {
        std::cout << "---- INCORECT RESULTS!----- " << std::endl;
        // clean-up
        delete[] h_in;
        delete[] h_out;

        checkCudaErrors(cudaFree(d_in));
        checkCudaErrors(cudaFree(d_out));
        exit(1);
    }

    #if !defined(RD_DEBUG) && !defined(RD_PROFILE) 
    std::cout << "Measure performance... " << std::endl;

    // warm-up
    kernel<blockThreads><<<gridSize, blockThreads>>>(d_in, d_out, nPoints);
    timer.Start();
    for (int i = 0; i < iterations; ++i)
    {
        kernel<blockThreads><<<gridSize, blockThreads>>>(d_in, d_out, nPoints);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    avgMilis = timer.ElapsedMillis() / float(iterations);
    checkCudaErrors(cudaDeviceSynchronize());

    gigaRate = nElements * 4 / avgMilis / 1000.0 / 1000.0;
    gigaBandwidth = gigaRate * sizeof(T);

    printf("-----   avgMilis: %f, gigaBandwidth: %f\n", avgMilis, gigaBandwidth);
    std::cout.flush();
    #endif

    // clean-up
    delete[] h_in;
    delete[] h_out;

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
}

int main()
{
    checkCudaErrors(deviceInit());

    test<float>();

    checkCudaErrors(cudaDeviceReset());

    return 0;
}
