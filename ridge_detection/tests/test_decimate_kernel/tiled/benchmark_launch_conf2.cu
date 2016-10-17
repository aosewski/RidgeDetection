/**
 * @file benchmark_launch_conf2.cu
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 * 
 * In this file we're timing kernels in a 'standard way'. That is we have simple kernel 
 * performing only one requested operation. Additionally we have another auxilliary
 * kernel performing data initialization. We use events to measure gpu execution time.
 */

#include <helper_cuda.h>

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>

// #ifndef RD_DEBUG
// #define NDEBUG      // for disabling assert macro
// #endif 
#include <assert.h>

#if defined(RD_DEBUG) && !defined(CUB_STDERR)
#define CUB_STDERR 
#endif
 
#ifdef RD_PROFILE
#   include <cuda_profiler_api.h>
// #   include <nvToolsExt.h>
#endif

#include "rd/gpu/device/tiled/tiled_tree.cuh"
#include "rd/gpu/device/tiled/tree_drawer.cuh"
#include "rd/gpu/device/device_choose.cuh"
#include "rd/gpu/device/device_tiled_decimate.cuh"
#include "rd/gpu/agent/agent_memcpy.cuh"

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "tests/test_util.hpp"

#include "cub/test_util.h"

//----------------------------------------------
// global variables / constants
//----------------------------------------------

static constexpr int BUILD_TREE_BLOCK_THREADS      = 128;
static constexpr int BUILD_TREE_POINTS_PER_THREAD  = 6;
static constexpr int MAX_TEST_DIM       = 3;
static constexpr int MAX_POINTS_NUM     = int(1e7);
static constexpr int RD_CUDA_MAX_SYNC_DEPTH = 10;
static constexpr size_t HUNDRED_MB_IN_BYTES = 100 * 1024 * 1024;

static const std::string LOG_FILE_NAME_SUFFIX   = "g_decimate_lc_log.txt";

#if defined(RD_PROFILE) || defined(RD_DEBUG)
    const int g_iterations = 1;
#else
    const int g_iterations = 100;
#endif

static std::ofstream *  g_logFile           = nullptr;
static std::string      g_devName           = "";
static int              g_devId             = 0;
static bool             g_logPerfResults    = false;
static int              g_devClockRate      = 0;
/**
 * @brief      Create if necessary and open log file. Allocate log file stream.
 */
template <typename T>
static void initializeLogFile()
{
    if (g_logPerfResults)
    {
        std::ostringstream logFileName;
        logFileName << typeid(T).name() << "_" << getCurrDateAndTime() << "_" <<
            g_devName << "_" << getBinConfSuffix() << LOG_FILE_NAME_SUFFIX;

        std::string logFilePath = rd::findPath("timings/", logFileName.str());
        g_logFile = new std::ofstream(logFilePath.c_str(), std::ios::out | std::ios::app);
        if (g_logFile->fail())
        {
            throw std::logic_error("Couldn't open file: " + logFileName.str());
        }
    }
}

//------------------------------------------------------------------------

static void configureDevice(
    size_t neededMemSize)
{
    checkCudaErrors(cudaDeviceReset());
    checkCudaErrors(cudaSetDevice(g_devId));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, RD_CUDA_MAX_SYNC_DEPTH));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, neededMemSize));
}

//------------------------------------------------------------------------

template <
    int                     DIM,
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename                T>
__launch_bounds__ (128)
static __global__ void dataCopyKernel(
    T const * d_in,
    T * d_out,
    int numPoints,
    int startOffset,
    int inStride,
    int outStride)
{
    typedef rd::gpu::BlockTileLoadPolicy<
            128,
            8,
            cub::LOAD_CS>
        BlockTileLoadPolicyT;

    typedef rd::gpu::BlockTileStorePolicy<
            128,
            8,
            cub::STORE_DEFAULT>
        BlockTileStorePolicyT;

    typedef rd::gpu::AgentMemcpy<
            BlockTileLoadPolicyT,
            BlockTileStorePolicyT,
            DIM,
            MEM_LAYOUT,
            MEM_LAYOUT,
            rd::gpu::IO_BACKEND_CUB,
            int,
            T>
        AgentMemcpyT;

    #ifdef RD_DEBUG
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("dataCopyKernel: startOffset: %d, numPoints: %d, inStride: %d, outStride: %d\n",
            startOffset, numPoints, inStride, outStride);
    }
    #endif

    AgentMemcpyT agent(d_in, d_out);
    agent.copyRange(startOffset, numPoints, inStride, outStride);
}

//------------------------------------------------------------------------

template <
    int                             DIM,
    rd::DataMemoryLayout            IN_MEM_LAYOUT,
    rd::DataMemoryLayout            OUT_MEM_LAYOUT,
    typename                        T>
struct TileProcessOp
{
    T r1;

    __device__ __forceinline__ TileProcessOp(
        T r1)
    :
        r1(r1)
    {}

    template <typename NodeT>
    __device__ __forceinline__ void operator()(NodeT * node) const
    {
        __syncthreads();
        #ifdef RD_DEBUG
        if (threadIdx.x == 0)
        {
            _CubLog("*** TileProcessOp ***** node id: %d, "
                "chosenPointsCapacity: %d, chosenSamples: [%p -- %p]\n",
                node->id, node->chosenPointsCapacity, node->chosenSamples,
                node->chosenSamples + node->chosenSamplesStride * DIM);
        }
        #endif

        if (threadIdx.x == 0)
        {
            int * d_tileChosenPtsNum = new int();
            assert(d_tileChosenPtsNum != nullptr);

            *d_tileChosenPtsNum = 0;
            // choose points within this tile
            cudaError_t err = cudaSuccess;
            err = rd::gpu::bruteForce::DeviceChoose::choose<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
                    node->samples, node->chosenSamples, node->pointsCnt, d_tileChosenPtsNum,
                    r1, node->samplesStride, node->chosenSamplesStride);
            rdDevCheckCall(err);
            rdDevCheckCall(cudaDeviceSynchronize());

            node->chosenPointsCnt = *d_tileChosenPtsNum;
            delete d_tileChosenPtsNum;
        }
        __syncthreads();
        // #ifdef RD_DEBUG
        if (threadIdx.x == 0)
        {
            _CubLog("*** TileProcessOp ***** node id: %d, chosenPointsCnt: %d\n",
                node->id, node->chosenPointsCnt);
        }
        // #endif
    }
};

//------------------------------------------------------------------------
// Test kernels
//------------------------------------------------------------------------

template <
    typename                TiledTreeT,
    typename                TileProcessOpT,
    int                     DIM,
    typename                T>
__launch_bounds__ (1)
static __global__ void buildTreeKernel(
    TiledTreeT *                        initTree,
    TiledTreeT *                        workTree,
    T const *                           inputPoints,
    int                                 pointsNum,
    rd::gpu::BoundingBox<DIM, T> *      d_globalBBox,
    int                                 maxTileCapacity,
    T                                   sphereRadius,
    T                                   extensionFactor,
    cub::ArrayWrapper<int, DIM> const   initTileCntPerDim,
    int                                 inPtsStride)
{
    // last arg-> debugSynchronous
    new(initTree) TiledTreeT(maxTileCapacity, sphereRadius, extensionFactor, false);

    TileProcessOpT tileProcessOp(sphereRadius);
    cudaStream_t buildTreeStream;
    rdDevCheckCall(cudaStreamCreateWithFlags(&buildTreeStream, cudaStreamNonBlocking));

    rdDevCheckCall(initTree->buildTree(
        inputPoints, pointsNum, initTileCntPerDim, d_globalBBox, tileProcessOp, buildTreeStream, 
        inPtsStride));
    rdDevCheckCall(cudaStreamDestroy(buildTreeStream));
    rdDevCheckCall(cudaDeviceSynchronize());

    initTree->clone(workTree);
}

template <
    typename                TiledTreeT,
    // int                     BLOCK_SIZE,
    rd::DataMemoryLayout    OUT_MEM_LAYOUT,
    int                     DIM>
__launch_bounds__ (1)
static __global__ void initTreeKernel(
    TiledTreeT *    initTree,
    TiledTreeT *    workTree)
{
    typedef typename TiledTreeT::NodeT NodeT;
     // allocate table for pointers to leaf nodes 
    NodeT **d_initTreeLeafs = new NodeT*[*initTree->d_leafCount];
    assert(d_initTreeLeafs != nullptr);
    NodeT **d_workTreeLeafs = new NodeT*[*initTree->d_leafCount];
    assert(d_workTreeLeafs != nullptr);
    
    int initTreeleafCounter = 0;
    // collect and initialize leafs
    initTree->forEachNodePreorder(
        [&d_initTreeLeafs, &initTreeleafCounter](NodeT * node) {
        // am I a non-empty leaf?
        if (!node->haveChildren() && !node->empty())
        {
            d_initTreeLeafs[initTreeleafCounter++] = node;
            // init evolve flag
            node->needEvolve = 1;
            #ifdef RD_DEBUG
            {
                // printf("node: %d, chosenPointsCnt: %d\n", node->id, node->chosenPointsCnt);
                printf("node: %d, pointsCnt: %d, pointsStride: %d, chosenPtsCnt: %d, "
                    "chPtsStride: %d, neighboursCnt: %d, neighboursStride: %d\n",
                    node->id, node->pointsCnt, node->samplesStride, node->chosenPointsCnt,
                    node->chosenSamplesStride, node->neighboursCnt, node->neighboursStride);
            }
            #endif
        }
    });

    int workTreeLeafCounter = 0;
    workTree->forEachNodePreorder(
        [&d_workTreeLeafs, &workTreeLeafCounter](NodeT * node) {
        // am I a non-empty leaf?
        if (!node->haveChildren() && !node->empty())
        {
            d_workTreeLeafs[workTreeLeafCounter++] = node;
            
            // init evolve flag
            node->needEvolve = 1;
            #ifdef RD_DEBUG
            {
                printf("node: %d, chosenPointsCnt: %d\n", node->id, node->chosenPointsCnt);
            }
            #endif
        }
    });

    // restore initial state
    for (int k = 0; k < workTreeLeafCounter; ++k)
    {
        cudaStream_t copyChosenSamplesStream;
        rdDevCheckCall(cudaStreamCreateWithFlags(&copyChosenSamplesStream, 
            cudaStreamNonBlocking));
        
        dim3 copyChosenSamplesGrid(1);
        copyChosenSamplesGrid.x = (((d_initTreeLeafs[k]->chosenPointsCnt + 7) / 
            8) + 127) / 128;
        
        #ifdef RD_DEBUG
        printf("Invoke dataCopyKernel <<<%d, 128, 0, %p>>> Copying data from leaf id: %d"
            " to leaf id: %d\n",
            copyChosenSamplesGrid.x, copyChosenSamplesStream, d_initTreeLeafs[k]->id,
            d_workTreeLeafs[k]->id);
        #endif

        dataCopyKernel<DIM, OUT_MEM_LAYOUT><<<copyChosenSamplesGrid, 128, 0, 
                copyChosenSamplesStream>>>(
            d_initTreeLeafs[k]->chosenSamples, d_workTreeLeafs[k]->chosenSamples, 
            d_initTreeLeafs[k]->chosenPointsCnt, 0, d_initTreeLeafs[k]->chosenSamplesStride, 
            d_workTreeLeafs[k]->chosenSamplesStride);
        rdDevCheckCall(cudaPeekAtLastError());
        
        rdDevCheckCall(cudaStreamDestroy(copyChosenSamplesStream));
        #ifdef RD_DEBUG
            rdDevCheckCall(cudaDeviceSynchronize());
        #endif
    }
    rdDevCheckCall(cudaDeviceSynchronize());

    delete[] d_initTreeLeafs;
    delete[] d_workTreeLeafs;
}

template <
    typename                TiledTreeT,
    int                     BLOCK_SIZE,
    rd::DataMemoryLayout    OUT_MEM_LAYOUT,
    int                     DIM,
    typename                T>
__launch_bounds__ (1)
static __global__ void globalDecimateKernel(
    TiledTreeT *    workTree,
    T               sphereRadius2)
{
    typedef typename TiledTreeT::NodeT NodeT;

    // TiledTreeT * workTree = initTree->clone();
    #if RD_DEBUG
    printf("\n------- END CLONE! ------ \n");
    #endif

    // allocate table for pointers to leaf nodes 
    NodeT **d_workTreeLeafs = new NodeT*[*workTree->d_leafCount];
    assert(d_workTreeLeafs != nullptr);
    
    int * d_chosenPointsNum = new int();
    assert(d_chosenPointsNum != nullptr);

    int workTreeLeafCounter = 0;
    workTree->forEachNodePreorder(
        [d_chosenPointsNum, &d_workTreeLeafs, &workTreeLeafCounter](NodeT * node) {
        // am I a non-empty leaf?
        if (!node->haveChildren() && !node->empty())
        {
            d_workTreeLeafs[workTreeLeafCounter++] = node;
            atomicAdd(d_chosenPointsNum, node->chosenPointsCnt);
            // init evolve flag
            node->needEvolve = 1;
            #ifdef RD_DEBUG
            {
                printf("node: %d, chosenPointsCnt: %d\n", node->id, node->chosenPointsCnt);
            }
            #endif
        }
    });
    
    int initChosenPointsNum = *d_chosenPointsNum;
    #ifdef RD_DEBUG
    {
        _CubLog("\n>>>>> ---------globalDecimate---------\n chosenPointsNum: %d\n",
            initChosenPointsNum);
    }
    #endif

    // restore initial state
    *d_chosenPointsNum = initChosenPointsNum;
    rd::gpu::tiled::detail::deviceDecimateKernel<DIM, BLOCK_SIZE, OUT_MEM_LAYOUT>
        <<<1, BLOCK_SIZE>>>(d_workTreeLeafs, workTreeLeafCounter, d_chosenPointsNum, 
            sphereRadius2);

    // Check for failure to launch
    rdDevCheckCall(cudaPeekAtLastError());
    rdDevCheckCall(cudaDeviceSynchronize());

    delete[] d_workTreeLeafs;
    delete d_chosenPointsNum;
}

template <
    typename TiledTreeT>
__launch_bounds__ (1)
static __global__ void deleteTiledTree(
    TiledTreeT *          tree)
{
    tree->~TiledTreeT();
}

//------------------------------------------------------------------------

template <
    typename TiledTreeT,
    rd::DataMemoryLayout MEM_LAYOUT,
    int DIM>
static float timeInitKernel(
    TiledTreeT * initTree,
    TiledTreeT * workTree)
{
    GpuTimer timer;
    float kernelTime;

    timer.Start();
    for (int i = 0; i < g_iterations; ++i)
    {
        initTreeKernel<TiledTreeT, MEM_LAYOUT, DIM><<<1,1>>>(initTree, workTree);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    return kernelTime;
}

template <
    typename TiledTreeT,
    rd::DataMemoryLayout MEM_LAYOUT,
    int BLOCK_SIZE,
    int DIM,
    typename T>
static float timeGlobalDecimateKernel(
    TiledTreeT * initTree,
    TiledTreeT * workTree,
    T sphereRadius2,
    float initTime)
{
    GpuTimer timer;
    float kernelTime;

    timer.Start();
    for (int i = 0; i < g_iterations; ++i)
    {
        initTreeKernel<TiledTreeT, MEM_LAYOUT, DIM><<<1,1>>>(
            initTree, workTree);
        checkCudaErrors(cudaGetLastError());
        globalDecimateKernel<TiledTreeT, BLOCK_SIZE, MEM_LAYOUT, DIM><<<1,1>>>(
            workTree, sphereRadius2);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    return (kernelTime - initTime) / float(g_iterations);
}

template <
    int DIM,
    int BLOCK_SIZE,
    typename TiledTreeT_RR,
    typename TiledTreeT_CR,
    typename TiledTreeT_CC,
    typename T>
void benchmark(
    TiledTreeT_RR *     d_initTree_RR,
    TiledTreeT_RR *     d_workTree_RR,
    TiledTreeT_CR *     d_initTree_CR,
    TiledTreeT_CR *     d_workTree_CR,
    TiledTreeT_CC *     d_initTree_CC,
    TiledTreeT_CC *     d_workTree_CC,
    float           initTime_RR,
    float           initTime_CR,
    float           initTime_CC,
    T               sphereRadius2)
{
    float avgKernelTime = 0;

    logValue(std::cout, BLOCK_SIZE);
    if (g_logPerfResults)
    {
        logValue(*g_logFile, BLOCK_SIZE);
    }

    #ifdef RD_DEBUG
        std::cout << "Invoking globalDecimateKernel<TiledTreeT_CR, " << BLOCK_SIZE << ", COL, ROW, "
            << DIM << ">" << std::endl;
    #endif

    avgKernelTime = timeGlobalDecimateKernel<TiledTreeT_CR, rd::ROW_MAJOR, BLOCK_SIZE, DIM>(
        d_initTree_CR, d_workTree_CR, sphereRadius2, initTime_CR);

    logValue(std::cout, avgKernelTime);
    if (g_logPerfResults)
    {
        logValue(*g_logFile, avgKernelTime);
    }

    #ifdef RD_DEBUG
        std::cout << "Invoking globalDecimateKernel<TiledTreeT_RR, " << BLOCK_SIZE << ", ROW, ROW, "
                << DIM << ">" << std::endl;
    #endif

    avgKernelTime = timeGlobalDecimateKernel<TiledTreeT_RR, rd::ROW_MAJOR, BLOCK_SIZE, DIM>(
        d_initTree_RR, d_workTree_RR, sphereRadius2, initTime_RR);

    logValue(std::cout, avgKernelTime);
    if (g_logPerfResults)
    {
        logValue(*g_logFile, avgKernelTime);
    }

    #ifdef RD_DEBUG
        std::cout << "Invoking globalDecimateKernel<TiledTreeT_CC, " << BLOCK_SIZE << ", COL, COL, "
                << DIM << ">" << std::endl;
    #endif

    avgKernelTime = timeGlobalDecimateKernel<TiledTreeT_CC, rd::COL_MAJOR, BLOCK_SIZE, DIM>(
        d_initTree_CC, d_workTree_CC, sphereRadius2, initTime_CC);

    logValue(std::cout, avgKernelTime);
    if (g_logPerfResults)
    {
        logValue(*g_logFile, avgKernelTime);
        *g_logFile << "\n";
    }
    std::cout << std::endl;
}

template <
    int DIM,
    typename TiledTreeT_RR,
    typename TiledTreeT_CR,
    typename TiledTreeT_CC,
    typename T>
void iterateKernelConf(
    TiledTreeT_RR *     d_initTree_RR,
    TiledTreeT_RR *     d_workTree_RR,
    TiledTreeT_CR *     d_initTree_CR,
    TiledTreeT_CR *     d_workTree_CR,
    TiledTreeT_CC *     d_initTree_CC,
    TiledTreeT_CC *     d_workTree_CC,
    float           initTime_RR,
    float           initTime_CR,
    float           initTime_CC,
    T                   sphereRadius2)
{
    #ifdef QUICK_TEST
        benchmark<DIM, 256>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
    #else
        benchmark<DIM,   64>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,   96>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  128>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  160>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  192>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  224>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  256>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  288>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  320>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  352>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  384>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  416>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  448>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  480>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  512>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  544>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  576>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  608>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  640>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  672>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  704>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  736>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  768>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  800>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  832>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  864>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  896>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  928>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  960>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM,  992>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
        benchmark<DIM, 1024>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_CC, sphereRadius2);
    #endif
}

template <
    int DIM,
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    typename TiledTreeT,
    typename T>
void buildTree(
    T const *   h_inputPoints,
    int         pointsNum,
    T           sphereRadius, 
    TiledTreeT *d_initTree, 
    TiledTreeT *d_workTree) 
{
    typedef TileProcessOp<
        DIM,
        IN_MEM_LAYOUT,
        OUT_MEM_LAYOUT,
        T>
    TileProcessOpT;

    int     maxTileCapacity = 0.17 * pointsNum;
    T       extensionFactor = 1.35;
    T *     d_inputPoints;
    cub::ArrayWrapper<int, DIM> initTileCntPerDim;
    int inPtsStride = DIM;

    if (IN_MEM_LAYOUT == rd::ROW_MAJOR)
    {
        checkCudaErrors(cudaMalloc(&d_inputPoints, pointsNum * DIM * sizeof(T)));
    }
    else if (IN_MEM_LAYOUT == rd::COL_MAJOR)
    {
        size_t pitch;
        checkCudaErrors(cudaMallocPitch(&d_inputPoints, &pitch, pointsNum * sizeof(T),
            DIM));
        inPtsStride = pitch / sizeof(T);
    }
    else
    {
        throw std::runtime_error("Unsupported memory layout!");
    }
    
    // Hardcoded!
    for (int k =0; k < DIM; ++k)
    {
        initTileCntPerDim.array[k] = 2;
        // initTileCntPerDim.array[k] = 1;
    }

    if (IN_MEM_LAYOUT == rd::ROW_MAJOR)
    {
        rd::gpu::rdMemcpy<rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_inputPoints, h_inputPoints, DIM, pointsNum, DIM, inPtsStride);
    }
    else if (IN_MEM_LAYOUT == rd::COL_MAJOR)
    {

        rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_inputPoints, h_inputPoints, DIM, pointsNum, inPtsStride * sizeof(T),
            DIM * sizeof(T));
    }
    else
    {
        throw std::runtime_error("Unsupported memory layout!");
    }

    checkCudaErrors(cudaDeviceSynchronize());

    rd::gpu::BoundingBox<DIM, T> globalBBox;
    checkCudaErrors(globalBBox.template findBounds<IN_MEM_LAYOUT>(
        d_inputPoints, pointsNum, inPtsStride));
    globalBBox.calcDistances();

    // allocate & copy memory for device global bounding box
    rd::gpu::BoundingBox<DIM, T> *d_globalBBox;
    checkCudaErrors(cudaMalloc(&d_globalBBox, sizeof(rd::gpu::BoundingBox<DIM,T>)));
    checkCudaErrors(cudaMemcpy(d_globalBBox, &globalBBox, sizeof(rd::gpu::BoundingBox<DIM,T>), 
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Invoking buildTreeKernel" << std::endl;

    buildTreeKernel<TiledTreeT, TileProcessOpT, DIM><<<1,1>>>(
        d_initTree, d_workTree, d_inputPoints, pointsNum, d_globalBBox, maxTileCapacity, 
        sphereRadius, extensionFactor, initTileCntPerDim, inPtsStride);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // draw initial chosen points
    // #ifdef QUICK_TEST
    // rd::gpu::tiled::util::TreeDrawer<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, TiledTreeT, T> treeDrawer(
    //     d_tree, d_inputPoints, pointsNum, inPtsStride);
    // treeDrawer.drawBounds();
    // treeDrawer.drawEachTile();
    // #endif

    //-------------------------------------------------------------------------------
    // clean-up
    //-------------------------------------------------------------------------------

    checkCudaErrors(cudaFree(d_inputPoints));
    checkCudaErrors(cudaFree(d_globalBBox));
}

template <
    int DIM,
    typename T>
void testMemLayout(
    int pointNum,
    PointCloud<T> const & pc)
{
    std::vector<T> && points = pc.extractPart(pointNum, DIM);
    T sphereRadius = 1.45f * pc.stddev_;
    T sphereRadius2 = 2.f * sphereRadius;

    typedef rd::gpu::tiled::TiledTreePolicy<
        BUILD_TREE_BLOCK_THREADS,
        BUILD_TREE_POINTS_PER_THREAD,
        cub::LOAD_LDG,
        rd::gpu::IO_BACKEND_CUB>
    TiledTreePolicyT;

    //-----------------------------------------------------------------
    // build trees in required layouts
    //-----------------------------------------------------------------

    typedef rd::gpu::tiled::TiledTree<
        TiledTreePolicyT, 
        DIM, 
        rd::ROW_MAJOR,
        rd::ROW_MAJOR,
        T>
    TiledTreeT_RR;

    TiledTreeT_RR * d_initTree_RR = nullptr;
    TiledTreeT_RR * d_workTree_RR = nullptr;
    checkCudaErrors(cudaMalloc(&d_initTree_RR, sizeof(TiledTreeT_RR)));
    checkCudaErrors(cudaMalloc(&d_workTree_RR, sizeof(TiledTreeT_RR)));

    std::cout << rd::HLINE << std::endl;
    std::cout << "<<<<< ROW_MAJOR, ROW_MAJOR >>>>>" << std::endl;
    buildTree<DIM, rd::ROW_MAJOR, rd::ROW_MAJOR>(points.data(), pointNum, sphereRadius, 
        d_initTree_RR, d_workTree_RR);
    float initTime_RR = timeInitKernel<TiledTreeT_RR, rd::ROW_MAJOR, DIM>(d_initTree_RR, 
        d_workTree_RR);

    typedef rd::gpu::tiled::TiledTree<
        TiledTreePolicyT, 
        DIM, 
        rd::COL_MAJOR, 
        rd::ROW_MAJOR, 
        T>
    TiledTreeT_CR;

    TiledTreeT_CR * d_initTree_CR = nullptr;
    TiledTreeT_CR * d_workTree_CR = nullptr;
    checkCudaErrors(cudaMalloc(&d_initTree_CR, sizeof(TiledTreeT_CR)));
    checkCudaErrors(cudaMalloc(&d_workTree_CR, sizeof(TiledTreeT_CR)));

    std::cout << rd::HLINE << std::endl;
    std::cout << "<<<<< COL_MAJOR, ROW_MAJOR >>>>>" << std::endl;
    buildTree<DIM, rd::COL_MAJOR, rd::ROW_MAJOR>(points.data(), pointNum, sphereRadius, 
        d_initTree_CR, d_workTree_CR);
    float initTime_CR = timeInitKernel<TiledTreeT_CR, rd::ROW_MAJOR, DIM>(d_initTree_CR, 
        d_workTree_CR);

    typedef rd::gpu::tiled::TiledTree<
        TiledTreePolicyT, 
        DIM, 
        rd::COL_MAJOR,
        rd::COL_MAJOR,
        T>
    TiledTreeT_CC;

    TiledTreeT_CC * d_initTree_CC = nullptr;
    TiledTreeT_CC * d_workTree_CC = nullptr;
    checkCudaErrors(cudaMalloc(&d_initTree_CC, sizeof(TiledTreeT_CC)));
    checkCudaErrors(cudaMalloc(&d_workTree_CC, sizeof(TiledTreeT_CC)));

    std::cout << rd::HLINE << std::endl;
    std::cout << "<<<<< COL_MAJOR, COL_MAJOR >>>>>" << std::endl;
    buildTree<DIM, rd::COL_MAJOR, rd::COL_MAJOR>(points.data(), pointNum, sphereRadius, 
        d_initTree_CC, d_workTree_CC);
    float initTime_CC = timeInitKernel<TiledTreeT_CC, rd::COL_MAJOR, DIM>(d_initTree_CC, 
        d_workTree_CC);

    //-----------------------------------------------------------------
    // benchmark global decimate kernel
    //-----------------------------------------------------------------

    iterateKernelConf<DIM>(d_initTree_RR, d_workTree_RR, d_initTree_CR, d_workTree_CR, 
        d_initTree_CC, d_workTree_CC, initTime_RR, initTime_CR, initTime_RR, sphereRadius2);

    //-----------------------------------------------------------------
    // clean-up
    //-----------------------------------------------------------------

    deleteTiledTree<TiledTreeT_CR><<<1, 1>>>(d_initTree_CR);
    checkCudaErrors(cudaGetLastError());
    deleteTiledTree<TiledTreeT_CR><<<1, 1>>>(d_workTree_CR);
    checkCudaErrors(cudaGetLastError());

    deleteTiledTree<TiledTreeT_RR><<<1, 1>>>(d_initTree_RR);
    checkCudaErrors(cudaGetLastError());
    deleteTiledTree<TiledTreeT_RR><<<1, 1>>>(d_workTree_RR);
    checkCudaErrors(cudaGetLastError());

    deleteTiledTree<TiledTreeT_CC><<<1, 1>>>(d_initTree_CC);
    checkCudaErrors(cudaGetLastError());
    deleteTiledTree<TiledTreeT_CC><<<1, 1>>>(d_workTree_CC);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
 
    checkCudaErrors(cudaFree(d_initTree_RR));
    checkCudaErrors(cudaFree(d_initTree_CR));
    checkCudaErrors(cudaFree(d_initTree_CC));
    checkCudaErrors(cudaFree(d_workTree_RR));
    checkCudaErrors(cudaFree(d_workTree_CR));
    checkCudaErrors(cudaFree(d_workTree_CC));
}

/**
 * @brief helper structure for static for loop over dimension
 */
struct IterateDimensions
{
    template <typename D, typename T>
    static void impl(
        D   idx,
        int pointNum,
        PointCloud<T> const & pc)
    {
        std::cout << rd::HLINE << std::endl;
        std::cout << ">>>> Dimension: " << idx << "D\n";

        if (g_logPerfResults)
        {
            T a, b;
            pc.getCloudParameters(a, b);
            *g_logFile << "%>>>> Dimension: " << idx << "D\n"
                << "% a: " << a << " b: " << b << " s: " << pc.stddev_ 
                << " pointsNum: " << pointNum << "\n";
        }

        testMemLayout<D::value>(pointNum, pc);

        if (g_logPerfResults)
        {
            g_logFile->flush();
        }
    }
};

template <
    int          DIM,
    typename     T>
struct TestDimensions
{
    static void impl(
        PointCloud<T> & pc,
        int pointNum)
    {
        static_assert(DIM != 0, "DIM equal to zero!\n");

        initializeLogFile<T>();
        pc.pointCnt_ = pointNum;
        pc.initializeData();

        size_t neededMemSize = 10 * pointNum * DIM * sizeof(T);
        neededMemSize = std::max(HUNDRED_MB_IN_BYTES, neededMemSize);

        std::cout << "Reserve " << float(neededMemSize) / 1024.f / 1024.f 
            << " Mb for malloc heap size" << std::endl;
        configureDevice(neededMemSize);

        std::cout << rd::HLINE << std::endl;
        std::cout << ">>>> Dimension: " << DIM << "D\n";

        if (g_logPerfResults)
        {
            T a, b;
            pc.getCloudParameters(a, b);
            *g_logFile << "%>>>> Dimension: " << DIM << "D\n"
                << "% a: " << a << " b: " << b << " s: " << pc.stddev_ 
                << " pointsNum: " << pointNum << "\n";
        }

        testMemLayout<DIM>(pointNum, pc);

        // clean-up
        if (g_logPerfResults)
        {
            g_logFile->close();
            delete g_logFile;
        }
    }
};

template <typename T>
struct TestDimensions<0, T>
{
    static void impl(
        PointCloud<T> & pc,
        int pointNum)
    {
        initializeLogFile<T>();
        pc.pointCnt_ = pointNum;
        pc.dim_ = MAX_TEST_DIM;
        pc.initializeData();

        size_t neededMemSize = 23 * pointNum * MAX_TEST_DIM * sizeof(T);
        neededMemSize = std::max(HUNDRED_MB_IN_BYTES, neededMemSize);

        std::cout << "Reserve " << float(neededMemSize) / 1024.f / 1024.f 
            << " Mb for malloc heap size" << std::endl;
        configureDevice(neededMemSize);

        StaticFor<1, MAX_TEST_DIM, IterateDimensions>::impl(pointNum, pc);

        // clean-up
        if (g_logPerfResults)
        {
            g_logFile->close();
            delete g_logFile;
        }
    }
};

template <
    typename    T,
    int         DIM = 0,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testSize(
    PointCloud<T> & pc,
    int pointNum = -1)
{
    if (pointNum > 0)
    {
        TestDimensions<DIM, T>::impl(pc, pointNum);
    }
    else
    {
        for (int k = 1000; k <= MAX_POINTS_NUM; k *= 10)
        {
            std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t pointNum: " << k  
                    << "\n//------------------------------------------\n";

            TestDimensions<DIM, T>::impl(pc, k);
        }
    }
}

int main(int argc, char const **argv)
{
    float a = -1.f, b = -1.f, stddev = -1.f;
    int pointNum = -1;

    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help")) 
    {
        printf("%s \n"
            "\t\t[--a <a parameter of spiral or length if dim > 3>]\n"
            "\t\t[--b <b parameter of spiral or ignored if dim > 3>]\n"
            "\t\t[--stddev <standard deviation of generated samples>]\n"
            "\t\t[--size <number of points>]\n"
            "\t\t[--log <log performance results>]\n"
            "\t\t[--d=<device id>]\n"
            "\n", argv[0]);
        exit(0);
    }

    if (args.CheckCmdLineFlag("a"))
    {
        args.GetCmdLineArgument("a", a);
    }
    if (args.CheckCmdLineFlag("b"))
    {
        args.GetCmdLineArgument("b", b);
    }
    if (args.CheckCmdLineFlag("stddev"))
    {
        args.GetCmdLineArgument("stddev", stddev);
    }
    if (args.CheckCmdLineFlag("size"))
    {
        args.GetCmdLineArgument("size", pointNum);
    }    
    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", g_devId);
    }
    if (args.CheckCmdLineFlag("log")) 
    {
        g_logPerfResults = true;
    }

    checkCudaErrors(deviceInit(g_devId));

    // set device name for logging and drawing purposes
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, g_devId));
    g_devName = devProp.name;
    g_devClockRate = devProp.clockRate;

    if (pointNum    < 0 ||
        a           < 0 ||
        b           < 0 ||
        stddev      < 0)
    {
        std::cout << "Have to specify parameters! Rerun with --help for help.\n";
        exit(1);
    }
    #ifdef QUICK_TEST

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (spiral) float: "  
                    << "\n//------------------------------------------\n";
        // PointCloud<float> && fpc = SpiralPointCloud<float>(a, b, pointNum, 2, stddev);
        // TestDimensions<2, float>::impl(fpc, pointNum);

        PointCloud<float> && fpc = SegmentPointCloud<float>(a, pointNum, 3, stddev);
        TestDimensions<3, float>::impl(fpc, pointNum);

        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t (spiral) double: "  
        //             << "\n//------------------------------------------\n";
        // PointCloud<double> && dpc = SpiralPointCloud<double>(a, b, pointNum, 2, stddev);
        // TestDimensions<2, double>::impl(dpc, pointNum);

    #else
        // 1e6 2D points, spiral a=22, b=10, stddev=4
        #ifndef RD_DOUBLE_PRECISION        

        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t (spiral) float: "  
        //             << "\n//------------------------------------------\n";
        // PointCloud<float> && fpc2d = SpiralPointCloud<float>(22.f, 10.f, 0, 2, 4.f);
        // PointCloud<float> && fpc3d = SpiralPointCloud<float>(22.f, 10.f, 0, 3, 4.f);
        // TestDimensions<2, float>::impl(fpc2d, int(1e6));
        // TestDimensions<3, float>::impl(fpc3d, int(1e6));

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) float: "  
                    << "\n//------------------------------------------\n";
        PointCloud<float> && fpc2 = SegmentPointCloud<float>(a, pointNum, 0, stddev);
        testSize<float>(fpc2);
        #else

        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t (spiral) double: "  
        //             << "\n//------------------------------------------\n";
        // PointCloud<double> && dpc2d = SpiralPointCloud<double>(22.0, 10.0, 0, 2, 4.0);
        // PointCloud<double> && dpc3d = SpiralPointCloud<double>(22.0, 10.0, 0, 3, 4.0);
        // TestDimensions<2, double>::impl(dpc2d, int(1e6));
        // TestDimensions<3, double>::impl(dpc3d, int(1e6));

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) double: "  
                    << "\n//------------------------------------------\n";
        PointCloud<double> && dpc2 = SegmentPointCloud<double>(a, pointNum, 0, stddev);
        testSize<double>(dpc2);
        #endif
    #endif

    checkCudaErrors(cudaDeviceReset());
    
    std::cout << rd::HLINE << std::endl;
    std::cout << "END!" << std::endl;
 
    return EXIT_SUCCESS;
}

