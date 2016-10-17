/**
 * @file benchmark_select_if_mem_dv.cu
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

#define BLOCK_TILE_LOAD_V4 1

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/memory.h" 
#include "rd/utils/name_traits.hpp"
#include "rd/utils/rd_params.hpp"

#include "rd/gpu/block/block_select_if_dv.cuh"
#include "rd/gpu/util/dev_samples_set.cuh"
#include "rd/gpu/util/dev_utilities.cuh"
#include "rd/gpu/util/dev_memcpy.cuh"

#include "tests/test_util.hpp"
#include "cub/test_util.h"
#include "cub/util_macro.cuh"

#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda_profiler_api.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <limits>

#include <cmath>
#include <functional>
#include <algorithm>
#include <utility>
#include <tuple>

#ifdef RD_USE_OPENMP
#include <omp.h>
#endif

#ifdef RD_DEBUG
#define CUB_STDERR
#endif

//------------------------------------------------------------
//  GLOBAL CONSTANTS / VARIABLES
//------------------------------------------------------------

static const std::string LOG_FILE_NAME_SUFFIX = "_select_if_mdv-timings.txt";
static const int SUBSCRIPTION_FACTOR = 1;

std::ofstream * g_logFile       = nullptr;
bool            g_logResults    = false;
bool            g_drawGraphs    = false;
std::string     g_devName;
int             g_devSMCount        = 0;
int             g_devMaxBlocksPerSM = 0;
int             g_devMaxBlockCnt    = 0;
bool            g_verbose           = false;

// forward declaration
struct KernelParametersConf;

// (selectRatio, DIM, kernelConf)
typedef std::tuple<float, unsigned int, KernelParametersConf> GraphDataT;

std::vector<GraphDataT>         g_graphData;

#if defined(RD_PROFILE) || defined(RD_DEBUG)
static const int g_iterations = 2;
#else
static const int g_iterations = 100;
#endif

//------------------------------------------------------------
//  Benchmark helper structures
//------------------------------------------------------------


struct KernelParametersConf
{
    int                         BLOCK_THREADS;
    int                         POINTS_PER_THREAD;
    int                         DIM;
    cub::CacheLoadModifier      LOAD_MODIFIER;
    rd::gpu::BlockTileIOBackend IO_BACKEND;
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT;
    bool                        STORE_TWO_PHASE;
    float                       avgMillis;
    float                       gigaBandwidth;

    KernelParametersConf()
    :
        BLOCK_THREADS(0),
        POINTS_PER_THREAD(0),
        DIM(0),
        LOAD_MODIFIER(cub::LOAD_DEFAULT),
        INPUT_MEM_LAYOUT(rd::ROW_MAJOR),
        IO_BACKEND(rd::gpu::IO_BACKEND_CUB),
        STORE_TWO_PHASE(false),
        avgMillis(std::numeric_limits<float>::max()),
        gigaBandwidth(std::numeric_limits<float>::lowest())
    {}

    KernelParametersConf(
        int                         _DIM,
        cub::CacheLoadModifier      _LOAD_MODIFIER,
        rd::gpu::BlockTileIOBackend _IO_BACKEND,
        rd::DataMemoryLayout        _INPUT_MEM_LAYOUT,
        bool                        _STORE_TWO_PHASE)
    :
        DIM(_DIM),
        LOAD_MODIFIER(_LOAD_MODIFIER),
        IO_BACKEND(_IO_BACKEND),
        INPUT_MEM_LAYOUT(_INPUT_MEM_LAYOUT),
        STORE_TWO_PHASE(_STORE_TWO_PHASE)
    {}

    void printLaunchConf(std::ostream& os) const
    {
        os << POINTS_PER_THREAD 
            << " " << BLOCK_THREADS 
            << " " << avgMillis
            << " " << gigaBandwidth << "\n";
    }

    bool hasSameAlgParams(KernelParametersConf const & kpc) const
    {
        return  LOAD_MODIFIER       == kpc.LOAD_MODIFIER &&
                IO_BACKEND          == kpc.IO_BACKEND &&
                INPUT_MEM_LAYOUT    == kpc.INPUT_MEM_LAYOUT &&
                STORE_TWO_PHASE     == kpc.STORE_TWO_PHASE;
    }
    
    KernelParametersConf getAlgParamCopy() const
    {
        KernelParametersConf out = *this;
        out.BLOCK_THREADS = 0;
        out.POINTS_PER_THREAD = 0;
        out.avgMillis = 0.f;
        out.gigaBandwidth = 0.f;
        return out;
    }
};

std::ostream & operator<<(std::ostream & os, KernelParametersConf const & kp)
{
    os << "\n dim: \t\t\t" << kp.DIM
        << "\n avgMillis: \t\t" << kp.avgMillis
        << "\n gigaBandwidth: \t" << kp.gigaBandwidth
        << "\n block threads: \t" << kp.BLOCK_THREADS
        << "\n points per thread: \t" << kp.POINTS_PER_THREAD
        << "\n load modifier: \t" << rd::getLoadModifierName(kp.LOAD_MODIFIER)
        << "\n mem layout: \t\t" << rd::getRDDataMemoryLayout(kp.INPUT_MEM_LAYOUT)
        << "\n io backend: \t\t" << rd::getRDTileIOBackend(kp.IO_BACKEND)
        << "\n storeTwoPhase \t\t" << std::boolalpha << kp.STORE_TWO_PHASE << "\n";
    return os;
}

typedef std::pair<float, float> KernelPerfT;

//------------------------------------------------------------
//  Gnuplot data file generation
//------------------------------------------------------------


template <typename T>
std::string createFinalGraphDataFile(
    unsigned int MIN_TEST_DIM,
    unsigned int MAX_TEST_DIM,
    int numPoints)
{
    // assume that all current elements in graphData have the same selecRatio.
    float selectRatio = std::get<0>(g_graphData[0]);

    std::ostringstream graphDataFileName;
    graphDataFileName << typeid(T).name() << "_" << g_devName 
        << "_selRatio" << selectRatio 
        << "_nPoints" << numPoints
        << "_graphData.dat";

    std::string filePath = rd::findPath("gnuplot_data/", graphDataFileName.str());
    std::ofstream gdataFile(filePath.c_str(), std::ios::out | std::ios::trunc);
    if (gdataFile.fail())
    {
        throw std::logic_error("Couldn't open file: " + graphDataFileName.str());
    }

    /**
     * group graph data into graph columns (readable form for gnuplot)
     * g_graphData has structure:
     * [0] DIM 1 (alg param set1)
     * [2] DIM 1 (alg param set2)
     * [3] DIM 1 (alg param set3)
     * [4] DIM 1 (alg param set4)
     *  ...
     * [n] DIM 2 (alg param set1)
     * [n+1] DIM 2 (alg param set2)
     * [n+2] DIM 2 (alg param set3)
     * [n+3] DIM 2 (alg param set4)
     *  ...
     */

    std::vector<std::pair<KernelParametersConf, std::vector<float>>> graphColumns;

    // read first dimension
    for (auto it = g_graphData.begin(); it != g_graphData.end();)
    {
        if (std::get<1>(*it) == MIN_TEST_DIM)
        {
            // add column
            KernelParametersConf tmp = std::get<2>(*it);
            graphColumns.emplace_back(tmp.getAlgParamCopy(), std::vector<float>{float(MIN_TEST_DIM), tmp.gigaBandwidth});
            // erase element from g_graphData
            it = g_graphData.erase(it);
        }
        else
        {
            break;
        }
    }

    // read other dimension
    for (unsigned int dim = MIN_TEST_DIM+1; dim <= MAX_TEST_DIM; ++dim)
    {
        unsigned int column = 0;
        for (auto it = g_graphData.begin(); it != g_graphData.end();)
        {
            if (std::get<1>(*it) == dim)
            {
                KernelParametersConf tmp = std::get<2>(*it);
                if (!tmp.hasSameAlgParams(graphColumns[column].first))
                {
                    std::cerr << "ERROR! incorrect columns order!" << std::endl;
                    std::cout << "\n tmp: \n" << tmp << std::endl;
                    std::cout << "\n graphColumns["<<column<<"]: \n" << graphColumns[column].first << std::endl;
                    exit(1);
                }
                graphColumns[column].second.push_back(dim);
                graphColumns[column].second.push_back(tmp.gigaBandwidth);
                // erase element from g_graphData
                it = g_graphData.erase(it);
                column++;
            }
            else
            {
                break;
            }
        }
    }

    for (auto const & e : graphColumns)
    {
        // prepare secname
        std::ostringstream secName;
        secName << rd::getLoadModifierName(e.first.LOAD_MODIFIER)
                << "_" << rd::getRDDataMemoryLayout(e.first.INPUT_MEM_LAYOUT)
                << "_" << rd::getRDTileIOBackend(e.first.IO_BACKEND)
                << "_" << std::boolalpha << e.first.STORE_TWO_PHASE;

        auto v = e.second;

        gdataFile << "# [" << secName.str() << "] \n";
        for (size_t i = 0; i < v.size() / 2; ++i)
        {
            gdataFile << std::right << std::fixed << std::setw(5) << std::setprecision(1) <<
                int(v[2*i]) << " " << v[2*i + 1] << "\n";
        }
        // two sequential blank records to reset $0 counter
        gdataFile << "\n\n";

    }

    gdataFile.close();
    return filePath;
}

//------------------------------------------------------------
//  Select Op
//------------------------------------------------------------

template <int DIM, typename T>
struct LessThan
{
    T val_;

    __host__ __device__ __forceinline__ LessThan(T v)
    : 
        val_(v)
    {}

    __host__ __device__ __forceinline__ bool operator()(T const * point) const
    {
        for (int d = 0; d < DIM; ++d)
        {
            if (point[d] >= val_) 
                return false;
        }
        return true;
    }
};

//------------------------------------------------------------
//  KERNEL 
//------------------------------------------------------------

template <
    typename                    BlockSelectIfPolicyT,
    int                         DIM,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    typename                    OffsetT,
    typename                    SampleT,
    typename                    SelectOpT,
    bool                        STORE_TWO_PHASE>    // Whether or not to perform two phase selected items store with items compatcion in shmem. Otherwise uses warp-wide store.
__launch_bounds__ (int(BlockSelectIfPolicyT::BLOCK_THREADS))
static __global__ void selectIfKernel(
    SampleT const *                         d_in,
    OffsetT                                 numPoints,
    SampleT **                              d_selectedItemsPtrs,
    OffsetT *                               d_selectedItemsCnt,
    SelectOpT                               selectOp,
    OffsetT                                 stride,
    bool                                    correctnessCheck)
{
    
    typedef rd::gpu::BlockDynamicVector<
        BlockSelectIfPolicyT::BLOCK_THREADS, 
        BlockSelectIfPolicyT::POINTS_PER_THREAD * DIM, 
        SampleT,
        6,
        float> VectorT;

    typedef rd::gpu::BlockSelectIf<
        BlockSelectIfPolicyT,
        DIM,
        INPUT_MEM_LAYOUT,
        SelectOpT,
        VectorT,
        SampleT,
        OffsetT,
        STORE_TWO_PHASE>
    BlockSelectIfT;

    if (numPoints == 0)
    {
        return;
    }

    // define dynamic vector resize thresholds
    VectorT dynVec({0.05f, 0.15f, 0.25f, 0.45f, 0.66f, 1.0f}, (float)numPoints * DIM);

    __shared__ typename BlockSelectIfT::TempStorage tempStorage;

    OffsetT selectedPointsCnt = BlockSelectIfT(tempStorage, d_in, dynVec, selectOp).scanRange(0, numPoints, stride);

    if (threadIdx.x == 0)
    {
        d_selectedItemsCnt[blockIdx.x] = selectedPointsCnt;
    }

    if (correctnessCheck)
    {

        typedef rd::gpu::BlockTileLoadPolicy<
            BlockSelectIfPolicyT::BLOCK_THREADS,
            BlockSelectIfPolicyT::POINTS_PER_THREAD,
            BlockSelectIfPolicyT::LOAD_MODIFIER> BlockTileLoadPolicyT;

        typedef rd::gpu::BlockTileStorePolicy<
            BlockSelectIfPolicyT::BLOCK_THREADS,
            BlockSelectIfPolicyT::POINTS_PER_THREAD,
            cub::STORE_DEFAULT> BlockTileStorePolicyT;

        typedef rd::gpu::AgentMemcpy<
            BlockTileLoadPolicyT,
            BlockTileStorePolicyT,
            DIM,
            rd::ROW_MAJOR,                       // memory layout
            rd::ROW_MAJOR,                       // private memory layout
            BlockSelectIfPolicyT::IO_BACKEND,
            OffsetT,
            SampleT> MemcpyEngineT;

        unsigned int vecSize = dynVec.size();
        #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                _CubLog("CorrectnessCheck! ,copying dynVec data, vec.begin(): %p, vec.size(): %u, vec.capacity(): %d\n",
                         dynVec.begin(), vecSize, dynVec.capacity());
            }
        #endif

        MemcpyEngineT(dynVec.begin(), d_selectedItemsPtrs[blockIdx.x]).copyRange(0, vecSize / DIM, stride, true);
        __syncthreads();
    
    }
    // #ifdef RD_DEBUG
    //     if (threadIdx.x == 0)
    //     {
    //         _CubLog("----  Clearing dynamic vector!\n", 0);
    //     }
    // #endif
    dynVec.clear();
}

//------------------------------------------------------------
//  KERNEL INVOCATION
//------------------------------------------------------------

struct KernelConfig
{
    int blockThreads;
    int itemsPerThread;
};

template <
    typename                    OffsetT,
    typename                    SampleT,
    typename                    SelectOpT,
    typename                    PartitionKernelPtrT>
static cudaError_t invoke(
    SampleT const *                 d_in,
    OffsetT                         numPoints,
    SampleT **                      d_selectedItemsPtrs,
    OffsetT *                       d_selectedItemsCnt,
    OffsetT                         stride,
    SelectOpT                       selectOp,
    cudaStream_t                    stream,
    bool                            correctnessCheck,
    bool                            debugSynchronous,
    PartitionKernelPtrT             partitionKernelPtr,
    KernelConfig                    kernelConfig)
{

    cudaError error = cudaSuccess;
    do
    {
        // get SM occupancy
        int smOccupancy;
        if(CubDebug(cub::MaxSmOccupancy(
            smOccupancy,
            partitionKernelPtr,
            kernelConfig.blockThreads)
        )) break;

        dim3 partitionGridSize(1);
        partitionGridSize.x = CUB_MIN(smOccupancy * g_devSMCount * SUBSCRIPTION_FACTOR, g_devMaxBlockCnt);

        if (debugSynchronous)
        {
            printf("Invoking selectIfKernel<<<%d, %d, 0, %lld>>> numPoints: %d, pointsPerThread: %d\n",
                partitionGridSize.x, kernelConfig.blockThreads, (long long)stream, numPoints, kernelConfig.itemsPerThread);
        }

        partitionKernelPtr<<<partitionGridSize, kernelConfig.blockThreads, 0, stream>>>(
            d_in,
            numPoints,
            d_selectedItemsPtrs,
            d_selectedItemsCnt,
            selectOp,
            stride,
            correctnessCheck);

        // Check for failure to launch
        if (CubDebug(error = cudaPeekAtLastError())) break;
        // Sync the stream if specified to flush runtime errors
        if (debugSynchronous && (CubDebug(error = cub::SyncStream(stream)))) break;


    } while (0);

    return error;
}

//------------------------------------------------------------
//  KERNEL DISPATCH
//------------------------------------------------------------

template <
    int                         BLOCK_THREADS,
    int                         POINTS_PER_THREAD,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    int                         DIM,
    bool                        STORE_TWO_PHASE,
    typename                    SelectOpT,
    typename                    OffsetT,
    typename                    T>
static void dispatchKernel(
    T const *                       d_in,
    OffsetT                         numPoints,
    T **                            d_selectedItemsPtrs,
    OffsetT *                       d_selectedItemsCnt,
    SelectOpT                       selectOp,
    OffsetT                         stride,
    int                             iterations,
    bool                            correctnessCheck = false,       ///< whether or not copy back data to device vector for correctnes check
    bool                            debugSynchronous = false)
{

    typedef rd::gpu::BlockSelectIfPolicy<
        BLOCK_THREADS,
        POINTS_PER_THREAD,
        LOAD_MODIFIER,
        IO_BACKEND>
    BlockSelectIfPolicyT;

    KernelConfig partitionConfig;
    partitionConfig.blockThreads = BLOCK_THREADS;
    partitionConfig.itemsPerThread = POINTS_PER_THREAD;

    auto partitionKernelPtr = selectIfKernel<BlockSelectIfPolicyT, DIM, INPUT_MEM_LAYOUT, OffsetT, T, SelectOpT, STORE_TWO_PHASE>;

    // If we use two-phase store algorithm, which compact's selections in smem, we prefer larger smem to L1 cache size.
    if (STORE_TWO_PHASE)
    {
        // set smem/L1 mem configuration
        // * - ::cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
        // * - ::cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
        // * - ::cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
        // * - ::cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory
        cudaFuncSetCacheConfig(partitionKernelPtr, cudaFuncCachePreferShared);

        if (sizeof(T) == 8)
        {
            // * - ::cudaSharedMemBankSizeDefault: use the device's shared memory configuration when launching this function.
            // * - ::cudaSharedMemBankSizeFourByte: set shared memory bank width to be four bytes natively when launching this function.
            // * - ::cudaSharedMemBankSizeEightByte: set shared memory bank width to be eight bytes natively when launching this function.
            cudaFuncSetSharedMemConfig(partitionKernelPtr, cudaSharedMemBankSizeEightByte);
        }
    }

    for (int i = 0; i < iterations; ++i)
    {
        CubDebugExit(invoke(
            d_in,
            numPoints,
            d_selectedItemsPtrs,
            d_selectedItemsCnt,
            stride,
            selectOp,
            0,
            correctnessCheck,
            debugSynchronous,
            partitionKernelPtr,
            partitionConfig));
    }

}

//------------------------------------------------------------
//  Test and benchmark specified kernel configuration 
//------------------------------------------------------------

template <
    int                         BLOCK_THREADS,
    int                         POINTS_PER_THREAD,
    int                         DIM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    bool                        STORE_TWO_PHASE,
    typename                    T>
KernelPerfT runSelectIf(
    int                             numPoints,
    float                           compare,
    T const *                       d_in,
    int                             stride,
    T const *                       h_reference,
    int const                       h_referenceSelectedPointsCnt)
{
    std::cout << rd::HLINE << std::endl;
    std::cout << "runSelectIf:" << std::endl;
    std::cout << "blockThreads: " << BLOCK_THREADS 
              << ", pointsPerThread: " << POINTS_PER_THREAD
              << ", load modifier: " << rd::LoadModifierNameTraits<LOAD_MODIFIER>::name
              << ", io backend: " << rd::BlockTileIONameTraits<IO_BACKEND>::name
              << ", mem layout: " << rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name
              << ", store two phase: " << std::boolalpha << STORE_TWO_PHASE
              << ", numPoints: " << numPoints << "\n";

    LessThan<DIM, T> selectOp(compare);

    // type definitions for kernel function pointer
    typedef LessThan<DIM, T> SelectOpT;
    typedef rd::gpu::BlockSelectIfPolicy<
        BLOCK_THREADS,
        POINTS_PER_THREAD,
        LOAD_MODIFIER,
        IO_BACKEND>
    BlockSelectIfPolicyT;

    auto kernelPtr = selectIfKernel<BlockSelectIfPolicyT, DIM, INPUT_MEM_LAYOUT, int, T, SelectOpT, STORE_TWO_PHASE>;

    // get SM occupancy
    int smOccupancy;
    CubDebugExit(cub::MaxSmOccupancy(
        smOccupancy,
        kernelPtr,
        BLOCK_THREADS)
    );

    int blockCount = CUB_MIN(smOccupancy * g_devSMCount * SUBSCRIPTION_FACTOR, g_devMaxBlockCnt);

    // Allocate device arrays
    T ** d_selectedPointsPtrs = nullptr;
    T ** h_dSelectedPointsPtrs = new T*[blockCount];
    int * d_selectedPointsCnt = nullptr;

    checkCudaErrors(cudaMalloc(&d_selectedPointsPtrs, blockCount * sizeof(T*)));
    checkCudaErrors(cudaMalloc(&d_selectedPointsCnt, blockCount * sizeof(int)));
    
    cudaStream_t auxStream;
    checkCudaErrors(cudaStreamCreateWithFlags(&auxStream, cudaStreamNonBlocking));
    checkCudaErrors(cudaMemsetAsync(d_selectedPointsCnt, 0, blockCount * sizeof(int), auxStream));

    for (int k = 0; k < blockCount; ++k)
    {
        checkCudaErrors(cudaMalloc(h_dSelectedPointsPtrs + k, h_referenceSelectedPointsCnt * DIM * sizeof(T)));
        checkCudaErrors(cudaMemsetAsync(h_dSelectedPointsPtrs[k], 0, h_referenceSelectedPointsCnt * DIM * sizeof(T), auxStream));
    }

    // Initialize device input
    checkCudaErrors(cudaMemcpy(d_selectedPointsPtrs, h_dSelectedPointsPtrs, blockCount * sizeof(T*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    // Run warm-up/correctness iteration
    dispatchKernel<BLOCK_THREADS, POINTS_PER_THREAD, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, DIM, STORE_TWO_PHASE>(
        d_in, numPoints, d_selectedPointsPtrs, d_selectedPointsCnt, selectOp, stride, 1, true, true);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // check results
    std::cout << "\nCheck results count ... ";
    if (CompareDeviceResults(h_referenceSelectedPointsCnt, d_selectedPointsCnt, blockCount))
    {
        std::cerr << "\n\n ERROR! Incorrect results count!" << std::endl;
        exit(1);
    }
    else
    {
        std::cout << " PASS!\n";
    }

    std::cout << "Check each block results ";
    #ifdef RD_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < blockCount; ++i)
    {
        // compareDeviceResults sorts device data
        if (CompareDeviceResults(h_reference, h_dSelectedPointsPtrs[i], h_referenceSelectedPointsCnt * DIM, true, true))
        {
            std::cerr << "\n\n ERROR! Incorrect device selected items!" << std::endl;
            exit(1);
        }
    }
    std::cout << " PASS!\n";

    // Measure performance
    GpuTimer timer;
    float elapsedMillis;

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    dispatchKernel<BLOCK_THREADS, POINTS_PER_THREAD, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, DIM, STORE_TWO_PHASE>(
        d_in, numPoints, d_selectedPointsPtrs, d_selectedPointsCnt, selectOp, stride, g_iterations);
    timer.Stop();
    elapsedMillis = timer.ElapsedMillis();
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif



    float   avgMillis           = elapsedMillis / g_iterations;
    int     selItemsCnt         = h_referenceSelectedPointsCnt * DIM * blockCount;
    size_t  numBytes            = sizeof(T) * blockCount * numPoints * DIM +              // every block scans entire data set
                                    blockCount * sizeof(int) +                            // storing selectedItems counters
                                    selItemsCnt * sizeof(T);                                      // storing selected items
    float   gigaBandwidth       =   float(numBytes) / avgMillis / 1000.0 / 1000.0;                    // conversion to GB/s

    if (g_logResults)
    {
        *g_logFile << POINTS_PER_THREAD << " " << BLOCK_THREADS << " " << avgMillis << " " << gigaBandwidth << "\n";
    }

    std::cout << avgMillis << " avg ms, "
              << gigaBandwidth << " logical GB/s\n";

    // cleanup
    checkCudaErrors(cudaStreamDestroy(auxStream));
    for (int k = 0; k < blockCount; ++k)
    {
        if (h_dSelectedPointsPtrs[k]) checkCudaErrors(cudaFree(h_dSelectedPointsPtrs[k]));
    }

    if (d_selectedPointsPtrs) checkCudaErrors(cudaFree(d_selectedPointsPtrs));
    if (d_selectedPointsCnt) checkCudaErrors(cudaFree(d_selectedPointsCnt));
    if (h_dSelectedPointsPtrs) delete[] h_dSelectedPointsPtrs;

    return std::make_pair(avgMillis, gigaBandwidth);
}

//------------------------------------------------------------
//  Test kernel block-threads / items-per-thread configurations
//------------------------------------------------------------

/*
 *  Test different kernel configurations (block size, points per thread)
 */
template <
    int                         DIM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    bool                        STORE_TWO_PHASE,
    typename                    T>
KernelParametersConf testKernelConf(
    int                             numPoints,
    float                           compare,
    T const *                       d_in,
    int                             stride,
    T const *                       h_reference,
    int const                       h_referenceSelectedPointsCnt)
{

    KernelParametersConf bestKernelParams(DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE);
    KernelPerfT bestPerf = std::make_pair(1e10f, -1.0f);

    auto checkBestConf = [&bestPerf, &bestKernelParams](int bs, int ppt, KernelPerfT kp)
    {
        if (kp.second > bestPerf.second)
        {
            bestPerf.first = kp.first;
            bestPerf.second = kp.second;
            bestKernelParams.avgMillis = kp.first;
            bestKernelParams.gigaBandwidth = kp.second;
            bestKernelParams.BLOCK_THREADS = bs;
            bestKernelParams.POINTS_PER_THREAD = ppt;
        }
    };

    #define runTest(blockSize, ppt) checkBestConf(blockSize, ppt, \
            runSelectIf< \
                    blockSize, \
                    ppt, \
                    DIM, \
                    LOAD_MODIFIER, \
                    IO_BACKEND, \
                    INPUT_MEM_LAYOUT, \
                    STORE_TWO_PHASE, T>( \
                numPoints, \
                compare, \
                d_in, \
                stride, \
                h_reference, \
                h_referenceSelectedPointsCnt));
    
    #ifdef QUICK_TEST
        runTest(128, 4);
    #else
        // runTest(64, 1);
        // runTest(64, 2);
        // runTest(64, 3);
        runTest(64, 4);
        // runTest(64, 5);
        // runTest(64, 6);
        // runTest(64, 7);
        // runTest(64, 8);
        // runTest(64, 9);
        // runTest(64, 10);

        // runTest(96, 1);
        // runTest(96, 2);
        // runTest(96, 3);
        // runTest(96, 4);
        // runTest(96, 5);
        // runTest(96, 6);
        // runTest(96, 7);
        // runTest(96, 8);
        // runTest(96, 9);
        // runTest(96, 10); 

        // runTest(128, 1);
        // runTest(128, 2);
        // runTest(128, 3);
        runTest(128, 4);
        // runTest(128, 5);
        // runTest(128, 6);
        // runTest(128, 7);
        // runTest(128, 8);
        // runTest(128, 9);
        // runTest(128, 10);

        // runTest(256, 1);
        // runTest(256, 2);
        // runTest(256, 3);
        // runTest(256, 4);
        // runTest(256, 5);
        // runTest(256, 6);
        // runTest(256, 7);
        // runTest(256, 8);
        // runTest(256, 9);
        // runTest(256, 10);

        // runTest(384, 1);
        // runTest(384, 2);
        // runTest(384, 3);
        // runTest(384, 4);
        // runTest(384, 5);
        // runTest(384, 6);
        // runTest(384, 7);
        // runTest(384, 8);
        // runTest(384, 9);
        // runTest(384, 10);

        // runTest(512, 1);
        // runTest(512, 2);
        // runTest(512, 3);
        // runTest(512, 4);
        // runTest(512, 5);
        // runTest(512, 6);
        // runTest(512, 7);
        // runTest(512, 8);
        // runTest(512, 9);
        // runTest(512, 10);
    #endif
    #undef runTest

    if (g_logResults)
    {
        *g_logFile << "% best: ";
         bestKernelParams.printLaunchConf(*g_logFile);
    }

    std::cout << "-------------------------------\n";
    std::cout << ">>>>>>> best performance conf: ";
    bestKernelParams.printLaunchConf(std::cout);
    std::cout << "\n-------------------------------\n";

    return bestKernelParams;
}


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Initialize problem
 */
template <typename    T>
static void Initialize(
    T*  h_in,
    int numItems)
{
    rd::fillRandomDataTable(h_in, numItems, T(0), T(126));
    if (g_verbose)
    {
        rd::printTable(h_in, numItems, "Input:");
    }
}

/**
 * @brief      Create if necessary and open log file. Allocate log file stream.
 */
template <typename T>
static void initializeLogFile()
{
    if (g_logResults)
    {
        std::ostringstream logFileName;
        // append device name to log file
        logFileName << g_devName << "_" << LOG_FILE_NAME_SUFFIX;

        std::string logFilePath = rd::findPath("timings/", logFileName.str());
        g_logFile = new std::ofstream(logFilePath.c_str(), std::ios::out | std::ios::app);
        if (g_logFile->fail())
        {
            throw std::logic_error("Couldn't open file: " + logFileName.str());
        }

        *g_logFile << "%" << rd::HLINE << std::endl;
        *g_logFile << "% " << typeid(T).name() << std::endl;
        *g_logFile << "%" << rd::HLINE << std::endl;
    }
}

/**
 * Reference selection problem solution.
 */
template <
    int         DIM,
    typename    T,
    typename    SelectOpT>
static void solve(
    T const *           h_in,
    size_t              numPoints,
    T *                 h_selectedPoints,
    int &               h_selectedPointsCnt,
    SelectOpT           selectOp)
{
    #ifdef RD_USE_OPENMP
    #pragma omp parallel for num_threads(8), schedule(static)
    #endif
    for (size_t k = 0; k < numPoints; ++k)
    {
        T const * point = h_in + k * DIM;

        if (selectOp(point))
        {
            int offset = 0;
            #ifdef RD_USE_OPENMP
            #pragma omp atomic capture
            #endif
            offset = h_selectedPointsCnt++;

            for (int d = 0; d < DIM; ++d)
            {
                h_selectedPoints[offset * DIM + d] = point[d];
            }
        }
    }
}

/**
 * @brief      Prepare and run test. Allocate and initialize test input data.
 */
template <
    int                         DIM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    bool                        STORE_TWO_PHASE,
    typename                    T>
KernelParametersConf prepareAndRunTest(
    int                             numPoints,
    float                           selectRatio)
{
    // allocate host arrays
    T * h_in            = new T[numPoints * DIM];
    T * h_reference     = new T[numPoints * DIM];
    int h_referenceSelectedPointsCnt = 0;

    // Initialize input
    Initialize(h_in, numPoints * DIM);

    // Select a comparison value that is selectRatio through the space of [0,127]
    T compare;
    if (selectRatio <= 0.0)
        InitValue(INTEGER_SEED, compare, 0);        // select none
    else if (selectRatio >= 1.0)
        InitValue(INTEGER_SEED, compare, 127);      // select all
    else
        InitValue(INTEGER_SEED, compare, int(double(double(127) * selectRatio)));

    LessThan<DIM, T> selectOp(compare);
    solve<DIM>(h_in, numPoints, h_reference, h_referenceSelectedPointsCnt, selectOp);

    // sort results because, points selected on GPU may be stored in different order.
    std::sort(h_reference, h_reference + h_referenceSelectedPointsCnt * DIM);

    std::cout << "\nTest: \n"
          << ", load modifier: " << rd::LoadModifierNameTraits<LOAD_MODIFIER>::name
          << ", io backend: " << rd::BlockTileIONameTraits<IO_BACKEND>::name
          << ", mem layout: " << rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name
          << ", store two phase: " << std::boolalpha << STORE_TWO_PHASE
          << ", numPoints: " << numPoints << "("<<DIM<<"-dim)\n";
    std::cout << "\nComparison item: " << compare 
          << ", " << h_referenceSelectedPointsCnt << " selected points (select ratio " << selectRatio << ")\n";

    if (g_logResults)
    {
        *g_logFile << "% "
              << " loadModifier=" << rd::LoadModifierNameTraits<LOAD_MODIFIER>::name
              << " ioBackend=" << rd::BlockTileIONameTraits<IO_BACKEND>::name
              << " memLayout=" << rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name
              << " storeTwoPhase=" << std::boolalpha << STORE_TWO_PHASE
              << " numPoints=" << numPoints 
              << " dim=" << DIM 
              << " compareItem=" << compare 
              << " selectRatio=" << selectRatio << "\n"; 
    }


    // Allocate device arrays
    T * d_in = nullptr;
    checkCudaErrors(cudaMalloc(&d_in, numPoints * DIM * sizeof(T)));
    rd::gpu::rdMemcpy<DIM, INPUT_MEM_LAYOUT, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(d_in, h_in, numPoints);

    int inDataStride = (INPUT_MEM_LAYOUT == rd::COL_MAJOR) ? numPoints : 1;

    // Run test kernel configurations
    KernelParametersConf bestKernelParams = testKernelConf<DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE>(
        numPoints,
        compare, 
        d_in, 
        inDataStride, 
        h_reference,
        h_referenceSelectedPointsCnt);

    // cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (d_in) checkCudaErrors(cudaFree(d_in));

    return bestKernelParams;
}

template <
    int                             DIM,
    cub::CacheLoadModifier          LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend     IO_BACKEND,
    rd::DataMemoryLayout            INPUT_MEM_LAYOUT,
    typename                        T>    
void testStoreAlgorithm(
    int             numPoints,
    float           selectRatio)
{
    KernelParametersConf kp;

    /**
     * store two-phase (with selected items compaction in smem)
     */
    kp = prepareAndRunTest<DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, true, T>(numPoints, selectRatio);
    auto perfRes = std::make_tuple(selectRatio, DIM, kp);
    g_graphData.push_back(perfRes);

    /**
     * store using warp-aggregated algorithm
     */
    // kp = prepareAndRunTest<DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, false, T>(numPoints, selectRatio);
    // std::get<2>(perfRes) = kp;
    // g_graphData.push_back(perfRes);
}

template <
    int                             DIM,
    rd::gpu::BlockTileIOBackend     IO_BACKEND,
    rd::DataMemoryLayout            INPUT_MEM_LAYOUT,
    typename                        T>
void testLoadModifier(
    int             numPoints,
    float           selectRatio)
{
    testStoreAlgorithm<DIM, cub::LOAD_LDG, IO_BACKEND, INPUT_MEM_LAYOUT, T>(numPoints, selectRatio);
    // Cache streaming (likely to be accessed once)
    // testStoreAlgorithm<DIM, cub::LOAD_CS, IO_BACKEND, INPUT_MEM_LAYOUT, T>(numPoints, selectRatio);
}

template <
    int                             DIM,
    rd::DataMemoryLayout            INPUT_MEM_LAYOUT,
    typename                        T>
void testIOBackend(
    int             numPoints,
    float           selectRatio)
{
    testLoadModifier<DIM, rd::gpu::IO_BACKEND_CUB,   INPUT_MEM_LAYOUT, T>(numPoints, selectRatio);
    // testLoadModifier<DIM, rd::gpu::IO_BACKEND_TROVE, INPUT_MEM_LAYOUT, T>(numPoints, selectRatio);
}

template <
    int         DIM,
    typename    T>
void testInputMemLayout(
    int             numPoints,
    float           selectRatio)
{
    testIOBackend<DIM, rd::ROW_MAJOR, T>(numPoints, selectRatio);
    // testIOBackend<DIM, rd::COL_MAJOR, T>(numPoints, selectRatio);

}


template <typename T>
static void allocateDevHeapMem(
    int DIM,
    int numPoints)
{
    size_t devFreeMem, devTotalMem;
    cudaMemGetInfo(&devFreeMem, &devTotalMem);

    size_t neededMemSize = size_t(numPoints * DIM * g_devMaxBlockCnt) * sizeof(T);
    if (neededMemSize > devFreeMem)
    {
        throw std::runtime_error("Insufficient device free memory available!");
    }
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, neededMemSize));

    size_t mb = 1024 * 1024;
    std::cout << "Device total mem: "<<devTotalMem / mb<<"(MB), free: "<<devFreeMem / mb<<"(MB), allocated heap size: "<<neededMemSize / mb<<"(MB)\n";
}

template <
    typename       T>
void testDim(
    int             numPoints,
    int             devId)
{
    #if !defined(RD_DEBUG) && !defined(RD_PROFILE)
    const unsigned int MIN_TEST_DIM = 1;
    const unsigned int MAX_TEST_DIM = 6;
    #else
    const unsigned int MIN_TEST_DIM = 2;
    const unsigned int MAX_TEST_DIM = 2;
    #endif
    
    checkCudaErrors(deviceInit(devId));
    allocateDevHeapMem<T>(MAX_TEST_DIM, numPoints);

    if (g_logResults)
    {
        initializeLogFile<T>();
    }


    // for (float selectRatio = 0; selectRatio <= 1.0f; selectRatio += 0.2f)
    for (float selectRatio = 0.6; selectRatio <= 0.6f; selectRatio += 0.2f)
    {
        #if defined(RD_DEBUG) || defined(RD_PROFILE) 
            std::cout << "\nTest DIM 2...\n\n";
            testInputMemLayout<2, T>(numPoints, selectRatio);
        #else
            std::cout << "\nTest DIM 1...\n\n";
            testInputMemLayout<1, T>(numPoints, selectRatio);
            std::cout << "\nTest DIM 2...\n\n";
            testInputMemLayout<2, T>(numPoints, selectRatio);
            std::cout << "\nTest DIM 3...\n\n";
            testInputMemLayout<3, T>(numPoints, selectRatio);
            std::cout << "\nTest DIM 4...\n\n";
            testInputMemLayout<4, T>(numPoints, selectRatio);
            std::cout << "\nTest DIM 5...\n\n";
            testInputMemLayout<5, T>(numPoints, selectRatio);
            std::cout << "\nTest DIM 6...\n\n";
            testInputMemLayout<6, T>(numPoints, selectRatio);
        #endif

        //---------------------------------------------------
        //  summarize results
        //---------------------------------------------------
    
        for (unsigned int d = MIN_TEST_DIM; d <= MAX_TEST_DIM; ++d)
        {
            KernelParametersConf bestKernelParams;
            bestKernelParams.gigaBandwidth = -1.0f;

            // find best configuration for given dimension
            for (auto const & e : g_graphData)
            {
                if (std::get<1>(e) == d)
                {
                    KernelParametersConf conf = std::get<2>(e);
                    if (conf.gigaBandwidth > bestKernelParams.gigaBandwidth)
                    {
                        bestKernelParams = conf;
                    }
                }
            }

            if (g_logResults)
            {
                *g_logFile << "\n% overallBest "
                    << " dim=" << d 
                    << " avgMillis=" << bestKernelParams.avgMillis
                    << " gigaBandwidth=" << bestKernelParams.gigaBandwidth
                    << " blockThreads=" << bestKernelParams.BLOCK_THREADS
                    << " pointsPerThread=" << bestKernelParams.POINTS_PER_THREAD
                    << " loadModifier=" << rd::getLoadModifierName(bestKernelParams.LOAD_MODIFIER)
                    << " memLayout=" << rd::getRDDataMemoryLayout(bestKernelParams.INPUT_MEM_LAYOUT)
                    << " ioBackend=" << rd::getRDTileIOBackend(bestKernelParams.IO_BACKEND)
                    << " storeTwoPhase=" << std::boolalpha << bestKernelParams.STORE_TWO_PHASE
                    << " numPoints=" << numPoints << "\n";
            }

            std::cout << "\n>>>>> overall best conf: " 
                << "\n dim: \t\t\t" << d
                << "\n avgMillis: \t\t" << bestKernelParams.avgMillis
                << "\n gigaBandwidth: \t" << bestKernelParams.gigaBandwidth
                << "\n block threads: \t" << bestKernelParams.BLOCK_THREADS
                << "\n points per thread: \t" << bestKernelParams.POINTS_PER_THREAD
                << "\n load modifier: \t" << rd::getLoadModifierName(bestKernelParams.LOAD_MODIFIER)
                << "\n mem layout: \t\t" << rd::getRDDataMemoryLayout(bestKernelParams.INPUT_MEM_LAYOUT)
                << "\n io backend: \t\t" << rd::getRDTileIOBackend(bestKernelParams.IO_BACKEND)
                << "\n storeTwoPhase \t\t" << std::boolalpha << bestKernelParams.STORE_TWO_PHASE
                << "\n numPoints: \t\t" << numPoints << "\n";
        }

        if (g_drawGraphs)
        {
           createFinalGraphDataFile<T>(MIN_TEST_DIM, MAX_TEST_DIM, numPoints);
        }

        g_graphData.clear();
    }


    if (g_logResults)
    {
        if (g_logFile) delete g_logFile;
    }

    checkCudaErrors(cudaDeviceReset());
}

template <
    typename        T>
void testSize(
    int             numPoints, 
    int             devId)
{
    if (numPoints < 0)
    {
        testDim<T>(0      , devId);
        testDim<T>(1      , devId);
        testDim<T>(100    , devId);
        testDim<T>(10000  , devId);
        testDim<T>(100000 , devId);
        testDim<T>(1000000, devId);
    }
    else
    {
        testDim<T>(numPoints, devId);
    }
}

int main(int argc, char const **argv)
{
    
    int numPoints       = -1;
    int devId           = 0;
    float selectRatio   = 0.5f;

    // Initialize mersenne generator
    unsigned int mersenne_init[4]=  {0x123, 0x234, 0x345, 0x456};
    mersenne::init_by_array(mersenne_init, 4);

    // Initialize command line
    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help")) 
    {
        printf("%s \n"
            "\t\t--np=<number of input points>\n"
            "\t\t[--device=<device id>]\n"
            "\t\t[--ratio=<selection ratio, default 0.5>]\n"
            "\t\t[--v <verbose>]\n"
            "\t\t[--drawGraphs <draw graphs>]\n"
            "\t\t[--logResults <log performance results>]\n"
            "\n", argv[0]);
        exit(0);
    }

    args.GetCmdLineArgument("np", numPoints);

    if (args.CheckCmdLineFlag("device")) 
    {
        args.GetCmdLineArgument("device", devId);
    }
    if (args.CheckCmdLineFlag("ratio")) 
    {
        args.GetCmdLineArgument("ratio", selectRatio);
    }
    if (args.CheckCmdLineFlag("v")) 
    {
        g_verbose = true;
    }

    checkCudaErrors(deviceInit(devId));

    // set device name for logging and drawing purposes
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, devId));
    g_devName = devProp.name;
    // read device SM count and determine max number of resident blocks per SM
    g_devSMCount = devProp.multiProcessorCount;
    g_devMaxBlocksPerSM = (devProp.major < 3) ? 8 : 16;

    g_devMaxBlockCnt = g_devSMCount * g_devMaxBlocksPerSM;
    #ifdef RD_DEBUG
    g_devMaxBlockCnt = 14;
    #endif

    //-----------------------------------------
    //  TESTS
    //-----------------------------------------

#ifdef QUICK_TEST

    if (numPoints < 0)
    {
        numPoints = 1000000;
    }

    allocateDevHeapMem<float>(3, numPoints);

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 2D: " << std::endl;
    prepareAndRunTest<2, cub::LOAD_LDG, rd::gpu::IO_BACKEND_CUB, rd::ROW_MAJOR, true, float>(numPoints, selectRatio);
    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 3D: " << std::endl;
    prepareAndRunTest<3, cub::LOAD_LDG, rd::gpu::IO_BACKEND_CUB, rd::ROW_MAJOR, true, float>(numPoints, selectRatio);
    std::cout << rd::HLINE << std::endl;
    
#else    

    if (args.CheckCmdLineFlag("drawGraphs")) 
    {
        g_drawGraphs = true;
    }
    if (args.CheckCmdLineFlag("logResults")) 
    {
        g_logResults = true;
    }

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    testSize<float>(numPoints, devId);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE: " << std::endl;
    testSize<double>(numPoints, devId);
    std::cout << rd::HLINE << std::endl;

#endif

    checkCudaErrors(cudaDeviceReset());

    std::cout << "END!" << std::endl;
    return 0;
}
