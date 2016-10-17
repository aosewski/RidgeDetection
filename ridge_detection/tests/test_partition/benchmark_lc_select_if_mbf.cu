/**
 * @file benchmark_lc_select_if_mem_bf.cu
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

#include <helper_cuda.h>
#ifdef RD_PROFILE
#   include <cuda_profiler_api.h>
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <stdexcept>
#include <string>

#include <cmath>
#include <type_traits>

#ifdef RD_USE_OPENMP
#include <omp.h>
#endif

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/memory.h" 
#include "rd/utils/name_traits.hpp"
#include "rd/utils/rd_params.hpp"

#include "rd/gpu/block/block_select_if.cuh"
#include "rd/gpu/util/dev_samples_set.cuh"
#include "rd/gpu/util/dev_utilities.cuh"
#include "rd/gpu/util/dev_memcpy.cuh"

#include "tests/test_util.hpp"
#include "cub/test_util.h"
#include "cub/util_device.cuh"
#include "cub/util_type.cuh"


//------------------------------------------------------------
//  GLOBAL CONSTANTS / VARIABLES
//------------------------------------------------------------

static const std::string LOG_FILE_NAME_SUFFIX = "_select_if_mbf-timings.txt";

static std::ofstream * g_logFile            = nullptr;
static std::string     g_devName            = "";
static int             g_devSMCount         = 0;
static int             g_devId              = 0;
static bool            g_logPerfResults     = false;
static bool            g_startBenchmark     = false;

#if defined(RD_PROFILE) || defined(RD_DEBUG)
static const int g_iterations = 1;
#else
static const int g_iterations = 100;
#endif

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

/**
 * @tparam     STORE_TWO_PHASE Whether or not to perform two phase selected items store 
 *                             with items compatcion in shmem. Oherwise uses warp-wide store.
 */
template <
    typename                    BlockSelectIfPolicyT,
    int                         DIM,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    typename                    OffsetT,
    typename                    SampleT,
    typename                    SelectOpT,
    bool                        STORE_TWO_PHASE>    
__launch_bounds__ (int(BlockSelectIfPolicyT::BLOCK_THREADS))
static __global__ void selectIfKernel(
    SampleT const *                         d_in,
    OffsetT                                 numPoints,
    SampleT **                              d_selectedItemsPtrs,
    OffsetT *                               d_selectedItemsCnt,
    SelectOpT                               selectOp,
    OffsetT                                 inStride,
    OffsetT                                 outStride)
{
    typedef rd::gpu::BlockSelectIf<
        BlockSelectIfPolicyT,
        DIM,
        INPUT_MEM_LAYOUT,
        SelectOpT,
        SampleT,
        OffsetT,
        STORE_TWO_PHASE>
    BlockSelectIfT;

    if (numPoints == 0)
    {
        return;
    }

    __shared__ typename BlockSelectIfT::TempStorage tempStorage;

    OffsetT selectedPointsCnt = BlockSelectIfT(tempStorage, d_in, d_selectedItemsPtrs[blockIdx.x], 
        selectOp).scanRange(0, numPoints, inStride, outStride);

    if (threadIdx.x == 0)
    {
        d_selectedItemsCnt[blockIdx.x] = selectedPointsCnt;
    }
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
    OffsetT                         inStride,
    OffsetT                         outStride,
    SelectOpT                       selectOp,
    cudaStream_t                    stream,
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
        partitionGridSize.x = smOccupancy * g_devSMCount;

        if (debugSynchronous)
        {
            printf("Invoking selectIfKernel<<<%d, %d, 0, %p>>> numPoints: %d, "
                "pointsPerThread: %d\n",
                partitionGridSize.x, kernelConfig.blockThreads, stream, numPoints, 
                kernelConfig.itemsPerThread);
        }

        partitionKernelPtr<<<partitionGridSize, kernelConfig.blockThreads, 0, stream>>>(
            d_in,
            numPoints,
            d_selectedItemsPtrs,
            d_selectedItemsCnt,
            selectOp,
            inStride,
            outStride);

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
    OffsetT                         inStride,
    OffsetT                         outStride,
    int                             iterations,
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

    auto partitionKernelPtr = selectIfKernel<BlockSelectIfPolicyT, DIM, INPUT_MEM_LAYOUT,
        OffsetT, T, SelectOpT, STORE_TWO_PHASE>;

    // If we use two-phase store algorithm, which compact's selections in smem, we prefer 
    // larger smem to L1 cache size.
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
            // * - ::cudaSharedMemBankSizeDefault: use the device's shared memory 
            // configuration when launching this function.
            // * - ::cudaSharedMemBankSizeFourByte: set shared memory bank width to be 
            // four bytes natively when launching this function.
            // * - ::cudaSharedMemBankSizeEightByte: set shared memory bank width to be 
            // eight bytes natively when launching this function.
            cudaFuncSetSharedMemConfig(partitionKernelPtr, cudaSharedMemBankSizeEightByte);
        }
    }
    else
    {
        cudaFuncSetCacheConfig(partitionKernelPtr, cudaFuncCachePreferL1);
    }

    for (int i = 0; i < iterations; ++i)
    {
        CubDebugExit(invoke(
            d_in,
            numPoints,
            d_selectedItemsPtrs,
            d_selectedItemsCnt,
            inStride,
            outStride,
            selectOp,
            0,
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
struct RunSelectIf
{
    typedef LessThan<DIM, T> SelectOpT;

    typedef rd::gpu::BlockSelectIfPolicy<
        BLOCK_THREADS, POINTS_PER_THREAD, LOAD_MODIFIER, IO_BACKEND> 
    BlockSelectIfPolicyT;

    typedef rd::gpu::BlockSelectIf<
        BlockSelectIfPolicyT, DIM, INPUT_MEM_LAYOUT, SelectOpT, T, int, STORE_TWO_PHASE>
    BlockSelectIfT;

    typedef typename BlockSelectIfT::TempStorage KernelTempStorageT;

    typedef typename cub::If<sizeof(KernelTempStorageT) <= 0xc000,
        std::true_type, std::false_type>::Type EnableKernelT;

    static void impl(
        int         numPoints,
        float       compare,
        T const *   d_in,
        int         inStride,
        int         outStride,
        T           selectRatio,
        int         selectedPointsCnt)
    {
        doImpl<EnableKernelT>(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    }

    template <
        typename EnableT,
        typename std::enable_if<
            std::is_same<EnableT, std::true_type>::value, int>::type * = nullptr>
    static void doImpl(
        int         numPoints,
        float       compare,
        T const *   d_in,
        int         inStride,
        int         outStride,
        T           selectRatio,
        int         selectedPointsCnt)
    {
        SelectOpT selectOp(compare);

        auto kernelPtr = selectIfKernel<BlockSelectIfPolicyT, DIM, INPUT_MEM_LAYOUT, int, T, 
                            SelectOpT, STORE_TWO_PHASE>;

        // get SM occupancy
        int smOccupancy;
        checkCudaErrors(cub::MaxSmOccupancy(smOccupancy, kernelPtr, BLOCK_THREADS));

        int blockCount = 1;
        blockCount = smOccupancy * g_devSMCount;

        // Allocate device arrays
        T ** d_selectedPointsPtrs = nullptr;
        T ** h_dSelectedPointsPtrs = new T*[blockCount];
        int * d_selectedPointsCnt = nullptr;

        checkCudaErrors(cudaMalloc(&d_selectedPointsPtrs, blockCount * sizeof(T*)));
        checkCudaErrors(cudaMalloc(&d_selectedPointsCnt, blockCount * sizeof(int)));
        
        cudaStream_t auxStream;
        checkCudaErrors(cudaStreamCreateWithFlags(&auxStream, cudaStreamNonBlocking));
        checkCudaErrors(cudaMemsetAsync(d_selectedPointsCnt, 0, blockCount * sizeof(int), 
            auxStream));

        for (int k = 0; k < blockCount; ++k)
        {
            checkCudaErrors(cudaMalloc(h_dSelectedPointsPtrs + k, 
                selectedPointsCnt * DIM * sizeof(T)));
            checkCudaErrors(cudaMemsetAsync(h_dSelectedPointsPtrs[k], 0, 
                selectedPointsCnt * DIM * sizeof(T), auxStream));
        }

        // Initialize device input
        checkCudaErrors(cudaMemcpy(d_selectedPointsPtrs, h_dSelectedPointsPtrs, 
            blockCount * sizeof(T*), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());

        // Run warm-up/correctness iteration
        dispatchKernel<BLOCK_THREADS, POINTS_PER_THREAD, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, 
            DIM, STORE_TWO_PHASE>(
                d_in, numPoints, d_selectedPointsPtrs, d_selectedPointsCnt, selectOp, inStride, 
                outStride, 1);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Measure performance
        GpuTimer timer;
        float elapsedMillis;

        #ifdef RD_PROFILE
        cudaProfilerStart();
        #endif
        timer.Start();
        dispatchKernel<BLOCK_THREADS, POINTS_PER_THREAD, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, 
            DIM, STORE_TWO_PHASE>(
                d_in, numPoints, d_selectedPointsPtrs, d_selectedPointsCnt, selectOp, inStride, 
                outStride, g_iterations);
        timer.Stop();
        elapsedMillis = timer.ElapsedMillis();
        #ifdef RD_PROFILE
        cudaProfilerStop();
        #endif

        float   avgMillis           = elapsedMillis / g_iterations;
        int     selItemsCnt         = selectedPointsCnt * DIM * blockCount;
        // every block scans entire data set
        unsigned long long int  numBytes = sizeof(T) * blockCount * numPoints * DIM +              
        // storing selectedItems counters
                                        blockCount * sizeof(int) +                            
        // storing selected items
                                        selItemsCnt * sizeof(T);
        // conversion to GB/s
        float   gigaBandwidth       =   float(numBytes) / avgMillis / 1000.0f / 1000.0f;                    
        KernelResourceUsage resUsage(kernelPtr, BLOCK_THREADS);

        if (g_logPerfResults)
        {
            logValues(*g_logFile, DIM, BLOCK_THREADS, POINTS_PER_THREAD,
                std::string(rd::LoadModifierNameTraits<LOAD_MODIFIER>::name),
                std::string(rd::BlockTileIONameTraits<IO_BACKEND>::name),
                std::string(rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name),
                (STORE_TWO_PHASE) ? "true" : "false",
                selectRatio, avgMillis, gigaBandwidth,
                resUsage.prettyPrint());
            *g_logFile << std::endl;
        }
        logValues(std::cout, DIM, BLOCK_THREADS, POINTS_PER_THREAD,
            std::string(rd::LoadModifierNameTraits<LOAD_MODIFIER>::name),
            std::string(rd::BlockTileIONameTraits<IO_BACKEND>::name),
            std::string(rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name),
            (STORE_TWO_PHASE) ? "true" : "false",
            selectRatio, avgMillis, gigaBandwidth,
            resUsage.prettyPrint());
        std::cout << std::endl;

        // cleanup
        checkCudaErrors(cudaStreamDestroy(auxStream));
        for (int k = 0; k < blockCount; ++k)
        {
            if (h_dSelectedPointsPtrs[k]) checkCudaErrors(cudaFree(h_dSelectedPointsPtrs[k]));
        }

        if (d_selectedPointsPtrs) checkCudaErrors(cudaFree(d_selectedPointsPtrs));
        if (d_selectedPointsCnt) checkCudaErrors(cudaFree(d_selectedPointsCnt));
        if (h_dSelectedPointsPtrs) delete[] h_dSelectedPointsPtrs;
    }
    
    template < 
        typename EnableT,
        typename std::enable_if<
            std::is_same<EnableT, std::false_type>::value, int>::type * = nullptr>
    static void doImpl(
        int         ,
        float       ,
        T const *   ,
        int         ,
        int         ,
        T           ,
        int         )
    {
    }
};

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
void iterateKernelConf(
    int         numPoints,
    float       compare,
    T const *   d_in,
    int         inStride,
    int         outStride,
    T           selectRatio,
    int         selectedPointsCnt)
{

    #ifdef QUICK_TEST
    RunSelectIf<RD_BLOCK_SIZE, 4, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    #else
    RunSelectIf<RD_BLOCK_SIZE, 1, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    RunSelectIf<RD_BLOCK_SIZE, 2, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    RunSelectIf<RD_BLOCK_SIZE, 3, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    RunSelectIf<RD_BLOCK_SIZE, 4, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    RunSelectIf<RD_BLOCK_SIZE, 5, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    RunSelectIf<RD_BLOCK_SIZE, 6, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    RunSelectIf<RD_BLOCK_SIZE, 7, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    RunSelectIf<RD_BLOCK_SIZE, 8, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    RunSelectIf<RD_BLOCK_SIZE, 9, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    RunSelectIf<RD_BLOCK_SIZE,10, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    RunSelectIf<RD_BLOCK_SIZE,11, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    RunSelectIf<RD_BLOCK_SIZE,12, DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE,
        T>::impl(numPoints, compare, d_in, inStride, outStride, selectRatio, selectedPointsCnt);
    #endif
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
}

/**
 * @brief      Create if necessary and open log file. Allocate log file stream.
 */
template <typename T>
static void initializeLogFile()
{
    if (g_logPerfResults)
    {
        std::ostringstream logFileName;
        // append device name to log file
        logFileName << typeid(T).name() << "_" << getCurrDate() << "_" << g_devName << "_" 
                    << LOG_FILE_NAME_SUFFIX;

        std::string logFilePath = rd::findPath("../timings/", logFileName.str());
        g_logFile = new std::ofstream(logFilePath.c_str(), std::ios::out | std::ios::app);
        if (g_logFile->fail())
        {
            throw std::logic_error("Couldn't open file: " + logFileName.str());
        }

        // legend
        if (g_startBenchmark)
        {
            *g_logFile << "% ";
            logValue(*g_logFile, "dim", 10);
            logValue(*g_logFile, "blockSize", 10);
            logValue(*g_logFile, "itmPerThr", 10);
            logValue(*g_logFile, "loadModif", 10);
            logValue(*g_logFile, "ioBackend", 16);
            logValue(*g_logFile, "inMemLout", 10);
            logValue(*g_logFile, "twoPhase", 10);
            logValue(*g_logFile, "selRatio", 10);
            logValue(*g_logFile, "avgMillis", 12);
            logValue(*g_logFile, "GBytes", 12);
            logValue(*g_logFile, "resUsage", 10);
            *g_logFile << "\n";
        }
    }
}


/**
 * Reference selection problem solution.
 */
template <
    int         DIM,
    typename    T,
    typename    SelectOpT>
static int solve(
    T const *   h_in,
    int         numPoints,
    SelectOpT   selectOp)
{
    int selectedPointsCnt = 0;

    #ifdef RD_USE_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:selectedPointsCnt)
    #endif
    for (int k = 0; k < numPoints; ++k)
    {
        T const * point = h_in + k * DIM;

        if (selectOp(point))
        {
            selectedPointsCnt++;
        }
    }

    return selectedPointsCnt;
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
void prepareAndRunTest(
    T *         d_in,
    int         numPoints,
    float       selectRatio,
    int         selectedPointsCnt,
    T           compareItem)
{
    int inDataStride = (INPUT_MEM_LAYOUT == rd::COL_MAJOR) ? numPoints : 1;
    int outDataStride = (INPUT_MEM_LAYOUT == rd::COL_MAJOR) ? selectedPointsCnt : 1;

    // Run test kernel configurations
    iterateKernelConf<DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, STORE_TWO_PHASE>(
            numPoints, compareItem, d_in, inDataStride, outDataStride, selectRatio, 
            selectedPointsCnt); 
}

template <
    int                             DIM,
    cub::CacheLoadModifier          LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend     IO_BACKEND,
    rd::DataMemoryLayout            INPUT_MEM_LAYOUT,
    typename                        T>    
void testStoreAlgorithm(
    T *             d_in,
    int             numPoints,
    float           selectRatio,
    int             selectedPointsCnt,
    T               compareItem)
{
    /**
     * store two-phase (with selected items compaction in smem)
     */
    // prepareAndRunTest<DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, true, T>(
    //     d_in,  numPoints, selectRatio, selectedPointsCnt, compareItem);

    /**
     * store using warp-aggregated algorithm
     */
    prepareAndRunTest<DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, false, T>(
        d_in,  numPoints, selectRatio, selectedPointsCnt, compareItem);
}

template <
    int                             DIM,
    rd::gpu::BlockTileIOBackend     IO_BACKEND,
    rd::DataMemoryLayout            INPUT_MEM_LAYOUT,
    typename                        T>
void testLoadModifier(
    T *             d_in,
    int             numPoints,
    float           selectRatio,
    int             selectedPointsCnt,
    T               compareItem)
{
    testStoreAlgorithm<DIM, cub::LOAD_LDG, IO_BACKEND, INPUT_MEM_LAYOUT, T>(d_in, numPoints, 
        selectRatio, selectedPointsCnt, compareItem);
    // Cache streaming (likely to be accessed once)
    testStoreAlgorithm<DIM, cub::LOAD_CG, IO_BACKEND, INPUT_MEM_LAYOUT, T>(d_in, numPoints, 
        selectRatio, selectedPointsCnt, compareItem);
}

template <
    int                             DIM,
    rd::DataMemoryLayout            INPUT_MEM_LAYOUT,
    typename                        T>
void testIOBackend(
    T *             d_in,
    int             numPoints,
    float           selectRatio,
    int             selectedPointsCnt,
    T               compareItem)
{
    testLoadModifier<DIM, rd::gpu::IO_BACKEND_CUB,   INPUT_MEM_LAYOUT, T>(d_in, numPoints, 
        selectRatio, selectedPointsCnt, compareItem);
    /**
     * 06.06.2016 Trove version causes misalinged address errors while storing data from smem.
     */
    // testLoadModifier<DIM, rd::gpu::IO_BACKEND_TROVE, INPUT_MEM_LAYOUT, T>(d_in, numPoints, 
    // selectRatio, selectedPointsCnt, compareItem);
}

template <
    int         DIM,
    typename    T>
void testInputMemLayout(
    int     numPoints,
    float   selectRatio)
{
    if (g_logPerfResults)
    {
        *g_logFile << "% pointsNum=" << numPoints << std::endl;
    }

    // allocate host arrays
    T * h_in            = new T[numPoints * DIM];
    // Initialize input
    Initialize(h_in, numPoints * DIM);

    // Select a comparison value that is selectRatio through the space of [0,127]
    T compareItem;
    if (selectRatio <= 0.0)
    {
        compareItem = 0;        // select none
    }
    else if (selectRatio >= 1.0)
    {
        compareItem = 127;      // select all
    }
    else
    {
        compareItem = int(T(T(127) * selectRatio));
    }

    LessThan<DIM, T> selectOp(compareItem);
    int selectedPointsCnt = solve<DIM>(h_in, numPoints, selectOp);

    // Allocate device arrays
    T * d_in = nullptr;
    checkCudaErrors(cudaMalloc(&d_in, numPoints * DIM * sizeof(T)));

    rd::gpu::rdMemcpy<rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
        d_in, h_in, DIM, numPoints, DIM, DIM);
    testIOBackend<DIM, rd::ROW_MAJOR, T>(d_in, numPoints, selectRatio, selectedPointsCnt, 
        compareItem);

    rd::gpu::rdMemcpy<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
        d_in, h_in, DIM, numPoints, numPoints, DIM);
    testIOBackend<DIM, rd::COL_MAJOR, T>(d_in, numPoints, selectRatio, selectedPointsCnt, 
        compareItem);


    // cleanup

    if (h_in) delete[] h_in;
    if (d_in) checkCudaErrors(cudaFree(d_in));
}

template <
    typename       T>
void testDim(
    int     numPoints,
    T       selectRatio)
{
    if (g_logPerfResults)
    {
        initializeLogFile<T>();
    }

    #if defined(RD_DEBUG) || defined(RD_PROFILE) || defined(QUICK_TEST)
        std::cout << "\nTest DIM 2...\n\n";
        testInputMemLayout<10, T>(numPoints, selectRatio);
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
        std::cout << "\nTest DIM 7...\n\n";
        testInputMemLayout<7, T>(numPoints, selectRatio);
        std::cout << "\nTest DIM 8...\n\n";
        testInputMemLayout<8, T>(numPoints, selectRatio);
        std::cout << "\nTest DIM 9...\n\n";
        testInputMemLayout<9, T>(numPoints, selectRatio);
        std::cout << "\nTest DIM 10...\n\n";
        testInputMemLayout<10, T>(numPoints, selectRatio);
        std::cout << "\nTest DIM 11...\n\n";
        testInputMemLayout<11, T>(numPoints, selectRatio);
        std::cout << "\nTest DIM 12...\n\n";
        testInputMemLayout<12, T>(numPoints, selectRatio);
    #endif

    if (g_logPerfResults)
    {
        if (g_logFile) delete g_logFile;
    }
}


int main(int argc, char const **argv)
{
    
    int numPoints       = -1;
    float selectRatio   = -1.f;

    // Initialize command line
    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help")) 
    {
        printf("%s \n"
            "\t\t--np=<number of input points>\n"
            "\t\t[--d=<device id>]\n"
            "\t\t[--ratio=<selection ratio, default 0.5>]\n"
            "\t\t[--log <log performance results>]\n"
            "\t\t[--start <mark start of benchmark in log file>]\n"
            // "\t\t[--end <mark end of benchmark in log file>]\n"
            "\n", argv[0]);
        exit(0);
    }

    args.GetCmdLineArgument("np", numPoints);

    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", g_devId);
    }
    if (args.CheckCmdLineFlag("ratio")) 
    {
        args.GetCmdLineArgument("ratio", selectRatio);
    }
    if (args.CheckCmdLineFlag("log")) 
    {
        g_logPerfResults = true;
    }
    if (args.CheckCmdLineFlag("start")) 
    {
        g_startBenchmark = true;
    }
    // if (args.CheckCmdLineFlag("end")) 
    // {
    //     g_endBenchmark = true;
    // }


    checkCudaErrors(deviceInit(g_devId));

    // set device name for logging and drawing purposes
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, g_devId));
    g_devName = devProp.name;
    // read device SM count and determine max number of resident blocks per SM
    g_devSMCount = devProp.multiProcessorCount;

    //-----------------------------------------
    //  TESTS
    //-----------------------------------------

    if (numPoints < 0 ||
        selectRatio < 0)
    {
        std::cout << "Have to specify parameters! Rerun with --help for more "
            "informations.\n";
        exit(1);
    }
#ifdef QUICK_TEST

    const int DIM = 10;

    // allocate host arrays
    float * h_in = new float[numPoints * DIM];
    // Initialize input
    Initialize(h_in, numPoints * DIM);

    if (g_logPerfResults)
    {
        initializeLogFile<float>();
    }

    // Select a comparison value that is selectRatio through the space of [0,127]
    float compareItem;
    if (selectRatio <= 0.0)
    {
        compareItem = 0;        // select none
    }
    else if (selectRatio >= 1.0)
    {
        compareItem = 127;      // select all
    }
    else
    {
        compareItem = int(float(float(127) * selectRatio));
    }

    LessThan<DIM, float> selectOp(compareItem);
    int selectedPointsCnt = solve<DIM>(h_in, numPoints, selectOp);

    // Allocate device arrays
    float * d_in = nullptr;
    checkCudaErrors(cudaMalloc(&d_in, numPoints * DIM * sizeof(float)));
    rd::gpu::rdMemcpy<rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
        d_in, h_in, DIM, numPoints, DIM, DIM);

    prepareAndRunTest<DIM, cub::LOAD_LDG, rd::gpu::IO_BACKEND_CUB, rd::ROW_MAJOR, true>(
        d_in, numPoints, selectRatio, selectedPointsCnt, compareItem);
    prepareAndRunTest<DIM, cub::LOAD_LDG, rd::gpu::IO_BACKEND_CUB, rd::ROW_MAJOR, false>(
        d_in, numPoints, selectRatio, selectedPointsCnt, compareItem);

    rd::gpu::rdMemcpy<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
        d_in, h_in, DIM, numPoints, numPoints, DIM);    
    prepareAndRunTest<DIM, cub::LOAD_LDG, rd::gpu::IO_BACKEND_CUB, rd::COL_MAJOR, true>(
        d_in, numPoints, selectRatio, selectedPointsCnt, compareItem);
    prepareAndRunTest<DIM, cub::LOAD_LDG, rd::gpu::IO_BACKEND_CUB, rd::COL_MAJOR, false>(
        d_in, numPoints, selectRatio, selectedPointsCnt, compareItem);

    std::cout << rd::HLINE << std::endl;
    
    // cleanup
    if (h_in) delete[] h_in;
    if (d_in) checkCudaErrors(cudaFree(d_in));
    
#else    
    #ifndef RD_TEST_DBL_PREC
    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    testDim<float>(numPoints, selectRatio);
    #else
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE: " << std::endl;
    testDim<double>(numPoints, selectRatio);
    #endif
    std::cout << rd::HLINE << std::endl;
#endif

    checkCudaErrors(deviceReset());

    std::cout << "END!" << std::endl;
    return 0;
}
