/**
 * @file benchmark_find_bounds.cu
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
#define CUB_STDERR
#define BLOCK_TILE_LOAD_V4 1

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <limits>

#include <cmath>
#include <utility>

#include <helper_cuda.h>
#ifdef RD_PROFILE
#include <cuda_profiler_api.h>
#endif

// #include <thrust/extrema.h>
// #include <thrust/device_vector.h>

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/bounding_box.hpp"
#include "rd/utils/name_traits.hpp"

#include "rd/gpu/util/dev_memcpy.cuh"
#include "rd/gpu/device/device_find_bounds.cuh"

#include "tests/test_util.hpp"
#include "cub/test_util.h"


//------------------------------------------------------------
//  GLOBAL CONSTANTS / VARIABLES
//------------------------------------------------------------

static const std::string LOG_FILE_NAME_SUFFIX = "_find_bounds_timings_v4.txt";

std::ofstream * g_logFile           = nullptr;
bool            g_drawResultsGraph  = false;
bool            g_logPerfResults    = false;
std::string     g_devName;
static int      g_devId             = 0;

std::vector<std::vector<float>> g_bestPerf;

#if defined(RD_PROFILE) || defined(RD_DEBUG) || defined(QUICK_TEST)
static const int g_iterations = 1;
#else
static const int g_iterations = 100;
#endif

#ifdef QUICK_TEST
static const int MAX_TEST_DIM = 4;
#else
static const int MAX_TEST_DIM = 12;
#endif

//------------------------------------------------------------
//  LOG FILE INITIALIZATION
//------------------------------------------------------------

template <typename T>
static void initializeLogFile()
{
    if (g_logPerfResults)
    {
        std::ostringstream logFileName;
        logFileName << getCurrDate() << "_" <<
            g_devName << "_" << LOG_FILE_NAME_SUFFIX;

        std::string logFilePath = rd::findPath("../timings/", logFileName.str());
        g_logFile = new std::ofstream(logFilePath.c_str(), std::ios::out | std::ios::app);
        if (g_logFile->fail())
        {
            throw std::logic_error("Couldn't open file: " + logFileName.str());
        }

        *g_logFile << "%" << rd::HLINE << std::endl;
        *g_logFile << "% " << typeid(T).name() << std::endl;
        *g_logFile << "%" << rd::HLINE << std::endl;
        
        // legend
        *g_logFile << "% "; 
        logValue(*g_logFile, "dim", 10);
        logValue(*g_logFile, "inPointsNum", 11);
        logValue(*g_logFile, "loadModifier", 12);
        logValue(*g_logFile, "ioBackend", 10);
        logValue(*g_logFile, "inMemLayout", 11);
        logValue(*g_logFile, "ptsPerThr", 10);
        logValue(*g_logFile, "blockThrs", 10);
        logValue(*g_logFile, "avgMillis", 10);
        logValue(*g_logFile, "GBytes", 10);
        logValue(*g_logFile, "ResUsage1", 10);
        logValue(*g_logFile, "ResUsage2", 10);
        *g_logFile << "\n";
        g_logFile->flush();
    }
}

//------------------------------------------------------------
//  REFERENCE FIND BOUNDS
//------------------------------------------------------------

template <int DIM, typename T>
bool compareBounds(
    rd::BoundingBox<T> const & bboxGold,
    T (&h_minBounds)[DIM],
    T (&h_maxBounds)[DIM])
{
    bool result = true;
    for (int d = 0; d < DIM; ++d)
    {
        #ifdef RD_DEBUG
        std::cout << "min[" << d << "] gpu: " << std::right << std::fixed << std::setw(12) 
                << std::setprecision(8) << h_minBounds[d] <<", cpu: "<< bboxGold.min(d) <<"\n";
        std::cout << "max[" << d << "] gpu: " << std::right << std::fixed << std::setw(12) 
                << std::setprecision(8) << h_maxBounds[d] <<", cpu: "<< bboxGold.max(d) <<"\n";
        #endif
        
        if (h_minBounds[d] != bboxGold.min(d))
        {
            result = false;
            std::cout << "ERROR!: min[" << d << "] is: "<< h_minBounds[d] 
                    <<", should be: "<< bboxGold.min(d) <<"\n";
        } 
        if (h_maxBounds[d] != bboxGold.max(d)) 
        {
            result = false;
            std::cout << "ERROR!: max[" << d << "] is: "<< h_maxBounds[d] 
                    <<", should be: "<< bboxGold.max(d) <<"\n";
        }
    }
    return result;
}

//------------------------------------------------------------
//  KERNEL DISPATCH
//------------------------------------------------------------

typedef std::pair<KernelResourceUsage, KernelResourceUsage> fbKernelsResUsageT;

template <
    int                         BLOCK_THREADS,
    int                         POINTS_PER_THREAD,
    cub::BlockReduceAlgorithm   REDUCE_ALGORITHM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    int                         DIM,
    typename                    SampleT,
    typename                    OffsetT>
void  dispatchFindBoundsKernel(
    void *                      d_tempStorage,              
    size_t &                    tempStorageBytes,           
    SampleT const *             d_in,
    SampleT *                   d_outMin,
    SampleT *                   d_outMax,
    int                         numPoints,
    OffsetT                     stride,
    int                         iterations,
    fbKernelsResUsageT *        resUsage = nullptr,
    bool                        debugSynchronous = false)
{
    typedef rd::gpu::detail::AgentFindBoundsPolicy<
        BLOCK_THREADS,
        POINTS_PER_THREAD,
        REDUCE_ALGORITHM,
        LOAD_MODIFIER,
        IO_BACKEND> AgentFindBoundsPolicyT;

    typedef rd::gpu::DispatchFindBounds<
        SampleT,
        OffsetT,
        DIM,
        INPUT_MEM_LAYOUT> DispatchFindBoundsT;

    typename DispatchFindBoundsT::KernelConfig findBoundsConfig;
    findBoundsConfig.blockThreads = BLOCK_THREADS;
    findBoundsConfig.itemsPerThread = POINTS_PER_THREAD;

    auto firstPassKernelPtr = rd::gpu::detail::deviceFindBoundsKernelFirstPass<
        AgentFindBoundsPolicyT, DIM, INPUT_MEM_LAYOUT, SampleT, OffsetT>;

    auto secondPassKernelPtr = rd::gpu::detail::deviceFindBoundsKernelSecondPass<
        AgentFindBoundsPolicyT, DIM, SampleT, OffsetT>;

    int ptxVersion;
    checkCudaErrors(cub::PtxVersion(ptxVersion));

    if (resUsage != nullptr)
    {
        resUsage->first = KernelResourceUsage(firstPassKernelPtr, BLOCK_THREADS);
        resUsage->second = KernelResourceUsage(secondPassKernelPtr, BLOCK_THREADS);
    }

    for (int i = 0; i < iterations; ++i)
    {
        CubDebugExit(DispatchFindBoundsT::invoke(
            d_tempStorage,
            tempStorageBytes,
            d_in,
            d_outMin,
            d_outMax,
            numPoints,
            stride,
            nullptr,
            debugSynchronous,
            firstPassKernelPtr,
            secondPassKernelPtr,
            findBoundsConfig,
            ptxVersion));
    }

}

//------------------------------------------------------------
//  Benchmark helper structures
//------------------------------------------------------------

struct KernelParametersConf
{
    int                         BLOCK_THREADS;
    int                         POINTS_PER_THREAD;
    int                         DIM;
    // cub::BlockReduceAlgorithm   REDUCE_ALGORITHM,
    cub::CacheLoadModifier      LOAD_MODIFIER;
    rd::gpu::BlockTileIOBackend IO_BACKEND;
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT;
    float                       avgMillis;
    float                       gigaBandwidth;

    KernelParametersConf()
    :
        LOAD_MODIFIER(cub::LOAD_DEFAULT),
        INPUT_MEM_LAYOUT(rd::ROW_MAJOR),
        IO_BACKEND(rd::gpu::IO_BACKEND_CUB)
    {}

    KernelParametersConf(
        int                         _DIM,
        cub::CacheLoadModifier      _LOAD_MODIFIER,
        rd::gpu::BlockTileIOBackend _IO_BACKEND,
        rd::DataMemoryLayout        _INPUT_MEM_LAYOUT)
    :
        DIM(_DIM),
        LOAD_MODIFIER(_LOAD_MODIFIER),
        IO_BACKEND(_IO_BACKEND),
        INPUT_MEM_LAYOUT(_INPUT_MEM_LAYOUT)
    {}

};

typedef std::pair<float, float> KernelPerfT;


//------------------------------------------------------------
//  TEST CONFIGURATION AND RUN
//------------------------------------------------------------

template <
    int                         BLOCK_THREADS,
    int                         POINTS_PER_THREAD,
    int                         DIM,
    cub::BlockReduceAlgorithm   REDUCE_ALGORITHM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    typename                    SampleT>
KernelPerfT runDeviceFindBounds(
    int                                 pointCnt,
    SampleT const *                     d_in,
    SampleT *                           d_outMin,
    SampleT *                           d_outMax,
    int                                 stride,
    rd::BoundingBox<SampleT> const &    bboxGold)
{
    // Allocate temporary storage
    void            *d_tempStorage = NULL;
    size_t          tempStorageBytes = 0;

    fbKernelsResUsageT resUsage;

    dispatchFindBoundsKernel<BLOCK_THREADS, POINTS_PER_THREAD, REDUCE_ALGORITHM,
        LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, DIM>(
        d_tempStorage, tempStorageBytes, d_in, d_outMin, d_outMax, pointCnt, 
        stride, 1, &resUsage, true); 

    checkCudaErrors(cudaMalloc((void**)&d_tempStorage, tempStorageBytes));

    //---------------------------------------------
    // Run warm-up/correctness iteration
    //---------------------------------------------
    dispatchFindBoundsKernel<BLOCK_THREADS, POINTS_PER_THREAD, REDUCE_ALGORITHM, 
        LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, DIM>(
        d_tempStorage, tempStorageBytes, d_in, d_outMin, d_outMax, pointCnt, 
        stride, 1, nullptr, true); 
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    SampleT h_minBounds[DIM];
    SampleT h_maxBounds[DIM];
    checkCudaErrors(cudaMemcpy(h_minBounds, d_outMin, DIM * sizeof(SampleT), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_maxBounds, d_outMax, DIM * sizeof(SampleT), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    bool result = compareBounds(bboxGold, h_minBounds, h_maxBounds);
    if (result)
    {
        std::cout << ">>>> CORRECT!\n";
    }
    else
    {
        std::cout << ">>>> ERROR!" << std::endl;
        // clean-up
        checkCudaErrors(cudaFree(d_tempStorage));
        return std::make_pair(1e10f, -1.0f);
    }

    //---------------------------------------------
    // Measure performance
    //---------------------------------------------

    GpuTimer timer;
    float elapsedMillis;

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();

    dispatchFindBoundsKernel<BLOCK_THREADS, POINTS_PER_THREAD, REDUCE_ALGORITHM, LOAD_MODIFIER, 
            IO_BACKEND, INPUT_MEM_LAYOUT, DIM>(
        d_tempStorage, tempStorageBytes, d_in, d_outMin, d_outMax, pointCnt, stride, 
        g_iterations); 

    timer.Stop();
    elapsedMillis = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    double avgMillis = double(elapsedMillis) / g_iterations;
    double gigaBandwidth = double(pointCnt * DIM * sizeof(SampleT)) / avgMillis / 1000.0 / 1000.0;

    if (g_logPerfResults)
    {
        logValues(*g_logFile,
            DIM,
            pointCnt,
            std::string(rd::LoadModifierNameTraits<LOAD_MODIFIER>::name),
            std::string(rd::BlockTileIONameTraits<IO_BACKEND>::name),
            std::string(rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name),
            POINTS_PER_THREAD, 
            BLOCK_THREADS,
            avgMillis, 
            gigaBandwidth,
            resUsage.first.prettyPrint(),
            resUsage.second.prettyPrint());
        *g_logFile << "\n";
    }

    logValues(std::cout,
            DIM,
            pointCnt,
            std::string(rd::LoadModifierNameTraits<LOAD_MODIFIER>::name),
            std::string(rd::BlockTileIONameTraits<IO_BACKEND>::name),
            std::string(rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name),
            POINTS_PER_THREAD, 
            BLOCK_THREADS,
            avgMillis, 
            gigaBandwidth,
            resUsage.first.prettyPrint(),
            resUsage.second.prettyPrint());
    std::cout << std::endl;

    // clean-up
    if (d_tempStorage != nullptr) checkCudaErrors(cudaFree(d_tempStorage));

    return std::make_pair(avgMillis, gigaBandwidth);
}

//------------------------------------------------------------
//  TEST SPECIALIZATIONS
//------------------------------------------------------------

/*
 *  Specialization for testing different points per thread
 */
template <
    int                         BLOCK_THREADS,
    int                         DIM,
    cub::BlockReduceAlgorithm   REDUCE_ALGORITHM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    typename                    SampleT>
KernelParametersConf testPointsPerThread(
    int                                 poitnCnt,
    SampleT const *                     d_in,
    SampleT *                           d_outMin,
    SampleT *                           d_outMax,
    int                                 stride,
    rd::BoundingBox<SampleT> const &    bboxGold)
{
    KernelParametersConf bestKernelParams(DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT);
    bestKernelParams.BLOCK_THREADS = BLOCK_THREADS;
    KernelPerfT bestPerf = std::make_pair(1e10f, -1.0f);

    auto processResults = [&bestPerf, &bestKernelParams](int ppt, KernelPerfT kp)
    {
        if (kp.second > bestPerf.second)
        {
            bestPerf.first = kp.first;
            bestPerf.second = kp.second;
            bestKernelParams.avgMillis = kp.first;
            bestKernelParams.gigaBandwidth = kp.second;
            bestKernelParams.POINTS_PER_THREAD = ppt;
        }
    };

    #ifdef QUICK_TEST
    bestKernelParams = runDeviceFindBounds<BLOCK_THREADS, 4, DIM, REDUCE_ALGORITHM, 
            LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT>( 
        poitnCnt, d_in, d_outMin, d_outMax, stride, bboxGold);
    #else

    #define runTest(ppt) processResults(ppt, runDeviceFindBounds<BLOCK_THREADS, ppt, DIM, \
        REDUCE_ALGORITHM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT>( \
        poitnCnt, d_in, d_outMin, d_outMax, stride, bboxGold));
    
    runTest(1);
    runTest(2);
    runTest(3);
    runTest(4);
    runTest(5);
    runTest(6);
    runTest(7);
    runTest(8);
    runTest(9);
    runTest(10);
    runTest(11);
    runTest(12);

    #undef runTest
    #endif

    if (g_logPerfResults)
    {
        *g_logFile << "\n%------------------------------------------------"
            "\n% best conf: "
            "\n%------------------------------------------------\n";
        logValues(*g_logFile,
            bestKernelParams.BLOCK_THREADS,
            bestKernelParams.POINTS_PER_THREAD,
            bestKernelParams.avgMillis,
            bestKernelParams.gigaBandwidth,
            rd::getLoadModifierName(bestKernelParams.LOAD_MODIFIER),
            rd::getRDDataMemoryLayout(bestKernelParams.INPUT_MEM_LAYOUT),
            rd::getRDTileIOBackend(bestKernelParams.IO_BACKEND));
        *g_logFile <<"\n\n\n\n"; 
    }

    std::cout << "\n%------------------------------------------------"
            "\n% best conf: "
            "\n%------------------------------------------------\n";
        logValues(std::cout,
            bestKernelParams.BLOCK_THREADS,
            bestKernelParams.POINTS_PER_THREAD,
            bestKernelParams.avgMillis,
            bestKernelParams.gigaBandwidth,
            rd::getLoadModifierName(bestKernelParams.LOAD_MODIFIER),
            rd::getRDDataMemoryLayout(bestKernelParams.INPUT_MEM_LAYOUT),
            rd::getRDTileIOBackend(bestKernelParams.IO_BACKEND));
        std::cout <<"\n\n\n"; 

    return bestKernelParams;
}

/*
 *  Specialization for testing different number of threads in a block 
 */
template <
    int                         DIM,
    cub::BlockReduceAlgorithm   REDUCE_ALGORITHM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    typename                    SampleT>
KernelParametersConf testBlockSize(
    int                                 pointCnt,
    SampleT const *                     d_in,
    SampleT *                           d_outMin,
    SampleT *                           d_outMax,
    int                                 stride,
    rd::BoundingBox<SampleT> const &    bboxGold)
{
    KernelParametersConf bestKernelParams;

    auto checkBestConf = [&bestKernelParams](KernelParametersConf kp)
    {
        if (kp.gigaBandwidth > bestKernelParams.gigaBandwidth)
        {
            bestKernelParams = kp;
        }
    };

    checkBestConf(testPointsPerThread<RD_BLOCK_SIZE, DIM, REDUCE_ALGORITHM, 
        LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT>( 
        pointCnt, d_in, d_outMin, d_outMax, stride, bboxGold));
    
    return bestKernelParams;
}

//------------------------------------------------------------
//  TEST SPECIFIED VARIANTS
//------------------------------------------------------------

template <int DIM, typename T>
void test(
    std::vector<T> && inData,
    int pointCnt)
{
	T *d_PRowMajor, *d_PColMajor;
    T *d_minBounds, *d_maxBounds;
    size_t pitch;
    int d_PColStride;

    // allocate containers
    checkCudaErrors(cudaMalloc((void**)&d_PRowMajor, pointCnt * DIM * sizeof(T)));
    checkCudaErrors(cudaMallocPitch((void**)&d_PColMajor, &pitch, pointCnt * sizeof(T), DIM));
    checkCudaErrors(cudaMalloc((void**)&d_minBounds, DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_maxBounds, DIM * sizeof(T)));

    d_PColStride = pitch / sizeof(T);

    // initialize data
    rd::gpu::rdMemcpy<rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_PRowMajor, inData.data(), DIM, pointCnt, DIM, DIM);    
    rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_PColMajor, inData.data(), DIM, pointCnt, pitch, DIM * sizeof(T));    

    //---------------------------------------------------
    //               REFERENCE BOUNDING BOX
    //---------------------------------------------------
    rd::BoundingBox<T> bboxGold(inData.data(), pointCnt, DIM);
    bboxGold.calcDistances();
    #ifdef RD_DEBUG
    bboxGold.print();
    #endif
    //---------------------------------------------------
    //               REFERENCE BOUNDING BOX using Thrust
    //---------------------------------------------------

    
    //---------------------------------------------------
    //               GPU BOUNDING BOX
    //---------------------------------------------------

    #ifdef QUICK_TEST

    const int BLOCK_THREADS = 128;
    const int POINTS_PER_THREAD = 4;
    runDeviceFindBounds<BLOCK_THREADS, POINTS_PER_THREAD, DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 
        cub::LOAD_LDG, rd::gpu::IO_BACKEND_CUB, rd::COL_MAJOR>(
            pointCnt, d_PColMajor, d_minBounds, d_maxBounds, d_PColStride, bboxGold);

    runDeviceFindBounds<BLOCK_THREADS, POINTS_PER_THREAD, DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 
        cub::LOAD_LDG, rd::gpu::IO_BACKEND_CUB, rd::ROW_MAJOR>(
            pointCnt, d_PRowMajor, d_minBounds, d_maxBounds, DIM, bboxGold);

    runDeviceFindBounds<BLOCK_THREADS, POINTS_PER_THREAD, DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 
        cub::LOAD_LDG, rd::gpu::IO_BACKEND_TROVE, rd::COL_MAJOR>(
            pointCnt, d_PColMajor, d_minBounds, d_maxBounds, d_PColStride, bboxGold);

    runDeviceFindBounds<BLOCK_THREADS, POINTS_PER_THREAD, DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 
        cub::LOAD_LDG, rd::gpu::IO_BACKEND_TROVE, rd::ROW_MAJOR>(
            pointCnt, d_PRowMajor, d_minBounds, d_maxBounds, DIM, bboxGold);
    #else

    std::cout << "\n" << rd::HLINE << "\n";

    testBlockSize<DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_LDG, 
        rd::gpu::IO_BACKEND_CUB, rd::COL_MAJOR>(pointCnt, d_PColMajor, d_minBounds, d_maxBounds,
            d_PColStride, bboxGold);
     
    testBlockSize<DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_LDG, 
        rd::gpu::IO_BACKEND_CUB, rd::ROW_MAJOR>(pointCnt, d_PRowMajor, d_minBounds, d_maxBounds,
            DIM, bboxGold);
    
    std::cout << "\n" << rd::HLINE << "\n";
     
    testBlockSize<DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_LDG, 
        rd::gpu::IO_BACKEND_TROVE, rd::COL_MAJOR>(pointCnt, d_PColMajor, d_minBounds, d_maxBounds,
            d_PColStride, bboxGold);
     
    testBlockSize<DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_LDG, 
        rd::gpu::IO_BACKEND_TROVE, rd::ROW_MAJOR>(pointCnt, d_PRowMajor, d_minBounds, d_maxBounds,
            DIM, bboxGold);
     
    std::cout << "\n" << rd::HLINE << "\n";
    #endif

    //---------------------------------------------------
    // clean-up
    checkCudaErrors(cudaFree(d_PRowMajor));
    checkCudaErrors(cudaFree(d_PColMajor));
    checkCudaErrors(cudaFree(d_minBounds));
    checkCudaErrors(cudaFree(d_maxBounds));
}

struct IterateDimensions
{
    template <typename D, typename T>
    static void impl(
        D   idx,
        int pointCnt,
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
                << " pointsNum: " << pointCnt << "\n";
        }

        test<D::value>(pc.extractPart(pointCnt, idx), pointCnt);
    }
};

/**
 * @brief Test detection time & quality relative to point dimension
 */
template <
    int          DIM,
    typename     T>
struct TestDimensions
{
    static void impl(
        PointCloud<T> & pc,
        int pointCnt)
    {
        static_assert(DIM != 0, "DIM equal to zero!\n");

        initializeLogFile<T>();
        pc.pointCnt_ = pointCnt;
        pc.initializeData();

        std::cout << rd::HLINE << std::endl;
        std::cout << ">>>> Dimension: " << DIM << "D\n";
        
        if (g_logPerfResults)
        {
            T a, b;
            pc.getCloudParameters(a, b);
            *g_logFile << "%>>>> Dimension: " << DIM << "D\n"
                << "% a: " << a << " b: " << b << " s: " << pc.stddev_ 
                << " pointsNum: " << pointCnt << "\n";
        }
        test<DIM>(pc.extractPart(pointCnt, DIM), pointCnt);
        
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
        int pointCnt)
    {
        initializeLogFile<T>();
        pc.pointCnt_ = pointCnt;
        pc.dim_ = MAX_TEST_DIM;
        pc.initializeData();

        StaticFor<1, MAX_TEST_DIM, IterateDimensions>::impl(pointCnt, pc);

        // clean-up
        if (g_logPerfResults)
        {
            g_logFile->close();
            delete g_logFile;
        }
    }
};

int main(int argc, char const **argv)
{
    float stddev = -1.f, segLength = -1.f;
    int pointCnt = -1;

    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help")) 
    {
        printf("%s \n"
            "\t\t[--log <log performance results>]\n"
            "\t\t[--rGraphs <draws perf resuls graphs >]\n"
            "\t\t[--segl <generated N-dimensional segment length>]\n"
            "\t\t[--stddev <standard deviation of generated samples>]\n"
            "\t\t[--size <number of points>]\n"
            "\t\t[--d=<device id>]\n"
            "\n", argv[0]);
        exit(0);
    }

    if (args.CheckCmdLineFlag("log"))
    {
        g_logPerfResults = true;
    }
    if (args.CheckCmdLineFlag("segl"))
    {
        args.GetCmdLineArgument("segl", segLength);
    }
    if (args.CheckCmdLineFlag("stddev"))
    {
        args.GetCmdLineArgument("stddev", stddev);
    }
    if (args.CheckCmdLineFlag("size"))
    {
        args.GetCmdLineArgument("size", pointCnt);
    }    
    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", g_devId);
    }
    if (args.CheckCmdLineFlag("rGraphs"))
    {
        g_drawResultsGraph = true;
    }

    checkCudaErrors(deviceInit(g_devId));

    // set device name for logging and drawing purposes
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, g_devId));
    g_devName = devProp.name;

    if (pointCnt    < 0 ||
        segLength   < 0 ||
        stddev      < 0)
    {
        std::cout << "Have to specify parameters! Rerun with --help for help.\n";
        exit(1);
    }
    #ifdef QUICK_TEST
        const int dim = 11;

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t float: "  
                    << "\n//------------------------------------------\n";
        PointCloud<float> && fpc = SegmentPointCloud<float>(segLength, pointCnt, dim, stddev);
        TestDimensions<dim, float>::impl(fpc, pointCnt);

        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t double: "  
        //             << "\n//------------------------------------------\n";
        // PointCloud<double> && dpc = SegmentPointCloud<double>(segLength, pointCnt, dim, stddev);
        // TestDimensions<dim, double>::impl(dpc, pointCnt);
    #else
        #ifndef RD_DOUBLE_PRECISION
        // --size=1000000 --segl=100 --stddev=2.17 --rGraphs --log --d=0
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) float: "  
                    << "\n//------------------------------------------\n";
        PointCloud<float> && fpc2 = SegmentPointCloud<float>(segLength, pointCnt, 0, stddev);
        TestDimensions<0, float>::impl(fpc2, pointCnt);
        #else
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) double: "  
                    << "\n//------------------------------------------\n";
        PointCloud<double> && dpc2 = SegmentPointCloud<double>(segLength, pointCnt, 0, stddev);
        TestDimensions<0, double>::impl(dpc2, pointCnt);
        #endif
    #endif

    checkCudaErrors(deviceReset());

    std::cout << "END!" << std::endl;
    return 0;
}
