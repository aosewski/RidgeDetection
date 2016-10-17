/**
 * @file benchmark_spatial_histogram.cu
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
#include <cuda_profiler_api.h>
#endif

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


#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/bounding_box.hpp"
#include "rd/utils/memory.h" 
#include "rd/utils/histogram.hpp"
#include "rd/utils/name_traits.hpp"

#include "rd/gpu/util/dev_memcpy.cuh"
#include "rd/gpu/device/device_spatial_histogram.cuh"

#include "tests/test_util.hpp"
#include "cub/test_util.h"

//------------------------------------------------------------
//  GLOBAL CONSTANTS / VARIABLES
//------------------------------------------------------------

static const std::string LOG_FILE_NAME_SUFFIX = "_histogram-timings_v4.txt";

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

#if !defined(QUICK_TEST)
static const float              g_graphColStep  = 0.3f;
// group columns count 
static const int                g_graphNCol     = 4;     
// number of dimensions to plot
static const int                g_graphNGroups  = MAX_TEST_DIM;     
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
        logValue(*g_logFile, "numBins", 10);
        logValue(*g_logFile, "loadModifier", 12);
        logValue(*g_logFile, "ioBackend", 10);
        logValue(*g_logFile, "inMemLayout", 11);
        logValue(*g_logFile, "ptsPerThr", 10);
        logValue(*g_logFile, "blockThrs", 10);
        logValue(*g_logFile, "avgMillis", 10);
        logValue(*g_logFile, "GBytes", 10);
        *g_logFile << "\n";
        g_logFile->flush();
    }
}

//------------------------------------------------------------
//  REFERENCE HISTOGRAM
//------------------------------------------------------------

template <typename T>
struct HistogramMapFuncGold
{
    rd::BoundingBox<T> const &bb;

    HistogramMapFuncGold(rd::BoundingBox<T> const &bb)
    :
        bb(bb)
    {
    }

    size_t operator()(
        T const *                     sample,
        std::vector<size_t> const &   binsCnt)
    {
        size_t bin = 0;
        size_t binIdx;            
        for (size_t i = 0; i < bb.dim; ++i)
        {
            // get sample's bin [x,y,z...n] idx
            /*
             * translate each sample coordinate to the common origin (by distracting minimum)
             * then divide shifted coordinate by current dimension bin width and get the 
             * floor of this value (counting from zero!) which is our bin idx we search for.
             */
            if (bb.dist[i] < std::numeric_limits<T>::epsilon())
            {
                binIdx = 0;
            }
            else
            {
                T normCord = std::abs(sample[i] - bb.min(i));
                T step = bb.dist[i] / binsCnt[i];

                if (std::abs(normCord - bb.dist[i]) <= std::numeric_limits<T>::epsilon())
                {
                    binIdx = binsCnt[i]-1;
                }
                else
                {
                    binIdx = std::floor(normCord / step);
                }
            }

            /*
             * Calculate global idx value linearizing bin idx
             * idx = k_0 + sum_{i=2}^{dim}{k_i mul_{j=i-1}^{1}bDim_j}
             */
            size_t tmp = 1;
            for (int j = (int)i - 1; j >= 0; --j)
            {
                tmp *= binsCnt[j];
            }
            bin += binIdx*tmp;
        }
        return bin;
    }
};

template <typename T>
void histogramGold(
    int pointCnt,
    T const *P,
    std::vector<size_t>const &binsCnt,
    rd::BoundingBox<T>const & bbox,
    rd::Histogram<T> &hist)
{
    HistogramMapFuncGold<T> mapFunc(bbox);

    hist.setBinCnt(binsCnt);
    hist.getHist(P, pointCnt, mapFunc);
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
    typename                    SampleT,
    typename                    CounterT,
    typename                    OffsetT,
    typename                    T>
void dispatchSpatialHistogramKernel(
    void *                      d_tempStorage,              
    size_t &                    tempStorageBytes,           
    SampleT const *             d_in,
    int                         numPoints,
    CounterT *                  d_outHistogram,
    OffsetT                     stride,
    int const                   (&binsCnt)[DIM],
    rd::gpu::BoundingBox<DIM, T> const & bbox,
    int                         iterations,
    bool                        useGmemPrivHist = true,
    bool                        debugSynchronous = false)
{
    typedef rd::gpu::detail::AgentSpatialHistogramPolicy<
        BLOCK_THREADS,
        POINTS_PER_THREAD,
        LOAD_MODIFIER,
        IO_BACKEND> AgentSpatialHistogramPolicyT;

    typedef rd::gpu::DispatchSpatialHistogram<
        SampleT,
        CounterT,
        OffsetT,
        DIM,
        INPUT_MEM_LAYOUT> DispatchSpatialHistogramT;

    typedef typename DispatchSpatialHistogramT::Transform PointDecodeOpT;
    PointDecodeOpT pointDecodeOp(binsCnt, bbox);

    typename DispatchSpatialHistogramT::KernelConfig histogramConfig;
    histogramConfig.blockThreads = BLOCK_THREADS;
    histogramConfig.itemsPerThread = POINTS_PER_THREAD;

    int numBins = 1;
    for (int d = 0; d < DIM; ++d)
    {
        numBins *= binsCnt[d];
    }

    auto kernelPtr = (useGmemPrivHist) ?
        rd::gpu::detail::deviceSpatialHistogramKernel<AgentSpatialHistogramPolicyT, 
            INPUT_MEM_LAYOUT, DIM, SampleT, CounterT, OffsetT, PointDecodeOpT, false, true> 
        :
        rd::gpu::detail::deviceSpatialHistogramKernel<AgentSpatialHistogramPolicyT, 
            INPUT_MEM_LAYOUT, DIM, SampleT, CounterT, OffsetT, PointDecodeOpT, false, false>;


    for (int i = 0; i < iterations; ++i)
    {
        CubDebugExit(DispatchSpatialHistogramT::invoke(
            d_tempStorage,
            tempStorageBytes,
            d_in,
            numPoints,
            d_outHistogram,
            numBins,
            stride,
            pointDecodeOp,
            false,
            useGmemPrivHist,
            nullptr,
            debugSynchronous,
            kernelPtr,
            histogramConfig));
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
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    typename                    T>
KernelPerfT runDeviceSpatialHistogram(
    int                             pointCnt,
    T const *                       d_P,
    int *                           d_hist,
    int                             stride,
    std::vector<size_t>const &      binsCnt,
    size_t                          numBins,
    rd::BoundingBox<T> const &      bboxGold,
    std::vector<int> const &        histGold)
{
    /*
     *  prepare kernel parameters
     */
    typedef int AliasedBinCnt[DIM];
    AliasedBinCnt aliasedBinCnt;
    rd::gpu::BoundingBox<DIM, T> d_bbox;

    for (int d = 0; d < DIM; ++d)
    {
        d_bbox.bbox[d * 2]      = bboxGold.min(d);
        d_bbox.bbox[d * 2 + 1]  = bboxGold.max(d);
        d_bbox.dist[d]          = bboxGold.dist[d];
        aliasedBinCnt[d]        = binsCnt[d];
    }

    // Allocate temporary storage
    void            *d_tempStorage = nullptr;
    size_t          tempStorageBytes = 0;

    bool useGmemPrivHist = numBins < 1 << 20;

    if (!useGmemPrivHist)
    {
        std::cout << "Fallback to simple strategy; numBins: " << numBins << std::endl;
    }

    if (useGmemPrivHist)
    {
        dispatchSpatialHistogramKernel<BLOCK_THREADS, POINTS_PER_THREAD, 
            LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, DIM>(
                d_tempStorage, tempStorageBytes, d_P, pointCnt, d_hist, 
                stride, aliasedBinCnt, d_bbox, 1, useGmemPrivHist, true); 
        checkCudaErrors(cudaMalloc((void**)&d_tempStorage, tempStorageBytes));
    }

    // Run warm-up/correctness iteration
    dispatchSpatialHistogramKernel<BLOCK_THREADS, POINTS_PER_THREAD, LOAD_MODIFIER, 
        IO_BACKEND, INPUT_MEM_LAYOUT, DIM>( d_tempStorage, tempStorageBytes, d_P, 
            pointCnt, d_hist, stride, aliasedBinCnt, d_bbox, 1, useGmemPrivHist, true);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << ">>>> Check results... ";
    int result = CompareDeviceResults(histGold.data(), d_hist, numBins);
    if (!result)
    {
        std::cout << ".... CORRECT!\n";
    }
    else
    {
        std::cout << ".... ERROR!" << std::endl;
        // clean-up
        if (d_tempStorage != nullptr) checkCudaErrors(cudaFree(d_tempStorage));
        return std::make_pair(1e10f, -1.0f);
    }

    //---------------------------------------------------------------------
    // Measure performance
    //---------------------------------------------------------------------

    GpuTimer timer;
    float elapsedMillis;

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();

    dispatchSpatialHistogramKernel<BLOCK_THREADS, POINTS_PER_THREAD, LOAD_MODIFIER, 
        IO_BACKEND, INPUT_MEM_LAYOUT, DIM>( d_tempStorage, tempStorageBytes, d_P, 
            pointCnt, d_hist, stride, aliasedBinCnt, d_bbox, g_iterations, useGmemPrivHist);

    timer.Stop();
    elapsedMillis = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    double avgMillis = double(elapsedMillis) / g_iterations;
    double gigaBandwidth = double(pointCnt * DIM * sizeof(T) + pointCnt * sizeof(int)) / 
        avgMillis / 1000.0 / 1000.0;

    if (g_logPerfResults)
    {
        logValues(*g_logFile,
            DIM,
            pointCnt,
            numBins,
            std::string(rd::LoadModifierNameTraits<LOAD_MODIFIER>::name),
            std::string(rd::BlockTileIONameTraits<IO_BACKEND>::name),
            std::string(rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name),
            POINTS_PER_THREAD, 
            BLOCK_THREADS,
            avgMillis, 
            gigaBandwidth);
        *g_logFile << "\n";
    }

    logValues(std::cout,
            DIM,
            pointCnt,
            numBins,
            std::string(rd::LoadModifierNameTraits<LOAD_MODIFIER>::name),
            std::string(rd::BlockTileIONameTraits<IO_BACKEND>::name),
            std::string(rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name),
            POINTS_PER_THREAD, 
            BLOCK_THREADS,
            avgMillis, 
            gigaBandwidth);
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
    int                         BLOCK_SIZE,
    int                         DIM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    typename                    T>
KernelParametersConf testPointsPerThread(
    int                             pointCnt,
    T const *                       d_P,
    int *                           d_hist,
    int                             stride,
    std::vector<size_t>const &      binsCnt,
    size_t                          numBins,
    rd::BoundingBox<T> const &      bboxGold,
    std::vector<int> const &        histGold)
{
    KernelParametersConf bestKernelParams(DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT);
    bestKernelParams.BLOCK_THREADS = BLOCK_SIZE;
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
    processResults(4, runDeviceSpatialHistogram<BLOCK_SIZE, 4, DIM, LOAD_MODIFIER, 
        IO_BACKEND, INPUT_MEM_LAYOUT, T>(
            pointCnt, d_P, d_hist, stride, binsCnt, numBins, bboxGold, histGold));
    #else

    #define runTest(ppt) processResults(ppt, runDeviceSpatialHistogram<BLOCK_SIZE, ppt, DIM, \
        LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, T>( \
            pointCnt, d_P, d_hist, stride, binsCnt, numBins, bboxGold, histGold));
    
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
 *  Specialization for testing different number of threads in block 
 */
template <
    int                         DIM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    typename                    T>
KernelParametersConf testBlockSize(
    int                             pointCnt,
    T const *                       d_P,
    int *                           d_hist,
    int                             stride,
    std::vector<size_t>const &      binsCnt,
    size_t                          numBins,
    rd::BoundingBox<T> const &      bboxGold,
    std::vector<int> const &        histGold)
{
    KernelParametersConf bestKernelParams;

    auto checkBestConf = [&bestKernelParams](KernelParametersConf kp)
    {
        if (kp.gigaBandwidth > bestKernelParams.gigaBandwidth)
        {
            bestKernelParams = kp;
        }
    };

    #ifdef RD_BLOCK_SIZE
    checkBestConf(testPointsPerThread<RD_BLOCK_SIZE, 
        DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, T>(pointCnt, d_P, d_hist,
        stride, binsCnt, numBins, bboxGold, histGold));
    #endif

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
    std::vector<size_t> binsCnt(DIM, 4);
    // if (DIM == 1)
    // {
    //     binsCnt[0] = 16;
    // }
    // else if (DIM >= 2)
    // {
    //     binsCnt[0] = 4;
    //     binsCnt[1] = 4;
    // }
    size_t numBins = std::accumulate(binsCnt.begin(), binsCnt.end(),
                         1, std::multiplies<size_t>());

    T *d_PRowMajor, *d_PColMajor;
    int *d_hist;

    // allocate containers
    checkCudaErrors(cudaMalloc((void**)&d_PRowMajor, pointCnt * DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_PColMajor, pointCnt * DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_hist, numBins * sizeof(int)));

    // initialize data
    rd::gpu::rdMemcpy<rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_PRowMajor, inData.data(), DIM, pointCnt, DIM, DIM);    
    rd::gpu::rdMemcpy<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_PColMajor, inData.data(), DIM, pointCnt, pointCnt, DIM);    
    
    //---------------------------------------------------
    //               REFERENCE HISTOGRAM
    //---------------------------------------------------
    std::cout << "binsCnt: [";
    for (size_t b : binsCnt)
    {
        std::cout << ", " << b;
    }
    std::cout << "], total: " << numBins << std::endl;

    rd::Histogram<T> histGold;
    rd::BoundingBox<T> bboxGold(inData.data(), pointCnt, DIM);
    bboxGold.calcDistances();
    
    histogramGold(pointCnt, inData.data(), binsCnt, bboxGold, histGold);

    #ifdef RD_DEBUG
        bboxGold.print();
        std::cout << "hist: \n";
        for (size_t i = 0; i < histGold.hist.size(); ++i)
        {
            if (histGold.hist[i])
            {
                std::cout << "hist["<<i<<"]: " << histGold.hist[i] << "\n";
            }
        }
        std::cout << std::endl;
    #endif

    std::vector<int> histGoldValues;
    // convert from size_t to int for results comparison
    for (size_t v : histGold.hist)
        histGoldValues.push_back((int)v);

    //---------------------------------------------------
    //               GPU HISTOGRAM
    //---------------------------------------------------

#ifdef QUICK_TEST 
    {
        const int BLOCK_THREADS = 128;
        const int POINTS_PER_THREAD = 4;
        runDeviceSpatialHistogram<BLOCK_THREADS, POINTS_PER_THREAD, DIM, cub::LOAD_LDG, 
            rd::gpu::IO_BACKEND_CUB, rd::COL_MAJOR>(
            pointCnt, d_PColMajor, d_hist, pointCnt, binsCnt, numBins, bboxGold, histGoldValues);
        runDeviceSpatialHistogram<BLOCK_THREADS, POINTS_PER_THREAD, DIM, cub::LOAD_LDG, 
            rd::gpu::IO_BACKEND_TROVE, rd::COL_MAJOR>(
            pointCnt, d_PColMajor, d_hist, pointCnt, binsCnt, numBins, bboxGold, histGoldValues);

        runDeviceSpatialHistogram<BLOCK_THREADS, POINTS_PER_THREAD, DIM, cub::LOAD_LDG, 
            rd::gpu::IO_BACKEND_CUB, rd::ROW_MAJOR>(
            pointCnt, d_PRowMajor, d_hist, 1, binsCnt, numBins, bboxGold, histGoldValues);
        runDeviceSpatialHistogram<BLOCK_THREADS, POINTS_PER_THREAD, DIM, cub::LOAD_LDG, 
            rd::gpu::IO_BACKEND_TROVE, rd::ROW_MAJOR>(
            pointCnt, d_PRowMajor, d_hist, 1, binsCnt, numBins, bboxGold, histGoldValues);
    }
#else    
    
    std::cout << "\n" << rd::HLINE << "\n";

    testBlockSize<DIM, cub::LOAD_LDG, rd::gpu::IO_BACKEND_CUB, rd::COL_MAJOR>(
        pointCnt, d_PColMajor, d_hist, pointCnt, binsCnt, numBins, 
        bboxGold, histGoldValues);
    testBlockSize<DIM, cub::LOAD_LDG, rd::gpu::IO_BACKEND_CUB, rd::ROW_MAJOR>(
        pointCnt, d_PRowMajor, d_hist, 1, binsCnt, numBins, bboxGold, 
        histGoldValues);

    std::cout << "\n" << rd::HLINE << "\n";

    testBlockSize<DIM, cub::LOAD_LDG, rd::gpu::IO_BACKEND_TROVE, rd::COL_MAJOR>(
        pointCnt, d_PColMajor, d_hist, pointCnt, binsCnt, numBins, 
        bboxGold, histGoldValues);
    testBlockSize<DIM, cub::LOAD_LDG, rd::gpu::IO_BACKEND_TROVE, rd::ROW_MAJOR>(
        pointCnt, d_PRowMajor, d_hist, 1, binsCnt, numBins, bboxGold, 
        histGoldValues);

    std::cout << "\n" << rd::HLINE << "\n";
#endif


    //---------------------------------------------------
    // clean-up
    checkCudaErrors(cudaFree(d_PRowMajor));
    checkCudaErrors(cudaFree(d_PColMajor));
    checkCudaErrors(cudaFree(d_hist));
}

#if !defined(QUICK_TEST)
template <typename T>
std::string createFinalGraphDataFile()
{
    //------------------------------------------
    // create data file for drawing graph
    //------------------------------------------

    std::ostringstream graphDataFile;
    graphDataFile << typeid(T).name() << "_" << getCurrDateAndTime() << "_" 
        << g_devName << "_graphData_v4.dat";

    std::string filePath = rd::findPath("../gnuplot_data/", graphDataFile.str());
    std::ofstream gdataFile(filePath.c_str(), std::ios::out | std::ios::trunc);
    if (gdataFile.fail())
    {
        throw std::logic_error("Couldn't open file: " + graphDataFile.str());
    }

    auto printData = [&gdataFile](std::vector<float> const &v, std::string secName)
    {
        gdataFile << "# [" << secName << "] \n";
        for (size_t i = 0; i < v.size()/2; ++i)
        {
            gdataFile << std::right << std::fixed << std::setw(5) << std::setprecision(1) <<
                v[2 * i] << " " << v[2 * i + 1] << "\n";
        }
        // two sequential blank records to reset $0 counter
        gdataFile << "\n\n";
    };

    printData(g_bestPerf[0], "ROW-(CUB)");
    printData(g_bestPerf[1], "COL-(CUB)");
    printData(g_bestPerf[2], "ROW-(trove)");
    printData(g_bestPerf[3], "COL-(trove)");

    gdataFile.close();
    return filePath;
}

template <typename T>
void drawFinalGraph(std::string graphDataFilePath)
{
        //------------------------------------------
        // drawing graph
        //------------------------------------------

        rd::GraphDrawer<float> gDrawer;
        std::ostringstream graphName;
        graphName << typeid(T).name() << "_" <<  getCurrDateAndTime() << "_" 
            << g_devName << "_hist_performance_v4.png";
        std::string filePath = rd::findPath("../img/", graphName.str());

        gDrawer.sendCmd("set output '" + filePath + "'");
        gDrawer.setXLabel("Wymiar danych.");
        gDrawer.setYLabel("GB/s");

        gDrawer.sendCmd("set key right top");
        gDrawer.sendCmd("set style fill solid 0.95 border rgb 'grey30'");

        gDrawer.sendCmd("colStep = " + std::to_string(g_graphColStep));
        gDrawer.sendCmd("bs = 2 * colStep");
        gDrawer.sendCmd("nCol = " + std::to_string(g_graphNCol));
        gDrawer.sendCmd("groupStep = (nCol+1) * bs");
        gDrawer.sendCmd("nGroups = " + std::to_string(g_graphNGroups));
        gDrawer.sendCmd("offset = 9 * colStep");
        gDrawer.sendCmd("xEnd = offset + (nGroups-1) * groupStep + 9 * colStep + 4");

        gDrawer.sendCmd("set xrange [0:xEnd]");
        // gDrawer.sendCmd("set xtics nomirror out ('2D' offset,'3D' offset + groupStep,"
        //      "'4D' offset + 2*groupStep, '5D' offset + 3*groupStep, '6D' offset + 4*groupStep)");

        std::ostringstream xticsCmd;
        xticsCmd << "set xtics nomirror out ('1D' offset";
        for (int i = 1; i < g_graphNGroups; ++i)
        {
            xticsCmd << ", '" << i+1 << "D' offset + " << i << "*groupStep";
        }
        gDrawer.sendCmd(xticsCmd.str());

        gDrawer.sendCmd("dataFile = '" + graphDataFilePath + "'");

        std::ostringstream cmd;
        cmd << "plot dataFile i 0 u (offset + $0 * groupStep - 3 * colStep):2:(bs) t 'ROW (CUB)' w boxes ls 1,";
        cmd << "    ''        i 1 u (offset + $0 * groupStep - 1 * colStep):2:(bs) t 'COL (CUB)' w boxes ls 2,";
        cmd << "    ''        i 2 u (offset + $0 * groupStep + 1 * colStep):2:(bs) t 'ROW (trove)' w boxes ls 3,";
        cmd << "    ''        i 3 u (offset + $0 * groupStep + 3 * colStep):2:(bs) t 'COL (trove)' w boxes ls 4,";
        cmd << "    ''        i 0 u (offset + $0 * groupStep - 3 * colStep):($2 + 0.6):2 notitle w labels rotate by 70 left,";
        cmd << "    ''        i 1 u (offset + $0 * groupStep - 1 * colStep):($2 + 0.6):2 notitle w labels rotate by 70 left,";
        cmd << "    ''        i 2 u (offset + $0 * groupStep + 1 * colStep):($2 + 0.6):2 notitle w labels rotate by 70 left,";
        cmd << "    ''        i 3 u (offset + $0 * groupStep + 3 * colStep):($2 + 0.6):2 notitle w labels rotate by 70 left ";

        gDrawer.sendCmd(cmd.str());
}
#endif  // !defined(QUICK_TEST)


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
        const int dim = 10;

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
