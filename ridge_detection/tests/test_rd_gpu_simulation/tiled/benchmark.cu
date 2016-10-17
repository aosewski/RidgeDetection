/**
 *  @file benchmark.cu
 *  @author Adam Rogowiec
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

// needed for name traits..
// FIXME: remove this!
#define BLOCK_TILE_LOAD_V4 1

#if defined(RD_DEBUG) && !defined(CUB_STDERR)
#define CUB_STDERR 
#endif

#include <helper_cuda.h>

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <limits>

#ifdef RD_USE_OPENMP
#include <omp.h>
#endif

#include "rd/gpu/device/tiled/simulation.cuh" 
#include "rd/gpu/util/dev_static_for.cuh"

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/memory.h"
#include "rd/utils/name_traits.hpp"

#include "tests/test_util.hpp"

#include "cub/test_util.h"
#include "cub/util_arch.cuh"

#include "rd/utils/rd_params.hpp"

//------------------------------------------------------------
//  GLOBAL CONSTANTS / VARIABLES
//------------------------------------------------------------

static const std::string LOG_FILE_NAME_SUFFIX = "gpu_tiled_timings.txt";

static std::ofstream * g_logFile           = nullptr;
static bool            g_logPerfResults    = false;
static bool            g_drawResultsGraph  = false;
static std::string     g_devName           = "";
static int             g_devId             = 0;
#ifdef RD_INNER_KERNEL_TIMING
static int             g_devClockRate      = 0;
#endif

#if defined(RD_PROFILE) || defined(RD_DEBUG) || defined(QUICK_TEST)
static const int g_iterations = 1;
#else
static const int g_iterations = 5;
#endif

#if defined(RD_TEST_SCALING) && defined(RD_TEST_DIMENSION)
static constexpr int MIN_TEST_DIM = RD_TEST_DIMENSION;
#else
static constexpr int MIN_TEST_DIM = 2;
#endif
#ifdef QUICK_TEST
static constexpr int MAX_TEST_DIM = 4;
#elif defined(RD_TEST_SCALING) && defined(RD_TEST_DIMENSION)
static constexpr int MAX_TEST_DIM = RD_TEST_DIMENSION;
#else
static constexpr int MAX_TEST_DIM = 12;
#endif
static constexpr int MAX_POINTS_NUM = int(1e7);
static constexpr int RD_CUDA_MAX_SYNC_DEPTH = 10;
static constexpr size_t HUNDRED_MB_IN_BYTES = 100 * 1024 * 1024;

//------------------------------------------------------------
//  Utils
//------------------------------------------------------------

template <typename T>
struct RdTiledData
{
    T const * inputPoints;
    std::vector<T> chosenPoints;
    int pointsNum;
    int chosenPointsNum;
    T r1, r2;
    T a, b, s;    
    int maxTileCapacity;
    std::vector<int> nTilesPerDim;
};

/**
 * @brief      Create if necessary and open log file. Allocate log file stream.
 */
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
        logValue(*g_logFile, "inPointsNum", 11);
        logValue(*g_logFile, "dim", 6);
        logValue(*g_logFile, "refinement", 10);
        logValue(*g_logFile, "r1", 10);
        logValue(*g_logFile, "r2", 10);
        logValue(*g_logFile, "maxTileCapacity", 15);
        logValue(*g_logFile, "nTilesPerDim", 12);
        logValue(*g_logFile, "inMemLayout", 16);
        logValue(*g_logFile, "outMemLayout", 16);
        logValue(*g_logFile, "rdTileAlg", 15);
        logValue(*g_logFile, "rdTilePolicy", 16);
        logValue(*g_logFile, "tileType", 16);
        logValue(*g_logFile, "chosenPointsNum", 15);
        logValue(*g_logFile, "avgCpuTime", 12);
        logValue(*g_logFile, "minCpuTime", 12);
        logValue(*g_logFile, "maxCpuTime", 12);
        #ifdef RD_INNER_KERNEL_TIMING
        logValue(*g_logFile, "avgGpuTime", 10);
        logValue(*g_logFile, "avgGpuRdTime", 12);
        logValue(*g_logFile, "avgGpuRefTime", 13);
        #endif
        logValue(*g_logFile, "hausdorffDist", 13);
        logValue(*g_logFile, "medianDist", 12);
        logValue(*g_logFile, "avgDist", 12);
        logValue(*g_logFile, "minDist", 12);
        logValue(*g_logFile, "maxDist", 12);
        *g_logFile << "\n";
    }
}


static void configureDevice(
    size_t neededMemSize)
{
    checkCudaErrors(cudaDeviceReset());
    checkCudaErrors(cudaSetDevice(g_devId));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, RD_CUDA_MAX_SYNC_DEPTH));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, neededMemSize));
}

//------------------------------------------------------------
//  INVOKE AND MEASURE 
//------------------------------------------------------------

template <
    int                     DIM,
    rd::DataMemoryLayout    IN_MEMORY_LAYOUT,
    rd::DataMemoryLayout    OUT_MEMORY_LAYOUT,
    rd::gpu::tiled::RidgeDetectionAlgorithm         RD_TILE_ALGORITHM,
    rd::gpu::tiled::TiledRidgeDetectionPolicy       RD_TILE_POLICY,
    rd::gpu::tiled::TileType                        RD_TILE_TYPE,
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void benchmark(
    RdTiledData<T> &                dataPack,
    rd::RDAssessmentQuality<T> *    qualityMeasure,
    bool                            endPhaseRefinement)
{
    using namespace rd::gpu;

    bool debugSynchronous = false;
    bool enableTiming = false;
    #ifdef RD_INNER_KERNEL_TIMING
    enableTiming = true;
    #endif
    #ifdef RD_DEBUG
    debugSynchronous = true;
    #endif

    tiled::RidgeDetection<DIM, IN_MEMORY_LAYOUT, OUT_MEMORY_LAYOUT, RD_TILE_ALGORITHM,
        RD_TILE_POLICY, RD_TILE_TYPE, T> rdGpu(
            dataPack.pointsNum, dataPack.inputPoints, enableTiming, debugSynchronous);

    std::cout << rd::HLINE << "\n";
    std::cout << "in mem layout: {" << rd::DataMemoryLayoutNameTraits<IN_MEMORY_LAYOUT>::name
            << "} out mem layout: {" << rd::DataMemoryLayoutNameTraits<OUT_MEMORY_LAYOUT>::name
            << "} rd tile algorithm: {" << 
                tiled::RidgeDetectionAlgorithmNameTraits<RD_TILE_ALGORITHM>::name
            << "} rd tile policy: {" << 
                tiled::TiledRidgeDetectionPolicyNameTraits<RD_TILE_POLICY>::name
            << "} tile type: {" << tiled::TileTypeNameTraits<RD_TILE_TYPE>::name
            << "} maxTileCapacity: {" << dataPack.maxTileCapacity 
            << "} nTilesPerDim: " << rdToString(dataPack.nTilesPerDim) 
            << " r1: {" << dataPack.r1 
            << "} r2: {" << dataPack.r2 
            << "} pointCnt: {" << dataPack.pointsNum 
            << "} endPhaseRefinement: {" << std::boolalpha << endPhaseRefinement << "}\n";

    rd::CpuTimer timer;
    T medianDist = 0, avgDist = 0, minDist = 0, maxDist = 0, hausdorffDist = 0;
    #ifdef RD_INNER_KERNEL_TIMING
    T avgGpuWholeTime = 0, avgGpuRdTime = 0, avgGpuRefinementTime = 0;
    rd::gpu::tiled::detail::TiledRidgeDetectionTimers gpuTimers;
    #endif

    float minCpuTime = std::numeric_limits<float>::max();
    float maxCpuTime = std::numeric_limits<float>::lowest();

    std::vector<T> chosenPoints;
    int dimTiles[DIM];

    for (int d = 0; d < DIM; ++d)
    {
        dimTiles[d] = dataPack.nTilesPerDim[d];
    }

    rd::CpuTimer qmesTimer, postprcsTimer;

    float testAvgCpuTime = 0.f;
    for (int k = 0; k < g_iterations; ++k)
    {
        chosenPoints.clear();
        timer.start();
        rdGpu(dataPack.r1, dataPack.r2, dataPack.maxTileCapacity, dimTiles, 
            endPhaseRefinement);
        timer.stop();

        #ifdef RD_INNER_KERNEL_TIMING
        gpuTimers = rdGpu.getTimers();
        avgGpuWholeTime += gpuTimers.wholeTime / g_devClockRate;
        avgGpuRdTime += gpuTimers.rdTilesTime / g_devClockRate;
        avgGpuRefinementTime += gpuTimers.refinementTime / g_devClockRate;
        #endif

        float currTime = timer.elapsedMillis(0);
        testAvgCpuTime += currTime;
        minCpuTime = min(currTime, minCpuTime);
        maxCpuTime = max(currTime, maxCpuTime);

        postprcsTimer.start();
        dataPack.chosenPointsNum = rdGpu.getChosenPointsNum();
        // std::cout << "Aggregate chosenPointsNum: " << dataPack.chosenPointsNum << std::endl;
        
        chosenPoints.resize(dataPack.chosenPointsNum * DIM);
        rdGpu.getChosenPoints(chosenPoints.data());

        qmesTimer.start();
        // measure assessment quality
        hausdorffDist += qualityMeasure->hausdorffDistance(chosenPoints);
        T median, avg, min, max;
        
        qualityMeasure->setDistanceStats(chosenPoints, median, avg, min, max);
        avgDist += avg;
        medianDist += median;
        minDist += min;
        maxDist += max;

        qmesTimer.stop();
        postprcsTimer.stop();
        std::cout << "postprocess (quality measure): " << std::fixed << std::right 
                << std::setw(12) << std::setprecision(3) << qmesTimer.elapsedMillis(0) << "ms" 
            << "\tpostprocess (all): " << std::fixed << std::right << std::setw(12) 
                << std::setprecision(3) << postprcsTimer.elapsedMillis(0) << "ms" 
            << "\tcomputation cpu time: " << std::fixed << std::right << std::setw(12) 
                << std::setprecision(3) << timer.elapsedMillis(0) << "ms" << std::endl;
    }

    if (DIM <= 3 && g_drawResultsGraph) 
    {
        rd::GraphDrawer<T> gDrawer;
        std::ostringstream graphName;

        graphName << typeid(T).name() << "_" << getCurrDateAndTime() << "_"
            << g_devName 
            << "_" << rd::DataMemoryLayoutNameTraits<IN_MEMORY_LAYOUT>::shortName
            << "_" << rd::DataMemoryLayoutNameTraits<OUT_MEMORY_LAYOUT>::shortName
            << "_" << tiled::RidgeDetectionAlgorithmNameTraits<RD_TILE_ALGORITHM>::shortName
            << "_" << tiled::TiledRidgeDetectionPolicyNameTraits<RD_TILE_POLICY>::shortName
            << "_" << tiled::TileTypeNameTraits<RD_TILE_TYPE>::shortName
            << "_tcap=" << dataPack.maxTileCapacity 
            << "_" << rdToString(dataPack.nTilesPerDim) 
            << "_ref=" << (endPhaseRefinement ? "T" : "F")
            << "_np=" << dataPack.pointsNum    
            << "_ns=" << dataPack.chosenPointsNum    
            << "_r1=" << dataPack.r1 
            << "_r2=" << dataPack.r2 
            << "_a=" << dataPack.a 
            << "_b=" << dataPack.b 
            << "_s=" << dataPack.s;

        std::string filePath = rd::findPath("../img/", graphName.str());

        gDrawer.startGraph(filePath, DIM);
        if (DIM == 3)
        {
            gDrawer.setGraph3DConf();
        }

        gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#B8E186' ps 0.5 ",
             dataPack.inputPoints, rd::GraphDrawer<T>::POINTS, dataPack.pointsNum);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#D73027' ps 1.3 ",
             chosenPoints.data(), rd::GraphDrawer<T>::POINTS, dataPack.chosenPointsNum);
        gDrawer.endGraph();
    }

    testAvgCpuTime /= g_iterations;
    hausdorffDist /= g_iterations;
    medianDist /= g_iterations;
    avgDist /= g_iterations;
    minDist /= g_iterations;
    maxDist /= g_iterations;

    #ifdef RD_INNER_KERNEL_TIMING
    avgGpuWholeTime /= g_iterations;
    avgGpuRdTime /= g_iterations;
    avgGpuRefinementTime /= g_iterations;
    #endif

    if (g_logPerfResults)
    {
        logValues(*g_logFile, dataPack.pointsNum, DIM);
        logValue<bool>(*g_logFile, endPhaseRefinement);
        logValues(*g_logFile,
            dataPack.r1, 
            dataPack.r2,
            dataPack.maxTileCapacity,
            rdToString(dataPack.nTilesPerDim),
            std::string(rd::DataMemoryLayoutNameTraits<IN_MEMORY_LAYOUT>::shortName),
            std::string(rd::DataMemoryLayoutNameTraits<OUT_MEMORY_LAYOUT>::shortName),
            std::string(tiled::RidgeDetectionAlgorithmNameTraits<RD_TILE_ALGORITHM>::shortName),
            std::string(tiled::TiledRidgeDetectionPolicyNameTraits<RD_TILE_POLICY>::shortName),
            std::string(tiled::TileTypeNameTraits<RD_TILE_TYPE>::shortName),
            dataPack.chosenPointsNum,
            testAvgCpuTime,
            minCpuTime,
            maxCpuTime,
            #ifdef RD_INNER_KERNEL_TIMING
            avgGpuWholeTime,
            avgGpuRdTime,
            avgGpuRefinementTime,
            #endif
            hausdorffDist,
            medianDist,
            avgDist,
            minDist,
            maxDist);
        
        *g_logFile << "\n";
        g_logFile->flush();
    }

    std::cout << "avg time: \t\t" << testAvgCpuTime << "ms\n"
              << "minCpuTime \t\t" << minCpuTime << "\n"
              << "maxCpuTime \t\t" << maxCpuTime << "\n"
              #ifdef RD_INNER_KERNEL_TIMING
              << "avgGpuWholeTime \t" << avgGpuWholeTime << "\n"
              << "avgGpuRdTime \t\t" << avgGpuRdTime << "\n"
              << "avgGpuRefinementTime \t" << avgGpuRefinementTime << "\n"
              #endif
              << "hausdorffDist \t\t" << hausdorffDist << "\n"
              << "medianDist \t\t" << medianDist << "\n"
              << "avgDist \t\t" << avgDist << "\n"
              << "minDist \t\t" << minDist << "\n"
              << "maxDist \t\t" << maxDist << "\n";

}

//------------------------------------------------------------
//  Test generation
//------------------------------------------------------------

template <
    int         DIM,
    rd::DataMemoryLayout    IN_MEMORY_LAYOUT,
    rd::DataMemoryLayout    OUT_MEMORY_LAYOUT,
    typename    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testAlgorithm(
    RdTiledData<T> &                dataPack,
    rd::RDAssessmentQuality<T> *    qualityMeasure,
    bool                            endPhaseRefinement)
{
    using namespace rd::gpu::tiled;

    #ifndef RD_TEST_SCALING
    benchmark<DIM, IN_MEMORY_LAYOUT, OUT_MEMORY_LAYOUT, RD_BRUTE_FORCE, RD_LOCAL, 
            RD_EXTENDED_TILE>(dataPack, qualityMeasure, endPhaseRefinement);
    std::cout << rd::HLINE << "\n";
    #endif

    benchmark<DIM, IN_MEMORY_LAYOUT, OUT_MEMORY_LAYOUT, RD_BRUTE_FORCE, RD_MIXED, 
            RD_EXTENDED_TILE>(dataPack, qualityMeasure, endPhaseRefinement);
    std::cout << rd::HLINE << "\n";
}

template <
    int         DIM,
    typename    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testMemLayout(
    RdTiledData<T> &                dataPack,
    rd::RDAssessmentQuality<T> *    qualityMeasure,
    bool                            endPhaseRefinement)
{
    #ifndef RD_TEST_SCALING
    testAlgorithm<DIM, rd::ROW_MAJOR, rd::ROW_MAJOR>(dataPack, qualityMeasure, endPhaseRefinement);
    testAlgorithm<DIM, rd::COL_MAJOR, rd::ROW_MAJOR>(dataPack, qualityMeasure, endPhaseRefinement);
    #endif
    testAlgorithm<DIM, rd::COL_MAJOR, rd::COL_MAJOR>(dataPack, qualityMeasure, endPhaseRefinement);
}

template <
    int        DIM,
    typename   T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testNTiles(
    RdTiledData<T> &                dataPack,
    rd::RDAssessmentQuality<T> *    qualityMeasure,
    bool                            endPhaseRefinement)
{
    #ifndef RD_TEST_SCALING
    #if defined(QUICK_TEST)
        int k = 3;
    #else
    for (int k = 3; k < 6; k++)
    #endif
    {
        dataPack.nTilesPerDim = std::vector<int>(DIM, k);
        testMemLayout<DIM>(dataPack, qualityMeasure, endPhaseRefinement);
    }
    #else
    dataPack.nTilesPerDim = std::vector<int>(DIM, 1);
    dataPack.nTilesPerDim[0] = 4;
    testMemLayout<DIM>(dataPack, qualityMeasure, endPhaseRefinement);
    #endif
}

template <
    int                     DIM,
    typename                T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testMaxTileCapacity(
    RdTiledData<T> &                dataPack,
    rd::RDAssessmentQuality<T> *    qualityMeasure,
    bool                            endPhaseRefinement)
{
    #if defined(QUICK_TEST) || defined(RD_TEST_SCALING)
    int k = 0.2 * dataPack.pointsNum;
    #else
    for (int k = 0.05 * dataPack.pointsNum; 
             k <= 0.25 * dataPack.pointsNum; 
             k += 0.05 * dataPack.pointsNum)
    #endif
    {
        dataPack.maxTileCapacity = k;
        testNTiles<DIM>(dataPack, qualityMeasure, endPhaseRefinement);
    }
}

/**
 * @brief Test detection time & quality relative to algorithm parameter values
 */
template <
    int         DIM,
    typename    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testRDParams(
    int                             pointCnt,
    PointCloud<T> const &           pc,
    std::vector<T> &&               points,
    rd::RDAssessmentQuality<T> *    qualityMeasure,
    bool                            endPhaseRefinement)
{
    #if defined(RD_TEST_SCALING)
    T r = 2.f * pc.stddev_;
    #else
    std::vector<T> r1Vals{0.1f, 0.2f, 0.5f, 1.0f, 1.2f, 1.5f, 1.8f, 2.0f, 3.f, 4.f, 5.f, 10.f};
    for (T& val : r1Vals)
    {
        val *= pc.stddev_;
    }

    for (T r : r1Vals)    
    #endif
    {
        RdTiledData<T> dataPack;
        dataPack.pointsNum = pointCnt;
        dataPack.r1 = r;
        dataPack.r2 = r*2.f;
        dataPack.s = pc.stddev_;
        dataPack.inputPoints = points.data();
        pc.getCloudParameters(dataPack.a, dataPack.b);
        testMaxTileCapacity<DIM>(dataPack, qualityMeasure, endPhaseRefinement);
    }
}

/**
 * @brief Test detection time & quality relative to end phase refinement 
 */
template <
    int         DIM,
    typename    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testEndPhaseRefinement(
    int                             pointCnt,
    PointCloud<T> const &           pc,
    std::vector<T> &&               points,
    rd::RDAssessmentQuality<T> *    qualityMeasure)
{
    testRDParams<DIM>(pointCnt, pc, std::forward<std::vector<T>>(points), qualityMeasure, false);
    // #ifndef QUICK_TEST
    // testRDParams<DIM>(pointCnt, pc, std::forward<std::vector<T>>(points), qualityMeasure, true);
    // #endif
    delete qualityMeasure;
}

/**
 * @brief helper structure for static for loop over dimension
 */
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

        testEndPhaseRefinement<D::value>(pointCnt, pc, pc.extractPart(pointCnt, idx), 
            pc.getQualityMeasurer(idx));
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
        if (pc.dim_ < DIM)
        {
            throw std::runtime_error("Input file data dimensionality"
                " is lower than requested!");
        }

        size_t neededMemSize = 8 * pointCnt * DIM * sizeof(T);
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
                << " pointsNum: " << pointCnt << "\n";
        }

        testEndPhaseRefinement<DIM>(pointCnt, pc, pc.extractPart(pointCnt, DIM), 
            pc.getQualityMeasurer(DIM));
        
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
        size_t neededMemSize = 0;
        if (pc.dim_ < MAX_TEST_DIM)
        {
            throw std::runtime_error("Input file data dimensionality"
                " is lower than requested!");
        }
        neededMemSize = 8 * pointCnt * MAX_TEST_DIM * sizeof(T);

        neededMemSize = std::max(HUNDRED_MB_IN_BYTES, neededMemSize);

        std::cout << "Reserve " << float(neededMemSize) / 1024.f / 1024.f 
            << " Mb for malloc heap size" << std::endl;
        configureDevice(neededMemSize);

        StaticFor<MIN_TEST_DIM, MAX_TEST_DIM, IterateDimensions>::impl(pointCnt, pc);

        // clean-up
        if (g_logPerfResults)
        {
            g_logFile->close();
            delete g_logFile;
        }
    }
};

/**
 * @brief Test detection time & quality relative to number of points
 */

template <
    typename    T,
    int         DIM = 0,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testSize(
    PointCloud<T> & pc,
    int pointCnt = -1,
    bool readFromFile = false)
{
    if (pointCnt > 0)
    {
        if (!readFromFile)
        {
            pc.pointCnt_ = pointCnt;
            pc.dim_ = (DIM == 0) ? MAX_TEST_DIM : DIM;
            pc.initializeData();
        }
        TestDimensions<DIM, T>::impl(pc, pointCnt);
    }
    else
    {
        if (!readFromFile) 
        {
            pc.pointCnt_ = MAX_POINTS_NUM;
            pc.dim_ = (DIM == 0) ? MAX_TEST_DIM : DIM;
            pc.initializeData();
        }
        for (int k = 1e3; k <= MAX_POINTS_NUM; k *= 10)
        {
            std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t pointCnt: " << k  
                    << "\n//------------------------------------------\n";
            TestDimensions<DIM, T>::impl(pc, k);
            if (k == MAX_POINTS_NUM) break;
            std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t pointCnt: " << k*2  
                    << "\n//------------------------------------------\n";
            TestDimensions<DIM, T>::impl(pc, k*2);
            std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t pointCnt: " << k*5 
                    << "\n//------------------------------------------\n";
            TestDimensions<DIM, T>::impl(pc, k*5);
        }
    }
}

//------------------------------------------------------------
//  MAIN
//------------------------------------------------------------

int main(int argc, char const **argv)
{

    float a = -1.f, b = -1.f, stddev = -1.f, segLength = -1.f;
    int pointCnt = -1;
    std::string inFilePath = "";
    int inFileDataDim = 0;
    bool loadDataFromFile = false;

    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help")) 
    {
        printf("%s \n"
            "\t\t[--log <log performance results>]\n"
            "\t\t[--rGraphs <draws resuls graphs (if dim <= 3)>]\n"
            "\t\t[--a <a parameter of spiral or length if dim > 3>]\n"
            "\t\t[--b <b parameter of spiral or ignored if dim > 3>]\n"
            "\t\t[--segl <generated N-dimensional segment length>]\n"
            "\t\t--stddev <standard deviation of generated samples>\n"
            "\t\t--size <number of points>\n"
            "\t\t[--d=<device id>]\n"
            "\t\t[--f=<relative to binary, input data file path>]\n"
            "\t\t[--fd=<data dimensonality>]\n"
            "\n", argv[0]);
        exit(0);
    }

    if (args.CheckCmdLineFlag("log"))
    {
        g_logPerfResults = true;
    }
    if (args.CheckCmdLineFlag("a"))
    {
        args.GetCmdLineArgument("a", a);
    }
    if (args.CheckCmdLineFlag("b"))
    {
        args.GetCmdLineArgument("b", b);
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
    if (args.CheckCmdLineFlag("f")) 
    {
        args.GetCmdLineArgument("f", inFilePath);
        loadDataFromFile = true;
    }
    if (args.CheckCmdLineFlag("fd")) 
    {
        args.GetCmdLineArgument("fd", inFileDataDim);
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
    #ifdef RD_INNER_KERNEL_TIMING
    g_devClockRate = devProp.clockRate;
    #endif

    if (pointCnt    < 0 ||
        segLength   < 0 ||
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

        const int dim = 2;

        PointCloud<float> && fpc = SpiralPointCloud<float>(inFilePath, a, b, int(1e7), 3, stddev);

        // PointCloud<float> && fpc = SpiralPointCloud<float>(a, b, pointCnt, dim, stddev);
        // TestDimensions<dim, float>::impl(fpc, pointCnt);
        // fpc.pointCnt_ = pointCnt;
        // fpc.dim_ = dim;
        // fpc.initializeData();

        initializeLogFile<float>();
        size_t neededMemSize = 5 * pointCnt * dim * sizeof(float);
        neededMemSize = std::max(HUNDRED_MB_IN_BYTES, neededMemSize);

        std::cout << "Reserve " << float(neededMemSize) / 1024.f / 1024.f 
            << " Mb for malloc heap size" << std::endl;
        configureDevice(neededMemSize);

        float r1 = 1.7f * fpc.stddev_;
        float r2 = 2.0f * r1;

        RdTiledData<float> dataPack;
        dataPack.pointsNum = pointCnt;
        dataPack.r1 = r1;
        dataPack.r2 = r2;
        dataPack.s = fpc.stddev_;
        auto vpoints = fpc.extractPart(pointCnt, dim);
        dataPack.inputPoints = vpoints.data();
        fpc.getCloudParameters(dataPack.a, dataPack.b);
        dataPack.maxTileCapacity = 0.2 * pointCnt;
        dataPack.nTilesPerDim = std::vector<int>(dim, 3);
        // dataPack.nTilesPerDim[0] = 4;

        // benchmark<dim, rd::ROW_MAJOR, rd::ROW_MAJOR, rd::gpu::tiled::RD_BRUTE_FORCE, 
        //     rd::gpu::tiled::RD_LOCAL, rd::gpu::tiled::RD_EXTENDED_TILE>(
        //     dataPack, fpc.getQualityMeasurer(dim), false);
        // benchmark<dim, rd::COL_MAJOR, rd::ROW_MAJOR, rd::gpu::tiled::RD_BRUTE_FORCE, 
        //     rd::gpu::tiled::RD_LOCAL, rd::gpu::tiled::RD_EXTENDED_TILE>(
        //     dataPack, fpc.getQualityMeasurer(dim), false);
        // benchmark<dim, rd::COL_MAJOR, rd::COL_MAJOR, rd::gpu::tiled::RD_BRUTE_FORCE, 
        //     rd::gpu::tiled::RD_LOCAL, rd::gpu::tiled::RD_EXTENDED_TILE>(
        //     dataPack, fpc.getQualityMeasurer(dim), false);

        benchmark<dim, rd::ROW_MAJOR, rd::ROW_MAJOR, rd::gpu::tiled::RD_BRUTE_FORCE, 
            rd::gpu::tiled::RD_MIXED, rd::gpu::tiled::RD_EXTENDED_TILE>(
            dataPack, fpc.getQualityMeasurer(dim), false);
        benchmark<dim, rd::COL_MAJOR, rd::ROW_MAJOR, rd::gpu::tiled::RD_BRUTE_FORCE, 
            rd::gpu::tiled::RD_MIXED, rd::gpu::tiled::RD_EXTENDED_TILE>(
            dataPack, fpc.getQualityMeasurer(dim), false);
        benchmark<dim, rd::COL_MAJOR, rd::COL_MAJOR, rd::gpu::tiled::RD_BRUTE_FORCE, 
            rd::gpu::tiled::RD_MIXED, rd::gpu::tiled::RD_EXTENDED_TILE>(
            dataPack, fpc.getQualityMeasurer(dim), false);

    #else
        // --size=1000000 --segl=0 --a=22.52 --b=11.31 --stddev=4.17
        #ifndef RD_DOUBLE_PRECISION
        #ifndef RD_TEST_SCALING
        
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (spiral) float: "  
                    << "\n//------------------------------------------\n";
        PointCloud<float> && fpc3d = loadDataFromFile ?
            SpiralPointCloud<float>(inFilePath, a, b, pointCnt, 3, stddev) :
            SpiralPointCloud<float>(a, b, pointCnt, 3, stddev);
        TestDimensions<2, float>::impl(fpc3d, pointCnt);
        TestDimensions<3, float>::impl(fpc3d, pointCnt);
        #else
        // --size=0 --segl=100 --a=0 --b=0 --stddev=2.17 --rGraphs --log --d=0
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) float: "  
                    << "\n//------------------------------------------\n";
        PointCloud<float> && fpc2 = loadDataFromFile ?
            SegmentPointCloud<float>(inFilePath, segLength, pointCnt, inFileDataDim, stddev) :
            SegmentPointCloud<float>(segLength, 0, 0, stddev);
        testSize<float, RD_TEST_DIMENSION>(fpc2, 0, loadDataFromFile);
        #endif  // RD_TEST_SCALING
        #else   // RD_DOUBLE_PRECISION
        #ifndef RD_TEST_SCALING
        
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (spiral) double: "  
                    << "\n//------------------------------------------\n";
        PointCloud<double> && dpc3d = loadDataFromFile ?
            SpiralPointCloud<double>(inFilePath, a, b, pointCnt, 3, stddev) :
            SpiralPointCloud<double>(a, b, pointCnt, 3, stddev);
        TestDimensions<2, double>::impl(dpc3d, pointCnt);
        TestDimensions<3, double>::impl(dpc3d, pointCnt);

        #else 
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) double: "  
                    << "\n//------------------------------------------\n";
        PointCloud<double> && dpc2 = loadDataFromFile ?
            SegmentPointCloud<double>(inFilePath, segLength, pointCnt, inFileDataDim, stddev) :
            SegmentPointCloud<double>(segLength, 0, 0, stddev);
        testSize<double, RD_TEST_DIMENSION>(dpc2);
        
        #endif  // RD_TEST_SCALING
        #endif  // RD_DOUBLE_PRECISION
    #endif

    checkCudaErrors(cudaDeviceReset());

    std::cout << rd::HLINE << std::endl;
    std::cout << "END!" << std::endl;

    return EXIT_SUCCESS;
}
