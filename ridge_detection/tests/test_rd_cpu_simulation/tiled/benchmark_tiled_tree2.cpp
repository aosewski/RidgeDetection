/**
 * @file benchmark_tiled_tree.cpp
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

#include "rd/utils/rd_params.hpp"

#include "rd/utils/utilities.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/assessment_quality.hpp"

#include "rd/cpu/tiled/rd_tiled.hpp"
#include "rd/cpu/samples_generator.hpp"

#include "tests/test_util.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <tuple>
#include <cmath>
#include <type_traits>
#include <limits>

#ifdef RD_USE_OPENMP
#include <omp.h>
#endif

//------------------------------------------------------------
//  GLOBAL CONSTANTS / VARIABLES
//------------------------------------------------------------

static const std::string LOG_FILE_NAME_SUFFIX = "cpu_tiled_timings.txt";

std::ofstream * g_logFile           = nullptr;
bool            g_logPerfResults    = false;
bool            g_drawResultsGraph  = false;
// (whole time, build tree time, ridge detection time, end phase refinement time)
typedef std::tuple<float, float, float, float> PerfResultsT;

#if defined(RD_PROFILE) || defined(RD_DEBUG) /*|| defined(QUICK_TEST)*/
static const int g_iterations = 1;
#else
static const int g_iterations = 5;
#endif

static int g_threads = 1;
static constexpr int MAX_TEST_DIM       = 12;
static constexpr int MIN_TEST_DIM       = 2;
static constexpr int MAX_POINTS_NUM     = int(1e7);

template <typename T> using Params = rd::RDTiledParams<T>;

//------------------------------------------------------------
//  Utils
//------------------------------------------------------------

/**
 * @brief      Create if necessary and open log file. Allocate log file stream.
 */
template <typename T>
static void initializeLogFile()
{
    if (g_logPerfResults)
    {
        std::ostringstream logFileName;
        logFileName << getCurrDate() << "_" << LOG_FILE_NAME_SUFFIX;

        std::string logFilePath = rd::findPath("timings/", logFileName.str());
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
        logValue(*g_logFile, "dim", 10);
        logValue(*g_logFile, "refinement", 10);
        logValue(*g_logFile, "r1", 10);
        logValue(*g_logFile, "r2", 10);
        logValue(*g_logFile, "maxTileCapacity", 15);
        logValue(*g_logFile, "nTilesPerDim", 12);
        logValue(*g_logFile, "algorithm", 16);
        logValue(*g_logFile, "threads", 10);
        logValue(*g_logFile, "chosenPointsNum", 15);
        logValue(*g_logFile, "avgCpuTime", 10);
        logValue(*g_logFile, "minCpuTime", 10);
        logValue(*g_logFile, "maxCpuTime", 10);
        logValue(*g_logFile, "avgBuildTreeTime", 16);
        logValue(*g_logFile, "avgRDTime", 10);
        logValue(*g_logFile, "avgRefTime", 10);
        logValue(*g_logFile, "hausdorffDist", 13);
        logValue(*g_logFile, "medianDist", 10);
        logValue(*g_logFile, "avgDist", 10);
        logValue(*g_logFile, "minDist", 10);
        logValue(*g_logFile, "maxDist", 10);
        *g_logFile << "\n";
    }
}

//------------------------------------------------------------
//  INVOKE AND MEASURE 
//------------------------------------------------------------

template <
    rd::tiled::TiledRDAlgorithm     ALGORITHM,
    typename                        T>
PerfResultsT benchmark(
    rd::tiled::rdTiledData<T> &     dataPack,
    int                             threads,
    bool                            endPhaseRefinement,
    rd::RDAssessmentQuality<T> *    qualityMeasure,
    bool                            verbose = false,
    bool                            drawTiles = false)
{
    std::cout << rd::HLINE << "\n";
    std::cout << rd::tiled::TiledRDAlgorithmNameTraits<ALGORITHM>::name
              << " threads: " << threads << "\n";
    std::cout << "% maxTileCapacity: " << dataPack.maxTileCapacity << " "
            << "nTilesPerDim: " << rdToString(dataPack.nTilesPerDim) << " "
            << "r1: " << dataPack.r1 << " "
            << "r2: " << dataPack.r2 << " "
            << "pointCnt: " << dataPack.np << " "
            << "endPhaseRefinement: " << std::boolalpha << endPhaseRefinement << "\n";


    // verbose = true;
    // drawTiles = true;
    T medianDist = 0, avgDist = 0, minDist = 0, maxDist = 0, hausdorffDist = 0;
    float minTime = std::numeric_limits<float>::max();
    float maxTime = std::numeric_limits<float>::lowest();

    PerfResultsT testAvgTime = std::make_tuple(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < g_iterations; ++k)
    {
        PerfResultsT perf = rd::tiled::tiledRidgeDetection<ALGORITHM>(
            dataPack,
            threads,
            endPhaseRefinement,
            verbose, drawTiles);

        printf("curr iter time: %12.3fms\n", std::get<0>(perf));

        // measure assessment quality
        hausdorffDist += qualityMeasure->hausdorffDistance(dataPack.S);
        T median, avg, min, max;
        
        qualityMeasure->setDistanceStats(dataPack.S, median, avg, min, max);
        avgDist += avg;
        medianDist += median;
        minDist += min;
        maxDist += max;

        dataPack.S.clear();
        // whole time
        std::get<0>(testAvgTime) += std::get<0>(perf);
        // build tree time
        std::get<1>(testAvgTime) += std::get<1>(perf);
        // ridge detection time
        std::get<2>(testAvgTime) += std::get<2>(perf);
        // end phase refinement time
        std::get<3>(testAvgTime) += std::get<3>(perf);

        minTime = std::min(std::get<0>(perf), minTime);
        maxTime = std::max(std::get<0>(perf), maxTime);
    }

    if (dataPack.dim <= 3 && g_drawResultsGraph) 
    {
        rd::GraphDrawer<T> gDrawer;
        std::ostringstream graphName;

        graphName << typeid(T).name() << "_" << getCurrDateAndTime() << "_"
            << rd::tiled::TiledRDAlgorithmNameTraits<ALGORITHM>::shortName
            << "_t-" << threads
            << "_mtc-" << dataPack.maxTileCapacity 
            << "_ntpd-" << rdToString(dataPack.nTilesPerDim) 
            << "_endRef-" << std::boolalpha << endPhaseRefinement 
            << "_np-" << dataPack.np    
            << "_r1-" << dataPack.r1 
            << "_r2-" << dataPack.r2 
            << "_a-" << dataPack.a 
            << "_b-" << dataPack.b 
            << "_s-" << dataPack.s
            << "_res";

        std::string filePath = rd::findPath("img/", graphName.str());
        gDrawer.startGraph(filePath, dataPack.dim);
        if (dataPack.dim == 3)
        {
            gDrawer.setGraph3DConf();
        }

        gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#B8E186' ps 0.5 ",
             dataPack.P, rd::GraphDrawer<T>::POINTS, dataPack.np);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#D73027' ps 1.3 ",
             dataPack.S.data(), rd::GraphDrawer<T>::POINTS, dataPack.ns);

        gDrawer.endGraph();
    }

    std::get<0>(testAvgTime) /= g_iterations;
    std::get<1>(testAvgTime) /= g_iterations;
    std::get<2>(testAvgTime) /= g_iterations;
    std::get<3>(testAvgTime) /= g_iterations;
    hausdorffDist /= g_iterations;
    medianDist /= g_iterations;
    avgDist /= g_iterations;
    minDist /= g_iterations;
    maxDist /= g_iterations;

    if (g_logPerfResults)
    {
        logValues(*g_logFile, dataPack.np, dataPack.dim);
        logValue<bool>(*g_logFile, endPhaseRefinement);
        logValues(*g_logFile,
            dataPack.r1,
            dataPack.r2,
            dataPack.maxTileCapacity,
            rdToString(dataPack.nTilesPerDim),
            std::string(rd::tiled::TiledRDAlgorithmNameTraits<ALGORITHM>::shortName),
            threads,
            dataPack.ns,
            std::get<0>(testAvgTime),
            minTime,
            maxTime,
            std::get<1>(testAvgTime),
            std::get<2>(testAvgTime),
            std::get<3>(testAvgTime),
            hausdorffDist,
            medianDist,
            avgDist,
            minDist,
            maxDist);

        *g_logFile << "\n";
        g_logFile->flush();
    }

    std::cout << "avg whole time: \t\t" << std::get<0>(testAvgTime) << "ms\n"
              << "min whole time: \t\t" << minTime << "ms\n"
              << "max whole time: \t\t" << maxTime << "ms\n"
              << "avg build tree time: \t" << std::get<1>(testAvgTime) << "ms\n"
              << "avg ridge detection time: \t" << std::get<2>(testAvgTime) << "ms\n"
              << "avg refinement time: \t" << std::get<3>(testAvgTime) << "ms\n"
              << "hausdorffDist \t\t" << hausdorffDist << "\n"
              << "medianDist \t\t" << medianDist << "\n"
              << "avgDist \t\t" << avgDist << "\n"
              << "minDist \t\t" << minDist << "\n"
              << "maxDist \t\t" << maxDist << "\n";


    return testAvgTime;
}

//------------------------------------------------------------
//  Test generation
//------------------------------------------------------------

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testAlgorithm(
    rd::tiled::rdTiledData<T> & dataPack,
    rd::RDAssessmentQuality<T> * qualityMeasure,
    bool endPhaseRefinement)
{

    #if !defined(RD_TEST_SCALING) && !defined(QUICK_TEST)
    benchmark<rd::tiled::TILED_LOCAL_TREE_LOCAL_RD>(dataPack, g_threads, endPhaseRefinement, 
        qualityMeasure);
    std::cout << rd::HLINE << "\n";

    benchmark<rd::tiled::TILED_LOCAL_TREE_MIXED_RD>(dataPack, g_threads, endPhaseRefinement, 
        qualityMeasure);
    std::cout << rd::HLINE << "\n";

    benchmark<rd::tiled::TILED_GROUPED_TREE_LOCAL_RD>(dataPack, g_threads, endPhaseRefinement, 
        qualityMeasure);
    std::cout << rd::HLINE << "\n";

    benchmark<rd::tiled::TILED_GROUPED_TREE_MIXED_RD>(dataPack, g_threads, endPhaseRefinement, 
        qualityMeasure);
    std::cout << rd::HLINE << "\n";
    #endif
    benchmark<rd::tiled::EXT_TILE_TREE_MIXED_RD>(dataPack, g_threads, endPhaseRefinement, 
        qualityMeasure);
    std::cout << rd::HLINE << "\n";

}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testNTilesPerDim(
    rd::tiled::rdTiledData<T> & dataPack,
    rd::RDAssessmentQuality<T> * qualityMeasure,
    bool endPhaseRefinement)
{
    #ifdef RD_TEST_SCALING
        dataPack.nTilesPerDim = std::vector<size_t>(dataPack.dim, 1);
        dataPack.nTilesPerDim[0] = 4;
        testAlgorithm(dataPack, qualityMeasure, endPhaseRefinement);
    #else
    #ifdef QUICK_TEST
        size_t k = 3;
    #else
    for (size_t k = 3; k < 6; k++)
    #endif
    {
        dataPack.nTilesPerDim = std::vector<size_t>(dataPack.dim, k);
        testAlgorithm(dataPack, qualityMeasure, endPhaseRefinement);
    }
    #endif
}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testMaxTileCapacity(
    rd::tiled::rdTiledData<T> & dataPack,
    rd::RDAssessmentQuality<T> * qualityMeasure,
    bool endPhaseRefinement)
{
    #if defined(QUICK_TEST) || defined(RD_TEST_SCALING)
        int k = 0.2 * dataPack.np;
    #else
    for (int k = 0.05 * dataPack.np;
             k <= 0.25 * dataPack.np; 
             k += 0.05 * dataPack.np)
    #endif
    {
        dataPack.maxTileCapacity = k;
        testNTilesPerDim(dataPack, qualityMeasure, endPhaseRefinement);
    }

}

/**
 * @brief Test detection time & quality relative to algorithm parameter values
 */
template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testRDParams(
    int pointCnt,
    int dim,
    PointCloud<T> const & pc,
    std::vector<T> && points,
    rd::RDAssessmentQuality<T> * qualityMeasure,
    bool endPhaseRefinement)
{
    #if defined(RD_TEST_SCALING)
    T r = 2.f * pc.stddev_;

    #else
    std::vector<T> r1Vals{0.1f, 0.2f, 0.5f, 1.0f, 1.2f, 1.5f, 1.8f, 2.0f, 3.f, 4.f, 5.f, 10.f};
    // std::vector<T> r1Vals{2.0f};
    for (T& val : r1Vals)
    {
        val *= pc.stddev_;
    }

    for (T r : r1Vals)    
    #endif
    {
        rd::tiled::rdTiledData<T> dataPack;
        dataPack.dim = dim;
        dataPack.np = pointCnt;
        dataPack.r1 = r;
        dataPack.r2 = r * 2.f;
        dataPack.s = pc.stddev_;
        dataPack.P = points.data();
        pc.getCloudParameters(dataPack.a, dataPack.b);
        
        testMaxTileCapacity(dataPack, qualityMeasure, endPhaseRefinement);
    }
}

/**
 * @brief Test detection time & quality relative to end phase refinement 
 */
template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testEndPhaseRefinement(
    int pointCnt,
    int dim,
    PointCloud<T> const & pc,
    std::vector<T> && points,
    rd::RDAssessmentQuality<T> * qualityMeasure)
{
    // testRDParams(pointCnt, dim, pc, std::forward<std::vector<T>>(points), qualityMeasure, true);
    testRDParams(pointCnt, dim, pc, std::forward<std::vector<T>>(points), qualityMeasure, false);
    delete qualityMeasure;
}

/**
 * @brief Test detection time & quality relative to point dimension
 */
template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testDimensions(
    int pointCnt,
    PointCloud<T> & pc,
    int dim = -1)
{
    initializeLogFile<T>();

    if (dim > 0)
    {
        if (pc.dim_ < dim)
        {
            throw std::runtime_error("Input file data dimensionality"
                " is lower than requested!");
        }

        std::cout << rd::HLINE << std::endl;
        std::cout << ">>>> Dimension: " << dim << "D\n";

        if (g_logPerfResults)
        {
            T a, b;
            pc.getCloudParameters(a, b);
            *g_logFile << "%>>>> Dimension: " << dim << "D\n"
                << "% a: " << a << " b: " << b << " s: " << pc.stddev_ 
                << " pointsNum: " << pointCnt << "\n";
        }

        testEndPhaseRefinement(pointCnt, dim, pc, pc.extractPart(pointCnt, dim), 
            pc.getQualityMeasurer(dim));
    }
    else
    {
        if (pc.dim_ < MAX_TEST_DIM)
        {
            throw std::runtime_error("Input file data dimensionality"
                " is lower than requested!");
        }

        for (int d = MIN_TEST_DIM; d <= MAX_TEST_DIM; ++d)
        {
            std::cout << rd::HLINE << std::endl;
            std::cout << ">>>> Dimension: " << d << "D\n";

            if (g_logPerfResults)
            {
                T a, b;
                pc.getCloudParameters(a, b);
                *g_logFile << "%>>>> Dimension: " << d << "D\n"
                    << "% a: " << a << " b: " << b << " s: " << pc.stddev_ 
                    << " pointsNum: " << pointCnt << "\n";
            }

            testEndPhaseRefinement(pointCnt, d, pc, pc.extractPart(pointCnt, d), 
                pc.getQualityMeasurer(d));
        }
    }

    // clean-up
    if (g_logPerfResults)
    {
        g_logFile->close();
        delete g_logFile;
    }
}

/**
 * @brief Test detection time & quality relative to number of points
 */
template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testSize(
    PointCloud<T> & pc,
    int pointCnt = -1,
    int dim = 0,
    bool readFromFile = false)
{
    if (pointCnt > 0)
    {
        if (!readFromFile)
        {
            pc.pointCnt_ = pointCnt;
            pc.dim_ = (dim == 0) ? MAX_TEST_DIM : dim;
            pc.initializeData();
        }
        testDimensions<T>(pointCnt, pc, dim);
    }
    else
    {
        if (!readFromFile) 
        {
            pc.pointCnt_ = MAX_POINTS_NUM;
            pc.dim_ = (dim == 0) ? MAX_TEST_DIM : dim;
            pc.initializeData();
        }
        for (int k = 1e3; k <= MAX_POINTS_NUM; k *= 10)
        {
            std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t\t pointCnt: " << k  
                    << "\n//------------------------------------------\n";
            testDimensions<T>(k, pc, dim);
            if (k == MAX_POINTS_NUM) break;
            std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t pointCnt: " << k*2  
                    << "\n//------------------------------------------\n";
            testDimensions<T>(k*2, pc, dim);
            std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t pointCnt: " << k*5 
                    << "\n//------------------------------------------\n";
            testDimensions<T>(k*5, pc, dim);
        }
    }
}

//------------------------------------------------------------
//  MAIN
//------------------------------------------------------------

int main(int argc, char const **argv)
{

    #ifdef RD_USE_OPENMP
        g_threads = omp_get_num_procs();
    #endif

    float a = -1.f, b = -1.f, stddev = -1.f, segLength = -1.f;
    int pointCnt = -1;
    int dim = 0;
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
            "\t\t[--stddev <standard deviation of generated samples>]\n"
            "\t\t[--size <number of points>]\n"
            "\t\t[--dim <point dimension>]\n"
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
    if (args.CheckCmdLineFlag("dim"))
    {
        args.GetCmdLineArgument("dim", dim);
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

    if (pointCnt    < 0 ||
        stddev      < 0)
    {
        std::cout << "Have to specify parameters! Rerun with --help for more "
            "informations.\n";
        exit(1);
    }
    #ifdef QUICK_TEST
        if (a           < 0 ||
            b           < 0 )
        {
            std::cout << "Have to specify parameters! Rerun with --help for more "
                "informations.\n";
            exit(1);
        }
        PointCloud<float> && fpc = SpiralPointCloud<float>(a, b, pointCnt, dim, stddev);
        testDimensions<float>(pointCnt, fpc, dim);

        // PointCloud<double> && dpc = SpiralPointCloud<double>(a, b, pointCnt, dim, stddev);
        // testDimensions<double>(pointCnt, dpc, dim);
    #else
        #ifndef RD_DOUBLE_PRECISION
        #ifndef RD_TEST_SCALING
        if (a           < 0 ||
            b           < 0 )
        {
            std::cout << "Have to specify parameters! Rerun with --help for more "
                "informations.\n";
            exit(1);
        }
        // --size=1000000 --segl=0 --a=22.52 --b=11.31 --stddev=4.17 --log
        std::cout << "\n//------------------------------------------" 
                << "\n//\t\t (spiral) float: "  
                << "\n//------------------------------------------\n";
        PointCloud<float> && fpc3d = loadDataFromFile ?
            SpiralPointCloud<float>(inFilePath, a, b, pointCnt, 3, stddev) :
            SpiralPointCloud<float>(a, b, pointCnt, 3, stddev);
        testDimensions<float>(pointCnt, fpc3d, 2);
        testDimensions<float>(pointCnt, fpc3d, 3);

        #else  // #ifndef RD_TEST_SCALING      
        if (segLength   < 0)
        {
            std::cout << "Have to specify parameters! Rerun with --help for more "
                "informations.\n";
            exit(1);
        }
        // --segl=100 --a=0 --b=0 --stddev=2.17 --log
        std::cout << "\n//------------------------------------------" 
                << "\n//\t\t (segment) float: "  
                << "\n//------------------------------------------\n";        
        PointCloud<float> && fpc2 = loadDataFromFile ?
            SegmentPointCloud<float>(inFilePath, segLength, pointCnt, inFileDataDim, stddev) :
            SegmentPointCloud<float>(segLength, 0, 0, stddev);
        testSize<float>(fpc2, 0, 0, loadDataFromFile);
        
        std::cout << "\n//------------------------------------------" 
                << "\n//\t\t (spiral) float: "  
                << "\n//------------------------------------------\n";
        PointCloud<float> && fpc3d = loadDataFromFile ?
            SpiralPointCloud<float>(inFilePath, a, b, pointCnt, 3, stddev) :
            SpiralPointCloud<float>(a, b, pointCnt, 3, stddev);
        testSize<float>(fpc3d, 0, 2, loadDataFromFile);
        testSize<float>(fpc3d, 0, 3, loadDataFromFile);
        
        #endif  // #ifndef RD_TEST_SCALING
        #else   // #ifndef RD_DOUBLE_PRECISION
        #ifndef RD_TEST_SCALING
        if (a           < 0 ||
            b           < 0 )
        {
            std::cout << "Have to specify parameters! Rerun with --help for more "
                "informations.\n";
            exit(1);
        }
        std::cout << "\n//------------------------------------------" 
                << "\n//\t\t (spiral) double: "  
                << "\n//------------------------------------------\n";        
        PointCloud<double> && dpc3d = loadDataFromFile ?
            SpiralPointCloud<double>(inFilePath, a, b, pointCnt, 3, stddev) :
            SpiralPointCloud<double>(a, b, pointCnt, 3, stddev);
        testDimensions<double>(pointCnt, dpc3d, 2);
        testDimensions<double>(pointCnt, dpc3d, 3);

        #else // #ifndef RD_TEST_SCALING
        if (segLength   < 0)
        {
            std::cout << "Have to specify parameters! Rerun with --help for more "
                "informations.\n";
            exit(1);
        }
        std::cout << "\n//------------------------------------------" 
                << "\n//\t\t (segment) double: "  
                << "\n//------------------------------------------\n";
        PointCloud<double> && dpc2 = loadDataFromFile ?
            SegmentPointCloud<double>(inFilePath, segLength, pointCnt, inFileDataDim, stddev) :
            SegmentPointCloud<double>(segLength, 0, 0, stddev);
        testSize<double>(dpc2, 0, 0, loadDataFromFile);

        #endif  // RD_TEST_SCALING
        #endif  // RD_DOUBLE_PRECISION
    #endif

    std::cout << rd::HLINE << std::endl;
    std::cout << "END!" << std::endl;

    return EXIT_SUCCESS;
}

