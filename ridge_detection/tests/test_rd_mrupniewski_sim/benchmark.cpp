/**
 * @file benchmark.cpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is supervised by prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */


#include "rd/utils/utilities.hpp"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/assessment_quality.hpp" 
#include "rd/cpu/samples_generator.hpp"

#include "rd/cpu/mrupniewski/choose.hpp"
#include "rd/cpu/mrupniewski/evolve.hpp"
#include "rd/cpu/mrupniewski/decimate.hpp"
#include "rd/cpu/mrupniewski/order.hpp"

#include "tests/test_util.hpp"

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <typeinfo>
#include <sstream>
#include <iomanip>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>

#ifdef RD_USE_OPENMP
#include <omp.h>
#endif

//------------------------------------------------------------
//  GLOBAL CONSTANTS / VARIABLES
//------------------------------------------------------------

static const std::string LOG_FILE_NAME_SUFFIX = "cpu_mr_rd_timings.txt";

std::ofstream * g_logFile           = nullptr;
bool            g_logPerfResults    = false;
bool            g_drawResultsGraph  = false;

#if defined(RD_PROFILE) || defined(RD_DEBUG)
static const int g_iterations = 1;
#else
static const int g_iterations = 5;
#endif

static constexpr int MAX_TEST_DIM       = 12;
static constexpr int MIN_TEST_DIM       = 7;
static constexpr int MAX_POINTS_NUM     = int(1e6);

//------------------------------------------------------------
//  Utils
//------------------------------------------------------------

/**
 * @brief      Create if necessary and open log file. Allocate log file stream.
 */
template <typename T>
static void initializeLogFile(int dim = 0)
{
    if (g_logPerfResults)
    {
        std::ostringstream logFileName;
        if (dim > 0)
        {
            logFileName << dim << "D_";
        }
        logFileName << getCurrDate() << "_" << LOG_FILE_NAME_SUFFIX;

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
        logValue(*g_logFile, "dim", 10);
        logValue(*g_logFile, "r1", 10);
        logValue(*g_logFile, "r2", 10);
        logValue(*g_logFile, "chosenPointsNum", 15);
        logValue(*g_logFile, "avgCpuTime(ms)", 10);
        logValue(*g_logFile, "avgCpuTime", 16);
        logValue(*g_logFile, "minCpuTime", 10);
        logValue(*g_logFile, "maxCpuTime", 10);
        logValue(*g_logFile, "hausdorffDist", 13);
        logValue(*g_logFile, "medianDist", 10);
        logValue(*g_logFile, "avgDist", 10);
        logValue(*g_logFile, "minDist", 10);
        logValue(*g_logFile, "maxDist", 10);
        *g_logFile << "\n";
        g_logFile->flush();
    }
}

std::string convertMsToH_M_S_MS(float ms)
{
    std::chrono::milliseconds durationMs((long int)(floorf(ms)));
          
    std::chrono::hours durationHr;
    std::chrono::minutes durationM;
    long int minutes;
    std::chrono::seconds durationS;
    long int seconds;
    long int milliseconds;

    std::ostringstream outTime;

    durationHr = std::chrono::duration_cast<std::chrono::hours>(durationMs);
    outTime << durationHr.count() << "h-";

    durationM = std::chrono::duration_cast<std::chrono::minutes>(durationMs);
    minutes = durationM.count() - durationHr.count() * 60;
    outTime << minutes << "m-";

    durationS = std::chrono::duration_cast<std::chrono::seconds>(durationMs);
    seconds = durationS.count() - durationM.count() * 60;
    outTime << seconds << "s-";

    milliseconds = durationMs.count() - durationS.count() * 1000;
    outTime << milliseconds << "ms";

    return outTime.str();
}

//------------------------------------------------------------
//  RD ALGORITHM
//------------------------------------------------------------

template <typename T>
int ridgeDetection(
    T const * P, 
    int np, 
    T * S,
    T r1, 
    T r2, 
    int dim)
{
    int chosenCnt = choose(const_cast<T*>(P), np, S, r1, dim, NULL);

    // if (verbose)
    // {
    //     std::cout << "Wybrano reprezentantów, liczność zbioru S: " 
    //         << chosenCnt << std::endl;
    //     std::cout << rd::HLINE << std::endl;

    //     rd::GraphDrawer<T> gDrawer;
    //     std::ostringstream os;
    //     os << typeid(T).name();
    //     os << "_" << dim
    //         << "D_np=" << np;
    //     os << "_init_choosen_set";
    //     gDrawer.showPoints(os.str(), S, chosenCnt, dim);
    // }

    int oldCnt = 0;
    int iter = 0;
    while(oldCnt != chosenCnt)
    {
        oldCnt = chosenCnt;
        evolve(const_cast<T*>(P), np, r1, S, chosenCnt, dim, NULL);
        decimate(S, &chosenCnt, r2, dim, NULL);
        iter++;
    } 

    return chosenCnt;
}

//------------------------------------------------------------
//  INVOKE AND MEASURE 
//------------------------------------------------------------

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testAlgorithm(
    RdData<T> &                     dataPack,
    rd::RDAssessmentQuality<T> *    qualityMeasure)
{
    rd::CpuTimer timer;
    std::cout << rd::HLINE << "\n";
    std::cout << " r1: " << dataPack.r1
            << " r2: " << dataPack.r2
            << " pointCnt: " << dataPack.np << "\n";

    T medianDist = 0, avgDist = 0, minDist = 0, maxDist = 0, hausdorffDist = 0;
    float minTime = std::numeric_limits<float>::max();
    float maxTime = std::numeric_limits<float>::lowest();
    std::vector<T> chosenPoints;

    float testAvgTime = 0.f;
    for (int k = 0; k < g_iterations; ++k)
    {
        chosenPoints.clear();
        chosenPoints.resize(dataPack.np * dataPack.dim);
        timer.start();
        dataPack.ns = ridgeDetection(
                dataPack.P, 
                (int)dataPack.np, 
                chosenPoints.data(),
                dataPack.r1, 
                dataPack.r2, 
                (int)dataPack.dim);
        timer.stop();
        float currTime = timer.elapsedMillis(0);
        testAvgTime += currTime;
        minTime = std::min(currTime, minTime);
        maxTime = std::max(currTime, maxTime);
        
        printf("curr iter time: %12.3fms\n", currTime);

        chosenPoints.resize(dataPack.ns * dataPack.dim);

        // measure assessment quality
        hausdorffDist += qualityMeasure->hausdorffDistance(chosenPoints);
        T median, avg, min, max;
        qualityMeasure->setDistanceStats(chosenPoints, median, avg, min, max);
        avgDist += avg;
        medianDist += median;
        minDist += min;
        maxDist += max;
    }

    if (dataPack.dim <= 3 && g_drawResultsGraph) 
    {
        rd::GraphDrawer<T> gDrawer;
        std::ostringstream graphName;

        graphName << typeid(T).name() << "_" << getCurrDateAndTime() << "_"
            << "_np-" << dataPack.np    
            << "_r1-" << dataPack.r1 
            << "_r2-" << dataPack.r2 
            << "_a-" << dataPack.a 
            << "_b-" << dataPack.b 
            << "_s-" << dataPack.s
            << "_result";

        std::string filePath = rd::findPath("../img/", graphName.str());
        gDrawer.startGraph(filePath, dataPack.dim);

        if (dataPack.dim == 3)
        {
            gDrawer.setGraph3DConf();
        }

        gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#B8E186' ps 0.5 ",
             dataPack.P, rd::GraphDrawer<T>::POINTS, dataPack.np);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#D73027' ps 1.3 ",
             chosenPoints.data(), rd::GraphDrawer<T>::POINTS, dataPack.ns);
        gDrawer.endGraph();
    }

    testAvgTime /= g_iterations;
    hausdorffDist /= g_iterations;
    medianDist /= g_iterations;
    avgDist /= g_iterations;
    minDist /= g_iterations;
    maxDist /= g_iterations;

    if (g_logFile != nullptr)
    {
        logValues(*g_logFile,
            dataPack.np, 
            dataPack.dim,
            dataPack.r1,
            dataPack.r2,
            dataPack.ns,
            testAvgTime,
            convertMsToH_M_S_MS(testAvgTime),
            minTime,
            maxTime,
            hausdorffDist,
            medianDist,
            avgDist,
            minDist,
            maxDist);
        *g_logFile << std::endl;
        g_logFile->flush();
    }

    std::cout << "avg time: \t\t" << testAvgTime << "ms "
              << "-> " << convertMsToH_M_S_MS(testAvgTime) << "\n "
              << "minTime \t\t" << minTime << "\n"
              << "maxTime \t\t" << maxTime << "\n"
              << "hausdorffDist \t\t" << hausdorffDist << "\n"
              << "medianDist \t\t" << medianDist << "\n"
              << "avgDist \t\t" << avgDist << "\n"
              << "minDist \t\t" << minDist << "\n"
              << "maxDist \t\t" << maxDist << "\n";
}

//------------------------------------------------------------
//  Test generation
//------------------------------------------------------------


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
    rd::RDAssessmentQuality<T> * qualityMeasure)
{
    T r = 2.f * pc.stddev_;
    RdData<T> dataPack;
    dataPack.dim = dim;
    dataPack.np = pointCnt;
    dataPack.r1 = r;
    dataPack.r2 = r * 2.f;
    dataPack.s = pc.stddev_;
    dataPack.P = points.data();
    pc.getCloudParameters(dataPack.a, dataPack.b);
    
    testAlgorithm(dataPack, qualityMeasure);

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
    initializeLogFile<T>(dim);
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
        testRDParams(pointCnt, dim, pc, pc.extractPart(pointCnt, dim), 
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
            testRDParams(pointCnt, d, pc, pc.extractPart(pointCnt, d), 
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
                    << "\n//\t\t pointCnt: " << k  
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
    float a = -1.f, b = -1.f, stddev = -1.f, segLength = -1.f;
    int pointCnt = -1;
    // int dim = RD_TEST_DIM;
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
    #ifndef RD_DOUBLE_PRECISION
        if (a           < 0 ||
            b           < 0 ||
            dim         < 0)
        {
            std::cout << "Have to specify parameters! Rerun with --help for more "
                "informations.\n";
            exit(1);
        }
        // --size=1000000 --segl=0 --a=22.52 --b=11.31 --stddev=4.17 --rGraphs --log
        std::cout << "\n//------------------------------------------"
                   << "\n//\t\t (spiral) float: "
                   << "\n//------------------------------------------\n";
        PointCloud<float> && fpc3d = loadDataFromFile ?
            SpiralPointCloud<float>(inFilePath, a, b, pointCnt, 3, stddev) :
            SpiralPointCloud<float>(a, b, pointCnt, 3, stddev);
        testDimensions<float>(pointCnt, fpc3d, 2);
        testDimensions<float>(pointCnt, fpc3d, 3);
        testSize<float>(fpc3d, 0, dim, loadDataFromFile);

        if (segLength   < 0 ||
            dim         < 0 )
        {
            std::cout << "Have to specify parameters! Rerun with --help for more "
                "informations.\n";
            exit(1);
        }
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) float: "  
                    << "\n//------------------------------------------\n";
        // --segl=100 --a=0 --b=0 --stddev=2.17 --rGraphs --log
        PointCloud<float> && fpc2 = loadDataFromFile ?
            SegmentPointCloud<float>(inFilePath, segLength, pointCnt, inFileDataDim, stddev) :
            SegmentPointCloud<float>(segLength, 0, 0, stddev);
        testSize<float>(fpc2, 1000000, dim, loadDataFromFile);

    #else

        std::cout << "\n//------------------------------------------"
                   << "\n//\t\t (spiral) double: "
                   << "\n//------------------------------------------\n";
        PointCloud<double> && dpc3d = loadDataFromFile ?
            SpiralPointCloud<double>(inFilePath, a, b, pointCnt, 3, stddev) :
            SpiralPointCloud<double>(a, b, pointCnt, 3, stddev);
        testDimensions<double>(pointCnt, dpc3d, 2);
        testDimensions<double>(pointCnt, dpc3d, 3);

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) double: "  
                    << "\n//------------------------------------------\n";
        PointCloud<double> && dpc2 = loadDataFromFile ?
            SegmentPointCloud<double>(inFilePath, segLength, pointCnt, inFileDataDim, stddev) :
            SegmentPointCloud<double>(segLength, 0, 0, stddev);
        testSize<double>(dpc2, 0, dim, loadDataFromFile);
    #endif

    std::cout << rd::HLINE << std::endl;
    std::cout << "END!" << std::endl;

    return EXIT_SUCCESS;
}


