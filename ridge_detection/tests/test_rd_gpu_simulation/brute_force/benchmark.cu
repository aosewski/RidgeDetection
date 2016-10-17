/**
 *  @file simulation.cu
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

#define BLOCK_TILE_LOAD_V4 1

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <limits>
#ifdef RD_USE_OPENMP
    #include <omp.h>
#endif

#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "rd/gpu/device/brute_force/simulation.cuh"
#include "rd/gpu/device/samples_generator.cuh"
#include "rd/gpu/util/dev_memcpy.cuh"
#include "rd/gpu/util/dev_static_for.cuh"

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/memory.h"
#include "rd/utils/name_traits.hpp"

#include "tests/test_util.hpp"

#include "cub/test_util.h"

#include "rd/utils/rd_params.hpp"


//------------------------------------------------------------
//  GLOBAL CONSTANTS / VARIABLES
//------------------------------------------------------------

static const std::string LOG_FILE_NAME_SUFFIX = "gpu_brute_force_timings.txt";

std::ofstream * g_logFile           = nullptr;
bool            g_logPerfResults    = false;
bool            g_drawResultsGraph  = false;
std::string     g_devName;

#if defined(RD_PROFILE) || defined(RD_DEBUG) || defined(QUICK_TEST)
static constexpr int g_iterations = 1;
#else
static constexpr int g_iterations = 5;
#endif

static int g_devId      = 0;

#ifdef QUICK_TEST
static constexpr int MAX_TEST_DIM = 4;
#else
static constexpr int MIN_TEST_DIM = 2;
static constexpr int MAX_TEST_DIM = 12;
#endif

static constexpr int MAX_POINTS_NUM = int(1e7);

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
        logFileName << getCurrDate() << "_" <<
            g_devName << "_" << LOG_FILE_NAME_SUFFIX;

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
        logValue(*g_logFile, "r1", 10);
        logValue(*g_logFile, "r2", 10);
        logValue(*g_logFile, "inMemLayout", 11);
        logValue(*g_logFile, "outMemLayout", 12);
        logValue(*g_logFile, "chosenPtsNum", 12);
        logValue(*g_logFile, "avgCpuTime", 10);
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

//------------------------------------------------------------
//  INVOKE AND MEASURE 
//------------------------------------------------------------

template <
    int                     DIM,
    rd::DataMemoryLayout    IN_MEMORY_LAYOUT,
    rd::DataMemoryLayout    OUT_MEMORY_LAYOUT,
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void benchmark(
    RdData<T> &                     dataPack,
    rd::RDAssessmentQuality<T> *    qualityMeasure,
    bool                            verbose = false)
{

    using namespace rd::gpu;

    // verbose = true;
    bruteForce::RidgeDetection<T, DIM, IN_MEMORY_LAYOUT, OUT_MEMORY_LAYOUT> rdGpu(
        dataPack.np,
        dataPack.r1, 
        dataPack.r2, 
        verbose);

    // copy and if necessary transpose input data to gpu device
    if (IN_MEMORY_LAYOUT == rd::ROW_MAJOR)
    {
        rdMemcpy<rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            rdGpu.dP_, dataPack.P, DIM, dataPack.np, DIM, DIM);
    }
    else
    {
        rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            rdGpu.dP_, dataPack.P, DIM, dataPack.np, rdGpu.pPitch_, DIM * sizeof(T));
    }

    checkCudaErrors(cudaDeviceSynchronize());

    rd::CpuTimer timer;

    T medianDist = 0, avgDist = 0, minDist = 0, maxDist = 0, hausdorffDist = 0;
    float minCpuTime = std::numeric_limits<float>::max();
    float maxCpuTime = std::numeric_limits<float>::lowest();
    std::vector<T> chosenPoints;

    float testAvgCpuTime = 0.f;
    for (int k = 0; k < g_iterations; ++k)
    {
        chosenPoints.clear();
        timer.start();
        rdGpu.ridgeDetection();
        timer.stop();
        
        float currTime = timer.elapsedMillis(0);
        testAvgCpuTime += currTime;
        minCpuTime = min(currTime, minCpuTime);
        maxCpuTime = max(currTime, maxCpuTime);

        rd::CpuTimer qmesTimer, postprcsTimer;

        postprcsTimer.start();
        rdGpu.getChosenSamplesCount();
        dataPack.ns = rdGpu.ns_;
        chosenPoints.resize(rdGpu.ns_ * dataPack.dim);
        
        // copy back to host results
        if (OUT_MEMORY_LAYOUT == rd::ROW_MAJOR)
        {
            rdMemcpy<rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyDeviceToHost>(
                chosenPoints.data(), rdGpu.dS_, DIM, rdGpu.ns_, DIM, DIM);
        }
        else
        {
            rdMemcpy2D<rd::ROW_MAJOR, rd::COL_MAJOR, cudaMemcpyDeviceToHost>(
                chosenPoints.data(), rdGpu.dS_, rdGpu.ns_, DIM, DIM * sizeof(T),
                rdGpu.sPitch_);
        }

        // chosenPoints.resize(rdGpu.ns_ * dataPack.dim);
        // measure assessment quality
        qmesTimer.start();
        hausdorffDist += qualityMeasure->hausdorffDistance(chosenPoints);
        T median, avg, min, max;
        
        qualityMeasure->setDistanceStats(chosenPoints, median, avg, min, max);
        avgDist += avg;
        medianDist += median;
        minDist += min;
        maxDist += max;
        qmesTimer.stop();
        postprcsTimer.stop();
        std::cout << "postprocess (quality measure): " << qmesTimer.elapsedMillis(0) << "ms" 
                << "\tpostprocess (all): " << postprcsTimer.elapsedMillis(0) << "ms" 
                << "\tcomputation cpu time: " << timer.elapsedMillis(0) << "ms" << std::endl;
    }

    if (dataPack.dim <= 3 && g_drawResultsGraph) 
    {
        rd::GraphDrawer<T> gDrawer;
        std::ostringstream graphName;

        graphName << typeid(T).name() << "_" << getCurrDateAndTime() << "_"
            << g_devName 
            << "_" << rd::DataMemoryLayoutNameTraits<IN_MEMORY_LAYOUT>::shortName
            << "_" << rd::DataMemoryLayoutNameTraits<OUT_MEMORY_LAYOUT>::shortName
            << "_np-" << dataPack.np    
            << "_r1-" << dataPack.r1 
            << "_r2-" << dataPack.r2 
            << "_a-" << dataPack.a 
            << "_b-" << dataPack.b 
            << "_s-" << dataPack.s
            << "_result";

        std::string filePath = rd::findPath("img/", graphName.str());
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

    testAvgCpuTime /= g_iterations;
    hausdorffDist /= g_iterations;
    medianDist /= g_iterations;
    avgDist /= g_iterations;
    minDist /= g_iterations;
    maxDist /= g_iterations;

    if (g_logFile != nullptr)
    {
        logValues(*g_logFile,
            dataPack.np,
            DIM,
            dataPack.r1, 
            dataPack.r2,
            std::string(rd::DataMemoryLayoutNameTraits<IN_MEMORY_LAYOUT>::shortName),
            std::string(rd::DataMemoryLayoutNameTraits<OUT_MEMORY_LAYOUT>::shortName),
            dataPack.ns,
            testAvgCpuTime,
            minCpuTime,
            maxCpuTime,
            hausdorffDist,
            medianDist,
            avgDist,
            minDist,
            maxDist);
        
        *g_logFile << "\n";
        g_logFile->flush();
    }

    logValues(std::cout,
        dataPack.np,
        DIM,
        dataPack.r1, 
        dataPack.r2,
        std::string(rd::DataMemoryLayoutNameTraits<IN_MEMORY_LAYOUT>::name),
        std::string(rd::DataMemoryLayoutNameTraits<OUT_MEMORY_LAYOUT>::name),
        dataPack.ns,
        testAvgCpuTime,
        minCpuTime,
        maxCpuTime,
        hausdorffDist,
        medianDist,
        avgDist,
        minDist,
        maxDist);
    std::cout << std::endl;

}

//------------------------------------------------------------
//  Test generation
//------------------------------------------------------------

template <
    int                     DIM,
    typename                T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testMemLayout(
    RdData<T> &                     dataPack,
    rd::RDAssessmentQuality<T> *    qualityMeasure)
{
    benchmark<DIM, rd::COL_MAJOR, rd::COL_MAJOR>(dataPack, qualityMeasure);
    #ifndef QUICK_TEST
    benchmark<DIM, rd::COL_MAJOR, rd::ROW_MAJOR>(dataPack, qualityMeasure);
    benchmark<DIM, rd::ROW_MAJOR, rd::ROW_MAJOR>(dataPack, qualityMeasure);
    #endif
}

/**
 * @brief Test detection time & quality relative to algorithm parameter values
 */
template <
    int         DIM,
    typename    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testRDParams(
    int pointCnt,
    int dim,
    PointCloud<T> const & pc,
    std::vector<T> && points,
    rd::RDAssessmentQuality<T> * qualityMeasure)
{
    #ifdef QUICK_TEST
    T r1 = 1.7f * pc.stddev_;
    T r2 = 2.0f * r1;
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
        RdData<T> dataPack;
        dataPack.dim = dim;
        dataPack.np = pointCnt;
        dataPack.r1 = r;
        dataPack.r2 = r * 2.f;
        dataPack.s = pc.stddev_;
        dataPack.P = points.data();
        pc.getCloudParameters(dataPack.a, dataPack.b);
        
        testMemLayout<DIM>(dataPack, qualityMeasure);
    }
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

        testRDParams<D::value>(pointCnt, idx,  pc,  pc.extractPart(pointCnt, idx), 
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

        testRDParams<DIM>(pointCnt, DIM, pc, pc.extractPart(pointCnt, DIM), 
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
        if (pc.dim_ < MAX_TEST_DIM)
        {
            throw std::runtime_error("Input file data dimensionality"
                " is lower than requested!");
        }

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
            "\t\t[--a <a parameter of spiral>]\n"
            "\t\t[--b <b parameter of spiral>]\n"
            "\t\t[--segl <generated N-dimensional segment length>]\n"
            "\t\t[--stddev <standard deviation of generated samples>]\n"
            "\t\t[--size <number of points>]\n"
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
        const int dim = 5;

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t float: "  
                    << "\n//------------------------------------------\n";
        // PointCloud<float> && fpc = SpiralPointCloud<float>(a, b, pointCnt, dim, stddev);
        // TestDimensions<dim, float>::impl(fpc, pointCnt);
        PointCloud<float> && fpc = SegmentPointCloud<float>(segLength, pointCnt, dim, stddev);
        TestDimensions<dim, float>::impl(fpc, pointCnt);

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t double: "  
                    << "\n//------------------------------------------\n";
        // PointCloud<double> && dpc = SpiralPointCloud<double>(a, b, pointCnt, 2, stddev);
        // TestDimensions<2, double>::impl(dpc, pointCnt);
        PointCloud<double> && dpc = SegmentPointCloud<double>(segLength, pointCnt, dim, stddev);
        TestDimensions<dim, double>::impl(dpc, pointCnt);
    #else
        #ifndef RD_DOUBLE_PRECISION
        // --size=1000000 --segl=1457.75 --a=22.52 --b=11.31 --stddev=4.17 --rGraphs --log --d=0
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (spiral) float: "  
                    << "\n//------------------------------------------\n";
        PointCloud<float> && fpc3d = loadDataFromFile ?
            SpiralPointCloud<float>(inFilePath, a, b, pointCnt, 3, stddev) :
            SpiralPointCloud<float>(a, b, pointCnt, 3, stddev);
        testSize<float, 2>(fpc3d, 0, loadDataFromFile);
        testSize<float, 3>(fpc3d, 0, loadDataFromFile);

        // --segl=100 --a=0 --b=0 --stddev=2.17 --rGraphs --log --d=0
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) float: "  
                    << "\n//------------------------------------------\n";
        PointCloud<float> && fpc2 = loadDataFromFile ?
            SegmentPointCloud<float>(inFilePath, segLength, pointCnt, inFileDataDim, stddev) :
            SegmentPointCloud<float>(segLength, 0, 0, stddev);
        testSize<float>(fpc2, 0, loadDataFromFile);
        #else
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (spiral) double: "  
                    << "\n//------------------------------------------\n";
        PointCloud<double> && dpc3d = loadDataFromFile ?
            SpiralPointCloud<double>(inFilePath, a, b, pointCnt, 3, stddev) :
            SpiralPointCloud<double>(a, b, pointCnt, 3, stddev);
        TestDimensions<2, double>::impl(dpc2d, pointCnt);
        TestDimensions<3, double>::impl(dpc3d, pointCnt);

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) double: "  
                    << "\n//------------------------------------------\n";
        PointCloud<double> && dpc2 = loadDataFromFile ?
            SegmentPointCloud<double>(inFilePath, segLength, pointCnt, inFileDataDim, stddev) :
            SegmentPointCloud<double>(segLength, 0, 0, stddev);
        testSize<double>(dpc2, 0, loadDataFromFile);
        #endif
    #endif

    checkCudaErrors(cudaDeviceReset());

    std::cout << rd::HLINE << std::endl;
    std::cout << "END!" << std::endl;

    return EXIT_SUCCESS;
}
