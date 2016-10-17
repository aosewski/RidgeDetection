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
#include "rd/cpu/tiled/rd_tiled.hpp"
#include "rd/utils/samples_set.hpp"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/assessment_quality.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <tuple>
#include <cmath>

//------------------------------------------------------------
//  GLOBAL CONSTANTS / VARIABLES
//------------------------------------------------------------

static const std::string LOG_FILE_NAME_SUFFIX = "cpu_tiled_timings.txt";

std::ofstream * g_logFile           = nullptr;
bool            g_drawGraphs        = false;
bool            g_logPerfResults    = false;
bool            g_drawTilesGraphs   = false;
// whole time, build tree time, ridge detection time, end phase refinement time
typedef std::tuple<float, float, float, float> PerfResultsT;

std::vector<std::vector<float>> g_perf;           // storage for gnuplot data
static const float              g_graphColStep  = 0.3f;
static const int                g_graphNCol     = 5;     // group's columns count 
static const int                g_graphNGroups  = 6;     // number of dimensions to plot

#if defined(RD_PROFILE) || defined(RD_DEBUG)
static const int g_iterations = 1;
#else
static const int g_iterations = 10;
#endif

template <typename T> using Params = rd::RDTiledParams<T>;

template <typename T>
std::string rdToString(std::vector<T> const & v)
{
    std::string result;
    char comma[3] = {'\0', ' ', '\0'};
    result.reserve(v.size()*2+2);
    result += "[";
    for (const auto & e : v)
    {
        result += comma + std::to_string(e);
        // s << comma << e;
        comma[0] = ',';
    }
    result += "]";

    return result;
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
    bool                            verbose = false)
{
    std::cout << rd::HLINE << "\n";
    std::cout << rd::tiled::TiledRDAlgorithmNameTraits<ALGORITHM>::name
              << " threads: " << threads << "\n";
    std::cout << "% maxTileCapacity: " << dataPack.maxTileCapacity << " "
            << "nTilesPerDim: " << rdToString(dataPack.nTilesPerDim) << " "
            << "extTileFactor: " << dataPack.extTileFactor << " "
            << "r1: " << dataPack.r1 << " "
            << "r2: " << dataPack.r2 << " "
            << "endPhaseRefinement: " << std::boolalpha << endPhaseRefinement << "\n";

    T medianDist = 0, avgDist = 0, minDist = 0, maxDist = 0, hausdorffDist = 0;

    PerfResultsT testAvgTime = std::make_tuple(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < g_iterations; ++k)
    {
        PerfResultsT perf = rd::tiled::tiledRidgeDetection<ALGORITHM>(
            dataPack,
            threads,
            endPhaseRefinement,
            verbose,
            g_drawTilesGraphs);

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
    }

    if (dataPack.dim <= 3) 
    {
        rd::GraphDrawer<T> gDrawer;
        std::ostringstream s;

        s << typeid(T).name() << "_" << dataPack.dim << "D_" 
            << rd::tiled::TiledRDAlgorithmNameTraits<ALGORITHM>::name << "_result";
        gDrawer.startGraph(s.str(), dataPack.dim);
        gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
             dataPack.P, rd::GraphDrawer<T>::POINTS, dataPack.np);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#004cbf' ps 1.3 ",
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

    if (g_logFile != nullptr)
    {
        *g_logFile << "% " << rd::tiled::TiledRDAlgorithmNameTraits<ALGORITHM>::name
                    << " threads: " << threads << "\n";

        *g_logFile  << "% maxTileCapacity: " << dataPack.maxTileCapacity << " "
                    << "nTilesPerDim: " << rdToString(dataPack.nTilesPerDim) << " "
                    << "extTileFactor: " << dataPack.extTileFactor << " "
                    << "endPhaseRefinement: " << std::boolalpha << endPhaseRefinement << " "
                    << "np: " << dataPack.np << " "   
                    << "r1: " << dataPack.r1 << " "
                    << "r2: " << dataPack.r2 << " "
                    << "a: " << dataPack.a << " "
                    << "b: " << dataPack.b << " "
                    << "s: " << dataPack.s << "\n";
        *g_logFile << std::get<0>(testAvgTime) << " "
                    << std::get<1>(testAvgTime) << " "
                    << std::get<2>(testAvgTime) << " "
                    << std::get<3>(testAvgTime) << " "
                    << hausdorffDist << " "
                    << medianDist << " "
                    << avgDist << " "
                    << minDist << " "
                    << maxDist << "\n";
    }

    std::cout << "whole time: \t\t" << std::get<0>(testAvgTime) << "ms\n"
              << "build tree time: \t" << std::get<1>(testAvgTime) << "ms\n"
              << "ridge detection time: \t" << std::get<2>(testAvgTime) << "ms\n"
              << "refinement time: \t" << std::get<3>(testAvgTime) << "ms\n"
              << "hausdorffDist \t\t" << hausdorffDist << "\n"
              << "medianDist \t\t" << medianDist << "\n"
              << "avgDist \t\t" << avgDist << "\n"
              << "minDist \t\t" << minDist << "\n"
              << "maxDist \t\t" << maxDist << "\n";


    return testAvgTime;
}

//------------------------------------------------------------
//  TEST
//------------------------------------------------------------

template <typename T>
void runBenchmarks(
    Params<T> &rdp,
    rd::RDSpiralParams<T> &rds,
    rd::Samples<T> &samplesSet,
    int DIM,
    int threads)
{
    rd::RDAssessmentQuality<T> * qualityMeasure;
    // if b equals zero, then we test on a segment
    if (rds.b == 0 || rdp.dim > 3)
    {
        qualityMeasure = new rd::RDSegmentAssessmentQuality<T>(
            static_cast<size_t>(std::ceil(0.1f * rdp.np)), rdp.dim);
    }
    else
    {
        qualityMeasure = new rd::RDSpiralAssessmentQuality<T>(
            static_cast<size_t>(std::ceil(0.1f * rdp.np)), rdp.dim, rds.a, rds.b);
    }

    std::cout << "Samples: " << std::endl;
    std::cout <<  "\t dimension: " << rdp.dim << std::endl;
    std::cout <<  "\t n_samples: " << rdp.np << std::endl;

    std::cout << "Spiral params: " << std::endl;
    if (DIM == 2 || DIM == 3) 
    {
        std::cout <<  "\t a: " << rds.a << std::endl;
        std::cout <<  "\t b: " << rds.b << std::endl;
    }
    else
    {
        std::cout <<  "\t seg length: " << rds.a << std::endl;
    }
    std::cout <<  "\t sigma: " << rds.sigma << std::endl; 

    //---------------------------------------------------
    //              TEST ALL ALGORITHM VARIANTS
    //---------------------------------------------------

    std::cout << "-----start------" << std::endl;

    rd::tiled::rdTiledData<T> dataPack;
    dataPack.dim = rdp.dim;
    dataPack.np = rdp.np;
    dataPack.maxTileCapacity = rdp.maxTileCapacity;
    dataPack.nTilesPerDim = std::vector<size_t>(
        rdp.nTilesPerDim.begin(), rdp.nTilesPerDim.begin() + rdp.dim);
    dataPack.extTileFactor = rdp.extTileFactor;
    dataPack.r1 = rdp.r1;
    dataPack.r2 = rdp.r2;
    dataPack.a = rds.a;
    dataPack.b = rds.b;
    dataPack.s = rds.sigma;
    dataPack.P = samplesSet.samples_;

    auto processResults = [rdp, DIM](int graphSec, PerfResultsT perf)
    {
        if (g_drawGraphs)
        {
            g_perf[graphSec].push_back(DIM);
            // number of points in cloud
            g_perf[graphSec].push_back(rdp.np);
            // whole time
            g_perf[graphSec].push_back(std::get<0>(perf));
            // build tree time
            g_perf[graphSec].push_back(std::get<1>(perf));
            // ridge detection time
            g_perf[graphSec].push_back(std::get<2>(perf));
            // end phase refinement time
            g_perf[graphSec].push_back(std::get<3>(perf));
        }
    };

    processResults(0, benchmark<rd::tiled::TILED_LOCAL_TREE_LOCAL_RD>(
        dataPack, threads, rdp.endPhaseRefinement, qualityMeasure, rdp.verbose));
    std::cout << rd::HLINE << "\n";
    processResults(1, benchmark<rd::tiled::TILED_LOCAL_TREE_MIXED_RD>(
        dataPack, threads, rdp.endPhaseRefinement, qualityMeasure, rdp.verbose));
    std::cout << rd::HLINE << "\n";
    processResults(2, benchmark<rd::tiled::TILED_GROUPED_TREE_LOCAL_RD>(
        dataPack, threads, rdp.endPhaseRefinement, qualityMeasure, rdp.verbose));
    std::cout << rd::HLINE << "\n";
    processResults(3, benchmark<rd::tiled::TILED_GROUPED_TREE_MIXED_RD>(
        dataPack, threads, rdp.endPhaseRefinement, qualityMeasure, rdp.verbose));
    std::cout << rd::HLINE << "\n";
    processResults(4, benchmark<rd::tiled::EXT_TILE_TREE_MIXED_RD>(
        dataPack, threads, rdp.endPhaseRefinement, qualityMeasure, rdp.verbose));
    std::cout << rd::HLINE << "\n";

    delete qualityMeasure;
}

template <typename T>
void test(
    Params<T> &rdp,
    rd::RDSpiralParams<T> &rds,
    int DIM, 
    int threads)
{
    //---------------------------------------------------
    // program arguments correctnes check
    //---------------------------------------------------
    if (int(rdp.nTilesPerDim.size()) < DIM)
    {
        std::cerr << "If you use --tcnt argument you must pass number of"
            " values equal to max tested dimension!" << std::endl;
        exit(1);
    }


    rdp.dim = DIM;
    //---------------------------------------------------
    // Prepare logFile if needed
    //---------------------------------------------------

    if (g_logPerfResults)
    {
        std::ostringstream logFileName;
        // append device name to log file
        logFileName << std::to_string(DIM) << "D" << LOG_FILE_NAME_SUFFIX;

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

    //---------------------------------------------------
    // Prepare input data set and run benchmarks
    //---------------------------------------------------

    if (rds.loadFromFile)
    {
        for (std::string & fileName : rds.files)
        {
            rds.file = fileName;
            std::vector<std::string> samplesDir{"../../../examples/data/nd_segments/",
                 "../../../examples/data/spirals/"};
            rd::Samples<T> samplesSet(rdp, rds, samplesDir, DIM);

            runBenchmarks(rdp, rds, samplesSet, DIM, threads);
        }
    }
    else
    {
        rd::Samples<T> samplesSet(rdp, rds);
        runBenchmarks(rdp, rds, samplesSet, DIM, threads);
    }

    //---------------------------------------------------
    // clean-up
    if (g_logPerfResults)
    {
        g_logFile->close();
        delete g_logFile;
    }
}

//------------------------------------------------------------
//  DRAWING GRAPHS UTILITY
//------------------------------------------------------------

template <typename T>
std::string createFinalGraphDataFile()
{
    //------------------------------------------
    // create data file for drawing graph
    //------------------------------------------

    std::ostringstream graphDataFile;
    graphDataFile << typeid(T).name() << "_dim_scaling_graphData.dat";

    std::string filePath = rd::findPath("gnuplot_data/", graphDataFile.str());
    std::ofstream gdataFile(filePath.c_str(), std::ios::out | std::ios::app);
    if (gdataFile.fail())
    {
        throw std::logic_error("Couldn't open file: " + graphDataFile.str());
    }

    auto printData = [&gdataFile](std::vector<float> const &v, std::string secName)
    {
        gdataFile << "# [" << secName << "] \n";
        // we have 6 values in one line
        for (size_t i = 0; i < v.size()/6; ++i)
        {
            gdataFile << std::right << std::fixed << std::setw(6) << std::setprecision(2)
                << v[5 * i + 0] << " "      // DIM
                << v[5 * i + 1] << " "      // points count
                << v[5 * i + 2] << " "      // whole time
                << v[5 * i + 3] << " "      // build tree time
                << v[5 * i + 4] << " "      // rd time
                << v[5 * i + 5] << "\n";    // refinement time
                                            // 
        }
        // two sequential blank records to reset $0 counter
        gdataFile << "\n\n";
    };
    printData(g_perf[0], "TILED_LOCAL_TREE_LOCAL_RD");
    printData(g_perf[1], "TILED_LOCAL_TREE_MIXED_RD");
    printData(g_perf[2], "TILED_GROUPED_TREE_LOCAL_RD");
    printData(g_perf[3], "TILED_GROUPED_TREE_MIXED_RD");
    printData(g_perf[4], "EXT_TILE_TREE_MIXED_RD");

    gdataFile.close();
    return filePath;
}

//------------------------------------------------------------
//  MAIN
//------------------------------------------------------------

void printHelpAndExit(const char * msg = "") 
{
    printf("%s \n"
        "\nUsage (parameters in [] brackets are optional): \n"
        "\t\t[--np=<P size>]\n"
        "\t\t--r1=<r1 param>\n"
        "\t\t--r2=<r2 param>\n"
        "\t\t--tcnt=<cnt_d1>,<cnt_d2>... initial tile count per subsequent dimensions\n"
        "\t\t--tmax-size=<max tile capacity (points)>\n"
        "\t\t--e=<tile extension factor>\n"
        "\t\t[--endref <end phase refinement>]\n"
        "\t\t[--a=<spiral param>]\n"
        "\t\t[--b=<spiral param>]\n"
        "\t\t[--s=<spiral noise sigma>]\n"
        "\t\t[--f=<file name(s) to load and run test on>]\n"
        "\t\t[--t=<number of threads>]\n"
        "\t\t[--g <draw performance summary graphs>]\n"
        "\t\t[--drawTiles <draw each tile separately>]\n"
        "\t\t[--log <log performance results>]\n"
        "\t\t[--v <verbose output>]\n"
        "\n", msg);
    exit(0);
}

int main(int argc, char const **argv)
{
    Params<double> dParams;
    rd::RDSpiralParams<double> dSParams;
    Params<float> fParams;
    rd::RDSpiralParams<float> fSParams;

    int threads = 4;

    // Initialize command line
    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help") ||
             (args.ParsedArgc() < 5 ))
    {
        printHelpAndExit();
    }

    if (args.CheckCmdLineFlag("r1"))
    {
        args.GetCmdLineArgument("r1", dParams.r1);
        args.GetCmdLineArgument("r2", dParams.r2);
    }
    else
    {
        printHelpAndExit("You have to pass 'r1' parameter!");
    }

    if (args.CheckCmdLineFlag("r2"))
    {
        args.GetCmdLineArgument("r1", fParams.r1);
        args.GetCmdLineArgument("r2", fParams.r2);
    }
    else
    {
        printHelpAndExit("You have to pass 'r2' parameter!");
    }

    if (args.CheckCmdLineFlag("tcnt"))
    {
        args.GetCmdLineArguments("tcnt", dParams.nTilesPerDim);
        args.GetCmdLineArguments("tcnt", fParams.nTilesPerDim);
    }
    else
    {
        printHelpAndExit("You have to pass 'tcnt' parameter!");
    }

    if (args.CheckCmdLineFlag("tmax-size"))
    {
        args.GetCmdLineArgument("tmax-size", dParams.maxTileCapacity);
        args.GetCmdLineArgument("tmax-size", fParams.maxTileCapacity);
    }
    else
    {
        printHelpAndExit("You have to pass 'tmax-size' parameter!");
    }

    if (args.CheckCmdLineFlag("e"))
    {
        args.GetCmdLineArgument("e", dParams.extTileFactor);
        args.GetCmdLineArgument("e", fParams.extTileFactor);
    }
    else
    {
        printHelpAndExit("You have to pass 'e' parameter!");
    }

    if (args.CheckCmdLineFlag("endref"))
    {
        dParams.endPhaseRefinement = true;
        fParams.endPhaseRefinement = true;
    }
    
    if (args.CheckCmdLineFlag("v"))
    {
        dParams.verbose = true;
        fParams.verbose = true;
    }
    if (args.CheckCmdLineFlag("log"))
    {
        g_logPerfResults = true;
    }
    if (args.CheckCmdLineFlag("g")) 
    {
        g_drawGraphs = true;
    }
    if (args.CheckCmdLineFlag("drawTiles")) 
    {
        g_drawTilesGraphs = true;
    }

    if (args.CheckCmdLineFlag("f"))
    {
        args.GetCmdLineArguments("f",fSParams.files);
        args.GetCmdLineArguments("f",dSParams.files);      
        fSParams.loadFromFile = true;
        dSParams.loadFromFile = true;
    } 
    else
    {
        fSParams.loadFromFile = false;
        dSParams.loadFromFile = false;
        if (args.CheckCmdLineFlag("np"))
        {
            args.GetCmdLineArgument("np", dParams.np);
            args.GetCmdLineArgument("np", fParams.np);
        }
        else
        {   
            printHelpAndExit("If you don't use 'f' parameter You have to pass 'np' parameter!");
        }

        if (args.CheckCmdLineFlag("a"))
        {
            args.GetCmdLineArgument("a", fSParams.a);
            args.GetCmdLineArgument("a", dSParams.a);
        }
        if (args.CheckCmdLineFlag("b"))
        {
            args.GetCmdLineArgument("b", fSParams.b);
            args.GetCmdLineArgument("b", dSParams.b);
        }
        if (args.CheckCmdLineFlag("s"))
        {
            args.GetCmdLineArgument("s", fSParams.sigma);
            args.GetCmdLineArgument("s", dSParams.sigma);
        }
    }

    if (args.CheckCmdLineFlag("t"))
    {
        args.GetCmdLineArgument("t", threads);
    }

    if (g_drawGraphs)
    {
        // initialize storage for graph data
        g_perf = std::vector<std::vector<float>>(g_graphNCol);
    }

    //-----------------------------------------
    //  TESTS
    //-----------------------------------------

    /*
     * TODO: potrzebuję testów:
     * 1) czas wykonania w stosunku do wymiaru danych
     * 2) czas wykonania w stosunku do rozmiaru danych
     * 3) jakość dopasowania w stosunku do wybranych parametrów ((r1, r2), s)   (?)
     */


    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 2D: " << std::endl;
    test(fParams, fSParams, 2, threads);
    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 3D: " << std::endl;
    test(fParams, fSParams, 3, threads);
    std::cout << rd::HLINE << std::endl;
    // std::cout << "FLOAT 4D: " << std::endl;
    // test(fParams, fSParams, 4, threads);
    // std::cout << rd::HLINE << std::endl;
    // std::cout << "FLOAT 5D: " << std::endl;
    // test(fParams, fSParams, 5, threads);
    // std::cout << rd::HLINE << std::endl;
    // std::cout << "FLOAT 6D: " << std::endl;
    // test(fParams, fSParams, 6, threads);
    // std::cout << rd::HLINE << std::endl;

    // if (g_drawGraphs)
    // {
    //     createFinalGraphDataFile<float>();
    //     g_perf.clear();
    //     g_perf = std::vector<std::vector<float>>(g_graphNCol);
    // }

    // std::cout << "DOUBLE 2D: " << std::endl;
    // test(dParams, dSParams, 2, threads);
    // std::cout << rd::HLINE << std::endl;
    // std::cout << "DOUBLE 3D: " << std::endl;
    // test(dParams, dSParams, 3, threads);
    // std::cout << rd::HLINE << std::endl;
    // std::cout << "DOUBLE 4D: " << std::endl;
    // test(dParams, dSParams, 4, threads);
    // std::cout << rd::HLINE << std::endl;
    // std::cout << "DOUBLE 5D: " << std::endl;
    // test(dParams, dSParams, 5, threads);
    // std::cout << rd::HLINE << std::endl;
    // std::cout << "DOUBLE 6D: " << std::endl;
    // test(dParams, dSParams, 6, threads);
    // std::cout << rd::HLINE << std::endl;

    // if (g_drawGraphs)
    // {
    //     createFinalGraphDataFile<double>();

    //     g_perf.clear();
    // }

    std::cout << rd::HLINE << std::endl;
    std::cout << "END!" << std::endl;

    return EXIT_SUCCESS;
}

