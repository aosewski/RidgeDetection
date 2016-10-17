/**
 * @file test_tiled_tree.cpp
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

#include "rd/utils/rd_params.hpp"

#include "rd/utils/utilities.hpp"
#include "rd/cpu/tiled/rd_tiled.hpp"
#include "rd/utils/samples_set.hpp"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>

template <typename T> using Params = rd::RDTiledParams<T>;

template <typename T>
void testCpuSimulation(Params<T> &rd_p,
                        rd::RDSpiralParams<T> &sp,
                        int threads);

void printHelpAndExit(const char * msg = "") 
{
    printf("%s \n"
        "\nUsage (parameters in [] brackets are optional): \n"
        "\t\t[--np=<P size>]\n"
        "\t\t--r1=<r1 param>\n"
        "\t\t--r2=<r2 param>\n"
        "\t\t--tcnt=<cnt_d1>,<cnt_d2>... initial tile count per subsequent dimensions\n"
        "\t\t--tmax-size=<max tile capacity (points)>\n"
        "\t\t[--e=<tile extension factor>]\n"
        "\t\t[--a=<spiral param>]\n"
        "\t\t[--b=<spiral param>]\n"
        "\t\t[--s=<spiral noise sigma>]\n"
        "\t\t--dim=<data dimension>\n"
        "\t\t[--f=<samples input file>]\n"
        "\t\t[--t=<number of threads>]\n"
        "\t\t[--v] \n"
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

    if (args.CheckCmdLineFlag("v"))
    {
        dParams.verbose = true;
        fParams.verbose = true;
    }
    if (args.CheckCmdLineFlag("dim"))
    {
        args.GetCmdLineArgument("dim", fParams.dim);
        args.GetCmdLineArgument("dim", dParams.dim);
    }
    else
    {
        printHelpAndExit("You have to pass 'dim' parameter!");
    }

    if (args.CheckCmdLineFlag("f"))
    {
        args.GetCmdLineArgument("f",fSParams.file);
        args.GetCmdLineArgument("f",dSParams.file);
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

    std::cout << "\n\n TEST FLOAT " << std::endl;
    testCpuSimulation(fParams, fSParams, threads);
    std::cout << rd::HLINE << std::endl;
//    std::cout << "\n\n TEST DOUBLE " << std::endl;
//    testCpuSimulation(dParams, dSParams, threads);

    std::cout << rd::HLINE << std::endl;
    std::cout << "END!" << std::endl;

    return EXIT_SUCCESS;
}

template <typename T>
void testCpuSimulation(Params<T> &rdp,
                        rd::RDSpiralParams<T> &sp,
                        int threads)
{

    rd::Samples<T> samples;
    rd::GraphDrawer<T> gDrawer;

    T *samplesPtr;
    std::vector<std::string> samplesDir{"../../../examples/data/nd_segments/", "../../../examples/data/spirals/"};
    rd::Samples<T> samplesSet(rdp, sp, samplesDir, rdp.dim);
    samplesPtr = samplesSet.samples_;

	std::ostringstream s;
    if (rdp.verbose && rdp.dim < 3)
    {
		s << typeid(T).name() << "_initial_samples_set";
		gDrawer.showPoints(s.str(), samplesPtr, rdp.np, rdp.dim);
		s.clear();
		s.str(std::string());
    }

    /************************************
     * 		START ALGORITHM
     ************************************/

    std::cout << "-----start------" << std::endl;

    rd::tiled::rdTiledData<T> dataPack;
    dataPack.dim                = rdp.dim;
    dataPack.np                 = rdp.np;
    dataPack.maxTileCapacity    = rdp.maxTileCapacity;
    dataPack.nTilesPerDim       = rdp.nTilesPerDim;
    dataPack.extTileFactor      = rdp.extTileFactor;
    dataPack.r1                 = rdp.r1;
    dataPack.r2                 = rdp.r2;
    dataPack.P                  = samplesPtr;

    std::cout << rd::HLINE;
    std::cout << "\n TILED_GROUPED_TREE_LOCAL_RD \n";
    std::cout << rd::HLINE << "\n";

    rd::tiled::tiledRidgeDetection<rd::tiled::TILED_GROUPED_TREE_LOCAL_RD>(
        dataPack,
        threads,
        true,
        rdp.verbose);

    dataPack.S.clear();
    
    std::cout << rd::HLINE;
    std::cout << "\n TILED_GROUPED_TREE_MIXED_RD \n";
    std::cout << rd::HLINE << "\n";

    rd::tiled::tiledRidgeDetection<rd::tiled::TILED_GROUPED_TREE_MIXED_RD>(
        dataPack,
        threads,
        true,
        rdp.verbose);

    dataPack.S.clear();
    
    std::cout << rd::HLINE;
    std::cout << "\n EXT_TILE_TREE_MIXED_RD \n";
    std::cout << rd::HLINE << "\n";

    rd::tiled::tiledRidgeDetection<rd::tiled::EXT_TILE_TREE_MIXED_RD>(
        dataPack,
        threads,
        true,
        rdp.verbose);
        
    dataPack.S.clear();

    std::cout << rd::HLINE;
    std::cout << "\n TILED_LOCAL_TREE_LOCAL_RD \n";
    std::cout << rd::HLINE << "\n";

    rd::tiled::tiledRidgeDetection<rd::tiled::TILED_LOCAL_TREE_LOCAL_RD>(
        dataPack,
        threads,
        true,
        rdp.verbose);

    dataPack.S.clear();

    std::cout << rd::HLINE;
    std::cout << "\n TILED_LOCAL_TREE_MIXED_RD \n";
    std::cout << rd::HLINE << "\n";

    rd::tiled::tiledRidgeDetection<rd::tiled::TILED_LOCAL_TREE_MIXED_RD>(
        dataPack,
        threads,
        true,
        rdp.verbose);

    std::cout << "-----end------" << std::endl;
}
