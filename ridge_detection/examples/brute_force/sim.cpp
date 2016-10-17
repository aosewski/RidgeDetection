/**
 *  @file sim.cpp
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

#include "rd/utils/rd_params.hpp"
#include "rd/cpu/samples_set.hpp"
#include "rd/cpu/brute_force/order.hpp"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"

#include "rd.cuh"

#include <cstdio>
#include <string>

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <cassert>


#define TEST_DIMENSION 3

typedef float DataType;

enum SAMPLES_MODE 
{ 
    SM_GENERATE,
    SM_FILE,
    SM_PIPE
};

int main(int argc, char const * argv[]) {

    rd::RDParams<DataType> rdp;
    rd::RDSpiralParams<DataType> sp;


    int devId = 0;
    // int cpuThreads = 1;

    SAMPLES_MODE sMode = SM_GENERATE;
    std::string inFileName, inFileDir;
    char valSeparator = ' ';

    // Read command line args
    CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help")   ||
        args.ParsedArgc() < 2           ||
        !args.CheckCmdLineFlag("r1")    ||
        !args.CheckCmdLineFlag("r2"))
    {
        printf("%s \n"
            "\t\t--r1=<r1 param>\n"
            "\t\t--r2=<r2 param>\n"
            "\t\t[--np=<P size>]\n"
            "\t\t[--a=<spiral param>]\n"
            "\t\t[--b=<spiral param>]\n"
            "\t\t[--s=<spiral noise sigma>]\n"
            "\t\t[--d=<device id>]\n"
            // "\t\t[--t=<CPU threads>]\n"
            "\t\t[--f=<source file name>] \n"
            "\t\t[--fdir=<source file directory relative to binary file>] \n"
            "\t\t[--vd=<values delimiter>] \n"
            "\t\t[--p<pipeline mode>] \n"
            "\t\t[--v] \n"
            "\n", argv[0]);
        exit(0);
    }


    if (args.CheckCmdLineFlag("r1")) 
    {
        args.GetCmdLineArgument("r1", rdp.r1);
    }
    if (args.CheckCmdLineFlag("r2")) 
    {
        args.GetCmdLineArgument("r2", rdp.r2);
    }
    if (args.CheckCmdLineFlag("v")) 
    {
        rdp.verbose = true;
    }
    if (args.CheckCmdLineFlag("np")) 
    {
        args.GetCmdLineArgument("np", rdp.np);
        sMode = SM_GENERATE;
    }
    if (args.CheckCmdLineFlag("a")) 
    {
        args.GetCmdLineArgument("a", sp.a);
    }
    if (args.CheckCmdLineFlag("b")) 
    {
        args.GetCmdLineArgument("b", sp.b);
    }
    if (args.CheckCmdLineFlag("s")) 
    {
        args.GetCmdLineArgument("s", sp.sigma);
    }
    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", devId);
    }
    // if (args.CheckCmdLineFlag("t")) 
    // {
    //     args.GetCmdLineArgument("t", cpuThreads);
    // }
    if (args.CheckCmdLineFlag("f")) 
    {
        args.GetCmdLineArgument("f",inFileName);
        sMode = SM_FILE;
    }
    if (args.CheckCmdLineFlag("fdir")) 
    {
        args.GetCmdLineArgument("fdir",inFileDir);
    }
    if (args.CheckCmdLineFlag("vd"))
    {
        args.GetCmdLineArgument("vd",valSeparator);
    }
    if (args.CheckCmdLineFlag("p"))
    {
        sMode = SM_PIPE;
    }


    rdp.dim = TEST_DIMENSION;

    rd::Samples<DataType> hPointCloud;
    rd::GraphDrawer<DataType> gDrawer;
    CpuTimer timer;
    
    switch (sMode)
    {
        case SM_GENERATE:
            sp.loadFromFile = false;

            break;
        case SM_FILE:
            sp.loadFromFile = true;

            hPointCloud.loadFromFile(inFileName, inFileDir, valSeparator);
            assert(hPointCloud.dim_ == TEST_DIMENSION);
            rdp.np = hPointCloud.size_;

            break;
        case SM_PIPE:
            sp.loadFromFile = true;

            hPointCloud.loadFromInputStream(valSeparator);
            assert(hPointCloud.dim_ == TEST_DIMENSION);
            rdp.np = hPointCloud.size_;
            
            break;
        default:
            throw std::logic_error("Unsupported samples mode!");
    }


    //-----------------------------------------------------------------
    rd::Samples<DataType> hGpuRidge;

    std::cout << "sample count: " << rdp.np << std::endl;
    std::cout << "r1: " << rdp.r1 << std::endl;
    std::cout << "r2: " << rdp.r2 << std::endl;

    if (sMode == SM_GENERATE)
    {
        std::cout << "Spiral params: " << std::endl;
        std::cout << "\n a: " << sp.a << ", b: " << sp.b << ", sigma: " << sp.sigma << std::endl;
    }

    std::cout << HLINE << std::endl;
    std::cout << "START!" << std::endl;

    timer.start();
    // gpu simulation
    ridgeDetection3D(hPointCloud.samples_, &hGpuRidge.samples_, rdp, sp, devId);
    timer.stop();
    timer.elapsedMillis(0);

    hGpuRidge.dim_ = TEST_DIMENSION;
    hGpuRidge.size_ = rdp.ns;

    //-----------------------------------------------------------------

    std::ostringstream os;
    os << typeid(DataType).name() << "_" << TEST_DIMENSION;
    os << "D_detected_ridge";
    gDrawer.showPoints(os.str(), hGpuRidge.samples_, hGpuRidge.size_,
     TEST_DIMENSION);

    std::ostringstream comment;
    comment << "Detected ridge: ns: " << hGpuRidge.size_;
    hGpuRidge.saveToFile("detected_ridge", comment.str());

    //-----------------------------------------------------------------

    DataType r2Sqr = rdp.r2 * rdp.r2;
    std::list<std::deque<DataType const*>> hChainList = rd::orderSamples<DataType>(
        hGpuRidge.samples_, TEST_DIMENSION, hGpuRidge.size_, 
        [&r2Sqr](DataType const *p1,
                 DataType const *p2,
                 size_t dim) -> bool{
            return squareEuclideanDistance(p1, p2, dim) <= 4.f * r2Sqr;
        });

    //-----------------------------------------------------------------
    std::cout << "Coherent point chains:" << std::endl;

    std::vector<DataType*> continuousMemChainsPoints;

    std::string plotcmd1 = "'-' w p pt ";
    std::string plotcmd2 = " lc rgb '#d64f4f' ps 1 ";

    int chainIdx = 0;
    gDrawer.startGraph("detected_ridge_chains", rdp.dim);
    for (auto chain : hChainList)
    {

        DataType *chainContMem = new DataType[chain.size()*rdp.dim];
        continuousMemChainsPoints.push_back(chainContMem);
        os.clear();
        os.str(std::string());
        os << plotcmd1 << (chainIdx % 15) + 1 << plotcmd2;
        gDrawer.addPlotCmd(os.str(), chainContMem,
            rd::GraphDrawer<DataType>::POINTS, chain.size());

        std::cout << "============ chain: " << ++chainIdx << "===========" << std::endl;
        std::cout.precision(4);
        for (auto ptr : chain)
        {
            for (size_t d = 0; d < rdp.dim; ++d)
            {
                *chainContMem++ = ptr[d];
                std::cout << std::right << std::fixed << std::setw(10) << ptr[d];
            }
            std::cout << std::endl;
        }

    }
    gDrawer.endGraph();

    for (DataType *ptr : continuousMemChainsPoints)
        delete[] ptr;

    //-----------------------------------------------------------------

    std::cout << HLINE << std::endl;
    std::cout << "END!" << std::endl;

    return 0;
}
