/**
 * @file test_rd_cpu_simulation.cpp
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
#include "rd/utils/samples_set.hpp"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/cpu/brute_force/ridge_detection.hpp"


#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <typeinfo>
#include <sstream>
#include <iomanip>
#include <vector>

template <typename T>
void testCpuSimulation(rd::RDParams<T> &rd_p, rd::RDSpiralParams<T> &sp, int threads);

int main(int argc, char const *argv[]) {

    rd::RDParams<double> dParams;
    rd::RDSpiralParams<double> dSParams;
    rd::RDParams<float> fParams;
    rd::RDSpiralParams<float> fSParams;

    int threads = 4;


    // Initialize command line
    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help") ||
             (args.ParsedArgc() < 3 )) {
        printf("%s \n"
            "\t\t--np=<P size>\n"
            "\t\t--r1=<r1 param>\n"
            "\t\t--r2=<r2 param>\n"
            "\t\t[--a=<spiral param>]\n"
            "\t\t[--b=<spiral param>]\n"
            "\t\t[--s=<spiral noise sigma>]\n"
            "\t\t[--dim=<data dimension>]\n"
            "\t\t[--f=<samples input file>]\n"
            "\t\t[--t=<number of threads>]\n"
            "\t\t[--o=<order samples>]\n"
            "\t\t[--v] \n"
            "\n", argv[0]);
        exit(0);
    }

    args.GetCmdLineArgument("r1", dParams.r1);
    args.GetCmdLineArgument("r2", dParams.r2);

    args.GetCmdLineArgument("r1", fParams.r1);
    args.GetCmdLineArgument("r2", fParams.r2);


    if (args.CheckCmdLineFlag("v")) {
        dParams.verbose = true;
        fParams.verbose = true;
    }
    if (args.CheckCmdLineFlag("o")) {
        dParams.order = true;
        fParams.order = true;
    }
    if (args.CheckCmdLineFlag("f")) {
        args.GetCmdLineArgument("f",fSParams.file);
        args.GetCmdLineArgument("f",dSParams.file);     
        fSParams.loadFromFile = true;
        dSParams.loadFromFile = true;
    } else {
        fSParams.loadFromFile = false;
        dSParams.loadFromFile = false;

        args.GetCmdLineArgument("np", dParams.np);
        args.GetCmdLineArgument("np", fParams.np);

        if (args.CheckCmdLineFlag("dim")) {
            args.GetCmdLineArgument("dim", fParams.dim);
            args.GetCmdLineArgument("dim", dParams.dim);
        }
        if (args.CheckCmdLineFlag("a")) {
            args.GetCmdLineArgument("a", fSParams.a);
            args.GetCmdLineArgument("a", dSParams.a);
        }
        if (args.CheckCmdLineFlag("b")) {
            args.GetCmdLineArgument("b", fSParams.b);
            args.GetCmdLineArgument("b", dSParams.b);
        }
        if (args.CheckCmdLineFlag("s")) {
            args.GetCmdLineArgument("s", fSParams.sigma);
            args.GetCmdLineArgument("s", dSParams.sigma);
        }
    }

    if (args.CheckCmdLineFlag("t")) {
        args.GetCmdLineArgument("t", threads);
    }

    std::cout << "\n\n TEST FLOAT " << "\n";
    testCpuSimulation(fParams, fSParams, threads);
    std::cout << rd::HLINE << "\n";
    std::cout << "\n\n TEST DOUBLE " << "\n";
    testCpuSimulation(dParams, dSParams, threads);

    std::cout << rd::HLINE << "\n";
    std::cout << "END!" << "\n";

    return EXIT_SUCCESS;
}

template <typename T>
void testCpuSimulation(rd::RDParams<T> &rdp, rd::RDSpiralParams<T> &sp, int threads) {

    rd::RidgeDetection<T> rd_cpu;
    rd_cpu.verbose_ = rdp.verbose;
    rd_cpu.order_ = rdp.order;
    rd_cpu.ompSetNumThreads(threads);

    rd::Samples<T> samples;
    rd::CpuTimer timer;
    rd::GraphDrawer<T> gDrawer;
    T *P;

    std::vector<std::string> samplesDir{"../../../examples/data/nd_segments/", "../../../examples/data/spirals/"};
    rd::Samples<T> samplesSet(rdp, sp, samplesDir, rdp.dim);
    P = samplesSet.samples_;

    std::ostringstream s;
    s << typeid(T).name() << "_initial_samples_set";
    gDrawer.showPoints(s.str(), P, rdp.np, rdp.dim);
    s.clear();
    s.str(std::string());

    T *S = rd::createTable<T>(rdp.np * rdp.dim, T(0));

    std::cout << rd::HLINE << "\n";
    std::cout << "CPU:" << "\n";

    timer.start();
    rd_cpu.ridgeDetection(P, rdp.np, S, rdp.r1, rdp.r2, rdp.dim);
    timer.stop();
    timer.elapsedMillis(0, true);

    s << typeid(T).name() << "_cpu_detected_ridge";
    gDrawer.showPoints(s.str(), S, rd_cpu.ns_, rdp.dim);

    if (rd_cpu.order_)
    {
        std::cout << "Coherent point chains:" << "\n";
        std::vector<T*> continuousMemChainsPoints;

        std::string plotcmd1 = "'-' w p pt ";
        std::string plotcmd2 = " lc rgb '#d64f4f' ps 1 ";
        std::string linesCmd = "'-' w l lt 2 lc rgb 'black' lw 2 ";

        int chainIdx = 0;
        s.clear();
        s.str(std::string());
        s << typeid(T).name() << "_detected_ridge_chains";
        gDrawer.startGraph(s.str(), rdp.dim);
        for (auto chain : rd_cpu.chainList_)
        {

            T *chainContMem = new T[chain.size()*rdp.dim];
            continuousMemChainsPoints.push_back(chainContMem);
            s.clear();
            s.str(std::string());
            s << plotcmd1 << (chainIdx % 15) + 1 << plotcmd2;
            gDrawer.addPlotCmd(s.str(), chainContMem, rd::GraphDrawer<T>::POINTS, chain.size());
            gDrawer.addPlotCmd(linesCmd, chainContMem, rd::GraphDrawer<T>::LINE, chain.size());

            std::cout << "============ chain: " << ++chainIdx << "===========" << "\n";
            std::cout.precision(4);
            for (auto ptr : chain)
            {
                for (size_t d = 0; d < rdp.dim; ++d)
                {
                    *chainContMem++ = ptr[d];
                    std::cout << std::right << std::fixed << std::setw(10) << ptr[d];
                }
                std::cout << "\n";
            }

        }
        gDrawer.endGraph();

        for (T *ptr : continuousMemChainsPoints)
            delete[] ptr;
    }
    delete[] S;
}
