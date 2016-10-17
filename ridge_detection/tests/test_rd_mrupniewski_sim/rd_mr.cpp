/**
 * @file rd_mr.cpp
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
#include "rd/utils/bounding_box.hpp"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"

#include "rd/cpu/mrupniewski/choose.hpp"
#include "rd/cpu/mrupniewski/evolve.hpp"
#include "rd/cpu/mrupniewski/decimate.hpp"
#include "rd/cpu/mrupniewski/order.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <chrono>
#include <cmath>

template <typename T> using Params = rd::RDParams<T>;

template <typename T>
void testCpuSimulation(Params<T> &rd_p,
                        rd::RDSpiralParams<T> &sp);

int main(int argc, char const **argv)
{
    Params<double> dParams;
    rd::RDSpiralParams<double> dSParams;
    Params<float> fParams;
    rd::RDSpiralParams<float> fSParams;

    dParams.dim = 2;
    dSParams.a = 4;
    dSParams.b = 0.5;
    dSParams.sigma = 4;

    fParams.dim = 2;
    fSParams.a = 4.0f;
    fSParams.b = 0.5f;
    fSParams.sigma = 4.0f;


    // Initialize command line
    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help") ||
             (args.ParsedArgc() < 5 )) {
        printf("%s \n"
            "\t\t--np=<P size>\n"
            "\t\t--r1=<r1 param>\n"
            "\t\t--r2=<r2 param>\n"
            "\t\t[--a=<spiral param>]\n"
            "\t\t[--b=<spiral param>]\n"
            "\t\t[--s=<spiral noise sigma>]\n"
            "\t\t[--dim=<data dimension>]\n"
            "\t\t[--file=<samples input file>]\n"
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
    if (args.CheckCmdLineFlag("file")) {
        args.GetCmdLineArgument("file",fSParams.file);
        args.GetCmdLineArgument("file",dSParams.file);      
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

    std::cout << "\n\n TEST FLOAT " << std::endl;
    testCpuSimulation(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    // std::cout << "\n\n TEST DOUBLE " << std::endl;
    // testCpuSimulation(dParams, dSParams);

    std::cout << rd::HLINE << std::endl;
    std::cout << "END!" << std::endl;

    return EXIT_SUCCESS;
}

template <typename T>
void testCpuSimulation(Params<T> &rdp,
                       rd::RDSpiralParams<T> &sp)
{

    rd::Samples<T> samples;
    rd::GraphDrawer<T> gDrawer;
    rd::CpuTimer rdTimer;

    T *samplesPtr;
    if (sp.loadFromFile) {
        samplesPtr = samples.loadFromFile(sp.file);
        rdp.dim = samples.dim_;
        rdp.np = samples.size_;
    } else {

        switch (rdp.dim) {
            case 2: samplesPtr = samples.genSpiral2D(
                        rdp.np, sp.a, sp.b, sp.sigma); break;
            case 3: samplesPtr = samples.genSpiral3D(
                        rdp.np, sp.a, sp.b, sp.sigma); break;
            default: samplesPtr = samples.genSegmentND(
                        rdp.np, rdp.dim, sp.sigma, T(100)); break;
        }
    }
    std::ostringstream s;
    s << typeid(T).name() << "_initial_samples_set";
    gDrawer.showPoints(s.str(), samplesPtr, rdp.np, rdp.dim);
    s.clear();
    s.str(std::string());

    rd::BoundingBox<T> bb(samplesPtr, rdp.np, rdp.dim);
    bb.calcDistances();
    T *chosenPtr = new T[bb.countSpheresInside(rdp.r1)];

    /************************************
     * ALGORITHM
     ************************************/
    std::cout << "start!" << std::endl;
    rdTimer.start();
	int chosenCnt = choose(samplesPtr, rdp.np, chosenPtr, rdp.r1,
                             rdp.dim, NULL);

    if (rdp.verbose)
    {
        std::cout << "Wybrano reprezentantów, liczność zbioru S: " 
            << chosenCnt << std::endl;
        std::cout << rd::HLINE << std::endl;

        std::ostringstream os;
        os << typeid(T).name();
        os << "_" << rdp.dim;
        os << "D_init_choosen_set";
        gDrawer.showPoints(os.str(), chosenPtr, chosenCnt, rdp.dim);
    }

	int oldCnt = 0;
	int iter = 0;
	while(oldCnt != chosenCnt) {
		oldCnt = chosenCnt;
		evolve(samplesPtr, rdp.np, rdp.r1, chosenPtr, chosenCnt, 
                rdp.dim, NULL);
        if (rdp.verbose)
        {
            std::cout << "Ewolucja nr: " << iter << std::endl;

            std::ostringstream os;
            os << typeid(T).name();
            os << "_" << rdp.dim;
            os << "D_iter_" << iter;
            os << "_a_evolution";
            if (rdp.np > 10000) {
                gDrawer.showCircles(os.str(), 0, rdp.np, chosenPtr, chosenCnt, rdp.r1, rdp.dim);
            } else {
                gDrawer.showCircles(os.str(), samplesPtr, rdp.np, chosenPtr, chosenCnt, rdp.r1, rdp.dim);
            }
        }
		decimate(chosenPtr, &chosenCnt, rdp.r2, rdp.dim, NULL);
        if (rdp.verbose)
        {
            std::cout << "Decymacja nr: " << iter << ", ns: " << chosenCnt << std::endl;

            std::ostringstream os;
            os << typeid(T).name();
            os << "_cpu_";
            os << rdp.dim;
            os << "D_iter_";
            os << iter;
            os << "_decimation";
            if (rdp.np > 10000) {
                gDrawer.showCircles(os.str(), 0, rdp.np, chosenPtr, chosenCnt, rdp.r2, rdp.dim);
            } else {
                gDrawer.showCircles(os.str(), samplesPtr, rdp.np, chosenPtr, chosenCnt, rdp.r2, rdp.dim);
            }
        }
		iter++;
	}

	// listaPunktow *lista = order(chosenPtr, chosenCnt, rdp.r2,
            //  rdp.dim, NULL); 
    rdTimer.stop();
    std::cout << "iter: " << iter << std::endl;

    /**
     *  TEST FLOAT 
     *  iter: 13
     *  ridge detection: 2.83838e+07ms =~ 7,884388889 h !!!
     */

     /**
      * a: 22.52 b: 11.31 s: 4.17 pointsNum: 1000000 r1: 7.089 r2: 14.178
      * 
      * 2D:
      * iter: 20
      * ridge detection:  37566284.000000ms 
      * And that is:  10h 26m 6s 284ms
      * 
      * 3D:
      * iter: 6
      * ridge detection:  13138943.000000ms 
      * And that is: 3h ...
      */

    std::cout.precision(6);
    std::cout << "ridge detection: " << std::fixed << std::right << std::setw(16)
        << rdTimer.elapsedMillis() << "ms \n";

    std::chrono::milliseconds durationMs((long int)(floorf(rdTimer.elapsedMillis())));
      
    std::chrono::hours durationHr;
    std::chrono::minutes durationM;
    long int minutes;
    std::chrono::seconds durationS;
    long int seconds;
    long int milliseconds;


    std::cout << "And that is: ";
    durationHr = std::chrono::duration_cast<std::chrono::hours>(durationMs);
    std::cout << durationHr.count() << "h ";

    durationM = std::chrono::duration_cast<std::chrono::minutes>(durationMs);
    minutes = durationM.count() - durationHr.count() * 60;
    std::cout << minutes << "m ";
    
    durationS = std::chrono::duration_cast<std::chrono::seconds>(durationMs);
    seconds = durationS.count() - durationM.count() * 60;
    std::cout << seconds << "s ";
    
    milliseconds = durationMs.count() - durationS.count() * 1000;
    std::cout << milliseconds << "ms " << std::endl;

    s << typeid(T).name() << "_chosen_samples";
    gDrawer.showPoints(s.str(), chosenPtr, chosenCnt, rdp.dim);

    delete[] chosenPtr;
}

