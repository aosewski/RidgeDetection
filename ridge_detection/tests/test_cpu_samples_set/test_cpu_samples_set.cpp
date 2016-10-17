/**
 * @file test_cpu_samples_set.cpp
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

#include "ut_cpu_samples_set.hpp"
#include "rd/utils/rd_params.hpp"

#include "rd/utils/cmd_line_parser.hpp"

#include <cstdlib>
#include <iostream>
#include <cmath>

template <typename T>
void testCpuSamplesSet(rd::RDParams<T> &rdp, rd::RDSpiralParams<T> &sp);

int main(int argc, char* argv[]) {

	rd::RDParams<double> dParams;
	rd::RDSpiralParams<double> dSpiralParams;

	// spiral params
	dSpiralParams.sigma = 2;
	dSpiralParams.a = 10;
    dSpiralParams.b = 0.2;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help") || args.ParsedArgc() < 1) {
        printf("%s \n"
            "\t\t--n=<samples number>\n"
        	"\t\t[--a=<spiral param a>]\n"
        	"\t\t[--b=<spiral param b>]\n"
            "\t\t[--s=<spiral sigma>]\n"
            "\n", argv[0]);
        exit(0);
    }

    args.GetCmdLineArgument("n", dParams.np);

    if (args.CheckCmdLineFlag("a")) {
		args.GetCmdLineArgument("a", dSpiralParams.a);
    }

    if (args.CheckCmdLineFlag("b")) {
    	args.GetCmdLineArgument("b", dSpiralParams.b);
    }

    if (args.CheckCmdLineFlag("s")) {
        args.GetCmdLineArgument("s", dSpiralParams.sigma);
    }

    testCpuSamplesSet(dParams, dSpiralParams);


	std::cout << HLINE << std::endl;
	std::cout << "END!" << std::endl;

	return EXIT_SUCCESS;
}

template <typename T>
void testCpuSamplesSet(rd::RDParams<T> &rdp, rd::RDSpiralParams<T> &sp) {

	CpuSamplesSetUnitTests<T> test(rdp.np, sp.a, sp.b, sp.sigma);

	test.testGenSpiral2D();
	test.testGenSpiral3D();
	test.testGenSegmentND(13);

}
