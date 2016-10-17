/**
 *	@file test_gpu_samples_set.cpp
 *	@author Adam Rogowiec
 */

#include "ut_gpu_samples_set.cuh"
#include "rd/utils/rd_params.hpp"

#include "cub/test_util.h"
#include "rd/utils/cmd_line_parser.hpp"

#include <helper_cuda.h>
#include <cstdlib>
#include <iostream>

template <typename T>
void testGpuSamplesSet(rd::RDParams<T> &rd_p, rd::RDSpiralParams<T> &sp);

int main(int argc, char const **argv) {

	rd::RDParams<double> dp;
	rd::RDSpiralParams<double> dsp;
	int devId = 0;

	// spiral params
	dsp.sigma = 2;
	dsp.a = 10;
	dsp.b = 0.2;

    // Initialize command line
    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help") || args.ParsedArgc() < 1) {
        printf("%s \n"
            "\t\t--n=<samples number>\n"
        	"\t\t[--a=<spiral param a>]\n"
        	"\t\t[--b=<spiral param b>]\n"
        	"\t\t[--s=<spiral sigma>]\n"
        	"\t\t[--device=<device id>]\n"
            "\t\t[--v] \n"
            "\n", argv[0]);
        exit(0);
    }

    args.GetCmdLineArgument("n", dp.np);

    if (args.CheckCmdLineFlag("a")) {
		args.GetCmdLineArgument("a", dsp.a);
    }
    if (args.CheckCmdLineFlag("b")) {
    	args.GetCmdLineArgument("b", dsp.b);
    }
    if (args.CheckCmdLineFlag("s")) {
    	args.GetCmdLineArgument("s", dsp.sigma);
    }
    if (args.CheckCmdLineFlag("device")) {
    	args.GetCmdLineArgument("device", devId);
    }
	checkCudaErrors(deviceInit(devId));

    testGpuSamplesSet(dp, dsp);

	std::cout << rd::HLINE << std::endl;
	std::cout << "END!" << std::endl;

	checkCudaErrors(deviceReset());

	return EXIT_SUCCESS;
}

template <typename T>
void testGpuSamplesSet(rd::RDParams<T> &rd_p, rd::RDSpiralParams<T> &sp) {

	GpuSamplesSetUnitTests<T> test(rd_p.np, sp.a, sp.b, sp.sigma);

	test.testGenSpiral2D();
	test.testGenSpiral3D();
	test.testGenSegmentND(13);

}
