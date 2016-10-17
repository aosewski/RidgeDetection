/**
 *	@file visualisation.cu
 *	@author Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of 
 *  estimation of multidimensional random variable density function ridge
 *  detection algorithm.",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 * 
 * ICCE Faculty of Electronics and Information Technology
 * Warsaw University of Technology 2016
 */

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h> // includes cuda_gl_interop.h

#include "../../../cub/test_util.h"
#include "../../util_test_params.hpp"
#include "../../test_util.hpp"
#include "../../../rd/gpu/version.h"
#include "../../../rd/utils/flags.h"
#include "../../../rd/utils/cmd_line_parser.hpp"

#include <iostream>

//---------------------------------------------------



#define TEST_DIMENSION 2


//---------------------------------------------------

int main(int argc, char* argv[]) {

	rd::RDParams<double> dParams;
	rd::RDSpiralParams<double> dSParams;
	rd::RDParams<float> fParams;
	rd::RDSpiralParams<float> fSParams;

	dSParams.a = 0.7;
	dSParams.b = 0.3;
	dSParams.sigma = 0.4;

	fSParams.a = 0.7f;
	fSParams.b = 0.3f;
	fSParams.sigma = 0.4f;

	int devId = 0;
	int cpuThreads = 1;
	int version = RD_BRUTE_FORCE_BEST_VERSION;

	//-----------------------------------------------------------------

    // Initialize command line
    CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help") || (args.ParsedArgc() < 3 )) {
        printf("%s \n"
            "\t\t--np=<P size>\n"
        	"\t\t--r1=<r1 param>\n"
        	"\t\t--r2=<r2 param>\n"
        	"\t\t[--a=<spiral param>]\n"
        	"\t\t[--b=<spiral param>]\n"
        	"\t\t[--s=<spiral noise sigma>]\n"
        	"\t\t[--d=<device id>]\n"
        	"\t\t[--t=<CPU threads>]\n"
            "\t\t[--v] \n"
        	"\t\t[--ver=<version>\n"
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

	args.GetCmdLineArgument("np", dParams.np);
	args.GetCmdLineArgument("np", fParams.np);

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
	if (args.CheckCmdLineFlag("d")) {
		args.GetCmdLineArgument("d", devId);
	}

	if (args.CheckCmdLineFlag("t")) {
		args.GetCmdLineArgument("t", cpuThreads);
	}

	if (args.CheckCmdLineFlag("ver")) {
		args.GetCmdLineArgument("ver", version);
	}

	fParams.dim = TEST_DIMENSION;
	dParams.dim = TEST_DIMENSION;
	fParams.version = version;
	dParams.version = version;

	checkCudaErrors(deviceInit(devId));	

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

}