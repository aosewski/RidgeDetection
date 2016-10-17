/**
 * @file test_inner_decimate.cu
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

#include "../../../rd/gpu/brute_force/kernels/test/decimate2.cuh"
#include "../../../rd/gpu/brute_force/kernels/decimate.cuh"
#include "../../../rd/gpu/brute_force/rd_globals.cuh"
#include "../../../rd/gpu/samples_generator.cuh"

#include "../../../rd/utils/graph_drawer.hpp"
#include "../../../rd/utils/cmd_line_parser.hpp"
#include "../../../rd/utils/utilities.hpp"
#include "../../../cub/test_util.h"

#include "../../util_test_params.hpp"
#include "../../test_util.hpp"

#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#ifdef RD_PROFILE
#include <cuda_profiler_api.h>
#endif

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <cmath>

#define TEST_DIM 2

template <typename T>
void testChooseKernel(rd::RDParams<T> const &rdp,
                      rd::RDSpiralParams<T> const &rds);

int main(int argc, char *argv[])
{

    rd::RDParams<double> dParams;
    rd::RDSpiralParams<double> dSParams;
    rd::RDParams<float> fParams;
    rd::RDSpiralParams<float> fSParams;

    dSParams.a = 30;
    dSParams.b = 75;
    dSParams.sigma = 50;

    fSParams.a = 30.f;
    fSParams.b = 75.f;
    fSParams.sigma = 50.f;

    dParams.np = 50000;
    dParams.r1 = 65;
    dParams.r2 = 65;
    
    fParams.np = 50000;
    fParams.r1 = 65;
    fParams.r2 = 65;

    int devId = -1;

    //-----------------------------------------------------------------

    // Initialize command line
    GpuCommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help")) 
    {
        printf("%s \n"
            "\t\t[--np=<P size>]\n"
            "\t\t[--r1=<r1 param>]\n"
            "\t\t[--r2=<r2 param>]\n"
            "\t\t[--a=<spiral param>]\n"
            "\t\t[--b=<spiral param>]\n"
            "\t\t[--s=<spiral noise sigma>]\n"
            "\t\t[--d=<device id>]\n"
            "\n", argv[0]);
        exit(0);
    }

    args.GetCmdLineArgument("r1", dParams.r1);
    args.GetCmdLineArgument("r2", dParams.r2);

    args.GetCmdLineArgument("r1", fParams.r1);
    args.GetCmdLineArgument("r2", fParams.r2);


    args.GetCmdLineArgument("np", dParams.np);
    args.GetCmdLineArgument("np", fParams.np);

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
    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", devId);
    }

    fParams.dim = TEST_DIM;
    dParams.dim = TEST_DIM;

    args.DeviceInit(devId);

    std::cout << HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    testChooseKernel<float>(fParams, fSParams);
    std::cout << HLINE << std::endl;
    std::cout << "DOUBLE: " << std::endl;
    testChooseKernel<double>(dParams, dSParams);
    std::cout << HLINE << std::endl;

    args.DeviceReset();

    std::cout << "END!" << std::endl;
    return 0;
}

template <typename T>
void testChooseKernel(rd::RDParams<T> const &rdp,
                      rd::RDSpiralParams<T> const &sp)
{

    std::cout << "Samples: " << std::endl;
    std::cout <<  "\t dimension: " << rdp.dim << std::endl;
    std::cout <<  "\t n_samples: " << rdp.np << std::endl;
    std::cout <<  "\t r1: " << rdp.r1 << std::endl;
    std::cout <<  "\t r2: " << rdp.r2 << std::endl;

    std::cout << "Spiral params: " << std::endl;
    std::cout <<  "\t a: " << sp.a << std::endl;
    std::cout <<  "\t b: " << sp.b << std::endl;
    std::cout <<  "\t sigma: " << sp.sigma << std::endl;


    rd::GraphDrawer<T> gDrawer;
    GpuTimer kernelTimer, memcpyTimer;

    T *d_S, *d_P;
    int *d_ns, h_ns;
    T *h_P, *h_S, *h_S2;

    const int INITIAL_S_SIZE = 0.1 * rdp.np;

    checkCudaErrors(cudaMalloc((void**)&d_S, INITIAL_S_SIZE * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_P, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_S, 0, INITIAL_S_SIZE * TEST_DIM * sizeof(T)));

    h_S     = createTable<T>(INITIAL_S_SIZE * TEST_DIM, 0);
    h_S2    = createTable<T>(INITIAL_S_SIZE * TEST_DIM, 0);
    h_P     = createTable<T>(rdp.np * TEST_DIM, 0);

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rdBruteForceNs));

    switch(TEST_DIM)
    {
        case 2:
            rd::SamplesGenerator<T>::template spiral2D<rd::ROW_MAJOR>(
                rdp.np, sp.a, sp.b, sp.sigma, d_P);
            break;
        case 3:
            rd::SamplesGenerator<T>::template spiral3D<rd::ROW_MAJOR>(
                rdp.np, sp.a, sp.b, sp.sigma, d_P);
            break;
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_P, d_P, rdp.np * TEST_DIM * sizeof(T),
        cudaMemcpyDeviceToHost));

    for (int i = 0, index = 0; i < INITIAL_S_SIZE; ++i)
    {
        index = getRandIndex(INITIAL_S_SIZE);
        for (int d = 0; d < TEST_DIM; ++d)
            h_S[i*TEST_DIM + d] = h_P[index * TEST_DIM + d];
    }

    std::ostringstream os;
    os << typeid(T).name() << "_" << TEST_DIM;
    os << "D_initial_chosen_set_";
    gDrawer.startGraph(os.str(), rdp.dim);
    gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
         h_P, rd::GraphDrawer<T>::POINTS, rdp.np);
    gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
         h_S, rd::GraphDrawer<T>::POINTS, INITIAL_S_SIZE);
    gDrawer.endGraph();
    os.clear();
    os.str(std::string());

    checkCudaErrors(cudaMemcpy(d_S, h_S, INITIAL_S_SIZE * TEST_DIM * sizeof(T), 
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    //---------------------------------------------------

    std::cout << HLINE << std::endl;
    std::cout << "Test inner_decimate kernel:" << std::endl;

    const int BLOCK_DIM_X   = 128;
    const int GRID_DIM_X    = 1;
    #if defined(DEBUG) || defined(RD_PROFILE)
    const int ITER          = 1;
    #else
    const int ITER          = 100;
    #endif
    const int VALS_PER_THREAD = 2;

    dim3 blockDim(BLOCK_DIM_X);
    dim3 gridDim(GRID_DIM_X);

    #ifndef DEBUG
    // warm-up
    checkCudaErrors(cudaMemcpyToSymbol(rdBruteForceNs, &INITIAL_S_SIZE, sizeof(int)));
    __inner_decimate<T, BLOCK_DIM_X, VALS_PER_THREAD, TEST_DIM>
        <<<gridDim, blockDim>>>(d_S, d_ns, rdp.r2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    #endif

    memcpyTimer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rdBruteForceNs, &INITIAL_S_SIZE, sizeof(int)));
        checkCudaErrors(cudaMemcpy(d_S, h_S, INITIAL_S_SIZE * TEST_DIM * sizeof(T), 
            cudaMemcpyHostToDevice));
    }
    memcpyTimer.Stop();
    float memcpyETime = memcpyTimer.ElapsedMillis();
    checkCudaErrors(cudaMemset(d_S, 0, INITIAL_S_SIZE * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaDeviceSynchronize());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    kernelTimer.Start();
    for (int i = 0; i < ITER; ++i) 
    {
        checkCudaErrors(cudaMemcpyToSymbol(rdBruteForceNs, &INITIAL_S_SIZE, sizeof(int)));
        checkCudaErrors(cudaMemcpy(d_S, h_S, INITIAL_S_SIZE * TEST_DIM * sizeof(T), 
            cudaMemcpyHostToDevice));
        __inner_decimate<T, BLOCK_DIM_X, VALS_PER_THREAD, TEST_DIM>
            <<<gridDim, blockDim>>>(d_S, d_ns, rdp.r2);
        checkCudaErrors(cudaGetLastError());
    }
    kernelTimer.Stop();
    float etime = kernelTimer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    // std::cout << "Avg gpu time: " << etime / ITER << "ms " << std::endl;
    std::cout << "Avg gpu time (v2): " << (etime - memcpyETime) / ITER << "ms " << std::endl;

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Chosen count (v2): " << h_ns << std::endl;

    checkCudaErrors(cudaMemcpy(h_S, d_S, h_ns * TEST_DIM * sizeof(T), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    int notNaNCount = 0;
    for (int i = 0; i < h_ns; ++i)
    {
        if (!std::isnan(h_S[i*TEST_DIM]))
        {
            std::cout << "h_S["<<i*TEST_DIM<<"]: " << h_S[i*TEST_DIM] << std::endl;
            notNaNCount++;
            for (int d = 0; d < TEST_DIM; ++d)
                h_S2[notNaNCount * TEST_DIM + d] = h_S[i * TEST_DIM + d];
        }
    }

    std::cout << "notNaNCount count (v2): " << notNaNCount << std::endl;

    os << typeid(T).name() << "_" << TEST_DIM << "D_gpu_chosen_setV2";
    gDrawer.startGraph(os.str(), rdp.dim);
    gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
         h_P, rd::GraphDrawer<T>::POINTS, rdp.np);
    gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
         // h_S2, rd::GraphDrawer<T>::POINTS, h_ns);
         h_S2, rd::GraphDrawer<T>::POINTS, notNaNCount);
    gDrawer.endGraph();
    os.clear();
    os.str(std::string());

    delete[] h_P;
    delete[] h_S;
    delete[] h_S2;

    checkCudaErrors(cudaFree(d_S));
    checkCudaErrors(cudaFree(d_P));
}