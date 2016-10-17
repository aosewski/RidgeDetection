/**
 * @file test_choose_v2.cu
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

#include "rd/gpu/device/brute_force/choose2.cuh"
#include "rd/gpu/device/brute_force/choose.cuh"
#include "rd/gpu/device/device_decimate.cuh"
#include "rd/gpu/device/brute_force/rd_globals.cuh"
#include "rd/gpu/device/samples_generator.cuh"

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/rd_params.hpp"

#include "cub/test_util.h"


#include <helper_cuda.h>

#ifdef RD_PROFILE
#include <cuda_profiler_api.h>
#endif

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>

#define TEST_DIM 2

template <typename T>
void testChooseKernel(rd::RDParams<T> const &rdp,
                      rd::RDSpiralParams<T> const &rds);

int main(int argc, char const **argv)
{
    rd::RDParams<float> fParams;
    rd::RDSpiralParams<float> fSParams;

    //-----------------------------------------------------------------

    // Initialize command line
    rd::CommandLineArgs args(argc, argv);
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

    args.GetCmdLineArgument("r1", fParams.r1);
    args.GetCmdLineArgument("r2", fParams.r2);


    args.GetCmdLineArgument("np", fParams.np);

    if (args.CheckCmdLineFlag("a")) 
    {
        args.GetCmdLineArgument("a", fSParams.a);
    }
    if (args.CheckCmdLineFlag("b")) 
    {
        args.GetCmdLineArgument("b", fSParams.b);
    }
    if (args.CheckCmdLineFlag("s")) 
    {
        args.GetCmdLineArgument("s", fSParams.sigma);
    }
    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", fParams.devId);
    }
    
    fParams.dim = TEST_DIM;

    checkCudaErrors(deviceInit(fParams.devId));

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    testChooseKernel<float>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;

    checkCudaErrors(deviceReset());

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
    GpuTimer kernelTimer, memcpyTimer, kernelTimerV1;

    T *d_P, *d_S, *d_Sv1;
    int *d_ns, h_ns, h_nsV1;
    T *h_P, *h_S, *h_Sv1;

    checkCudaErrors(cudaMalloc((void**)&d_P, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_S, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_Sv1, rdp.np * TEST_DIM * sizeof(T)));

    checkCudaErrors(cudaMemset(d_P, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_Sv1, 0, rdp.np * TEST_DIM * sizeof(T)));

    h_P = new T[rdp.np * TEST_DIM];
    h_S = new T[rdp.np * TEST_DIM];
    h_Sv1 = new T[rdp.np * TEST_DIM];

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));

    switch(TEST_DIM)
    {
        case 2:
            rd::gpu::SamplesGenerator<T>::template spiral2D<rd::COL_MAJOR>(
                rdp.np, sp.a, sp.b, sp.sigma, d_P);
            break;
        case 3:
            rd::gpu::SamplesGenerator<T>::template spiral3D<rd::COL_MAJOR>(
                rdp.np, sp.a, sp.b, sp.sigma, d_P);
            break;
        default:
            throw std::invalid_argument("Unsupported dimension!");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_P, d_P, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::transposeInPlace(h_P, h_P + rdp.np * TEST_DIM, rdp.np);

    std::ostringstream os;
    os << typeid(T).name() << "_" << TEST_DIM;
    os << "D_initial_samples_set_";
    gDrawer.showPoints(os.str(), h_P, rdp.np, TEST_DIM);
    os.clear();
    os.str(std::string());

    //---------------------------------------------------

    std::cout << rd::HLINE << std::endl;
    std::cout << "Test choosev2 kernel:" << std::endl;

    const int BLOCK_DIM_X       = 64;
    const int GRID_DIM_X        = 64;
    const int TILE              = 64;
    const int VAL_PER_THREAD    = 4;
    #if defined(DEBUG) || defined(RD_PROFILE)
    const int ITER          = 1;
    #else
    const int ITER          = 100;
    #endif
    const int ZERO          = 0;

    dim3 blockDim(BLOCK_DIM_X);
    dim3 gridDim(GRID_DIM_X);


    #ifndef DEBUG
    // warm-up
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    __choose_kernel_v3<T, VAL_PER_THREAD, TILE, BLOCK_DIM_X, TEST_DIM><<<gridDim, blockDim>>>(
        d_P, d_S, rdp.np, rdp.r1, d_ns);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    #endif

    memcpyTimer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    }
    memcpyTimer.Stop();
    float memcpyETime = memcpyTimer.ElapsedMillis();
    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaDeviceSynchronize());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    kernelTimer.Start();
    for (int i = 0; i < ITER; ++i) 
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
        __choose_kernel_v3<T, VAL_PER_THREAD, TILE, BLOCK_DIM_X, TEST_DIM><<<gridDim, blockDim>>>(
            d_P, d_S, rdp.np, rdp.r1, d_ns);
        rd::gpu::bruteForce::DeviceDecimate::decimate<TEST_DIM, rd::ROW_MAJOR>(d_S, d_ns, rdp.r2);
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

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Chosen count (v2): " << h_ns << std::endl;

    checkCudaErrors(cudaMemcpy(h_S, d_S, h_ns * TEST_DIM * sizeof(T), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    // -----------------------------------------------------------------------

    #if !defined(DEBUG) && !defined(RD_PROFILE)

    std::cout << rd::HLINE << std::endl;
    std::cout << "Compare to choosev1 kernel:" << std::endl;

    #ifndef DEBUG
    // warm-up
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    __choose_kernel_v1<T, 320><<<1, 320>>>(
        d_P, d_Sv1, rdp.np, rdp.r1, d_ns, TEST_DIM);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    #endif

    kernelTimer.Start();
    for (int i = 0; i < ITER; ++i) 
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
        __choose_kernel_v1<T, 320><<<1, 320>>>(
            d_P, d_Sv1, rdp.np, rdp.r1, d_ns, TEST_DIM);
        checkCudaErrors(cudaGetLastError());
    }
    kernelTimer.Stop();
    etime = kernelTimer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    // std::cout << "Avg gpu time: " << etime / ITER << "ms " << std::endl;
    std::cout << "Avg gpu time (v1): " << (etime - memcpyETime) / ITER << "ms " << std::endl;

    checkCudaErrors(cudaMemcpyFromSymbol(&h_nsV1, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Chosen count (v1): " << h_nsV1 << std::endl;

    checkCudaErrors(cudaMemcpy(h_Sv1, d_Sv1, h_nsV1 * TEST_DIM * sizeof(T), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    #endif
    // -----------------------------------------------------------------------

    os << typeid(T).name() << "_" << TEST_DIM << "D_gpu_chosen_setV2";
    gDrawer.startGraph(os.str(), rdp.dim);
    gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
         h_P, rd::GraphDrawer<T>::POINTS, rdp.np);
    gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
         h_S, rd::GraphDrawer<T>::POINTS, h_ns);
    gDrawer.endGraph();
    os.clear();
    os.str(std::string());

    #if !defined(DEBUG) && !defined(RD_PROFILE)
    os << typeid(T).name() << "_" << TEST_DIM << "D_gpu_chosen_setV1";
    gDrawer.startGraph(os.str(), rdp.dim);
    gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
         h_P, rd::GraphDrawer<T>::POINTS, rdp.np);
    gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
         h_Sv1, rd::GraphDrawer<T>::POINTS, h_nsV1);
    gDrawer.endGraph();
    os.clear();
    os.str(std::string());

    #endif

    delete[] h_P;
    delete[] h_S;
    delete[] h_Sv1;

    checkCudaErrors(cudaFree(d_P));
    checkCudaErrors(cudaFree(d_S));
    checkCudaErrors(cudaFree(d_Sv1));

}
