/**
 * @file benchmark_count_neighbours.cu
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

#include "rd/cpu/brute_force/ridge_detection.hpp"
#include "rd/gpu/device/brute_force/rd_globals.cuh"
#include "rd/gpu/device/samples_generator.cuh"
#include "rd/gpu/util/data_order_traits.hpp"
#include "rd/gpu/cta_count_neighbour_points.cuh"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/rd_samples.cuh"
#include "cub/test_util.h"
#include "cub/util_device.cuh"

#include "rd/utils/rd_params.hpp"

#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda_profiler_api.h>

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <string>

static const int TEST_DIM = 3;
static const std::string LOG_FILE_NAME_SUFFIX = "_neighbours-timings.txt";

#ifdef RD_PROFILE
const int ITER = 1;
#else
const int ITER = 100;
#endif


template <typename T>
void testCountNeighboursKernel(rd::RDParams<T> &rdp,
                      rd::RDSpiralParams<T> const &rds);

int main(int argc, char const **argv)
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

    //-----------------------------------------------------------------

    // Initialize command line
    CommandLineArgs args(argc, argv);
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
            "\t\t[--dim=<dim>]\n"
            "\t\t[--v <verbose>]\n"
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
        args.GetCmdLineArgument("d", fParams.devId);
        args.GetCmdLineArgument("d", dParams.devId);
    }
    if (args.CheckCmdLineFlag("dim")) 
    {
        args.GetCmdLineArgument("dim", fParams.dim);
        args.GetCmdLineArgument("dim", dParams.dim);
    }
    if (args.CheckCmdLineFlag("v")) 
    {
        fParams.verbose = true;
        dParams.verbose = true;
    }

    deviceInit(fParams.devId);

    std::cout << HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    testCountNeighboursKernel<float>(fParams, fSParams);
    std::cout << HLINE << std::endl;
    std::cout << "DOUBLE: " << std::endl;
    testCountNeighboursKernel<double>(dParams, dSParams);
    std::cout << HLINE << std::endl;

    deviceReset();

    std::cout << "END!" << std::endl;
    return 0;
}

template <typename T>
bool countNeighboursGold(
    rd::RDParams<T> &rdp,
    T *P,
    T *S,
    int treshold)
{
    rd::RidgeDetection<T> rd(TEST_DIM, rdp.np, rdp.r1, rdp.r2, P, S);
    rd.choose();

    rdp.ns = rd.ns_;
    std::cout << "Chosen count: " << rdp.ns << std::endl;

    return countNeighbouringPoints(S, rdp.ns, S,
         TEST_DIM, rdp.r1 * rdp.r1, treshold);
}

struct KernelConf
{
    float time;
    int blockSize;

    KernelConf() : time(std::numeric_limits<float>::max()),
        blockSize(0)
    {
    }
};

template <
    int         BLOCK_SIZE,
    typename    T>
__global__ void __dispatch_count_neighbours_row_major(
        T const *  points,
        int        np,
        T const *  srcP,
        int        dim,
        T          r2,
        int        threshold,
        int *      result)
{
    int res = ctaCountNeighbouringPoints<T, BLOCK_SIZE>(points, np, srcP, dim, r2,
                 threshold, rd::rowMajorOrderTag());
    if (threadIdx.x == 0)
        *result = res;

}

template<
    int         BLOCK_SIZE, 
    typename    T>
float dispatchCountNeighboursRowMajorOrder(
    rd::RDParams<T> const & rdp,
    T const *               d_S,
    int *                   d_result,
    int                     NEIGHBOURS_THRESHOLD,
    std::ofstream *         logFile = nullptr)
{

    GpuTimer timer;
    float kernelTime;

    // warm-up
    __dispatch_count_neighbours_row_major<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_S, rdp.ns, d_S, TEST_DIM,
         rdp.r1 * rdp.r1, NEIGHBOURS_THRESHOLD, d_result);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        __dispatch_count_neighbours_row_major<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_S, rdp.ns, d_S, TEST_DIM,
            rdp.r1 * rdp.r1, NEIGHBOURS_THRESHOLD, d_result);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    kernelTime = kernelTime  / static_cast<float>(ITER);
    if (rdp.verbose)
    {
        *logFile << " " << BLOCK_SIZE << " " << kernelTime << std::endl;
    }

    return kernelTime;
}

template <typename T>
void benchmarkCountNeighboursRowMajorOrder(
    rd::RDParams<T> &   rdp,
    T const *           d_S,
    int                 NEIGHBOURS_THRESHOLD,
    std::ofstream *     logFile = nullptr)
{

    std::cout << HLINE << std::endl;
    std::cout << "benchmarkCountNeighboursRowMajorOrder:" << std::endl;

    if (rdp.verbose)
        *logFile << "%---------------benchmarkCountNeighboursRowMajorOrder---------------" << std::endl;

    int *d_result;
    checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(int)));

    KernelConf bestConf;

    auto checkBestConf = [&](float kernelTime, int BLOCK_SIZE) {
        if (kernelTime < bestConf.time)
        {
            bestConf.time = kernelTime;
            bestConf.blockSize = BLOCK_SIZE;
        }
    };

    #if !defined(RD_PROFILE)
    checkBestConf(dispatchCountNeighboursRowMajorOrder<  64>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile),  64);
    checkBestConf(dispatchCountNeighboursRowMajorOrder<  96>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile),  96);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 128>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 128);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 160>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 160);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 192>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 192);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 224>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 224);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 256>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 256);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 288>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 288);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 320>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 320);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 352>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 352);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 384>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 384);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 416>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 416);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 448>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 448);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 480>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 480);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 512>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 512);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 544>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 544);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 576>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 576);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 608>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 608);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 640>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 640);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 672>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 672);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 704>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 704);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 736>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 736);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 768>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 768);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 800>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 800);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 832>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 832);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 864>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 864);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 896>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 896);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 928>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 928);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 960>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 960);
    checkBestConf(dispatchCountNeighboursRowMajorOrder< 992>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 992);
    #endif
    checkBestConf(dispatchCountNeighboursRowMajorOrder<1024>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile),1024);


    std::cout << "best conf: \n " << bestConf.blockSize << " " << bestConf.time << std::endl;
    if (rdp.verbose)
    {
        *logFile << "%----------BEST CONF---------" << std::endl;
        *logFile << "% " << bestConf.blockSize << " " << bestConf.time << std::endl;
    }

    checkCudaErrors(cudaFree(d_result));
}


template <
    int         DIM,
    int         BLOCK_SIZE,
    typename    T>
__global__ void __dispatch_count_neighbours_row_major_v2(
        T const *  points,
        int        np,
        T const *  srcP,
        T          r2,
        int        threshold,
        int *      result)
{
    int res = ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(points, np, srcP, r2,
                 threshold, rd::rowMajorOrderTag());
    if (threadIdx.x == 0)
    {
        *result = (res >= threshold) ? 1 : 0;
    }
}


template<
    int         BLOCK_SIZE, 
    typename    T>
float dispatchCountNeighboursRowMajorOrder_v2(
    rd::RDParams<T> const & rdp,
    T const *               d_S,
    int *                   d_result,
    int                     NEIGHBOURS_THRESHOLD,
    std::ofstream *         logFile = nullptr)
{

    GpuTimer timer;
    float kernelTime;

    // warm-up
    __dispatch_count_neighbours_row_major_v2<TEST_DIM, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_S, rdp.ns, d_S,
         rdp.r1 * rdp.r1, NEIGHBOURS_THRESHOLD, d_result);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        __dispatch_count_neighbours_row_major_v2<TEST_DIM, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_S, rdp.ns, d_S,
            rdp.r1 * rdp.r1, NEIGHBOURS_THRESHOLD, d_result);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    kernelTime = kernelTime  / static_cast<float>(ITER);
    if (rdp.verbose)
    {
        *logFile << " " << BLOCK_SIZE << " " << kernelTime << std::endl;
    }

    return kernelTime;
}


template <typename T>
void benchmarkCountNeighboursRowMajorOrder_v2(
    rd::RDParams<T> &   rdp,
    T const *           d_S,
    int                 NEIGHBOURS_THRESHOLD, 
    std::ofstream *     logFile = nullptr)
{

    std::cout << HLINE << std::endl;
    std::cout << "benchmarkCountNeighboursRowMajorOrder_v2:" << std::endl;

    if (rdp.verbose)
        *logFile << "%---------------benchmarkCountNeighboursRowMajorOrder_v2---------------" << std::endl;

    int *d_result;
    checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(int)));

    KernelConf bestConf;

    auto checkBestConf = [&](float kernelTime, int BLOCK_SIZE) {
        if (kernelTime < bestConf.time)
        {
            bestConf.time = kernelTime;
            bestConf.blockSize = BLOCK_SIZE;
        }
    };

    #if !defined(RD_PROFILE)
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2<  64>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile),  64);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2<  96>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile),  96);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 128>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 128);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 160>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 160);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 192>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 192);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 224>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 224);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 256>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 256);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 288>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 288);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 320>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 320);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 352>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 352);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 384>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 384);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 416>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 416);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 448>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 448);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 480>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 480);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 512>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 512);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 544>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 544);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 576>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 576);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 608>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 608);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 640>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 640);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 672>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 672);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 704>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 704);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 736>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 736);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 768>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 768);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 800>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 800);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 832>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 832);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 864>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 864);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 896>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 896);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 928>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 928);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 960>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 960);
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2< 992>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile), 992);
    #endif
    checkBestConf(dispatchCountNeighboursRowMajorOrder_v2<1024>(rdp, d_S, d_result, NEIGHBOURS_THRESHOLD, logFile),1024);


    std::cout << "best conf: \n " << bestConf.blockSize << " " << bestConf.time << std::endl;
    if (rdp.verbose)
    {
        *logFile << "%----------BEST CONF---------" << std::endl;
        *logFile << "% " << bestConf.blockSize << " " << bestConf.time << std::endl;
    }

    checkCudaErrors(cudaFree(d_result));
}


template <typename T>
void testCountNeighboursKernel(rd::RDParams<T> &rdp,
                      rd::RDSpiralParams<T> const &sp)
{

    std::cout << "Samples: " << std::endl;
    std::cout <<  "\t dimension: " << TEST_DIM << std::endl;
    std::cout <<  "\t n_samples: " << rdp.np << std::endl;
    std::cout <<  "\t r1: " << rdp.r1 << std::endl;
    std::cout <<  "\t r2: " << rdp.r2 << std::endl;

    std::cout << "Spiral params: " << std::endl;
    std::cout <<  "\t a: " << sp.a << std::endl;
    std::cout <<  "\t b: " << sp.b << std::endl;
    std::cout <<  "\t sigma: " << sp.sigma << std::endl; 

    const int NEIGHBOURS_THRESHOLD = 4;

    T *d_P, *d_S;
    T *h_P, *h_S;

    checkCudaErrors(cudaMalloc((void**)&d_P, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_S, rdp.np * TEST_DIM * sizeof(T)));

    checkCudaErrors(cudaMemset(d_P, 0, rdp.np * TEST_DIM * sizeof(T)));

    h_P = new T[rdp.np * TEST_DIM];
    h_S = new T[rdp.np * TEST_DIM];

    switch(TEST_DIM)
    {
        case 2:
            rd::SamplesGenerator<T>::template spiral2D<rd::COL_MAJOR>(
                rdp.np, sp.a, sp.b, sp.sigma, d_P);
            break;
        case 3:
            rd::SamplesGenerator<T>::template spiral3D<rd::COL_MAJOR>(
                rdp.np, sp.a, sp.b, sp.sigma, d_P);
            break;
        default:
            throw std::logic_error("Not supported dimension!");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_P, d_P, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    transposeInPlace(h_P, h_P + rdp.np * TEST_DIM, rdp.np);

    //---------------------------------------------------
    //               REFERENCE COUNT_NEIGHBOURS 
    //---------------------------------------------------

    countNeighboursGold(rdp, h_P, h_S, NEIGHBOURS_THRESHOLD);

    // get chosen samples to device memory properly ordered
    checkCudaErrors(cudaMemcpy(d_S, h_S, rdp.ns * TEST_DIM * sizeof(T), cudaMemcpyHostToDevice));

    //-------------------------------------------------------
    
    std::ofstream *logFile = nullptr;
    rdp.devId = (rdp.devId != -1) ? rdp.devId : 0;

    if (rdp.verbose)
    {
        cudaDeviceProp devProp;
        checkCudaErrors(cudaGetDeviceProperties(&devProp, rdp.devId));

        std::ostringstream logFileName;
        logFileName << devProp.name << "_" << std::to_string(TEST_DIM) <<
             "D" << LOG_FILE_NAME_SUFFIX;

        std::string logFilePath = findPath("", logFileName.str());
        logFile = new std::ofstream(logFilePath.c_str(), std::ios::out | std::ios::app);
        if (logFile->fail())
        {
            throw std::logic_error("Couldn't open file: " + logFileName.str());
        }

        *logFile << "%" << HLINE << std::endl;
        *logFile << "% " << typeid(T).name() << std::endl;
    }


    //---------------------------------------------------
    //               GPU COUNT_NEIGHBOURS 
    //---------------------------------------------------
    // int smVersion;
    // checkCudaErrors(cub::SmVersion(smVersion, rdp.devId));

    benchmarkCountNeighboursRowMajorOrder(rdp, d_S, NEIGHBOURS_THRESHOLD, logFile);
    benchmarkCountNeighboursRowMajorOrder_v2(rdp, d_S, NEIGHBOURS_THRESHOLD, logFile);

    // clean-up
    
    if (rdp.verbose)
    {
        logFile->close();
        delete logFile;
    }

    delete[] h_P;
    delete[] h_S;

    checkCudaErrors(cudaFree(d_P));
    checkCudaErrors(cudaFree(d_S));
}

