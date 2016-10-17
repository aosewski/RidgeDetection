/**
 * @file benchmark_decimate.cu
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

#include <helper_cuda.h>
#ifdef RD_PROFILE
#include <cuda_profiler_api.h>
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <stdexcept>
#include <string>

#include "rd/gpu/device/brute_force/rd_globals.cuh"
#include "rd/gpu/device/brute_force/decimate.cuh"
#include "rd/gpu/device/device_choose.cuh"
#include "rd/gpu/device/samples_generator.cuh"
#include "rd/gpu/util/dev_memcpy.cuh"

#include "rd/gpu/device/brute_force/test/decimate_nan_marker1.cuh"
#include "rd/gpu/device/brute_force/decimate_dist_mtx.cuh"

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/memory.h"
#include "rd/utils/name_traits.hpp"
#include "rd/utils/rd_params.hpp"
#include "tests/test_util.hpp"

#include "cub/test_util.h"


static const std::string LOG_FILE_NAME_SUFFIX = "decimate-perf.txt";

static std::ofstream   *g_logFile = nullptr;
static std::string      g_devName;
static int              g_devId = 0;
static bool             g_logPerfResults = false;
static bool             g_startBenchmark = false;

#if defined(RD_PROFILE) || defined(RD_DEBUG)
const int ITER = 1;
#else
const int ITER = 100;
#endif


template <int TEST_DIM, typename T>
void benchmarkDecimateKernel(rd::RDParams<T> &rdp,
                      rd::RDSpiralParams<T> const &rds);


template <typename T>
static void initializeLogFile()
{
    if (g_logPerfResults)
    {
        std::ostringstream logFileName;
        logFileName << typeid(T).name() << "_" <<  getCurrDate() << "_" 
                    << g_devName << "_" << getBinConfSuffix() << LOG_FILE_NAME_SUFFIX;

        std::string logFilePath = rd::findPath("../timings/", logFileName.str());
        g_logFile = new std::ofstream(logFilePath.c_str(), std::ios::out | std::ios::app);
        if (g_logFile->fail())
        {
            throw std::runtime_error("Couldn't open file: " + logFilePath);
        }

        // legend
        #ifdef CUB_CDP
        if (g_startBenchmark)
        {
            *g_logFile << "% "; 
            logValue(*g_logFile, "dim", 10);
            logValue(*g_logFile, "blockSize", 10);
            logValue(*g_logFile, "row-major(distMtx)", 13);
            logValue(*g_logFile, "col-major(distMtx)", 13);
            *g_logFile << "\n";
            *g_logFile << "% values in brackets(regs used, local mem used, smem used) \n";
        }
        #else
        if (g_startBenchmark)
        {
            *g_logFile << "% "; 
            logValue(*g_logFile, "dim", 10);
            logValue(*g_logFile, "blockSize", 10);
            logValue(*g_logFile, "row-major(v1)", 13);
            logValue(*g_logFile, "row-major(nan)", 14);
            logValue(*g_logFile, "col-major(v1)", 13);
            logValue(*g_logFile, "col-major(nan)", 14);
            *g_logFile << "\n";
            *g_logFile << "% values in brackets(regs used, local mem used, smem used) \n";
        }
        #endif
    }
}

int main(int argc, char const **argv)
{

    rd::RDParams<double> dParams;
    rd::RDSpiralParams<double> dSParams;
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
            "\t\t[--log <save results to file>]\n"
            "\t\t[--start <mark start of benchmark in log file>]\n"
            // "\t\t[--end <mark end of benchmark in log file>]\n"
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
        args.GetCmdLineArgument("d", g_devId);
    }
    if (args.CheckCmdLineFlag("log")) 
    {
        g_logPerfResults = true;
    }
    if (args.CheckCmdLineFlag("start")) 
    {
        g_startBenchmark = true;
    }
    // if (args.CheckCmdLineFlag("end")) 
    // {
    //     g_endBenchmark = true;
    // }

    deviceInit(g_devId);

    // set device name for logging and drawing purposes
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, g_devId));
    g_devName = devProp.name;

    std::cout << "Samples: " << std::endl;
    // std::cout <<  "\t dimension: " << TEST_DIM << std::endl;
    std::cout <<  "\t n_samples: " << fParams.np << std::endl;
    std::cout <<  "\t r1: " << fParams.r1 << std::endl;
    std::cout <<  "\t r2: " << fParams.r2 << std::endl;

    std::cout << "Segment params: " << std::endl;
    std::cout <<  "\t a: " << fSParams.a << std::endl;
    std::cout <<  "\t b: " << fSParams.b << std::endl;
    std::cout <<  "\t sigma: " << fSParams.sigma << std::endl; 


    if (g_logPerfResults)
    {
        initializeLogFile<float>();
    }

    if (g_logPerfResults && g_startBenchmark)
    {
        *g_logFile << "%np=" << fParams.np << " a=" << fSParams.a 
            << " b=" << fSParams.b << " s=" << fSParams.sigma
            << " r1=" << fParams.r1  << " r2=" << fParams.r2 << std::endl;
    }

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    benchmarkDecimateKernel<2, float>(fParams, fSParams);
    benchmarkDecimateKernel<3, float>(fParams, fSParams);
    benchmarkDecimateKernel<4, float>(fParams, fSParams);
    benchmarkDecimateKernel<5, float>(fParams, fSParams);
    benchmarkDecimateKernel<6, float>(fParams, fSParams);
    // benchmarkDecimateKernel<10, float>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    

    if (g_logPerfResults)
    {
        g_logFile->close();
        delete g_logFile;
        // initializeLogFile<double>();
    }

    // if (g_logPerfResults && g_startBenchmark)
    // {
    //     *g_logFile << "%np=" << rdp.np << " a=" << sp.a << " b=" << sp.b << " s=" << sp.sigma
    //     << " r1=" << rdp.r1  << " r2=" << rdp.r2 << std::endl;
    // }

    // std::cout << rd::HLINE << std::endl;
    // std::cout << "DOUBLE: " << std::endl;
    // benchmarkDecimateKernel<2, double>(dParams, dSParams);
    // benchmarkDecimateKernel<3, double>(dParams, dSParams);
    // std::cout << rd::HLINE << std::endl;

    // if (g_logPerfResults)
    // {
    //     g_logFile->close();
    //     delete g_logFile;
    // }


    deviceReset();

    std::cout << "END!" << std::endl;
    return 0;
}

template <
    int                     DIM,
    int                     BLOCK_SIZE,
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename            T>
__launch_bounds__ (BLOCK_SIZE)
__global__ void DeviceDecimateKernel_v1(
        T *     S,
        int *   ns,
        T       r,
        int     stride = 0)
{
    typedef rd::gpu::bruteForce::BlockDecimate<T, DIM, BLOCK_SIZE, MEM_LAYOUT> BlockDecimateT;
    BlockDecimateT().decimate(S, ns, r, stride);
}

template<
    int                     TEST_DIM,
    rd::DataMemoryLayout    MEM_LAYOUT,
    int                     BLOCK_SIZE, 
    typename                T>
void dispatchDecimate_v1(
    rd::RDParams<T> const &rdp,
    T *         d_S,
    int         sStride,
    T const *   d_chosenS,
    int         chsStride,
    int *       d_ns,
    float       memcpyTime)
{
    GpuTimer timer;
    float kernelTime;

    // warm-up
    #ifndef RD_PROFILE
    DeviceDecimateKernel_v1<TEST_DIM, BLOCK_SIZE, MEM_LAYOUT><<<1, BLOCK_SIZE>>>(
        d_S, d_ns, rdp.r2, sStride);
    checkCudaErrors(cudaGetLastError());
    #endif

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    if (MEM_LAYOUT == rd::ROW_MAJOR)
    {
        timer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.ns * TEST_DIM * sizeof(T), 
                cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &rdp.ns, sizeof(int)));
            DeviceDecimateKernel_v1<TEST_DIM, BLOCK_SIZE, MEM_LAYOUT><<<1, BLOCK_SIZE>>>(
                d_S, d_ns, rdp.r2, sStride);
            checkCudaErrors(cudaGetLastError());
        }
        timer.Stop();
        kernelTime = timer.ElapsedMillis();
    }
    else
    {
        timer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemcpy2D(d_S, sStride * sizeof(T), d_chosenS, 
                chsStride * sizeof(T), rdp.ns * sizeof(T), TEST_DIM, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &rdp.ns, sizeof(int)));
            DeviceDecimateKernel_v1<TEST_DIM, BLOCK_SIZE, MEM_LAYOUT><<<1, BLOCK_SIZE>>>(
                d_S, d_ns, rdp.r2, sStride);
            checkCudaErrors(cudaGetLastError());
        }
        timer.Stop();
        kernelTime = timer.ElapsedMillis();
    }
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    KernelResourceUsage kernelResUsage(
        DeviceDecimateKernel_v1<TEST_DIM, BLOCK_SIZE, MEM_LAYOUT, T>);

    kernelTime = (kernelTime - memcpyTime) / static_cast<float>(ITER);
    if (g_logPerfResults)
    {
        logValue(*g_logFile, kernelTime);
        *g_logFile << kernelResUsage.prettyPrint();
    }
    logValue(std::cout, kernelTime);
    std::cout << kernelResUsage.prettyPrint();
}

template<
    int                     TEST_DIM,
    int                     BLOCK_SIZE,
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename                T>
void dispatchDecimateNaNMark(
    rd::RDParams<T> const & rdp,
    T *         d_S,
    int         sStride,
    T const *   d_chosenS,
    int         chsStride,
    int *       d_ns,
    float       memcpyTime)
{
    GpuTimer timer;
    float kernelTime = -1;

    typedef void (*KernelPtrT)(T*, int*, T const, int, cub::Int2Type<MEM_LAYOUT>);
    KernelPtrT kernelPtr = rd::gpu::bruteForce::decimateNanMarker1<BLOCK_SIZE, TEST_DIM, T>;

    // warm-up
    #ifndef RD_PROFILE
    kernelPtr<<<1, BLOCK_SIZE>>>(
        d_S, d_ns, rdp.r2, sStride, cub::Int2Type<MEM_LAYOUT>());
    checkCudaErrors(cudaGetLastError());
    #endif

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    if (MEM_LAYOUT == rd::ROW_MAJOR)
    {
        timer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.ns * TEST_DIM * sizeof(T), 
                cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &rdp.ns, sizeof(int)));
            kernelPtr<<<1, BLOCK_SIZE>>>(
                d_S, d_ns, rdp.r2, sStride, cub::Int2Type<MEM_LAYOUT>());
            checkCudaErrors(cudaGetLastError());
        }
        timer.Stop();
        kernelTime = timer.ElapsedMillis();
    }
    else
    {
        timer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemcpy2D(d_S, sStride * sizeof(T), d_chosenS, 
                chsStride * sizeof(T), rdp.ns * sizeof(T), TEST_DIM, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &rdp.ns, sizeof(int)));
            kernelPtr<<<1, BLOCK_SIZE>>>(
                d_S, d_ns, rdp.r2, sStride, cub::Int2Type<MEM_LAYOUT>());
            checkCudaErrors(cudaGetLastError());
        }
        timer.Stop();
        kernelTime = timer.ElapsedMillis();
    }
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    KernelResourceUsage kernelResUsage(kernelPtr);

    kernelTime = (kernelTime - memcpyTime) / static_cast<float>(ITER);
    if (g_logPerfResults)
    {
        logValue(*g_logFile, kernelTime);
        *g_logFile << kernelResUsage.prettyPrint();
    }
    logValue(std::cout, kernelTime);
    std::cout << kernelResUsage.prettyPrint();
}

template<
    int                     TEST_DIM,
    int                     BLOCK_SIZE,
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename                T>
void dispatchDecimate_v1AndNanMark(
    rd::RDParams<T> const & rdp,
    T *         d_S,
    int         sStride,
    T const *   d_chosenS,
    int         chsStride,
    int *       d_ns,
    float       memcpyTime)
{
    if (g_logPerfResults)
    {
        logValues(*g_logFile, TEST_DIM, BLOCK_SIZE, 
            std::string(rd::DataMemoryLayoutNameTraits<MEM_LAYOUT>::name));
    }
    logValues(std::cout, TEST_DIM, BLOCK_SIZE,
        std::string(rd::DataMemoryLayoutNameTraits<MEM_LAYOUT>::name));

    dispatchDecimate_v1<TEST_DIM, MEM_LAYOUT, BLOCK_SIZE>(rdp, d_S, sStride, d_chosenS, 
        chsStride, d_ns, memcpyTime);
    dispatchDecimateNaNMark<TEST_DIM, BLOCK_SIZE, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS, 
        chsStride, d_ns, memcpyTime);

    if (g_logPerfResults)
    {
        *g_logFile << std::endl;
    }
    std::cout << std::endl;
}

template<
    int                     TEST_DIM,
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename                T>
void dispatchDecimate_v1AndNanMark(
    rd::RDParams<T> const & rdp,
    T *         d_S,
    int         sStride,
    T const *   d_chosenS,
    int         chsStride,
    int *       d_ns,
    float       memcpyTime)
{
    #ifdef QUICK_TEST
    dispatchDecimate_v1AndNanMark<TEST_DIM, 512, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    #else
    dispatchDecimate_v1AndNanMark<TEST_DIM, 64, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 96, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 128, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 160, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 192, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 224, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 256, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 288, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 320, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 352, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 384, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 416, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 448, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 480, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 512, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 544, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 576, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 608, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 640, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 672, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 704, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 736, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 768, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 800, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 832, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 864, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 896, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 928, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 960, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 992, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    dispatchDecimate_v1AndNanMark<TEST_DIM, 1024, MEM_LAYOUT>(rdp, d_S, sStride, d_chosenS,
        chsStride, d_ns, memcpyTime);
    #endif
}


template <
    int                     TEST_DIM,
    rd::DataMemoryLayout    MEM_LAYOUT,
    int                     BLOCK_SIZE,
    typename                T>
static void dispatchDecimateDistMtx(
    rd::RDParams<T> const &rdp,
    T * d_S, 
    int * d_ns, 
    int sStride, 
    T const * d_chosenS,
    int chsStride,
    T * d_distMtx, 
    size_t distMtxPitch, 
    char * d_mask, 
    float initTime)
{
    int distMtxStride = distMtxPitch / sizeof(T);

    GpuTimer timer;
    float kernelTime = -1;

    auto kernelPtr = rd::gpu::bruteForce::decimateDistMtx<TEST_DIM, BLOCK_SIZE, MEM_LAYOUT, T>;

    // warm-up
    #ifndef RD_PROFILE
    kernelPtr<<<1, BLOCK_SIZE>>>(d_S, d_ns, sStride, d_distMtx, distMtxStride, d_mask, rdp.r2);
    checkCudaErrors(cudaGetLastError());
    #endif

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    if (MEM_LAYOUT == rd::ROW_MAJOR)
    {
        timer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemset2D(d_distMtx, distMtxPitch, 0, rdp.ns * sizeof(T), rdp.ns));
            checkCudaErrors(cudaMemset(d_mask, 1, rdp.ns * sizeof(char)));
            checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.ns * TEST_DIM * sizeof(T),
                cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &rdp.ns, sizeof(int)));
            kernelPtr<<<1, BLOCK_SIZE>>>(d_S, d_ns, sStride, d_distMtx, distMtxStride, d_mask, 
                rdp.r2);
        }
        timer.Stop();
    }
    else if (MEM_LAYOUT == rd::COL_MAJOR)
    {
        timer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemset2D(d_distMtx, distMtxPitch, 0, rdp.ns * sizeof(T), 
                rdp.ns));
            checkCudaErrors(cudaMemset(d_mask, 1, rdp.ns * sizeof(char)));
            rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::COL_MAJOR, cudaMemcpyDeviceToDevice>(
                    d_S, d_chosenS, rdp.ns, TEST_DIM, sStride * sizeof(T), chsStride * sizeof(T));
            checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &rdp.ns, sizeof(int)));
            kernelPtr<<<1, BLOCK_SIZE>>>(d_S, d_ns, sStride, d_distMtx, distMtxStride, d_mask, 
                rdp.r2);
        }
        timer.Stop();
    }

    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    KernelResourceUsage kernelResUsage(kernelPtr);

    kernelTime = (kernelTime - initTime) / static_cast<float>(ITER);
    if (g_logPerfResults)
    {
        logValues(*g_logFile, TEST_DIM, BLOCK_SIZE, 
            std::string(rd::DataMemoryLayoutNameTraits<MEM_LAYOUT>::name), kernelTime);
        *g_logFile << kernelResUsage.prettyPrint();
        *g_logFile << std::endl;
    }
    logValues(std::cout, TEST_DIM, BLOCK_SIZE, 
            std::string(rd::DataMemoryLayoutNameTraits<MEM_LAYOUT>::name), kernelTime);
    std::cout << kernelResUsage.prettyPrint();
    std::cout << std::endl;
}

template <
    int                     TEST_DIM,
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename                T>
static void benchmarkDecimateDistMtx(
    rd::RDParams<T> const & rdp,
    T *         d_S,
    int         sStride,
    T const *   d_chosenS,
    int         chsStride,
    int *       d_ns,
    float       memcpyTime)
{

    size_t distMtxPitch;
    T *d_distMtx;
    char *d_mask;

    checkCudaErrors(cudaMallocPitch(&d_distMtx, &distMtxPitch, rdp.ns * sizeof(T), rdp.ns));
    checkCudaErrors(cudaMalloc(&d_mask, rdp.ns * sizeof(char)));

    float initTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemset2D(d_distMtx, distMtxPitch, 0, rdp.ns * sizeof(T), 
        rdp.ns));
        // checkCudaErrors(cudaMemset(d_mask, 1, rdp.ns * sizeof(char)));
    }
    timer.Stop();
    initTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    // dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, RD_BLOCK_SIZE>(rdp, d_S, d_ns, sStride, 
    //     d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    #ifdef QUICK_TEST
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 512>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    #else
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 64>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 96>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 128>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 160>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 192>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 224>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 256>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 288>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 320>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 352>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 384>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 416>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 448>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 480>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 512>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 544>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 576>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 608>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 640>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 672>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 704>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 736>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 768>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 800>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 832>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 864>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 896>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 928>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 960>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 992>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    dispatchDecimateDistMtx<TEST_DIM, MEM_LAYOUT, 1024>(rdp, d_S, d_ns, sStride, 
        d_chosenS, chsStride, d_distMtx, distMtxPitch, d_mask, initTime + memcpyTime);
    #endif
    checkCudaErrors(cudaFree(d_distMtx));
        checkCudaErrors(cudaFree(d_mask));
}

// ROW_MAJOR version
template <int TEST_DIM, typename T>
void benchmarkDecimate(
    rd::RDParams<T> & rdp,
    T const * d_P)
{
    T * d_S, * d_chosenS;
    checkCudaErrors(cudaMalloc(&d_chosenS, rdp.np * TEST_DIM * sizeof(T)));

    int *d_ns, zero = 0;
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &zero, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::gpu::bruteForce::DeviceChoose::choose<TEST_DIM, rd::ROW_MAJOR, rd::ROW_MAJOR>(
        d_P, d_chosenS, rdp.np, d_ns, rdp.r1, TEST_DIM, TEST_DIM);
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaMemcpyFromSymbol(&rdp.ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Chosen points count: " << rdp.ns << std::endl;

    checkCudaErrors(cudaMalloc(&d_S, rdp.ns * TEST_DIM * sizeof(T)));

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.ns * TEST_DIM * sizeof(T), 
            cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &rdp.ns, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    #ifndef CUB_CDP
    dispatchDecimate_v1AndNanMark<TEST_DIM, rd::ROW_MAJOR>(rdp, d_S, TEST_DIM, d_chosenS, 
        TEST_DIM, d_ns, memcpyTime);
    #else
    benchmarkDecimateDistMtx<TEST_DIM, rd::ROW_MAJOR>(rdp, d_S, TEST_DIM, d_chosenS, TEST_DIM,
        d_ns, memcpyTime);
    #endif

    checkCudaErrors(cudaFree(d_S));
    checkCudaErrors(cudaFree(d_chosenS));
}

// COL_MAJOR version
template <int TEST_DIM, typename T>
void benchmarkDecimate(
    rd::RDParams<T> & rdp,
    T const * d_P,
    int pStride)
{
    size_t sPitch, chsPitch;
    T * d_S, * d_chosenS;
    checkCudaErrors(cudaMallocPitch(&d_chosenS, &chsPitch, rdp.np * sizeof(T), TEST_DIM));

    int *d_ns, zero = 0;
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &zero, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::gpu::bruteForce::DeviceChoose::choose<TEST_DIM, rd::COL_MAJOR, rd::COL_MAJOR>(
        d_P, d_chosenS, rdp.np, d_ns, rdp.r1, pStride, chsPitch / sizeof(T));
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaMemcpyFromSymbol(&rdp.ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Chosen samples count: " << rdp.ns << std::endl;

    checkCudaErrors(cudaMallocPitch(&d_S, &sPitch, rdp.ns * sizeof(T), TEST_DIM));

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpy2D(d_S, sPitch, d_chosenS, chsPitch, 
            rdp.ns * sizeof(T), TEST_DIM, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &rdp.ns, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    #ifndef CUB_CDP
    dispatchDecimate_v1AndNanMark<TEST_DIM, rd::COL_MAJOR>(rdp, d_S, sPitch / sizeof(T), d_chosenS, 
        chsPitch / sizeof(T), d_ns, memcpyTime);
    #else
    benchmarkDecimateDistMtx<TEST_DIM, rd::COL_MAJOR>(rdp, d_S, sPitch / sizeof(T), d_chosenS, 
        chsPitch / sizeof(T), d_ns, memcpyTime);
    #endif
}

template <int TEST_DIM, typename T>
void benchmarkDecimateKernel(
    rd::RDParams<T>             &rdp,
    rd::RDSpiralParams<T> const &sp)
{
    size_t pPitch;
    T *d_Pcol, *d_Prow;
    checkCudaErrors(cudaMalloc(&d_Prow, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMallocPitch(&d_Pcol, &pPitch, rdp.np * sizeof(T), TEST_DIM));

    rd::gpu::SamplesGenerator<T>::template segmentND<rd::ROW_MAJOR>(
        rdp.np, TEST_DIM, sp.sigma, sp.a, d_Prow);
    rd::gpu::SamplesGenerator<T>::template segmentND<rd::COL_MAJOR>(
        rdp.np, TEST_DIM, sp.sigma, sp.a, d_Pcol, pPitch / sizeof(T));
    checkCudaErrors(cudaDeviceSynchronize());

    // if (TEST_DIM == 2 || TEST_DIM == 3)
    // {
    //     T * h_P = new T[rdp.np * TEST_DIM];

    //     checkCudaErrors(cudaMemcpy(h_P, d_Prow, rdp.np * TEST_DIM * sizeof(T), 
    //         cudaMemcpyDeviceToHost));
    //     checkCudaErrors(cudaDeviceSynchronize());

    //     std::ostringstream os;
    //     rd::GraphDrawer<T> gDrawer;
    //     os << typeid(T).name() << "_" << TEST_DIM;
    //     os << "D_initial_samples_set_row_major";
    //     gDrawer.showPoints(os.str(), h_P, rdp.np, TEST_DIM);
    //     os.clear();
    //     os.str(std::string());

    //     rd::fillTable(h_P, T(0), rdp.np * TEST_DIM);
    //     rd::gpu::rdMemcpy2D<rd::ROW_MAJOR, rd::COL_MAJOR, cudaMemcpyDeviceToHost>(
    //         h_P, d_Pcol, rdp.np, TEST_DIM, TEST_DIM * sizeof(T), pPitch);

    //     os << typeid(T).name() << "_" << TEST_DIM;
    //     os << "D_initial_samples_set_col_major";
    //     gDrawer.showPoints(os.str(), h_P, rdp.np, TEST_DIM);
    //     os.clear();
    //     os.str(std::string());
    //     delete[] h_P;
    // }
    
    std::cout << "start benchmark!" << std::endl;

    benchmarkDecimate<TEST_DIM>(rdp, d_Prow);
    benchmarkDecimate<TEST_DIM>(rdp, d_Pcol, pPitch / sizeof(T));

    checkCudaErrors(cudaFree(d_Prow));
    checkCudaErrors(cudaFree(d_Pcol));
}

