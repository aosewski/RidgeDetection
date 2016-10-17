/**
 * @file benchmark_choose.cu
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
#include <cuda_profiler_api.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <limits>

#include "rd/gpu/device/brute_force/choose.cuh"
#include "rd/gpu/device/brute_force/rd_globals.cuh"
#include "rd/gpu/device/samples_generator.cuh"
#include "rd/gpu/util/data_order_traits.hpp"

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/rd_params.hpp"
#include "tests/test_util.hpp"

#include "cub/test_util.h"

static const int ZERO          = 0;
static const std::string LOG_FILE_NAME_SUFFIX = "_choose-timings.txt";

static std::ofstream   *g_logFile = nullptr;
static std::string      g_devName;
static int              g_devId = 0;
static bool             g_logPerfResults = false;
static bool             g_startBenchmark = false;
// static bool             g_endBenchmark = false;

#if defined(RD_PROFILE) || defined(DEBUG)
const int ITER = 1;
#else
const int ITER = 100;
#endif

template <int TEST_DIM, typename T>
void testChooseKernel(rd::RDParams<T> &rdp,
                      rd::RDSpiralParams<T> const &rds);


template <typename T>
static void initializeLogFile()
{
    if (g_logPerfResults)
    {
        std::ostringstream logFileName;
        logFileName << typeid(T).name() << "_" <<  getCurrDate() << "_" 
                    << g_devName << "_" << LOG_FILE_NAME_SUFFIX;

        std::string logFilePath = rd::findPath("../timings/", logFileName.str());
        g_logFile = new std::ofstream(logFilePath.c_str(), std::ios::out | std::ios::app);
        if (g_logFile->fail())
        {
            throw std::logic_error("Couldn't open file: " + logFilePath);
        }

        // legend
        if (g_startBenchmark)
        {
            *g_logFile << "% "; 
            logValue(*g_logFile, "dim", 10);
            logValue(*g_logFile, "blockSize", 10);
            logValue(*g_logFile, "row-major(v1)", 13);
            logValue(*g_logFile, "col-major(v1)", 13);
            logValue(*g_logFile, "mixed-ord(v1)", 13);
            logValue(*g_logFile, "soa(v1)", 10);
            logValue(*g_logFile, "row-major(v2)", 13);
            logValue(*g_logFile, "col-major(v2)", 13);
            logValue(*g_logFile, "mixed-ord(v2)", 13);
            *g_logFile << "\n";
        }
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

    if (g_logPerfResults)
    {
        initializeLogFile<float>();
    }

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    testChooseKernel<2, float>(fParams, fSParams);
    testChooseKernel<3, float>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    

    if (g_logPerfResults)
    {
        g_logFile->close();
        delete g_logFile;
        initializeLogFile<double>();
    }

    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE: " << std::endl;
    testChooseKernel<2, double>(dParams, dSParams);
    testChooseKernel<3, double>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;

    if (g_logPerfResults)
    {
        g_logFile->close();
        delete g_logFile;
    }

    deviceReset();

    std::cout << "END!" << std::endl;
    return 0;
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
    typename                T,
    int                     DIM,
    int                     BLOCK_SIZE,
    rd::DataMemoryLayout  INPUT_MEM_LAYOUT,
    rd::DataMemoryLayout  OUTPUT_MEM_LAYOUT>
__launch_bounds__ (BLOCK_SIZE)
__global__ void DeviceChooseKernel(
        T const * __restrict__ P,
        T * S,
        int np,
        int *ns,
        T r,
        int pStride,
        int sStride)
{

    typedef rd::gpu::bruteForce::BlockChoose<T, DIM, BLOCK_SIZE, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT> AgentChooseT;
    AgentChooseT().choose(P, S, np, ns, r, pStride, sStride);
}


template<
    int         TEST_DIM,
    int         BLOCK_SIZE, 
    typename    T>
float dispatchChooseMixedOrder(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    T *d_S,
    int *d_ns,
    float memcpyTime)
{

    GpuTimer timer;
    float kernelTime;

    // warm-up
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    __choose_kernel_v1<T, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_P, d_S, rdp.np, rdp.r1, d_ns, TEST_DIM);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
        __choose_kernel_v1<T, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(
            d_P, d_S, rdp.np, rdp.r1, d_ns, TEST_DIM);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    kernelTime = (kernelTime - memcpyTime) / static_cast<float>(ITER);
    if (g_logPerfResults)
    {
        *g_logFile << " " << kernelTime;
    }
    std::cout << " " << BLOCK_SIZE << " " << kernelTime << std::endl;

    return kernelTime;
}

template<int TEST_DIM, typename T>
void benchChooseMixedOrder(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    T *d_S)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "benchChooseMixedOrder:" << std::endl;

    int *d_ns;
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    dispatchChooseMixedOrder<TEST_DIM, RD_BLOCK_SIZE>(rdp, d_P, d_S, d_ns, memcpyTime);
}

template<
    int         TEST_DIM,
    int         BLOCK_SIZE, 
    typename    T>
float dispatchChooseMixedOrder_v2(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    int pStride,
    T *d_S,
    int *d_ns,
    float memcpyTime)
{

    GpuTimer timer;
    float kernelTime;

    // warm-up
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    DeviceChooseKernel<T, TEST_DIM, BLOCK_SIZE, rd::COL_MAJOR, rd::ROW_MAJOR><<<1, BLOCK_SIZE>>>(
        d_P, d_S, rdp.np, d_ns, rdp.r1, pStride, 0);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
        DeviceChooseKernel<T, TEST_DIM, BLOCK_SIZE, rd::COL_MAJOR, rd::ROW_MAJOR><<<1, BLOCK_SIZE>>>(
            d_P, d_S, rdp.np, d_ns, rdp.r1, pStride, 0);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    KernelResourceUsage resUsage(DeviceChooseKernel<T, TEST_DIM, BLOCK_SIZE, rd::COL_MAJOR, 
        rd::ROW_MAJOR>);
    kernelTime = (kernelTime - memcpyTime) / static_cast<float>(ITER);
    if (g_logPerfResults)
    {
        *g_logFile << " " << kernelTime;
    }
    std::cout << " " << BLOCK_SIZE << " " << kernelTime << resUsage.prettyPrint() << std::endl;

    return kernelTime;
}

template<int TEST_DIM, typename T>
void benchChooseMixedOrder_v2(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    T *d_S)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "benchChooseMixedOrder_v2:" << std::endl;

    int *d_ns;
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));

    size_t pPitch = 0;
    T *d_P2;
    checkCudaErrors(cudaMallocPitch(&d_P2, &pPitch, rdp.np * sizeof(T), TEST_DIM));
    checkCudaErrors(cudaMemcpy2D(d_P2, pPitch, d_P, rdp.np * sizeof(T), rdp.np * sizeof(T), 
        TEST_DIM, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    dispatchChooseMixedOrder_v2<TEST_DIM, RD_BLOCK_SIZE>(rdp, d_P2, pPitch / sizeof(T), d_S, 
        d_ns, memcpyTime);
}

template<
    int         TEST_DIM,
    int         BLOCK_SIZE, 
    typename    T>
float dispatchChooseRowMajorOrder(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    T *d_S,
    int *d_ns,
    float memcpyTime)
{

    GpuTimer timer;
    float kernelTime;

    // warm-up
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    __choose_kernel_v1<T, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(
        d_P, d_S, rdp.np, rdp.r1, d_ns, TEST_DIM, rd::gpu::rowMajorOrderTag());
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
        __choose_kernel_v1<T, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(
            d_P, d_S, rdp.np, rdp.r1, d_ns, TEST_DIM, rd::gpu::rowMajorOrderTag());
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    kernelTime = (kernelTime - memcpyTime) / static_cast<float>(ITER);
    if (g_logPerfResults)
    {
        *g_logFile << " " << kernelTime;
    }
    std::cout << " " << BLOCK_SIZE << " " << kernelTime << std::endl;

    return kernelTime;
}

template <int TEST_DIM, typename T>
void benchChooseRowMajorOrder(
    rd::RDParams<T> const &rdp,
    T *d_S,
    T const *h_P)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "benchChooseRowMajorOrder:" << std::endl;

    T *d_PRowMajor;
    int *d_ns;

    checkCudaErrors(cudaMalloc((void**)&d_PRowMajor, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemcpy(d_PRowMajor, h_P, rdp.np * TEST_DIM * sizeof(T),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    dispatchChooseRowMajorOrder<TEST_DIM, RD_BLOCK_SIZE>(rdp, d_PRowMajor, d_S, d_ns, memcpyTime);
    checkCudaErrors(cudaFree(d_PRowMajor));
}

template<
    int         TEST_DIM,
    int         BLOCK_SIZE, 
    typename    T>
float dispatchChooseRowMajorOrder_v2(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    T *d_S,
    int *d_ns,
    float memcpyTime)
{

    GpuTimer timer;
    float kernelTime;

    // warm-up
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    DeviceChooseKernel<T, TEST_DIM, BLOCK_SIZE, rd::ROW_MAJOR, rd::ROW_MAJOR><<<1, BLOCK_SIZE>>>(
        d_P, d_S, rdp.np, d_ns, rdp.r1, 0, 0);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
        DeviceChooseKernel<T, TEST_DIM, BLOCK_SIZE, rd::ROW_MAJOR, rd::ROW_MAJOR>
            <<<1, BLOCK_SIZE>>>(d_P, d_S, rdp.np, d_ns, rdp.r1, 0, 0);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    KernelResourceUsage resUsage(DeviceChooseKernel<T, TEST_DIM, BLOCK_SIZE, rd::ROW_MAJOR, 
        rd::ROW_MAJOR>);
    kernelTime = (kernelTime - memcpyTime) / static_cast<float>(ITER);
    if (g_logPerfResults)
    {
        *g_logFile << " " << kernelTime;
    }
    std::cout << " " << BLOCK_SIZE << " " << kernelTime << resUsage.prettyPrint() << std::endl;

    return kernelTime;
}

template <int TEST_DIM, typename T>
void benchChooseRowMajorOrder_v2(
    rd::RDParams<T> const &rdp,
    T *d_S,
    T const *h_P)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "benchChooseRowMajorOrder_v2:" << std::endl;

    T *d_PRowMajor;
    int *d_ns;

    checkCudaErrors(cudaMalloc((void**)&d_PRowMajor, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemcpy(d_PRowMajor, h_P, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    dispatchChooseRowMajorOrder_v2<TEST_DIM, RD_BLOCK_SIZE>(rdp, d_PRowMajor, d_S, d_ns, memcpyTime);
    checkCudaErrors(cudaFree(d_PRowMajor));
}

template<
    int         TEST_DIM,
    int         BLOCK_SIZE, 
    typename    T>
float dispatchChooseColMajorOrder(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    T *d_S,
    int *d_ns,
    float memcpyTime)
{

    GpuTimer timer;
    float kernelTime;

    // warm-up
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    __choose_kernel_v1<T, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(
        d_P, d_S, rdp.np, rdp.r1, d_ns, TEST_DIM, rd::gpu::colMajorOrderTag());
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
        __choose_kernel_v1<T, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(
            d_P, d_S, rdp.np, rdp.r1, d_ns, TEST_DIM, rd::gpu::colMajorOrderTag());
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    kernelTime = (kernelTime - memcpyTime) / static_cast<float>(ITER);
    if (g_logPerfResults)
    {
        *g_logFile << " " << kernelTime;
    }
    std::cout << " " << BLOCK_SIZE << " " << kernelTime << std::endl;

    return kernelTime;
}


template <int TEST_DIM, typename T>
void benchChooseColMajorOrder(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    T *d_S)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "benchChooseColMajorOrder:" << std::endl;

    int *d_ns;

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaDeviceSynchronize());

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    dispatchChooseColMajorOrder<TEST_DIM, RD_BLOCK_SIZE>(rdp, d_P, d_S, d_ns, memcpyTime);
}

template<
    int         TEST_DIM,
    int         BLOCK_SIZE, 
    typename    T>
float dispatchChooseColMajorOrder_v2(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    int pStride,
    T *d_S,
    int sStride, 
    int *d_ns,
    float memcpyTime)
{

    GpuTimer timer;
    float kernelTime;

    // warm-up
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    DeviceChooseKernel<T, TEST_DIM, BLOCK_SIZE, rd::COL_MAJOR, rd::COL_MAJOR><<<1, BLOCK_SIZE>>>(
        d_P, d_S, rdp.np, d_ns, rdp.r1, pStride, sStride);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
        DeviceChooseKernel<T, TEST_DIM, BLOCK_SIZE, rd::COL_MAJOR, rd::COL_MAJOR>
            <<<1, BLOCK_SIZE>>>(d_P, d_S, rdp.np, d_ns, rdp.r1, pStride, sStride);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    KernelResourceUsage resUsage(DeviceChooseKernel<T, TEST_DIM, BLOCK_SIZE, rd::COL_MAJOR, 
        rd::COL_MAJOR>);
    kernelTime = (kernelTime - memcpyTime) / static_cast<float>(ITER);
    if (g_logPerfResults)
    {
        *g_logFile << " " << kernelTime;
    }
    std::cout << " " << BLOCK_SIZE << " " << kernelTime << resUsage.prettyPrint() << std::endl;

    return kernelTime;
}


template <int TEST_DIM, typename T>
void benchChooseColMajorOrder_v2(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    T *d_S)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "benchChooseColMajorOrder_v2:" << std::endl;

    int *d_ns;

    size_t pPitch, sPitch;
    T * d_P2, *d_S2;

    checkCudaErrors(cudaMallocPitch(&d_P2, &pPitch, rdp.np * sizeof(T), TEST_DIM));
    checkCudaErrors(cudaMallocPitch(&d_S2, &sPitch, rdp.np * sizeof(T), TEST_DIM));
    checkCudaErrors(cudaMemset2D(d_S2, sPitch, 0, rdp.np * sizeof(T), TEST_DIM));
    checkCudaErrors(cudaMemcpy2D(d_P2, pPitch, d_P, rdp.np * sizeof(T), rdp.np * sizeof(T), 
        TEST_DIM, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaDeviceSynchronize());

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    dispatchChooseColMajorOrder_v2<TEST_DIM, RD_BLOCK_SIZE>(rdp, d_P2, pPitch / sizeof(T), 
        d_S2, sPitch / sizeof(T), d_ns, memcpyTime);

    checkCudaErrors(cudaFree(d_P2));
    checkCudaErrors(cudaFree(d_S2));
}


template<
    int         TEST_DIM,
    int         BLOCK_SIZE,
    typename    SamplesDevT,
    typename    T>
float dispatchChooseSOA(
    rd::RDParams<T> const &rdp,
    SamplesDevT const *d_P,
    SamplesDevT *d_S,
    int *d_ns,
    float memcpyTime)
{

    GpuTimer timer;
    float kernelTime;

    // warm-up
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    __choose_kernel_v1<T, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(
        d_P->dSamples, d_S->dSamples, rdp.np, rdp.r1, d_ns, TEST_DIM);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
        __choose_kernel_v1<T, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(
            d_P->dSamples, d_S->dSamples, rdp.np, rdp.r1, d_ns, TEST_DIM);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif


    kernelTime = (kernelTime - memcpyTime) / static_cast<float>(ITER);
    if (g_logPerfResults)
    {
        *g_logFile << " " << kernelTime;
    }
    std::cout << " " << BLOCK_SIZE << " " << kernelTime << std::endl;

    return kernelTime;
}

template <int TEST_DIM, typename T>
void benchChooseSOA(
    rd::RDParams<T> const &rdp,
    T const *h_P)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "benchChooseSOA:" << std::endl;

    typedef rd::ColMajorDeviceSamples<T, TEST_DIM> SamplesDevT;

    SamplesDevT *d_P, *d_S;
    int *d_ns;

    d_P = new SamplesDevT(rdp.np);
    d_S = new SamplesDevT(rdp.np);

    T *h_aux = new T[rdp.np * TEST_DIM];
    rd::copyTable(h_P, h_aux, rdp.np * TEST_DIM);
    rd::transposeInPlace(h_aux, h_aux + rdp.np * TEST_DIM, TEST_DIM);
    d_P->copyFromContinuousData(h_aux, rdp.np);

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    dispatchChooseSOA<TEST_DIM, RD_BLOCK_SIZE>(rdp, d_P, d_S, d_ns, memcpyTime);

    delete[] h_aux;
    delete d_P;
    delete d_S;
}

template <int TEST_DIM, typename T>
void benchmarkChoose(
    rd::RDParams<T> const & rdp,
    rd::RDSpiralParams<T> const &sp,
    T * d_S,
    T * d_P,
    T * h_P)
{
    if (g_logPerfResults && g_startBenchmark)
    {
        *g_logFile << "%np=" << rdp.np << " a=" << sp.a << " b=" << sp.b << " s=" << sp.sigma
        << " r1=" << rdp.r1  << " r2=" << rdp.r2 << std::endl;
    }
    if (g_logPerfResults)
    {
        *g_logFile << TEST_DIM << " " << RD_BLOCK_SIZE;
    }

    benchChooseRowMajorOrder<TEST_DIM>(rdp, d_S, h_P);
    benchChooseColMajorOrder<TEST_DIM>(rdp, d_P, d_S);
    benchChooseMixedOrder<TEST_DIM>(rdp, d_P, d_S);
    // benchChooseSOA<TEST_DIM>(rdp, h_P);

    benchChooseRowMajorOrder_v2<TEST_DIM>(rdp, d_S, h_P);
    benchChooseColMajorOrder_v2<TEST_DIM>(rdp, d_P, d_S);
    benchChooseMixedOrder_v2<TEST_DIM>(rdp, d_P, d_S);

    if (g_logPerfResults)
    {
        *g_logFile << std::endl;
    }
}

template <int TEST_DIM, typename T>
void testChooseKernel(rd::RDParams<T> &rdp,
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

    rd::GraphDrawer<T> gDrawer;

    T *d_P, *d_S;
    T *h_P;

    checkCudaErrors(cudaMalloc((void**)&d_P, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_S, rdp.np * TEST_DIM * sizeof(T)));

    checkCudaErrors(cudaMemset(d_P, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));

    h_P = new T[rdp.np * TEST_DIM];

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
            throw std::logic_error("Not supported dimension!");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_P, d_P, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::transposeInPlace(h_P, h_P + rdp.np * TEST_DIM, rdp.np);

    // std::ostringstream os;
    // if (rdp.verbose)
    // {
    //     os << typeid(T).name() << "_" << TEST_DIM;
    //     os << "D_initial_samples_set_";
    //     gDrawer.showPoints(os.str(), h_P, rdp.np, TEST_DIM);
    //     os.clear();
    //     os.str(std::string());
    // }

    //---------------------------------------------------
    //               BENCHMARK CHOOSE 
    //---------------------------------------------------

    benchmarkChoose<TEST_DIM>(rdp, sp, d_S, d_P, h_P);

    // clean-up
    delete[] h_P;

    checkCudaErrors(cudaFree(d_P));
    checkCudaErrors(cudaFree(d_S));
}

