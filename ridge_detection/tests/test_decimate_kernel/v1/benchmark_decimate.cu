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
#include <cuda_profiler_api.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <limits>

#include "rd/cpu/brute_force/choose.hpp"
#include "rd/gpu/device/brute_force/decimate.cuh"
#include "rd/gpu/device/brute_force/rd_globals.cuh"
#include "rd/gpu/device/samples_generator.cuh"
#include "rd/gpu/util/data_order_traits.hpp"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/rd_samples.cuh"
#include "rd/utils/memory.h"
#include "tests/test_util.hpp"

#include "cub/test_util.h"
#include "cub/util_device.cuh"

#include "rd/utils/rd_params.hpp"

static const std::string LOG_FILE_NAME_SUFFIX = "_decimate-timings.txt";

static std::ofstream   *g_logFile = nullptr;
static std::string      g_devName;
static int              g_devId = 0;
static bool             g_logPerfResults = false;
static bool             g_startBenchmark = false;

#ifdef RD_PROFILE
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
            logValue(*g_logFile, "row-major(v2)", 13);
            logValue(*g_logFile, "col-major(v2)", 13);
            // logValue(*g_logFile, "soa(v2)", 13);
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
    benchmarkDecimateKernel<2, float>(fParams, fSParams);
    benchmarkDecimateKernel<3, float>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    

    if (g_logPerfResults)
    {
        g_logFile->close();
        delete g_logFile;
        initializeLogFile<double>();
    }

    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE: " << std::endl;
    benchmarkDecimateKernel<2, double>(dParams, dSParams);
    benchmarkDecimateKernel<3, double>(dParams, dSParams);
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

template <int TEST_DIM, typename T>
void initializeChosenSamples(
    rd::RDParams<T> &rdp,
    T *P,
    T *S)
{
    std::list<T*> csList;
    rd::choose(P, S, csList, rdp.np, rdp.ns, TEST_DIM, rdp.r1);

    std::cout << "Chosen count: " << rdp.ns << std::endl;
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
    int                     DIM,
    int                     BLOCK_SIZE,
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename            T>
__launch_bounds__ (BLOCK_SIZE)
__global__ void DeviceDecimateKernel(
        T *     S,
        int *   ns,
        T       r,
        int     stride = 0)
{
    typedef rd::gpu::bruteForce::BlockDecimate<T, DIM, BLOCK_SIZE, MEM_LAYOUT> BlockDecimateT;
    BlockDecimateT().decimate(S, ns, r, stride);
}

template<
    int         TEST_DIM,
    int         BLOCK_SIZE, 
    typename    T>
float dispatchDecimateRowMajorOrder(
    rd::RDParams<T> const &rdp,
    T *d_S,
    T const *d_chosenS,
    int *d_ns,
    int h_chosenCount,
    float memcpyTime)
{

    GpuTimer timer;
    float kernelTime;

    // warm-up
    checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    DeviceDecimateKernel<TEST_DIM, BLOCK_SIZE, rd::ROW_MAJOR><<<1, BLOCK_SIZE>>>(
        d_S, d_ns, rdp.r2);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.np * TEST_DIM * sizeof(T), 
            cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
        DeviceDecimateKernel<TEST_DIM, BLOCK_SIZE, rd::ROW_MAJOR><<<1, BLOCK_SIZE>>>(
            d_S, d_ns, rdp.r2);
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
void benchmarkDecimateRowMajorOrder(
    rd::RDParams<T> const &rdp,
    T *d_S,
    T const *h_chosenS,
    int h_chosenCount)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "benchmarkDecimateRowMajorOrder:" << std::endl;

    T *d_chosenS;
    int *d_ns;

    checkCudaErrors(cudaMalloc((void**)&d_chosenS, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_chosenS, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));

    // get chosen samples to device memory properly ordered
    checkCudaErrors(cudaMemcpy(d_chosenS, h_chosenS, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaDeviceSynchronize());

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.np * TEST_DIM * sizeof(T), 
            cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    dispatchDecimateRowMajorOrder<TEST_DIM, RD_BLOCK_SIZE>(rdp, d_S, d_chosenS, d_ns, 
        h_chosenCount, memcpyTime);
    checkCudaErrors(cudaFree(d_chosenS));
}

template<
    int         TEST_DIM,
    int         BLOCK_SIZE, 
    typename    T>
float dispatchDecimateColMajorOrder(
    rd::RDParams<T> const &rdp,
    T *d_S,
    T const *d_chosenS,
    int *d_ns,
    int h_chosenCount,
    float memcpyTime)
{

    GpuTimer timer;
    float kernelTime;

    // warm-up
    checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    DeviceDecimateKernel<TEST_DIM, BLOCK_SIZE, rd::COL_MAJOR><<<1, BLOCK_SIZE>>>(
        d_S, d_ns, rdp.r2);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.np * TEST_DIM * sizeof(T), 
            cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
        DeviceDecimateKernel<TEST_DIM, BLOCK_SIZE, rd::COL_MAJOR><<<1, BLOCK_SIZE>>>(
            d_S, d_ns, rdp.r2);
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
void benchmarkDecimateColMajorOrder(
    rd::RDParams<T> const &rdp,
    T *d_S,
    T const *h_chosenS,
    int h_chosenCount)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "benchmarkDecimateColMajorOrder:" << std::endl;

    T *d_chosenS, *h_auxS;
    int *d_ns;

    // get chosen samples to device memory properly ordered
    h_auxS = new T[rdp.np * TEST_DIM];
    rd::copyTable(h_chosenS, h_auxS, h_chosenCount * TEST_DIM);
    rd::transposeInPlace(h_auxS, h_auxS + rdp.np * TEST_DIM, TEST_DIM);
    
    checkCudaErrors(cudaMalloc((void**)&d_chosenS, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_chosenS, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));

    // get chosen samples to device memory properly ordered
    checkCudaErrors(cudaMemcpy(d_chosenS, h_auxS, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaDeviceSynchronize());

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.np * TEST_DIM * sizeof(T), 
            cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    dispatchDecimateColMajorOrder<TEST_DIM, RD_BLOCK_SIZE>(rdp, d_S, d_chosenS, d_ns, 
        h_chosenCount, memcpyTime);
    checkCudaErrors(cudaFree(d_chosenS));
    delete[] h_auxS;
}

template<
    int         TEST_DIM,
    int         BLOCK_SIZE,
    typename    SamplesDevT,
    typename    T>
float dispatchDecimateSOA(
    rd::RDParams<T> const &rdp,
    SamplesDevT *d_S,
    SamplesDevT const *d_chosenS,
    int *d_ns,
    int h_chosenCount,
    float memcpyTime)
{

    GpuTimer timer;
    float kernelTime;

    // warm-up
    d_S->copy(*d_chosenS);
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    __decimate_kernel_v1<T, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_S->dSamples, d_ns, rdp.r2, TEST_DIM);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        d_S->copy(*d_chosenS);
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
            __decimate_kernel_v1<T, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_S->dSamples, d_ns, rdp.r2, TEST_DIM);
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
void benchmarkDecimateSOA(
    rd::RDParams<T> const &rdp,
    T const *h_chosenS,
    int h_chosenCount)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "benchmarkDecimateSOA:" << std::endl;

    typedef rd::ColMajorDeviceSamples<T, TEST_DIM> SamplesDevT;

    SamplesDevT *d_S, *d_chosenS;
    int *d_ns;

    d_S = new SamplesDevT(rdp.np);
    d_chosenS = new SamplesDevT(rdp.np);

    T *h_aux = new T[rdp.np * TEST_DIM];
    rd::copyTable(h_chosenS, h_aux, h_chosenCount * TEST_DIM);
    rd::transposeInPlace(h_aux, h_aux + rdp.np * TEST_DIM, TEST_DIM);
    d_chosenS->copyFromContinuousData(h_aux, rdp.np);

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaDeviceSynchronize());

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        d_S->copy(*d_chosenS);
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    dispatchDecimateSOA<TEST_DIM, RD_BLOCK_SIZE>(rdp, d_S, d_chosenS, d_ns, h_chosenCount, 
        memcpyTime);

    delete[] h_aux;
    delete d_S;
    delete d_chosenS;
}

template<
    int         TEST_DIM,
    int         BLOCK_SIZE,
    typename    T>
float dispatchDecimateNaNMark(
    rd::RDParams<T> const &rdp,
    T *d_S,
    T const *d_chosenS,
    int *d_ns,
    int h_chosenCount,
    float memcpyTime)
{

    GpuTimer timer;
    float kernelTime = -1;

    // warm-up
    checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    __decimate_v3<BLOCK_SIZE, TEST_DIM><<<1, BLOCK_SIZE>>>(d_S, d_ns, rdp.r2, 
        rd::gpu::rowMajorOrderTag());
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.np * TEST_DIM * sizeof(T), 
            cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
        __decimate_v3<BLOCK_SIZE, TEST_DIM><<<1, BLOCK_SIZE>>>(d_S, d_ns, rdp.r2, 
            rd::gpu::rowMajorOrderTag());
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
void benchmarkDecimateNaNMark(
    rd::RDParams<T> const &rdp,
    T *d_S,
    T const *h_chosenS,
    int h_chosenCount)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "benchmarkDecimateNaNMark:" << std::endl;

    T *d_chosenS;
    int *d_ns;

    checkCudaErrors(cudaMalloc((void**)&d_chosenS, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_chosenS, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));

    // get chosen samples to device memory properly ordered
    checkCudaErrors(cudaMemcpy(d_chosenS, h_chosenS, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaDeviceSynchronize());

    float memcpyTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpy(d_S, d_chosenS, rdp.np * TEST_DIM * sizeof(T), 
            cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    }
    timer.Stop();
    memcpyTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    dispatchDecimateNaNMark<TEST_DIM, RD_BLOCK_SIZE>(rdp, d_S, d_chosenS, d_ns, h_chosenCount, 
        memcpyTime);
    checkCudaErrors(cudaFree(d_chosenS));
}


template <int TEST_DIM, typename T>
void benchmarkDecimate(
    rd::RDParams<T> const & rdp,
    rd::RDSpiralParams<T> const &sp,
    T * d_S,
    T * h_S)
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

    // int smVersion;
    // checkCudaErrors(cub::SmVersion(smVersion, rdp.devId));

    benchmarkDecimateRowMajorOrder<TEST_DIM>(rdp, d_S, h_S, rdp.ns);
    benchmarkDecimateColMajorOrder<TEST_DIM>(rdp, d_S, h_S, rdp.ns);
    // benchmarkDecimateSOA<TEST_DIM>(rdp, h_S, rdp.ns);
    
    // if (smVersion >= 300)
    // {
    //     benchmarkDecimateNaNMark(rdp, d_S, h_S, rdp.ns);
    // }

    if (g_logPerfResults)
    {
        *g_logFile << std::endl;
    }
}

template <int TEST_DIM, typename T>
void benchmarkDecimateKernel(
    rd::RDParams<T>             &rdp,
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
    T *h_P, *h_S, *h_chosenS;

    checkCudaErrors(cudaMalloc((void**)&d_P, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_S, rdp.np * TEST_DIM * sizeof(T)));

    checkCudaErrors(cudaMemset(d_P, 0, rdp.np * TEST_DIM * sizeof(T)));

    h_P = new T[rdp.np * TEST_DIM];
    h_S = new T[rdp.np * TEST_DIM];
    h_chosenS = new T[rdp.np * TEST_DIM];

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
    // if (g_logPerfResults)
    // {
    //     os << typeid(T).name() << "_" << TEST_DIM;
    //     os << "D_initial_samples_set_";
    //     gDrawer.showPoints(os.str(), h_P, rdp.np, TEST_DIM);
    //     os.clear();
    //     os.str(std::string());
    // }

    initializeChosenSamples<TEST_DIM>(rdp, h_P, h_S);

    //---------------------------------------------------
    //               GPU CHOOSE 
    //---------------------------------------------------

    benchmarkDecimate<TEST_DIM>(rdp, sp, d_S, h_S);

    delete[] h_P;
    delete[] h_S;
    delete[] h_chosenS;

    checkCudaErrors(cudaFree(d_P));
    checkCudaErrors(cudaFree(d_S));
}

