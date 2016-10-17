/**
 * @file benchmark_dev_evolve.cu
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

#if defined(RD_DEBUG) && !defined(CUB_STDERR)
#define CUB_STDERR 
#endif

#include <helper_cuda.h>

#ifdef RD_PROFILE
#   include <cuda_profiler_api.h>
#   include <nvToolsExt.h>
#endif

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <limits>
#include <fstream>
#include <iomanip>

#include "rd/gpu/device/brute_force/evolve.cuh"
#include "rd/gpu/device/device_choose.cuh"
// #include "rd/gpu/device/device_evolve.cuh"
#include "rd/gpu/device/brute_force/rd_globals.cuh"
#include "rd/gpu/device/dispatch/dispatch_evolve.cuh"
#include "rd/gpu/util/dev_memcpy.cuh"

#include "rd/cpu/samples_generator.hpp"
#include "rd/cpu/brute_force/choose.hpp"

#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/memory.h"
#include "rd/utils/name_traits.hpp"
#include "tests/test_util.hpp"

#include "cub/test_util.h"
#include "cub/util_arch.cuh"
#include "cub/util_device.cuh"

/*
 *  Global variables 
 */

static const std::string LOG_FILE_NAME_SUFFIX   = "evolve_perf.txt";
static constexpr int MAX_TEST_DIM               = 16;


#if defined(RD_PROFILE) || defined(RD_DEBUG)
    const int ITER = 1;
#else
    const int ITER = 100;
#endif

static std::ofstream *  g_logFile           = nullptr;
static bool             g_logPerfResults    = false;
static std::string      g_devName           = "";
static int              g_devId             = 0;

//------------------------------------------------------------
//  Utils
//------------------------------------------------------------

/**
 * @brief      Create if necessary and open log file. Allocate log file stream.
 */
template <typename T>
static void initializeLogFile()
{
    if (g_logPerfResults)
    {
        std::ostringstream logFileName;
        logFileName << getCurrDate() << "_" <<
            g_devName << "_" << getBinConfSuffix() 
        #ifdef RD_CCSC
            << "_ccsc_"
        #endif
        #ifdef RD_STMC
            << "_stmc_"
        #endif
            << LOG_FILE_NAME_SUFFIX;

        std::string logFilePath = rd::findPath("../timings/", logFileName.str());
        g_logFile = new std::ofstream(logFilePath.c_str(), std::ios::out | std::ios::app);
        if (g_logFile->fail())
        {
            throw std::logic_error("Couldn't open file: " + logFileName.str());
        }

        *g_logFile << "%" << rd::HLINE << std::endl;
        *g_logFile << "% " << typeid(T).name() << std::endl;
        *g_logFile << "%" << rd::HLINE << std::endl;
        // legend
        *g_logFile << "% "; 
        logValue(*g_logFile, "dim", 10);
        logValue(*g_logFile, "itemsPerThr", 11);
        logValue(*g_logFile, "blockSize", 10);
        logValue(*g_logFile, "inMemLayout", 16);
        logValue(*g_logFile, "outMemLayout", 16);
        logValue(*g_logFile, "blockCount(v1)", 14);
        logValue(*g_logFile, "avgMillis(v1)", 13);
        logValue(*g_logFile, "gigaBytes(v1)", 13);
        logValue(*g_logFile, "gigaFlops(v1)", 13);
        *g_logFile << "\n";
    }
}

struct KernelConf
{
    float time;
    float gflops;
    float gbytes;
    int itemsPerThread;
    int blockSize;
    int blockCount;
};

// Function for selecting best kernel configuration (execution time)
void checkBestConf(
    KernelConf &    bestConf,
    float           initTime,
    float           kernelTime,
    unsigned long long int flops,
    unsigned long long int numBytes,
    int             blockCount,
    int             ITEMS_PER_THREAD,
    int             BLOCK_SIZE,
    KernelResourceUsage resUsage) 
{

    float avgMillis = (kernelTime - initTime) / static_cast<float>(ITER);
    float gigaBandwidth = float(numBytes) / avgMillis / 1e6f;
    float gigaFlops = float(flops) / avgMillis / 1e6f;

    if (g_logPerfResults)
    {
        logValues(*g_logFile, blockCount, avgMillis, gigaBandwidth, gigaFlops, 
            resUsage.prettyPrint());
    }

    logValues(std::cout, blockCount, avgMillis, gigaBandwidth, gigaFlops, 
        resUsage.prettyPrint());

    if (bestConf.time > avgMillis)
    {
        bestConf.time = avgMillis;
        bestConf.gflops = gigaFlops;
        bestConf.gbytes = gigaBandwidth;
        bestConf.itemsPerThread = ITEMS_PER_THREAD;
        bestConf.blockSize = BLOCK_SIZE;
        bestConf.blockCount = blockCount;
    }
};

/***************************************************************************
 *  Dispatch kernel routines
 ***************************************************************************/

template <
    int         DIM, 
    typename    CCSCKernelPtrT,
    typename    T>
float dispatch_and_measure_ccsc_kernel(
    T const *               d_P,
    int                     np,
    T const *               d_S,
    int                     ns,
    T                       r1,
    T *                     d_cordSums,
    int *                   d_spherePointCount,
    dim3                    gridBlocks,
    dim3                    blockThreads,
    CCSCKernelPtrT          ccscKernelPtr,
    int                     pStride,
    int                     sStride,
    int                     csStride)
{
    GpuTimer kernelTimer;
    float kernelTime = 0;

    // warm-up
    if (csStride != DIM)
    {
        checkCudaErrors(cudaMemset2D(d_cordSums, csStride * sizeof(T), 0, 
                ns * sizeof(T), DIM));
    }
    else
    {
        checkCudaErrors(cudaMemset(d_cordSums, 0, ns * DIM * sizeof(T)));
    }
    checkCudaErrors(cudaMemset(d_spherePointCount, 0, ns * sizeof(int)));
    ccscKernelPtr<<<gridBlocks, blockThreads>>>(
        d_P, d_S, d_cordSums, d_spherePointCount, np, ns, r1, pStride,
        csStride, sStride);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
        cudaProfilerStart();
    #endif
    if (csStride != DIM)
    {
        kernelTimer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemset2D(d_cordSums, csStride * sizeof(T), 0, 
                ns * sizeof(T), DIM));
            checkCudaErrors(cudaMemset(d_spherePointCount, 0, ns * sizeof(int)));
            ccscKernelPtr<<<gridBlocks, blockThreads>>>(
                d_P, d_S, d_cordSums, d_spherePointCount, np, ns, r1, pStride, 
                csStride, sStride);
            checkCudaErrors(cudaGetLastError());
        }
        kernelTimer.Stop();
    }
    else
    {
        kernelTimer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemset(d_cordSums, 0, ns * DIM * sizeof(T)));
            checkCudaErrors(cudaMemset(d_spherePointCount, 0, ns * sizeof(int)));
            ccscKernelPtr<<<gridBlocks, blockThreads>>>(
                d_P, d_S, d_cordSums, d_spherePointCount, np, ns, r1, pStride, 
                csStride, sStride);
            checkCudaErrors(cudaGetLastError());
        }
        kernelTimer.Stop();
    }

    kernelTime = kernelTimer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
        cudaProfilerStop();
    #endif

    return kernelTime;
}


template<
    int         DIM, 
    typename    STMCKernelPtrT,
    typename    T>
float dispatch_and_measure_stmc_kernel(
    T *             d_S,
    int             ns,
    T const *       d_cordSums,
    int const *     d_spherePointCount,
    T const *       d_SInitial,
    dim3            gridBlocks,
    dim3            blockThreads,
    STMCKernelPtrT  stmcKernelPtr,
    int             sStride,
    int             csStride,
    int             siStride)
{
    GpuTimer kernelTimer;

    float kernelTime = 0;

    // warm-up
    if (sStride != DIM && siStride != DIM)
    {
        checkCudaErrors(cudaMemcpy2D(d_S, sStride * sizeof(T), d_SInitial, 
            siStride * sizeof(T), ns * sizeof(T), DIM, cudaMemcpyDeviceToDevice));
    }
    else
    {
        checkCudaErrors(cudaMemcpy(d_S, d_SInitial, ns * DIM * sizeof(T), 
            cudaMemcpyDeviceToDevice));
    }
    stmcKernelPtr<<<gridBlocks, blockThreads>>>(d_S, d_cordSums, d_spherePointCount, ns,
        csStride, sStride);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
        cudaProfilerStart();
    #endif
    if (sStride != DIM && siStride != DIM)
    {
        kernelTimer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemcpy2D(d_S, sStride * sizeof(T), d_SInitial, 
                siStride * sizeof(T), ns * sizeof(T), DIM, cudaMemcpyDeviceToDevice));
            stmcKernelPtr<<<gridBlocks, blockThreads>>>(
                d_S, d_cordSums, d_spherePointCount, ns, csStride, sStride);
            checkCudaErrors(cudaGetLastError());
        }
        kernelTimer.Stop();
    }
    else
    {
        kernelTimer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemcpy(d_S, d_SInitial, ns * DIM * sizeof(T), 
                cudaMemcpyDeviceToDevice));
            stmcKernelPtr<<<gridBlocks, blockThreads>>>(
                d_S, d_cordSums, d_spherePointCount, ns, csStride, sStride);
            checkCudaErrors(cudaGetLastError());
        }
        kernelTimer.Stop();
    }
    kernelTime = kernelTimer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
        cudaProfilerStop();
    #endif

    return kernelTime;
}

#if 0
    template <
        int         DIM,
        rd::DataMemoryLayout  IN_MEM_LAYOUT,
        rd::DataMemoryLayout  OUT_MEM_LAYOUT,
        typename    T>
    float dispatch_evolve_kernel(
        T const *       d_P,
        T *             d_S,
        T *             d_cordSums, 
        int *           d_spherePointCount,
        T const *       d_SInitial,
        int             h_ns,
        rd::RDParams<T> const &rdp, 
        int             pStride = 1,
        int             sStride = 1)
    {

        GpuTimer kernelTimer;

        #ifdef RD_PROFILE
            const int ITER = 1;
        #else
            const int ITER = 100;
        #endif
        float kernelTime = 0;

        #ifdef RD_PROFILE
            cudaProfilerStart();
        #endif
        kernelTimer.Start();
        for (int i = -1; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemcpy(d_S, d_SInitial, h_ns * DIM * sizeof(T),
                cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemset(d_cordSums, 0, h_ns * DIM * sizeof(T)));
            checkCudaErrors(cudaMemset(d_spherePointCount, 0, h_ns * sizeof(int)));
            rd::gpu::bruteForce::evolve<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
                d_P,
                d_S,
                d_cordSums,
                d_spherePointCount,
                rdp.np,
                h_ns,
                rdp.r1,
                pStride,
                sStride);
            checkCudaErrors(cudaGetLastError());
        }
        kernelTimer.Stop();
        kernelTime = kernelTimer.ElapsedMillis();
        checkCudaErrors(cudaDeviceSynchronize());
        #ifdef RD_PROFILE
            cudaProfilerStop();
        #endif

        return kernelTime;
    }
#endif // 0

/***************************************************************************
 *  Prepare kernel configuration parameters to benchmark
 ***************************************************************************/

/*******************************************************************************
 *      Evolve whole pass
 *******************************************************************************/

#if 0
    /**
     * @brief      Test whole evolve pass
     *
     * @param      rdp                 Simulation parameters
     * @param[in]  h_ns                Number of chosen samples
     * @param      d_P                 All samples in device memory.
     * @param      d_S                 Chosen samples in device memory.
     * @param      d_cordSums          Device memory ptr to store cord sums
     * @param      d_spherePointCount  Device memory ptr to store sphere point count
     * @param[in]  pStride             Distance between consecutive coordinates in
     *                                 @p d_P table.
     * @param[in]  sStride             Distance between consecutive coordinates in
     *                                 @p d_S table.
     *
     * @tparam     DIM                 Data dimensionality
     * @tparam     IN_MEM_LAYOUT       Input memory layout (data samples and cord
     *                                 sums)
     * @tparam     OUT_MEM_LAYOUT      Output memory layout (chosen samples)
     * @tparam     T                   Samples data type.
     */
    template <
        int         DIM,
        rd::DataMemoryLayout  IN_MEM_LAYOUT,
        rd::DataMemoryLayout  OUT_MEM_LAYOUT,
        typename    T>
    void benchmark_evolve(
        rd::RDParams<T> const &rdp, 
        int h_ns,
        T const * d_P,
        T * d_S,
        T * d_cordSums,
        int * d_spherePointCount,
        T const * d_SInitial,
        std::ofstream *dstFile = nullptr,
        int pStride = 1,
        int sStride = 1)
    {

        std::cout << "--------__evolve_kernel__--------" << std::endl;

        if (rdp.verbose)
            *dstFile << "% __evolve_kernel__" << std::endl;

        GpuTimer timer;
        float initDataTime = 0;

        #ifdef RD_PROFILE
            const int ITER = 1;
        #else
            const int ITER = 100;
        #endif

        // measure memset time
        timer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemcpy(d_S, d_SInitial, h_ns * DIM * sizeof(T),
                cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemset(d_cordSums, 0, h_ns * DIM * sizeof(T)));
            checkCudaErrors(cudaMemset(d_spherePointCount, 0, h_ns * sizeof(int)));
        }
        timer.Stop();
        initDataTime = timer.ElapsedMillis();

        float kernelTime = dispatch_evolve_kernel<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
                d_P, d_S, d_cordSums, d_spherePointCount, d_SInitial, h_ns, rdp, pStride, sStride);

        kernelTime = (kernelTime - initDataTime) / static_cast<float>(ITER);

        if (rdp.verbose)
        {
            *dstFile << "% (whole) evolve kernel time: " << std::endl;
            *dstFile << kernelTime << std::endl;
        }

        std::cout << "% (whole) evolve kernel time: " << std::endl;
        std::cout << kernelTime << std::endl;

    }
#endif // 0
 
/*******************************************************************************
 *      Benchmark configuration and data initialization
 *******************************************************************************/

template <
    int         ITEMS_PER_THREAD,
    int         BLOCK_SIZE,
    int         DIM,
    rd::DataMemoryLayout  IN_MEM_LAYOUT,
    rd::DataMemoryLayout  OUT_MEM_LAYOUT,
    typename    T>
void benchmarkAlg_ccsc(
    T const *   d_P, 
    int         np, 
    T *         d_S, 
    int         ns, 
    T           r1,
    T *         d_cordSums, 
    int *       d_spherePointCount, 
    int         pStride, 
    int         sStride,
    int         csStride,
    float       memsetTime,
    KernelConf &  bestConf_v1)
{
    typedef rd::gpu::bruteForce::AgentClosestSpherePolicy<
            BLOCK_SIZE,
            ITEMS_PER_THREAD>
        v1_PolicyT;

    auto v1_kernelPtr = rd::gpu::bruteForce::detail::DeviceClosestSphereKernel<
        v1_PolicyT, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, T>;
    KernelResourceUsage v1_resUsage(v1_kernelPtr);

    unsigned long long int ns2 = ns;
    unsigned long long int np2 = np;

    unsigned long long int numElements = ns2 * np2;
    // calculate euclideanSqrDist for each pair of point x chosenPoint
    // update sphere counters
    // update cordSums (read samples + write cordSums)
    unsigned long long int flops = numElements * DIM * 3 + np2 + np2 * DIM;
    // unsigned long long int numBytes = numElements * DIM * sizeof(T) / ITEMS_PER_THREAD / BLOCK_SIZE +
    unsigned long long int numBytes = numElements * DIM * sizeof(T) + np2 * sizeof(int) + 
        np2 * DIM * sizeof(T);

    dim3 gridSize, blockThreads;
    float kernelTime = 0;
    int blockCount;

    int ptxVersion = 0;
    checkCudaErrors(cub::PtxVersion(ptxVersion));

    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, g_devId));

    // Get SM count
    int smCount = devProp.multiProcessorCount;

    // log results
    if (g_logPerfResults)
    {
        logValues(*g_logFile, DIM, ITEMS_PER_THREAD,  BLOCK_SIZE, 
            std::string(rd::DataMemoryLayoutNameTraits<IN_MEM_LAYOUT>::shortName),
            std::string(rd::DataMemoryLayoutNameTraits<OUT_MEM_LAYOUT>::shortName));
    }
    logValues(std::cout, DIM, ITEMS_PER_THREAD,  BLOCK_SIZE, 
            std::string(rd::DataMemoryLayoutNameTraits<IN_MEM_LAYOUT>::shortName),
            std::string(rd::DataMemoryLayoutNameTraits<OUT_MEM_LAYOUT>::shortName));

    //-----------------------------------------------------------------------------
    // V1
    //-----------------------------------------------------------------------------

    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blockCount, v1_kernelPtr, BLOCK_SIZE, 0));

    gridSize = dim3(blockCount * smCount * CUB_SUBSCRIPTION_FACTOR(ptxVersion));
    blockThreads = dim3(BLOCK_SIZE);

    kernelTime = dispatch_and_measure_ccsc_kernel<DIM>(
        d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, gridSize, blockThreads, 
        v1_kernelPtr, pStride, sStride, csStride);
    checkBestConf(bestConf_v1, memsetTime, kernelTime, flops, numBytes, gridSize.x,
        ITEMS_PER_THREAD, BLOCK_SIZE, v1_resUsage);

    if (g_logPerfResults)
    {
        *g_logFile << "\n";
    }
    std::cout << std::endl;
}

template <
    int         ITEMS_PER_THREAD,
    int         BLOCK_SIZE,
    int         DIM,
    rd::DataMemoryLayout  IN_MEM_LAYOUT,
    rd::DataMemoryLayout  OUT_MEM_LAYOUT,
    typename    T>
void benchmarkAlg_stmc(
    T *         d_S, 
    T const *   d_SInitial, 
    int         ns, 
    T const *   d_cordSums, 
    int const * d_spherePointCount, 
    int         sStride,
    int         csStride, 
    int         siStride,
    float       memcpyTime,
    KernelConf &  bestConf_v1)
{
    typedef rd::gpu::bruteForce::AgentShiftSpherePolicy<
            BLOCK_SIZE,
            ITEMS_PER_THREAD>
        v1_PolicyT;

    auto v1_kernelPtr = rd::gpu::bruteForce::detail::DeviceShiftSphereKernel<
        v1_PolicyT, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, T>;
    KernelResourceUsage v1_resUsage(v1_kernelPtr);

    unsigned long long int numElements = ns;
    // calculate mass center for each sphere
    unsigned long long int flops = numElements * DIM;
    // calculate mass center for each sphere (read 2x)
    unsigned long long int numBytes = numElements * DIM * 2ull * sizeof(T) + 
        // shift condition evaluation (read 2x)
        numElements * DIM * 2ull * sizeof(int) + 
        // update chosen sample coordinates
        numElements * DIM * sizeof(T);

    dim3 gridSize, blockThreads;
    float kernelTime = 0;
    int blockCount;

    int ptxVersion = 0;
    checkCudaErrors(cub::PtxVersion(ptxVersion));

    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, g_devId));

    // Get SM count
    int smCount = devProp.multiProcessorCount;

    // log results

    if (g_logPerfResults)
    {
        logValues(*g_logFile, DIM, ITEMS_PER_THREAD,  BLOCK_SIZE, 
            std::string(rd::DataMemoryLayoutNameTraits<IN_MEM_LAYOUT>::shortName),
            std::string(rd::DataMemoryLayoutNameTraits<OUT_MEM_LAYOUT>::shortName));
    }
    logValues(std::cout, DIM, ITEMS_PER_THREAD,  BLOCK_SIZE, 
            std::string(rd::DataMemoryLayoutNameTraits<IN_MEM_LAYOUT>::shortName),
            std::string(rd::DataMemoryLayoutNameTraits<OUT_MEM_LAYOUT>::shortName));

    //-----------------------------------------------------------------------------
    // V1
    //-----------------------------------------------------------------------------

    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blockCount, v1_kernelPtr, BLOCK_SIZE, 0));

    gridSize = dim3(blockCount * smCount * CUB_SUBSCRIPTION_FACTOR(ptxVersion));
    blockThreads = dim3(BLOCK_SIZE);

    kernelTime = dispatch_and_measure_stmc_kernel<DIM>(
        d_S, ns, d_cordSums, d_spherePointCount, d_SInitial, gridSize, blockThreads, 
        v1_kernelPtr, sStride, csStride, siStride);
    checkBestConf(bestConf_v1, memcpyTime, kernelTime, flops, numBytes, gridSize.x,
        ITEMS_PER_THREAD, BLOCK_SIZE, v1_resUsage);

    if (g_logPerfResults)
    {
        *g_logFile << "\n";
    }
    std::cout << std::endl;

}

template <
    int         BLOCK_SIZE,
    int         DIM,
    rd::DataMemoryLayout  IN_MEM_LAYOUT,
    rd::DataMemoryLayout  OUT_MEM_LAYOUT,
    typename    T>
void benchmark_ccsc(
    T const *   d_P, 
    int         np, 
    T *         d_S, 
    int         ns, 
    T           r1,
    T *         d_cordSums, 
    int *       d_spherePointCount, 
    int         pStride, 
    int         sStride,
    int         csStride,
    float       memsetTime,
    KernelConf &  bestConf_v1)
{
    // iterate over items per thread
    #ifdef QUICK_TEST
        benchmarkAlg_ccsc<3, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
    #else
        benchmarkAlg_ccsc<1, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        benchmarkAlg_ccsc<2, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        benchmarkAlg_ccsc<3, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        benchmarkAlg_ccsc<4, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        benchmarkAlg_ccsc<5, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        benchmarkAlg_ccsc<6, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        benchmarkAlg_ccsc<7, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        benchmarkAlg_ccsc<8, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        benchmarkAlg_ccsc<9, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        benchmarkAlg_ccsc<10, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        benchmarkAlg_ccsc<11, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        benchmarkAlg_ccsc<12, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
            d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        // benchmarkAlg_ccsc<13, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
        //     d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        // benchmarkAlg_ccsc<14, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
        //     d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        // benchmarkAlg_ccsc<15, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
        //     d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
        // benchmarkAlg_ccsc<16, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, 
        //     d_cordSums, d_spherePointCount, pStride, sStride, csStride, memsetTime, bestConf_v1);
    #endif
}

template <
    int         BLOCK_SIZE,
    int         DIM,
    rd::DataMemoryLayout  IN_MEM_LAYOUT,
    rd::DataMemoryLayout  OUT_MEM_LAYOUT,
    typename    T>
void benchmark_stmc(
    T *         d_S, 
    T const *   d_SInitial, 
    int         ns, 
    T const *   d_cordSums, 
    int const * d_spherePointCount, 
    int         sStride,
    int         csStride,
    int         siStride,
    float       memcpyTime,
    KernelConf &  bestConf_v1)
{
    // iterate over items per thread
    #ifdef QUICK_TEST
        benchmarkAlg_stmc<3, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
    #else
        benchmarkAlg_stmc<1, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
        benchmarkAlg_stmc<2, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
        benchmarkAlg_stmc<3, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
        benchmarkAlg_stmc<4, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
        benchmarkAlg_stmc<5, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
        benchmarkAlg_stmc<6, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
        benchmarkAlg_stmc<7, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
        benchmarkAlg_stmc<8, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
        benchmarkAlg_stmc<9, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
        benchmarkAlg_stmc<10, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
        benchmarkAlg_stmc<11, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
        benchmarkAlg_stmc<12, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, 
            d_cordSums, d_spherePointCount, sStride, csStride, siStride, memcpyTime, bestConf_v1);
    #endif
}

template <
    int         DIM,
    rd::DataMemoryLayout  IN_MEM_LAYOUT,
    rd::DataMemoryLayout  OUT_MEM_LAYOUT,
    typename    T>
void benchmark_ccsc(
    T const *   d_P, 
    int         np, 
    T *         d_S, 
    int         ns, 
    T           r1,
    T *         d_cordSums, 
    int *       d_spherePointCount, 
    int         pStride, 
    int         sStride,
    int         csStride)
{
    std::cout << "\n%//-----------------------------------------------\n"
        << "%// benchmark_ccsc\n" 
        << "%//-----------------------------------------------\n" << std::endl;

    if (g_logPerfResults)
    {
        *g_logFile << "\n%//-----------------------------------------------\n"
              << "%// benchmark_ccsc\n" 
              << "%//-----------------------------------------------\n" << std::endl;
    }

    GpuTimer memsetTimer;
    float memsetTime = 0;

    // measure memset time
    if (IN_MEM_LAYOUT == rd::ROW_MAJOR)
    {
        memsetTimer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemset(d_cordSums, 0, ns * DIM * sizeof(T)));
            checkCudaErrors(cudaMemset(d_spherePointCount, 0, ns * sizeof(int)));
        }
        memsetTimer.Stop();
    }
    else
    {
        memsetTimer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemset2D(d_cordSums, csStride * sizeof(T), 0, 
                ns * sizeof(T), DIM));
            checkCudaErrors(cudaMemset(d_spherePointCount, 0, ns * sizeof(int)));
        }
        memsetTimer.Stop();
    }
    memsetTime = memsetTimer.ElapsedMillis();

    KernelConf bestConf_v1;
    bestConf_v1.time = std::numeric_limits<T>::max();

    benchmark_ccsc<RD_BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
        d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, csStride,
        memsetTime, bestConf_v1);

    if (g_logPerfResults)
    {
        *g_logFile << "\n%################################################\n";
        *g_logFile << "% ver, time, gflops, gbytes, itemsPerThread, blockSize, blockCount\n";
        logValues(*g_logFile, "%v1", bestConf_v1.time, bestConf_v1.gflops, bestConf_v1.gbytes, 
            bestConf_v1.itemsPerThread, bestConf_v1.blockSize, bestConf_v1.blockCount);
        *g_logFile << "\n";
        *g_logFile << "%################################################\n\n";
    }

    std::cout << "\n%################################################\n";
    std::cout << "% ver, time, gflops, gbytes, itemsPerThread, blockSize, blockCount\n";
    logValues(std::cout, "%v1", bestConf_v1.time, bestConf_v1.gflops, bestConf_v1.gbytes, 
        bestConf_v1.itemsPerThread, bestConf_v1.blockSize, bestConf_v1.blockCount);
    std::cout << "\n";
    std::cout << "%################################################\n" << std::endl;
}


template <
    int         DIM,
    rd::DataMemoryLayout  IN_MEM_LAYOUT,
    rd::DataMemoryLayout  OUT_MEM_LAYOUT,
    typename    T>
void benchmark_stmc(
    T *         d_S, 
    T const *   d_SInitial, 
    int         ns, 
    T const *   d_cordSums, 
    int const * d_spherePointCount, 
    int         sStride,
    int         csStride,
    int         siStride)
{
    std::cout << "\n//-----------------------------------------------\n"
              << "// benchmark_stmc\n" 
              << "//-----------------------------------------------\n" << std::endl;

    if (g_logPerfResults)
    {
        *g_logFile << "\n%//-----------------------------------------------\n"
              << "%// benchmark_stmc\n" 
              << "%//-----------------------------------------------\n" << std::endl;
    }

    GpuTimer memcpyTimer;
    float memcpyTime = 0;

    // measure memcpy time
    if (OUT_MEM_LAYOUT == rd::ROW_MAJOR)
    {
        memcpyTimer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemcpy(d_S, d_SInitial, ns * DIM * sizeof(T), 
                cudaMemcpyDeviceToDevice));
        }
        memcpyTimer.Stop();
    }
    else
    {
        memcpyTimer.Start();
        for (int i = 0; i < ITER; ++i)
        {
            checkCudaErrors(cudaMemcpy2D(d_S, sStride * sizeof(T), d_SInitial, 
                siStride * sizeof(T), ns * sizeof(T), DIM, cudaMemcpyDeviceToDevice));
        }
        memcpyTimer.Stop();
    }
    memcpyTime = memcpyTimer.ElapsedMillis();

    KernelConf bestConf_v1;
    bestConf_v1.time = std::numeric_limits<T>::max();

    benchmark_stmc<RD_BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
        d_S, d_SInitial, ns, d_cordSums, d_spherePointCount, sStride, csStride, siStride, 
        memcpyTime, bestConf_v1);

    if (g_logPerfResults)
    {
        *g_logFile << "\n%################################################\n";
        *g_logFile << "% ver, time, gflops, gbytes, itemsPerThread, blockSize, blockCount\n";
        logValues(*g_logFile, "%v1", bestConf_v1.time, bestConf_v1.gflops, bestConf_v1.gbytes, 
            bestConf_v1.itemsPerThread, bestConf_v1.blockSize, bestConf_v1.blockCount);
        *g_logFile << "\n";
        *g_logFile << "%################################################\n\n";
    }

    std::cout << "\n%################################################\n";
    std::cout << "% ver, time, gflops, gbytes, itemsPerThread, blockSize, blockCount\n";
    logValues(std::cout, "%v1", bestConf_v1.time, bestConf_v1.gflops, bestConf_v1.gbytes, 
        bestConf_v1.itemsPerThread, bestConf_v1.blockSize, bestConf_v1.blockCount);
    std::cout << "\n";
    std::cout << "%################################################\n" << std::endl;
}

template <
    int DIM,
    rd::DataMemoryLayout  IN_MEM_LAYOUT,
    rd::DataMemoryLayout  OUT_MEM_LAYOUT,
    typename T>
void benchmark_individual_kernels(
    RdData<T> const & dataPack)                   // row-major
{
    T *d_S, *d_SInitial, *d_P, *d_cordSums;
    int *d_spherePointCount;
    int pStride = DIM, sStride = DIM, csStride = DIM, siStride = DIM;
    size_t pPitch = 1, sPitch = 1, csPitch = 1, siPitch = 1;
        
    // allocate needed memory
    if (IN_MEM_LAYOUT == rd::COL_MAJOR)
    {
        checkCudaErrors(cudaMallocPitch(&d_P, &pPitch, dataPack.np * sizeof(T), DIM));
        checkCudaErrors(cudaMallocPitch(&d_cordSums, &csPitch, dataPack.ns * sizeof(T), DIM));
    }
    else 
    {
        checkCudaErrors(cudaMalloc(&d_P, dataPack.np * DIM * sizeof(T)));
        checkCudaErrors(cudaMalloc(&d_cordSums, dataPack.ns * DIM * sizeof(T)));
    }

    if (OUT_MEM_LAYOUT == rd::COL_MAJOR)
    {
        checkCudaErrors(cudaMallocPitch(&d_S, &sPitch, dataPack.ns * sizeof(T), DIM));
        checkCudaErrors(cudaMallocPitch(&d_SInitial, &siPitch, dataPack.ns * sizeof(T), DIM));
    }
    else
    {
        checkCudaErrors(cudaMalloc(&d_S, dataPack.ns * DIM * sizeof(T)));
        checkCudaErrors(cudaMalloc(&d_SInitial, dataPack.ns * DIM * sizeof(T)));
    }
    checkCudaErrors(cudaMalloc((void**)&d_spherePointCount, dataPack.ns * sizeof(int)));
    
    /************************************************************************
     *      GPU VERSION respective kernels
     ************************************************************************/

    // prepare data in appropriate memory layout
    if (IN_MEM_LAYOUT == rd::COL_MAJOR)
    {
        pStride = pPitch / sizeof(T);
        csStride = csPitch / sizeof(T);

        rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_P, dataPack.P, DIM, dataPack.np, pPitch, DIM * sizeof(T));
    }
    else
    {
        rd::gpu::rdMemcpy<rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_P, dataPack.P, DIM, dataPack.np, pStride, DIM);
    }

    if (OUT_MEM_LAYOUT == rd::COL_MAJOR)
    {
        sStride = sPitch / sizeof(T);
        siStride = siPitch / sizeof(T);
        rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_S, dataPack.S.data(), DIM, dataPack.ns, sPitch, DIM * sizeof(T));
    }
    else
    {
        rd::gpu::rdMemcpy<rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_S, dataPack.S.data(), DIM, dataPack.ns, sStride, DIM);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    #ifdef RD_CCSC
    benchmark_ccsc<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
        d_P, dataPack.np, d_S, dataPack.ns, dataPack.r1, d_cordSums, 
        d_spherePointCount, pStride, sStride, csStride);
    #endif


    if (OUT_MEM_LAYOUT == rd::COL_MAJOR)
    {
        rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::COL_MAJOR, cudaMemcpyDeviceToDevice>(
            d_SInitial, d_S, dataPack.ns, DIM, siPitch, sPitch);
    }
    else
    {
        rd::gpu::rdMemcpy<rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyDeviceToDevice>(
            d_SInitial, d_S, DIM, dataPack.ns, siStride, sStride);
    }

    #ifdef RD_STMC
    benchmark_stmc<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
        d_S, d_SInitial, dataPack.ns, d_cordSums, d_spherePointCount, sStride, 
        csStride, siStride);
    #endif

    // clean-up
    checkCudaErrors(cudaFree(d_S));
    checkCudaErrors(cudaFree(d_SInitial));
    checkCudaErrors(cudaFree(d_P));
    checkCudaErrors(cudaFree(d_cordSums));
    checkCudaErrors(cudaFree(d_spherePointCount));
}

/**
 * @brief      Invoke benchmarks for individual kernels.
 *
 * @tparam     DIM                 Point dimension
 * @tparam     T                   Samples data type.
 */
template <
    int                     DIM,
    rd::DataMemoryLayout    IN_MEM_LAYOUT,
    rd::DataMemoryLayout    OUT_MEM_LAYOUT,
    typename                T>
void test(RdData<T> const & dataPack)
{
    benchmark_individual_kernels<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(dataPack);
    // benchmark_evolve<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(dataPack);
}

template <
    int         DIM,
    typename    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testMemLayout(
    int                             pointCnt,
    PointCloud<T> const &           pc,
    T                               r1,
    std::vector<T> &&               points)
{
    RdData<T> dataPack;
    dataPack.np = pointCnt;
    dataPack.r1 = r1;
    dataPack.r2 = 0;
    // dataPack.r1 = 5.f;
    dataPack.s = pc.stddev_;
    dataPack.P = points.data();
    pc.getCloudParameters(dataPack.a, dataPack.b);

    //---------------------------------------------------
    //               CHOOSE 
    //---------------------------------------------------

    dataPack.S.resize(pointCnt*DIM);

    {
        // std::list<T*> csList;
        // rd::CpuTimer chooseTimer;
        // chooseTimer.start();
        // rd::choose(dataPack.P, dataPack.S.data(), csList, pointCnt, dataPack.ns,
        //     size_t(DIM), r1);
        // chooseTimer.stop();
        // std::cout << "(cpu) choose time: " << chooseTimer.elapsedMillis(0) << "ms" << std::endl;
        
        T *d_inP, *d_outS;
        size_t pPitch;
        int * d_ns;
        int zero = 0;

        checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
        checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &zero, sizeof(int)));
        checkCudaErrors(cudaMallocPitch(&d_inP, &pPitch, pointCnt * sizeof(T), DIM));
        checkCudaErrors(cudaMalloc(&d_outS, pointCnt * DIM * sizeof(T)));

        rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_inP, dataPack.P, DIM, pointCnt, pPitch, DIM * sizeof(T));
        cudaError_t err = rd::gpu::bruteForce::DeviceChoose::
            setCacheConfig<DIM, rd::COL_MAJOR, rd::ROW_MAJOR, T>();
        checkCudaErrors(err);
        checkCudaErrors(cudaDeviceSynchronize());

        err = rd::gpu::bruteForce::DeviceChoose::choose<
            DIM, rd::COL_MAJOR, rd::ROW_MAJOR>(
            d_inP, d_outS, pointCnt, d_ns, r1, pPitch / sizeof(T), DIM);
        checkCudaErrors(err);
        checkCudaErrors(cudaMemcpyFromSymbol(&dataPack.ns, rd::gpu::rdBruteForceNs, sizeof(int)));
        rd::gpu::rdMemcpy2D<rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyDeviceToHost>(
            dataPack.S.data(), d_outS, DIM, dataPack.ns, DIM * sizeof(T), DIM * sizeof(T));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // release unused memory
    dataPack.S.resize(dataPack.ns * DIM);
    // checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &dataPack.ns, sizeof(int)));

    std::cout << "Chosen count: " << dataPack.ns << std::endl;

    if (g_logPerfResults)
    {
        *g_logFile << "% Chosen count: " << dataPack.ns << std::endl;
    }

    //---------------------------------------------------
    //          TEST EVOLVE
    //---------------------------------------------------

    std::cout << rd::HLINE << std::endl;
    std::cout << ">>>> ROW_MAJOR - ROW_MAJOR" << std::endl;
    test<DIM, rd::ROW_MAJOR, rd::ROW_MAJOR>(dataPack);

    // #ifndef QUICK_TEST
    std::cout << rd::HLINE << std::endl;
    std::cout << ">>>> COL_MAJOR - ROW_MAJOR" << std::endl;
    test<DIM, rd::COL_MAJOR, rd::ROW_MAJOR>(dataPack);

    std::cout << rd::HLINE << std::endl;
    std::cout << ">>>> COL_MAJOR - COL_MAJOR" << std::endl;
    test<DIM, rd::COL_MAJOR, rd::COL_MAJOR>(dataPack);
    // #endif
}

/**
 * @brief helper structure for static for loop over dimension
 */
struct IterateDimensions
{
    template <typename D, typename T>
    static void impl(
        D   idx,
        int pointCnt,
        T r1,
        PointCloud<T> const & pc)
    {
        std::cout << rd::HLINE << std::endl;
        std::cout << ">>>> Dimension: " << idx << "D\n";

        if (g_logPerfResults)
        {
            T a, b;
            pc.getCloudParameters(a, b);
            *g_logFile << "%>>>> Dimension: " << idx << "D\n"
                << "% a: " << a << " b: " << b << " s: " << pc.stddev_ 
                << " pointsNum: " << pointCnt << " r1: " << r1 << "\n";
        }

        testMemLayout<D::value>(pointCnt, pc, r1, pc.extractPart(pointCnt, idx));
    }
};

/**
 * @brief Test evolve time relative to point dimension
 */
template <
    int          DIM,
    typename     T>
struct TestDimensions
{
    static void impl(
        PointCloud<T> & pc,
        T r1, 
        int pointCnt)
    {
        static_assert(DIM != 0, "DIM equal to zero!\n");

        initializeLogFile<T>();
        pc.pointCnt_ = pointCnt;
        pc.initializeData();

        std::cout << rd::HLINE << std::endl;
        std::cout << ">>>> Dimension: " << DIM << "D\n";
        
        if (g_logPerfResults)
        {
            T a, b;
            pc.getCloudParameters(a, b);
            *g_logFile << "%>>>> Dimension: " << DIM << "D\n"
                << "% a: " << a << " b: " << b << " s: " << pc.stddev_ 
                << " pointsNum: " << pointCnt << " r1: " << r1 << "\n";
        }

        testMemLayout<DIM>(pointCnt, pc, r1, pc.extractPart(pointCnt, DIM));
        
        // clean-up
        if (g_logPerfResults)
        {
            g_logFile->close();
            delete g_logFile;
        }
    }
};

template <typename T>
struct TestDimensions<0, T>
{
    static void impl(
        PointCloud<T> & pc,
        T r1,
        int pointCnt)
    {
        initializeLogFile<T>();
        pc.pointCnt_ = pointCnt;
        pc.dim_ = MAX_TEST_DIM;
        pc.initializeData();

        StaticFor<1, MAX_TEST_DIM, IterateDimensions>::impl(pointCnt, r1, pc);

        // clean-up
        if (g_logPerfResults)
        {
            g_logFile->close();
            delete g_logFile;
        }
    }
};

//------------------------------------------------------------
//  MAIN
//------------------------------------------------------------

int main(int argc, char const **argv)
{

    float a = -1.f, b = -1.f, stddev = -1.f;
    float r1 = 0.f;
    int pointCnt = -1;
    //-----------------------------------------------------------------

    // Initialize command line
    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help") && args.ParsedArgc() < 6) 
    {
        printf("%s \n"
            "\t\t[--np=<P size>]\n"
            "\t\t[--r1=<r1 param>]\n"
            "\t\t[--a=<spiral param>]\n"
            "\t\t[--b=<spiral param>]\n"
            "\t\t[--s=<spiral noise sigma>]\n"
            "\t\t[--d=<device id>]\n"
            "\t\t[--log <log performance to file>]\n"
            "\n", argv[0]);
        exit(0);
    }

    args.GetCmdLineArgument("r1", r1);

    args.GetCmdLineArgument("np", pointCnt);

    if (args.CheckCmdLineFlag("a")) 
    {
        args.GetCmdLineArgument("a", a);
    }
    if (args.CheckCmdLineFlag("b")) 
    {
        args.GetCmdLineArgument("b", b);
    }
    if (args.CheckCmdLineFlag("s")) 
    {
        args.GetCmdLineArgument("s", stddev);
    }
    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", g_devId);
    }
    if (args.CheckCmdLineFlag("log")) 
    {
        g_logPerfResults = true;
    }

    checkCudaErrors(deviceInit(g_devId));

    // set device name for logging and drawing purposes
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, g_devId));
    g_devName = devProp.name;

    if (pointCnt    < 0 ||
        a           < 0 ||
        b           < 0 ||
        stddev      < 0)
    {
        std::cout << "Have to specify parameters! Rerun with --help for help.\n";
        exit(1);
    }
    #ifdef QUICK_TEST
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (spiral) float: "  
                    << "\n//------------------------------------------\n";

        const int dim = 6;

        // PointCloud<float> && fpc = SpiralPointCloud<float>(a, b, pointCnt, dim, stddev);
        PointCloud<float> && fpc = SegmentPointCloud<float>(a, pointCnt, dim, stddev);
        TestDimensions<dim, float>::impl(fpc, r1, pointCnt);

        // initializeLogFile<float>();
        // fpc.pointCnt_ = pointCnt;
        // fpc.dim_ = dim;
        // fpc.initializeData();

        // RdData<float> dataPack;
        // dataPack.np = pointCnt;
        // dataPack.r1 = 5.65f;
        // dataPack.r2 = 0.f;
        // dataPack.s = fpc.stddev_;
        // auto vpoints = fpc.extractPart(pointCnt, dim);
        // dataPack.P = vpoints.data();
        // fpc.getCloudParameters(dataPack.a, dataPack.b);

        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t (spiral) double: "  
        //             << "\n//------------------------------------------\n";
        // PointCloud<double> && dpc = SpiralPointCloud<double>(a, b, pointCnt, dim, stddev);
        // TestDimensions<dim, double>::impl(dpc, r1, pointCnt);

    #else
        // 1e6 2D points, spiral a=22, b=10, stddev=4
        // PointCloud<float> && fpc2d = SpiralPointCloud<float>(a, b, 0, 2, stddev);
        // PointCloud<float> && fpc3d = SpiralPointCloud<float>(a, b, 0, 3, stddev);
        // PointCloud<double> && dpc2d = SpiralPointCloud<double>(a, b, 0, 2, stddev);
        // PointCloud<double> && dpc3d = SpiralPointCloud<double>(a, b, 0, 3, stddev);

        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t (spiral) float: "  
        //             << "\n//------------------------------------------\n";
        // TestDimensions<2, float>::impl(fpc2d, r1, int(pointCnt));
        // TestDimensions<3, float>::impl(fpc3d, r1, int(pointCnt));

        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t (spiral) double: "  
        //             << "\n//------------------------------------------\n";
        // TestDimensions<2, double>::impl(dpc2d, r1, int(pointCnt));
        // TestDimensions<3, double>::impl(dpc3d, r1, int(pointCnt));
        

        #ifndef RD_DOUBLE_PRECISION
        const int dim = 0;
        PointCloud<float> && fpc = SegmentPointCloud<float>(a, pointCnt, dim, stddev);
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) float: "  
                    << "\n//------------------------------------------\n";
        TestDimensions<dim, float>::impl(fpc, r1, pointCnt);            

        #else
        PointCloud<double> && dpc = SegmentPointCloud<double>(a, pointCnt, 0, stddev);
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) double: "  
                    << "\n//------------------------------------------\n";
        TestDimensions<0, double>::impl(dpc, r1, pointCnt);
        #endif

    #endif

    checkCudaErrors(cudaDeviceReset());

    std::cout << rd::HLINE << std::endl;
    std::cout << "END!" << std::endl;

    return EXIT_SUCCESS;
}

