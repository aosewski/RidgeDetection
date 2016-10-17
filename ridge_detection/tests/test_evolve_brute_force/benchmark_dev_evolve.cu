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

// needed for name traits..
// FIXME: remove this!
#define BLOCK_TILE_LOAD_V4 1

#if defined(RD_DEBUG) && !defined(CUB_STDERR)
#define CUB_STDERR 
#endif

#include <helper_cuda.h>

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <limits>
#include <fstream>
#include <iomanip>

#include "rd/gpu/device/brute_force/evolve.cuh"
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

/*
 *  Global variables 
 */

static const std::string LOG_FILE_NAME_SUFFIX   = "gpu_evolve-timings.txt";
static constexpr int MAX_TEST_DIM               = 3;
static constexpr int MAX_POINTS_NUM             = int(1e7f);


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
        logFileName << getCurrDateAndTime() << "_" <<
            g_devName << "_" << LOG_FILE_NAME_SUFFIX;

        std::string logFilePath = rd::findPath("timings/", logFileName.str());
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
        logValue(*g_logFile, "inPointsNum", 11);
        logValue(*g_logFile, "chosenPtsNum", 12);
        logValue(*g_logFile, "inMemLayout", 16);
        logValue(*g_logFile, "outMemLayout", 16);
        logValue(*g_logFile, "blockSize", 10);
        logValue(*g_logFile, "itemsPerThr", 11);
        logValue(*g_logFile, "version", 10);
        logValue(*g_logFile, "blockCount", 10);
        logValue(*g_logFile, "avgMillis", 10);
        logValue(*g_logFile, "gigaBytes", 10);
        logValue(*g_logFile, "gigaFlops", 10);
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
    int                     pStride = 1,
    int                     sStride = 1)
{
    GpuTimer kernelTimer;
    float kernelTime = 0;

    // warm-up
    checkCudaErrors(cudaMemset(d_cordSums, 0, ns * DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_spherePointCount, 0, ns * sizeof(int)));
    ccscKernelPtr<<<gridBlocks, blockThreads>>>(
        d_P, d_S, d_cordSums, d_spherePointCount, np, ns, r1, pStride, sStride);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
        cudaProfilerStart();
    #endif
    kernelTimer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemset(d_cordSums, 0, ns * DIM * sizeof(T)));
        checkCudaErrors(cudaMemset(d_spherePointCount, 0, ns * sizeof(int)));
        ccscKernelPtr<<<gridBlocks, blockThreads>>>(
            d_P, d_S, d_cordSums, d_spherePointCount, np, ns, r1, pStride, sStride);
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
    int             sStride = 1)
{
    GpuTimer kernelTimer;

    float kernelTime = 0;

    // warm-up
    checkCudaErrors(cudaMemcpy(d_S, d_SInitial, ns * DIM * sizeof(T), cudaMemcpyDeviceToDevice));
    stmcKernelPtr<<<gridBlocks, blockThreads>>>(
        d_S, d_cordSums, d_spherePointCount, ns, sStride);
    checkCudaErrors(cudaGetLastError());

    #ifdef RD_PROFILE
        cudaProfilerStart();
    #endif
    kernelTimer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpy(d_S, d_SInitial, ns * DIM * sizeof(T), 
            cudaMemcpyDeviceToDevice));
        stmcKernelPtr<<<gridBlocks, blockThreads>>>(
            d_S, d_cordSums, d_spherePointCount, ns, sStride);
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

// Function for selecting best kernel configuration (execution time)
void checkBestConf(
    KernelConf &    bestConf,
    float           initTime,
    float           kernelTime,
    size_t          flops,
    size_t          numBytes,
    int             ITEMS_PER_THREAD,
    int             BLOCK_SIZE,
    int             blockCount,
    int             DIM,
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    int             np,
    int             ns,
    std::string     version) 
{
    float avgMillis = (kernelTime - initTime) / static_cast<float>(ITER);
    float gigaBandwidth = float(numBytes) / avgMillis / 1e6f;
    float gigaFlops = float(flops) / avgMillis / 1e6f;

    if (g_logPerfResults)
    {
        logValues(*g_logFile, DIM, np, ns, rd::getRDDataMemoryLayout(IN_MEM_LAYOUT),
            rd::getRDDataMemoryLayout(OUT_MEM_LAYOUT), BLOCK_SIZE, ITEMS_PER_THREAD, 
            version, blockCount, avgMillis, gigaBandwidth, gigaFlops);
        *g_logFile << "\n";
    }

    std::cout << " " << std::setw(2) << ITEMS_PER_THREAD 
    << " " << std::setw(4) << BLOCK_SIZE 
    << " " << std::setw(4) << blockCount 
    << " " << std::setw(8) << std::fixed << std::right << std::setprecision(3) << avgMillis << "(ms)"
    << std::setw(8) << std::fixed << std::right << std::setprecision(3) << gigaBandwidth << "(GB/s)"
    << std::setw(8) << std::fixed << std::right << std::setprecision(3) << gigaFlops << "(GFlop/s)" 
    << std::endl;

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
    float       memsetTime,
    KernelConf &  bestConf_v1,
    KernelConf &  bestConf_v2_CUB,
    KernelConf &  bestConf_v2_trove)
{
    typedef rd::gpu::bruteForce::AgentClosestSpherePolicy<
            BLOCK_SIZE,
            ITEMS_PER_THREAD>
        v1_PolicyT;

    auto v1_kernelPtr = rd::gpu::bruteForce::detail::DeviceClosestSphereKernel<
        v1_PolicyT, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, T>;

    size_t numElements = ns * np;
    // calculate euclideanSqrDist for each pair of point x chosenPoint
    // update sphere counters
    // update cordSums (read samples + write cordSums)
    size_t flops = numElements * (DIM * 3) + np + np * DIM;
    size_t numBytes = numElements * DIM * sizeof(T) + np * sizeof(int) + np * 2 * DIM * sizeof(T);

    dim3 gridSize, blockThreads;
    float kernelTime = 0;
    int blockCount;

    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, g_devId));

    // Get SM count
    int smCount = devProp.multiProcessorCount;

    //-----------------------------------------------------------------------------
    // V1
    //-----------------------------------------------------------------------------

    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blockCount, v1_kernelPtr, BLOCK_SIZE, 0));

    gridSize = dim3(blockCount * smCount);
    blockThreads = dim3(BLOCK_SIZE);

    kernelTime = dispatch_and_measure_ccsc_kernel<DIM>(
        d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, gridSize, blockThreads, v1_kernelPtr,
        pStride, sStride);
    checkBestConf(bestConf_v1, memsetTime, kernelTime, flops, numBytes,
        ITEMS_PER_THREAD, BLOCK_SIZE, blockCount * smCount, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT,
        np, ns, std::string("v1"));

    // Run few times as many blocks, to check wheather it is better configuration
    gridSize = dim3(blockCount * smCount * CUB_PTX_SUBSCRIPTION_FACTOR);
    kernelTime = dispatch_and_measure_ccsc_kernel<DIM>(
        d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, gridSize, blockThreads, v1_kernelPtr, 
        pStride, sStride);
    checkBestConf(bestConf_v1, memsetTime, kernelTime, flops, numBytes,
        ITEMS_PER_THREAD, BLOCK_SIZE, blockCount * smCount * CUB_PTX_SUBSCRIPTION_FACTOR, DIM,
         IN_MEM_LAYOUT, OUT_MEM_LAYOUT, np, ns, std::string("v1"));
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
    int         np, 
    T const *   d_cordSums, 
    int const * d_spherePointCount, 
    int         sStride,
    float       memcpyTime,
    KernelConf &  bestConf_v1)
{
    typedef rd::gpu::bruteForce::AgentShiftSpherePolicy<
            BLOCK_SIZE,
            ITEMS_PER_THREAD>
        v1_PolicyT;

    auto v1_kernelPtr = rd::gpu::bruteForce::detail::DeviceShiftSphereKernel<
        v1_PolicyT, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, T>;

    size_t numElements = ns;
    // calculate mass center for each sphere
    size_t flops = numElements * (DIM);
    // calculate mass center for each sphere (read 2x)
    // shift condition evaluation (read 2x)
    // update chosen sample coordinates
    size_t numBytes = numElements * DIM * 2 * sizeof(T) + 
        numElements * DIM * 2 * sizeof(int) + 
        numElements * DIM * sizeof(T);

    dim3 gridSize, blockThreads;
    float kernelTime = 0;
    int blockCount;

    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, g_devId));

    // Get SM count
    int smCount = devProp.multiProcessorCount;

    //-----------------------------------------------------------------------------
    // V1
    //-----------------------------------------------------------------------------

    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blockCount, v1_kernelPtr, BLOCK_SIZE, 0));

    gridSize = dim3(blockCount * smCount);
    blockThreads = dim3(BLOCK_SIZE);

    kernelTime = dispatch_and_measure_stmc_kernel<DIM>(
        d_S, ns, d_cordSums, d_spherePointCount, d_SInitial, gridSize, blockThreads, v1_kernelPtr,
        sStride);
    checkBestConf(bestConf_v1, memcpyTime, kernelTime, flops, numBytes,
        ITEMS_PER_THREAD, BLOCK_SIZE, blockCount * smCount, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT,
        np, ns, std::string("v1"));

    // Run few times as many blocks, to check wheather it is better configuration
    gridSize = dim3(blockCount * smCount * CUB_PTX_SUBSCRIPTION_FACTOR);
    kernelTime = dispatch_and_measure_stmc_kernel<DIM>(
        d_S, ns, d_cordSums, d_spherePointCount, d_SInitial, gridSize, blockThreads, v1_kernelPtr, 
        sStride);
    checkBestConf(bestConf_v1, memcpyTime, kernelTime, flops, numBytes,
        ITEMS_PER_THREAD, BLOCK_SIZE, blockCount * smCount * CUB_PTX_SUBSCRIPTION_FACTOR, DIM,
         IN_MEM_LAYOUT, OUT_MEM_LAYOUT, np, ns, std::string("v1"));
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
    float       memsetTime,
    KernelConf &  bestConf_v1)
{
    // iterate over items per thread
    #ifdef QUICK_TEST
    benchmarkAlg_ccsc<6, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    #else
    benchmarkAlg_ccsc<1, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmarkAlg_ccsc<2, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmarkAlg_ccsc<3, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmarkAlg_ccsc<4, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmarkAlg_ccsc<5, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmarkAlg_ccsc<6, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmarkAlg_ccsc<7, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmarkAlg_ccsc<8, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmarkAlg_ccsc<9, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmarkAlg_ccsc<10, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmarkAlg_ccsc<11, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmarkAlg_ccsc<12, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
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
    int         np, 
    T const *   d_cordSums, 
    int const * d_spherePointCount, 
    int         sStride,
    float       memcpyTime,
    KernelConf &  bestConf_v1)
{
    // iterate over items per thread
    #ifdef QUICK_TEST
    benchmarkAlg_stmc<6, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    #else
    benchmarkAlg_stmc<1, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmarkAlg_stmc<2, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmarkAlg_stmc<3, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmarkAlg_stmc<4, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmarkAlg_stmc<5, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmarkAlg_stmc<6, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmarkAlg_stmc<7, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmarkAlg_stmc<8, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmarkAlg_stmc<9, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmarkAlg_stmc<10, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmarkAlg_stmc<11, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmarkAlg_stmc<12, BLOCK_SIZE, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
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
    float       memsetTime,
    KernelConf &  bestConf_v1)
{
    // iterate over test block size
    #ifdef QUICK_TEST
    benchmark_ccsc<128, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    #else
    benchmark_ccsc<64, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmark_ccsc<96, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmark_ccsc<128, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmark_ccsc<160, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmark_ccsc<192, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmark_ccsc<224, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmark_ccsc<256, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmark_ccsc<320, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    benchmark_ccsc<512, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride, memsetTime, bestConf_v1);
    #endif
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
    int         np,
    T const *   d_cordSums, 
    int const * d_spherePointCount, 
    int         sStride,
    float       memcpyTime,
    KernelConf &  bestConf_v1)
{
    // iterate over test block size
    #ifdef QUICK_TEST
    benchmark_stmc<128, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    #else
    benchmark_stmc<64, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmark_stmc<96, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmark_stmc<128, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmark_stmc<160, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmark_stmc<192, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmark_stmc<224, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmark_stmc<256, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmark_stmc<320, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
    benchmark_stmc<512, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride, memcpyTime, bestConf_v1);
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
    int         sStride)
{
    std::cout << "\n//-----------------------------------------------\n"
              << "// benchmark_ccsc\n" 
              << "//-----------------------------------------------\n" << std::endl;

    if (g_logPerfResults)
    {
        *g_logFile << "\n%//-----------------------------------------------\n"
              << "%// benchmark_ccsc\n" 
              << "%//-----------------------------------------------\n" << std::endl;
    }

    GpuTimer memsetTimer;
    float memsetTime = 0;

    // measure memset time
    memsetTimer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemset(d_cordSums, 0, ns * DIM * sizeof(T)));
        checkCudaErrors(cudaMemset(d_spherePointCount, 0, ns * sizeof(int)));
    }
    memsetTimer.Stop();
    memsetTime = memsetTimer.ElapsedMillis();

    KernelConf bestConf_v1, bestConf_v2_CUB, bestConf_v2_trove;
    bestConf_v1.time = std::numeric_limits<T>::max();

    benchmark_ccsc<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
        d_P, np, d_S, ns, r1, d_cordSums, d_spherePointCount, pStride, sStride,
        memsetTime, bestConf_v1, bestConf_v2_CUB, bestConf_v2_trove);

    if (g_logPerfResults)
    {
        *g_logFile << "\n%################################################\n";
        *g_logFile << "% best configuration v1: \n"
                 << "% time: " << bestConf_v1.time
                 << ", gflops: " << bestConf_v1.gflops
                 << ", gbytes: " << bestConf_v1.gbytes
                 << ", itemsPerThread: " << bestConf_v1.itemsPerThread
                 << ", blockSize: " << bestConf_v1.blockSize
                 << ", blockCount: " << bestConf_v1.blockCount << std::endl;
    }

    std::cout << "\n################################################\n";
    std::cout << "best configuration v1: \n"
             << " time: " << bestConf_v1.time
             << ", gflops: " << bestConf_v1.gflops
             << ", gbytes: " << bestConf_v1.gbytes
             << ", itemsPerThread: " << bestConf_v1.itemsPerThread
             << ", blockSize: " << bestConf_v1.blockSize
             << ", blockCount: " << bestConf_v1.blockCount << std::endl;
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
    int         np, 
    T const *   d_cordSums, 
    int const * d_spherePointCount, 
    int         sStride)
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
    memcpyTimer.Start();
    for (int i = 0; i < ITER; ++i)
    {
        checkCudaErrors(cudaMemcpy(d_S, d_SInitial, ns * DIM * sizeof(T), 
            cudaMemcpyDeviceToDevice));
    }
    memcpyTimer.Stop();
    memcpyTime = memcpyTimer.ElapsedMillis();

    KernelConf bestConf_v1, bestConf_v2_CUB, bestConf_v2_trove;
    bestConf_v1.time = std::numeric_limits<T>::max();

    benchmark_stmc<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
        d_S, d_SInitial, ns, np, d_cordSums, d_spherePointCount, sStride,
        memcpyTime, bestConf_v1, bestConf_v2_CUB, bestConf_v2_trove);

    if (g_logPerfResults)
    {
        *g_logFile << "\n%################################################\n";
        *g_logFile << "% best configuration v1: \n"
                 << "% time: " << bestConf_v1.time
                 << ", gflops: " << bestConf_v1.gflops
                 << ", gbytes: " << bestConf_v1.gbytes
                 << ", itemsPerThread: " << bestConf_v1.itemsPerThread
                 << ", blockSize: " << bestConf_v1.blockSize
                 << ", blockCount: " << bestConf_v1.blockCount << std::endl;
    }

    std::cout << "\n################################################\n";
    std::cout << "best configuration v1: \n"
             << " time: " << bestConf_v1.time
             << ", gflops: " << bestConf_v1.gflops
             << ", gbytes: " << bestConf_v1.gbytes
             << ", itemsPerThread: " << bestConf_v1.itemsPerThread
             << ", blockSize: " << bestConf_v1.blockSize
             << ", blockCount: " << bestConf_v1.blockCount << std::endl;
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
    int pStride, sStride;
        
    pStride = (OUT_MEM_LAYOUT == rd::ROW_MAJOR) ? 1 : dataPack.np;
    sStride = (OUT_MEM_LAYOUT == rd::ROW_MAJOR) ? 1 : dataPack.ns;

    // allocate needed memory

    checkCudaErrors(cudaMalloc((void**)&d_S, dataPack.ns * DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_SInitial, dataPack.ns * DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_P, dataPack.np * DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_cordSums, dataPack.ns * DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_spherePointCount, dataPack.ns * sizeof(int)));
    
    /************************************************************************
     *      GPU VERSION respective kernels
     ************************************************************************/

    // prepare data in appropriate memory layout
    rd::gpu::rdMemcpy<DIM, IN_MEM_LAYOUT, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
        d_P, dataPack.P, dataPack.np, pStride);

    rd::gpu::rdMemcpy<DIM, OUT_MEM_LAYOUT, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
        d_S, dataPack.S.data(), dataPack.ns, sStride);
    checkCudaErrors(cudaDeviceSynchronize());
    rd::gpu::rdMemcpy<DIM, OUT_MEM_LAYOUT, OUT_MEM_LAYOUT, cudaMemcpyDeviceToDevice>(
        d_SInitial, d_S, dataPack.ns);
    
    benchmark_ccsc<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
        d_P, dataPack.np, d_S, dataPack.ns, dataPack.r1, d_cordSums, 
        d_spherePointCount, pStride, sStride);

    benchmark_stmc<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
        d_S, d_SInitial, dataPack.ns, dataPack.np, d_cordSums, d_spherePointCount, sStride);

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
    #ifdef INDIVIDUAL_KERNELS
    benchmark_individual_kernels<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(dataPack);
    #elif defined(WHOLE_EVOLVE_PASS)
    benchmark_evolve<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(dataPack);
    #endif
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
        std::list<T*> csList;
        rd::choose(dataPack.P, dataPack.S.data(), csList, pointCnt, dataPack.ns,
            size_t(DIM), r1);
    }

    // release unused memory
    dataPack.S.resize(dataPack.ns * DIM);
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &dataPack.ns, sizeof(int)));

    std::cout << "Chosen count: " << dataPack.ns << std::endl;

    //---------------------------------------------------
    //          TEST EVOLVE
    //---------------------------------------------------

    std::cout << rd::HLINE << std::endl;
    std::cout << ">>>> ROW_MAJOR - ROW_MAJOR" << std::endl;
    #if defined(RD_PROFILE)
    nvtxRangeId_t r_rr = nvtxRangeStart("ROW_MAJOR-ROW_MAJOR");
    #endif
    test<DIM, rd::ROW_MAJOR, rd::ROW_MAJOR>(dataPack);
    #if defined(RD_PROFILE)
    nvtxRangeEnd(r_rr);
    nvtxRangeId_t r_cr = nvtxRangeStart("COL_MAJOR-ROW_MAJOR");
    #endif

    std::cout << rd::HLINE << std::endl;
    std::cout << ">>>> COL_MAJOR - ROW_MAJOR" << std::endl;
    test<DIM, rd::COL_MAJOR, rd::ROW_MAJOR>(dataPack);
    #if defined(RD_PROFILE)
    nvtxRangeEnd(r_cr);
    nvtxRangeId_t r_cc = nvtxRangeStart("COL_MAJOR-COL_MAJOR");
    #endif

    std::cout << rd::HLINE << std::endl;
    std::cout << ">>>> COL_MAJOR - COL_MAJOR" << std::endl;
    test<DIM, rd::COL_MAJOR, rd::COL_MAJOR>(dataPack);
    #if defined(RD_PROFILE)
    nvtxRangeEnd(r_cc);
    #endif
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

/**
 * @brief Test evolve time relative to number of points
 */

template <
    typename    T,
    int         DIM = 0,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testSize(
    PointCloud<T> & pc,
    T r1,
    int pointCnt = -1)
{
    if (pointCnt > 0)
    {
        TestDimensions<DIM, T>::impl(pc, r1, pointCnt);
    }
    else
    {
        for (int k = 10; k <= MAX_POINTS_NUM; k *= 10)
        {
            std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t pointCnt: " << k  
                    << "\n//------------------------------------------\n";

            TestDimensions<DIM, T>::impl(pc, r1, k);
        }
    }
}

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

#ifdef QUICK_TEST
        if (pointCnt    < 0 ||
            a           < 0 ||
            b           < 0 ||
            stddev      < 0)
        {
            std::cout << "Have to specify parameters! Rerun with --help for help.\n";
            exit(1);
        }
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (spiral) float: "  
                    << "\n//------------------------------------------\n";

        const int dim = 2;

        PointCloud<float> && fpc = SpiralPointCloud<float>(a, b, pointCnt, dim, stddev);
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
        PointCloud<float> && fpc2d = SpiralPointCloud<float>(22.f, 10.f, 0, 2, 4.f);
        PointCloud<float> && fpc3d = SpiralPointCloud<float>(22.f, 10.f, 0, 3, 4.f);
        PointCloud<double> && dpc2d = SpiralPointCloud<double>(22.0, 10.0, 0, 2, 4.0);
        PointCloud<double> && dpc3d = SpiralPointCloud<double>(22.0, 10.0, 0, 3, 4.0);

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (spiral) float: "  
                    << "\n//------------------------------------------\n";
        TestDimensions<2, float>::impl(fpc2d, r1, int(1e6));
        TestDimensions<3, float>::impl(fpc3d, r1, int(1e6));

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (spiral) double: "  
                    << "\n//------------------------------------------\n";
        TestDimensions<2, double>::impl(dpc2d, r1, int(1e6));
        TestDimensions<3, double>::impl(dpc3d, r1, int(1e6));

        PointCloud<float> && fpc2 = SegmentPointCloud<float>(1000.f, 0, 0, 4.f);
        PointCloud<double> && dpc2 = SegmentPointCloud<double>(1000.0, 0, 0, 4.0);;

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) float: "  
                    << "\n//------------------------------------------\n";
        testSize<float>(fpc2, r1);

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) double: "  
                    << "\n//------------------------------------------\n";
        testSize<double>(dpc2, r1);
    #endif

    checkCudaErrors(cudaDeviceReset());

    std::cout << rd::HLINE << std::endl;
    std::cout << "END!" << std::endl;

    return EXIT_SUCCESS;
}

