/**
 *  @file benchmark_dist_mtx.cu
 *  @author Adam Rogowiec
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

#ifdef RD_USE_OPENMP
#include <omp.h>
#endif
#ifdef RD_PROFILE
#include <cuda_profiler_api.h>
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <string>
#include <stdexcept>

#include "rd/gpu/agent/agent_dist_mtx.cuh"
#include "rd/gpu/device/device_distance_mtx.cuh"

#include "rd/gpu/util/dev_memcpy.cuh"
#include "rd/utils/memory.h"
#include "rd/utils/name_traits.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "tests/test_util.hpp"

#include "cub/test_util.h"
#include "cub/util_arch.cuh"

//------------------------------------------------------------
//  GLOBAL CONSTANTS / VARIABLES
//------------------------------------------------------------

static const std::string LOG_FILE_NAME_SUFFIX = "dist_mtx_perf.txt";

static std::ofstream * g_logFile           = nullptr;
static bool            g_logPerfResults    = false;
static std::string     g_devName           = "";
static int             g_devId             = 0;

#if defined(RD_PROFILE) || defined(RD_DEBUG)
static const int g_iterations = 1;
#else
static const int g_iterations = 100;
#endif

static constexpr int MAX_POINTS_NUM = int(1e3);
static constexpr int MAX_POINTS_DIM = int(1e3);


//------------------------------------------------------------
//  
//------------------------------------------------------------

template <typename T>
static void initializeLogFile()
{
    if (g_logPerfResults)
    {
        std::ostringstream logFileName;
        logFileName << typeid(T).name() << "_" <<  getCurrDateAndTime() << "_" 
                    << g_devName << "_" << LOG_FILE_NAME_SUFFIX;

        std::string logFilePath = rd::findPath("timings/", logFileName.str());
        g_logFile = new std::ofstream(logFilePath.c_str(), std::ios::out | std::ios::app);
        if (g_logFile->fail())
        {
            throw std::runtime_error("Couldn't open file: " + logFilePath);
        }

        // legend
        *g_logFile << "% "; 
        logValue(*g_logFile, "BLOCK_W", 10);
        logValue(*g_logFile, "BLOCK_H", 10);
        logValue(*g_logFile, "grid.x", 10);
        logValue(*g_logFile, "grid.y", 10);
        logValue(*g_logFile, "GFlops", 10);
        logValue(*g_logFile, "avgTime", 10);
        logValue(*g_logFile, "resUsage", 10);
        *g_logFile << "\n";
        *g_logFile << "% values in brackets(regs used, local mem used, smem used) \n";
    }
}

//------------------------------------------------------------
//  
//------------------------------------------------------------

template <typename T>
static void symDistMtx_cpu(
    T const *   h_in,
    T *         h_out,
    int         w,
    int         h)
{
    #ifdef RD_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            if (j <= i)
            {
                T sqrDist = 0;
                T diff = 0;
                for (int k = 0; k < w; ++k)
                {
                    diff = h_in[i * w + k] - h_in[j*w + k];
                    sqrDist += diff * diff;
                }
                h_out[i * h + j] = sqrDist;
                if (j < i)
                {
                    h_out[j * h + i] = sqrDist;
                }
            }
        }
    }
}

template<
    int                     BLOCK_W,
    int                     BLOCK_H,
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename                T>
__launch_bounds__ (BLOCK_H * BLOCK_W)
__global__ void symDistMtxKernel(
    T const * __restrict__  d_in,
    T *                     d_out,
    int                     width,
    int                     height,
    int                     inStride,
    int                     outStride)
{
    typedef rd::gpu::detail::AgentDistMtx<BLOCK_W, BLOCK_H, MEM_LAYOUT, T> AgentT;

    typedef typename AgentT::TempStorage TempStorageT;
    __shared__ TempStorageT smem;

    AgentT(smem).symDist(d_in, d_out, width, height, inStride, outStride);
}
//------------------------------------------------------------
//  
//------------------------------------------------------------

template<
    int                     BLOCK_W,
    int                     BLOCK_H,
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename                T>
static void dispatchSymDistMtx(
    T const * d_in,
    T *       d_out,
    int       width,
    int       height,
    size_t    d_inPitch,
    size_t    d_outPitch,
    float     memsetTime,
    T const * h_gold,
    T *       h_gpu_out)
{

    GpuTimer timer;
    float kernelTime = -1;

    int inStride = static_cast<int>(d_inPitch / sizeof(T));
    int outStride = static_cast<int>(d_outPitch / sizeof(T));

    #ifndef QUICK_TEST
    int ptxVersion = 0;
    checkCudaErrors(cub::PtxVersion(ptxVersion));

    // Get SM count
    int smCount;
    checkCudaErrors(cudaDeviceGetAttribute(
        &smCount, cudaDevAttrMultiProcessorCount, g_devId));

    auto kptr = symDistMtxKernel<BLOCK_W, BLOCK_H, MEM_LAYOUT, T>;

    // get SM occupancy
    int smOccupancy;
    checkCudaErrors(cub::MaxSmOccupancy(smOccupancy, kptr, BLOCK_W * BLOCK_H));
    int blockCount = smCount * smOccupancy * CUB_SUBSCRIPTION_FACTOR(ptxVersion);
    float root = sqrtf(float(blockCount));
    int rc = static_cast<int>(ceilf(root));
    int rf = static_cast<int>(floorf(root));

    dim3 dimGrid(1), dimBlock(1);
    dimGrid.x = rc;
    dimGrid.y = rf;
    dimBlock.x = BLOCK_W;
    dimBlock.y = BLOCK_H;
    #endif

    // std::cout << "bw: " << dimBlock.x << ", bh: " << dimBlock.y 
    //           << ", gw: " << dimGrid.x << ", gh: " << dimGrid.y 
    //           << ", inStride: " << inStride << ", outStride: " << outStride
    //           << ", blockCount: " << blockCount
    //           << std::endl;

    rd::fillTable_omp(h_gpu_out, T(0), height * height);

    // warm-up & correctness check
    #ifndef RD_PROFILE
    checkCudaErrors(cudaMemset(d_out, 0, height * d_outPitch));
    #ifndef QUICK_TEST
    kptr<<<dimGrid, dimBlock>>>(d_in, d_out, width, 
        height, inStride, outStride);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    #else
    rd::gpu::DeviceDistanceMtx::symmetricDistMtx<MEM_LAYOUT>(
        d_in, d_out, width, height, inStride, outStride, 0, true);
    #endif

    checkCudaErrors(cudaMemcpy2D(h_gpu_out, height * sizeof(T), d_out, d_outPitch, 
        height * sizeof(T), height, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    /*double maxErr = */
    rd::checkResult(h_gold, h_gpu_out, height * height);
    // if(maxErr > 1e-1)
    // {
    //     throw std::runtime_error("Incorrect results!");
    // }
    // rd::printTable(h_gpu_out, height, height, "gpu_output");
    #endif

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < g_iterations; ++i)
    {
        checkCudaErrors(cudaMemset2D(d_out, d_outPitch, 0, height * sizeof(T), height));
        #ifndef QUICK_TEST
        kptr<<<dimGrid, dimBlock>>>(d_in, d_out, width, 
            height, inStride, outStride);
        checkCudaErrors(cudaGetLastError());
        #else
        rd::gpu::DeviceDistanceMtx::symmetricDistMtx<MEM_LAYOUT>(
            d_in, d_out, width, height, inStride, outStride);
        #endif
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    kernelTime = (kernelTime - memsetTime) / static_cast<float>(g_iterations);

    size_t numElements = height * height;
    size_t flops = numElements * width * 3;
    float GFlops = flops / kernelTime / 1000.f / 1000.f;

    #ifndef QUICK_TEST
    KernelResourceUsage kernelResUsage(kptr);
    if (g_logPerfResults)
    {
        logValues(*g_logFile, std::string(rd::DataMemoryLayoutNameTraits<MEM_LAYOUT>::name), 
            BLOCK_W, BLOCK_H, dimGrid.x, dimGrid.y, GFlops, kernelTime);
        *g_logFile << " " << kernelResUsage.prettyPrint();
        *g_logFile << "\n";
    }
    logValues(std::cout, std::string(rd::DataMemoryLayoutNameTraits<MEM_LAYOUT>::name), 
        BLOCK_W, BLOCK_H, dimGrid.x, dimGrid.y, GFlops, kernelTime);
    std::cout << " " << kernelResUsage.prettyPrint();
    #else
    logValues(std::cout, std::string(rd::DataMemoryLayoutNameTraits<MEM_LAYOUT>::name), 
        GFlops, kernelTime);
    #endif
    std::cout << std::endl;
}
//------------------------------------------------------------
//  
//------------------------------------------------------------

template <
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename                T>
static void test(
    int w,
    int h,
    std::vector<T> && points)
{
    //allocate memory
    T * d_in;
    T * d_out;
    T * h_gpu_out, *h_cpu_out;

    size_t d_inPitch = 0, d_outPitch = 0;
    if (MEM_LAYOUT == rd::ROW_MAJOR)
    {
        checkCudaErrors(cudaMallocPitch(&d_in, &d_inPitch, w * sizeof(T), h));
        checkCudaErrors(cudaMemcpy2D(d_in, d_inPitch, points.data(), w * sizeof(T), w * sizeof(T),
            h, cudaMemcpyHostToDevice));
    }
    else if (MEM_LAYOUT == rd::COL_MAJOR)
    {
        checkCudaErrors(cudaMallocPitch(&d_in, &d_inPitch, h * sizeof(T), w));
        rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_in, points.data(), w, h, d_inPitch, w * sizeof(T));
    }
    checkCudaErrors(cudaMallocPitch(&d_out, &d_outPitch, h * sizeof(T), h));
    h_gpu_out = new T[h * h];
    h_cpu_out = new T[h * h];
    

    symDistMtx_cpu(points.data(), h_cpu_out, w, h);

    float memsetTime = 0;
    GpuTimer timer;

    timer.Start();
    for (int i = 0; i < g_iterations; ++i)
    {
        checkCudaErrors(cudaMemset2D(d_out, d_outPitch, 0, h * sizeof(T), h));
    }
    timer.Stop();
    memsetTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());

    try
    {

        #ifndef QUICK_TEST
        dispatchSymDistMtx<32,16, MEM_LAYOUT>(d_in, d_out, w, h, d_inPitch, d_outPitch, memsetTime, 
            h_cpu_out, h_gpu_out);
        dispatchSymDistMtx<32, 8, MEM_LAYOUT>(d_in, d_out, w, h, d_inPitch, d_outPitch, memsetTime, 
            h_cpu_out, h_gpu_out);
        dispatchSymDistMtx<32, 4, MEM_LAYOUT>(d_in, d_out, w, h, d_inPitch, d_outPitch, memsetTime, 
            h_cpu_out, h_gpu_out);
        dispatchSymDistMtx<32, 2, MEM_LAYOUT>(d_in, d_out, w, h, d_inPitch, d_outPitch, memsetTime, 
            h_cpu_out, h_gpu_out);
        #else
        dispatchSymDistMtx<0, 0, MEM_LAYOUT>(d_in, d_out, w, h, d_inPitch, d_outPitch, memsetTime, 
            h_cpu_out, h_gpu_out);
        #endif
    }
    catch(std::runtime_error const & e)
    {
        std::cerr << e.what() << std::endl;
        checkCudaErrors(cudaFree(d_in));
        checkCudaErrors(cudaFree(d_out));
        delete[] h_gpu_out;
        delete[] h_cpu_out;
        throw;
    }

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
    delete[] h_gpu_out;
    delete[] h_cpu_out;
}

template <typename T>
static void testMemLayout(
    int w,
    int h,
    std::vector<T> && points)
{
    test<rd::ROW_MAJOR>(w, h, std::forward<std::vector<T>>(points));
    test<rd::COL_MAJOR>(w, h, std::forward<std::vector<T>>(points));
}

template <
    typename    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testSize(
    PointCloud<T> & pc)
{
    if (g_logPerfResults)
    {
        initializeLogFile<T>();
    }

    pc.pointCnt_ = MAX_POINTS_NUM;
    pc.dim_ = MAX_POINTS_DIM;
    pc.initializeData();

    try
    {
    for (int h = 100; h <= MAX_POINTS_NUM; h += 100)
    {
        for (int w = 100; w <= MAX_POINTS_DIM; w *= 2)
        {
            std::cout << "\n//------------------------------------------" 
                << "\n//\t\t h: " << h << ", w: " << w
                << "\n//------------------------------------------\n";

            if (g_logPerfResults)
            {
                *g_logFile << "\n\n%//------------------------------------------ h: " 
                           << "\n%//\t\t h: " << h << ", w: " << w
                           << "\n%//------------------------------------------\n";
            }

            testMemLayout(w, h, pc.extractPart(h, w));
            if (g_logPerfResults)
            {
                (*g_logFile).flush();
            }
        }
    }
    }
    catch (std::runtime_error const & e)
    {
        // clean-up
        if (g_logPerfResults)
        {
            g_logFile->close();
            delete g_logFile;
        }
        checkCudaErrors(cudaDeviceReset());
        exit(1);
    }

    // clean-up
    if (g_logPerfResults)
    {
        g_logFile->close();
        delete g_logFile;
    }
}

//------------------------------------------------------------
//  MAIN
//------------------------------------------------------------

int main(int argc, char const **argv)
{

    int dim = -1;
    float stddev = 10;
    int pointCnt = -1;

    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help")) 
    {
        printf("%s \n"
            "\t\t[--log  <log performance results>]\n"
            "\t\t[--size <number of points>]\n"
            "\t\t[--dim  <points dimension>]\n"
            "\t\t[--d    <device id>]\n"
            "\n", argv[0]);
        exit(0);
    }

    if (args.CheckCmdLineFlag("log"))
    {
        g_logPerfResults = true;
    }
    if (args.CheckCmdLineFlag("size"))
    {
        args.GetCmdLineArgument("size", pointCnt);
    }    
    if (args.CheckCmdLineFlag("dim")) 
    {
        args.GetCmdLineArgument("dim", dim);
    }    
    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", g_devId);
    }

    checkCudaErrors(deviceInit(g_devId));

    // set device name for logging
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, g_devId));
    g_devName = devProp.name;

    #ifdef QUICK_TEST
        if (pointCnt < 0 || dim < 0 )
        {
            std::cout << "Have to specify parameters! Rerun with --help for help.\n";
            exit(1);
        }
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t float: "  
                    << "\n//------------------------------------------\n";

        PointCloud<float> && fpc = SegmentPointCloud<float>(1000.f, pointCnt, dim, stddev);
        fpc.initializeData();

        // std::vector<float> h_in(pointCnt * dim);
        // rd::fillTable_omp(h_in.data(), 1.f, pointCnt * dim);

        // rd::printTable(h_in.data(), dim, pointCnt, "input");

        // run benchmark
        testMemLayout<float>(dim, pointCnt, fpc.extractPart(pointCnt, dim));
        // test<float>(dim, pointCnt, std::move(h_in));

        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t double: "  
        //             << "\n//------------------------------------------\n";
        // PointCloud<double> && dpc = SegmentPointCloud<double>(1000.0, pointCnt, dim, stddev);
        
    #else
        #ifndef RD_DOUBLE_PRECISION
        PointCloud<float> && fpc = SegmentPointCloud<float>(1000.f, 0, 0, stddev);
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t float: "  
                    << "\n//------------------------------------------\n";
        testSize<float>(fpc);

        #else
        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t double: "  
        //             << "\n//------------------------------------------\n";
        // PointCloud<double> && dpc = SegmentPointCloud<double>(1000.0, 0, 0, stddev);
        // testSize<double>(dpc);
        #endif
    #endif

    checkCudaErrors(cudaDeviceReset());

    std::cout << rd::HLINE << std::endl;
    std::cout << "END!" << std::endl;

    return EXIT_SUCCESS;
}
