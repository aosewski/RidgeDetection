/**
 * @file benchmark_find_bounds.cu
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
#define CUB_STDERR
#define BLOCK_TILE_LOAD_V1 1

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/bounding_box.hpp"
#include "rd/utils/memory.h" 
#include "rd/utils/name_traits.hpp"
#include "rd/utils/samples_set.hpp"
#include "rd/utils/rd_params.hpp"
#include "rd/utils/bounding_box.hpp"

#include "rd/gpu/device/samples_generator.cuh"
#include "rd/gpu/device/device_find_bounds.cuh"

#include "cub/test_util.h"

#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda_profiler_api.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <limits>

#include <cmath>
#include <utility>

//------------------------------------------------------------
//  GLOBAL CONSTANTS / VARIABLES
//------------------------------------------------------------

static const std::string LOG_FILE_NAME_SUFFIX = "_find_bounds_timings_v1.txt";

std::ofstream * g_logFile       = nullptr;
bool            g_drawGraphs    = false;
std::string     g_devName;

std::vector<std::vector<float>> g_bestPerf;
static const float              g_graphColStep  = 0.3f;
static const int                g_graphNCol     = 4;     // group's columns count 
static const int                g_graphNGroups  = 6;     // number of dimensions to plot


#if defined(RD_PROFILE) || defined(RD_DEBUG)
static const int g_iterations = 1;
#else
static const int g_iterations = 100;
#endif


//------------------------------------------------------------
//  REFERENCE FIND BOUNDS
//------------------------------------------------------------

template <typename T>
void findBoundsGold(
    rd::RDParams<T> &rdp,
    T const *P,
    rd::BoundingBox<T> &bbox)
{
    bbox.dim = rdp.dim;
    bbox.findBounds(P, rdp.np, rdp.dim);
}

template <int DIM, typename T>
bool compareBounds(
    rd::BoundingBox<T> const & bboxGold,
    T (&h_minBounds)[DIM],
    T (&h_maxBounds)[DIM])
{
    bool result = true;
    for (int d = 0; d < DIM; ++d)
    {
        #ifdef RD_DEBUG
        std::cout << "min[" << d << "] gpu: " << std::right << std::fixed << std::setw(12) << std::setprecision(8) <<
            h_minBounds[d] <<", cpu: "<< bboxGold.min(d) <<"\n";
        std::cout << "max[" << d << "] gpu: " << std::right << std::fixed << std::setw(12) << std::setprecision(8) <<
            h_maxBounds[d] <<", cpu: "<< bboxGold.max(d) <<"\n";
        #endif
        
        if (h_minBounds[d] != bboxGold.min(d))
        {
            result = false;
            std::cout << "ERROR!: min[" << d << "] is: "<< h_minBounds[d] <<", should be: "<< bboxGold.min(d) <<"\n";
        } 
        if (h_maxBounds[d] != bboxGold.max(d)) 
        {
            result = false;
            std::cout << "ERROR!: max[" << d << "] is: "<< h_maxBounds[d] <<", should be: "<< bboxGold.max(d) <<"\n";
        }
    }
    return result;
}

//------------------------------------------------------------
//  KERNEL DISPATCH
//------------------------------------------------------------

template <
    int                         BLOCK_THREADS,
    int                         POINTS_PER_THREAD,
    cub::BlockReduceAlgorithm   REDUCE_ALGORITHM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    int                         DIM,
    typename                    SampleT,
    typename                    OffsetT>
void dispatchFindBoundsKernel(
    void *                      d_tempStorage,              
    size_t &                    tempStorageBytes,           
    SampleT const *             d_in,
    SampleT *                   d_outMin,
    SampleT *                   d_outMax,
    int                         numPoints,
    OffsetT                     stride,
    int                         iterations,
    bool                        debugSynchronous = false)
{
    typedef rd::gpu::detail::AgentFindBoundsPolicy<
        BLOCK_THREADS,
        POINTS_PER_THREAD,
        REDUCE_ALGORITHM,
        LOAD_MODIFIER,
        IO_BACKEND> AgentFindBoundsPolicyT;

    typedef rd::gpu::DispatchFindBounds<
        SampleT,
        OffsetT,
        DIM,
        INPUT_MEM_LAYOUT> DispatchFindBoundsT;

    typename DispatchFindBoundsT::KernelConfig findBoundsConfig;
    findBoundsConfig.blockThreads = BLOCK_THREADS;
    findBoundsConfig.itemsPerThread = POINTS_PER_THREAD;

    for (int i = 0; i < iterations; ++i)
    {
        CubDebugExit(DispatchFindBoundsT::invoke(
            d_tempStorage,
            tempStorageBytes,
            d_in,
            d_outMin,
            d_outMax,
            numPoints,
            stride,
            0,
            debugSynchronous,
            rd::gpu::detail::deviceFindBoundsKernelFirstPass<AgentFindBoundsPolicyT, DIM, INPUT_MEM_LAYOUT, SampleT, OffsetT>,
            rd::gpu::detail::deviceFindBoundsKernelSecondPass<AgentFindBoundsPolicyT, DIM, SampleT , OffsetT>,
            findBoundsConfig));
    }

}

//------------------------------------------------------------
//  Benchmark helper structures
//------------------------------------------------------------

struct KernelParametersConf
{
    int                         BLOCK_THREADS;
    int                         POINTS_PER_THREAD;
    int                         DIM;
    // cub::BlockReduceAlgorithm   REDUCE_ALGORITHM,
    cub::CacheLoadModifier      LOAD_MODIFIER;
    rd::gpu::BlockTileIOBackend IO_BACKEND;
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT;
    float                       avgMillis;
    float                       gigaBandwidth;

    KernelParametersConf()
    :
        LOAD_MODIFIER(cub::LOAD_DEFAULT),
        INPUT_MEM_LAYOUT(rd::ROW_MAJOR),
        IO_BACKEND(rd::gpu::IO_BACKEND_CUB)
    {}

    KernelParametersConf(
        int                         _DIM,
        cub::CacheLoadModifier      _LOAD_MODIFIER,
        rd::gpu::BlockTileIOBackend _IO_BACKEND,
        rd::DataMemoryLayout        _INPUT_MEM_LAYOUT)
    :
        DIM(_DIM),
        LOAD_MODIFIER(_LOAD_MODIFIER),
        IO_BACKEND(_IO_BACKEND),
        INPUT_MEM_LAYOUT(_INPUT_MEM_LAYOUT)
    {}

};

typedef std::pair<float, float> KernelPerfT;


//------------------------------------------------------------
//  TEST CONFIGURATION AND RUN
//------------------------------------------------------------

template <
    int                         BLOCK_THREADS,
    int                         POINTS_PER_THREAD,
    int                         DIM,
    cub::BlockReduceAlgorithm   REDUCE_ALGORITHM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    typename                    SampleT>
KernelPerfT runDeviceFindBounds(
    rd::RDParams<SampleT> const &       rdp,
    SampleT const *                     d_in,
    SampleT *                           d_outMin,
    SampleT *                           d_outMax,
    int                                 stride,
    rd::BoundingBox<SampleT> const &    bboxGold)
{
    std::cout << rd::HLINE << std::endl;
    std::cout << "runDeviceFindBounds:" << std::endl;
    std::cout << "blockThreads: " << BLOCK_THREADS 
              << ", pointsPerThread: " << POINTS_PER_THREAD
              << ", load modifier: " << rd::LoadModifierNameTraits<LOAD_MODIFIER>::name
              << ", io backend: " << rd::BlockTileIONameTraits<IO_BACKEND>::name
              << ", mem layout: " << rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name
              << ", numPoints: " << rdp.np << "\n";

    // Allocate temporary storage
    void            *d_tempStorage = NULL;
    size_t          tempStorageBytes = 0;

    dispatchFindBoundsKernel<BLOCK_THREADS, POINTS_PER_THREAD, REDUCE_ALGORITHM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, DIM>(
        d_tempStorage, tempStorageBytes, d_in, d_outMin, d_outMax, int(rdp.np), stride, 1, true); 

    checkCudaErrors(cudaMalloc((void**)&d_tempStorage, tempStorageBytes));

    //---------------------------------------------
    // Run warm-up/correctness iteration
    //---------------------------------------------
    dispatchFindBoundsKernel<BLOCK_THREADS, POINTS_PER_THREAD, REDUCE_ALGORITHM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, DIM>(
        d_tempStorage, tempStorageBytes, d_in, d_outMin, d_outMax, int(rdp.np), stride, 1, true); 
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    SampleT h_minBounds[DIM];
    SampleT h_maxBounds[DIM];
    checkCudaErrors(cudaMemcpy(h_minBounds, d_outMin, DIM * sizeof(SampleT), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_maxBounds, d_outMax, DIM * sizeof(SampleT), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    bool result = compareBounds(bboxGold, h_minBounds, h_maxBounds);
    if (result)
    {
        std::cout << ">>>> CORRECT!\n";
    }
    else
    {
        std::cout << ">>>> ERROR!" << std::endl;
        // clean-up
        checkCudaErrors(cudaFree(d_tempStorage));
        return std::make_pair(1e10f, -1.0f);
    }

    //---------------------------------------------
    // Measure performance
    //---------------------------------------------

    GpuTimer timer;
    float elapsedMillis;

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();

    dispatchFindBoundsKernel<BLOCK_THREADS, POINTS_PER_THREAD, REDUCE_ALGORITHM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, DIM>(
        d_tempStorage, tempStorageBytes, d_in, d_outMin, d_outMax, int(rdp.np), stride, g_iterations); 

    timer.Stop();
    elapsedMillis = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    double avgMillis = double(elapsedMillis) / g_iterations;
    double gigaBandwidth = double(rdp.np * DIM * sizeof(SampleT)) / avgMillis / 1000.0 / 1000.0;

    if (rdp.verbose)
    {
        *g_logFile << POINTS_PER_THREAD << " " << BLOCK_THREADS << " " << avgMillis << " " << gigaBandwidth << "\n";
    }

    std::cout << avgMillis << " avg ms, "
              << gigaBandwidth << " logical GB/s\n";

    // clean-up
    checkCudaErrors(cudaFree(d_tempStorage));

    return std::make_pair(avgMillis, gigaBandwidth);
}

//------------------------------------------------------------
//  TEST SPECIALIZATIONS
//------------------------------------------------------------

/*
 *  Specialization for testing different number of threads in a block 
 */
template <
    int                         POINTS_PER_THREAD,
    int                         DIM,
    cub::BlockReduceAlgorithm   REDUCE_ALGORITHM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    typename                    SampleT>
KernelParametersConf testBlockSize(
    rd::RDParams<SampleT> const &       rdp,
    SampleT const *                     d_in,
    SampleT *                           d_outMin,
    SampleT *                           d_outMax,
    int                                 stride,
    rd::BoundingBox<SampleT> const &    bboxGold)
{
    if (rdp.verbose)
    {
        *g_logFile << "% test block size: \n%"
              << "pointsPerThread: " << POINTS_PER_THREAD
              << ", load modifier: " << rd::LoadModifierNameTraits<LOAD_MODIFIER>::name
              << ", io backend: " << rd::BlockTileIONameTraits<IO_BACKEND>::name
              << ", mem layout: " << rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name
              << ", numPoints: " << rdp.np << "\n";
    }

    KernelParametersConf bestKernelParams(DIM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT);
    bestKernelParams.POINTS_PER_THREAD = POINTS_PER_THREAD;
    KernelPerfT bestPerf = std::make_pair(1e10f, -1.0f);

    auto processResults = [&bestPerf, &bestKernelParams](int bs, KernelPerfT kp)
    {
        if (kp.second > bestPerf.second)
        {
            bestPerf.first = kp.first;
            bestPerf.second = kp.second;
            bestKernelParams.avgMillis = kp.first;
            bestKernelParams.gigaBandwidth = kp.second;
            bestKernelParams.BLOCK_THREADS = bs;
        }
    };

#define runTest(blockSize) processResults(blockSize, \
        runDeviceFindBounds<blockSize, POINTS_PER_THREAD, DIM, REDUCE_ALGORITHM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT>( \
            rdp, d_in, d_outMin, d_outMax, stride, bboxGold));
    
    runTest(64);
    runTest(96);
    runTest(128);
    runTest(256);
    runTest(384);
    runTest(512);

#undef runTest

    return bestKernelParams;
}

/*
 *  Specialization for testing different points per thread
 */
template <
    int                         DIM,
    cub::BlockReduceAlgorithm   REDUCE_ALGORITHM,
    cub::CacheLoadModifier      LOAD_MODIFIER,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    rd::DataMemoryLayout        INPUT_MEM_LAYOUT,
    typename                    SampleT>
KernelParametersConf testPointsPerThread(
    rd::RDParams<SampleT> const &       rdp,
    SampleT const *                     d_in,
    SampleT *                           d_outMin,
    SampleT *                           d_outMax,
    int                                 stride,
    rd::BoundingBox<SampleT> const &    bboxGold)
{
    if (rdp.verbose)
    {
        *g_logFile << "% test points per thread: \n%"
              << ", load modifier: " << rd::LoadModifierNameTraits<LOAD_MODIFIER>::name
              << ", io backend: " << rd::BlockTileIONameTraits<IO_BACKEND>::name
              << ", mem layout: " << rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name
              << ", numPoints: " << rdp.np << "\n";
    }

    KernelParametersConf bestKernelParams;

    auto checkBestConf = [&bestKernelParams](KernelParametersConf kp)
    {
        if (kp.gigaBandwidth > bestKernelParams.gigaBandwidth)
        {
            bestKernelParams = kp;
        }
    };

    bestKernelParams = testBlockSize<1, DIM, REDUCE_ALGORITHM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, SampleT>(
            rdp, d_in, d_outMin, d_outMax, stride, bboxGold);

#define runTest(ppt) checkBestConf(testBlockSize<ppt, DIM, REDUCE_ALGORITHM, LOAD_MODIFIER, IO_BACKEND, INPUT_MEM_LAYOUT, SampleT>( \
                                        rdp, d_in, d_outMin, d_outMax, stride, bboxGold));
    
    runTest(2);
    runTest(3);
    runTest(4);
    runTest(5);
    runTest(6);
    runTest(7);
    runTest(8);
    runTest(9);
    runTest(10);

#undef runTest
    
    if (rdp.verbose)
    {
        *g_logFile << "% best performance conf: " << bestKernelParams.BLOCK_THREADS 
                    << ", " << bestKernelParams.POINTS_PER_THREAD 
                    << ", " << bestKernelParams.avgMillis
                    << ", " << bestKernelParams.gigaBandwidth << "\n"; 
    }

    std::cout << ">>>>>>> best performance conf: " << bestKernelParams.BLOCK_THREADS 
                << ", " << bestKernelParams.POINTS_PER_THREAD 
                << ", " << bestKernelParams.avgMillis
                << ", " << bestKernelParams.gigaBandwidth; 

    return bestKernelParams;
}

//------------------------------------------------------------
//  TEST SPECIFIED VARIANTS
//------------------------------------------------------------

template <int DIM, typename T>
void test(rd::RDParams<T> &rdp,
          rd::RDSpiralParams<T> &rds)
{
    rdp.dim = DIM;
    std::vector<std::string> samplesDir{"../../examples/data/nd_segments/", "../../examples/data/spirals/"};
    rd::Samples<T> d_samplesSet(rdp, rds, samplesDir, DIM);

    std::cout << "Samples: " << std::endl;
    std::cout <<  "\t dimension: " << rdp.dim << std::endl;
    std::cout <<  "\t n_samples: " << rdp.np << std::endl;

    std::cout << "Spiral params: " << std::endl;
    if (DIM == 2 || DIM == 3) 
    {
        std::cout <<  "\t a: " << rds.a << std::endl;
        std::cout <<  "\t b: " << rds.b << std::endl;
    }
    else
    {
        std::cout <<  "\t seg length: " << rds.a << std::endl;
    }
    std::cout <<  "\t sigma: " << rds.sigma << std::endl; 

    T *d_PRowMajor, *d_PColMajor;
    T *h_P;
    T *d_minBounds, *d_maxBounds;

    // allocate containers
    checkCudaErrors(cudaMalloc((void**)&d_PRowMajor, rdp.np * DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_PColMajor, rdp.np * DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_minBounds, DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_maxBounds, DIM * sizeof(T)));
    h_P = new T[rdp.np * DIM];

    // initialize data
    checkCudaErrors(cudaMemcpy(d_PRowMajor, d_samplesSet.samples_, rdp.np * DIM * sizeof(T), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(h_P, d_samplesSet.samples_, rdp.np * DIM * sizeof(T), cudaMemcpyDeviceToHost));

    T * tmp = new T[rdp.np * DIM];
    rd::transposeTable(h_P, tmp, rdp.np, DIM);
    checkCudaErrors(cudaMemcpy(d_PColMajor, tmp, rdp.np * DIM * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    delete[] tmp;
    tmp = nullptr;

    //---------------------------------------------------
    // Prepare logFile if needed
    //---------------------------------------------------

    if (rdp.verbose)
    {
        std::ostringstream logFileName;
        // append device name to log file
        logFileName << g_devName << "_" << std::to_string(DIM) <<
             "D" << LOG_FILE_NAME_SUFFIX;

        std::string logFilePath = rd::findPath("timings/", logFileName.str());
        g_logFile = new std::ofstream(logFilePath.c_str(), std::ios::out | std::ios::app);
        if (g_logFile->fail())
        {
            throw std::logic_error("Couldn't open file: " + logFileName.str());
        }

        *g_logFile << "%" << rd::HLINE << std::endl;
        *g_logFile << "% " << typeid(T).name() << std::endl;
        *g_logFile << "%" << rd::HLINE << std::endl;
    }

    //---------------------------------------------------
    //               REFERENCE BOUNDING BOX
    //---------------------------------------------------
    rd::BoundingBox<T> bboxGold(h_P, rdp.np, rdp.dim);
    bboxGold.calcDistances();
    
    if (rdp.verbose)
    {
        bboxGold.print();
    }

    //---------------------------------------------------
    //               GPU BOUNDING BOX
    //---------------------------------------------------

    std::vector<KernelParametersConf> bestConfigurations;
    KernelParametersConf bestKernelParams;

    auto processResults = [&bestKernelParams](int graphSec, KernelParametersConf params)
    {
        if (params.gigaBandwidth > bestKernelParams.gigaBandwidth)
        {
            bestKernelParams = params;
        }
        if (g_drawGraphs)
        {
            g_bestPerf[graphSec].push_back(DIM);
            g_bestPerf[graphSec].push_back(params.gigaBandwidth);
        }
    };

    std::cout << "\n" << rd::HLINE << "\n";

    bestKernelParams = testPointsPerThread<DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_LDG, rd::gpu::IO_BACKEND_CUB, rd::COL_MAJOR>(rdp, d_PColMajor, d_minBounds, d_maxBounds, rdp.np, bboxGold);
    if (g_drawGraphs)
    {
        g_bestPerf[0].push_back(DIM);
        g_bestPerf[0].push_back(bestKernelParams.gigaBandwidth);
    }
    processResults(1, testPointsPerThread<DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_LDG, rd::gpu::IO_BACKEND_CUB, rd::ROW_MAJOR>(rdp, d_PRowMajor, d_minBounds, d_maxBounds, 1, bboxGold));

    std::cout << "\n" << rd::HLINE << "\n";

    processResults(2, testPointsPerThread<DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_LDG, rd::gpu::IO_BACKEND_TROVE, rd::COL_MAJOR>(rdp, d_PColMajor, d_minBounds, d_maxBounds, rdp.np, bboxGold));
    processResults(3, testPointsPerThread<DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_LDG, rd::gpu::IO_BACKEND_TROVE, rd::ROW_MAJOR>(rdp, d_PRowMajor, d_minBounds, d_maxBounds, 1, bboxGold));

    std::cout << "\n" << rd::HLINE << "\n";

    //---------------------------------------------------
    //  summarize results
    //---------------------------------------------------

    if (rdp.verbose)
    {
        *g_logFile << "\n% overall best conf: " 
            << "\n%avgMillis: \t\t" << bestKernelParams.avgMillis
            << "\n%gigaBandwidth: \t" << bestKernelParams.gigaBandwidth
            << "\n%block threads: \t" << bestKernelParams.BLOCK_THREADS
            << "\n%points per thread: \t" << bestKernelParams.POINTS_PER_THREAD
            << "\n%load modifier: \t" << rd::getLoadModifierName(bestKernelParams.LOAD_MODIFIER)
            << "\n%mem layout: \t\t" << rd::getRDDataMemoryLayout(bestKernelParams.INPUT_MEM_LAYOUT)
            << "\n%io backend: \t\t" << rd::getRDTileIOBackend(bestKernelParams.IO_BACKEND)
            << "\n%numPoints: \t\t" << rdp.np << "\n";
    }

    std::cout << ">>>>> overall best conf: \n" 
        << "\n avgMillis: \t\t" << bestKernelParams.avgMillis
        << "\n gigaBandwidth: \t" << bestKernelParams.gigaBandwidth
        << "\n block threads: \t" << bestKernelParams.BLOCK_THREADS
        << "\n points per thread: \t" << bestKernelParams.POINTS_PER_THREAD
        << "\n load modifier: \t" << rd::getLoadModifierName(bestKernelParams.LOAD_MODIFIER)
        << "\n mem layout: \t\t" << rd::getRDDataMemoryLayout(bestKernelParams.INPUT_MEM_LAYOUT)
        << "\n io backend: \t\t" << rd::getRDTileIOBackend(bestKernelParams.IO_BACKEND)
        << "\n numPoints: \t\t" << rdp.np << "\n";

    //---------------------------------------------------
    // clean-up
    if (rdp.verbose)
    {
        g_logFile->close();
        delete g_logFile;
    }
    
    delete[] h_P;

    checkCudaErrors(cudaFree(d_PRowMajor));
    checkCudaErrors(cudaFree(d_PColMajor));
    checkCudaErrors(cudaFree(d_minBounds));
    checkCudaErrors(cudaFree(d_maxBounds));
}

template <typename T>
std::string createFinalGraphDataFile()
{
    //------------------------------------------
    // create data file for drawing graph
    //------------------------------------------

    std::ostringstream graphDataFile;
    graphDataFile << typeid(T).name() << "_" << g_devName << "_graphData_v1.dat";

    std::string filePath = rd::findPath("gnuplot_data/", graphDataFile.str());
    std::ofstream gdataFile(filePath.c_str(), std::ios::out | std::ios::trunc);
    if (gdataFile.fail())
    {
        throw std::logic_error("Couldn't open file: " + graphDataFile.str());
    }

    auto printData = [&gdataFile](std::vector<float> const &v, std::string secName)
    {
        gdataFile << "# [" << secName << "] \n";
        for (size_t i = 0; i < v.size()/2; ++i)
        {
            gdataFile << std::right << std::fixed << std::setw(5) << std::setprecision(1) <<
                v[2 * i] << " " << v[2 * i + 1] << "\n";
        }
        // two sequential blank records to reset $0 counter
        gdataFile << "\n\n";
    };

    printData(g_bestPerf[0], "ROW-(CUB)");
    printData(g_bestPerf[1], "COL-(CUB)");
    printData(g_bestPerf[2], "ROW-(trove)");
    printData(g_bestPerf[3], "COL-(trove)");

    gdataFile.close();
    return filePath;
}

template <typename T>
void drawFinalGraph(std::string graphDataFilePath)
{
        //------------------------------------------
        // drawing graph
        //------------------------------------------

        rd::GraphDrawer<float> gDrawer;
        std::ostringstream graphName;
        graphName << typeid(T).name() << "_" << g_devName << "_hist_performance_v1.png";
        std::string filePath = rd::findPath("img/", graphName.str());

        gDrawer.sendCmd("set output '" + filePath + "'");
        gDrawer.setXLabel("Wymiar danych.");
        gDrawer.setYLabel("GB/s");

        gDrawer.sendCmd("set key right top");
        gDrawer.sendCmd("set style fill solid 0.95 border rgb 'grey30'");

        gDrawer.sendCmd("colStep = " + std::to_string(g_graphColStep));
        gDrawer.sendCmd("bs = 2 * colStep");
        gDrawer.sendCmd("nCol = " + std::to_string(g_graphNCol));
        gDrawer.sendCmd("groupStep = (nCol+1) * bs");
        gDrawer.sendCmd("nGroups = " + std::to_string(g_graphNGroups));
        gDrawer.sendCmd("offset = 9 * colStep");
        gDrawer.sendCmd("xEnd = offset + (nGroups-1) * groupStep + 9 * colStep + 4");

        gDrawer.sendCmd("set xrange [0:xEnd]");
        gDrawer.sendCmd("set xtics nomirror out ('2D' offset,'3D' offset + groupStep,"
             "'4D' offset + 2*groupStep, '5D' offset + 3*groupStep, '6D' offset + 4*groupStep)");
        gDrawer.sendCmd("dataFile = '" + graphDataFilePath + "'");

        std::ostringstream cmd;
        cmd << "plot dataFile i 0 u (offset + $0 * groupStep - 3 * colStep):2:(bs) t 'ROW (CUB)' w boxes ls 1,";
        cmd << "    ''        i 1 u (offset + $0 * groupStep - 1 * colStep):2:(bs) t 'COL (CUB)' w boxes ls 2,";
        cmd << "    ''        i 2 u (offset + $0 * groupStep + 1 * colStep):2:(bs) t 'ROW (trove)' w boxes ls 3,";
        cmd << "    ''        i 3 u (offset + $0 * groupStep + 3 * colStep):2:(bs) t 'COL (trove)' w boxes ls 4,";
        cmd << "    ''        i 0 u (offset + $0 * groupStep - 3 * colStep):($2 + 0.6):2 notitle w labels rotate by 70 left,";
        cmd << "    ''        i 1 u (offset + $0 * groupStep - 1 * colStep):($2 + 0.6):2 notitle w labels rotate by 70 left,";
        cmd << "    ''        i 2 u (offset + $0 * groupStep + 1 * colStep):($2 + 0.6):2 notitle w labels rotate by 70 left,";
        cmd << "    ''        i 3 u (offset + $0 * groupStep + 3 * colStep):($2 + 0.6):2 notitle w labels rotate by 70 left ";

        gDrawer.sendCmd(cmd.str());
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
            "\t\t[--a=<spiral param>]\n"
            "\t\t[--b=<spiral param>]\n"
            "\t\t[--s=<spiral noise sigma>]\n"
            "\t\t[--d=<device id>]\n"
            "\t\t[--v <verbose>]\n"
            "\t\t[--f=<file name to load>]\n"
            "\t\t[--g <draw graphs>]\n"
            "\n", argv[0]);
        exit(0);
    }

    if (args.CheckCmdLineFlag("f"))
    {
        args.GetCmdLineArgument("f", fSParams.file);
        args.GetCmdLineArgument("f", dSParams.file);
        fSParams.loadFromFile = true;
        dSParams.loadFromFile = true;
    }
    else
    {
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
    }
    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", fParams.devId);
        args.GetCmdLineArgument("d", dParams.devId);
    }
    if (args.CheckCmdLineFlag("v")) 
    {
        fParams.verbose = true;
        dParams.verbose = true;
    }
    if (args.CheckCmdLineFlag("g")) 
    {
        g_drawGraphs = true;
    }

    checkCudaErrors(deviceInit(fParams.devId));

    // set device name for logging and drawing purposes
    fParams.devId = (fParams.devId != -1) ? fParams.devId : 0;
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, fParams.devId));
    g_devName = devProp.name;

    if (g_drawGraphs)
    {
        // initialize storage for graph data
        g_bestPerf = std::vector<std::vector<float>>(g_graphNCol);
    }

    //-----------------------------------------
    //  TESTS
    //-----------------------------------------

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 2D: " << std::endl;
    test<2>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 3D: " << std::endl;
    test<3>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 4D: " << std::endl;
    test<4>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 5D: " << std::endl;
    test<5>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 6D: " << std::endl;
    test<6>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;

    if (g_drawGraphs)
    {
        drawFinalGraph<float>(createFinalGraphDataFile<float>());

        g_bestPerf.clear();
        g_bestPerf = std::vector<std::vector<float>>(8);
    }

    std::cout << "DOUBLE 2D: " << std::endl;
    test<2>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE 3D: " << std::endl;
    test<3>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE 4D: " << std::endl;
    test<4>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE 5D: " << std::endl;
    test<5>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE 6D: " << std::endl;
    test<6>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;

    if (g_drawGraphs)
    {
        drawFinalGraph<double>(createFinalGraphDataFile<double>());

        g_bestPerf.clear();
    }

    checkCudaErrors(deviceReset());

    std::cout << "END!" << std::endl;
    return 0;
}
