/**
 * @file test_tiled_global_decimate.cu
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */

#include <helper_cuda.h>

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>

// #ifndef RD_DEBUG
// #define NDEBUG      // for disabling assert macro
// #endif 
#include <assert.h>

#if defined(RD_DEBUG) && !defined(CUB_STDERR)
#define CUB_STDERR 
#endif
 
#include "rd/gpu/device/tiled/tiled_tree.cuh"
#include "rd/gpu/device/tiled/tree_drawer.cuh"
#include "rd/gpu/device/device_choose.cuh"
#include "rd/gpu/device/device_tiled_decimate.cuh"

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "tests/test_util.hpp"

#include "cub/test_util.h"

//----------------------------------------------
// global variables / constants
//----------------------------------------------

static constexpr int BLOCK_THREADS      = 128;
static constexpr int POINTS_PER_THREAD  = 4;
static constexpr int MAX_TEST_DIM       = 3;
static constexpr int MAX_POINTS_NUM     = int(1e7);
static constexpr int RD_CUDA_MAX_SYNC_DEPTH = 10;

static constexpr size_t HUNDRED_MB_IN_BYTES = 100 * 1024 * 1024;

static int g_devId              = 0;
static bool g_verbose           = false;
static std::string g_devName    = "";

//------------------------------------------------------------------------

static void configureDevice(
    size_t neededMemSize)
{
    checkCudaErrors(cudaDeviceReset());
    checkCudaErrors(cudaSetDevice(g_devId));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, RD_CUDA_MAX_SYNC_DEPTH));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, neededMemSize));
}

//------------------------------------------------------------------------
template <
    int                             DIM,
    rd::DataMemoryLayout            IN_MEM_LAYOUT,
    rd::DataMemoryLayout            OUT_MEM_LAYOUT,
    typename                        T>
struct TileProcessOp
{
    T r1;

    __device__ __forceinline__ TileProcessOp(
        T r1)
    :
        r1(r1)
    {}

    template <typename NodeT>
    __device__ __forceinline__ void operator()(NodeT * node) const
    {
        __syncthreads();
        #ifdef RD_DEBUG
        if (threadIdx.x == 0)
        {
            _CubLog("*** TileProcessOp ***** node id: %d, "
                "chosenPointsCapacity: %d, chosenSamples: [%p -- %p]\n",
                node->id, node->chosenPointsCapacity, node->chosenSamples,
                (OUT_MEM_LAYOUT == rd::COL_MAJOR) ? 
                node->chosenSamples + node->chosenSamplesStride * DIM : 
                node->chosenSamples + node->chosenPointsCapacity * DIM);
        }
        #endif

        if (threadIdx.x == 0)
        {
            int * d_tileChosenPtsNum = new int();
            assert(d_tileChosenPtsNum != nullptr);

            *d_tileChosenPtsNum = 0;
            // choose points within this tile
            cudaError_t err = cudaSuccess;
            err = rd::gpu::bruteForce::DeviceChoose::choose<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
                    node->samples, node->chosenSamples, node->pointsCnt, d_tileChosenPtsNum,
                    r1, node->samplesStride, node->chosenSamplesStride);
            rdDevCheckCall(err);
            rdDevCheckCall(cudaDeviceSynchronize());

            node->chosenPointsCnt = *d_tileChosenPtsNum;
            delete d_tileChosenPtsNum;
        }
        __syncthreads();
        #ifdef RD_DEBUG
        if (threadIdx.x == 0)
        {
            _CubLog("*** TileProcessOp ***** node id: %d, chosenPointsCnt: %d\n",
                node->id, node->chosenPointsCnt);
        }
        #endif
    }
};

//------------------------------------------------------------------------
// Test kernels
//------------------------------------------------------------------------

template <
    typename                TiledTreeT,
    typename                TileProcessOpT,
    int                     DIM,
    typename                T>
__launch_bounds__ (1)
static __global__ void buildTreeKernel(
    TiledTreeT *                        tree,
    T const *                           inputPoints,
    int                                 pointsNum,
    rd::gpu::BoundingBox<DIM, T> *      d_globalBBox,
    int                                 maxTileCapacity,
    T                                   sphereRadius,
    T                                   extensionFactor,
    cub::ArrayWrapper<int, DIM> const   initTileCntPerDim,
    int                                 inPtsStride)
{
    // last arg: true -> debugSynchronous
    new(tree) TiledTreeT(maxTileCapacity, sphereRadius, extensionFactor, true);

    TileProcessOpT tileProcessOp(sphereRadius);
    cudaStream_t buildTreeStream;
    rdDevCheckCall(cudaStreamCreateWithFlags(&buildTreeStream, cudaStreamNonBlocking));

    rdDevCheckCall(tree->buildTree(
        inputPoints, pointsNum, initTileCntPerDim, d_globalBBox, tileProcessOp, buildTreeStream, 
        inPtsStride));
    rdDevCheckCall(cudaStreamDestroy(buildTreeStream));
    rdDevCheckCall(cudaDeviceSynchronize());
}

template <
    typename                TiledTreeT,
    rd::DataMemoryLayout    OUT_MEM_LAYOUT,
    int                     DIM,
    typename                T>
__launch_bounds__ (1)
static __global__ void globalDecimateKernel(
    TiledTreeT * tree,
    T            sphereRadius2)
{
    /*
     * Main part of ridge detection
     */
    typedef typename TiledTreeT::NodeT NodeT;

    // allocate table for pointers to leaf nodes 
    NodeT **d_treeLeafs = new NodeT*[*tree->d_leafCount];
    int * d_chosenPointsNum = new int();
    assert(d_treeLeafs != nullptr);
    assert(d_chosenPointsNum != nullptr);

    int leafCounter = 0;

    // collect and initialize leafs
    tree->forEachNodePreorder(
        [d_chosenPointsNum, &d_treeLeafs, &leafCounter](NodeT * node) {
        // am I a non-empty leaf?
        if (!node->haveChildren() && !node->empty())
        {
            d_treeLeafs[leafCounter++] = node;
            atomicAdd(d_chosenPointsNum, node->chosenPointsCnt);
            
            // init evolve flag
            node->needEvolve = 1;
            #ifdef RD_DEBUG
            {
                printf("node: %d, chosenPointsCnt: %d\n", node->id, node->chosenPointsCnt);
            }
            #endif
        }
    });

    #ifdef RD_DEBUG
    if (threadIdx.x == 0)
    {
        _CubLog("\n>>>>> ---------globalDecimate---------\n", 1);
    }
    #endif

    // last arg 'true' -> debugSynchronous
    rd::gpu::tiled::DeviceDecimate::globalDecimate<DIM, OUT_MEM_LAYOUT>(
        d_treeLeafs, leafCounter, d_chosenPointsNum, sphereRadius2, nullptr, true);
    rdDevCheckCall(cudaDeviceSynchronize());

    delete[] d_treeLeafs;
    delete d_chosenPointsNum;
}

template <
    typename TiledTreeT>
__launch_bounds__ (1)
static __global__ void deleteTiledTree(
    TiledTreeT *          tree)
{
    tree->~TiledTreeT();
}

//------------------------------------------------------------------------

template <
    int DIM,
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    typename T>
void testBuildTree(
    T const *   h_inputPoints,
    int         pointsNum,
    T           sphereRadius, 
    T           sphereRadius2)
{
    typedef rd::gpu::tiled::TiledTreePolicy<
        BLOCK_THREADS,
        POINTS_PER_THREAD,
        cub::LOAD_LDG,
        rd::gpu::IO_BACKEND_CUB>
    TiledTreePolicyT;

    typedef rd::gpu::tiled::TiledTree<
        TiledTreePolicyT, 
        DIM, 
        IN_MEM_LAYOUT, 
        OUT_MEM_LAYOUT, 
        T>
    TiledTreeT;

    typedef TileProcessOp<
        DIM,
        IN_MEM_LAYOUT,
        OUT_MEM_LAYOUT,
        T>
    TileProcessOpT;

    int     maxTileCapacity = 0.17 * pointsNum;
    // int     maxTileCapacity = pointsNum;
    T       extensionFactor = 1.35;
    T *     d_inputPoints;
    TiledTreeT * d_tree;
    cub::ArrayWrapper<int, DIM> initTileCntPerDim;
    int inPtsStride = DIM;

    if (IN_MEM_LAYOUT == rd::ROW_MAJOR)
    {
        checkCudaErrors(cudaMalloc(&d_inputPoints, pointsNum * DIM * sizeof(T)));
    }
    else if (IN_MEM_LAYOUT == rd::COL_MAJOR)
    {
        size_t pitch = 0;
        checkCudaErrors(cudaMallocPitch(&d_inputPoints, &pitch, pointsNum * sizeof(T),
            DIM));
        inPtsStride = pitch / sizeof(T);
    }
    else
    {
        throw std::runtime_error("Unsupported memory layout!");
    }
    checkCudaErrors(cudaMalloc(&d_tree, sizeof(TiledTreeT)));

    // Hardcoded for 3 tiles per dimension
    for (int k =0; k < DIM; ++k)
    {
        initTileCntPerDim.array[k] = 2;
        // initTileCntPerDim.array[k] = 1;
    }

    if (IN_MEM_LAYOUT == rd::ROW_MAJOR)
    {
        rd::gpu::rdMemcpy<rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_inputPoints, h_inputPoints, DIM, pointsNum, DIM, inPtsStride);
    }
    else if (IN_MEM_LAYOUT == rd::COL_MAJOR)
    {

        rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_inputPoints, h_inputPoints, DIM, pointsNum, inPtsStride * sizeof(T),
            DIM * sizeof(T));
    }
    else
    {
        throw std::runtime_error("Unsupported memory layout!");
    }
    checkCudaErrors(cudaDeviceSynchronize());

    rd::gpu::BoundingBox<DIM, T> globalBBox;
    checkCudaErrors(globalBBox.template findBounds<IN_MEM_LAYOUT>(
        d_inputPoints, pointsNum, inPtsStride));
    globalBBox.calcDistances();

    // allocate & copy memory for device global bounding box
    rd::gpu::BoundingBox<DIM, T> *d_globalBBox;
    checkCudaErrors(cudaMalloc(&d_globalBBox, sizeof(rd::gpu::BoundingBox<DIM,T>)));
    checkCudaErrors(cudaMemcpy(d_globalBBox, &globalBBox, sizeof(rd::gpu::BoundingBox<DIM,T>), 
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Invoking buildTreeKernel" << std::endl;

    buildTreeKernel<TiledTreeT, TileProcessOpT, DIM><<<1,1>>>(
        d_tree, d_inputPoints, pointsNum, d_globalBBox, maxTileCapacity, sphereRadius, 
        extensionFactor, initTileCntPerDim, inPtsStride);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // draw initial chosen points
    rd::gpu::tiled::util::TreeDrawer<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, 
        TiledTreeT, T> treeDrawer(d_tree, d_inputPoints, pointsNum, inPtsStride);
    treeDrawer.drawBounds();
    treeDrawer.drawEachTile();

    std::cout << "Invoking globalDecimateKernel" << std::endl;

    globalDecimateKernel<TiledTreeT, OUT_MEM_LAYOUT, DIM><<<1,1>>>(d_tree, sphereRadius2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // draw chosen points after decimation
    treeDrawer.collectTreeData();
    treeDrawer.drawEachTile();

    deleteTiledTree<TiledTreeT><<<1, 1>>>(d_tree);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());    

    //-------------------------------------------------------------------------------
    // clean-up
    //-------------------------------------------------------------------------------

    checkCudaErrors(cudaFree(d_tree));
    checkCudaErrors(cudaFree(d_inputPoints));
    checkCudaErrors(cudaFree(d_globalBBox));
}

template <
    int DIM,
    typename T>
void testMemLayout(
    int pointNum,
    PointCloud<T> const & pc)
{
    std::vector<T> && points = pc.extractPart(pointNum, DIM);
    T sphereRadius = 1.45f * pc.stddev_;
    T sphereRadius2 = 2.f * sphereRadius;

    if (g_verbose && DIM <= 3)
    {
        rd::GraphDrawer<T> gDrawer;
        std::ostringstream os;
        os << typeid(T).name() << "_" << DIM << "D_" << g_devName;
        os << "_initial_samples_set";
        gDrawer.showPoints(os.str(), points.data(), pointNum, DIM);
    }

    std::cout << rd::HLINE << std::endl;
    std::cout << "<<<<< ROW_MAJOR, ROW_MAJOR >>>>>" << std::endl;
    testBuildTree<DIM, rd::ROW_MAJOR, rd::ROW_MAJOR>(points.data(), pointNum, sphereRadius, 
        sphereRadius2);

    std::cout << rd::HLINE << std::endl;
    std::cout << "<<<<< COL_MAJOR, ROW_MAJOR >>>>>" << std::endl;
    testBuildTree<DIM, rd::COL_MAJOR, rd::ROW_MAJOR>(points.data(), pointNum, sphereRadius, 
        sphereRadius2);

    std::cout << rd::HLINE << std::endl;
    std::cout << "<<<<< COL_MAJOR, COL_MAJOR >>>>>" << std::endl;
    testBuildTree<DIM, rd::COL_MAJOR, rd::COL_MAJOR>(points.data(), pointNum, sphereRadius, 
        sphereRadius2);
}

/**
 * @brief helper structure for static for loop over dimension
 */
struct IterateDimensions
{
    template <typename D, typename T>
    static void impl(
        D   idx,
        int pointNum,
        PointCloud<T> const & pc)
    {
        std::cout << rd::HLINE << std::endl;
        std::cout << ">>>> Dimension: " << idx << "D\n";
        testMemLayout<D::value>(pointNum, pc);
    }
};

template <
    int          DIM,
    typename     T>
struct TestDimensions
{
    static void impl(
        PointCloud<T> & pc,
        int pointNum)
    {
        static_assert(DIM != 0, "DIM equal to zero!\n");

        pc.pointCnt_ = pointNum;
        pc.initializeData();

        size_t neededMemSize = 5 * pointNum * DIM * sizeof(T);
        neededMemSize = std::max(HUNDRED_MB_IN_BYTES, neededMemSize);

        std::cout << "Reserve " << float(neededMemSize) / 1024.f / 1024.f 
            << " Mb for malloc heap size" << std::endl;
        configureDevice(neededMemSize);

        std::cout << rd::HLINE << std::endl;
        std::cout << ">>>> Dimension: " << DIM << "D\n";
        testMemLayout<DIM>(pointNum, pc);
    }
};

template <typename T>
struct TestDimensions<0, T>
{
    static void impl(
        PointCloud<T> & pc,
        int pointNum)
    {
        pc.pointCnt_ = pointNum;
        pc.dim_ = MAX_TEST_DIM;
        pc.initializeData();

        size_t neededMemSize = 10 * pointNum * MAX_TEST_DIM * sizeof(T);
        neededMemSize = std::max(HUNDRED_MB_IN_BYTES, neededMemSize);

        std::cout << "Reserve " << float(neededMemSize) / 1024.f / 1024.f 
            << " Mb for malloc heap size" << std::endl;
        configureDevice(neededMemSize);

        StaticFor<1, MAX_TEST_DIM, IterateDimensions>::impl(pointNum, pc);
    }
};

template <
    typename    T,
    int         DIM = 0,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void testSize(
    PointCloud<T> & pc,
    int pointNum = -1)
{
    if (pointNum > 0)
    {
        TestDimensions<DIM, T>::impl(pc, pointNum);
    }
    else
    {
        for (int k = 1000; k <= MAX_POINTS_NUM; k *= 10)
        {
            std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t pointNum: " << k  
                    << "\n//------------------------------------------\n";

            TestDimensions<DIM, T>::impl(pc, k);
        }
    }
}

int main(int argc, char const **argv)
{
    float a = -1.f, b = -1.f, stddev = -1.f;
    int pointNum = -1;

    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help")) 
    {
        printf("%s \n"
            "\t\t[--a <a parameter of spiral or length if dim > 3>]\n"
            "\t\t[--b <b parameter of spiral or ignored if dim > 3>]\n"
            "\t\t[--stddev <standard deviation of generated samples>]\n"
            "\t\t[--size <number of points>]\n"
            "\t\t[--d=<device id>]\n"
            "\t\t[--v <verbose>]\n"
            "\n", argv[0]);
        exit(0);
    }

    if (args.CheckCmdLineFlag("a"))
    {
        args.GetCmdLineArgument("a", a);
    }
    if (args.CheckCmdLineFlag("b"))
    {
        args.GetCmdLineArgument("b", b);
    }
    if (args.CheckCmdLineFlag("stddev"))
    {
        args.GetCmdLineArgument("stddev", stddev);
    }
    if (args.CheckCmdLineFlag("size"))
    {
        args.GetCmdLineArgument("size", pointNum);
    }    
    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", g_devId);
    }
    if (args.CheckCmdLineFlag("v")) 
    {
        g_verbose = true;
    }

    checkCudaErrors(deviceInit(g_devId));

    // set device name for logging and drawing purposes
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, g_devId));
    g_devName = devProp.name;

    #ifdef QUICK_TEST
        if (pointNum < 0)
        {
            pointNum = 50000;
        }

        const int dim = 3;
        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t (spiral) float: "  
        //             << "\n//------------------------------------------\n";
        // PointCloud<float> && fpc = SpiralPointCloud<float>(a, b, pointNum, 2, stddev);
        // TestDimensions<2, float>::impl(fpc, pointNum);

        PointCloud<float> && fpc2 = SegmentPointCloud<float>(1000.f, pointNum, dim, stddev);
        TestDimensions<dim, float>::impl(fpc2, pointNum);

        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t (spiral) double: "  
        //             << "\n//------------------------------------------\n";
        // PointCloud<double> && dpc = SpiralPointCloud<double>(a, b, pointNum, 2, stddev);
        // TestDimensions<2, double>::impl(dpc, pointNum);

    #else
        // 1e6 2D points, spiral a=22, b=10, stddev=4
        #ifndef RD_DOUBLE_PRECISION
        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t (spiral) float: "  
        //             << "\n//------------------------------------------\n";
        // PointCloud<float> && fpc2d = SpiralPointCloud<float>(22.f, 10.f, 0, 2, 4.f);
        // PointCloud<float> && fpc3d = SpiralPointCloud<float>(22.f, 10.f, 0, 3, 4.f);
        // TestDimensions<2, float>::impl(fpc2d, int(1e6));
        // TestDimensions<3, float>::impl(fpc3d, int(1e6));
        

        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) float: "  
                    << "\n//------------------------------------------\n";
        PointCloud<float> && fpc2 = SegmentPointCloud<float>(1000.f, 0, 0, 4.f);
        testSize<float>(fpc2);
        #else
        // std::cout << "\n//------------------------------------------" 
        //             << "\n//\t\t (spiral) double: "  
        //             << "\n//------------------------------------------\n";
        // PointCloud<double> && dpc2d = SpiralPointCloud<double>(22.0, 10.0, 0, 2, 4.0);
        // PointCloud<double> && dpc3d = SpiralPointCloud<double>(22.0, 10.0, 0, 3, 4.0);
        // TestDimensions<2, double>::impl(dpc2d, int(1e6));
        // TestDimensions<3, double>::impl(dpc3d, int(1e6));
        
        std::cout << "\n//------------------------------------------" 
                    << "\n//\t\t (segment) double: "  
                    << "\n//------------------------------------------\n";
        PointCloud<double> && dpc2 = SegmentPointCloud<double>(1000.0, 0, 0, 4.0);
        testSize<double>(dpc2);
        #endif
    #endif

    checkCudaErrors(cudaDeviceReset());
    
    std::cout << rd::HLINE << std::endl;
    std::cout << "END!" << std::endl;
 
    return EXIT_SUCCESS;
}

