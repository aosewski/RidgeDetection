/**
 * @file tiled_tree.cuh
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is conducted under the supervision of prof. dr hab. inż. Marka
 * Nałęcza.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */

#pragma once

#ifndef BLOCK_TILE_LOAD_V4
#   define BLOCK_TILE_LOAD_V4 1
#endif

#include "rd/utils/macro.h"
#include "rd/utils/memory.h"

#include "rd/gpu/util/dev_arch.cuh"
#include "rd/gpu/util/dev_utilities.cuh"
#include "rd/gpu/block/block_tile_load_store4.cuh"

#include "rd/gpu/device/tiled/tiled_tree_root.cuh"
#include "rd/gpu/device/tiled/tiled_tree_node.cuh"
#include "rd/gpu/device/tiled/agent_build_tiled_tree.cuh"
#include "rd/gpu/device/device_spatial_histogram.cuh"
#include "rd/gpu/device/bounding_box.cuh"

#include "cub/util_arch.cuh"
#include "cub/util_type.cuh"
#include "cub/util_device.cuh"
#include "cub/util_debug.cuh"
#include "cub/thread/thread_load.cuh"

#include <assert.h>
#include <utility>

namespace rd
{
namespace gpu
{
namespace tiled
{
namespace detail 
{

template <
    int         DIM,
    typename    T>
static __global__ void initTileBoundsKernel(
    TiledTreeNode<DIM, T> *             nodes,
    cub::ArrayWrapper<int, DIM> const   initTileCntPerDim,
    int                                 initTileCnt,
    BoundingBox<DIM, T> * const         parentBBox)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < initTileCnt)
    {
        nodes[tid].initTileBounds(tid, initTileCntPerDim, initTileCnt, *parentBBox);
    }
}
 
template <
    int                             BLOCK_THREADS,
    int                             POINTS_PER_THREAD,
    DataMemoryLayout                IN_MEM_LAYOUT,
    DataMemoryLayout                OUT_MEM_LAYOUT,
    int                             DIM,
    typename                        SampleT,
    typename                        TileProcessingOpT>
__launch_bounds__ (BLOCK_THREADS)
static __global__ void addTreeRootNodesKernel(
    TiledTreeNode<DIM, SampleT> *       nodes,
    SampleT const *                     samples,
    int                                 pointsCnt,
    int *                               pointsHist,
    int *                               neighboursHist,
    int                                 maxTileCapacity,
    SampleT                             sphereRadius,
    TileProcessingOpT                   tileProcessingOp,
    int *                               d_leafCount,
    int                                 stride)
{

    typedef AgentBuildTiledTree<
        BLOCK_THREADS, 
        POINTS_PER_THREAD, 
        DIM,
        IN_MEM_LAYOUT,
        OUT_MEM_LAYOUT,
        TileProcessingOpT,
        SampleT> AgentBuildTiledTreeT;

    // Shared memory needed for this block of threads
    __shared__ typename AgentBuildTiledTreeT::TempStorage tempStorage;

    // TODO: use dynamic grid mapping
    AgentBuildTiledTreeT(
        tempStorage,
        nodes + blockIdx.x, 
        sphereRadius,
        tileProcessingOp).addRootNode(
            samples,
            pointsCnt,
            pointsHist[blockIdx.x],
            neighboursHist[blockIdx.x],
            maxTileCapacity, 
            sphereRadius,
            d_leafCount,
            stride);
}

} // end namespace detail

//--------------------------------------------------------------------------------
//
//  BUILDING TILED TREE
//
//--------------------------------------------------------------------------------

template <
    int                         _BLOCK_THREADS,
    int                         _POINTS_PER_THREAD>
struct TiledTreePolicy
{
    enum 
    {
        BLOCK_THREADS           = _BLOCK_THREADS,
        POINTS_PER_THREAD       = _POINTS_PER_THREAD,
    };
};


/**
 * @brief      Provides abstraction of cut-off bin processing. Splits input data into tiles and
 *             process them concurrently.
 *
 * @tparam     TiledTreePolicyT  Policy used for running addRootNodes kernel.
 * @tparam     DIM               Input data point dimension
 * @tparam     IN_MEM_LAYOUT     Input data memory layout (input points).
 * @tparam     OUT_MEM_LAYOUT    Output data memory layout (chosen points).
 * @tparam     SampleT           Input data point coordinate type.
 */
template <
    typename            TiledTreePolicyT,
    int                 DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT,
    typename            SampleT>
class TiledTree
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS           = TiledTreePolicyT::BLOCK_THREADS,
        POINTS_PER_THREAD       = TiledTreePolicyT::POINTS_PER_THREAD,
    };

public:
    typedef TiledTreeNode<DIM, SampleT>     NodeT;
    typedef TiledTreeRoot<NodeT>            RootT;
    typedef typename NodeT::BBoxT           BBoxT;

    // Mamximum number of points inside tile, after which we split it on two halves.
    int maxTileCapacity;
    // Ridge detection choose phase parameter, needed for chosen points count estimation.
    SampleT sphereRadius;
    // wheather or not to synchronize after each kernel launch
    bool debugSynchronous;
    // number of builded tree leafs.
    int * d_leafCount;

    #ifdef RD_DRAW_TREE_TILES
    unsigned int * d_referenceCount;
    #endif

    /**
     * @brief      Constructor
     *
     * @param[in]  tileCapacity    All tiles count. This is the product of @p initTilesCntPerDim
     * @param[in]  radius          Ridge detection parameter for choosing samples.
     * @param[in]  dbgSynchronous  Whether to synchronize after each kernel call.
     */
    __device__ __forceinline__ TiledTree(
        int         tileCapacity,
        SampleT     radius,
        bool        dbgSynchronous = false)
    :
        maxTileCapacity(tileCapacity),
        sphereRadius(radius),
        debugSynchronous(dbgSynchronous),
        d_leafCount(nullptr)
    {
        #ifdef RD_DRAW_TREE_TILES
        {
            d_referenceCount = new unsigned int();
            assert(d_referenceCount != nullptr);
            *d_referenceCount = 1;
        }
        #endif
        #ifdef RD_DEBUG
            printf("TiledTree() maxTileCapacity: %d, sphereRadius %f\n",
                maxTileCapacity, sphereRadius);
        #endif
    }

    __device__ __forceinline__ ~TiledTree()
    {
        #ifdef RD_DRAW_TREE_TILES
        #ifdef RD_DEBUG
        printf("~TiledTree() *d_referenceCount: %d\n", *d_referenceCount);
        #endif
        (*d_referenceCount)--;
        if (*d_referenceCount == 0)
        {
            delete d_referenceCount;
        }
        #else
        #ifdef RD_DEBUG
            printf("~TiledTree() \n", 1);
        #endif
        #endif
        if (d_leafCount != nullptr)
        {
            delete d_leafCount;
        }
        // printf("~TiledTree() end!\n");
    }

    /**
     * @brief      Builds a Tree.
     *
     * @param[in]  samples            Pointer to source data.
     * @param[in]  pointsCnt          Number of points in @p samples set
     * @param      initTileCntPerDim  Number of initial tiles per dimension, to split data into.
     * @param[in]  tileProcessingOp   The tile processing op
     * @param[in]  stride             Number of samples between single point subsequent coordinates.
     * @param[in]  stream             Stream to run computation in.
     *
     * @tparam     TileProcessingOpT  Functor for tile data processing.
     *
     * @return     cudaError or cudaSuccess.
     *
     * @note       This should be run with a single thread kernel.
     */
    template <typename TileProcessingOpT>
    __device__ __forceinline__ cudaError_t buildTree(
        SampleT const *                     samples,
        int                                 pointsCnt,
        cub::ArrayWrapper<int, DIM> const   initTileCntPerDim,
        BBoxT *                             globalBbox,
        TileProcessingOpT                   tileProcessingOp,
        cudaStream_t                        stream,      
        int                                 stride)
    {
        cudaError error = cudaSuccess;
        #ifndef CUB_RUNTIME_ENABLED
            // Kernel launch not supported from this device
            CubDebug(error = cudaErrorNotSupported);
        #else
        do {
            // initialize leaf counter
            if (d_leafCount == nullptr)
            {
                d_leafCount = new int(0);
                assert(d_leafCount != nullptr);
            }
            *d_leafCount = 0;

            int initTileCnt = 1;
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                initTileCnt *= initTileCntPerDim.array[d];
            }
            root.init(initTileCnt);

            #ifdef RD_DEBUG
            {
                printf("buildTree(): initTileCnt: %d, samples %p, pCnt %d, stride %d\n",
                    initTileCnt, samples, pointsCnt, stride);
            }
            #endif

            // Get device ordinal
            int deviceOrdinal;
            if (CubDebug(error = cudaGetDevice(&deviceOrdinal))) break;

            // get max x-dimension of grid
            int maxDimX;
            if (CubDebug(error = cudaDeviceGetAttribute(
                &maxDimX, cudaDevAttrMaxGridDimX, deviceOrdinal))) break;

            // get grid size for adding new tiles
            dim3 gridSize(1);
            gridSize.y = ((unsigned int) initTileCnt + maxDimX - 1) / maxDimX;
            gridSize.x = CUB_MIN((initTileCnt + 127) / 128, maxDimX);

            // log initTileBoundsKernel configuration
            if (debugSynchronous)
            {
                printf("globalBbox: \n");
                globalBbox->print();
                printf("Invoking initTileBoundsKernel<%d><<<{%d, %d, %d}, %d, 0, %p>>>,"
                    " initTileCnt: %d\n",
                    DIM, gridSize.x, gridSize.y, gridSize.z, 128, stream, initTileCnt);
            }

            //---------------------------------
            // calculate each tile's bounds
            //---------------------------------
            detail::initTileBoundsKernel<<<gridSize, 128, 0, stream>>>(
                root.children,
                initTileCntPerDim,
                initTileCnt,
                globalBbox);
                
            // Check for failure to launch
            if (error = CubDebug(cudaPeekAtLastError())) break;
            if (error = CubDebug(cub::SyncStream(stream))) break;

            #ifdef RD_DEBUG
            {
                printf("nodes bounds: \n");
                for (int k = 0; k < root.childrenCount; ++k)
                {
                    root.children[k].bounds.print();
                }
            }
            #endif

            //---------------------------------------------------------
            // calculate number of points falling into respective tile
            //---------------------------------------------------------
            int * d_tilesPointsHist, *d_tilesNeighboursHist;
            
            d_tilesPointsHist       = new int[initTileCnt];
            d_tilesNeighboursHist   = new int[initTileCnt];
            
            assert(d_tilesPointsHist     != nullptr);
            assert(d_tilesNeighboursHist != nullptr);

            #ifdef RD_DEBUG
            {
                printf("buildTree(): invoking calcHistogram kernels \n", 1);
            }
            #endif

            calcHistogram(samples, pointsCnt, d_tilesPointsHist, d_tilesNeighboursHist,
                initTileCntPerDim.array, *globalBbox, root.children, initTileCnt, stride);

            #ifdef RD_DEBUG
            {
                for (int k = 0; k < initTileCnt; ++k)
                {
                    printf("Tile %d points: %d, neighbours: %d\n",
                        k, d_tilesPointsHist[k], d_tilesNeighboursHist[k]);
                }
            }
            #endif

            //---------------------------------------------------------
            // allocate containers for tile samples & neighbours and partition data.
            //---------------------------------------------------------

            gridSize.x = CUB_MIN(initTileCnt , maxDimX);

            // log addTreeRootNodesKernel configuration
            if (debugSynchronous)
            {
                printf("Invoking addTreeRootNodesKernel<%d, %d><<<{%d, %d, %d}, %d, 0, %p>>>, initTileCnt: %d\n",
                    BLOCK_THREADS, DIM, gridSize.x, gridSize.y, gridSize.z, BLOCK_THREADS, stream, initTileCnt);
            }

            detail::addTreeRootNodesKernel<
                BLOCK_THREADS, 
                POINTS_PER_THREAD,
                IN_MEM_LAYOUT,
                OUT_MEM_LAYOUT,
                DIM, 
                SampleT, 
                TileProcessingOpT>
                <<<gridSize, BLOCK_THREADS, 0, stream>>>(
                    root.children,
                    samples,
                    pointsCnt,
                    d_tilesPointsHist,
                    d_tilesNeighboursHist,
                    maxTileCapacity, 
                    sphereRadius,
                    tileProcessingOp,
                    d_leafCount,
                    stride);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;
            // wait for kernel end to release histograms
            if (CubDebug(error = cub::SyncStream(stream))) break;

            delete[] d_tilesPointsHist;
            delete[] d_tilesNeighboursHist;

        } 
        while (0);
        #endif
        return error;
    }

    /**
     * @brief      Visit each tree node and perform @p f function on it.
     *
     * @param      f              Functor object defining function to perform.
     *
     * @tparam     UnaryFunction  Functor class with defined function to perform
     * on nodes' data.
     */
    template <typename UnaryFunction>
    __device__ __forceinline__ void forEachNodePreorder(
        UnaryFunction const &f)
    {
        if (root.children == nullptr)
        {
            return;
        }
        else
        {
            for (int k = 0; k < root.childrenCount; ++k)
            {
                visitNode(root.children + k, f);
            }
        }
    }

    __device__ __forceinline__ RootT getRoot() const { return root; }

    __device__ __forceinline__ void setRoot(RootT const & r) { root = r; }
    __device__ __forceinline__ void setRoot(RootT && r) { root = std::move(r); }

    __device__ __forceinline__ TiledTree* clone(TiledTree * treeClone = nullptr) const
    {
        #ifdef RD_DEBUG
        printf("TiledTree::clone()\n");
        #endif
        if (treeClone != nullptr)
        {
            new(treeClone) TiledTree(maxTileCapacity, sphereRadius, 
                debugSynchronous);
        }
        else
        {
            treeClone = new TiledTree(maxTileCapacity, sphereRadius, 
                debugSynchronous);
        }
        if (d_leafCount != nullptr)
        {
            treeClone->d_leafCount = new int(0);
            *treeClone->d_leafCount = *d_leafCount;
        }
        treeClone->setRoot(root.template clone<IN_MEM_LAYOUT, OUT_MEM_LAYOUT>());
        return treeClone;
    }

    __device__ __forceinline__ TiledTree(TiledTree const & rhs) 
    {
        root = rhs.getRoot();
        maxTileCapacity = rhs.maxTileCapacity;
        sphereRadius = rhs.sphereRadius;
        debugSynchronous = rhs.debugSynchronous;
        d_leafCount = rhs.d_leafCount;
    #ifdef RD_DRAW_TREE_TILES
        d_referenceCount = rhs.d_referenceCount;
        (*d_referenceCount)++;
        #ifdef RD_DEBUG
            printf("TiledTree[copy constr], *d_referenceCount: %d\n", *d_referenceCount);
        #endif
    #endif
    }

    __device__ __forceinline__ TiledTree & operator=(TiledTree const & rhs)
    {
        root = rhs.getRoot();
        maxTileCapacity = rhs.maxTileCapacity;
        sphereRadius = rhs.sphereRadius;
        debugSynchronous = rhs.debugSynchronous;
        d_leafCount = rhs.d_leafCount;
    #ifdef RD_DRAW_TREE_TILES
        d_referenceCount = rhs.d_referenceCount;
        (*d_referenceCount)++;
        #ifdef RD_DEBUG
            printf("TiledTree[copy constr], *d_referenceCount: %d\n", *d_referenceCount);
        #endif
    #endif
        return *this;
    }

private:
    RootT root;

    /**
     * @brief      Calculates spatial histogram for @p d_samples set
     *
     * @param[in]  d_samples              Samples set we calculate histogram for.
     * @param[in]  pointsCnt              Number of points in @p d_samples set.
     * @param[out] d_tilesPointsHist      Output points histogram.
     * @param[out] d_tilesNeighboursHist  Output neighbours histogram.
     * @param[in]  binsCnt                Table with number of bins for respecitve dimensions.
     * @param[in]  bbox                   Bounding box of input points set
     * @param      nodes                  Nodes representing histogram bins
     * @param[in]  nodesCnt               Overall number of generated tiles.
     * @param[in]  stride                 Stride value for @d_samples set. Number of samples between
     *                                    point's consecutive coordinates.
     *                                    
     * @note       This should be run with a single thread.
     */
    __device__ __forceinline__ void calcHistogram(
        SampleT const * d_samples,
        int             pointsCnt,
        int *           d_tilesPointsHist,
        int *           d_tilesNeighboursHist,
        int const       (&binsCnt)[DIM],
        BBoxT const &   bbox,
        NodeT *         nodes,
        int             nodesCnt,
        int             stride) const
    {
        #ifndef CUB_RUNTIME_ENABLED
            // Kernel launch not supported from this device
            rdDevCheckCall(cudaErrorNotSupported);
        #else

            cudaStream_t pHistStream = 0, nHistStream = 0;
            rdDevCheckCall(cudaStreamCreateWithFlags(&pHistStream, cudaStreamNonBlocking));
            rdDevCheckCall(cudaStreamCreateWithFlags(&nHistStream, cudaStreamNonBlocking));

            #ifdef RD_DEBUG
                printf("nHistDecodeOp: nodes: (%p,%p), nodesCnt: %d\n",
                    nodes, nodes+nodesCnt-1, nodesCnt);
            #endif

            SampleT extRadius = this->sphereRadius;
            // decodeOp for neighbours histogram
            auto nHistDecodeOp = [nodes, nodesCnt, extRadius] (
                SampleT point[DIM],
                int *   d_histogram) 
            {
                for (int n = 0; n < nodesCnt; ++n)
                {
                    if (nodes[n].bounds.isNearby(point, extRadius))
                    {
                        atomicAdd(d_histogram + n, 1);
                    // #ifdef RD_DEBUG
                    // {
                    //     _CubLog(" ######## Got neighbour!: [%f, %f] ##########\n", point[0], point[1]);
                    // }
                    // #endif
                    }
                }
            };

            // query for and allocate temporary storage
            void * d_pHistTempStorage = nullptr, *d_nHistTempStorage = nullptr;
            size_t pHistTempStorageBytes = 0, nHistTempStorageBytes = 0;
            cudaError err = cudaSuccess;

            // if we exceed 1^10 of bins, we fallback to simple histogram impl, 
            // without use of private gmem histograms
            bool useGmemPrivHist = nodesCnt < 1 << 10;
            if (useGmemPrivHist)
            {
                err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_pHistTempStorage,
                    pHistTempStorageBytes, d_samples, d_tilesPointsHist, pointsCnt, binsCnt, bbox,
                    stride, useGmemPrivHist, pHistStream, debugSynchronous);
                rdDevCheckCall(err);

                err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_nHistTempStorage,
                    nHistTempStorageBytes, d_samples, d_tilesNeighboursHist, pointsCnt, nodesCnt, 
                    nHistDecodeOp, stride, false, useGmemPrivHist, nHistStream, debugSynchronous);
                rdDevCheckCall(err);

                d_pHistTempStorage = new char[pHistTempStorageBytes];
                d_nHistTempStorage = new char[nHistTempStorageBytes];

                assert(d_pHistTempStorage != nullptr);
                assert(d_nHistTempStorage != nullptr);
            }

            // run histograms
            if (debugSynchronous)
            {
                _CubLog("Invoking spatialHistogram for tile points! "
                    "pHistTempStorage: %p, tempStorageBytes: %lld, samples: %p, pointsCnt: %d"
                    " d_tilesPointsHist: %p, useGmemPrivHist %s\n",
                    d_pHistTempStorage, pHistTempStorageBytes, d_samples, pointsCnt,
                    d_tilesPointsHist, (useGmemPrivHist ? "true" : "false"));
            }

            err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_pHistTempStorage, 
                pHistTempStorageBytes, d_samples, d_tilesPointsHist, pointsCnt, binsCnt, bbox, 
                stride, useGmemPrivHist, pHistStream, debugSynchronous);
            rdDevCheckCall(err);

            if (debugSynchronous)
            {
                _CubLog("Invoking spatialHistogram for tile neighbours! " 
                    "nHistTempStorage: %p, nHistTempStorageBytes %lld, samples: %p, " 
                    "d_tilesNeighboursHist: %p, pointsCnt: %d, nodesCnt: %d, stride: %d"
                    ", useGmemPrivHist: %s\n",
                    d_nHistTempStorage, nHistTempStorageBytes, d_samples, d_tilesNeighboursHist, 
                    pointsCnt, nodesCnt, stride, (useGmemPrivHist ? "true" : "false"));
            }

            err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_nHistTempStorage, 
                nHistTempStorageBytes, d_samples, d_tilesNeighboursHist, pointsCnt, nodesCnt, 
                nHistDecodeOp, stride, false, useGmemPrivHist, nHistStream, debugSynchronous);
            rdDevCheckCall(err);

            rdDevCheckCall(cub::SyncStream(pHistStream));
            rdDevCheckCall(cub::SyncStream(nHistStream));

            rdDevCheckCall(cudaStreamDestroy(pHistStream));
            rdDevCheckCall(cudaStreamDestroy(nHistStream));

            if (d_pHistTempStorage != nullptr) delete[] d_pHistTempStorage;
            if (d_nHistTempStorage != nullptr) delete[] d_nHistTempStorage;
        #endif
    }

    /**
     * @brief      Visits @p node and all of its children and performs @p f
     *              function on each data.
     *
     * @param[in]  node           Node we want to visit.
     * @param      f              Functor object with function to perform.
     *
     * @tparam     UnaryFunction  Functor object with function which gets
     *              NodeT pointer.
     */
    template <typename UnaryFunction>
    __device__ __forceinline__ void visitNode(
        NodeT *                 node, 
        UnaryFunction const &   f)
    {
        f(node);
        if (node->left != nullptr)
        {
            visitNode(node->left, f);
        }
        if (node->right != nullptr)
        {
            visitNode(node->right, f);
        }
    }

};

} // end namespace tiled
} // end namespace gpu
} // end namespace rd