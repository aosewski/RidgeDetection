/**
 * @file agent_build_tiled_tree.cuh
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

#include "rd/utils/macro.h"
#include "rd/utils/memory.h"

#include "rd/gpu/device/tiled/tiled_tree_node.cuh"
#include "rd/gpu/device/bounding_box.cuh"
#include "rd/gpu/device/device_spatial_histogram.cuh"

#include "rd/gpu/block/block_select_if.cuh"

#include "cub/thread/thread_load.cuh"
#include "cub/util_type.cuh"

namespace rd
{
namespace gpu
{
namespace tiled
{

template <
    int         BLOCK_THREADS,
    int         POINTS_PER_THREAD,
    int         DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT,
    typename    TileProcessingOpT,
    typename    T>
class AgentBuildTiledTree;

namespace detail
{

template <
    int                             BLOCK_THREADS,
    int                             POINTS_PER_THREAD,
    DataMemoryLayout                IN_MEM_LAYOUT,
    DataMemoryLayout                OUT_MEM_LAYOUT,
    int                             DIM,
    typename                        SampleT,
    typename                        TileProcessingOpT>
__launch_bounds__ (BLOCK_THREADS)
static __global__ void addNodesKernel(
    TiledTreeNode<DIM, SampleT> *       parentNode,
    int                                 maxTileCapacity,
    SampleT                             sphereRadius,
    SampleT                             extRadius,
    TileProcessingOpT                   tileProcessingOp,
    int *                               d_leafCount)
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

    TiledTreeNode<DIM, SampleT> * node = (blockIdx.x == 0) ? parentNode->left : parentNode->right;

    AgentBuildTiledTreeT(
        tempStorage,
        node, 
        extRadius, 
        tileProcessingOp).addNode(
            parentNode,
            maxTileCapacity, 
            sphereRadius,
            d_leafCount);

}

} // end namespace detail

/**
 * @brief      Block of threads abstraction for creation of TiledTreeNodes
 *
 * @tparam     DIM                { description }
 * @tparam     NodeT              { description }
 * @tparam     BBoxT              { description }
 * @tparam     TileProcessingOpT  { description }
 * @tparam     T                  { description }
 */

template <
    int         BLOCK_THREADS,
    int         POINTS_PER_THREAD,
    int         DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT,
    typename    TileProcessingOpT,
    typename    T>
class AgentBuildTiledTree
{
    //-----------------------------------------------------
    // Types and constants
    //-----------------------------------------------------

    typedef TiledTreeNode<DIM, T> NodeT;
    typedef BoundingBox<DIM, T> BBoxT;

    struct SelectTilePointsT
    {
        BBoxT const & bounds;

        __device__ __forceinline__ SelectTilePointsT(
            BBoxT const & b)
        :
            bounds(b)
        {}

        __device__ __forceinline__ bool operator()(T const * point) const
        {
            return bounds.isInside(point);
        }
    };


    struct SelectTileNeighboursT
    {
        BBoxT const & bounds;
        /// How much extend the tile for searching neighbours.
        T extRadius;

        __device__ __forceinline__ SelectTileNeighboursT(
            BBoxT const &    b,
            T                radius)
        :
            bounds(b),
            extRadius(radius)
        {}

        __device__ __forceinline__ bool operator()(T const * point) const
        {
            return bounds.isNearby(point, extRadius);
        }
    };

public:
    // decode operator for mapping neighbouring points onto bins
    // left child's tile bin has index 0, 
    // right child's tile bin has index 1
    struct NeighboursHistDecodeOp
    {
        NodeT const & dividedNode;
        T extensionRadius;

        __device__ __forceinline__ NeighboursHistDecodeOp(
            NodeT const & n,
            T r)
        :
            dividedNode(n),
            extensionRadius(r)
        {}

        __device__ __forceinline__ void operator()(
            T point[DIM],
            int * d_histogram) const
        {
            if (dividedNode.left->bounds.isNearby(point, extensionRadius))
            {
                atomicAdd(d_histogram, 1);
            }
            if (dividedNode.right->bounds.isNearby(point, extensionRadius))
            {
                atomicAdd(d_histogram + 1, 1);
            }
        }
    };

private:
    typedef BlockSelectIfPolicy<
        BLOCK_THREADS,
        POINTS_PER_THREAD,
        cub::LOAD_LDG,               /// cache as read-only in non-uniform cache
        // cub::LOAD_CG,                   /// cache globally only in L2
        IO_BACKEND_CUB> BlockSelectIfPolicyT;

    typedef BlockSelectIf<
        BlockSelectIfPolicyT,
        DIM,
        IN_MEM_LAYOUT,
        SelectTilePointsT,
        T,
        int,
        false> BlockSelectIfTilePointsT;

    typedef BlockSelectIf<
        BlockSelectIfPolicyT,
        DIM,
        IN_MEM_LAYOUT,
        SelectTileNeighboursT,
        T,
        int,
        false> BlockSelectIfTileNeighboursT;

    // Shared memory type for this threadblock
    union _TempStorage
    {
        // Smem needed for points inside tile selection
        typename BlockSelectIfTilePointsT::TempStorage selectIfPointsSmem;
        // Smem needed for neighbouring tile points selection
        typename BlockSelectIfTileNeighboursT::TempStorage selectIfNeighboursSmem;
    };

public:
    struct TempStorage : cub::Uninitialized<_TempStorage> {};

private:
    //-----------------------------------------------------
    // per-thread  fields
    //-----------------------------------------------------
    
    /// reference to thread block smem
    _TempStorage & tempStorage;
    /// this thread block node
    NodeT * node;
    /// Functor for tile data processing
    TileProcessingOpT tileProcessOp;

    /// operator for selecting tile points 
    SelectTilePointsT selectTilePointsOp;
    /// operator for selecting tile neighbouring points
    SelectTileNeighboursT selectTileNeighboursOp;

    //-----------------------------------------------------
    // points partitioning
    //-----------------------------------------------------

    /**
     * @brief      Selects this node tile's points.
     *
     * @param      parentSamples        Pointer to parent points data
     * @param[in]  parentPointsCnt      The parent points number
     * @param[in]  parentSamplesStride  Distance between single point consecutive coordinates
     */
    __device__ __forceinline__ void getTilePoints(
        T const *   parentSamples,
        int         parentPointsCnt,
        int         parentSamplesStride)
    {
        int num = BlockSelectIfTilePointsT(
            tempStorage.selectIfPointsSmem,
            parentSamples, 
            node->samples, 
            selectTilePointsOp).scanRange(
                0, parentPointsCnt, parentSamplesStride, node->samplesStride);
        
        #ifdef RD_DEBUG
        if (threadIdx.x == 0 && num != node->pointsCnt)
        {
            _CubLog("getTilePoints(): node id: %d, parentPointsCnt: %d, expected pointsCnt: %d, "
                "selected pointsCnt: %d treeLevel: %d\n",
                node->id, parentPointsCnt, node->pointsCnt, num, node->treeLevel);
        }
        #endif
        if (threadIdx.x == 0 && num != node->pointsCnt)
        {
            // XXX: hack to handle slight divergence between histogram and block_select_if results!
            // there will be results discrepancy becaouse histogram handles last tile in row/column 
            // case correctly (e.g. point lying exactly on max bound) and bounding box mark point as
            // lying inside if it pass condition (min >= x > max)
            // assert(num >= int(0.995f * float(node->pointsCnt)));
            node->pointsCnt = num;
        }
    }

    /**
     * @brief      Selects the tile neighbours.
     *
     * @param      parentSamples           The parent samples
     * @param      parentNeighbours        The parent neighbours
     * @param[in]  parentPointCnt          The parent point count
     * @param[in]  parentNeighboursCnt     The parent neighbours count
     * @param[in]  parentSamplesStride     The parent samples stride
     * @param[in]  parentNeighboursStride  The parent neighbours stride
     */
    __device__ __forceinline__ void getTileNeighbours(
        T const *   parentSamples,
        T const *   parentNeighbours,
        int         parentPointCnt,
        int         parentNeighboursCnt,
        int         parentSamplesStride,
        int         parentNeighboursStride)
    {
        int num = 0;
        if (parentNeighbours != nullptr)
        {
            // #ifdef RD_DEBUG
            // if (threadIdx.x == 0)
            // {
            //     _CubLog(">>>>\t Invoking BlockSelectIf [neighbour] on parent neighbours: %p, "
            //         "neighboursCnt: %d, out: %p\n",
            //         parentNeighbours, parentNeighboursCnt, node->neighbours);
            // }
            // #endif

            num = BlockSelectIfTileNeighboursT(
                tempStorage.selectIfNeighboursSmem, 
                parentNeighbours, 
                node->neighbours, 
                selectTileNeighboursOp).scanRange(
                    0, parentNeighboursCnt, parentNeighboursStride, node->neighboursStride);
        }
        __syncthreads();

        // #ifdef RD_DEBUG
        // if (threadIdx.x == 0)
        // {
        //     _CubLog(">>>>\t Invoking BlockSelectIf [neighbour] on parent points: %p, "
        //         "pointsCnt: %d, out: %p, alreadySelectedNum: %d\n",
        //         parentSamples, parentPointCnt, node->neighbours, num);
        // }
        // #endif

        num += BlockSelectIfTileNeighboursT(
            tempStorage.selectIfNeighboursSmem, 
            parentSamples, 
            node->neighbours, 
            selectTileNeighboursOp,
            num).scanRange(
                0, parentPointCnt, parentSamplesStride, node->neighboursStride);

        #ifdef RD_DEBUG
        if (threadIdx.x == 0 && num != node->neighboursCnt)
        {
            _CubLog("getTileNeighbours(): node id: %d, parentNeighboursCnt: %d, "
                "expected neighboursCnt: %d, selected neighboursCnt: %d treeLevel: %d\n",
                node->id, parentNeighboursCnt, node->neighboursCnt, num, node->treeLevel);
        }
        #endif
        if (threadIdx.x == 0 && num != node->neighboursCnt)
        {
            // XXX: hack to handle slight divergence between histogram and block_select_if results!"
            // there will be results discrepancy becaouse histogram handles last tile in row/column 
            // case correctly (e.g. point lying exactly on max bound) and bounding box mark point as
            // lying inside if it pass condition (min >= x > max)
            assert(num >= int(0.995f * float(node->neighboursCnt)));
            node->neighboursCnt = num;
        }
    }

   /**
     * @brief      Calculates spatial histogram for @p d_samples set
     *
     * @param[out] d_tilesPointsHist      Output points histogram.
     * @param[out] d_tilesNeighboursHist  Output neighbours histogram.
     */
    __device__ __forceinline__ void calcHistograms(
        int *           d_tilesPointsHist,
        int *           d_tilesNeighboursHist) const
    {
        cudaStream_t pHistStream, nHistStream;
        rdDevCheckCall(cudaStreamCreateWithFlags(&pHistStream, cudaStreamNonBlocking));
        rdDevCheckCall(cudaStreamCreateWithFlags(&nHistStream, cudaStreamNonBlocking));

        // number of bins in subsequent dimensions
        int binsCnt[DIM];
        #pragma unroll
        for (int d = 0; d < DIM; ++d)
        {
            binsCnt[d] = 1;
        }
        binsCnt[node->treeLevel % DIM] = 2;

        // query for and allocate temporary storage
        void * d_pHistTempStorage = nullptr, *d_nHistTempStorage = nullptr;
        size_t pHistTempStorageBytes = 0, nHistTempStorageBytes = 0;
        NeighboursHistDecodeOp nHistDecodeOp(*node, selectTileNeighboursOp.extRadius);

        cudaError err = cudaSuccess;
        #ifdef RD_DEBUG
            // last arg is: debugSynchronous = true
            // third from the end is whether to use gmem priv hists
            err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_pHistTempStorage,
                pHistTempStorageBytes, node->samples, d_tilesPointsHist, node->pointsCnt, binsCnt, 
                node->bounds, node->samplesStride, true, pHistStream, true);
            rdDevCheckCall(err);

            err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_nHistTempStorage, 
                nHistTempStorageBytes, node->samples, d_tilesNeighboursHist, node->pointsCnt, 2, 
                nHistDecodeOp, node->samplesStride, false, true, nHistStream, true);
            rdDevCheckCall(err);

        #else
            err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_pHistTempStorage,
                pHistTempStorageBytes, node->samples, d_tilesPointsHist, node->pointsCnt, binsCnt, 
                node->bounds, node->samplesStride, true, pHistStream);
            rdDevCheckCall(err);

            err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_nHistTempStorage, 
                nHistTempStorageBytes, node->samples, d_tilesNeighboursHist, node->pointsCnt, 2, 
                nHistDecodeOp, node->samplesStride, false, true, nHistStream);
            rdDevCheckCall(err);

        #endif

        d_pHistTempStorage = new char[pHistTempStorageBytes];
        d_nHistTempStorage = new char[nHistTempStorageBytes];

        assert(d_pHistTempStorage != nullptr);
        assert(d_nHistTempStorage != nullptr);

        // run histograms
        #ifdef RD_DEBUG
                _CubLog("Invoking spatialHistogram for tile points!\n"
                    "pHistTempStorage: %p, tempStorageBytes: %lld, samples: %p, pointsCnt: %d"
                    " d_tilesPointsHist: %p\n",
                    d_pHistTempStorage, pHistTempStorageBytes, node->samples, node->pointsCnt,
                    d_tilesPointsHist);

            err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_pHistTempStorage, 
                pHistTempStorageBytes, node->samples, d_tilesPointsHist, node->pointsCnt, binsCnt, 
                node->bounds, node->samplesStride, true, pHistStream, true);
            rdDevCheckCall(err);
            rdDevCheckCall(cudaDeviceSynchronize());
        #else

            err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_pHistTempStorage, 
                pHistTempStorageBytes, node->samples, d_tilesPointsHist, node->pointsCnt, binsCnt, 
                node->bounds, node->samplesStride, true, pHistStream);
            rdDevCheckCall(err);
        #endif

        // We must call histogram twice. One for (in) tile samples, and second for neighbouring 
        // tile samples. Because when we divide tile, part of subtile neighbours will lie inside 
        // parent tile, and part will lie outside parent tile (that is within its neighbours) 
        #ifdef RD_DEBUG
                _CubLog("Invoking spatialHistogram for tile neighbours on node samples!\n"
                    "nHistTempStorage: %p, tempStorageBytes: %lld, neighbours: %p, pointsCnt: %d"
                    " d_tilesNeighboursHist: %p\n",
                    d_nHistTempStorage, nHistTempStorageBytes, node->samples, node->pointsCnt,
                    d_tilesNeighboursHist);
            
            err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_nHistTempStorage, 
                nHistTempStorageBytes, node->samples, d_tilesNeighboursHist, node->pointsCnt, 2, 
                nHistDecodeOp, node->samplesStride, false, true, nHistStream, true);
            rdDevCheckCall(err);
            rdDevCheckCall(cudaDeviceSynchronize());

            _CubLog("Invoking spatialHistogram for tile neighbours on node neighbours!\n"
                    "neighbours: %p, pointsCnt: %d, actualHist[0]: %d, actualHist[1]: %d\n",
                    node->neighbours, node->neighboursCnt, d_tilesNeighboursHist[0],
                    d_tilesNeighboursHist[1]);

            err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_nHistTempStorage, 
                nHistTempStorageBytes, node->neighbours, d_tilesNeighboursHist, node->neighboursCnt,
                2, nHistDecodeOp, node->neighboursStride, true, true, nHistStream, true);
            rdDevCheckCall(err);
            rdDevCheckCall(cudaDeviceSynchronize());

        #else

            err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_nHistTempStorage, 
                nHistTempStorageBytes, node->samples, d_tilesNeighboursHist, node->pointsCnt, 2, 
                nHistDecodeOp, node->samplesStride, false, true, nHistStream);
            rdDevCheckCall(err);

            err = DeviceHistogram::spatialHistogram<DIM, IN_MEM_LAYOUT>(d_nHistTempStorage, 
                nHistTempStorageBytes, node->neighbours, d_tilesNeighboursHist, node->neighboursCnt,
                2, nHistDecodeOp, node->neighboursStride, true, true, nHistStream);
            rdDevCheckCall(err);
        #endif

        rdDevCheckCall(cub::SyncStream(pHistStream));
        rdDevCheckCall(cub::SyncStream(nHistStream));

        rdDevCheckCall(cudaStreamDestroy(pHistStream));
        rdDevCheckCall(cudaStreamDestroy(nHistStream));

        delete[] d_pHistTempStorage;
        delete[] d_nHistTempStorage;
    }

    //-----------------------------------------------------
    // tile subdivision
    //-----------------------------------------------------

    /**
     * @brief      Halfsplit this tile and recursively process its data.
     *
     * @param[in]  maxTileCapacity  Maximum number of points tile can contain
     * @param[in]  sphereRadius     Parameter used to chosen points upper bound count estimation.
     * @param      d_leafCount      Pointer to global leaf counter.
     */
    __device__ __forceinline__ void subdivide(
        int         maxTileCapacity,
        T           sphereRadius,
        int *       d_leafCount)
    {
        
        if (threadIdx.x == 0)
        {
            #ifdef RD_DEBUG
            {
                _CubLog("\n>>>>> agent::subdivide()\n\n",1);
            }
            #endif

            // allocate children nodes
            node->left = new NodeT(node);
            node->right = new NodeT(node);
            assert(node->left != nullptr);
            assert(node->right != nullptr);

            // creating tile which is half of parentNode's tile size
            cub::ArrayWrapper<int, DIM> tilesPerDim;
            for (int d = 0; d < DIM; ++d)
            {
                tilesPerDim.array[d] = 1;
            }
            tilesPerDim.array[node->treeLevel % DIM] = 2;

            #ifdef RD_DEBUG
            {
                _CubLog("subdivide:: initTileBounds() on children\n",1);
            }
            #endif

            node->left->initTileBounds(0, tilesPerDim, 2, node->bounds);
            node->right->initTileBounds(1, tilesPerDim, 2, node->bounds);

            #ifdef RD_DEBUG
            {
                node->left->bounds.print();
                node->right->bounds.print();
            }
            #endif

            // calculate spatial histogram
            int * pointsHist = new int[2];
            int * neighboursHist = new int[2];
            assert(pointsHist != nullptr);
            assert(neighboursHist != nullptr);

            calcHistograms(pointsHist, neighboursHist);

            #ifdef RD_DEBUG
            {
                _CubLog("Tile[left] points: %d, neighbours: %d\n",
                    pointsHist[0], neighboursHist[0]);
                _CubLog("Tile[right] points: %d, neighbours: %d\n",
                    pointsHist[1], neighboursHist[1]);
            }
            #endif

            node->left->pointsCnt = pointsHist[0];
            node->right->pointsCnt = pointsHist[1];
            node->left->neighboursCnt = neighboursHist[0];
            node->right->neighboursCnt = neighboursHist[1];
            
            delete[] pointsHist;
            delete[] neighboursHist;

            // launch adding new tiles in new kernel
            cudaStream_t subdivideStream;
            rdDevCheckCall(cudaStreamCreateWithFlags(&subdivideStream, cudaStreamNonBlocking));

            #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                _CubLog("---- Invoking addNodesKernel()\n",1);
            }
            #endif

            detail::addNodesKernel<
                BLOCK_THREADS, 
                POINTS_PER_THREAD,
                IN_MEM_LAYOUT,
                OUT_MEM_LAYOUT,
                DIM,
                T,
                TileProcessingOpT>
                <<<2, BLOCK_THREADS, 0, subdivideStream>>>(
                    node,
                    maxTileCapacity,
                    sphereRadius,
                    selectTileNeighboursOp.extRadius,
                    tileProcessOp,
                    d_leafCount);

            rdDevCheckCall(cudaPeekAtLastError());
            rdDevCheckCall(cudaStreamDestroy(subdivideStream));
        } // threadIdx.x == 0
    }

public:
    //-----------------------------------------------------
    // Interface
    //-----------------------------------------------------
    
    __device__ __forceinline__ AgentBuildTiledTree(
        TempStorage &       tempStorage,
        NodeT *             n,
        T                   extRadius,
        TileProcessingOpT   processOp)
    :
        tempStorage(tempStorage.Alias()),
        node(n),
        tileProcessOp(processOp),
        selectTilePointsOp(node->bounds),
        selectTileNeighboursOp(node->bounds, extRadius)
    {
    }


    /**
     * @brief      Add child node to tree's root.
     *
     * @param      parentSamples        The parent samples
     * @param[in]  parentPointsCnt      The parent points count
     * @param[in]  tilePointCnt         This node tile point count
     * @param[in]  tileNeighboursCnt    This node tile neighbours count
     * @param[in]  maxTileCapacity      Mamximum number of points inside tile, after which we split
     *                                  it on two halves.
     * @param[in]  sphereRadius         Ridge detection choose phase parameter, needed for chosen
     *                                  points count estimation.
     * @param      d_leafCount          Pointer to global leaf counter.
     * @param[in]  parentSamplesStride  Distance between single point's subsequent coordinates
     *                                  (number of samples)
     */
    __device__ __forceinline__ void addRootNode(
        T const *   parentSamples,
        int         parentPointsCnt,
        int         tilePointCnt,
        int         tileNeighboursCnt,
        int         maxTileCapacity,         
        T           sphereRadius,  
        int *       d_leafCount,          
        int         parentSamplesStride)     
    {
        #ifdef RD_DEBUG
        if (threadIdx.x == 0)
        {
            _CubLog("AgentBuildTiledTree::addRootNode(): parentPointsCnt:%d, tilePointCnt :%d,"
                " tileNeighboursCnt: %d, maxTileCapacity: %d, sphereRadius: %f, "
                "parentSamplesStride: %d\n",
                 parentPointsCnt, tilePointCnt, tileNeighboursCnt, maxTileCapacity, sphereRadius,
                 parentSamplesStride);
        }
        #endif
        if (threadIdx.x == 0 && tilePointCnt > 1)
        {
            rdDevCheckCall(rdDevAllocMem(&node->samples, &node->samplesStride, DIM, 
                tilePointCnt, cub::Int2Type<IN_MEM_LAYOUT>()));
            rdDevCheckCall(rdDevAllocMem(&node->neighbours, &node->neighboursStride, DIM, 
                tileNeighboursCnt, cub::Int2Type<IN_MEM_LAYOUT>()));

            node->pointsCnt = tilePointCnt;
            node->neighboursCnt = tileNeighboursCnt;
            node->treeLevel = 1;
        }
        __syncthreads();

        // Stop if tile has few points inside.
        if (tilePointCnt < 2)
        {
            return;
        }

        getTilePoints(parentSamples, parentPointsCnt, parentSamplesStride);
        getTileNeighbours(parentSamples, nullptr, parentPointsCnt, 0, parentSamplesStride, 0);
        __syncthreads();

        if (threadIdx.x == 0 && tilePointCnt <= maxTileCapacity)
        {
            node->chosenPointsCapacity = node->bounds.countSpheresInside(sphereRadius);
            rdDevCheckCall(rdDevAllocMem(&node->chosenSamples, &node->chosenSamplesStride,
                DIM, min(node->chosenPointsCapacity, tilePointCnt), 
                cub::Int2Type<OUT_MEM_LAYOUT>()));
        }

        if (tilePointCnt > maxTileCapacity)
        {
            subdivide(maxTileCapacity, sphereRadius, d_leafCount);
        }
        else
        {
            if (threadIdx.x == 0)
            {
                atomicAdd(d_leafCount, 1);
            }
            tileProcessOp(node);
        }
    }

    /**
     * @brief      Adds a node to the @p parentNode.
     *
     * @param      parentNode       Pointer to the parent node
     * @param[in]  maxTileCapacity  Mamximum number of points inside tile, after which we split it
     *                              on two halves.
     * @param[in]  sphereRadius     Ridge detection choose phase parameter, needed for chosen points
     *                              count estimation.
     * @param      d_leafCount      Pointer to global leaf counter
     */
    __device__ __forceinline__ void addNode(
        NodeT *     parentNode, 
        int         maxTileCapacity,        
        T           sphereRadius,
        int *       d_leafCount)           
    {
        #ifdef RD_DEBUG
        if (threadIdx.x == 0)
        {
            _CubLog("agent::addNode() node id: %d, treeLevel: %d, points: %d, neighbours: %d\n",
                node->id, node->treeLevel, node->pointsCnt, node->neighboursCnt);
        }
        #endif

        if (threadIdx.x == 0 && node->pointsCnt > 1)
        {
            rdDevCheckCall(rdDevAllocMem(&node->samples, &node->samplesStride, DIM, 
                node->pointsCnt, cub::Int2Type<IN_MEM_LAYOUT>()));
            rdDevCheckCall(rdDevAllocMem(&node->neighbours, &node->neighboursStride, DIM, 
                node->neighboursCnt, cub::Int2Type<IN_MEM_LAYOUT>()));
        }
        __syncthreads();

        // Stop if tile has few points inside.
        if (node->pointsCnt < 2)
        {
            if (threadIdx.x == 0)
            {
                atomicCAS(&parentNode->needEvolve, 0, 1);
                parentNode->releaseChild(node->id);
            }
            return;
        }

        getTilePoints(parentNode->samples, parentNode->pointsCnt, parentNode->samplesStride);

        #ifdef RD_DEBUG
        if (threadIdx.x == 0)
        {
            _CubLog(">>>> addNode:: selected tile points! node id: %d\n", node->id);
        }
        #endif

        getTileNeighbours(parentNode->samples, parentNode->neighbours, parentNode->pointsCnt, 
            parentNode->neighboursCnt, parentNode->samplesStride, parentNode->neighboursStride);
        __syncthreads();

        #ifdef RD_DEBUG
        if (threadIdx.x == 0)
        {
            _CubLog(">>>> addNode:: selected tile neighbours! node id: %d\n", node->id);
        }
        #endif

        // release parent's memory if I'am the last block who process parent data.
        if (threadIdx.x == 0)
        {
            if (atomicCAS(&parentNode->needEvolve, 0, 1))
            {
                #ifdef RD_DEBUG
                    _CubLog("clearing parent data! node id: %d, parentNode id: %d, treeLevel: %d\n",
                        node->id, parentNode->id, node->treeLevel);
                #endif
                parentNode->clearData();
            }
        }

        if (threadIdx.x == 0 && node->pointsCnt <= maxTileCapacity)
        {
            node->chosenPointsCapacity = node->bounds.countSpheresInside(sphereRadius);
            rdDevCheckCall(rdDevAllocMem(&node->chosenSamples, &node->chosenSamplesStride,
                DIM, min(node->chosenPointsCapacity, node->pointsCnt), 
                cub::Int2Type<OUT_MEM_LAYOUT>()));
        }
        __syncthreads();

        if (node->pointsCnt > maxTileCapacity)
        {
            subdivide(maxTileCapacity, sphereRadius, d_leafCount);
        }
        else
        {
            if (threadIdx.x == 0)
            {
                atomicAdd(d_leafCount, 1);
            }
            tileProcessOp(node);
        }
    }
};

} // end namespace tiled
} // end namespace gpu
} // end namespace rd