/**
 * @file agent_tiled_ridge_detection.cuh
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

#include "rd/gpu/agent/agent_memcpy.cuh"
#include "rd/gpu/device/tiled/tiled_tree.cuh"
#include "rd/gpu/device/device_ridge_detection.cuh"
#include "rd/gpu/device/device_evolve.cuh"
#include "rd/gpu/device/device_choose.cuh"
#include "rd/gpu/device/device_decimate.cuh"
#include "rd/gpu/device/device_tiled_decimate.cuh"

#include "rd/gpu/util/dev_utilities.cuh"
#include "rd/utils/memory.h"
#include "rd/utils/macro.h"

#include "cub/thread/thread_load.cuh"
#include "cub/thread/thread_store.cuh"
#include "cub/util_type.cuh"

namespace rd
{
namespace gpu
{
namespace tiled
{

/**
 * @brief      Describes available tile types to use with ridge detection algorithm
 */
enum TileType
{
    /**
     * @par Overview
     *     Represents a tile which consider given point as a neighbouring point, if it lies within
     *     its enlarged bounds.
     *
     * @par Performance Consideratons
     *     Extended tile has greater memory needs, because it keeps all of its neighbouring points
     *     inside additional table.
     */
    RD_EXTENDED_TILE,
};

/**
 * @brief      Describes general ridge detection behaviour relative to all tiles.
 */
enum TiledRidgeDetectionPolicy
{
    /**
     * @par Overview
     *     With this policy ridge detection is performed individually by each tile and with only its
     *     data.
     *
     * @par Performance Considerations
     *     This is the fastest policy, since all computations are performed end-to-end by each tile
     *     individually. However as a consequence we may obtain globally worse results than with
     *     RD_MIXED policy.
     */
    RD_LOCAL,

    /**
     * @par Overview
     *     With this policy ridge detection algorithm is divided into local and global part. The
     *     local part include choose and evolve passes and the remaining decimate belongs to global
     *     part. Local part calculations are performed individually within each tile as in RD_LOCAL.
     *     However global part is performed on all tiles' data.
     *
     * @par Performance Considerations
     *     Such work decomposition requires few synchronization points. First after evolve, before
     *     global decimate, and second after global decimate is done. This entails slower execution
     *     time than RD_LOCAL. In exchange we may obtain better, more coherent results.
     */
    RD_MIXED,
};

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

namespace detail
{

struct RD_MEM_ALIGN(16) TiledRidgeDetectionTimers
{
    // long long int buildTreeTime;
    long long int rdTilesTime;
    long long int refinementTime;
    long long int wholeTime;
};

template <
    int                 DIM,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    DataMemoryLayout    OUT_MEM_LAYOUT,
    typename            T>
__launch_bounds__ (BLOCK_THREADS)
__global__ void deviceCollectChosenPointsKernel(
    T *                             d_chosenPoints,
    int *                           d_chosenPointsNum,
    TiledTreeNode<DIM,T> const * const * d_treeLeafs,
    int                             chosenPointsStride)
{
    __shared__ int auxOffset;
    int chosenPointsOffset = 0;
    TiledTreeNode<DIM,T> const *node = d_treeLeafs[blockIdx.x];

    if (threadIdx.x == 0)
    {
        // update global counter
        chosenPointsOffset = atomicAdd(d_chosenPointsNum, node->chosenPointsCnt);
        auxOffset = chosenPointsOffset;
    }

    __syncthreads();
    chosenPointsOffset = auxOffset;

    // copy points
    
    // Type definitions to perform efficient copy
    typedef BlockTileLoadPolicy<
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            cub::LOAD_LDG>
        BlockTileLoadPolicyT;

    typedef BlockTileStorePolicy<
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            cub::STORE_DEFAULT>
            // cub::STORE_CS/CG>
        BlockTileStorePolicyT;

    typedef AgentMemcpy<
            BlockTileLoadPolicyT,
            BlockTileStorePolicyT,
            DIM,
            OUT_MEM_LAYOUT,
            OUT_MEM_LAYOUT,
            IO_BACKEND_CUB,
            int,
            T>
        AgentMemcpyT;

    AgentMemcpyT(node->chosenSamples, d_chosenPoints, chosenPointsOffset).copyRange(
        0, node->chosenPointsCnt, node->chosenSamplesStride, chosenPointsStride, true);
}

} // end namespace detail

//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------

template< 
    int     _BLOCK_THREADS,
    int     _ITEMS_PER_THREAD>
struct AgentTiledRidgeDetectionPolicy
{
    enum 
    {  
        BLOCK_THREADS       = _BLOCK_THREADS,
        ITEMS_PER_THREAD    = _ITEMS_PER_THREAD,
    };
};

template <
    typename                        AgentTiledRidgeDetectionPolicyT,
    int                             DIM,
    DataMemoryLayout                IN_MEM_LAYOUT,
    DataMemoryLayout                OUT_MEM_LAYOUT,
    RidgeDetectionAlgorithm         RD_TILE_ALGORITHM,
    TiledRidgeDetectionPolicy       RD_TILE_POLICY,
    TileType                        RD_TILE_TYPE,
    typename                        T>
class AgentTiledRidgeDetection
{
    //-----------------------------------------------------
    // Types and constants
    //-----------------------------------------------------
    
    enum 
    {  
        BLOCK_THREADS       = AgentTiledRidgeDetectionPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD    = AgentTiledRidgeDetectionPolicyT::ITEMS_PER_THREAD,
    };
    
#ifdef RD_DRAW_TREE_TILES
public:
#endif
    typedef TiledTreePolicy<
            BLOCK_THREADS,
            ITEMS_PER_THREAD>
        TiledTreePolicyT;

    typedef TiledTree<
            TiledTreePolicyT,
            DIM,
            IN_MEM_LAYOUT,
            OUT_MEM_LAYOUT,
            T>
        TiledTreeT;

    typedef typename TiledTreeT::NodeT NodeT;

    //-----------------------------------------------------
    // Internal helper structure with tree's leaf tile processing operation.
    //-----------------------------------------------------
public:
    template<
        RidgeDetectionAlgorithm         _RD_TILE_ALGORITHM,
        TiledRidgeDetectionPolicy       _RD_TILE_POLICY,
        TileType                        _RD_TILE_TYPE,
        int                             DUMMY>
    struct TileProcessingOp
    {
    };

    template<int DUMMY>
    struct TileProcessingOp<RD_BRUTE_FORCE, RD_LOCAL, RD_EXTENDED_TILE, DUMMY>
    {
        T r1, r2;
        T *     d_chosenPoints; 
        int *   d_chosenPointsNum;
        int     chosenPointsStride;

        __device__ __forceinline__ TileProcessingOp(
            T r1,
            T r2,
            T *     d_chosenPoints, 
            int *   d_chosenPointsNum,
            int     chosenPointsStride)
        :
            r1(r1),
            r2(r2),
            d_chosenPoints(d_chosenPoints),
            d_chosenPointsNum(d_chosenPointsNum),
            chosenPointsStride(chosenPointsStride)
        {}

        __device__ __forceinline__ void operator()(NodeT * node)
        {
            #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                _CubLog("TileProcessingOp<BRUTE_FORCE, LOCAL, EXT_TILE>, node id: %d, "
                    "r1: %f, r2: %f, chosenPoints: %p, d_chosenPointsNum: %p, chosenPointsStride: %d\n",
                    node->id, r1, r2, d_chosenPoints, d_chosenPointsNum, chosenPointsStride);
            }
            #endif

            if (threadIdx.x == 0)
            {
                // launch local tile ridge detection
                cudaError_t err = cudaSuccess;
                err = DeviceRidgeDetection::approximate<DIM, IN_MEM_LAYOUT,
                        OUT_MEM_LAYOUT, RD_BRUTE_FORCE>(
                    node->samples, node->neighbours, node->chosenSamples, node->pointsCnt, 
                    node->neighboursCnt, &node->chosenPointsCnt, r1, r2, node->samplesStride, 
                    node->neighboursStride, node->chosenSamplesStride, false);   // debugSynchronous
                rdDevCheckCall(err);
                // make sure that all scheduled kernels (by any thread in current block) are done
                rdDevCheckCall(cudaDeviceSynchronize());
            }
            __syncthreads();

            #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                node->print();
                _CubLog("   >>>>>   node id: %d, Finished processing, collect tiles...\n", node->id);
            }
            #endif

            // collect chosen points from all tiles
            __shared__ int auxOffset;
            int chosenPointsOffset = 0;
            
            if (threadIdx.x == 0)
            {
                // update global counter
                chosenPointsOffset = atomicAdd(d_chosenPointsNum, node->chosenPointsCnt);
                auxOffset = chosenPointsOffset;
            }

            __syncthreads();
            chosenPointsOffset = auxOffset;

            // copy points
            
            // Type definitions to perform efficient copy
            typedef BlockTileLoadPolicy<
                    BLOCK_THREADS,
                    ITEMS_PER_THREAD,
                    cub::LOAD_LDG>
                BlockTileLoadPolicyT;

            typedef BlockTileStorePolicy<
                    BLOCK_THREADS,
                    ITEMS_PER_THREAD,
                    cub::STORE_DEFAULT>
                    // cub::STORE_CS/CG>
                BlockTileStorePolicyT;

            typedef AgentMemcpy<
                    BlockTileLoadPolicyT,
                    BlockTileStorePolicyT,
                    DIM,
                    OUT_MEM_LAYOUT,
                    OUT_MEM_LAYOUT,
                    IO_BACKEND_CUB,
                    int,
                    T>
                AgentMemcpyT;

            AgentMemcpyT(node->chosenSamples, d_chosenPoints, chosenPointsOffset).copyRange(
                0, node->chosenPointsCnt, node->chosenSamplesStride, chosenPointsStride, true);
        }
    };

    template<int DUMMY>
    struct TileProcessingOp<RD_BRUTE_FORCE, RD_MIXED, RD_EXTENDED_TILE, DUMMY>
    {
        T r1;

        __device__ __forceinline__ TileProcessingOp(
            T r1)
        :
            r1(r1)
        {}

        __device__ __forceinline__ void operator()(NodeT * node)
        {
            #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                _CubLog("TileProcessingOp<BRUTE_FORCE, MIXED, EXT_TILE>, node id: %d, "
                    "chosenPointsCapacity: %d, chosenSamples: [%p -- %p]\n",
                    node->id, node->chosenPointsCapacity, node->chosenSamples,
                    (OUT_MEM_LAYOUT == COL_MAJOR) ? 
                    node->chosenSamples + node->chosenSamplesStride * DIM : 
                    node->chosenSamples + node->chosenPointsCapacity * DIM);
            }
            #endif

            if (threadIdx.x == 0)
            {
                /*
                 * XXX: 17.08.2016 Dlaczego nie mogę (?) podać normalnie node->chosenPointsCnt zamiast
                 * alokować specjalnie counter do tego???
                 */
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
        }
    };

private:
    //-----------------------------------------------------
    // Internal helper structure for ridge detection alg private impl
    //-----------------------------------------------------

    template <
        RidgeDetectionAlgorithm         _RD_TILE_ALGORITHM,
        TiledRidgeDetectionPolicy       _RD_TILE_POLICY,
        TileType                        _RD_TILE_TYPE,
        int                             DUMMY>
    struct InternalRDImpl
    {
    };

    template <int DUMMY>
    struct InternalRDImpl<RD_BRUTE_FORCE, RD_LOCAL, RD_EXTENDED_TILE, DUMMY>
    {
        __device__ __forceinline__ void approximate(
            TiledTreeT &                tree,
            T const *                   d_inputPoints,
            T *                         d_chosenPoints,
            int                         inPointsNum,
            int *                       d_chosenPointsNum,
            T                           r1,
            T                           r2,
            int                         inPointsStride,
            int                         chosenPointsStride,
            BoundingBox<DIM, T> *       d_globalBBox,
            cub::ArrayWrapper<int,DIM>  dimTiles,
            detail::TiledRidgeDetectionTimers * ,     // -unused!
            bool                        )       // -unused!
        {
            #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                _CubLog("[InternalRDImpl<BRUTE_FORCE, LOCAL, EXT_TILE>] approximate\n", 1);
            }
            #endif

            cudaStream_t buildTreeStream;
            rdDevCheckCall(cudaStreamCreateWithFlags(&buildTreeStream, cudaStreamNonBlocking));

            typedef TileProcessingOp<
                    RD_BRUTE_FORCE, 
                    RD_LOCAL, 
                    RD_EXTENDED_TILE, 
                    DUMMY>
                TileProcessingOpT;

            // build tiled tree and perform local ridge detection within each tile individually
            rdDevCheckCall(tree.buildTree(d_inputPoints, inPointsNum, dimTiles, d_globalBBox, 
                TileProcessingOpT(r1, r2, d_chosenPoints, d_chosenPointsNum, chosenPointsStride), 
                buildTreeStream, inPointsStride));

            rdDevCheckCall(cudaStreamDestroy(buildTreeStream));
        }

    };

    template <int DUMMY>
    struct InternalRDImpl<RD_BRUTE_FORCE, RD_MIXED, RD_EXTENDED_TILE, DUMMY>
    {
        __device__ __forceinline__ void approximate(
            TiledTreeT &                tree,
            T const *                   d_inputPoints,
            T *                         d_chosenPoints,
            int                         inPointsNum,
            int *                       d_chosenPointsNum,
            T                           r1,
            T                           r2,
            int                         inPointsStride,
            int                         chosenPointsStride,
            BoundingBox<DIM, T> *       d_globalBBox,
            cub::ArrayWrapper<int,DIM>  dimTiles,
            detail::TiledRidgeDetectionTimers * d_rdTimers,
            bool                        debugSynchronous)
        {
            #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                _CubLog("[InternalRDImpl<BRUTE_FORCE, MIXED, EXT_TILE>] approximate\n", 1);
            }
            #endif

            cudaStream_t buildTreeStream;
            rdDevCheckCall(cudaStreamCreateWithFlags(&buildTreeStream, cudaStreamNonBlocking));

            typedef TileProcessingOp<
                    RD_BRUTE_FORCE, 
                    RD_MIXED, 
                    RD_EXTENDED_TILE, 
                    DUMMY>
                TileProcessingOpT;

            // build tree and choose points within tiles
            rdDevCheckCall(tree.buildTree(d_inputPoints, inPointsNum, dimTiles, d_globalBBox, 
                TileProcessingOpT(r1), buildTreeStream, inPointsStride));
            rdDevCheckCall(cudaStreamDestroy(buildTreeStream));
            // make sure building tree and choosing points is done
            rdDevCheckCall(cudaDeviceSynchronize());

            /*
             * Main part of ridge detection
             */

            #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                _CubLog(">>>>> main part ridge detection: collecting leafs... \n", 1);
            }
            #endif

            // allocate table for pointers to leaf nodes and leaf evolveStreams
            NodeT **d_treeLeafs = new NodeT*[*tree.d_leafCount];
            assert(d_treeLeafs != nullptr);
            cudaStream_t *leafEvolveStreams = new cudaStream_t[*tree.d_leafCount];
            assert(leafEvolveStreams != nullptr);

            int leafCounter = 0;

            // collect and initialize leafs
            tree.forEachNodePreorder(
                [d_chosenPointsNum, &d_treeLeafs, &leafCounter, &leafEvolveStreams](NodeT * node) {
                // am I a non-empty leaf?
                if (!node->haveChildren() && !node->empty())
                {
                    d_treeLeafs[leafCounter] = node;
                    atomicAdd(d_chosenPointsNum, node->chosenPointsCnt);
                    
                    // init evolve flag
                    node->needEvolve = 1;
                    rdDevCheckCall(cudaStreamCreateWithFlags(leafEvolveStreams + leafCounter++,
                        cudaStreamNonBlocking));

                    #ifdef RD_DEBUG
                    {
                        printf("node: %d, chosenPointsCnt: %d\n", node->id, node->chosenPointsCnt);
                    }
                    #endif

                    // allocate temporary storage for evolve phase
                    rdDevAllocMem(&node->cordSums, &node->cordSumsStride, DIM, 
                        node->chosenPointsCnt, cub::Int2Type<IN_MEM_LAYOUT>());
                    node->spherePointCnt = new int[node->chosenPointsCnt];

                    assert(node->cordSums != nullptr);
                    assert(node->spherePointCnt != nullptr);
                }
            });

            int oldChosenPtsCnt = 0;
            cudaError_t err = cudaSuccess;

            // repeat untill there's no change in point's count
            while (oldChosenPtsCnt != *d_chosenPointsNum)
            {
                oldChosenPtsCnt = *d_chosenPointsNum;
                #ifdef RD_DEBUG
                {
                    printf("\n\n\t\t >>>>>>\t d_chosenPointsNum: %d\n\n", oldChosenPtsCnt);
                }
                #endif

                for (int k = 0; k < leafCounter; ++k)
                {
                    NodeT * node = d_treeLeafs[k];
                     // We need to check whether we really have to do evolution for particular tile,
                     // since one tile may end stabilize its chosen points position earlier than 
                     // other. Thus during decimation we mark tile with needEvolve flag if there was 
                     // change in number of chosen points.
                    if (node->needEvolve)
                    {
                        #ifdef RD_DEBUG
                        if (threadIdx.x == 0)
                        {
                            _CubLog("\t>>>>> node %d local evolve ----> chosenPointsCnt: %d\n\n",
                                node->id, node->chosenPointsCnt);
                        }
                        #endif
                        
                        err = 
                        rd::gpu::bruteForce::DeviceEvolve::evolve<DIM, IN_MEM_LAYOUT, 
                                OUT_MEM_LAYOUT>(
                            node->samples, node->neighbours, node->chosenSamples, node->cordSums, 
                            node->spherePointCnt, node->pointsCnt, node->neighboursCnt,
                            node->chosenPointsCnt, r1, node->samplesStride, node->neighboursStride,
                            node->chosenSamplesStride, node->cordSumsStride, leafEvolveStreams[k], 
                            debugSynchronous);
                        rdDevCheckCall(err);
                    }
                    #ifdef RD_DEBUG
                    else
                    {
                        if (threadIdx.x == 0)
                        {
                            _CubLog("\t>>>>> node %d don't need local evolve ----> chosenPointsCnt: %d\n\n",
                                node->id, node->chosenPointsCnt);
                        }
                    }
                    #endif
                    node->needEvolve = 0;

                }
                rdDevCheckCall(cudaDeviceSynchronize());

                #ifdef RD_DEBUG
                if (threadIdx.x == 0)
                {
                    _CubLog("\n>>>>> ---------globalDecimate---------\n", 1);
                }
                #endif

                DeviceDecimate::globalDecimate<DIM, OUT_MEM_LAYOUT>(
                    d_treeLeafs, leafCounter, d_chosenPointsNum, r2, nullptr, debugSynchronous);
                rdDevCheckCall(cudaDeviceSynchronize());
                __syncthreads();
            }

            for (int k = 0; k < leafCounter; ++k)
            {
                rdDevCheckCall(cudaStreamDestroy(leafEvolveStreams[k]));
            }

            collectChosenPoints(d_treeLeafs, d_chosenPoints, d_chosenPointsNum, chosenPointsStride,
                leafCounter, debugSynchronous);
            rdDevCheckCall(cudaDeviceSynchronize());

            delete[] d_treeLeafs;
            delete[] leafEvolveStreams;
        }

        __device__ __forceinline__ void collectChosenPoints(
            NodeT ** d_treeLeafs,
            T *      d_chosenPoints,
            int *    d_chosenPointsNum,
            int      chosenPointsStride,
            int      leafCounter,
            bool     debugSynchronous)
        {
            if (debugSynchronous && threadIdx.x == 0)
            {
                _CubLog("[InternalRDImpl<RD_BRUTE_FORCE, RD_MIXED, RD_EXTENDED_TILE>] invoke "
                    "deviceCollectChosenPointsKernel<<<%d, %d>>>, leafCounter: %d, "
                    "*d_chosenPointsNum: %d\n",
                    leafCounter, BLOCK_THREADS, leafCounter, *d_chosenPointsNum);
            }
            // need to zeroize counter in order to get correct tile offsets.
            *d_chosenPointsNum = 0;

            detail::deviceCollectChosenPointsKernel<DIM, BLOCK_THREADS, 
                ITEMS_PER_THREAD, OUT_MEM_LAYOUT><<<leafCounter, BLOCK_THREADS>>>(
                    d_chosenPoints, d_chosenPointsNum, d_treeLeafs, chosenPointsStride);
            rdDevCheckCall(cudaGetLastError());
        }
    };


    //-----------------------------------------------------
    // Type definition for internal rd algorithm type
    //-----------------------------------------------------
    
    typedef InternalRDImpl<RD_TILE_ALGORITHM, RD_TILE_POLICY, RD_TILE_TYPE, 0> InternalRDImplT;

    //-----------------------------------------------------
    // per-thread  fields
    //-----------------------------------------------------
    
    InternalRDImplT rdImpl;
    TiledTreeT      tree;

    /**
     * @brief      Performs global ridge detection results refinements.
     */
    __device__ __forceinline__ void refineResults(
        T const *   d_inputPoints,
        T *         d_chosenPoints,
        int *       d_chosenPointsNum,
        int         inPointsNum,
        T           r1,
        T           r2,
        int         inPointsStride,
        int         chosenPointsStride,
        bool        debugSynchronous)
    {
        int chosenPtsNum = *d_chosenPointsNum;
        
        int cordSumsStride;
        int distMtxStride;
        
        T * d_cordSums;
        T * d_distMtx = nullptr;
        
        char * d_pointsMask = new char[chosenPtsNum];
        int * d_spherePointCount = new int[chosenPtsNum];
        assert(d_pointsMask != nullptr);
        assert(d_spherePointCount != nullptr);
        
        rdDevCheckCall(rdDevAllocMem(&d_cordSums, &cordSumsStride, DIM, chosenPtsNum, 
            cub::Int2Type<IN_MEM_LAYOUT>()));
        rdDevCheckCall(rdDevAllocMem(&d_distMtx, &distMtxStride, chosenPtsNum, chosenPtsNum, 
            cub::Int2Type<COL_MAJOR>()));

        #ifdef RD_DEBUG
        if (threadIdx.x == 0)
        {
            _CubLog("AgentTiledRidgeDetection::refineResults() \n",1);
        }
        #endif


        cudaError_t err = cudaSuccess;
        err = rd::gpu::bruteForce::DeviceDecimate::decimateDistMtx<DIM, OUT_MEM_LAYOUT>(
            d_chosenPoints, d_chosenPointsNum, chosenPointsStride, d_distMtx, distMtxStride,
            d_pointsMask, r2, nullptr, debugSynchronous);
        rdDevCheckCall(err);

        err = rd::gpu::bruteForce::DeviceEvolve::evolve<DIM, IN_MEM_LAYOUT, 
                OUT_MEM_LAYOUT>(
            d_inputPoints, d_chosenPoints, d_cordSums, d_spherePointCount, inPointsNum, 
            chosenPtsNum, r1, inPointsStride, chosenPointsStride, cordSumsStride, nullptr,
            debugSynchronous);
        rdDevCheckCall(err);

        rdDevCheckCall(cudaDeviceSynchronize());
        delete[] d_cordSums;
        delete[] d_spherePointCount;
        delete[] d_distMtx;
        delete[] d_pointsMask;
    }

public:
    //-----------------------------------------------------------------------------
    //  Interface
    //-----------------------------------------------------------------------------

    __device__ __forceinline__ AgentTiledRidgeDetection(
        int     maxTileCapacity,
        T       r1,
        bool    debugSynchronous)
    :
        tree(maxTileCapacity, r1, debugSynchronous)
    {}

    __device__ __forceinline__ void approximate(
        T const *                   d_inputPoints,
        T *                         d_chosenPoints,
        int                         inPointsNum,
        int *                       d_chosenPointsNum,
        T                           r1,
        T                           r2,
        int                         inPointsStride,
        int                         chosenPointsStride,
        BoundingBox<DIM, T> *       d_globalBBox,
        cub::ArrayWrapper<int,DIM>  dimTiles,
        bool                        endPhaseRefinement,
        detail::TiledRidgeDetectionTimers * d_rdTimers,
        bool                        debugSynchronous)
    {
        #ifdef RD_DEBUG
            _CubLog("AgentTiledRidgeDetection::approximate() inPointsNum: %d *d_chosenPointsNum: %d" 
                " inPointsStride: %d, chosenPointsStride: %d\n",
                inPointsNum, *d_chosenPointsNum, inPointsStride, chosenPointsStride);
        #endif

        #ifdef RD_INNER_KERNEL_TIMING
        long long int startRdTilesTime = 0, endRdTilesTime = 0;
        startRdTilesTime = clock64();
        #endif

        // initialize chosen pts counter
        *d_chosenPointsNum = 0;
        rdImpl.approximate(tree, d_inputPoints, d_chosenPoints, inPointsNum, d_chosenPointsNum,
            r1, r2, inPointsStride, chosenPointsStride, d_globalBBox, dimTiles, d_rdTimers, 
            debugSynchronous);
        rdDevCheckCall(cudaDeviceSynchronize());

        #ifdef RD_INNER_KERNEL_TIMING
        endRdTilesTime = clock64();
        if (d_rdTimers != nullptr)
        {
            d_rdTimers->rdTilesTime = endRdTilesTime - startRdTilesTime;
        }
        #endif

        if (endPhaseRefinement)
        {
            #ifdef RD_INNER_KERNEL_TIMING
            long long int startRefTime = 0, endRefTime = 0;
            startRefTime = clock64();
            #endif

            refineResults(d_inputPoints, d_chosenPoints, d_chosenPointsNum, inPointsNum, r1, r2, 
                inPointsStride, chosenPointsStride, debugSynchronous);

            #ifdef RD_INNER_KERNEL_TIMING
            endRefTime = clock64();
            if (d_rdTimers != nullptr)
            {
                d_rdTimers->refinementTime = endRefTime - startRefTime;
            }
            #endif
        }
    }

    #ifdef RD_DRAW_TREE_TILES

    __device__ __forceinline__ TiledTreeT getTree() const
    {
        return tree;
    }

    #endif

};

} // end namespace tiled
} // end namespace gpu
} // end namespace rd
