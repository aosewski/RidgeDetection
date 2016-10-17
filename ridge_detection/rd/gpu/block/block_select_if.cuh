/**
 * @file block_select_if.cuh
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and Information
 * Technology Warsaw University of Technology 2016
 */

#pragma once

#include <assert.h>

#include "rd/utils/memory.h"

#include "rd/gpu/block/block_tile_load_store4.cuh"
#include "rd/gpu/warp/warp_functions.cuh"

#include "cub/util_type.cuh"
#include "cub/util_ptx.cuh"

namespace rd {
namespace gpu {
    
/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for BlockSelectIf
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _POINTS_PER_THREAD,             ///< Items per thread (per tile of input)
    cub::CacheLoadModifier      _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    BlockTileIOBackend          _IO_BACKEND>                                                                  
struct BlockSelectIfPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        POINTS_PER_THREAD       = _POINTS_PER_THREAD,            ///< Items per thread (per tile of input)
    };

    static const cub::CacheLoadModifier     LOAD_MODIFIER           = _LOAD_MODIFIER;       ///< Cache load modifier for reading input elements
    static const BlockTileIOBackend         IO_BACKEND              = _IO_BACKEND;
};




/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/


/**
 * \brief BlockSelectIf implements a stateful abstraction of CUDA thread blocks for participating in device-wide selection
 *
 * Performs functor-based selection
 */
template <
    typename            BlockSelectIfPolicyT,           ///< Parameterized BlockSelectIfPolicy tuning policy type
    int                 DIM,                            ///< Input data dimension
    DataMemoryLayout    MEM_LAYOUT,                     ///< Data memory layout.
    typename            SelectOpT,                      ///< Selection operator type
    typename            SampleT,                        ///< Input data single coordinate data type.
    typename            OffsetT,                        ///< Signed integer type for global offsets
    bool                STORE_TWO_PHASE>                ///< Wheather or not to perform data compaction in smem when storing selected itmes to gmem
class BlockSelectIf
{

private:
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        BLOCK_THREADS           = BlockSelectIfPolicyT::BLOCK_THREADS,
        POINTS_PER_THREAD       = BlockSelectIfPolicyT::POINTS_PER_THREAD,
        TILE_POINTS             = BLOCK_THREADS * POINTS_PER_THREAD,

        SAMPLES_PER_THREAD      = POINTS_PER_THREAD * DIM,
        TILE_SAMPLES            = SAMPLES_PER_THREAD * BLOCK_THREADS,
    };

    // tile loading policy types
    typedef BlockTileLoadPolicy<
            BLOCK_THREADS,
            POINTS_PER_THREAD,
            BlockSelectIfPolicyT::LOAD_MODIFIER>
        BlockTileLoadPolicyT;

    // Tile loading types definitions
    typedef BlockTileLoad<
            BlockTileLoadPolicyT,
            DIM,
            MEM_LAYOUT,
            BlockSelectIfPolicyT::IO_BACKEND,
            SampleT,
            OffsetT>
        BlockTileLoadT;

    // input data point type
    typedef typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR>::PointT PointT;

    //---------------------------------------------------------------------
    // Store selections algorithm
    //---------------------------------------------------------------------

    // helper structure
    template <bool USE_TWO_PHASE, DataMemoryLayout OUT_MEM_LAYOUT, int DUMMY>
    struct StoreSelections;

    // Specialization for two phase storage
    template <DataMemoryLayout OUT_MEM_LAYOUT, int DUMMY>
    struct StoreSelections<true, OUT_MEM_LAYOUT, DUMMY> 
    {
        typedef BlockTileStorePolicy<
                BLOCK_THREADS,
                1,
                cub::STORE_DEFAULT>
            BlockPartialTileStorePolicyT;

        typedef BlockTileStore<
                BlockPartialTileStorePolicyT,
                DIM,
                OUT_MEM_LAYOUT,
                BlockSelectIfPolicyT::IO_BACKEND,
                SampleT,
                OffsetT>
            BlockPartialTileStoreT;

        typedef BlockTileStorePolicy<
                BLOCK_THREADS,
                POINTS_PER_THREAD,
                cub::STORE_DEFAULT>
            BlockFullTileStorePolicyT;

        typedef BlockTileStore<
                BlockFullTileStorePolicyT,
                DIM,
                OUT_MEM_LAYOUT,
                BlockSelectIfPolicyT::IO_BACKEND,
                SampleT,
                OffsetT>
            BlockFullTileStoreT;

        struct _TempStorage
        {
            /// (global) number of points selected by this block of threads
            OffsetT numSelectedPoints;
            // Item exchange type
            PointT rawExchange[TILE_POINTS];

            // indicates how many selected points are already stored in smem by current tile
            OffsetT tileSelectedPointsOffset;
        };

        struct TempStorage : cub::Uninitialized<_TempStorage> {};

        // thread-private fields
        _TempStorage &  auxStorage;
        SampleT *       d_selectedOut;
        
        /**
         * @brief      Constructor
         *
         * @param      ts     reference to shared memory storage
         * @param      d_out  Pointer to device memory, where we store selected items.
         */
        __device__ __forceinline__ StoreSelections(
            TempStorage&    ts,
            SampleT *       d_out,
            OffsetT         outOffset)
        :
            auxStorage(ts.Alias()),
            d_selectedOut(d_out)
        {
            if (threadIdx.x == 0)
            {
                auxStorage.numSelectedPoints = 0;
            }
            if (OUT_MEM_LAYOUT == COL_MAJOR)
            {
                d_selectedOut += outOffset;
            }
            else if (OUT_MEM_LAYOUT == ROW_MAJOR)
            {
                d_selectedOut += outOffset * DIM;
            }
            else
            {
                assert(0);
            }
        }

        __device__ __forceinline__ void compact(
            PointT          (&items)[POINTS_PER_THREAD],
            unsigned int    &selectionMask)
        {
            /**
             * Compact selected items in smem, with warp-level selection count aggregation.
             * 
             * In CUB's selectIf implementation they use scan to determine indexes where individual threads should store
             * their selected values. However that solution need additional registers to store selection flags and to 
             * store output indicies. Which IMHO is unnecessary, thus I use warp aggregation.
             */
            #pragma unroll
            for (int ITEM = 0; ITEM < POINTS_PER_THREAD; ++ITEM)
            {
                // count how many threads have selected ITEM
                int count = selectionMask & (1 << ITEM);
                int warpMask = __ballot(count);
                count = __popc(warpMask);

                if (count == 0)
                {
                    continue;
                }

                OffsetT selectedPointsOffset = 0;
                // determine warp offset to store selected points
                if (cub::LaneId() == 0)
                {
                    selectedPointsOffset = atomicAdd(&auxStorage.tileSelectedPointsOffset, count);
                }
                broadcast(selectedPointsOffset, 0);
                // calculate this thread idx
                // warpMask &= (1 << cub::LaneId()) - 1;
                int idx = __popc(warpMask & cub::LaneMaskLt());

                // store data
                if (selectionMask & (1 << ITEM))
                {
                    auxStorage.rawExchange[selectedPointsOffset + idx] = items[ITEM];
                }
            }
            // make sure all writes to smem are done
            __syncthreads();
        }

        __device__ __forceinline__ void store(
            OffsetT     globalOutputOffset,
            OffsetT     selPointsCnt,
            OffsetT     stride)
        {
            typedef SampleT AliasedFullTilePoints[POINTS_PER_THREAD][DIM];
            typedef SampleT AliasedPartTilePoints[1][DIM];

            if (selPointsCnt == TILE_POINTS)
            {
                // XXX: BANK CONFLICTS! for any point type >= 2D
                BlockFullTileStoreT::storeTileFromRowM(
                    d_selectedOut,
                    *reinterpret_cast<AliasedFullTilePoints*>(auxStorage.rawExchange + 
                        threadIdx.x * POINTS_PER_THREAD),
                    globalOutputOffset,
                    stride);
            }
            else
            {
                #pragma unroll
                for (int p = 0; p < POINTS_PER_THREAD; ++p)
                {
                    if (BLOCK_THREADS <= selPointsCnt)
                    {
                        // XXX: BANK CONFLICTS! for any point type >= 2D
                        BlockPartialTileStoreT::storeTileFromRowM(
                            d_selectedOut,
                            *reinterpret_cast<AliasedPartTilePoints*>(auxStorage.rawExchange + 
                                p * BLOCK_THREADS + threadIdx.x),
                            globalOutputOffset,
                            stride);
                    }
                    else
                    {
                        BlockPartialTileStoreT::storeTileFromRowM(
                            d_selectedOut,
                            *reinterpret_cast<AliasedPartTilePoints*>(auxStorage.rawExchange + 
                                p * BLOCK_THREADS + threadIdx.x),
                            globalOutputOffset,
                            selPointsCnt,
                            stride);

                    }
                    globalOutputOffset += BLOCK_THREADS;
                    selPointsCnt -= BLOCK_THREADS;
                    if (selPointsCnt <= 0)
                    {
                        break;
                    }
                }
            }
        }

        /**
         * Store flagged items to output
         */
        __device__ __forceinline__ void operator()(
            PointT         (&items)[POINTS_PER_THREAD],
            unsigned int    selectionMask,
            int             stride)
        {
            if (threadIdx.x == 0)
            {
                auxStorage.tileSelectedPointsOffset = 0;
            }
            __syncthreads();

            // compact items in shared memory
            compact(items, selectionMask);

            // read current output offset 
            OffsetT globalOutputOffset = auxStorage.numSelectedPoints;
            OffsetT selPointsCnt = auxStorage.tileSelectedPointsOffset;
            __syncthreads();

            // update block overall selected points counter
            if (threadIdx.x == 0)
            {
                atomicAdd(&auxStorage.numSelectedPoints, selPointsCnt);
            }

            // store data to gmem
            store(globalOutputOffset, selPointsCnt, stride);
        }
    };

    /**
     * Specialization for warp direct store algorithm
     */
    template <DataMemoryLayout OUT_MEM_LAYOUT, int DUMMY>
    struct StoreSelections<false, OUT_MEM_LAYOUT, DUMMY> 
    {
        // shared memory 
        struct _TempStorage
        {
            /// (global) number of points selected by this block of threads
            OffsetT numSelectedPoints;
        };

        struct TempStorage : cub::Uninitialized<_TempStorage> {};
        
        // thread-private fields
        _TempStorage &  tempStorage;
        SampleT *       d_selectedOut;     
        
        __device__ __forceinline__ StoreSelections(
            TempStorage&    ts,
            SampleT *       d_out,
            OffsetT         outOffset)
        :
            tempStorage(ts.Alias()),
            d_selectedOut(d_out)
        {
            if (threadIdx.x == 0)
            {
                tempStorage.numSelectedPoints = 0;
            }
            if (OUT_MEM_LAYOUT == COL_MAJOR)
            {
                d_selectedOut += outOffset;
            }
            else if (OUT_MEM_LAYOUT == ROW_MAJOR)
            {
                d_selectedOut += outOffset * DIM;
            }
            else
            {
                assert(0);
            }
        }

        __device__ __forceinline__ void store(
            PointT const                &point,
            OffsetT                     offset,
            OffsetT                     stride,
            cub::Int2Type<ROW_MAJOR>    memLayout)
        {
            reinterpret_cast<PointT*>(d_selectedOut)[offset] = point;
        }

        __device__ __forceinline__ void store(
            PointT const                &point,
            OffsetT                     offset,
            OffsetT                     stride,
            cub::Int2Type<COL_MAJOR>    memLayout)
        {
            for (int d = 0; d < DIM; ++d)
            {
                d_selectedOut[d * stride + offset] = point.array[d];
            }
        }

        /**
         * Store flagged items to output
         */
        __device__ __forceinline__ void operator()(
            PointT         (&items)[POINTS_PER_THREAD],
            unsigned int    selectionMask,
            OffsetT         stride)
        {
            /**
             *  Store selections using warp aggregated fashion. There is no need of any 
             *  synchronization and data compression in smem. However this implementation suffers
             *  from divergence.
             *  
             *  We're not using selection flags scan (as in CUB's select_if) in order to reduce 
             *  register usage. 
             */
            
            #pragma unroll
            for (int ITEM = 0; ITEM < POINTS_PER_THREAD; ++ITEM)
            {
                // count how many threads have selected ITEM
                int count = selectionMask & (1 << ITEM);
                int warpMask = __ballot(count);
                count = __popc(warpMask);

                if (count == 0)
                {
                    continue;
                }

                OffsetT selectedPointsOffset = 0;
                // determine warp offset to store selected points
                if (cub::LaneId() == 0)
                {
                    selectedPointsOffset = atomicAdd(&tempStorage.numSelectedPoints, count);
                }
                broadcast(selectedPointsOffset, 0);

                // calculate this thread idx
                // warpMask &= (1 << cub::LaneId()) - 1;
                int idx = __popc(warpMask & cub::LaneMaskLt());

                // store data
                if (selectionMask & (1 << ITEM))
                {
                    // reinterpret_cast<PointT*>(d_selectedOut)[selectedPointsOffset + idx] 
                    //   = items[ITEM];
                    store(items[ITEM], selectedPointsOffset + idx, stride, 
                        cub::Int2Type<OUT_MEM_LAYOUT>());
                }
            }
        }
    };

    //---------------------------------------------------------------------
    //  Type definition for internal store selections algorithm
    //---------------------------------------------------------------------
    
    typedef typename cub::If<STORE_TWO_PHASE,
                StoreSelections<true, MEM_LAYOUT, 0>,
                StoreSelections<false, MEM_LAYOUT, 0>>::Type StoreSelectionsT;

    // Shared memory type for this threadblock
    typedef typename StoreSelectionsT::TempStorage _TempStorage;


public:
    // Alias wrapper allowing storage to be unioned
    struct TempStorage : cub::Uninitialized<_TempStorage> {};
private:

    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage&       tempStorage;       ///< Reference to tempStorage
    SampleT const *     d_in;              ///< Input items
    SelectOpT           selectOp;          ///< Selection operator
    StoreSelectionsT    internalStore;

    //---------------------------------------------------------------------
    // Utility methods for initializing the selections
    //---------------------------------------------------------------------

    /**
     * Initialize selections (full tile)
     */
    __device__ __forceinline__ void initializeSelections(
        PointT                      (&items)[POINTS_PER_THREAD],
        unsigned int                &selectionFlags)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < POINTS_PER_THREAD; ++ITEM)
        {
            selectionFlags |= (selectOp(items[ITEM].array) << ITEM);
        }
    }

    /**
     * Initialize selections (partial tile)
     */
    __device__ __forceinline__ void initializeSelections(
        OffsetT                     validPoints,
        PointT                      (&items)[POINTS_PER_THREAD],
        unsigned int                &selectionFlags)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < POINTS_PER_THREAD; ++ITEM)
        {
            if (BLOCK_THREADS <= validPoints)
            {
                selectionFlags |= (selectOp(items[ITEM].array) << ITEM);
            }
            else if (threadIdx.x < validPoints)
            {
                selectionFlags |= (selectOp(items[ITEM].array) << ITEM);
            }
            validPoints -= BLOCK_THREADS;
            // stop when processed all valid input
            if (validPoints <= 0)
            {
                break;
            }
        }
    }

    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles
    //---------------------------------------------------------------------

    /**
     * Process full tile.
     */
    __device__ __forceinline__ void consumeFullTile(
        OffsetT     tilePointsOffset,       ///< Tile offset
        OffsetT     inStride,               ///< Distance between point's subsequent coordinates
        OffsetT     outStride)              ///< Distance between point's subsequent coordinates
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> items;
        unsigned int selectionMask = 0;     // bitfield mask indicating which loaded items are selected.  

        BlockTileLoadT::loadTile2RowM(d_in, items.data, tilePointsOffset, inStride);

        // initialize selectionMask
        initializeSelections(items.pointsRef(), selectionMask);   

        // store flagged items
        internalStore(items.pointsRef(), selectionMask, outStride);
    }

    /**
     * Process partial tile.
     */
    __device__ __forceinline__ void consumePartialTile(
        OffsetT     tilePointsOffset,       ///< Tile offset
        OffsetT     validPoints,            ///< Number of valid points within this tile of data
        OffsetT     inStride,               ///< Distance between point's subsequent coordinates
        OffsetT     outStride)              ///< Distance between point's subsequent coordinates
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> items;
        unsigned int selectionMask = 0;     // bitfield mask indicating which loaded items are selected.  

        BlockTileLoadT::loadTile2RowM(d_in, items.data, tilePointsOffset, validPoints, inStride);

        // initialize selectionMask
        initializeSelections(validPoints, items.pointsRef(), selectionMask);   

        // store flagged items
        internalStore(items.pointsRef(), selectionMask, outStride);
    }

    /**
     * Process a sequence of input tiles
     *
     * @param[in]  numPoints  The number of input points
     * @param[in]  inStride   Input points stride
     * @param[in]  outStride  Output points stride
     *
     * @return     Number of selected points
     */
    __device__ __forceinline__ OffsetT consumeTiles(
        OffsetT numPoints,
        OffsetT inStride,
        OffsetT outStride)
    {
        OffsetT numTiles = (numPoints + TILE_POINTS - 1) / TILE_POINTS;

        for (int t = 0, tileOffset = 0; t < numTiles; ++t, tileOffset += TILE_POINTS)
        {
            // #ifdef RD_DEBUG
            //     if (threadIdx.x == 0)
            //     {
            //         _CubLog("tileOffset: %d\n", tileOffset);
            //     }
            // #endif

            if (tileOffset + TILE_POINTS > numPoints)
            {
                consumePartialTile(tileOffset, numPoints - tileOffset, inStride, outStride);
            }
            else
            {
                consumeFullTile(tileOffset, inStride, outStride);
            }
        }
        // synchronize to make sure that all warps updated numSelectedPoints
        __syncthreads();

        // #ifdef RD_DEBUG
        //     if (threadIdx.x == 0)
        //     {
        //         _CubLog("numSelectedPoints: %d\n", tempStorage.Alias().numSelectedPoints);
        //     }
        // #endif

        return tempStorage.Alias().numSelectedPoints;
    }

public:
    //-----------------------------------------------------------------------------
    //  Interface
    //-----------------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    BlockSelectIf(
        TempStorage         &tempStorage,       ///< Reference to tempStorage
        SampleT const *     d_in,               ///< Input data
        SampleT *           d_selectedOut,      ///< Output data
        SelectOpT           selectOp,           ///< Selection operator
        OffsetT             outOffset = 0)      ///< Offset for selected points container
    :
        tempStorage(tempStorage.Alias()),
        d_in(d_in),
        selectOp(selectOp),
        internalStore(tempStorage.Alias(), d_selectedOut, outOffset)
    {}


    /**
     * @brief      Scan range of points and selects data according to given criteria.
     *
     * @param[in]  startOffset  Defines position from which we start scanning input data.
     * @param[in]  numPoints    Number of points to scan.
     * @param[in]  inStride     Distance (in terms of number of samples) between point's consecutive
     *                          coordinates.
     * @param[in]  outStride    As @p inStride but for output storage.
     *
     * @return     Number of selected items.
     */
    __device__ __forceinline__ OffsetT scanRange(
        OffsetT     startOffset,
        OffsetT     numPoints,
        OffsetT     inStride,
        OffsetT     outStride)
    {
        d_in += startOffset;

        if (threadIdx.x == 0)
        {
            tempStorage.Alias().numSelectedPoints = 0;
        }
        __syncthreads();
        return consumeTiles(numPoints, inStride, outStride);
    }

};

} // end namespace gpu
} // end namespace rd
