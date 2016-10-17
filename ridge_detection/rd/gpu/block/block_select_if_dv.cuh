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

#include "rd/utils/memory.h"

#include "rd/gpu/block/block_tile_load_store4.cuh"
#include "rd/gpu/block/block_dynamic_vector.cuh"
#include "rd/gpu/warp/warp_functions.cuh"

#include "cub/util_type.cuh"
#include "cub/util_ptx.cuh"
#include "cub/iterator/cache_modified_input_iterator.cuh"

#include <type_traits>

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
    DataMemoryLayout    INPUT_MEM_LAYOUT,               ///< Input data memory layout.
    typename            SelectOpT,                      ///< Selection operator type
    typename            VectorT,                        ///< Vector data type responsible for output data storage management
    typename            SampleT,                        ///< Input data single coordinate data type.
    typename            OffsetT,                        ///< Signed integer type for global offsets
    bool                STORE_TWO_PHASE>                ///< Wheather or not to perform two phase when storing selected itmes to gmem
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
            INPUT_MEM_LAYOUT,
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
    template <bool USE_TWO_PHASE, int DUMMY>
    struct StoreSelections;

    // Specialization for two phase storage
    template <int DUMMY>
    struct StoreSelections<true, DUMMY> 
    {

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
        _TempStorage & auxStorage;
        VectorT & outputVector;     ///< Dynamic vector type, responsible for selected output items storage management
        
        __device__ __forceinline__ StoreSelections(
            TempStorage&    ts,
            VectorT &       vec)
        :
            auxStorage(ts.Alias()),
            outputVector(vec)
        {
            if (threadIdx.x == 0)
            {
                auxStorage.numSelectedPoints = 0;
            }
        }

        /**
         * Store flagged items to output
         */
        __device__ __forceinline__ void impl(
            PointT         (&items)[POINTS_PER_THREAD],
            unsigned int    &selectionMask)
        {
            if (threadIdx.x == 0)
            {
                auxStorage.tileSelectedPointsOffset = 0;
            }
            __syncthreads();

            /**
             * Compact selected items in smem
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
                warpMask &= (1 << cub::LaneId()) - 1;
                int idx = __popc(warpMask);

                // store data
                if (selectionMask & (1 << ITEM))
                {
                    auxStorage.rawExchange[selectedPointsOffset + idx] = items[ITEM];
                }
            }
            // make sure all writes to smem are done
            __syncthreads();

            // read current output offset 
            OffsetT globalOutputOffset = auxStorage.numSelectedPoints;
            OffsetT selPointsCnt = auxStorage.tileSelectedPointsOffset;
            __syncthreads();

            // update block overall selected points counter
            if (threadIdx.x == 0)
            {
                atomicAdd(&auxStorage.numSelectedPoints, selPointsCnt);
            }

            // make sure we have enough space to store new points
            outputVector.resize((globalOutputOffset + selPointsCnt) * DIM);
            // update vector's elements count
            outputVector.blockIncrementItemsCnt(selPointsCnt * DIM);

            /*
             *   Store tile of selected points to output destination
             *   
             *   BlockTileLoadStore works only with data in thread-private storage, so I can't use it here.
             *   And probably the cost of moving data again from smem to local would be to expensive.
             */
            for (int k = threadIdx.x; k < selPointsCnt; k += BLOCK_THREADS)
            {
                reinterpret_cast<PointT*>(outputVector.begin())[globalOutputOffset + k] = auxStorage.rawExchange[k];
            }

        }
    };

    // specialization for warp direct store algorithm
    template <int DUMMY>
    struct StoreSelections <false, DUMMY>
    {

        // shared memory 
        struct _TempStorage
        {
            OffsetT numSelectedPoints;
        };

        struct TempStorage : cub::Uninitialized<_TempStorage> {};
        
        // thread-private fields
        _TempStorage &tempStorage;
        VectorT & outputVector;     ///< Dynamic vector type, responsible for selected output items storage management
        
        __device__ __forceinline__ StoreSelections(
            TempStorage&    ts,
            VectorT &       vec)
        :
            tempStorage(ts.Alias()),
            outputVector(vec)   
        {
            if (threadIdx.x == 0)
            {
                tempStorage.numSelectedPoints = 0;
            }
        }

        /**
         * Store flagged items to output
         */
        __device__ __forceinline__ void impl(
            PointT         (&items)[POINTS_PER_THREAD],
            unsigned int    &selectionMask)
        {
            /**
             *  Store without warp selections compression, with numerous divergence, however simple implementation.
             *  No need of any synchronization and data compression in smem.
             *  
             *  XXX: Wersja dla wierszowego zapisu danych w pamięci!!
             *  
             *  Don't use selection flags scan (as in CUB's select_if) in order to reduce register usage. 
             */
            #pragma unroll
            for (int ITEM = 0; ITEM < POINTS_PER_THREAD; ++ITEM)
            {
                // count how many threads have selected ITEM
                int count = selectionMask & (1 << ITEM);
                int warpMask = __ballot(count);
                count = __popc(warpMask);

                OffsetT selectedPointsOffset = 0;
                if (count)
                {
                    // determine warp offset to store selected points
                    if (cub::LaneId() == 0)
                    {
                        selectedPointsOffset = atomicAdd(&tempStorage.numSelectedPoints, count);
                    }
                    broadcast(selectedPointsOffset, 0);
                }
                __syncthreads();

                // make sure we have enough space to store new points
                outputVector.resize(tempStorage.numSelectedPoints * DIM);
                // update vector's elements count
                outputVector.warpIncrementItemsCnt(count * DIM);

                if (count)
                {
                    // calculate this thread idx
                    warpMask &= (1 << cub::LaneId()) - 1;
                    int idx = __popc(warpMask);

                    // store data
                    if (selectionMask & (1 << ITEM))
                    {
                        reinterpret_cast<PointT*>(outputVector.begin())[selectedPointsOffset + idx] = items[ITEM];
                    }
                }
            }
        }
    };

    //---------------------------------------------------------------------
    //  Type definition for internal store selections algorithm
    //---------------------------------------------------------------------
    
    typedef typename cub::If<STORE_TWO_PHASE,
                StoreSelections<true, 0>,
                StoreSelections<false, 0>>::Type StoreSelectionsT;


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
        OffsetT     tilePointsOffset,   ///< Tile offset
        OffsetT     stride)             ///< Distance between point's subsequent coordinates
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> items;
        unsigned int selectionMask = 0;     // bitfield mask indicating which loaded items are selected.  

        BlockTileLoadT::loadTile2Row(d_in, items.data, tilePointsOffset, stride);

        // initialize selectionMask
        initializeSelections(items.pointsRef(), selectionMask);   

        // store flagged items
        internalStore.impl(items.pointsRef(), selectionMask);
    }

    /**
     * Process partial tile.
     */
    __device__ __forceinline__ void consumePartialTile(
        OffsetT     tilePointsOffset,   ///< Tile offset
        OffsetT     validPoints,        ///< Number of valid points within this tile of data
        OffsetT     stride)             ///< Distance between point's subsequent coordinates
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> items;
        unsigned int selectionMask = 0;     // bitfield mask indicating which loaded items are selected.  

        BlockTileLoadT::loadTile2Row(d_in, items.data, tilePointsOffset, validPoints, stride);

        // initialize selectionMask
        initializeSelections(validPoints, items.pointsRef(), selectionMask);   

        // store flagged items
        internalStore.impl(items.pointsRef(), selectionMask);
    }

    /**
     * Process a sequence of input tiles
     */
    __device__ __forceinline__ OffsetT consumeTiles(
        OffsetT numPoints,
        OffsetT stride)
    {
        OffsetT numTiles = (numPoints + TILE_POINTS - 1) / TILE_POINTS;

        for (int t = 0, tileOffset = 0; t < numTiles; ++t, tileOffset += TILE_POINTS)
        {

            // #ifdef RD_DEBUG
            //     if (threadIdx.x == 0)
            //     {
            //         _CubLog(">>>>       tileOffset: %d      <<<<<\n", tileOffset);
            //     }
            // #endif

            if (tileOffset + TILE_POINTS > numPoints)
            {
                consumePartialTile(tileOffset, numPoints - tileOffset, stride);
            }
            else
            {
                consumeFullTile(tileOffset, stride);
            }
        }
        // synchronize to make sure that all warps updated numSelectedPoints
        __syncthreads();

        #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                _CubLog("numSelectedPoints:\t %d, outVec.begin(): %p\n", tempStorage.Alias().numSelectedPoints, internalStore.outputVector.begin());
            }
        #endif

        return tempStorage.Alias().numSelectedPoints;
    }

public:
    //-----------------------------------------------------------------------------
    //  Interface
    //-----------------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__ BlockSelectIf(
        TempStorage         &ts,                ///< Reference to tempStorage
        SampleT const *     d_in,               ///< Input data
        VectorT &           d_out,              ///< dynamic vector responsible for output storage management 
        SelectOpT           selectOp)           ///< Selection operator
    :
        tempStorage(ts.Alias()),
        d_in(d_in),
        selectOp(selectOp),
        internalStore(ts.Alias(), d_out)
    {
    }


    /**
     * @brief      Scan range of points and selects data according to given criteria.
     *
     * @param[in]  startOffset  Defines position from which we start scanning input data.
     * @param[in]  numPoints    Number of points to scan.
     * @param[in]  stride       Distance (in terms of number of samples) between point's consecutive coordinates.
     *
     * @return     Number of selected items.
     */
    __device__ __forceinline__ OffsetT scanRange(
        OffsetT     startOffset,
        OffsetT     numPoints,
        OffsetT     stride)
    {
        d_in += startOffset;

        if (threadIdx.x == 0)
        {
            tempStorage.Alias().numSelectedPoints = 0;
        }
        __syncthreads();
        return consumeTiles(numPoints, stride);
    }

};

} // end namespace gpu
} // end namespace rd
