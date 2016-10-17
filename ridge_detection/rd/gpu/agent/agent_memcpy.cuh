/**
 * @file agent_memcpy.cuh
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

#include "cub/util_ptx.cuh"
#include "cub/util_debug.cuh"

#include <common_functions.h>
#include <assert.h>

namespace rd
{
namespace gpu
{


//---------------------------------------------------------------------
// Thread block abstraction for efficient memory copy inside device global memory.
//---------------------------------------------------------------------

/**
 * @brief      AgentMemcpy provides efficient implementation of memory copy between two device
 *             memory regions.
 *
 * @tparam     BlockTileLoadPolicyT   Parameterized tile loading tuning policy.
 * @tparam     BlockTileStorePolicyT  Parameterized tile storing tuning policy.
 * @tparam     DIM                    Data dimension
 * @tparam     MEM_LAYOUT             Input and output memory layout.
 * @tparam     PRIVATE_MEM_LAYOUT     The memory layout we use to store read data in aux registers.
 * @tparam     IO_BACKEND             Backend responsible for copying data.
 * @tparam     OffsetT                Offset type - integral type.
 * @tparam     T                      Data type.
 */
template <
    typename                    BlockTileLoadPolicyT,
    typename                    BlockTileStorePolicyT,
    int                         DIM,
    rd::DataMemoryLayout        MEM_LAYOUT,
    rd::DataMemoryLayout        PRIVATE_MEM_LAYOUT,
    rd::gpu::BlockTileIOBackend IO_BACKEND,
    typename                    OffsetT,
    typename                    T>
class AgentMemcpy
{

    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// The sample type 
    typedef T SampleT;

    /// Constants
    enum
    {
        BLOCK_THREADS           = BlockTileLoadPolicyT::BLOCK_THREADS,

        POINTS_PER_THREAD       = BlockTileLoadPolicyT::POINTS_PER_THREAD,
        SAMPLES_PER_THREAD      = POINTS_PER_THREAD * DIM,

        TILE_POINTS             = POINTS_PER_THREAD * BLOCK_THREADS,
    };

    typedef rd::gpu::BlockTileLoad<
        BlockTileLoadPolicyT,
        DIM,
        MEM_LAYOUT, 
        IO_BACKEND,
        SampleT, 
        OffsetT> BlockTileLoadT;

    typedef rd::gpu::BlockTileStore<
        BlockTileStorePolicyT,
        DIM,
        MEM_LAYOUT,
        IO_BACKEND,
        SampleT,
        OffsetT> BlockTileStoreT;

    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    /// Native pointer for input data 
    SampleT const * d_in;
    /// Native pointer for output data 
    SampleT * d_out;

    //---------------------------------------------------------------------
    // Consume a full tile of data samples.
    //---------------------------------------------------------------------
    __device__ __forceinline__ void consumeFullTile(
        OffsetT                     tilePointsOffset,
        OffsetT                     inStride,
        OffsetT                     outStride,
        cub::Int2Type<rd::ROW_MAJOR>    privateMemLayout)
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> samples;

        BlockTileLoadT::loadTile2RowM(d_in, samples.data, tilePointsOffset, inStride);
        BlockTileStoreT::storeTileFromRowM(d_out, samples.data, tilePointsOffset, outStride);
    }
    __device__ __forceinline__ void consumeFullTile(
        OffsetT                     tilePointsOffset,
        OffsetT                     inStride,
        OffsetT                     outStride,
        cub::Int2Type<rd::COL_MAJOR>    privateMemLayout)
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::COL_MAJOR> samples;
        BlockTileLoadT::loadTile2ColM(d_in, samples.data, tilePointsOffset, inStride);
        BlockTileStoreT::storeTileFromColM(d_out, samples.data, tilePointsOffset, outStride);
    }

    //---------------------------------------------------------------------
    // Consume a partial tile of data samples.
    //---------------------------------------------------------------------
    __device__ __forceinline__ void consumePartialTile(
        OffsetT                     tilePointsOffset,
        int                         validPoints,
        OffsetT                     inStride,
        OffsetT                     outStride,
        cub::Int2Type<rd::ROW_MAJOR>    privateMemLayout)
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> samples;
        BlockTileLoadT::loadTile2RowM(d_in, samples.data, tilePointsOffset, validPoints, inStride);
        BlockTileStoreT::storeTileFromRowM(d_out, samples.data, tilePointsOffset, validPoints, 
            outStride);
    }
    __device__ __forceinline__ void consumePartialTile(
        OffsetT                     tilePointsOffset,
        int                         validPoints,
        OffsetT                     inStride,
        OffsetT                     outStride,
        cub::Int2Type<rd::COL_MAJOR>    privateMemLayout)
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::COL_MAJOR> samples;
        BlockTileLoadT::loadTile2ColM(d_in, samples.data, tilePointsOffset, validPoints, inStride);
        BlockTileStoreT::storeTileFromColM(d_out, samples.data, tilePointsOffset, validPoints, 
            outStride);
    }

    //---------------------------------------------------------------------
    // Tile processing
    //---------------------------------------------------------------------
    __device__ __forceinline__ void consumeTiles(
        OffsetT     numPoints,
        OffsetT     inStride,
        OffsetT     outStride,
        const int   blockId = blockIdx.x,
        const int   gridSize = gridDim.x)
    {
        OffsetT numTiles = (numPoints + TILE_POINTS - 1) / TILE_POINTS;
        for (int t = blockId; t < numTiles; t += gridSize)
        {
            OffsetT tileOffset = t * TILE_POINTS;

            if (tileOffset + TILE_POINTS > numPoints)
            {
                // #ifdef RD_DEBUG
                // if (threadIdx.x == 0)
                // {
                //     _CubLog("<<<< __AgentMemcpy__: [partial tile], tilePointsOffset %d, validPoints: %d, address range[%p - %p]\n",
                //         tileOffset, numPoints - tileOffset, d_in + tileOffset * DIM, d_in + numPoints * DIM);
                // }
                // #endif
                consumePartialTile(tileOffset, numPoints - tileOffset, inStride, outStride,
                    cub::Int2Type<PRIVATE_MEM_LAYOUT>());
            }
            else
            {
                // #ifdef RD_DEBUG
                // if (threadIdx.x == 0)
                // {
                //     _CubLog("<<<< __AgentMemcpy__: [full tile], tilePointsOffset %d, address range[%p - %p]\n", 
                //         tileOffset, d_in + tileOffset * DIM, d_in + (tileOffset + TILE_POINTS) * DIM);
                // }
                // #endif
                consumeFullTile(tileOffset, inStride, outStride, 
                    cub::Int2Type<PRIVATE_MEM_LAYOUT>());
            }
        }

            // #ifdef RD_DEBUG
            // if (threadIdx.x == 0)
            // {
            //     _CubLog("<<<< __AgentMemcpy__: consumed %d tiles!\n", numTiles);
            // }
            // #endif
    }  

public:
    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------


    /**
     * Constructor
     *
     * @param      d_input    Pointer to input data.
     * @param      d_output   Pointer to output memory region.
     * @param[in]  outOffset  Offset for output data, i.e. when we don't want to write from the
     *                        beginning but append.
     */
    __device__ __forceinline__ AgentMemcpy(
        T const *               d_input,
        T *                     d_output,
        OffsetT                 outOffset = 0)
    :
        d_in(d_input),
        d_out(d_output)
    {
        // #ifdef RD_DEBUG
        // if (threadIdx.x == 0)
        // {
        //     _CubLog("AgentMemcpy(Constructor): d_in: %p, d_out: %p\n", d_in, d_out);
        // }
        // #endif
        
        if (MEM_LAYOUT == COL_MAJOR)
        {
            d_out += outOffset;
        }
        else if (MEM_LAYOUT == ROW_MAJOR)
        {
            d_out += outOffset * DIM;
        }
        else
        {
            assert(0);
        }
    }

    /**
     * @brief Copy specifed range of input data.
     *
     * @param[in]  startOffset  The input data start offset.
     * @param[in]  numPoints    The number of points to copy.
     * @param[in]  inStride     Number of elements between input data point's consecutive
     *                          coordinates.
     * @param[in]  outStride    Number of elements between output data point's consecutive
     *                          coordinates.
     * @param[in]  singleblock  Whether to perform copy using only this block of threads or perform
     *                          device-wide copy.
     */
    __device__ __forceinline__ void copyRange(
        OffsetT startOffset,
        OffsetT numPoints,
        OffsetT inStride,
        OffsetT outStride,
        bool    singleblock = false)
    {
        d_in += startOffset;
        // #ifdef RD_DEBUG
        //     if (threadIdx.x == 0)
        //     {
        //         _CubLog("AgentMemcpy.copyRange() start: %d, numPoints: %d, inStride: %d\n", startOffset, numPoints, inStride);
        //     }
        // #endif
        if (singleblock)
        {
            consumeTiles(numPoints, inStride, outStride, 0, 1);
        }
        else
        {
            consumeTiles(numPoints, inStride, outStride);
        }
    }
};

} // end namespace gpu
} // end namespace rd