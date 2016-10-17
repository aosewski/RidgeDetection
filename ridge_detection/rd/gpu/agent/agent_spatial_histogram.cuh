/**
 * @file agent_spatial_histogram.cuh
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

#include "cub/util_type.cuh"
#include "cub/thread/thread_load.cuh"

// #ifndef RD_DEBUG
// #define NDEBUG      // for disabling assert macro
// #endif 
#include <assert.h>

namespace rd
{
namespace gpu
{
namespace detail
{

/**
 * Parameterized tuning policy for AgentSpatialHistogram.
 *
 * @tparam     _BLOCK_THREADS      Threads per thread block.
 * @tparam     _POINTS_PER_THREAD  Number of points processed by each thread simultaneously.
 * @tparam     _LOAD_MODIFIER      Cache load modifier for reading input elements.
 * @tparam     _IO_BACKEND         I/O backend (CUB/Trove) used to perform memory load/write.
 */
template <
    int                     _BLOCK_THREADS,
    int                     _POINTS_PER_THREAD,
    cub::CacheLoadModifier  _LOAD_MODIFIER,                 
    BlockTileIOBackend      _IO_BACKEND>                                                        
struct AgentSpatialHistogramPolicy
{
    enum 
    {
        BLOCK_THREADS       = _BLOCK_THREADS,
        POINTS_PER_THREAD   = _POINTS_PER_THREAD,
    };

    static const cub::CacheLoadModifier     LOAD_MODIFIER           = _LOAD_MODIFIER;
    static const BlockTileIOBackend         IO_BACKEND              = _IO_BACKEND;
};

/******************************************************************************
 * 
 ******************************************************************************/

/*
 * @brief      Thread block abstractions for cooperative device-wide histogram calculation
 *
 * @tparam     AgentSpatialHistogramPolicyT  Tuning parameters policy.
 * @tparam     INPUT_MEMORY_LAYOUT           Data memory layout (COL/ROW_MAJOR)
 * @tparam     DIM                           Input points dimension.
 * @tparam     SampleT                       Point coordinate data type.
 * @tparam     CounterT                      Histogram bin's data type.
 * @tparam     OffsetT                       Integer type for offsets, strides, etc.
 * @tparam     PointDecodeOpT                Functor providing mappint point's onto bins operation.
 * @tparam     CUSTOM_DECODE_OP              Whether or not we should use default scheme for point
 *                                           mapping, where we assume that one point belongs to 
 *                                           only one bin.
 * @tparam     USE_GMEM_PRIV_HIST            Whether or not to use thread block's privatized
 *                                           intermediate histogram stored in global memory.
 */
template <
    typename            AgentSpatialHistogramPolicyT,
    DataMemoryLayout    INPUT_MEMORY_LAYOUT,
    int                 DIM,
    typename            SampleT,
    typename            CounterT,
    typename            OffsetT,
    typename            PointDecodeOpT,
    bool                CUSTOM_DECODE_OP,
    bool                USE_GMEM_PRIV_HIST>
class AgentSpatialHistogram
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS           = AgentSpatialHistogramPolicyT::BLOCK_THREADS,
        POINTS_PER_THREAD       = AgentSpatialHistogramPolicyT::POINTS_PER_THREAD,

        TILE_POINTS             = POINTS_PER_THREAD * BLOCK_THREADS,
    };

    typedef BlockTileLoadPolicy<
            BLOCK_THREADS,
            POINTS_PER_THREAD, 
            AgentSpatialHistogramPolicyT::LOAD_MODIFIER> 
        BlockTileLoadPolicyT;

    typedef BlockTileLoad<
            BlockTileLoadPolicyT,
            DIM,
            INPUT_MEMORY_LAYOUT, 
            AgentSpatialHistogramPolicyT::IO_BACKEND,
            SampleT, 
        OffsetT> BlockTileLoadT;

    //---------------------------------------------------------------------
    // Bin values accumulation through single tile of data
    //---------------------------------------------------------------------

    // Point accumulation to block privatized histograms. (full tile specialization)
    template <bool USE_CUSTOM_DECODE_OP, int DUMMY>
    struct PointAccumulator {};

    template <int DUMMY>
    struct PointAccumulator<true, DUMMY>
    {
        /// The transform operator for determining output bin-ids from point coordinates
        PointDecodeOpT &pointDecodeOp;
        int numBins;

        __device__ __forceinline__ PointAccumulator(
            PointDecodeOpT& decodeOp,
            int             nb)
        :
            pointDecodeOp(decodeOp),
            numBins(nb)
        {}

        __device__ __forceinline__ void operator()(
            CounterT*   histogram,
            SampleT     samples[POINTS_PER_THREAD][DIM])
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                pointDecodeOp(samples[p], histogram);
            }
        }
   
        // Point accumulation to (block privatized) histogram. (partial tile specialization)
        __device__ __forceinline__ void operator()(
            CounterT*   histogram,
            SampleT     samples[POINTS_PER_THREAD][DIM],
            int         validPoints)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if ((threadIdx.x + p * BLOCK_THREADS) < validPoints)
                {
                    pointDecodeOp(samples[p], histogram);
                }
            }
        }
    };

    template <int DUMMY>
    struct PointAccumulator<false, DUMMY>
    {
        /// The transform operator for determining output bin-ids from point coordinates
        PointDecodeOpT &pointDecodeOp;
        int numBins;

        __device__ __forceinline__ PointAccumulator(
            PointDecodeOpT& decodeOp,
            int             nb)
        :   
            pointDecodeOp(decodeOp),
            numBins(nb)
        {}

        __device__ __forceinline__ void operator()(
            CounterT*   histogram,
            SampleT     samples[POINTS_PER_THREAD][DIM])
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                int bin = pointDecodeOp(samples[p]);
                assert(bin >= 0 && bin < numBins);
                atomicAdd(histogram + bin, 1);
            }
        }
       
        // Point accumulation to (block privatized) histograms. (partial tile specialization)
        __device__ __forceinline__ void operator()(
            CounterT*   histogram,
            SampleT     samples[POINTS_PER_THREAD][DIM],
            int         validPoints)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if ((threadIdx.x + p * BLOCK_THREADS) < validPoints)
                {
                    int bin = pointDecodeOp(samples[p]);
                    assert(bin >= 0 && bin < numBins);
                    atomicAdd(histogram + bin, 1);
                }
            }
        }
    };

    typedef PointAccumulator<CUSTOM_DECODE_OP, 0> PointAccumulatorT;

    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    /// pointer to input data (gmem)
    SampleT const * inputSamples;
    /// pointer to privatized (per block) histogram (gmem)
    CounterT* privatizedHistogram;
    /// pointer to final output histogram (gmem)
    CounterT* outputHistogram;
    /// specialized implementation of histogram point's accumulation.
    PointAccumulatorT pointAccumulator;
    int numBins;

    //---------------------------------------------------------------------
    // Initialize bin counters
    //---------------------------------------------------------------------

    __device__ __forceinline__ void initPrivateBinCounters(
        CounterT *  privatizedHist)
    {
        // Initialize the location of this block's privatized histogram
        privatizedHistogram = privatizedHist + blockIdx.x * numBins;
        
        for (int bin = threadIdx.x; bin < numBins; bin += BLOCK_THREADS)
        {
            privatizedHistogram[bin] = 0;
        }

        __syncthreads();
    }

   
    //---------------------------------------------------------------------
    // Privatized histograms reduction
    //---------------------------------------------------------------------

    __device__ __forceinline__ void storeOutput()
    {
        CounterT binValue;
        int binsLeft = numBins;

        for (int b = 0; b < numBins; b += BLOCK_THREADS)
        {
            if (threadIdx.x < binsLeft)
            {
                binValue = cub::ThreadLoad<cub::LOAD_CS>(privatizedHistogram + b + threadIdx.x);
                // binValue = *(privatizedHistogram + b + threadIdx.x);
                atomicAdd(outputHistogram + b + threadIdx.x, binValue);
            }

            binsLeft -= BLOCK_THREADS;
        }
    }

    //---------------------------------------------------------------------
    // Tile processing
    //---------------------------------------------------------------------

    // Consume a full tile of data samples
    __device__ __forceinline__ void consumeFullTile(
        OffsetT     tilePointsOffset,
        OffsetT     stride)
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> samples;
        BlockTileLoadT::loadTile2RowM(inputSamples, samples.data, tilePointsOffset, stride);
        
        pointAccumulator(privatizedHistogram, samples.data);
    }

    // Consume a partial tile of data samples
    __device__ __forceinline__ void consumePartialTile(
        OffsetT     tilePointsOffset,
        int         validPoints,
        OffsetT     stride)
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> samples;
        BlockTileLoadT::loadTile2RowM(inputSamples, samples.data, tilePointsOffset, validPoints, 
            stride);
        
        pointAccumulator(privatizedHistogram, samples.data, validPoints);
    }

    /**
     * @brief      Partition input data into fixed size tiles and process them.
     *
     * @param[in]  numPoints  Number of points to be partitioned into tiles.
     * @param[in]  stride     Number of elements between point consecutive coordinates.
     */
    __device__ __forceinline__ void consumeTiles(
        OffsetT numPoints,
        OffsetT stride)
    {
        OffsetT numTiles = (numPoints + TILE_POINTS - 1) / TILE_POINTS;

        for (int t = blockIdx.x; t < numTiles; t += gridDim.x)
        {
            OffsetT tileOffset = t * TILE_POINTS;
            // #ifdef RD_DEBUG
            //     if (threadIdx.x == 0)
            //     {
            //         _CubLog("tileOffset: %d\n", tileOffset);
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
    }

public:
    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------


    /**
     * Constructor
     */
    __device__ __forceinline__ AgentSpatialHistogram(
        SampleT const *     inSamples,
        int                 numBins,
        CounterT*           outHist,
        CounterT*           privHist,
        PointDecodeOpT &    pointDecodeOp)
    :
        outputHistogram(outHist),
        pointAccumulator(pointDecodeOp, numBins),
        inputSamples(inSamples),
        numBins(numBins)
    {
        if (USE_GMEM_PRIV_HIST)
        {
            initPrivateBinCounters(privHist);
        }
        else
        {
            privatizedHistogram = outHist;
        }
    }

    /**
     * Consume input data.
     *
     * @param[in]  numPoints  Number of n-dimensional points to build histogram from.
     * @param[in]  stride     Number of elements between point's consecutive coordinates.
     */
    __device__ __forceinline__ void consumeRange(
        OffsetT numPoints,
        OffsetT stride)         
    {
        consumeTiles(numPoints, stride);
        // make sure all threads updated their bins
        __syncthreads();

        // only blocks which have done some work have to store their results
        OffsetT numTiles = (numPoints + TILE_POINTS - 1) / TILE_POINTS;
        if (USE_GMEM_PRIV_HIST && blockIdx.x < numTiles)
        {
            storeOutput();
        }
    }
};

} // end namespace detail
} // end namespace gpu
} // end namespace rd
