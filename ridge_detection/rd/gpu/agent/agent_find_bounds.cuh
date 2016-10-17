/**
 * @file agent_find_bounds.cuh
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
#include "cub/block/block_reduce.cuh"

namespace rd
{
namespace gpu
{
namespace detail
{

/******************************************************************************
 * Tuning policy
 ******************************************************************************/

/**
 * @brief      Parameterizable tuning policy for AgentFindBounds
 *
 * @tparam     _BLOCK_THREADS      Number of threads per thread block
 * @tparam     _POINTS_PER_THREAD  Number of points processed by one thread at the same time
 * @tparam     _REDUCE_ALGORITHM   Algorithm used to reduce block intermediate results.
 * @tparam     _LOAD_MODIFIER      Cache load modifier for input data.
 * @tparam     _IO_BACKEND         I/O backend used to perform data load (CUB/Trove)
 */
template <
    int                         _BLOCK_THREADS,
    int                         _POINTS_PER_THREAD,
    cub::BlockReduceAlgorithm   _REDUCE_ALGORITHM,              
    cub::CacheLoadModifier      _LOAD_MODIFIER,                 
    BlockTileIOBackend          _IO_BACKEND>
struct AgentFindBoundsPolicy
{
    enum
    {
        BLOCK_THREADS       = _BLOCK_THREADS,
        POINTS_PER_THREAD   = _POINTS_PER_THREAD
    };

    /// The BlockReduce algorithm to use
    static const cub::BlockReduceAlgorithm   REDUCE_ALGORITHM    = _REDUCE_ALGORITHM;        
    /// Cache load modifier for reading input elements
    static const cub::CacheLoadModifier      LOAD_MODIFIER       = _LOAD_MODIFIER;           
    /// I/O backend (CUB/Trove) responsible for reading data
    static const BlockTileIOBackend          IO_BACKEND          = _IO_BACKEND;                                                                                             
};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * @brief      Defines types, constants, per-thread fields and methods to initialize them.
 *
 * @tparam     AgentFindBoundsPolicyT  Algorithm tuning parameters.
 * @tparam     INPUT_MEM_LAYOUT        Input data memory layout (col/row-major)
 * @tparam     DIM                     Input points dimension
 * @tparam     SampleT                 Type of point's coordinates
 * @tparam     OffsetT                 Integer type for offsets
 */
template <
    typename            AgentFindBoundsPolicyT,
    DataMemoryLayout    INPUT_MEM_LAYOUT,
    int                 DIM,
    typename            SampleT,
    typename            OffsetT>                        
class AgentFindBoundsBase
{
protected:
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS        = AgentFindBoundsPolicyT::BLOCK_THREADS,
        POINTS_PER_THREAD    = AgentFindBoundsPolicyT::POINTS_PER_THREAD,
        TILE_POINTS          = POINTS_PER_THREAD * BLOCK_THREADS
    };

    typedef BlockTileLoadPolicy<
            BLOCK_THREADS, 
            POINTS_PER_THREAD, 
            AgentFindBoundsPolicyT::LOAD_MODIFIER> 
        BlockTileLoadPolicyT;

    typedef BlockTileLoad<
        BlockTileLoadPolicyT,
        DIM,
        INPUT_MEM_LAYOUT, 
        AgentFindBoundsPolicyT::IO_BACKEND,
        SampleT, 
        OffsetT> BlockTileLoadT;

    /// Parameterized BlockReduce type for block-wide intermediate results reduction
    typedef cub::BlockReduce<
            SampleT,
            BLOCK_THREADS,
            AgentFindBoundsPolicyT::REDUCE_ALGORITHM>
        BlockReduceT;

    /// Shared memory type required by this thread block
    struct _TempStorage
    {
        /*
         * XXX: dla wersji n-wymiarowej należałoby to jakoś ograniczyć, albo najlepiej zrobić
         * tak jak w CUB, czyli cześć granic mogę trzymać w smem (co się zmieści), a resztę muszę w
         * gmem. Coś jak:
         * CounterT histograms[NUM_ACTIVE_CHANNELS][PRIVATIZED_SMEM_BINS + 1];     // Smem needed for block-privatized smem histogram (with 1 word of padding)
         */

        union
        {
            // Smem needed for reducing thread's private results 
            typename BlockReduceT::TempStorage sampleReduce;       
        };
    };

public:
    /// Temporary storage type (unionable)
    struct TempStorage : cub::Uninitialized<_TempStorage> {};

protected:
    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    /// Reference to tempStorage
    _TempStorage &tempStorage;

    /// Native pointer for input samples (possibly NULL if unavailable)
    SampleT const * d_inputSamples;
    /*
        XXX: dla wersji n-wymiarowej będę potrzebował:
     
        /// The number of output bins for each channel
        int (&num_output_bins)[NUM_ACTIVE_CHANNELS];

        /// The number of privatized bins for each channel
        int (&num_privatized_bins)[NUM_ACTIVE_CHANNELS];

        /// Reference to gmem privatized histograms for each channel
        CounterT* d_privatized_histograms[NUM_ACTIVE_CHANNELS];
    */

    // Per thread private bounds
    SampleT privateMinBounds[DIM];
    SampleT privateMaxBounds[DIM];

    /// Reference to final output bounds (gmem)
    SampleT* (&d_outputBboxMin)[DIM];
    SampleT* (&d_outputBboxMax)[DIM];

    //---------------------------------------------------------------------
    // Initialize privatized bounds
    //---------------------------------------------------------------------

    // Initialize private per-thread bounds
    __device__ __forceinline__ void initThreadPrivateBounds()
    {
        // Initialize bounds to max and min values
        #pragma unroll
        for (int d = 0; d < DIM; ++d)
        {
            privateMinBounds[d] = cub::NumericTraits<SampleT>::Max();
            privateMaxBounds[d] = cub::NumericTraits<SampleT>::Lowest();
        }
    }

    //---------------------------------------------------------------------
    // Update final output bounds
    //---------------------------------------------------------------------

    // // Update final output bounds from thread private bounds.
    __device__ __forceinline__ void storePrivateThreadOutput()
    {
        // Apply thread private bounds to output bounds
        if (threadIdx.x == 0)
        {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                d_outputBboxMin[d][blockIdx.x] = privateMinBounds[d];
                d_outputBboxMax[d][blockIdx.x] = privateMaxBounds[d];
            }
        }
    }

    //---------------------------------------------------------------------
    // Debug utility
    //---------------------------------------------------------------------

    __device__ __forceinline__ void printPrivateBounds(
        int tid = -1)
    {
        if (tid >= 0)
        {
            if (threadIdx.x == tid)
            {
                for (int d = 0; d < DIM; ++d)
                {
                    _CubLog("min[%d]: %8.6e, max[%d]: %8.6e \n",
                        d, privateMinBounds[d], d, privateMaxBounds[d]);
                }
            }
        }
        else
        {
            for (int d = 0; d < DIM; ++d)
            {
                _CubLog("min[%d]: %8.6e, max[%d]: %8.6e \n",
                    d, privateMinBounds[d], d, privateMaxBounds[d]);
            }
        }
    }

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

public:
    /**
     * Constructor
     *
     * @param      tempStorage      Reference to thread block's tempStorage
     * @param      d_samples        Input data to reduce
     * @param      d_outputBboxMin  Reference to final output bounds (min)
     * @param      d_outputBboxMax  Reference to final output bounds (max)
     */
    __device__ __forceinline__ AgentFindBoundsBase(
        TempStorage         &tempStorage,
        SampleT const *     d_samples,
        SampleT*            (&d_outputBboxMin)[DIM],
        SampleT*            (&d_outputBboxMax)[DIM])
    :
        tempStorage(tempStorage.Alias()),
        d_inputSamples(d_samples),
        d_outputBboxMin(d_outputBboxMin),
        d_outputBboxMax(d_outputBboxMax)
    {
        initThreadPrivateBounds();
    }
};


/**
 * @brief      Defines first phase of two-phase finding bounds for input point cloud data
 *             algorithm.
 *
 *             In first pass each block of threads reads and process tiles of input data. It finds
 *             both max and min bounds for each dimension and stores it's (block) private results 
 *             to the output temporaray storage (gmem).
 *
 * @tparam     AgentFindBoundsPolicyT  Algorithm tuning parameters
 * @tparam     INPUT_MEM_LAYOUT        Input data memory layout
 * @tparam     DIM                     Input points dimension
 * @tparam     SampleT                 Point coordinates' data type
 * @tparam     OffsetT                 Integer type for offsets
 */
template <
    typename            AgentFindBoundsPolicyT,
    DataMemoryLayout    INPUT_MEM_LAYOUT,
    int                 DIM,
    typename            SampleT,
    typename            OffsetT>                        
class AgentFindBoundsFirstPass : 
    public AgentFindBoundsBase<AgentFindBoundsPolicyT, INPUT_MEM_LAYOUT, DIM, SampleT, OffsetT>
{};

/*
 * Specialization for ROW_MAJOR order data layout 
 */
template <
    typename            AgentFindBoundsPolicyT,
    int                 DIM,
    typename            SampleT,
    typename            OffsetT>                        
class AgentFindBoundsFirstPass<AgentFindBoundsPolicyT, ROW_MAJOR, DIM, SampleT, OffsetT> 
    : public AgentFindBoundsBase<AgentFindBoundsPolicyT, ROW_MAJOR, DIM, SampleT, OffsetT>
{

    //---------------------------------------------------------------------
    // Typedefs and constants
    //---------------------------------------------------------------------
    typedef AgentFindBoundsBase<AgentFindBoundsPolicyT, ROW_MAJOR, DIM, SampleT, OffsetT> BaseT;
public:
    typedef typename BaseT::TempStorage               TempStorage;
private:
    typedef typename BaseT::BlockTileLoadT            BlockTileLoadT;
    typedef typename BaseT::BlockReduceT              BlockReduceT;

    /// Constants
    enum
    {
        BLOCK_THREADS        = AgentFindBoundsPolicyT::BLOCK_THREADS,
        POINTS_PER_THREAD    = AgentFindBoundsPolicyT::POINTS_PER_THREAD,
        TILE_POINTS          = POINTS_PER_THREAD * BLOCK_THREADS
    };

    //---------------------------------------------------------------------
    // Search tile bounds
    //---------------------------------------------------------------------

    // Find bounds. Specialized for full tile.
    __device__ __forceinline__ void findBounds(
        SampleT     (&samples)[POINTS_PER_THREAD][DIM])
    {
        #pragma unroll
        for (int p = 0; p < POINTS_PER_THREAD; ++p)
        {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                this->privateMinBounds[d] = min(this->privateMinBounds[d], samples[p][d]);
                this->privateMaxBounds[d] = max(this->privateMaxBounds[d], samples[p][d]);
            }
        }
    }

    // Find bounds. Specialized for partial tile
    __device__ __forceinline__ void findBounds(
        SampleT     (&samples)[POINTS_PER_THREAD][DIM],
        int         validPoints)
    {
        #pragma unroll
        for (int p = 0; p < POINTS_PER_THREAD; ++p)
        {
            if ((threadIdx.x + p * BLOCK_THREADS) < validPoints)
            {
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    this->privateMinBounds[d] = min(this->privateMinBounds[d], samples[p][d]);
                    this->privateMaxBounds[d] = max(this->privateMaxBounds[d], samples[p][d]);
                }
            }
        }
    }

    //---------------------------------------------------------------------
    // Tile processing
    //---------------------------------------------------------------------

    // Consume a full tile of data samples
    __device__ __forceinline__ void consumeFullTile(
        OffsetT     blockOffset)
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> samples;
        BlockTileLoadT::loadTile2RowM(this->d_inputSamples + blockOffset * DIM, samples.data, 1);
        
        findBounds(samples.data);
    }

    // Consume a partial tile of data samples
    __device__ __forceinline__ void consumePartialTile(
        OffsetT     blockOffset,
        int         validPoints)
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> samples;
        BlockTileLoadT::loadTile2RowM(this->d_inputSamples + blockOffset * DIM, validPoints, 
            samples.data, 1);
        
        findBounds(samples.data, validPoints);
    }

public:

    /**
     * Constructor
     *
     * @param      tempStorage      Reference to thread block tempStorage
     * @param      d_samples        Input data to reduce
     * @param      d_outputBboxMin  Reference to final output bounds
     * @param      d_outputBboxMax  Reference to final output bounds
     */
    __device__ __forceinline__ AgentFindBoundsFirstPass(
        TempStorage         &tempStorage, 
        SampleT const *     d_samples, 
        SampleT*            (&d_outputBboxMin)[DIM],
        SampleT*            (&d_outputBboxMax)[DIM])
    :
        BaseT(tempStorage, d_samples, d_outputBboxMin, d_outputBboxMax)
    {
    }

    /**
     * Consume input data partitioned into tiles
     *
     * @param[in]  numPoints  The number of multi-dimensional points to consume
     * @param[in]  (stride)   The number of samples between starts of consecutive points's
     *                        coordinates. Unused in this specialization.
     */
    __device__ __forceinline__ void consumeTiles(
        OffsetT     numPoints,
        OffsetT)
    {
        OffsetT numTiles = (numPoints + TILE_POINTS - 1) / TILE_POINTS;

        for (int t = blockIdx.x; t < numTiles; t += gridDim.x)
        {
            OffsetT tileOffset = t * TILE_POINTS;

            if (tileOffset + TILE_POINTS > numPoints)
            {
                // Consume partial tile
                consumePartialTile(tileOffset, numPoints - tileOffset);
            }
            else
            {
                // Consume full tile
                consumeFullTile(tileOffset);
            }
        }
        // reduce block-aggregated results
        BlockReduceT blockReductor(this->tempStorage.sampleReduce);

        #pragma unroll
        for (int d = 0; d < DIM; ++d)
        {
            __syncthreads();
            this->privateMinBounds[d] = blockReductor.Reduce(
                this->privateMinBounds[d], cub::Min());
            __syncthreads();
            this->privateMaxBounds[d] = blockReductor.Reduce(
                this->privateMaxBounds[d], cub::Max());
        }
        this->storePrivateThreadOutput();
    }
};


/*
 * Specialization for COL_MAJOR order data layout 
 */
template <
    typename            AgentFindBoundsPolicyT,
    int                 DIM,
    typename            SampleT,
    typename            OffsetT>        ///< Signed integer type for global offsets
class AgentFindBoundsFirstPass<AgentFindBoundsPolicyT, COL_MAJOR, DIM, SampleT, OffsetT> 
    : public AgentFindBoundsBase<AgentFindBoundsPolicyT, COL_MAJOR, DIM, SampleT, OffsetT>
{

    //---------------------------------------------------------------------
    // Typedefs and constants
    //---------------------------------------------------------------------
    typedef AgentFindBoundsBase<AgentFindBoundsPolicyT, COL_MAJOR, DIM, SampleT, OffsetT> BaseT;
public:
    typedef typename BaseT::TempStorage               TempStorage;
private:
    typedef typename BaseT::BlockTileLoadT            BlockTileLoadT;
    typedef typename BaseT::BlockReduceT              BlockReduceT;

    /// Constants
    enum
    {
        BLOCK_THREADS        = AgentFindBoundsPolicyT::BLOCK_THREADS,
        POINTS_PER_THREAD    = AgentFindBoundsPolicyT::POINTS_PER_THREAD,
        TILE_POINTS          = POINTS_PER_THREAD * BLOCK_THREADS
    };

    //---------------------------------------------------------------------
    // Search tile bounds
    //---------------------------------------------------------------------

    // Find bounds. Specialized for full tile.
    __device__ __forceinline__ void findBounds(
        SampleT    (&samples)[DIM][POINTS_PER_THREAD])
    {
        #pragma unroll
        for (int d = 0; d < DIM; ++d)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                this->privateMinBounds[d] = min(this->privateMinBounds[d], samples[d][p]);
                this->privateMaxBounds[d] = max(this->privateMaxBounds[d], samples[d][p]);
            }
        }
    }

    // Find bounds. Specialized for partial tile
    __device__ __forceinline__ void findBounds(
        SampleT     (&samples)[DIM][POINTS_PER_THREAD],
        int         validPoints)
    {
        #pragma unroll
        for (int p = 0; p < POINTS_PER_THREAD; ++p)
        {
            if ((threadIdx.x + p * BLOCK_THREADS) < validPoints)
            {
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    this->privateMinBounds[d] = min(this->privateMinBounds[d], samples[d][p]);
                    this->privateMaxBounds[d] = max(this->privateMaxBounds[d], samples[d][p]);
                }
            }
        }
    }

    //---------------------------------------------------------------------
    // Tile processing
    //---------------------------------------------------------------------
    // Consume a full tile of data samples
    __device__ __forceinline__ void consumeTile(
        OffsetT     blockOffset,
        OffsetT     stride)
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::COL_MAJOR> samples;
        BlockTileLoadT::loadTile2ColM(this->d_inputSamples + blockOffset, samples.data, stride);
        
        findBounds(samples.data);
    }

    // Consume a partial tile of data samples
    __device__ __forceinline__ void consumeTile(
        OffsetT     blockOffset,
        int         validPoints,
        OffsetT     stride)
    {
        typename BlockTileLoadT::ThreadPrivatePoints<rd::COL_MAJOR> samples;
        BlockTileLoadT::loadTile2ColM(this->d_inputSamples + blockOffset, validPoints, 
            samples.data, stride);
        findBounds(samples.data, validPoints);
    }

public:

    /**
     * Constructor
     *
     * @param      tempStorage      Reference to thread block's tempStorage
     * @param      d_samples        Input data to reduce
     * @param      d_outputBboxMin  Reference to final output bounds (min)
     * @param      d_outputBboxMax  Reference to final output bounds (max)
     */
    __device__ __forceinline__ AgentFindBoundsFirstPass(
        TempStorage         &tempStorage,
        SampleT const *     d_samples,
        SampleT*            (&d_outputBboxMin)[DIM],
        SampleT*            (&d_outputBboxMax)[DIM])
    :
        BaseT(tempStorage, d_samples, d_outputBboxMin, d_outputBboxMax)
    {
    }

    
    /**
     * Consume input data partitioned into tiles
     *
     * @param[in]  numPoints  The number of multi-dimensional points to consume
     * @param[in]  stride     The number of samples between starts of consecutive points's
     *                        coordinates.
     */
    __device__ __forceinline__ void consumeTiles(
        OffsetT     numPoints,
        OffsetT     stride)
    {

        OffsetT numTiles = (numPoints + TILE_POINTS - 1) / TILE_POINTS;

        for (int t = blockIdx.x; t < numTiles; t += gridDim.x)
        {
            OffsetT tileOffset = t * TILE_POINTS;

            if (tileOffset + TILE_POINTS > numPoints)
            {
                // Consume partial tile
                consumeTile(tileOffset, numPoints - tileOffset, stride);
            }
            else
            {
                // Consume full tile
                consumeTile(tileOffset, stride);
            }
        }

        // reduce block-aggregated results
        BlockReduceT blockReductor(this->tempStorage.sampleReduce);

        #pragma unroll
        for (int d = 0; d < DIM; ++d)
        {
            __syncthreads();
            this->privateMinBounds[d] = blockReductor.Reduce(this->privateMinBounds[d], cub::Min());
            __syncthreads();
            this->privateMaxBounds[d] = blockReductor.Reduce(this->privateMaxBounds[d], cub::Max());
        }
        this->storePrivateThreadOutput();
    }
};


/******************************************************************************
 *
 * Thread block abstractions for second Pass Find Bounds Kernel
 * 
 ******************************************************************************/

/**
 * @brief      Defines second phase of two-pass find bounds algorithm.
 *
 *             In this pass there are as many blocks running as point's dimensions times two. 
 *             Evenly indexed blocks perform max reduction of block-private results from algorithm 
 *             first pass for subsequent dimensions. While odd indexed thread blocks perform min 
 *             reduction.
 *
 * @tparam     AgentFindBoundsPolicyT  Algorithm tuning parameters
 * @tparam     DIM                     Point dimension
 * @tparam     SampleT                 Point coordinate's data type
 * @tparam     OffsetT                 Integer type for offsets
 */
template <
    typename            AgentFindBoundsPolicyT,
    int                 DIM,
    typename            SampleT,
    typename            OffsetT>
class AgentFindBoundsSecondPass
{
protected:
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS        = AgentFindBoundsPolicyT::BLOCK_THREADS,
        POINTS_PER_THREAD    = AgentFindBoundsPolicyT::POINTS_PER_THREAD,
        TILE_POINTS          = POINTS_PER_THREAD * BLOCK_THREADS
    };

    typedef BlockTileLoadPolicy<
            BLOCK_THREADS, 
            POINTS_PER_THREAD, 
            AgentFindBoundsPolicyT::LOAD_MODIFIER>
        BlockTileLoadPolicyT;

    typedef BlockTileLoad<
            BlockTileLoadPolicyT,
            1,
            ROW_MAJOR, 
            AgentFindBoundsPolicyT::IO_BACKEND,
            SampleT, 
            OffsetT> 
        BlockTileLoadT;

    /// Parameterized BlockReduce type block-wide results reduction
    typedef cub::BlockReduce<
            SampleT,
            BLOCK_THREADS,
            AgentFindBoundsPolicyT::REDUCE_ALGORITHM>
        BlockReduceT;

    /// Shared memory type required by this thread block
    struct _TempStorage
    {
        /*
         * XXX: dla wersji n-wymiarowej należałoby to jakoś ograniczyć, albo najlepiej zrobić
         * tak jak w CUB, czyli cześć granic mogę trzymać w smem (co się zmieści), a resztę muszę w
         * gmem. Coś jak:
         * CounterT histograms[NUM_ACTIVE_CHANNELS][PRIVATIZED_SMEM_BINS + 1];     // Smem needed for block-privatized smem histogram (with 1 word of padding)
         */

        union
        {
            typename BlockReduceT::TempStorage sampleReduce;       // Smem needed for reducing thread's private results 
        };
    };

public:
    /// Temporary storage type (unionable)
    struct TempStorage : cub::Uninitialized<_TempStorage> {};

protected:
    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    /// Reference to tempStorage
    _TempStorage &tempStorage;

    /// Native pointer for input samples 
    SampleT const * d_inputSamples;

    // Per thread private bounds
    SampleT privateBound;

    /// pointers to final bounds (gmem)
    SampleT* d_outputBound;

    //---------------------------------------------------------------------
    // Update final output bounds
    //---------------------------------------------------------------------

    // Update final output bounds from thread private bounds.
    __device__ __forceinline__ void storePrivateThreadOutput()
    {
        // Apply thread private bounds to output bounds
        if (threadIdx.x == 0)
        {
            *d_outputBound = privateBound;
        }
    }

    //---------------------------------------------------------------------
    // Debug utility
    //---------------------------------------------------------------------

    __device__ __forceinline__ 
    void printPrivateBounds(int tid = -1)
    {
        if (tid >= 0)
        {
            if (threadIdx.x == tid)
            {
                printf("[bid: %3d][tid: %3d], value: %8.6e \n", blockIdx.x, threadIdx.x, privateBound);
            }
        }
        else
        {
            printf("[bid: %3d][tid: %3d], value: %8.6e \n", blockIdx.x, threadIdx.x, privateBound);
        }
    }

    //---------------------------------------------------------------------
    // Search tile bounds
    //---------------------------------------------------------------------

    // Find bounds. Specialized for full tile.
    __device__ __forceinline__ void findBounds(
        SampleT     (&samples)[POINTS_PER_THREAD])
    {
        // even indexed blocks perform max reduction
        if ((blockIdx.x & 1) == 0)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                privateBound = max(privateBound, samples[p]);
            }
        }
        else
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                privateBound = min(privateBound, samples[p]);
            }
        }
    }

    // Find bounds. Specialized for partial tile
    __device__ __forceinline__ void findBounds(
        SampleT     (&samples)[POINTS_PER_THREAD],
        int         validPoints)
    {
        // even indexed blocks perform max reduction
        if ((blockIdx.x & 1) == 0)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if ((threadIdx.x + p * BLOCK_THREADS) < validPoints)
                {
                    privateBound = max(privateBound, samples[p]);
                }
            }
        }
        else
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if ((threadIdx.x + p * BLOCK_THREADS) < validPoints)
                {
                    privateBound = min(privateBound, samples[p]);
                }
            }
        }
    }

    //---------------------------------------------------------------------
    // Tile processing
    //---------------------------------------------------------------------

    // Consume a full tile of data samples
    __device__ __forceinline__ void consumeTile(
        OffsetT     blockOffset)
    {
        typedef SampleT AliasedSamples[POINTS_PER_THREAD];
        typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> samples;
        BlockTileLoadT::loadTile2RowM(d_inputSamples + blockOffset, samples.data, 1);
        
        findBounds(reinterpret_cast<AliasedSamples&>(samples.data));
    }

    // Consume a partial tile of data samples
    __device__ __forceinline__ void consumeTile(
        OffsetT     blockOffset,
        int         validPoints)
    {
        typedef SampleT AliasedSamples[POINTS_PER_THREAD];
        typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> samples;
        BlockTileLoadT::loadTile2RowM(d_inputSamples + blockOffset, validPoints, samples.data, 1);
        
        findBounds(reinterpret_cast<AliasedSamples&>(samples.data), validPoints);
    }

    //------------------------------------------------------------------------
    //  INTERFACE
    //------------------------------------------------------------------------

public:

    /**
     * Constructor
     *
     * @param      tempStorage      Reference to thread block's tempStorage
     * @param      d_inMin          Reference to input data to reduce. It's an array of pointers to
     *                              memory regions with intermediate thread block's results.
     * @param      d_inMax          Reference to input data to reduce. It's an array of pointers to
     *                              memory regions with intermediate thread block's results.
     * @param      d_outputBboxMin  Pointer to final output bounds (min)
     * @param      d_outputBboxMax  Pointer to final output bounds (max)
     */
    __device__ __forceinline__ AgentFindBoundsSecondPass(
        TempStorage &                               tempStorage,
        cub::ArrayWrapper<SampleT *, DIM> const &   d_inMin,
        cub::ArrayWrapper<SampleT *, DIM> const &   d_inMax,
        SampleT *                                   d_outputBboxMin,
        SampleT *                                   d_outputBboxMax)
    :
        tempStorage(tempStorage.Alias())
    {       
        // even indexed blocks perform max reduction
        if ((blockIdx.x & 1) == 0)
        {
            privateBound = cub::NumericTraits<SampleT>::Lowest();
            d_inputSamples = d_inMax.array[blockIdx.x >> 1];
            d_outputBound = d_outputBboxMax + (blockIdx.x >> 1);
        }
        else
        {
            privateBound = cub::NumericTraits<SampleT>::Max();
            d_inputSamples = d_inMin.array[(blockIdx.x - 1) >> 1];
            d_outputBound = d_outputBboxMin + ((blockIdx.x - 1) >> 1);
        }
    }

    /**
     * @brief Consume input data partitioned into tiles.
     *
     * @param[in]  numPoints  The number of points to consume
     */
    __device__ __forceinline__ void consumeTiles(
        OffsetT     numPoints)
    {

        OffsetT numTiles = (numPoints + TILE_POINTS - 1) / TILE_POINTS;

        for (int t = 0; t < numTiles; ++t)
        {
            OffsetT tileOffset = t * TILE_POINTS;

            if (tileOffset + TILE_POINTS > numPoints)
            {
                // Consume partial tile
                consumeTile(tileOffset, numPoints - tileOffset);
            }
            else
            {
                // Consume full tile
                consumeTile(tileOffset);
            }
        }

        __syncthreads();

        // reduce block-aggregated results
        if ((blockIdx.x & 1) == 0)
        {
            privateBound = BlockReduceT(tempStorage.sampleReduce).Reduce(privateBound, cub::Max());
        }
        else
        {
            privateBound = BlockReduceT(tempStorage.sampleReduce).Reduce(privateBound, cub::Min());
        }    
        storePrivateThreadOutput();
    }
};




} // end namespace detail
} // end namespace gpu
} // end namespace rd