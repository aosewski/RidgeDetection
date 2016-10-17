/**
 * @file block_tile_load_store5.cuh
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

#include "cub/util_debug.cuh"
#include "cub/util_type.cuh"
#include "cub/util_ptx.cuh"
#include "cub/thread/thread_load.cuh"
#include "cub/thread/thread_store.cuh"

#include "trove/aos.h"

namespace rd
{
namespace gpu
{

enum BlockTileIOBackend
{
    IO_BACKEND_CUB,            /// Use CUB library routines to load and store data
    IO_BACKEND_TROVE           /// Use TROVE library routines to load and store data
};

//------------------------------------------------------------
//  TILE IO POLICY
//------------------------------------------------------------

template <
    int                         _BLOCK_THREADS,
    int                         _POINTS_PER_THREAD,
    cub::CacheLoadModifier      _LOAD_MODIFIER>
struct BlockTileLoadPolicy
{
    enum 
    {
        BLOCK_THREADS           = _BLOCK_THREADS,
        POINTS_PER_THREAD       = _POINTS_PER_THREAD,
    };
        
    static const cub::CacheLoadModifier     LOAD_MODIFIER           = _LOAD_MODIFIER;
};

template <
    int                         _BLOCK_THREADS,
    int                         _POINTS_PER_THREAD,
    cub::CacheStoreModifier     _STORE_MODIFIER>
struct BlockTileStorePolicy
{
    enum 
    {
        BLOCK_THREADS           = _BLOCK_THREADS,
        POINTS_PER_THREAD       = _POINTS_PER_THREAD,
    };

    static const cub::CacheStoreModifier    STORE_MODIFIER          = _STORE_MODIFIER;
};

//------------------------------------------------------------
//  TILE LOAD CLASS
//------------------------------------------------------------

/**
 * @brief      The BlockTileLoad class provides collective methods for data loading.
 *
 * @tparam     BlockTileLoadPolicyT  Policy for loading data.
 * @tparam     DIM                   Points dimension
 * @tparam     MEM_LAYOUT            Input memory layout.
 * @tparam     IO_BACKEND            The underlaying backend to use {CUB or trove}
 * @tparam     OffsetT               Block offset type
 * @tparam     SampleT               Data type of point single coordinate.
 */

// Specialization for data ROW_MAJOR order
template <
    typename                BlockTileLoadPolicyT,
    int                     DIM,
    rd::DataMemoryLayout    MEM_LAYOUT,
    BlockTileIOBackend      IO_BACKEND,
    typename                SampleT,
    typename                OffsetT>
class BlockTileLoad
{
    //---------------------------------------------------------------------
    // constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS           = BlockTileLoadPolicyT::BLOCK_THREADS,
        POINTS_PER_THREAD       = BlockTileLoadPolicyT::POINTS_PER_THREAD,
    };


    //---------------------------------------------------------------------
    // Internal helper loader class
    //---------------------------------------------------------------------

    template <BlockTileIOBackend _BACKEND, int DUMMY>
    struct LoadInternal;

    /**************************************************************************
     *  IO_BACKEND_CUB
     **************************************************************************/
    template <int DUMMY>
    struct LoadInternal<IO_BACKEND_CUB, DUMMY>
    {
        //---------------------------------------------------------------------
        // constants and type definitions
        //---------------------------------------------------------------------

        /// The point type of SampleT
        typedef cub::CubVector<SampleT, DIM>    PointT;

        // Biggest memory access word (16B)
        typedef ulonglong2 DeviceWord;

        /// Constants
        enum
        {
            WORD_ALIGN_BYTES     = cub::AlignBytes<DeviceWord>::ALIGN_BYTES,
            POINT_ALIGN_BYTES    = cub::AlignBytes<PointT>::ALIGN_BYTES,
            
            IS_MULTIPLE          = (sizeof(DeviceWord) % sizeof(PointT) == 0) && (WORD_ALIGN_BYTES % POINT_ALIGN_BYTES == 0),
            WORD_POINTS          = sizeof(DeviceWord) / sizeof(PointT),
            VECTOR_SIZE          = (IS_MULTIPLE == 1 && WORD_POINTS > 1) ? WORD_POINTS : 1,
            
            TRY_VECTORIZE        = ((VECTOR_SIZE > 1) && (POINTS_PER_THREAD > VECTOR_SIZE) && (POINTS_PER_THREAD % VECTOR_SIZE == 0)) ? 1 : 0
        };

        /// Cache load modifier for reading input elements
        static const cub::CacheLoadModifier LOAD_MODIFIER = BlockTileLoadPolicyT::LOAD_MODIFIER;

        //---------------------------------------------------------------------
        // Utility
        //---------------------------------------------------------------------

        // Whether or not the input is aligned with the DeviceWord
        template <typename T>
        static __device__ __forceinline__ bool isAligned(
            T const *    d_in)
        {
            return (size_t(d_in) & (sizeof(DeviceWord) - 1)) == 0;
        }

        //---------------------------------------------------------------------
        // Tile loading interface
        //---------------------------------------------------------------------

        // Load full tile into ROW_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void loadTile2Row(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            cub::Int2Type<true>             canVectorize)
        {
            enum 
            {
                /// to suppress errors when POINTS_PER_THREAD equals 1
                N_VECTORS = POINTS_PER_THREAD / VECTOR_SIZE,
                VECTORS_PER_THREAD = (N_VECTORS > 0) ? N_VECTORS : 1        
            };

            typedef DeviceWord AliasedPoints[VECTORS_PER_THREAD];

           // if (blockIdx.x == 0 && threadIdx.x == 0)
           // {
           //     _CubLog(">>>>>>>>\n WORD_ALIGN_BYTES: %d\n POINT_ALIGN_BYTES: %d\n IS_MULTIPLE: %d\n WORD_POINTS: %d\n VECTOR_SIZE: %d\n TRY_VECTORIZE: %d\n sizeof(DeviceWord): %ld\n sizeof(PointT): %ld \n >>>>>>>>>> \n",
           //             WORD_ALIGN_BYTES, POINT_ALIGN_BYTES, IS_MULTIPLE, WORD_POINTS, VECTOR_SIZE, TRY_VECTORIZE, sizeof(DeviceWord), sizeof(PointT));
           // }

            AliasedPoints &aliasedPoints = reinterpret_cast<AliasedPoints&>(samples);
            DeviceWord const *vecPtr = reinterpret_cast<DeviceWord const *>(d_inputSamples);

           // if (threadIdx.x == 0 && cub::__isLocal(aliasedPoints))
           // {
           //     _CubLog(">>>> aliasedPoints is local! %p\n", aliasedPoints);
           // }
            #pragma unroll
            for (int p = 0; p < VECTORS_PER_THREAD; ++p)    
            {
               // if ((threadIdx.x & 31) == 0 && (size_t(aliasedPoints + p) & (WORD_ALIGN_BYTES-1)) != 0)
               // {
               //     _CubLog("%p is misaligned! %ld\n", aliasedPoints + p, size_t(aliasedPoints + p) & (WORD_ALIGN_BYTES-1));
               // }
                aliasedPoints[p] = cub::ThreadLoad<LOAD_MODIFIER>(vecPtr + p * BLOCK_THREADS + threadIdx.x);
            }    
        }

        // Load full tile into COL_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void loadTile2Col(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            cub::Int2Type<true>             canVectorize)
        {
            enum 
            {
                /// to suppress errors when POINTS_PER_THREAD equals 1
                N_VECTORS = POINTS_PER_THREAD / VECTOR_SIZE,
                VECTORS_PER_THREAD = (N_VECTORS > 0) ? N_VECTORS : 1 
            };

            typedef SampleT AliasedDeviceWord[WORD_POINTS][DIM];
            DeviceWord const *vecPtr = reinterpret_cast<DeviceWord const *>(d_inputSamples);

            #pragma unroll
            for (int p = 0; p < VECTORS_PER_THREAD; ++p)    
            {
                DeviceWord auxWord = cub::ThreadLoad<LOAD_MODIFIER>(vecPtr + p * BLOCK_THREADS + threadIdx.x);
                AliasedDeviceWord &aliasedWord = reinterpret_cast<AliasedDeviceWord&>(auxWord);
                #pragma unroll
                for (int k = 0; k < WORD_POINTS; ++k)
                {
                    for (int d = 0; d < DIM; ++d)
                    {
                        samples[d][p * WORD_POINTS + k] = aliasedWord[k][d];
                    }
                }
            }    
        }

        // Load full tile into ROW_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void loadTile2Row(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            cub::Int2Type<false>            canVectorize)
        {
            typedef PointT AliasedPoints[POINTS_PER_THREAD];

            AliasedPoints &points = reinterpret_cast<AliasedPoints&>(samples);
            PointT const *d_wrappedPoints = reinterpret_cast<PointT const *>(d_inputSamples + threadIdx.x * DIM);
           // if (blockIdx.x == 0 && threadIdx.x == 0)
           // {
           //     _CubLog(">>>>>>>>\n WORD_ALIGN_BYTES: %d\n POINT_ALIGN_BYTES: %d\n IS_MULTIPLE: %d\n WORD_POINTS: %d\n VECTOR_SIZE: %d\n TRY_VECTORIZE: %d\n sizeof(DeviceWord): %ld\n sizeof(PointT): %ld \n >>>>>>>>>> \n",
           //             WORD_ALIGN_BYTES, POINT_ALIGN_BYTES, IS_MULTIPLE, WORD_POINTS, VECTOR_SIZE, TRY_VECTORIZE, sizeof(DeviceWord), sizeof(PointT));
           // }
           // if ((threadIdx.x == 0) && cub::__isLocal(points))
           // {
           //     _CubLog(">>>> points is local! %p\n", points);
           // }

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)    
            {

               // if ((threadIdx.x & 31) == 0 && (size_t(points + p) & (POINT_ALIGN_BYTES-1)) != 0)
               // {
               //     _CubLog("%p is misaligned! %ld\n", points + p, size_t(points + p) & (POINT_ALIGN_BYTES-1));
               // }
                // _CubLog("points[%d] address: %p, load addres: %p\n", p, &(points[p]), d_wrappedPoints + p * BLOCK_THREADS);
                points[p] = cub::ThreadLoad<LOAD_MODIFIER>(d_wrappedPoints + p * BLOCK_THREADS);
            }  
        }

        // Load full tile into COL_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void loadTile2Col(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            cub::Int2Type<false>            canVectorize)
        {
            PointT const *d_wrappedPoints = reinterpret_cast<PointT const *>(d_inputSamples + threadIdx.x * DIM);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)    
            {
                PointT auxPoint = cub::ThreadLoad<LOAD_MODIFIER>(d_wrappedPoints + p * BLOCK_THREADS);

                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    samples[d][p] = ((SampleT *)(&auxPoint))[d];
                }
            }  
        }

        // Load partial tile into ROW_MAJOR ordered thread-private table.
        static __device__ __forceinline__ void loadTile2Row(
            SampleT const *     d_inputSamples,
            int                 validPoints,
            SampleT             (&samples)[POINTS_PER_THREAD][DIM])
        {
            typedef PointT AliasedPoints[POINTS_PER_THREAD];

            AliasedPoints &aliasedPoints = reinterpret_cast<AliasedPoints&>(samples);
            PointT const * d_wrappedPoints = reinterpret_cast<PointT const *>(d_inputSamples);

           // if ((threadIdx.x == 0) && cub::__isLocal(aliasedPoints))
           // {
           //     _CubLog(">>>> aliasedPoints is local! %p\n", aliasedPoints);
           // }
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
               // if ((threadIdx.x & 31) == 0 && (size_t(aliasedPoints + p) & (POINT_ALIGN_BYTES-1)) != 0)
               // {
               //     _CubLog("%p is misaligned! %ld\n", aliasedPoints + p, size_t(aliasedPoints + p) & (POINT_ALIGN_BYTES-1));
               // }
                // _CubLog("aliasedPoints[%d] address: %p, load addres: %p\n", p, &(aliasedPoints[p]), d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x);

                if (BLOCK_THREADS <= validPoints)
                {
                    aliasedPoints[p] = cub::ThreadLoad<LOAD_MODIFIER>(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x);
                }
                else if (threadIdx.x < validPoints)
                {
                    aliasedPoints[p] = cub::ThreadLoad<LOAD_MODIFIER>(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x);
                }
                validPoints -= BLOCK_THREADS;
                // stop loading if we've loaded all valid points
                if (validPoints <= 0)
                    break;
            }
        } 

        // Load partial tile into COL_MAJOR ordered thread-private table.
        static __device__ __forceinline__ void loadTile2Col(
            SampleT const *     d_inputSamples,
            int                 validPoints,
            SampleT             (&samples)[DIM][POINTS_PER_THREAD])
        {
            PointT const * d_wrappedPoints = reinterpret_cast<PointT const *>(d_inputSamples);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                PointT auxPoint;

                if (BLOCK_THREADS <= validPoints)
                {
                    auxPoint = cub::ThreadLoad<LOAD_MODIFIER>(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x);
                }
                else if (threadIdx.x < validPoints)
                {
                    auxPoint = cub::ThreadLoad<LOAD_MODIFIER>(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x);
                }

                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    samples[d][p] = ((SampleT *)(&auxPoint))[d];
                }

                validPoints -= BLOCK_THREADS;
                // stop loading if we've loaded all valid points
                if (validPoints <= 0)
                    break;
            }    
        }   
    };

    /**************************************************************************
     *  IO_BACKEND_TROVE
     **************************************************************************/
    template <int DUMMY>
    struct LoadInternal<IO_BACKEND_TROVE, DUMMY>
    {
        //---------------------------------------------------------------------
        // constants and type definitions
        //---------------------------------------------------------------------

        /// The point type of SampleT
        typedef cub::CubVector<SampleT, DIM>    PointT;

        /// Constants
        enum
        {
            /*
             * Trove decides whether or not to vectorize loads. It does vectorization when:
             * -- type size is multiple of 4B
             * -- and when sizeof(T) / sizeof(int/int2/int4) is in range (1, 64)
             */
            TRY_VECTORIZE        = 1
        };

        /// Cache load modifier for reading input elements
        static const cub::CacheLoadModifier LOAD_MODIFIER = BlockTileLoadPolicyT::LOAD_MODIFIER;

        //---------------------------------------------------------------------
        // Utility
        //---------------------------------------------------------------------

        // Whether or not the input is aligned with the DeviceWord
        template <typename T>
        static __device__ __forceinline__ bool isAligned(
            T const *    d_in)
        {
            return (size_t(d_in) & (sizeof(PointT) - 1)) == 0;
        }

        //---------------------------------------------------------------------
        // Tile loading interface
        //---------------------------------------------------------------------

        // Load full tile into ROW_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void loadFullTile2Row(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM])
        {
            typedef PointT AliasedPoints[POINTS_PER_THREAD];
            AliasedPoints &points = reinterpret_cast<AliasedPoints&>(samples);
            PointT const *d_wrappedPoints = reinterpret_cast<PointT const *>(d_inputSamples + threadIdx.x * DIM);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)    
            {
                points[p] = trove::load_warp_contiguous(d_wrappedPoints + p * BLOCK_THREADS);
            }
        }

        // Load full tile into COL_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void loadFullTile2Col(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD])
        {
            PointT const *d_wrappedPoints = reinterpret_cast<PointT const *>(d_inputSamples + threadIdx.x * DIM);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)    
            {
                PointT auxPoint = trove::load_warp_contiguous(d_wrappedPoints + p * BLOCK_THREADS);
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    samples[d][p] = ((SampleT *)(&auxPoint))[d];
                }
            } 
        }

        // Load full tile into ROW_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void loadTile2Row(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            cub::Int2Type<true>             canVectorize)
        {
             loadFullTile2Row(d_inputSamples, samples);
        }

        // Load full tile into COL_MAJOR ordered thread-private table(vectorized).
        static __device__ __forceinline__ void loadTile2Col(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            cub::Int2Type<true>             canVectorize)
        {
             loadFullTile2Col(d_inputSamples, samples);
        }

        // Load full tile into ROW_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void loadTile2Row(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            cub::Int2Type<false>             canVectorize)
        {
             loadFullTile2Row(d_inputSamples, samples);
        }

        // Load full tile into COL_MAJOR ordered thread-private table(non-vectorized).
        static __device__ __forceinline__ void loadTile2Col(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            cub::Int2Type<false>             canVectorize)
        {
             loadFullTile2Col(d_inputSamples, samples);
        }

        // Load partial tile into ROW_MAJOR ordered thread-private table
        static __device__ __forceinline__ void loadTile2Row(
            SampleT const *     d_inputSamples,
            int                 validPoints,
            SampleT             (&samples)[POINTS_PER_THREAD][DIM])
        {
            typedef PointT AliasedPoints[POINTS_PER_THREAD];

            AliasedPoints &aliasedPoints = reinterpret_cast<AliasedPoints&>(samples);
            PointT const * d_wrappedPoints = reinterpret_cast<PointT const *>(d_inputSamples);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if (BLOCK_THREADS <= validPoints)
                {
                    aliasedPoints[p] = trove::load_warp_contiguous(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x);
                }
                else if (threadIdx.x < validPoints)
                {
                    aliasedPoints[p] = *(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x);
                }
                validPoints -= BLOCK_THREADS;
                // stop loading if we've loaded all valid points
                if (validPoints <= 0)
                    break;
            }
        } 

        // Load partial tile into COL_MAJOR ordered thread-private table
        static __device__ __forceinline__ void loadTile2Col(
            SampleT const *     d_inputSamples,
            int                 validPoints,
            SampleT             (&samples)[DIM][POINTS_PER_THREAD])
        {
            PointT const * d_wrappedPoints = reinterpret_cast<PointT const *>(d_inputSamples);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                PointT auxPoint;

                if (BLOCK_THREADS <= validPoints)
                {
                    auxPoint = trove::load_warp_contiguous(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x);
                }
                else if (threadIdx.x < validPoints)
                {
                    auxPoint = *(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x);
                }

                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    samples[d][p] = reinterpret_cast<SampleT *>(&auxPoint)[d];
                }

                validPoints -= BLOCK_THREADS;
                // stop loading if we've loaded all valid points
                if (validPoints <= 0)
                    break;
            }    
        }   
    };

    //---------------------------------------------------------------------
    // Type definitions
    //---------------------------------------------------------------------

    /// Internal load implementation to use
    typedef LoadInternal<IO_BACKEND, 0> InternalLoad;

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------
public:
    /**
     * Constructor
     */
    __device__ __forceinline__ BlockTileLoad()
    {
    }

    // Load full tile into ROW_MAJOR ordered thread-private table
    static __device__ __forceinline__ void loadTile2Row(
        SampleT const *     d_inputSamples,
        SampleT             (&samples)[POINTS_PER_THREAD][DIM],
        OffsetT             )
    {
        if (InternalLoad::isAligned(d_inputSamples) && InternalLoad::TRY_VECTORIZE)
        {
            InternalLoad::loadTile2Row(d_inputSamples, samples, cub::Int2Type<true>());   
        }
        else
        {
            InternalLoad::loadTile2Row(d_inputSamples, samples, cub::Int2Type<false>());   
        }
    }

    // Load partial tile into ROW_MAJOR ordered thread-private table
    static __device__ __forceinline__ void loadTile2Row(
        SampleT const *     d_inputSamples,
        int                 validPoints,
        SampleT             (&samples)[POINTS_PER_THREAD][DIM],
        OffsetT             )
    {
        InternalLoad::loadTile2Row(d_inputSamples, validPoints, samples);   
    }

    // Load full tile into COL_MAJOR ordered thread-private table
    static __device__ __forceinline__ void loadTile2Col(
        SampleT const *     d_inputSamples,
        SampleT             (&samples)[DIM][POINTS_PER_THREAD],
        OffsetT             )
    {
        if (InternalLoad::isAligned(d_inputSamples) && InternalLoad::TRY_VECTORIZE)
        {
            InternalLoad::loadTile2Col(d_inputSamples, samples, cub::Int2Type<true>());   
        }
        else
        {
            InternalLoad::loadTile2Col(d_inputSamples, samples, cub::Int2Type<false>());   
        }
    }

    // Load partial tile into COL_MAJOR ordered thread-private table

    static __device__ __forceinline__ void loadTile2Col(
        SampleT const *     d_inputSamples,
        int                 validPoints,
        SampleT             (&samples)[DIM][POINTS_PER_THREAD],
        OffsetT             )
    {
        InternalLoad::loadTile2Col(d_inputSamples, validPoints, samples);   
    }

};

// Specialization for data COL_MAJOR order
template <
    typename                BlockTileLoadPolicyT,
    int                     DIM,
    BlockTileIOBackend      IO_BACKEND,
    typename                SampleT,
    typename                OffsetT>
class BlockTileLoad<BlockTileLoadPolicyT, DIM, rd::COL_MAJOR, IO_BACKEND, SampleT, OffsetT>
{
    //---------------------------------------------------------------------
    // constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS           = BlockTileLoadPolicyT::BLOCK_THREADS,
        POINTS_PER_THREAD       = BlockTileLoadPolicyT::POINTS_PER_THREAD,
    };


    //---------------------------------------------------------------------
    // Internal helper loader class
    //---------------------------------------------------------------------

    template <BlockTileIOBackend _BACKEND, int DUMMY>
    struct LoadInternal;

    /**************************************************************************
     *  IO_BACKEND_CUB
     **************************************************************************/
    template <int DUMMY>
    struct LoadInternal<IO_BACKEND_CUB, DUMMY>
    {
        //---------------------------------------------------------------------
        // constants and type definitions
        //---------------------------------------------------------------------

        /// The point type of SampleT
        typedef cub::CubVector<SampleT, DIM>    PointT;

        /// Constants
        enum
        {
            ROW_SAMPLES_SIZE    = sizeof(SampleT) * POINTS_PER_THREAD,
            DEVICE_WORD_SIZE    = (ROW_SAMPLES_SIZE % 16 == 0) ? 16 
                                    : (ROW_SAMPLES_SIZE % 8 == 0) ? 8
                                        : 1,
            
            SAMPLE_ALIGN_BYTES  = cub::AlignBytes<SampleT>::ALIGN_BYTES,
            IS_MULTIPLE         = (DEVICE_WORD_SIZE % sizeof(SampleT) == 0) && (DEVICE_WORD_SIZE % SAMPLE_ALIGN_BYTES == 0),
            
            DEVICE_WORD_SAMPLES = DEVICE_WORD_SIZE / sizeof(SampleT),
            VECTOR_SIZE         = (IS_MULTIPLE == 1 && DEVICE_WORD_SAMPLES > 1) ? DEVICE_WORD_SAMPLES : 1,
            
            TRY_VECTORIZE       = (VECTOR_SIZE > 1) ? 1 : 0
        };

        // Biggest memory access word that ROW_SAMPLES_SIZE is a whole multiple of 
        typedef typename cub::If<DEVICE_WORD_SIZE == 16, 
            ulonglong2,
            typename cub::If<DEVICE_WORD_SIZE == 8, 
                unsigned long long,
                unsigned int>::Type>::Type      DeviceWord;


        /// Cache load modifier for reading input elements
        static const cub::CacheLoadModifier LOAD_MODIFIER = BlockTileLoadPolicyT::LOAD_MODIFIER;

        //---------------------------------------------------------------------
        // Utility
        //---------------------------------------------------------------------

        // Whether or not the input is aligned with the DeviceWord
        template <typename T>
        static __device__ __forceinline__ bool isAligned(
            T const *    d_in)
        {
            return (size_t(d_in) & (sizeof(DeviceWord) - 1)) == 0;
        }

        //---------------------------------------------------------------------
        // Tile loading interface
        //---------------------------------------------------------------------

        // Loads a full tile of samples into ROW_MAJOR ordered thread private table
        static __device__ __forceinline__ void loadFullTile2Row(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride)
        {
            d_inputSamples += threadIdx.x;

            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                #pragma unroll
                for (int p = 0; p < POINTS_PER_THREAD; ++p)    
                {
                    samples[p][d] = cub::ThreadLoad<LOAD_MODIFIER>(d_inputSamples + p * BLOCK_THREADS);
                } 
                d_inputSamples += stride;   
            }            
        }

        // // Loads a full tile of samples into ROW_MAJOR ordered thread private table. (vectorized)
        static __device__ __forceinline__ void loadTile2Row(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride,
            cub::Int2Type<true>             canVectorize)
        {
            loadFullTile2Row(d_inputSamples, samples, stride);
        }

        // Loads a full tile of samples into ROW_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void loadTile2Row(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride,
            cub::Int2Type<false>            canVectorize)
        {
            loadFullTile2Row(d_inputSamples, samples, stride);
        }
        
        // Loads a full tile of samples into COL_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void loadTile2Col(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            OffsetT                         stride,
            cub::Int2Type<true>             canVectorize)
        {
            enum 
            {
                VECTORS_PER_THREAD = POINTS_PER_THREAD / VECTOR_SIZE
            };

            typedef DeviceWord AliasedPoints[DIM][VECTORS_PER_THREAD];
            AliasedPoints &aliasedPoints = reinterpret_cast<AliasedPoints&>(samples);

            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                DeviceWord const *vecPtr = reinterpret_cast<DeviceWord const *>(
                    d_inputSamples + threadIdx.x * VECTOR_SIZE + d * stride);
                #pragma unroll
                for (int p = 0; p < VECTORS_PER_THREAD; ++p)    
                {
                    aliasedPoints[d][p] = cub::ThreadLoad<LOAD_MODIFIER>(vecPtr + p * BLOCK_THREADS);
                }  
                // can't advance vecPtr by stride here, because it would move it out of bounds of input samples!
            }  
        }

        // Loads a full tile of samples into COL_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void loadTile2Col(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            OffsetT                         stride,
            cub::Int2Type<false>            canVectorize)
        {
            d_inputSamples += threadIdx.x;

            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                #pragma unroll
                for (int p = 0; p < POINTS_PER_THREAD; ++p)    
                {
                    samples[d][p] = cub::ThreadLoad<LOAD_MODIFIER>(d_inputSamples + p * BLOCK_THREADS);
                }    
                d_inputSamples += stride;
            }  
        }

        // Loads a partial tile of samples into ROW_MAJOR ordered thread-private table
        static __device__ __forceinline__ void loadTile2Row(
            SampleT const *                 d_inputSamples,
            int                             validPoints,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if (BLOCK_THREADS <= validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        samples[p][d] = cub::ThreadLoad<LOAD_MODIFIER>(d_inputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x);
                    }
                }
                else if (threadIdx.x < validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        samples[p][d] = cub::ThreadLoad<LOAD_MODIFIER>(d_inputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x);
                    }
                }
                validPoints -= BLOCK_THREADS;
                // stop loading if we've loaded all valid points
                if (validPoints <= 0)
                    break;
            }   
        } 

        // Loads a partial tile of samples into COL_MAJOR ordered thread-private table
        static __device__ __forceinline__ void loadTile2Col(
            SampleT const *                 d_inputSamples,
            int                             validPoints,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            OffsetT                         stride)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if (BLOCK_THREADS <= validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        samples[d][p] = cub::ThreadLoad<LOAD_MODIFIER>(d_inputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x);
                    }
                }
                else if (threadIdx.x < validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        samples[d][p] = cub::ThreadLoad<LOAD_MODIFIER>(d_inputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x);
                    }
                }
                validPoints -= BLOCK_THREADS;
                // stop loading if we've loaded all valid points
                if (validPoints <= 0)
                    break;
            }    
        }   
    };

    /**************************************************************************
     *  IO_BACKEND_TROVE
     **************************************************************************/
    template <int DUMMY>
    struct LoadInternal<IO_BACKEND_TROVE, DUMMY>
    {
        //---------------------------------------------------------------------
        // constants and type definitions
        //---------------------------------------------------------------------

        /// Constants
        enum
        {
            ROW_SAMPLES_SIZE    = sizeof(SampleT) * POINTS_PER_THREAD,
            DEVICE_WORD_SIZE    = (ROW_SAMPLES_SIZE % 16 == 0) ? 16 
                                    : (ROW_SAMPLES_SIZE % 8 == 0) ? 8
                                        : 1,
            
            SAMPLE_ALIGN_BYTES  = cub::AlignBytes<SampleT>::ALIGN_BYTES,
            IS_MULTIPLE         = (DEVICE_WORD_SIZE % sizeof(SampleT) == 0) && (DEVICE_WORD_SIZE % SAMPLE_ALIGN_BYTES == 0),
            
            DEVICE_WORD_SAMPLES = DEVICE_WORD_SIZE / sizeof(SampleT),
            VECTOR_SIZE         = (IS_MULTIPLE == 1 && DEVICE_WORD_SAMPLES > 1) ? DEVICE_WORD_SAMPLES : 1,
            
            TRY_VECTORIZE       = (VECTOR_SIZE > 1) ? 1 : 0
        };

        // Biggest memory access word that ROW_SAMPLES_SIZE is a whole multiple of 
        typedef typename cub::If<DEVICE_WORD_SIZE == 16, 
            ulonglong2,
            typename cub::If<DEVICE_WORD_SIZE == 8, 
                unsigned long long,
                unsigned int>::Type>::Type      DeviceWord;

        //---------------------------------------------------------------------
        // Utility
        //---------------------------------------------------------------------

        // Whether or not the input is aligned with the DeviceWord
        template <typename T>
        static __device__ __forceinline__ bool isAligned(
            T const *    d_in)
        {
            return (size_t(d_in) & (sizeof(DeviceWord) - 1)) == 0;
        }

        //---------------------------------------------------------------------
        // Tile loading interface
        //---------------------------------------------------------------------

        // Loads a full tile of samples into ROW_MAJOR ordered thread private table
        static __device__ __forceinline__ void loadFullTile2Row(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride)
        {
            d_inputSamples += threadIdx.x;

            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                #pragma unroll
                for (int p = 0; p < POINTS_PER_THREAD; ++p)    
                {
                    samples[p][d] = trove::load_warp_contiguous(d_inputSamples + p * BLOCK_THREADS);
                } 
                d_inputSamples += stride;   
            }            
        }

        // // Loads a full tile of samples into ROW_MAJOR ordered thread private table. (vectorized)
        static __device__ __forceinline__ void loadTile2Row(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride,
            cub::Int2Type<true>             canVectorize)
        {
            loadFullTile2Row(d_inputSamples, samples, stride);
        }

        // Loads a full tile of samples into ROW_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void loadTile2Row(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride,
            cub::Int2Type<false>            canVectorize)
        {
            loadFullTile2Row(d_inputSamples, samples, stride);
        }
        
        // Load a full tile of samples into COL_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void loadTile2Col(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            OffsetT                         stride,
            cub::Int2Type<true>             canVectorize)
        {
            enum 
            {
                VECTORS_PER_THREAD = POINTS_PER_THREAD / VECTOR_SIZE
            };

            typedef DeviceWord AliasedPoints[DIM][VECTORS_PER_THREAD];
            AliasedPoints &aliasedPoints = reinterpret_cast<AliasedPoints&>(samples);

            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                DeviceWord const *vecPtr = reinterpret_cast<DeviceWord const *>(
                    d_inputSamples + threadIdx.x * VECTOR_SIZE + d * stride);
                #pragma unroll
                for (int p = 0; p < VECTORS_PER_THREAD; ++p)    
                {
                    aliasedPoints[d][p] = trove::load_warp_contiguous(vecPtr + p * BLOCK_THREADS);
                }    
                // can't advance vecPtr by stride here, because it would move it out of bounds of input samples!
            }  
        }

        // Load a full tile of samples into COL_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void loadTile2Col(
            SampleT const *                 d_inputSamples,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            OffsetT                         stride,
            cub::Int2Type<false>            canVectorize)
        {
            d_inputSamples += threadIdx.x;
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                #pragma unroll
                for (int p = 0; p < POINTS_PER_THREAD; ++p)    
                {
                    samples[d][p] = trove::load_warp_contiguous(d_inputSamples + p * BLOCK_THREADS);
                }    
                d_inputSamples += stride;
            }  
        }

        // Load a partial tile of samples into ROW_MAJOR ordered thread-private table
        static __device__ __forceinline__ void loadTile2Row(
            SampleT const *                 d_inputSamples,
            int                             validPoints,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if (BLOCK_THREADS <= validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        samples[p][d] = trove::load_warp_contiguous(d_inputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x);
                    }
                }
                else if (threadIdx.x < validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        samples[p][d] = *(d_inputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x);
                    }
                }
                validPoints -= BLOCK_THREADS;
                // stop loading if we've loaded all valid points
                if (validPoints <= 0)
                    break;
            }   
        } 

        // Load a partial tile of samples into COL_MAJOR ordered thread-private table
        static __device__ __forceinline__ void loadTile2Col(
            SampleT const *                 d_inputSamples,
            int                             validPoints,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            OffsetT                         stride)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if (BLOCK_THREADS <= validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        samples[d][p] = trove::load_warp_contiguous(d_inputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x);
                    }
                }
                else if (threadIdx.x < validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        samples[d][p] = *(d_inputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x);
                    }
                }
                validPoints -= BLOCK_THREADS;
                // stop loading if we've loaded all valid points
                if (validPoints <= 0)
                    break;
            }    
        }   
    };

    //---------------------------------------------------------------------
    // Type definitions
    //---------------------------------------------------------------------

    /// Internal load implementation to use
    typedef LoadInternal<IO_BACKEND, 0> InternalLoad;

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------
public:
    /**
     * Constructor
     */
    __device__ __forceinline__ BlockTileLoad()
    {
    }

    // Load full tile into ROW_MAJOR ordered thread-private table
    static __device__ __forceinline__ void loadTile2Row(
        SampleT const *     d_inputSamples,
        SampleT             (&samples)[POINTS_PER_THREAD][DIM],
        OffsetT             stride)     /// distance (expressed in number of samples) between point's consecutive coordinates
    {
        if (InternalLoad::isAligned(d_inputSamples) && InternalLoad::TRY_VECTORIZE)
        {
            InternalLoad::loadTile2Row(d_inputSamples, samples, stride, cub::Int2Type<true>());   
        }
        else
        {
            InternalLoad::loadTile2Row(d_inputSamples, samples, stride, cub::Int2Type<false>());   
        }
    }
    // Load full tile into COL_MAJOR ordered thread-private table
    static __device__ __forceinline__ void loadTile2Col(
        SampleT const *     d_inputSamples,
        SampleT             (&samples)[DIM][POINTS_PER_THREAD],
        OffsetT             stride)     /// distance (expressed in number of samples) between point's consecutive coordinates
    {
        if (InternalLoad::isAligned(d_inputSamples) && InternalLoad::TRY_VECTORIZE)
        {
            InternalLoad::loadTile2Col(d_inputSamples, samples, stride, cub::Int2Type<true>());   
        }
        else
        {
            InternalLoad::loadTile2Col(d_inputSamples, samples, stride, cub::Int2Type<false>());   
        }
    }

    // Load partial tile into ROW_MAJOR ordered thread-private table
    static __device__ __forceinline__ void loadTile2Row(
        SampleT const *     d_inputSamples,
        int                 validPoints,
        SampleT             (&samples)[POINTS_PER_THREAD][DIM],
        OffsetT             stride)     /// distance (expressed in number of samples) between point's consecutive coordinates
    {
        InternalLoad::loadTile2Row(d_inputSamples, validPoints, samples, stride);   
    }

    // Load partial tile int COL_MAJOR ordered thread-private table
    static __device__ __forceinline__ void loadTile2Col(
        SampleT const *     d_inputSamples,
        int                 validPoints,
        SampleT             (&samples)[DIM][POINTS_PER_THREAD],
        OffsetT             stride)     /// distance (expressed in number of samples) between point's consecutive coordinates
    {
        InternalLoad::loadTile2Col(d_inputSamples, validPoints, samples, stride);   
    }

};

//------------------------------------------------------------
//  TILE STORE CLASS
//------------------------------------------------------------

/**
 * @brief      The BlockTileStore class provides collective methods for data storing.
 *
 * @tparam     BlockTileStorePolicyT Policy for storing data.
 * @tparam     DIM                   Points dimension
 * @tparam     MEM_LAYOUT            Input memory layout.
 * @tparam     IO_BACKEND            The underlaying backend to use {CUB or trove}
 * @tparam     OffsetT               Block offset type
 * @tparam     SampleT               Data type of point single coordinate.
 */

// Specialization for data ROW_MAJOR order
template <
    typename                BlockTileStorePolicyT,
    int                     DIM,
    rd::DataMemoryLayout    MEM_LAYOUT,
    BlockTileIOBackend      IO_BACKEND,
    typename                SampleT,
    typename                OffsetT>
class BlockTileStore
{
    //---------------------------------------------------------------------
    // constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS           = BlockTileStorePolicyT::BLOCK_THREADS,
        POINTS_PER_THREAD       = BlockTileStorePolicyT::POINTS_PER_THREAD,
    };


    //---------------------------------------------------------------------
    // Internal helper storer class
    //---------------------------------------------------------------------

    template <BlockTileIOBackend _BACKEND, int DUMMY>
    struct StoreInternal;

    /**************************************************************************
     *  IO_BACKEND_CUB
     **************************************************************************/
    template <int DUMMY>
    struct StoreInternal<IO_BACKEND_CUB, DUMMY>
    {
        //---------------------------------------------------------------------
        // constants and type definitions
        //---------------------------------------------------------------------

        /// The point type of SampleT
        typedef cub::CubVector<SampleT, DIM>    PointT;

        // Biggest memory access word (16B)
        typedef ulonglong2 DeviceWord;

        /// Constants
        enum
        {
            WORD_ALIGN_BYTES     = cub::AlignBytes<DeviceWord>::ALIGN_BYTES,
            POINT_ALIGN_BYTES    = cub::AlignBytes<PointT>::ALIGN_BYTES,
            
            IS_MULTIPLE          = (sizeof(DeviceWord) % sizeof(PointT) == 0) && (WORD_ALIGN_BYTES % POINT_ALIGN_BYTES == 0),
            WORD_POINTS          = sizeof(DeviceWord) / sizeof(PointT),
            VECTOR_SIZE          = (IS_MULTIPLE == 1 && WORD_POINTS > 1) ? WORD_POINTS : 1,
            
            TRY_VECTORIZE        = (VECTOR_SIZE > 1 && POINTS_PER_THREAD > VECTOR_SIZE && (POINTS_PER_THREAD % VECTOR_SIZE == 0)) ? 1 : 0
        };

        /// Cache Store modifier for writing output elements
        static const cub::CacheStoreModifier STORE_MODIFIER = BlockTileStorePolicyT::STORE_MODIFIER;

        //---------------------------------------------------------------------
        // Utility
        //---------------------------------------------------------------------

        // Whether or not the output is aligned with the DeviceWord
        template <typename T>
        static __device__ __forceinline__ bool isAligned(
            T const *    d_out)
        {
            return (size_t(d_out) & (sizeof(DeviceWord) - 1)) == 0;
        }

        //---------------------------------------------------------------------
        // Tile storing interface
        //---------------------------------------------------------------------

        // Store full tile from ROW_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void storeTile2Row(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[POINTS_PER_THREAD][DIM],
            cub::Int2Type<true>       canVectorize)
        {
            enum 
            {
                /// to suppress errors when POINTS_PER_THREAD equals 1
                N_VECTORS = POINTS_PER_THREAD / VECTOR_SIZE,
                VECTORS_PER_THREAD = (N_VECTORS > 0) ? N_VECTORS : 1 
            };

            typedef DeviceWord AliasedPoints[VECTORS_PER_THREAD];

            AliasedPoints &aliasedPoints = reinterpret_cast<AliasedPoints&>(samples);
            DeviceWord *vecPtr = reinterpret_cast<DeviceWord *>(d_outputSamples);

            #pragma unroll
            for (int p = 0; p < VECTORS_PER_THREAD; ++p)    
            {
                cub::ThreadStore<STORE_MODIFIER>(vecPtr + p * BLOCK_THREADS + threadIdx.x, aliasedPoints[p]);
            }    
        }

        // Store full tile from COL_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void storeTile2Col(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[DIM][POINTS_PER_THREAD],
            cub::Int2Type<true>       canVectorize)
        {
            enum 
            {
                /// to suppress errors when POINTS_PER_THREAD equals 1
                N_VECTORS = POINTS_PER_THREAD / VECTOR_SIZE,
                VECTORS_PER_THREAD = (N_VECTORS > 0) ? N_VECTORS : 1 
            };

            typedef SampleT AliasedDeviceWord[WORD_POINTS][DIM];
            DeviceWord *vecPtr = reinterpret_cast<DeviceWord *>(d_outputSamples);

            #pragma unroll
            for (int p = 0; p < VECTORS_PER_THREAD; ++p)    
            {
                DeviceWord auxWord;
                AliasedDeviceWord &aliasedWord = reinterpret_cast<AliasedDeviceWord&>(auxWord);

                #pragma unroll
                for (int k = 0; k < WORD_POINTS; ++k)
                {
                    for (int d = 0; d < DIM; ++d)
                    {
                        aliasedWord[k][d] = samples[d][p * WORD_POINTS + k];
                    }
                }

                cub::ThreadStore<STORE_MODIFIER>(vecPtr + p * BLOCK_THREADS + threadIdx.x, auxWord);
            }    
        }

        // Store full tile from ROW_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void storeTile2Row(
            SampleT *                 d_outputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            cub::Int2Type<false>            canVectorize)
        {
            typedef PointT AliasedPoints[POINTS_PER_THREAD];
            AliasedPoints &points = reinterpret_cast<AliasedPoints&>(samples);
            PointT *d_wrappedPoints = reinterpret_cast<PointT *>(d_outputSamples + threadIdx.x * DIM);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)    
            {
                cub::ThreadStore<STORE_MODIFIER>(d_wrappedPoints + p * BLOCK_THREADS, points[p]);
            }  
        }

        // Store full tile from COL_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void storeTile2Col(
            SampleT *                 d_outputSamples,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            cub::Int2Type<false>            canVectorize)
        {
            PointT *d_wrappedPoints = reinterpret_cast<PointT *>(d_outputSamples + threadIdx.x * DIM);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)    
            {
                PointT auxPoint;

                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    ((SampleT *)(&auxPoint))[d] = samples[d][p];
                }
                cub::ThreadStore<STORE_MODIFIER>(d_wrappedPoints + p * BLOCK_THREADS, auxPoint);
            }  
        }

        // Store partial tile from ROW_MAJOR ordered thread-private table.
        static __device__ __forceinline__ void storeTile2Row(
            SampleT *     d_outputSamples,
            int           validPoints,
            SampleT       (&samples)[POINTS_PER_THREAD][DIM])
        {
            typedef PointT AliasedPoints[POINTS_PER_THREAD];
            AliasedPoints &aliasedPoints = reinterpret_cast<AliasedPoints&>(samples);
            PointT * d_wrappedPoints = reinterpret_cast<PointT *>(d_outputSamples);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if (BLOCK_THREADS <= validPoints)
                {
                    cub::ThreadStore<STORE_MODIFIER>(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x, aliasedPoints[p]);
                }
                else if (threadIdx.x < validPoints)
                {
                    cub::ThreadStore<STORE_MODIFIER>(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x, aliasedPoints[p]);
                }
                validPoints -= BLOCK_THREADS;
                // stop storing if we've stored all valid points
                if (validPoints <= 0)
                    break;
            }    
        } 

        // Store partial tile from COL_MAJOR ordered thread-private table.
        static __device__ __forceinline__ void storeTile2Col(
            SampleT *     d_outputSamples,
            int                 validPoints,
            SampleT             (&samples)[DIM][POINTS_PER_THREAD])
        {
            PointT * d_wrappedPoints = reinterpret_cast<PointT *>(d_outputSamples);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                PointT auxPoint;
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    ((SampleT *)(&auxPoint))[d] = samples[d][p];
                }

                if (BLOCK_THREADS <= validPoints)
                {
                    cub::ThreadStore<STORE_MODIFIER>(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x, auxPoint);
                }
                else if (threadIdx.x < validPoints)
                {
                    cub::ThreadStore<STORE_MODIFIER>(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x, auxPoint);
                }

                validPoints -= BLOCK_THREADS;
                // stop storing if we've stored all valid points
                if (validPoints <= 0)
                    break;
            }    
        }   
    };

    /**************************************************************************
     *  IO_BACKEND_TROVE
     **************************************************************************/
    template <int DUMMY>
    struct StoreInternal<IO_BACKEND_TROVE, DUMMY>
    {
        //---------------------------------------------------------------------
        // constants and type definitions
        //---------------------------------------------------------------------

        /// The point type of SampleT
        typedef cub::CubVector<SampleT, DIM>    PointT;

        /// Constants
        enum
        {
            /*
             * Trove decides whether or not to vectorize loads. It does vectorization when:
             * -- type size is multiple of 4B
             * -- and when sizeof(T) / sizeof(int/int2/int4) is in range (1, 64)
             */
            TRY_VECTORIZE        = 1
        };

        //---------------------------------------------------------------------
        // Utility
        //---------------------------------------------------------------------

        // Whether or not the output is aligned with the PointT
        template <typename T>
        static __device__ __forceinline__ bool isAligned(
            T const *    d_out)
        {
            return (size_t(d_out) & (sizeof(PointT) - 1)) == 0;
        }

        //---------------------------------------------------------------------
        // Tile storing interface
        //---------------------------------------------------------------------

        // Store full tile from ROW_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void storeFullTile2Row(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[POINTS_PER_THREAD][DIM])
        {
            typedef PointT AliasedPoints[POINTS_PER_THREAD];
            AliasedPoints &points = reinterpret_cast<AliasedPoints&>(samples);
            PointT *d_wrappedPoints = reinterpret_cast<PointT *>(d_outputSamples + threadIdx.x * DIM);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)    
            {
                trove::store_warp_contiguous(points[p], d_wrappedPoints + p * BLOCK_THREADS);
            } 
        }

        // Store full tile from COL_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void storeFullTile2Col(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[DIM][POINTS_PER_THREAD])
        {
            PointT *d_wrappedPoints = reinterpret_cast<PointT *>(d_outputSamples + threadIdx.x * DIM);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)    
            {
                PointT auxPoint;
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    ((SampleT *)(&auxPoint))[d] = samples[d][p];
                }
                trove::store_warp_contiguous(auxPoint, d_wrappedPoints + p * BLOCK_THREADS);
            } 
        }

        // Store full tile from ROW_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void storeTile2Row(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[POINTS_PER_THREAD][DIM],
            cub::Int2Type<true>       canVectorize)
        {
             storeFullTile2Row(d_outputSamples, samples);
        }

        // Store full tile from ROW_MAJOR ordered thread-private table(non-vectorized).
        static __device__ __forceinline__ void storeTile2Row(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[POINTS_PER_THREAD][DIM],
            cub::Int2Type<false>      canVectorize)
        {
            storeFullTile2Row(d_outputSamples, samples);
        }

        // Store full tile from COL_MAJOR ordered thread-private table(vectorized).
        static __device__ __forceinline__ void storeTile2Col(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[DIM][POINTS_PER_THREAD],
            cub::Int2Type<true>       canVectorize)
        {
             storeFullTile2Col(d_outputSamples, samples);
        }

        // Store full tile from COL_MAJOR ordered thread-private table(non-vectorized).
        static __device__ __forceinline__ void storeTile2Col(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[DIM][POINTS_PER_THREAD],
            cub::Int2Type<false>      canVectorize)
        {
            storeFullTile2Col(d_outputSamples, samples);
        }

        // Store partial tile from ROW_MAJOR ordered thread-private table
        static __device__ __forceinline__ void storeTile2Row(
            SampleT *     d_outputSamples,
            int           validPoints,
            SampleT       (&samples)[POINTS_PER_THREAD][DIM])
        {
            typedef PointT AliasedPoints[POINTS_PER_THREAD];
            AliasedPoints &aliasedPoints = reinterpret_cast<AliasedPoints&>(samples);
            PointT *d_wrappedPoints = reinterpret_cast<PointT *>(d_outputSamples);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if (BLOCK_THREADS <= validPoints)
                {
                    trove::store_warp_contiguous(aliasedPoints[p], d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x);
                }
                else if (threadIdx.x < validPoints)
                {
                    *(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x) = aliasedPoints[p];
                }
                validPoints -= BLOCK_THREADS;
                // stop storing if we've stored all valid points
                if (validPoints <= 0)
                    break;
            }    
        } 

        // Store partial tile from COL_MAJOR ordered thread-private table
        static __device__ __forceinline__ void storeTile2Col(
            SampleT *     d_outputSamples,
            int                 validPoints,
            SampleT             (&samples)[DIM][POINTS_PER_THREAD])
        {
            PointT *d_wrappedPoints = reinterpret_cast<PointT *>(d_outputSamples);

            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                PointT auxPoint;
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    ((SampleT *)(&auxPoint))[d] = samples[d][p];
                }

                if (BLOCK_THREADS <= validPoints)
                {
                    trove::store_warp_contiguous(auxPoint, d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x);
                }
                else if (threadIdx.x < validPoints)
                {
                    *(d_wrappedPoints + p * BLOCK_THREADS + threadIdx.x) = auxPoint;
                }


                validPoints -= BLOCK_THREADS;
                // stop storing if we've stored all valid points
                if (validPoints <= 0)
                    break;
            }    
        }   
    };

    //---------------------------------------------------------------------
    // Type definitions
    //---------------------------------------------------------------------

    /// Internal Store implementation to use
    typedef StoreInternal<IO_BACKEND, 0> InternalStore;

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------
public:
    /**
     * Constructor
     */
    __device__ __forceinline__ BlockTileStore()
    {
    }

    // Store full tile from ROW_MAJOR ordered thread-private table
    static __device__ __forceinline__ void storeTile2Row(
        SampleT *     d_outputSamples,
        SampleT       (&samples)[POINTS_PER_THREAD][DIM],
        OffsetT       )
    {
        if (InternalStore::isAligned(d_outputSamples) && InternalStore::TRY_VECTORIZE)
        {
            InternalStore::storeTile2Row(d_outputSamples, samples, cub::Int2Type<true>());   
        }
        else
        {
            InternalStore::storeTile2Row(d_outputSamples, samples, cub::Int2Type<false>());   
        }
    }

    // Store partial tile from ROW_MAJOR ordered thread-private table
    static __device__ __forceinline__ void storeTile2Row(
        SampleT *     d_outputSamples,
        int           validPoints,
        SampleT       (&samples)[POINTS_PER_THREAD][DIM],
        OffsetT       )
    {
        InternalStore::storeTile2Row(d_outputSamples, validPoints, samples);   
    }

    // Store full tile from COL_MAJOR ordered thread-private table
    static __device__ __forceinline__ void storeTile2Col(
        SampleT *     d_outputSamples,
        SampleT       (&samples)[DIM][POINTS_PER_THREAD],
        OffsetT       )
    {
        if (InternalStore::isAligned(d_outputSamples) && InternalStore::TRY_VECTORIZE)
        {
            InternalStore::storeTile2Col(d_outputSamples, samples, cub::Int2Type<true>());   
        }
        else
        {
            InternalStore::storeTile2Col(d_outputSamples, samples, cub::Int2Type<false>());   
        }
    }

    // Store partial tile from COL_MAJOR ordered thread-private table
    static __device__ __forceinline__ void storeTile2Col(
        SampleT *     d_outputSamples,
        int           validPoints,
        SampleT       (&samples)[DIM][POINTS_PER_THREAD],
        OffsetT       )
    {
        InternalStore::storeTile2Col(d_outputSamples, validPoints, samples);   
    }

};

// Specialization for data COL_MAJOR order
template <
    typename                BlockTileStorePolicyT,
    int                     DIM,
    BlockTileIOBackend      IO_BACKEND,
    typename                SampleT,
    typename                OffsetT>
class BlockTileStore<BlockTileStorePolicyT, DIM, rd::COL_MAJOR, IO_BACKEND, SampleT, OffsetT>
{
    //---------------------------------------------------------------------
    // constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS           = BlockTileStorePolicyT::BLOCK_THREADS,
        POINTS_PER_THREAD       = BlockTileStorePolicyT::POINTS_PER_THREAD,
    };


    //---------------------------------------------------------------------
    // Internal helper storer class
    //---------------------------------------------------------------------

    template <BlockTileIOBackend _BACKEND, int DUMMY>
    struct StoreInternal;

    /**************************************************************************
     *  IO_BACKEND_CUB
     **************************************************************************/
    template <int DUMMY>
    struct StoreInternal<IO_BACKEND_CUB, DUMMY>
    {
        //---------------------------------------------------------------------
        // constants and type definitions
        //---------------------------------------------------------------------

        /// The point type of SampleT
        typedef cub::CubVector<SampleT, DIM>    PointT;

        /// Constants
        enum
        {
            ROW_SAMPLES_SIZE    = sizeof(SampleT) * POINTS_PER_THREAD,
            DEVICE_WORD_SIZE    = (ROW_SAMPLES_SIZE % 16 == 0) ? 16
                                    : (ROW_SAMPLES_SIZE % 8 == 0) ? 8
                                        : 1,

            SAMPLE_ALIGN_BYTES  = cub::AlignBytes<SampleT>::ALIGN_BYTES,
            IS_MULTIPLE         = (DEVICE_WORD_SIZE % sizeof(SampleT) == 0) && (DEVICE_WORD_SIZE % SAMPLE_ALIGN_BYTES == 0),

            DEVICE_WORD_SAMPLES = DEVICE_WORD_SIZE / sizeof(SampleT),
            VECTOR_SIZE         = (IS_MULTIPLE == 1 && DEVICE_WORD_SAMPLES > 1) ? DEVICE_WORD_SAMPLES : 1,

            TRY_VECTORIZE       = (VECTOR_SIZE > 1) ? 1 : 0
        };

        // Biggest memory access word that ROW_SAMPLES_SIZE is a whole multiple of
        typedef typename cub::If<DEVICE_WORD_SIZE == 16,
            ulonglong2,
            typename cub::If<DEVICE_WORD_SIZE == 8,
                unsigned long long,
                unsigned int>::Type>::Type      DeviceWord;


        /// Cache Store modifier for writing output elements
        static const cub::CacheStoreModifier STORE_MODIFIER = BlockTileStorePolicyT::STORE_MODIFIER;

        //---------------------------------------------------------------------
        // Utility
        //---------------------------------------------------------------------

        // Whether or not the output is aligned with the DeviceWord
        template <typename T>
        static __device__ __forceinline__ bool isAligned(
            T const *    d_out)
        {
            return (size_t(d_out) & (sizeof(DeviceWord) - 1)) == 0;
        }

        //---------------------------------------------------------------------
        // Tile storing interface
        //---------------------------------------------------------------------

        // Store a full tile of samples from ROW_MAJOR ordered thread private table
        static __device__ __forceinline__ void storeFullTile2Row(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                   stride)
        {
            d_outputSamples += threadIdx.x;

            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                #pragma unroll
                for (int p = 0; p < POINTS_PER_THREAD; ++p)    
                {
                    cub::ThreadStore<STORE_MODIFIER>(d_outputSamples + p * BLOCK_THREADS, samples[p][d]);
                } 
                d_outputSamples += stride;   
            }            
        }

        // // Store a full tile of samples from ROW_MAJOR ordered thread private table. (vectorized)
        static __device__ __forceinline__ void storeTile2Row(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                   stride,
            cub::Int2Type<true>       canVectorize)
        {
            storeFullTile2Row(d_outputSamples, samples, stride);
        }

        // Store a full tile of samples from ROW_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void storeTile2Row(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                   stride,
            cub::Int2Type<false>      canVectorize)
        {
            storeFullTile2Row(d_outputSamples, samples, stride);
        }
        
        // Store a full tile of samples from COL_MAJOR ordered thread-private table to a COL_MAJOR global table (vectorized).
        static __device__ __forceinline__ void storeTile2Col(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[DIM][POINTS_PER_THREAD],
            OffsetT                   stride,
            cub::Int2Type<true>       canVectorize)
        {
            enum 
            {
                VECTORS_PER_THREAD = POINTS_PER_THREAD / VECTOR_SIZE
            };

            typedef DeviceWord AliasedPoints[DIM][VECTORS_PER_THREAD];
            AliasedPoints &aliasedPoints = reinterpret_cast<AliasedPoints&>(samples);

            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                DeviceWord *vecPtr = reinterpret_cast<DeviceWord *>(
                    d_outputSamples + threadIdx.x * VECTOR_SIZE + d * stride);
                #pragma unroll
                for (int p = 0; p < VECTORS_PER_THREAD; ++p)    
                {
                    cub::ThreadStore<STORE_MODIFIER>(vecPtr + p * BLOCK_THREADS, aliasedPoints[d][p]);
                }    
                // can't advance vecPtr by stride here, because it would move it out of bounds of output samples!
            }  
        }

        // Store a full tile of samples from COL_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void storeTile2Col(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[DIM][POINTS_PER_THREAD],
            OffsetT                   stride,
            cub::Int2Type<false>      canVectorize)
        {
            d_outputSamples += threadIdx.x;

            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                #pragma unroll
                for (int p = 0; p < POINTS_PER_THREAD; ++p)    
                {
                    cub::ThreadStore<STORE_MODIFIER>(d_outputSamples + p * BLOCK_THREADS, samples[d][p]);
                }    
                d_outputSamples += stride;
            }  
        }

        // Store a partial tile of samples from ROW_MAJOR ordered thread-private table
        static __device__ __forceinline__ void storeTile2Row(
            SampleT *                 d_outputSamples,
            int                             validPoints,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if (BLOCK_THREADS <= validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        cub::ThreadStore<STORE_MODIFIER>(d_outputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x, samples[p][d]);
                    }
                }
                else if (threadIdx.x < validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        cub::ThreadStore<STORE_MODIFIER>(d_outputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x, samples[p][d]);
                    }
                }
                validPoints -= BLOCK_THREADS;
                // stop storing if we've loaded all valid points
                if (validPoints <= 0)
                    break;
            }   
        } 

        // Store a partial tile of samples from COL_MAJOR ordered thread-private table
        static __device__ __forceinline__ void storeTile2Col(
            SampleT *                 d_outputSamples,
            int                             validPoints,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            OffsetT                         stride)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if (BLOCK_THREADS <= validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        cub::ThreadStore<STORE_MODIFIER>(d_outputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x, samples[d][p]);
                    }
                }
                else if (threadIdx.x < validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        cub::ThreadStore<STORE_MODIFIER>(d_outputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x, samples[d][p]);
                    }
                }
                validPoints -= BLOCK_THREADS;
                // stop storing if we've loaded all valid points
                if (validPoints <= 0)
                    break;
            }    
        }   
    };

    /**************************************************************************
     *  IO_BACKEND_TROVE
     **************************************************************************/
    template <int DUMMY>
    struct StoreInternal<IO_BACKEND_TROVE, DUMMY>
    {
        //---------------------------------------------------------------------
        // constants and type definitions
        //---------------------------------------------------------------------

        /// The point type of SampleT
        typedef cub::CubVector<SampleT, DIM>    PointT;

        /// Constants
        enum
        {
            ROW_SAMPLES_SIZE    = sizeof(SampleT) * POINTS_PER_THREAD,
            DEVICE_WORD_SIZE    = (ROW_SAMPLES_SIZE % 16 == 0) ? 16 
                                    : (ROW_SAMPLES_SIZE % 8 == 0) ? 8
                                        : 1,
            
            SAMPLE_ALIGN_BYTES  = cub::AlignBytes<SampleT>::ALIGN_BYTES,
            IS_MULTIPLE         = (DEVICE_WORD_SIZE % sizeof(SampleT) == 0) && (DEVICE_WORD_SIZE % SAMPLE_ALIGN_BYTES == 0),
            
            DEVICE_WORD_SAMPLES = DEVICE_WORD_SIZE / sizeof(SampleT),
            VECTOR_SIZE         = (IS_MULTIPLE == 1 && DEVICE_WORD_SAMPLES > 1) ? DEVICE_WORD_SAMPLES : 1,
            
            TRY_VECTORIZE       = (VECTOR_SIZE > 1) ? 1 : 0
        };

        // Biggest memory access word that ROW_SAMPLES_SIZE is a whole multiple of 
        typedef typename cub::If<DEVICE_WORD_SIZE == 16, 
            ulonglong2,
            typename cub::If<DEVICE_WORD_SIZE == 8, 
                unsigned long long,
                unsigned int>::Type>::Type      DeviceWord;

        //---------------------------------------------------------------------
        // Utility
        //---------------------------------------------------------------------

        // Whether or not the output is aligned with the DeviceWord
        template <typename T>
        static __device__ __forceinline__ bool isAligned(
            T const *    d_out)
        {
            return (size_t(d_out) & (sizeof(DeviceWord) - 1)) == 0;
        }

        //---------------------------------------------------------------------
        // Tile storing interface
        //---------------------------------------------------------------------

        // Store a full tile of samples from ROW_MAJOR ordered thread private table
        static __device__ __forceinline__ void storeFullTile2Row(
            SampleT *                 d_outputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride)
        {
            d_outputSamples += threadIdx.x;

            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                #pragma unroll
                for (int p = 0; p < POINTS_PER_THREAD; ++p)    
                {
                    trove::store_warp_contiguous(samples[p][d], d_outputSamples + p * BLOCK_THREADS);
                } 
                d_outputSamples += stride;   
            }            
        }

        // // Store a full tile of samples from ROW_MAJOR ordered thread private table. (vectorized)
        static __device__ __forceinline__ void storeTile2Row(
            SampleT *                 d_outputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride,
            cub::Int2Type<true>             canVectorize)
        {
            storeFullTile2Row(d_outputSamples, samples, stride);
        }

        // Store a full tile of samples from ROW_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void storeTile2Row(
            SampleT *                 d_outputSamples,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride,
            cub::Int2Type<false>            canVectorize)
        {
            storeFullTile2Row(d_outputSamples, samples, stride);
        }
        
        // Store a full tile of samples from COL_MAJOR ordered thread-private table (vectorized).
        static __device__ __forceinline__ void storeTile2Col(
            SampleT *                 d_outputSamples,
            SampleT                         (&samples)[DIM][POINTS_PER_THREAD],
            OffsetT                         stride,
            cub::Int2Type<true>             canVectorize)
        {
            enum 
            {
                VECTORS_PER_THREAD = POINTS_PER_THREAD / VECTOR_SIZE
            };

            typedef DeviceWord AliasedPoints[DIM][VECTORS_PER_THREAD];
            AliasedPoints &aliasedPoints = reinterpret_cast<AliasedPoints&>(samples);

            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                DeviceWord *vecPtr = reinterpret_cast<DeviceWord *>(
                    d_outputSamples + threadIdx.x * VECTOR_SIZE + d * stride);
                #pragma unroll
                for (int p = 0; p < VECTORS_PER_THREAD; ++p)    
                {
                    trove::store_warp_contiguous(aliasedPoints[d][p], vecPtr + p * BLOCK_THREADS);
                }    
                // can't advance vecPtr by stride here, because it would move it out of bounds of output samples!
            }  
        }

        // Store a full tile of samples from COL_MAJOR ordered thread-private table (non-vectorized).
        static __device__ __forceinline__ void storeTile2Col(
            SampleT *                 d_outputSamples,
            SampleT                   (&samples)[DIM][POINTS_PER_THREAD],
            OffsetT                   stride,
            cub::Int2Type<false>      canVectorize)
        {
            d_outputSamples += threadIdx.x;
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                #pragma unroll
                for (int p = 0; p < POINTS_PER_THREAD; ++p)    
                {
                    trove::store_warp_contiguous(samples[d][p], d_outputSamples + p * BLOCK_THREADS);
                }    
                d_outputSamples += stride;
            }  
        }

        // Store a partial tile of samples from ROW_MAJOR ordered thread-private table
        static __device__ __forceinline__ void storeTile2Row(
            SampleT *                 d_outputSamples,
            int                             validPoints,
            SampleT                         (&samples)[POINTS_PER_THREAD][DIM],
            OffsetT                         stride)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if (BLOCK_THREADS <= validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        trove::store_warp_contiguous(samples[p][d], d_outputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x);
                    }
                }
                else if (threadIdx.x < validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        *(d_outputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x) = samples[p][d];
                    }
                }
                validPoints -= BLOCK_THREADS;
                // stop storing if we've loaded all valid points
                if (validPoints <= 0)
                    break;
            }   
        } 

        // Store a partial tile of samples from COL_MAJOR ordered thread-private table
        static __device__ __forceinline__ void storeTile2Col(
            SampleT *                 d_outputSamples,
            int                       validPoints,
            SampleT                   (&samples)[DIM][POINTS_PER_THREAD],
            OffsetT                   stride)
        {
            #pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p)
            {
                if (BLOCK_THREADS <= validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        trove::store_warp_contiguous(samples[d][p], d_outputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x);
                    }
                }
                else if (threadIdx.x < validPoints)
                {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        *(d_outputSamples + p * BLOCK_THREADS + d * stride + threadIdx.x) = samples[d][p];
                    }
                }
                validPoints -= BLOCK_THREADS;
                // stop storing if we've loaded all valid points
                if (validPoints <= 0)
                    break;
            }    
        }   
    };

    //---------------------------------------------------------------------
    // Type definitions
    //---------------------------------------------------------------------

    /// Internal Store implementation to use
    typedef StoreInternal<IO_BACKEND, 0> InternalStore;

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------
public:
    /**
     * Constructor
     */
    __device__ __forceinline__ BlockTileStore()
    {
    }

    // Store full tile from ROW_MAJOR ordered thread-private table
    static __device__ __forceinline__ void storeTile2Row(
        SampleT *     d_outputSamples,
        SampleT       (&samples)[POINTS_PER_THREAD][DIM],
        OffsetT       stride)     /// distance (expressed in number of samples) between point's consecutive coordinates
    {
        if (InternalStore::isAligned(d_outputSamples) && InternalStore::TRY_VECTORIZE)
        {
            InternalStore::storeTile2Row(d_outputSamples, samples, stride, cub::Int2Type<true>());   
        }
        else
        {
            InternalStore::storeTile2Row(d_outputSamples, samples, stride, cub::Int2Type<false>());   
        }
    }
    // Store full tile from COL_MAJOR ordered thread-private table
    static __device__ __forceinline__ void storeTile2Col(
        SampleT *     d_outputSamples,
        SampleT       (&samples)[DIM][POINTS_PER_THREAD],
        OffsetT       stride)     /// distance (expressed in number of samples) between point's consecutive coordinates
    {
        if (InternalStore::isAligned(d_outputSamples) && InternalStore::TRY_VECTORIZE)
        {
            InternalStore::storeTile2Col(d_outputSamples, samples, stride, cub::Int2Type<true>());   
        }
        else
        {
            InternalStore::storeTile2Col(d_outputSamples, samples, stride, cub::Int2Type<false>());   
        }
    }

    // Store partial tile from ROW_MAJOR ordered thread-private table
    static __device__ __forceinline__ void storeTile2Row(
        SampleT *     d_outputSamples,
        int           validPoints,
        SampleT       (&samples)[POINTS_PER_THREAD][DIM],
        OffsetT       stride)     /// distance (expressed in number of samples) between point's consecutive coordinates
    {
        InternalStore::storeTile2Row(d_outputSamples, validPoints, samples, stride);   
    }

    // Store partial tile int COL_MAJOR ordered thread-private table
    static __device__ __forceinline__ void storeTile2Col(
        SampleT *     d_outputSamples,
        int           validPoints,
        SampleT       (&samples)[DIM][POINTS_PER_THREAD],
        OffsetT       stride)     /// distance (expressed in number of samples) between point's consecutive coordinates
    {
        InternalStore::storeTile2Col(d_outputSamples, validPoints, samples, stride);   
    }

};

} // end namespace gpu 
} // end namespace rd
