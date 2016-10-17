/**
 * @file agent_dist_mtx.cuh
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

#include "cub/util_type.cuh"

namespace rd
{
namespace gpu
{
namespace detail
{


/**
 * @brief      This class is a thread block abstraction for participating in device-wide 
 *             calculation of symmetric distance matrix S = AA^T  or S = A^TA.
 *
 * @par Algorithm
 *     It uses effective algorithm described by Vasily Volkov, where output matrix is a sum of 
 *     set of matrices each of which is a result of outer product of k-th A column and k-th A^T \
 *     row. Such decomposition characterizes high data locality and thus high data reusage. This 
 *     of course reduces global memory bandwidth usage and has a positive performance impact.
 *
 * @tparam     BLOCK_W     Thread block width
 * @tparam     BLOCK_H     Thread block height
 * @tparam     MEM_LAYOUT  Input data memory layout (ROW_MAJOR or COL_MAJOR)
 * @tparam     T           Input data type.
 */
template <
    int                 BLOCK_W,
    int                 BLOCK_H,
    DataMemoryLayout    MEM_LAYOUT,
    typename            T>
class AgentDistMtx
{

    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    enum
    { 
        TILE_H              = 32,
        TILE_W              = 32,
        /*
         * In case of row-major memory layout we read consecutive rows of A matrix and store them
         * into consecutive rows of smem, and for A^T matrix we read consecutive rows and store 
         * them into columns of smem. 
         * In case of col-major we have opposite situation. We read consecutive rows of A and 
         * store them into columns of smem, and for A^T we read rows and store them as rows in 
         * smem. When we store data into columns of smem, we padd it with extra column to avoid 
         * bank conflicts
         */
        TILE_W1             = TILE_W + 1,
        TILE_H1             = TILE_H + 1,
        ITEMS_PER_THREAD    = TILE_H / BLOCK_H
    };

    static_assert(BLOCK_W == TILE_W, "BLOCK_W must equal to TILE_W!");
    static_assert((TILE_H % BLOCK_H) == 0, "BLOCK_H must evenly divide TILE_H!");

private:
    //---------------------------------------------------------------------
    // Tile loader helper structure
    //---------------------------------------------------------------------

    template <DataMemoryLayout _MEM_LAYOUT, int DUMMY>
    struct TileLoader {};

    template <int DUMMY>
    struct TileLoader<ROW_MAJOR, DUMMY> 
    {
        struct _TempStorage
        {
            T As[TILE_H][TILE_W];
            T AsT[TILE_W][TILE_H1];
        };

        struct TempStorage: cub::Uninitialized<_TempStorage> {};
        
        _TempStorage &smem;

        __device__ __forceinline__ TileLoader(TempStorage &storage)
        :
            smem(storage.Alias())
        {}

        __device__ __forceinline__ void loadFullTile(
            T const * __restrict__ A,
            const int   t,
            const int   tileRowOffset,
            const int   rowOffset,
            const int   colOffset,
            const int   A_stride)
        {
            const int column = t * TILE_W + threadIdx.x;
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                smem.As[tileRowOffset + r][threadIdx.x] = 
                    A[(rowOffset + tileRowOffset + r) * A_stride + column];
                smem.AsT[threadIdx.x][tileRowOffset + r] = 
                    A[(colOffset + tileRowOffset + r) * A_stride + column];
            }
        }

        __device__ __forceinline__ void loadPartialCTile(
            T const * __restrict__ A,
            const int   t,
            const int   width,
            const int   tileRowOffset,
            const int   rowOffset,
            const int   colOffset,
            const int   A_stride)
        {
            const int column = t * TILE_W + threadIdx.x;
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                smem.As[tileRowOffset + r][threadIdx.x] = (column < width) ? 
                    A[(rowOffset + tileRowOffset + r) * A_stride + column] : 0;
                smem.AsT[threadIdx.x][tileRowOffset + r] = (column < width) ? 
                    A[(colOffset + tileRowOffset + r) * A_stride + column] : 0;
            }
        }

        __device__ __forceinline__ void loadPartialR1Tile(
            T const * __restrict__ A,
            const int   t,
            const int   height,
            const int   tileRowOffset,
            const int   rowOffset,
            const int   colOffset,
            const int   A_stride)
        {
            int column = t * TILE_W + threadIdx.x;
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                smem.As[tileRowOffset + r][threadIdx.x] = (rowOffset + tileRowOffset + r < height) ?
                    A[(rowOffset + tileRowOffset + r) * A_stride + column] : 0;
                smem.AsT[threadIdx.x][tileRowOffset + r] = 
                    A[(colOffset + tileRowOffset + r) * A_stride + column];
            }
        }

        __device__ __forceinline__ void loadPartialCR1Tile(
            T const * __restrict__ A,
            const int   t,
            const int   width,
            const int   height,
            const int   tileRowOffset,
            const int   rowOffset,
            const int   colOffset,
            const int   A_stride)
        {
            int column = t * TILE_W + threadIdx.x;
            #pragma unroll 
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                smem.As[tileRowOffset + r][threadIdx.x] = (column < width && 
                    rowOffset + tileRowOffset + r < height) ? 
                    A[(rowOffset + tileRowOffset + r) * A_stride + column] : 0;
                smem.AsT[threadIdx.x][tileRowOffset + r] = (column < width) ? 
                    A[(colOffset + tileRowOffset + r) * A_stride + column] : 0;
            }
        }

        __device__ __forceinline__ void loadPartialRRTile(
            T const * __restrict__ A,
            const int   t,
            const int   height,
            const int   tileRowOffset,
            const int   rowOffset,
            const int   colOffset,
            const int   A_stride)
        {
            int column = t * TILE_W + threadIdx.x;
            #pragma unroll 
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                smem.As[tileRowOffset + r][threadIdx.x] = (rowOffset + tileRowOffset + r < height) ? 
                    A[(rowOffset + tileRowOffset + r) * A_stride + column] : 0;
                smem.AsT[threadIdx.x][tileRowOffset + r] = (colOffset + tileRowOffset + r < height) ?
                    A[(colOffset + tileRowOffset + r) * A_stride + column] : 0;
            }
        }

        __device__ __forceinline__ void loadPartialCRRTile(
            T const * __restrict__ A,
            const int   t,
            const int   width,
            const int   height,
            const int   tileRowOffset,
            const int   rowOffset,
            const int   colOffset,
            const int   A_stride)
        {
            int column = t * TILE_W + threadIdx.x;
            #pragma unroll 
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                smem.As[tileRowOffset + r][threadIdx.x] = (column < width && 
                    rowOffset + tileRowOffset + r < height) ? 
                    A[(rowOffset + tileRowOffset + r) * A_stride + column] : 0;
                smem.AsT[threadIdx.x][tileRowOffset + r] = (column < width && 
                    colOffset + tileRowOffset + r < height) ?
                    A[(colOffset + tileRowOffset + r) * A_stride + column] : 0;
            }
        }
    };

    template <int DUMMY>
    struct TileLoader<COL_MAJOR, DUMMY>
    {
        struct _TempStorage
        {
            T As[TILE_H][TILE_W1];
            T AsT[TILE_W][TILE_H];
        };

        struct TempStorage: cub::Uninitialized<_TempStorage> {};
        _TempStorage &smem;

        __device__ __forceinline__ TileLoader(TempStorage &storage)
        :
            smem(storage.Alias())
        {}


        __device__ __forceinline__ void loadFullTile(
            T const * __restrict__ A,
            const int   t,
            const int   tileRowOffset,
            const int   rowOffset,
            const int   colOffset,
            const int   A_stride)
        {
            const int column = t * TILE_W + tileRowOffset;
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                smem.As[threadIdx.x][tileRowOffset + r] = 
                    A[(column + r) * A_stride + rowOffset + threadIdx.x];
                smem.AsT[tileRowOffset + r][threadIdx.x] = 
                    A[(column + r) * A_stride + colOffset + threadIdx.x];
            }
        }

        __device__ __forceinline__ void loadPartialCTile(
            T const * __restrict__ A,
            const int   t,
            const int   width,
            const int   tileRowOffset,
            const int   rowOffset,
            const int   colOffset,
            const int   A_stride)
        {
            const int column = t * TILE_W + tileRowOffset;
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                smem.As[threadIdx.x][tileRowOffset + r] = (column + r < width) ?
                    A[(column + r) * A_stride + rowOffset + threadIdx.x] : 0;
                smem.AsT[tileRowOffset + r][threadIdx.x] = (column + r < width) ?
                    A[(column + r) * A_stride + colOffset + threadIdx.x] : 0;
            }
        }

        __device__ __forceinline__ void loadPartialR1Tile(
            T const * __restrict__ A,
            const int   t,
            const int   height,
            const int   tileRowOffset,
            const int   rowOffset,
            const int   colOffset,
            const int   A_stride)
        {
            int column = t * TILE_W + tileRowOffset;
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                smem.As[threadIdx.x][tileRowOffset + r] = (rowOffset + threadIdx.x < height) ?
                    A[(column + r) * A_stride + rowOffset + threadIdx.x] : 0;
                smem.AsT[tileRowOffset + r][threadIdx.x] = 
                    A[(column + r) * A_stride + colOffset + threadIdx.x];
            }
        }

        __device__ __forceinline__ void loadPartialCR1Tile(
            T const * __restrict__ A,
            const int   t,
            const int   width,
            const int   height,
            const int   tileRowOffset,
            const int   rowOffset,
            const int   colOffset,
            const int   A_stride)
        {
            int column = t * TILE_W + tileRowOffset;
            #pragma unroll 
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                smem.As[threadIdx.x][tileRowOffset + r] = (column + r < width && 
                    rowOffset + threadIdx.x < height) ?
                    A[(column + r) * A_stride + rowOffset + threadIdx.x] : 0;
                smem.AsT[tileRowOffset + r][threadIdx.x] = (column + r < width) ?
                    A[(column + r) * A_stride + colOffset + threadIdx.x] : 0;
            }
        }

        __device__ __forceinline__ void loadPartialRRTile(
            T const * __restrict__ A,
            const int   t,
            const int   height,
            const int   tileRowOffset,
            const int   rowOffset,
            const int   colOffset,
            const int   A_stride)
        {
            int column = t * TILE_W + tileRowOffset;
            #pragma unroll 
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                smem.As[threadIdx.x][tileRowOffset + r] = (rowOffset + threadIdx.x < height) ?
                    A[(column + r) * A_stride + rowOffset + threadIdx.x] : 0;
                smem.AsT[tileRowOffset + r][threadIdx.x] = (colOffset + threadIdx.x < height) ?
                    A[(column + r) * A_stride + colOffset + threadIdx.x] : 0;
            }
        }

        __device__ __forceinline__ void loadPartialCRRTile(
            T const * __restrict__ A,
            const int   t,
            const int   width,
            const int   height,
            const int   tileRowOffset,
            const int   rowOffset,
            const int   colOffset,
            const int   A_stride)
        {
            int column = t * TILE_W + tileRowOffset;
            #pragma unroll 
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                smem.As[threadIdx.x][tileRowOffset + r] = (column + r < width && 
                    rowOffset + threadIdx.x < height) ?
                    A[(column + r) * A_stride + rowOffset + threadIdx.x] : 0;
                smem.AsT[tileRowOffset + r][threadIdx.x] = (column + r < width && 
                    colOffset + threadIdx.x < height) ?
                    A[(column + r) * A_stride + colOffset + threadIdx.x] : 0;
            }
        }
    };

    typedef TileLoader<MEM_LAYOUT, 0> InternalTileLoader;
    typedef typename InternalTileLoader::TempStorage _TempStorage;
public:
    struct TempStorage: cub::Uninitialized<_TempStorage> {};
private:
    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    InternalTileLoader tileLoader;
    // squared distance
    T sqrDist[ITEMS_PER_THREAD];

    //---------------------------------------------------------------------
    // Computations
    //---------------------------------------------------------------------

    __device__ __forceinline__ void consumeFullTile(
        T const __restrict__ *  A,
        const int               rowOffset,
        const int               colOffset,
        const int               A_stride,
        const int               t)
    {
        int tileRowOffset = threadIdx.y * ITEMS_PER_THREAD;
        tileLoader.loadFullTile(A, t, tileRowOffset, rowOffset, colOffset, A_stride);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_W; ++k)
        {
            T atVal = tileLoader.smem.AsT[k][threadIdx.x];
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                T diff = tileLoader.smem.As[tileRowOffset + r][k] - atVal;
                sqrDist[r] += diff * diff;
            }
        }
        __syncthreads();
    }

    __device__ __forceinline__ void consumePartialCTile(
        T const __restrict__ *  A,
        const int               width,
        const int               rowOffset,
        const int               colOffset,
        const int               A_stride,
        const int               t)
    {
        int tileRowOffset = threadIdx.y * ITEMS_PER_THREAD;
        tileLoader.loadPartialCTile(A, t, width, tileRowOffset, rowOffset, colOffset, 
            A_stride);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < width - t * TILE_W; ++k)
        {
            T atVal = tileLoader.smem.AsT[k][threadIdx.x];
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                T diff = tileLoader.smem.As[tileRowOffset + r][k] - atVal;
                sqrDist[r] += diff * diff;
            }
        }
        __syncthreads();
    }

    __device__ __forceinline__ void consumePartialR1Tile(
        T const __restrict__ *  A,
        const int               height,
        const int               rowOffset,
        const int               colOffset,
        const int               A_stride,
        const int               t)
    {
        int tileRowOffset = threadIdx.y * ITEMS_PER_THREAD;
        tileLoader.loadPartialR1Tile(A, t, height, tileRowOffset, rowOffset, colOffset, 
            A_stride);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_W; ++k)
        {
            T atVal = tileLoader.smem.AsT[k][threadIdx.x];
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                // we omit checking condition (threadIdx.y * ITEMS_PER_THREAD + r < height)
                // because we have already zeroed that part of As, thus it won't affect 
                // overall result
                T diff = tileLoader.smem.As[tileRowOffset + r][k] - atVal;
                sqrDist[r] += diff * diff;
            }
        }
        __syncthreads();
    }

    __device__ __forceinline__ void consumePartialCR1Tile(
        T const __restrict__ *  A,
        const int               width,
        const int               height,
        const int               rowOffset,
        const int               colOffset,
        const int               A_stride,
        const int               t)
    {
        int tileRowOffset = threadIdx.y * ITEMS_PER_THREAD;
        tileLoader.loadPartialCR1Tile(A, t, width, height, tileRowOffset, rowOffset, colOffset, 
            A_stride);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < width - t * TILE_W; ++k)
        {
            T atVal = tileLoader.smem.AsT[k][threadIdx.x];
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                // we omit checking condition (threadIdx.y * ITEMS_PER_THREAD + r < height)
                // because we have already zeroed that part of As, thus it won't affect 
                // overall result
                T diff = tileLoader.smem.As[tileRowOffset + r][k] - atVal;
                sqrDist[r] += diff * diff;
            }
        }
        __syncthreads();
    }

    __device__ __forceinline__ void consumePartialRRTile(
        T const __restrict__ *  A,
        const int               height,
        const int               rowOffset,
        const int               colOffset,
        const int               A_stride,
        const int               t)
    {
        int tileRowOffset = threadIdx.y * ITEMS_PER_THREAD;
        tileLoader.loadPartialRRTile(A, t, height, tileRowOffset, rowOffset, colOffset, 
            A_stride);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_W; ++k)
        {
            // we could check condition (threadIdx.x + colOffset < height), 
            // but we already have zeroed this part, and thus we can avoid divergence
            T atVal = tileLoader.smem.AsT[k][threadIdx.x];
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                // we omit checking condition (threadIdx.y * ITEMS_PER_THREAD + r < height)
                // because we have already zeroed that part of As, thus it won't affect 
                // overall result
                T diff = tileLoader.smem.As[tileRowOffset + r][k] - atVal;
                sqrDist[r] += diff * diff;
            }
        }
        __syncthreads();
    }

    __device__ __forceinline__ void consumePartialCRRTile(
        T const __restrict__ *  A,
        const int               width,
        const int               height,
        const int               rowOffset,
        const int               colOffset,
        const int               A_stride,
        const int               t)
    {
        int tileRowOffset = threadIdx.y * ITEMS_PER_THREAD;
        tileLoader.loadPartialCRRTile(A, t, width, height, tileRowOffset, rowOffset, colOffset, 
            A_stride);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < width - t * TILE_W; ++k)
        {
            // we could check condition (threadIdx.x + colOffset < height), 
            // but we already have zeroed this part, and thus we can avoid divergence
            T atVal = tileLoader.smem.AsT[k][threadIdx.x];
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r)
            {
                // we omit checking condition (threadIdx.y * ITEMS_PER_THREAD + r < height)
                // because we have already zeroed that part of As, thus it won't affect 
                // overall result
                T diff = tileLoader.smem.As[tileRowOffset + r][k] - atVal;
                sqrDist[r] += diff * diff;
            }
        }
        __syncthreads();
    }

    __device__ __forceinline__ void storeFullTileOutput(
        T *                     C,
        const int               rowOffset,
        const int               colOffset,
        const int               C_stride)
    {
        // if we're on C's diagonal, we don't need to write mirror tile
        if (rowOffset == colOffset)
        {
            T *Cptr = C + (rowOffset + threadIdx.y * ITEMS_PER_THREAD) * C_stride + 
                colOffset + threadIdx.x;
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r, Cptr += C_stride)
            {
                *Cptr = sqrDist[r];
            }
        }
        else
        {
            T *Cptr = C + (rowOffset + threadIdx.y * ITEMS_PER_THREAD) * C_stride + 
                colOffset + threadIdx.x;

            // in the 'meantime' we store mirror tile in shared memory in order to have 
            // coalesced gmem writes
            // we always want to write to padded smem table, to avoid bank conflicts,
            // so we must choose appropriate one
            if (MEM_LAYOUT == ROW_MAJOR)
            {
                #pragma unroll
                for (int r = 0; r < ITEMS_PER_THREAD; ++r, Cptr += C_stride)
                {
                    *Cptr = sqrDist[r];
                    tileLoader.smem.AsT[threadIdx.x][threadIdx.y * ITEMS_PER_THREAD + r] = sqrDist[r];
                }

                __syncthreads();

                Cptr =  C + (colOffset + threadIdx.y * ITEMS_PER_THREAD) * C_stride + 
                    rowOffset + threadIdx.x;
                #pragma unroll
                for (int r = 0; r < ITEMS_PER_THREAD; ++r, Cptr += C_stride)
                {
                    *Cptr = tileLoader.smem.AsT[threadIdx.y * ITEMS_PER_THREAD + r][threadIdx.x];
                }
            }
            else if (MEM_LAYOUT == COL_MAJOR)
            {
                #pragma unroll
                for (int r = 0; r < ITEMS_PER_THREAD; ++r, Cptr += C_stride)
                {
                    *Cptr = sqrDist[r];
                    tileLoader.smem.As[threadIdx.x][threadIdx.y * ITEMS_PER_THREAD + r] = 
                        sqrDist[r];
                }

                __syncthreads();

                Cptr =  C + (colOffset + threadIdx.y * ITEMS_PER_THREAD) * C_stride + 
                    rowOffset + threadIdx.x;
                #pragma unroll
                for (int r = 0; r < ITEMS_PER_THREAD; ++r, Cptr += C_stride)
                {
                    *Cptr = tileLoader.smem.As[threadIdx.y * ITEMS_PER_THREAD + r][threadIdx.x];
                }    
            }

            __syncthreads();
        }
    }

    __device__ __forceinline__ void storePartialCR1TileOutput(
        T *                     C,
        int                     height,
        int                     rowOffset,
        int                     colOffset,
        int                     C_stride)
    {
        T *Cptr = C + (rowOffset + threadIdx.y * ITEMS_PER_THREAD) * C_stride + 
                colOffset + threadIdx.x;

        // in the 'meantime' we store mirror tile in shared memory in order to have 
        // coalesced gmem writes
        // we always want to write to padded smem table, to avoid bank conflicts,
        // so we must choose appropriate one
        if (MEM_LAYOUT == ROW_MAJOR)                
        {
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r, Cptr += C_stride)
            {
                if (rowOffset + threadIdx.y * ITEMS_PER_THREAD + r < height)
                {
                    *Cptr = sqrDist[r];
                    tileLoader.smem.AsT[threadIdx.x][threadIdx.y * ITEMS_PER_THREAD + r] = 
                        sqrDist[r];
                }
            }
            __syncthreads();

            Cptr = C + (colOffset + threadIdx.y * ITEMS_PER_THREAD) * C_stride + 
                rowOffset + threadIdx.x;

            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r, Cptr += C_stride)
            {
                if (rowOffset + threadIdx.x < height)
                {
                    *Cptr = tileLoader.smem.AsT[threadIdx.y * ITEMS_PER_THREAD + r][threadIdx.x];
                }
            }
        }
        else if (MEM_LAYOUT == COL_MAJOR)
        {
            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r, Cptr += C_stride)
            {
                if (rowOffset + threadIdx.y * ITEMS_PER_THREAD + r < height)
                {
                    *Cptr = sqrDist[r];
                    tileLoader.smem.As[threadIdx.x][threadIdx.y * ITEMS_PER_THREAD + r] = 
                        sqrDist[r];
                }
            }
            __syncthreads();

            Cptr = C + (colOffset + threadIdx.y * ITEMS_PER_THREAD) * C_stride + 
                rowOffset + threadIdx.x;

            #pragma unroll
            for (int r = 0; r < ITEMS_PER_THREAD; ++r, Cptr += C_stride)
            {
                if (rowOffset + threadIdx.x < height)
                {
                    *Cptr = tileLoader.smem.As[threadIdx.y * ITEMS_PER_THREAD + r][threadIdx.x];
                }
            }
        }

        __syncthreads();
    }

    __device__ __forceinline__ void calcFullTile(
        T const __restrict__ *  A,
        T *                     C,
        int                     width,
        const int               p1,
        const int               rowOffset,
        const int               colOffset,
        const int               A_stride,
        const int               C_stride)
    {
        #pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; ++k)
        {
            sqrDist[k] = 0;
        }

        int t;
        for (t = 0; t < (p1 / TILE_W) - 1; ++t)
        {
            consumeFullTile(A, rowOffset, colOffset, A_stride, t);
        }
        consumePartialCTile(A, width, rowOffset, colOffset, A_stride, t);

        storeFullTileOutput(C, rowOffset, colOffset, C_stride);
    }


    __device__ __forceinline__ void calcPartialTile(
        T const __restrict__ *  A,
        T *                     C,
        int                     width,
        int                     height,
        int                     p1,
        int                     rowOffset,
        int                     colOffset,
        int                     A_stride,
        int                     C_stride)
    {
        #pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; ++k)
        {
            sqrDist[k] = 0;
        }

        int t;
        // CR1 case
        if (rowOffset + TILE_H > height && colOffset + TILE_H < height)
        {
            for (t = 0; t < (p1 / TILE_W) - 1; ++t)
            {
                consumePartialR1Tile(A, height, rowOffset, colOffset, A_stride, t);
            }
            consumePartialCR1Tile(A, width, height, rowOffset, colOffset, A_stride, t);

            storePartialCR1TileOutput(C, height, rowOffset, colOffset, C_stride);
        }
        // CRR case 
        else
        {
            for (t = 0; t < (p1 / TILE_W) - 1; ++t)
            {
                consumePartialRRTile(A, height, rowOffset, colOffset, A_stride, t);
            }
            consumePartialCRRTile(A, width, height, rowOffset, colOffset, A_stride, t);
    
            T *Cptr1 = C + (rowOffset + threadIdx.y * ITEMS_PER_THREAD) * C_stride + 
                colOffset + threadIdx.x;
            if (colOffset + threadIdx.x < height)
            {
                #pragma unroll
                for (int r = 0; r < ITEMS_PER_THREAD; ++r, Cptr1 += C_stride)
                {
                    if (rowOffset + threadIdx.y * ITEMS_PER_THREAD + r < height)
                    {
                        *Cptr1 = sqrDist[r];
                    }
                }
            }
        }


    }

    __device__ __forceinline__ void calcOutputTiles(
        T const __restrict__ *  A,
        T *                     C,
        const int               width,
        const int               height,
        const int               A_stride,
        const int               C_stride)
    {
        // rounded up (to full tiles) number of C rows
        // rounded up (to full tiles) number of C cols
        const int m1 = (height + TILE_H - 1) & ~(TILE_H - 1);
        // rounded up (to full tiles) number of A columns
        const int p1 = (width + TILE_W - 1) & ~(TILE_W - 1);

        for (int rowOffset = blockIdx.y * TILE_H;
                 rowOffset < m1;
                 rowOffset += gridDim.y * TILE_H)
        {
            for (int colOffset = blockIdx.x * TILE_H;
                     colOffset < m1;
                     colOffset += gridDim.x * TILE_H)
            {
                // we're computing a symmetric matrix, so we need to compute only a half of it
                // (above diagonal, or below diagonal triangular matrix)
                if (colOffset <= rowOffset)
                {
                    if (rowOffset + TILE_H <= height &&
                        colOffset + TILE_H <= height )
                    {
                        calcFullTile(A, C, width, p1, rowOffset, colOffset, A_stride, C_stride);
                    }
                    else 
                    {
                        calcPartialTile(A, C, width, height, p1, rowOffset, colOffset, 
                            A_stride, C_stride);
                    }
                }
            }
        }
    }

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

public:

    __device__ __forceinline__ AgentDistMtx(
        TempStorage     &sharedStorage)
    :
        tileLoader(sharedStorage.Alias())
    {
    }

    /**
     * @brief      Calculate C = dist(A,A) - symmetric, Euclidean distance matrix
     * 
     * A may have row or column major order, however C is always row-major.
     *
     * @param      A         Input matrix
     * @param      C         Output (squared, symmetric) matrix
     * @param[in]  width     Input matrix width - points dimensions
     * @param[in]  height    Input matrix height - number of points
     * @param[in]  A_stride  Input matrix stride - (number of elements in a leading dimension)
     * @param[in]  C_stride  Output matrix stride - (number of elements in a row)
     */
    __device__ __forceinline__ void symDist(
        T const __restrict__ *  A,
        T *                     C,
        const int               width,
        const int               height,
        const int               A_stride,
        const int               C_stride)
    {
        calcOutputTiles(A, C, width, height, A_stride, C_stride);
    }

};


} // end namespace detail
} // end namespace gpu
} // end namespace rd