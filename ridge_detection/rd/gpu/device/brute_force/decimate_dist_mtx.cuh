/**
 * @file decimate_dist_mtx.cuh
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
 * 
 * This version performs decimation based on distance matrix with a mask of decimated points
 */

#pragma once

#include <assert.h>

#include "rd/gpu/warp/warp_functions.cuh"
#include "rd/gpu/device/device_distance_mtx.cuh"

#include "rd/gpu/util/dev_math.cuh"
#include "rd/gpu/util/dev_utilities.cuh"
#include "rd/utils/memory.h"

#include "cub/util_type.cuh"
#include "cub/util_ptx.cuh"
#include "cub/util_arch.cuh"

namespace rd
{
namespace gpu
{
namespace bruteForce
{

/**
 * @brief      Fill memory with given value
 *
 * @param      out    Pointer to device memory to fill 
 * @param[in]  size   Number of elements to fill
 * @param[in]  value  The value
 */
__launch_bounds__(256)
static __global__ void rdMemset(char * out, int size, int value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // get number of 4-element vectors to cover size
    const int m = (size + 3) / 4;
    // round up to blockDim.x multiply
    const int n = (m + blockDim.x - 1) & ~(blockDim.x - 1);

    char4 * ptr = reinterpret_cast<char4*>(out);
    char oval = value;

    for (; i < n - blockDim.x; i += gridDim.x * blockDim.x)
    {
        ptr[i] = make_char4(oval, oval, oval, oval);
    }

    // last (probably not full) tile of data
    if (i < m - 1)
    {
        ptr[i] = make_char4(oval, oval, oval, oval);
    }

    // last output vector to store
    if (i == m - 1)
    {
        i *= 4;
        for (; i < size; ++i)
        {
            out[i] = char(value);
        }
    }
}


//----------------------------------------------------------
//
//----------------------------------------------------------

/**
 * @brief      Update block-common counter if specified condition is satisfied
 *
 *             Update is performed with one atomic operation per warp.
 *
 * @param[in]  dist       The distance
 * @param[in]  mask       The mask
 * @param[in]  rSqr       Squared radius value we compare @p dist to
 * @param      counter    Pointer to smem counter
 * @param[in]  threshold  The maximum number of points we seek
 *
 * @tparam     T          Distance data type
 *
 * @return     1 if we exceed @p threshold, otherwise 0
 */
template <typename T>
static __device__ __forceinline__ int updateCounter(
    T       dist,
    int     mask,
    T       rSqr,
    int *   counter,
    int     threshold)
{
    int haveNeighbour = __ballot(mask && dist >= 0 && dist <= rSqr);
    int nCount = __popc(haveNeighbour);
    if (laneId() == __ffs(haveNeighbour) - 1)
    {
        (void) atomicAdd(counter, nCount);
    }

    __syncthreads();
    if (*counter >= threshold)
    {
        return 1;
    }
    return 0;
}

/**
 * @brief      Counts how many points in @p points are closer than @p rSqr
 *
 * @param      points      Pointer to memory region containing points distances
 * @param[in]  width       The number of points in @p points
 * @param      pointsMask  The mask we use to select whether or not to examine respective points
 * @param[in]  rSqr        The squared radius (squared distance) value we compare @p points to
 * @param[in]  threshold   The maximum number of points we seek
 *
 * @tparam     BLOCK_SIZE  Number of threads within block
 * @tparam     T           Distance data type.
 *
 * @return     Number of points found.
 */
template <
    int         BLOCK_SIZE,
    typename    T>
static __device__ int countPoints(
    T const * __restrict__      points,
    const int                   width,
    char const * __restrict__   pointsMask,
    T                           rSqr,
    int                         threshold)
{
    __shared__ int counter;

    if (width <= 0)
    {
        return 0;
    }

    if (threadIdx.x == 0)
    {
        counter = 0;
    }
    __syncthreads();

    T dist = 0;
    int mask = 1;
    T distBuff = 0;
    short maskBuff = 0;
    int offset = 0;
    // numer of steps
    int m = (width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    // load first tile to smem
    distBuff = (threadIdx.x < width) ? points[threadIdx.x] : T(-1);
    maskBuff = (threadIdx.x < width) ? pointsMask[threadIdx.x] : 0;

    // work on full tiles
    for (offset += BLOCK_SIZE; offset <= m-2*BLOCK_SIZE; offset += BLOCK_SIZE)
    {
        // if (threadIdx.x == 0)
        // {
        //     printf("(full tile) offset: %d, counter: %d\n", offset, counter);
        // }
        // load data from smem to registers
        dist = distBuff;
        mask = static_cast<int>(maskBuff);
        // start loading next tile;
        distBuff = points[offset + threadIdx.x];
        maskBuff = pointsMask[offset + threadIdx.x];

        // process tile
        if(updateCounter(dist, mask, rSqr, &counter, threshold))
        {
            return counter;
        }
    }

    // work on (probably last) full tile
    for (; offset <= m-BLOCK_SIZE; offset += BLOCK_SIZE)
    {
        // if (threadIdx.x == 0)
        // {
        //     printf("(last full tile) offset: %d, counter: %d\n", offset, counter);
        // }        
        // load data from smem to registers
        dist = distBuff;
        mask = static_cast<int>(maskBuff);
        // start loading last, probably partial tile to registers;
        distBuff = (offset + threadIdx.x < width) ? points[offset + threadIdx.x] : T(-1);
        maskBuff = (offset + threadIdx.x < width) ? pointsMask[offset + threadIdx.x] : 0;
        
        // process tile
        if(updateCounter(dist, mask, rSqr, &counter, threshold))
        {
            return counter;
        }
    }

    // if (threadIdx.x == 0)
    // {
    //     printf("(partial tile) offset: %d, counter: %d\n", offset, counter);
    // }

    // load last tile from smem to registers
    dist = distBuff;
    mask = static_cast<int>(maskBuff);

    // process tile
    if(updateCounter(dist, mask, rSqr, &counter, threshold))
    {
        return counter;
    }

    __syncthreads();
    return counter;
}

/*******************************************
 * Utilities to load / store tile of points
 *******************************************/

template <
    int         N,
    typename    T>
static __device__ __forceinline__ void loadTile(
    T const * __restrict__ in,
    int offset,
    T   (&buff)[N],
    int ,
    int isValid,
    cub::Int2Type<ROW_MAJOR>)
{
    #pragma unroll
    for (int i = 0; i < N; ++i)
    {
        buff[i] = (isValid) ? in[offset * N + i]: 0;
    }
}

template <
    int         N,
    typename    T>
static __device__ __forceinline__ void loadTile(
    T const * __restrict__ in,
    int offset,
    T   (&buff)[N],
    int stride,
    int isValid,
    cub::Int2Type<COL_MAJOR>)
{
    #pragma unroll
    for (int i = 0; i < N; ++i)
    {
        buff[i] = (isValid) ? in[offset + i * stride]: 0;
    }
}

template <
    int         N,
    typename    T>
static __device__ __forceinline__ void storeTile(
    T * out,
    int offset,
    T   (&buff)[N],
    int ,
    cub::Int2Type<ROW_MAJOR>)
{
    #pragma unroll
    for (int i = 0; i < N; ++i)
    {
        out[offset * N + i] = buff[i];
    }
}

template <
    int         N,
    typename    T>
static __device__ __forceinline__ void storeTile(
    T * out,
    int offset,
    T   (&buff)[N],
    int stride,
    cub::Int2Type<COL_MAJOR>)
{
    #pragma unroll
    for (int i = 0; i < N; ++i)
    {
        out[offset + i * stride] = buff[i];
    }
}

/**
 * @brief      Calculates offset at which thread may write data
 *
 * @param[in]  havePoint   Whether or not we have a valid point
 * @param      sOutOffset  Block common offset counter (residing in smem)
 *
 * @return     The offset.
 */
static __device__ __forceinline__ int getOutOffset(
    int     havePoint,
    int *   sOutOffset)
{
    int offset = 0;
    // how many threads have a point
    int warpMask = __ballot(havePoint);
    // find my warp offset
    // select leader & update counter
    if (cub::LaneId() == __ffs(warpMask)-1)
    {
        offset = rdAtomicAdd(sOutOffset, __popc(warpMask));
    }
    // let everybody know offset
    broadcast(offset,__ffs(warpMask)-1);
    // count my position in warp
    offset += __popc(warpMask & cub::LaneMaskLt());
      return offset;
}

/**
 * @brief      Removes marked points by @p pMask from @p points set
 *
 * @param      points      Pointer to points set.
 * @param      pMask       The mask we use to decide whether or not to remove a point
 * @param[in]  pStride     Distance between consecutive point's coordinates in @p points set.
 * @param[in]  size        Number of points in @p points set
 *
 * @tparam     DIM         Point's dimension
 * @tparam     BLOCK_SIZE  Number of threads within block
 * @tparam     MEM_LAYOUT  Data layout in memory [row/col-major]
 * @tparam     T           Point coordinate data type.
 */
template <
    int                 DIM,
    int                 BLOCK_SIZE,
    DataMemoryLayout    MEM_LAYOUT,
    typename            T>
static __device__ __forceinline__ void reducePoints(
    T *     points,
    char const * __restrict__ pMask,
    int     pStride,
    int     size) 
{
    __shared__ int outOffset;

    if (threadIdx.x == 0)
    {
        outOffset = 0;
    }
    __syncthreads();

    // round-up to BLOCK_SIZE multiply
    int m = (size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    T buff[DIM];
    int mask;
    int chosenOffset = 0;
    int x;

    // process full tiles
    for (x = threadIdx.x; 
         x < m - BLOCK_SIZE; 
         x += BLOCK_SIZE)
    {
        mask = static_cast<int>(pMask[x]);

        loadTile(points, x, buff, pStride, mask, cub::Int2Type<MEM_LAYOUT>());
        chosenOffset = getOutOffset(mask, &outOffset);

        if (mask)
        {
            storeTile(points, chosenOffset, buff, pStride, cub::Int2Type<MEM_LAYOUT>());
        }
    }

    mask = (x < size) ? static_cast<int>(pMask[x]) : 0;
    loadTile(points, x, buff, pStride, mask, cub::Int2Type<MEM_LAYOUT>());
    chosenOffset = getOutOffset(mask, &outOffset);
    if (mask)
    {
        storeTile(points, chosenOffset, buff, pStride, cub::Int2Type<MEM_LAYOUT>());
    }
}


template <int _BLOCK_SIZE>
struct DecimateDistMtxPolicy
{
    enum 
    {
        BLOCK_SIZE = _BLOCK_SIZE
    };
};

/**
 * @brief Removes redundant points from set S.
 *
 * The function removes points satisfying at least one of the two
 * following conditions:
 * @li point has at least 4 neighbours in 2R neighbourhood
 * @li point has at most 2 neighbours in 4R neighbourhood
 *
 * The algorithm stops when there is no more than 3 points left, or when
 * there are no points satisfying specified criteria.
 * First it marks redundant points with NaN, and then reduces table.
 */
template <
    int                 DIM,
    int                 BLOCK_SIZE,
    DataMemoryLayout    MEM_LAYOUT,
    typename            T>
__launch_bounds__ (BLOCK_SIZE)
static __global__ void decimateDistMtx(
    T *         S,
    int *       ns,
    const int   sStride,
    T *         distSMtx,
    const int   distSMtxStride,
    char *      mask,
    T const     r)
{
    const T rSqr = r*r;
    int prevCnt = 0;
    int neighbours = 0;
    int currCnt = *ns;
    const int chosenPtsCnt = currCnt;

    if (threadIdx.x == 0)
    {
        cudaStream_t distMtxStream = nullptr;
        cudaStream_t maskMemsetStream = nullptr;
        rdDevCheckCall(cudaStreamCreateWithFlags(&distMtxStream, cudaStreamNonBlocking));
        rdDevCheckCall(cudaStreamCreateWithFlags(&maskMemsetStream, cudaStreamNonBlocking));

        DeviceDistanceMtx::symmetricDistMtx<MEM_LAYOUT>(S, distSMtx, DIM, chosenPtsCnt, 
            sStride, distSMtxStride, distMtxStream);
        // rdDevCheckCall(cudaMemsetAsync(mask, 1, chosenPtsCnt, maskMemsetStream));

        // get current device id
        int deviceOrdinal;
        rdDevCheckCall(cudaGetDevice(&deviceOrdinal));
        // Get SM count
        int smCount;
        rdDevCheckCall(cudaDeviceGetAttribute(&smCount, 
            cudaDevAttrMultiProcessorCount, deviceOrdinal));
        // Get SM occupancy
        int smOccupancy;
        rdDevCheckCall(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &smOccupancy, rdMemset, 256, 0));

        int dimGrid = smCount * smOccupancy * CUB_PTX_SUBSCRIPTION_FACTOR;
        rdMemset<<<dimGrid, 256, 0, maskMemsetStream>>>(mask, chosenPtsCnt, 1);
        rdDevCheckCall(cudaPeekAtLastError());
        
        rdDevCheckCall(cudaStreamDestroy(distMtxStream));
        rdDevCheckCall(cudaStreamDestroy(maskMemsetStream));
        rdDevCheckCall(cudaDeviceSynchronize());
    }
    __syncthreads();

    while (prevCnt != currCnt && currCnt > 3)
    {
        prevCnt = currCnt;
        for (int i = 0; i < chosenPtsCnt; ++i)
        {
            __syncthreads();
            if (mask[i])
            {
                T const * distMtxRowPtr = distSMtx + i * distSMtxStride;
                neighbours = countPoints<BLOCK_SIZE>(distMtxRowPtr, chosenPtsCnt, mask, rSqr, 4);
                if (neighbours >= 4)
                {
                    currCnt--;
                    // mark point to remove
                    if (threadIdx.x == 0)
                    {
                        mask[i] = 0;
                    }
                    if (currCnt < 3)
                    {
                        break;
                    }
                    continue;
                }

                neighbours = countPoints<BLOCK_SIZE>(distMtxRowPtr, chosenPtsCnt, mask, 4.f*rSqr, 3);
                if (neighbours <= 2)
                {
                    currCnt--;
                    // mark point to remove
                    if (threadIdx.x == 0)
                    {
                        mask[i] = 0;
                    }
                    if (currCnt < 3)
                    {
                        break;
                    }
                } 
            }
        }
    }

    if (threadIdx.x == 0)
    {
        *ns = currCnt;
    }

    reducePoints<DIM, BLOCK_SIZE, MEM_LAYOUT>(S, mask, sStride, chosenPtsCnt);
}

}   // end namespace bruteForce
}   // end namespace gpu
}   // end namespace rd
