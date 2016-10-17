/**
 * @file decimate_nan_marker1.cuh
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

#include "rd/gpu/warp/warp_functions.cuh"
#include "rd/gpu/thread/thread_sqr_euclidean_dist.cuh"
#include "rd/gpu/util/dev_math.cuh"
#include "rd/gpu/util/data_order_traits.hpp"

#include "cub/util_type.cuh"
#include "cub/util_ptx.cuh"

namespace rd
{
namespace gpu
{
namespace bruteForce
{

template <
    int         DIM,
    int         BLOCK_SIZE,
    typename    T>
static __device__ int ctaCountNeighbouringPoints_nanCheck1(
    T const * __restrict__  points,
    int                     np,
    T const * __restrict__  srcP,
    T                       rSqr,
    int                     threshold,
    rowMajorOrderTag )
{
    T refPoint[DIM];
    __shared__ int s_threashold;
    __shared__ T smem[BLOCK_SIZE * DIM];

    if (np <= 0)
    {
        return 0;
    }

    if (threadIdx.x == 0)
    {
        s_threashold = 0;
    }

    #pragma unroll
    for (int d = 0; d < DIM; ++d)
    {
        refPoint[d] = srcP[d];
    }

    // numer of steps
    int m = (np + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    for (int offset = 0; offset < m; offset += BLOCK_SIZE)
    {
        #pragma unroll
        for (int d = 0; d < DIM; ++d)
        {
            int idx = offset*DIM + threadIdx.x + d * BLOCK_SIZE;
            smem[d * BLOCK_SIZE + threadIdx.x] = (idx < np*DIM) ? points[idx] : 0;
        }
        __syncthreads();

        if (offset + threadIdx.x < np) 
        {
            if(!isnan(smem[threadIdx.x*DIM]))
            {
                if(threadSqrEuclideanDistanceRowMajor(smem + threadIdx.x*DIM, refPoint, DIM) 
                    <= rSqr)
                {
                    int haveNeighbour = __ballot(true);
                    int nCount = __popc(haveNeighbour);
                    if (laneId() == __ffs(haveNeighbour) - 1)
                    {
                        (void) atomicAdd(&s_threashold, nCount);
                    }
                }
            }
        }

        __syncthreads();
        if (s_threashold >= threshold)
        {
            return s_threashold;
        }
    }

    __syncthreads();
    return s_threashold;
}

/*
 *  col - col major order
 */
template <
    int         DIM,
    int         BLOCK_SIZE,
    typename    T>
static __device__ int ctaCountNeighbouringPoints_nanCheck1(
    T const * __restrict__  points,
    int                     np,
    int                     stride1,
    T const * __restrict__  srcP,
    int                     stride2,
    T                       rSqr,
    int                     threshold,
    colMajorOrderTag)
{
    T refPoint[DIM];
    __shared__ int s_threashold;
    __shared__ T smem[BLOCK_SIZE * DIM];

    if (np <= 0)
    {
        return 0;
    }

    if (threadIdx.x == 0)
    {
        s_threashold = 0;
    }

    #pragma unroll
    for (int d = 0; d < DIM; ++d)
    {
        refPoint[d] = srcP[d * stride2];
    }

    // numer of steps
    int m = (np + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    for (int offset = 0; offset < m; offset += BLOCK_SIZE)
    {
        #pragma unroll
        for (int d = 0; d < DIM; ++d)
        {
            smem[d * BLOCK_SIZE + threadIdx.x] = (offset + threadIdx.x < np) ? 
                    points[d * stride1 + offset + threadIdx.x] : 0;
        }
        __syncthreads();
        
        if (offset + threadIdx.x < np) 
        {
            if(!isnan(smem[threadIdx.x]))
            {
                if(threadSqrEuclideanDistance(smem + threadIdx.x, BLOCK_SIZE, refPoint, 1, DIM) 
                    <= rSqr)
                {
                    int haveNeighbour = __ballot(true);
                    int nCount = __popc(haveNeighbour);
                    if (laneId() == __ffs(haveNeighbour) - 1)
                    {
                        (void) atomicAdd(&s_threashold, nCount);
                    }
                }
            }
        }

        __syncthreads();
        if (s_threashold >= threshold)
        {
            return s_threashold;
        }
    }

    __syncthreads();
    return s_threashold;
}

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
    int         BLOCK_SIZE,
    int         DIM,
    typename    T>
__launch_bounds__ (BLOCK_SIZE)
static __global__ void decimateNanMarker1(
    T *     S,
    int *   ns,
    T const r,
    int ,
    cub::Int2Type<ROW_MAJOR>)
{

    const T rSqr = r*r;
    int prevCnt = 0;
    int neighbours = 0;
    int currCnt = *ns;
    const int chosenPtsCnt = currCnt;

    while (prevCnt != currCnt && currCnt > 3)
    {
        prevCnt = currCnt;
        for (int i = 0; i < chosenPtsCnt; ++i)
        {
            if (!isnan(S[i*DIM]))
            {
                neighbours = ctaCountNeighbouringPoints_nanCheck1<DIM, BLOCK_SIZE>(
                    S, chosenPtsCnt, S + i*DIM, rSqr, 4, rowMajorOrderTag());
                __syncthreads();
                if (neighbours >= 4)
                {
                    currCnt--;
                    // mark point to remove
                    if (threadIdx.x == 0)
                    {
                        S[i*DIM] = GetNaN<T>::value();
                    }
                    if (currCnt < 3)
                        break;
                    continue;
                }

                neighbours = ctaCountNeighbouringPoints_nanCheck1<DIM, BLOCK_SIZE>(
                    S, chosenPtsCnt, S + i*DIM, 4.f*rSqr, 3, rowMajorOrderTag());
                __syncthreads();
                if (neighbours <= 2)
                {

                    currCnt--;
                    // mark point to remove
                    if (threadIdx.x == 0)
                    {
                        S[i*DIM] = GetNaN<T>::value();
                    }
                    if (currCnt < 3)
                        break;
                } 
            } 
        }
    }

    __shared__ int nsCounter;

    if (threadIdx.x == 0)
    {
        *ns = currCnt;
        nsCounter = 0;
    }
    __syncthreads();

    // round-up to BLOCK_SIZE multiply
    int m = (chosenPtsCnt + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    // remove unnecessary points.
    for (int x = threadIdx.x; 
             x < m; 
             x += BLOCK_SIZE)
    {
        T buff[DIM];

        if (x < chosenPtsCnt) 
        {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                buff[d] = S[x*DIM + d];
            }
        }

        int notHaveNaN = 0;
        int chosenOffset = 0;
        int innerWarpIdx = 0;
        if (x < chosenPtsCnt) 
        {
            notHaveNaN = 1 ^ isnan(buff[0]);
            #if __CUDA_ARCH__ >= 300
            // how many threads don't have NaN?
            int warpMask = __ballot(notHaveNaN);
            int count = __popc(warpMask);
            // find my warp offset
            // select leader & update counter
            if (cub::LaneId() == __ffs(warpMask)-1)
            {
                chosenOffset = rdAtomicAdd(&nsCounter, count);
            }
            // let everybody know offset
            broadcast(chosenOffset,__ffs(warpMask)-1);
            // count my position in warp
            innerWarpIdx = __popc(warpMask & cub::LaneMaskLt());
            #else
            if (notHaveNaN)
            {
                chosenOffset = rdAtomicAdd(&nsCounter, 1);
            }
            #endif
        }

        if (notHaveNaN && x < chosenPtsCnt)
        {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                S[(chosenOffset + innerWarpIdx) * DIM + d] = buff[d];
            }
        }
    }
}

template <
    int         BLOCK_SIZE,
    int         DIM,
    typename    T>
__launch_bounds__ (BLOCK_SIZE)
static __global__ void decimateNanMarker1(
    T *     S,
    int *   ns,
    T const r,
    int     stride,
    cub::Int2Type<COL_MAJOR>)
{

    const T rSqr = r*r;
    int prevCnt = 0;
    int neighbours = 0;
    int currCnt = *ns;
    const int chosenPtsCnt = currCnt;

    while (prevCnt != currCnt && currCnt > 3)
    {
        prevCnt = currCnt;
        for (int i = 0; i < chosenPtsCnt; ++i)
        {
            if (!isnan(S[i]))
            {
                neighbours = ctaCountNeighbouringPoints_nanCheck1<DIM, BLOCK_SIZE>(
                    S, chosenPtsCnt, stride, S + i, stride, rSqr, 4, colMajorOrderTag());
                __syncthreads();
                if (neighbours >= 4)
                {
                    currCnt--;
                    // mark point to remove
                    if (threadIdx.x == 0)
                        S[i] = GetNaN<T>::value();
                    if (currCnt < 3)
                        break;
                    continue;
                }

                neighbours = ctaCountNeighbouringPoints_nanCheck1<DIM, BLOCK_SIZE>(
                    S, chosenPtsCnt, stride, S + i, stride, 4.f*rSqr, 3, colMajorOrderTag());
                __syncthreads();
                if (neighbours <= 2)
                {
                    currCnt--;
                    // mark point to remove
                    if (threadIdx.x == 0)
                        S[i] = GetNaN<T>::value();
                    if (currCnt < 3)
                        break;
                } 
            } 
        }
    }

    __shared__ int nsCounter;

    if (threadIdx.x == 0)
    {
        *ns = currCnt;
        nsCounter = 0;
    }
    __syncthreads();

    // round-up to BLOCK_SIZE multiply
    int m = (chosenPtsCnt + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    // remove unnecessary points.
    for (int x = threadIdx.x; 
             x < m; 
             x += BLOCK_SIZE)
    {
        T buff[DIM];

        if (x < chosenPtsCnt) 
        {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                buff[d] = S[x + d * stride];
            }
        }

        int notHaveNaN = 0;
        int chosenOffset = 0;
        int innerWarpIdx = 0;
        if (x < chosenPtsCnt) 
        {
            notHaveNaN = 1 ^ isnan(buff[0]);
            #if __CUDA_ARCH__ >= 300
            // how many threads don't have NaN?
            int warpMask = __ballot(notHaveNaN);
            int count = __popc(warpMask);
            // find my warp offset
            // select leader & update counter
            if (cub::LaneId() == __ffs(warpMask)-1)
            {
                chosenOffset = rdAtomicAdd(&nsCounter, count);
            }
            // let everybody know offset
            broadcast(chosenOffset,__ffs(warpMask)-1);
            // count my position in warp
            innerWarpIdx = __popc(warpMask & cub::LaneMaskLt());
            #else
            if (notHaveNaN)
            {
                chosenOffset = rdAtomicAdd(&nsCounter, 1);
            }
            #endif
        }

        if (notHaveNaN && x < chosenPtsCnt)
        {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                S[(chosenOffset + innerWarpIdx) + d * stride] = buff[d];
            }
        }
    }
}

}   // end namespace bruteForce
}   // end namespace gpu
}   // end namespace rd
