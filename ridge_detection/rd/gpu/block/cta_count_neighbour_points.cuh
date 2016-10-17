/**
 * @file cta_count_neighbour_points.cuh
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

#ifndef CTA_COUNT_NEIGHBOUR_POINTS_CUH_
#define CTA_COUNT_NEIGHBOUR_POINTS_CUH_

#include "rd/gpu/util/data_order_traits.hpp"
#include "rd/gpu/util/dev_math.cuh"
#include "rd/gpu/thread/thread_sqr_euclidean_dist.cuh"
#include "rd/gpu/warp/warp_functions.cuh"

#include "cub/util_ptx.cuh"

#include <device_launch_parameters.h>
#include <host_defines.h>
#include <device_functions.h>

namespace rd
{
namespace gpu
{

/**
 * @brief Counts points in @p rSqr neighbourhood of @p srcP
 * @param points - [in] set of points to examine
 * @param np - [in] size of the input set
 * @param srcP - [in] reference point
 * @param dim - [in] dimension of points
 * @param rSqr - [in] search radius
 * @param threshold - [in] minimal number of points to find
 * @return true if we find at least threshold points in @p rSqr euclidean
 * 			distance from @p srcP
 */
template <
	typename 	T, 
	int 		blockSize>
__device__ int ctaCountNeighbouringPoints(
	T const * __restrict__ points,
	int np,
	T const * __restrict__ srcP,
	int dim,
	T rSqr,
	int threshold,
	rowMajorOrderTag)
{

	__shared__ T shmem[blockSize];
	__shared__ int sThreshold_;

	// XXX: this limits data dimension
	shmem[threadIdx.x] = (threadIdx.x < dim) ? srcP[threadIdx.x] : 0;

	if (threadIdx.x == 0) {
		sThreshold_ = threshold;
	}

	int k = ((np + blockSize - 1) / blockSize );
	int x = threadIdx.x + threadIdx.y * blockDim.x;
	__syncthreads();

	for (int i = 0; i < k; ++i, x += blockSize) {
		if (x < np) {
			if (threadSqrEuclideanDistanceRowMajor(points + x*dim, shmem, dim) <= rSqr) {
				(void) atomicSub(&sThreshold_, 1);
			}
		}
		__syncthreads();
		if (sThreshold_ <= 0) {
			return 1;
		}
	}
	return 0;
}

/**
 *
 * @param stride [in] offset between signle point consecutive coordinates
 * @return
 */
template <
	typename 	T, 
	int 		blockSize>
__device__ int ctaCountNeighbouringPoints(
	T const * __restrict__ points,
	int np,
	int stride,
	T const * __restrict__ srcP,
	int dim,
	T rSqr,
	int threshold,
	colMajorOrderTag)
{
	__shared__ T shmem[blockSize];
	__shared__ int sThreshold_;

	shmem[threadIdx.x] = (threadIdx.x < dim) ? srcP[threadIdx.x * stride] : 0;

	if (threadIdx.x == 0)
	{
		sThreshold_ = threshold;
	}

	int k = ((np + blockSize - 1) / blockSize );
	int x = threadIdx.x + threadIdx.y * blockDim.x;
	__syncthreads();

	for (int i = 0; i < k; ++i, x += blockSize)
	{
		if (x < np)
		{
			if (threadSqrEuclideanDistance(points + x, stride, shmem, 1, dim) <= rSqr)
			{
				(void) atomicSub(&sThreshold_, 1);
			}
		}
		__syncthreads();
		if (sThreshold_ <= 0)
		{
			return 1;
		}
	}

	return 0;
}


/**
 * @brief This version is a bit more general than the previous one.
 *
 * @param pStride [in] - it's offset between single point (in @p points set)
 * 			consecutive coordinates
 * @param srcpStride [in] - it's offset between single point (in set containing
 * 			@p srcP) consecutive coordinates
 */
template <
	typename 	T, 
	int 		blockSize>
__device__ int ctaCountNeighbouringPoints(
	T const * __restrict__ points,
	int np,
	int pStride,
	T const * __restrict__ srcP,
	int srcpStride,
	int dim,
	T rSqr,
	int threshold)
{
	__shared__ T shmem[blockSize];
	__shared__ int sThreshold_;

	shmem[threadIdx.x] = (threadIdx.x < dim) ? srcP[threadIdx.x * srcpStride] : 0;

	if (threadIdx.x == 0) {
		sThreshold_ = threshold;
	}

	int k = ((np + blockSize - 1) / blockSize );
	int x = threadIdx.x + threadIdx.y * blockDim.x;
	__syncthreads();

	for (int i = 0; i < k; ++i, x += blockSize) {
		if (x < np) {
			if (threadSqrEuclideanDistance(points + x * dim, pStride,
					shmem, 1, dim) <= rSqr) {
				(void) atomicSub(&sThreshold_, 1);
			}
		}
		__syncthreads();
		if (sThreshold_ <= 0) {
			return 1;
		}
	}

	return 0;
}

/**********************************************************
 * 
 * 				
 * 
 **********************************************************/


/**
 * @brief      Counts points in @p rSqr neighbourhood of @p srcP
 *
 * @param      points      - [in] set of points to examine
 * @param      np          - [in] size of the input set
 * @param      srcP        - [in] reference point
 * @param      rSqr          - [in] search radius
 * @param      threshold   - [in] minimal number of points to find
 * @param[in]  <unnamed>   - Auxilliary parameter for choosing right implementation
 * @param      dim   - [in] dimension of points
 *
 * @tparam     DIM         Input points dimension
 * @tparam     BLOCK_SIZE  Number of launched threads within block.
 * @tparam     T           Input point coordinate data type.
 *
 * @return     Number of found points in neighbourhood if it doesn't exceed the threshold, otherwise
 *             the treshold.
 */
template <
	int 		DIM,
	int 		BLOCK_SIZE,
	typename 	T>
__device__ int ctaCountNeighbouringPoints_v2(
	T const * __restrict__ 	points,
	int 					np,
	T const * __restrict__ 	srcP,
	T 						rSqr,
	int 					threshold,
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
			if(threadSqrEuclideanDistanceRowMajor(smem + threadIdx.x*DIM, refPoint, DIM) <= rSqr)
			{
				int haveNeighbour = __ballot(true);
				int nCount = __popc(haveNeighbour);
				if (cub::LaneId() == __ffs(haveNeighbour) - 1)
				{
					(void) atomicAdd(&s_threashold, nCount);
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
 *	Mixed order: points in row-major, srcP in col-major
 */

template <
	int 		DIM,
	int 		BLOCK_SIZE,
	typename 	T>
__device__ int ctaCountNeighbouringPoints_v2(
	T const * __restrict__ 	points,
	int 					np,
	T const * __restrict__ 	srcP,
	int 					stride,
	T 						rSqr,
	int 					threshold,
	rowMajorOrderTag)
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
		refPoint[d] = srcP[d * stride];
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
			if(threadSqrEuclideanDistanceRowMajor(smem + threadIdx.x*DIM, refPoint, DIM) <= rSqr)
			{
				int haveNeighbour = __ballot(true);
				int nCount = __popc(haveNeighbour);
				if (cub::LaneId() == __ffs(haveNeighbour) - 1)
				{
					(void) atomicAdd(&s_threashold, nCount);
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
 *	col - col major order
 */
template <
	int 		DIM,
	int 		BLOCK_SIZE,
	typename 	T>
__device__ int ctaCountNeighbouringPoints_v2(
	T const * __restrict__ 	points,
	int 					np,
	int 					stride1,
	T const * __restrict__ 	srcP,
	int 					stride2,
	T 						rSqr,
	int 					threshold,
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
			if(threadSqrEuclideanDistance(smem + threadIdx.x, BLOCK_SIZE, refPoint, 1, DIM) <= rSqr)
			{
				int haveNeighbour = __ballot(true);
				int nCount = __popc(haveNeighbour);
				if (cub::LaneId() == __ffs(haveNeighbour) - 1)
				{
					(void) atomicAdd(&s_threashold, nCount);
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

} // end namspace gpu
} // end namspace rd


#endif /* CTA_COUNT_NEIGHBOUR_POINTS_CUH_ */
