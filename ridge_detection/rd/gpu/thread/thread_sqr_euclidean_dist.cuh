/**
 * @file thread_sqr_euclidean_dist.cuh
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
 */


#ifndef THREAD_SQR_EUCLIDEAN_DIST_CUH_
#define THREAD_SQR_EUCLIDEAN_DIST_CUH_

#include <device_launch_parameters.h>
#include <host_defines.h>

namespace rd 
{
namespace gpu 
{

/**
 * @brief Compute square eculidean distance between points @p p1 and @p p2
 *	 	  using single thread.
 * @note version for SoA data layout.
 *
 * @param p1
 * @param p2
 * @param dim
 * @return
 */
template <typename T>
__device__ __forceinline__ T threadSqrEuclideanDistance(
	T const * __restrict__ const * __restrict__ p1,
	int p1Idx,
	T const * __restrict__ p2,
	int dim)
{

	T dist = 0, t;
	for (int i = 0; i < dim; ++i) 
	{
		t = p1[i][p1Idx] - p2[i];
		dist += t*t;
	}
	return dist;
}

/**
 *
 * @param p1 - pointer to data in row-major order
 * @param p2 - pointer to linear row-major order (single point)
 * @param dim - points dimension
 * @param stride - number of points in one column in @p p1's matrix
 * @return Euclidean distance between @p p1 and @p p2
 */
template <typename T>
__device__ __forceinline__ T threadSqrEuclideanDistanceRowMajor(
	T const * __restrict__ p1,
	T const * __restrict__ p2,
	int dim)
{
	T dist = 0, t;
	for (int i = 0; i < dim; ++i)
	{
		t = p1[i] - p2[i];
		dist += t*t;
	}
	return dist;
}

template <
	int DIM,
	typename T>
__device__ __forceinline__ T threadSqrEuclideanDistanceRowMajor(
	T const * __restrict__ p1,
	T const * __restrict__ p2)
{
	T dist = 0, t;
	#pragma unroll DIM
	for (int i = 0; i < DIM; ++i)
	{
		t = p1[i] - p2[i];
		dist += t*t;
	}
	return dist;
}

/**
 *
 * @param p1 - pointer to data in col-major order
 * @param p2 - pointer to second point
 * @param dim - points dimension
 * @param stride - number of points in one column in @p p1's matrix
 * @return Euclidean distance between @p p1 and @p p2
 */
template <typename T>
__device__ __forceinline__ T threadSqrEuclideanDistance(
	T const * __restrict__ p1,
	int p1Stride,
	T const * __restrict__ p2,
	int p2Stride,
	int dim)
{
	T dist = 0, t;
	for (int i = 0; i < dim; ++i) 
	{
		t = p1[i*p1Stride] - p2[i*p2Stride];
		dist += t*t;
	}
	return dist;
}


/**********************************************************
 * 
 * 				
 * 
 **********************************************************/

/**
 *
 * @param p1 - pointer to data in row-major order
 * @param p2 - pointer to linear row-major order (single point)
 * @param dim - points dimension
 * @param stride - number of points in one column in @p p1's matrix
 * @return Euclidean distance between @p p1 and @p p2
 */
template <
	int 		DIM,
	typename 	T>
__device__ __forceinline__ T threadSqrEuclideanDistance(
	T const * __restrict__ p1,
	T const * __restrict__ p2,
	int dim)
{
	T dist = 0, t;
	#pragma unroll
	for (int d = 0; d < dim; ++d)
	{
		t = p1[d] - p2[d];
		dist += t*t;
	}
	return dist;
}

} // end namespace rd
} // end namespace gpu

#endif /* THREAD_SQR_EUCLIDEAN_DIST_CUH_ */
