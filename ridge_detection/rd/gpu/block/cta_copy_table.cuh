/**
 * @file cta_copy_table.cuh
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

#ifndef CTA_COPY_TABLE_CUH_
#define CTA_COPY_TABLE_CUH_

#include "rd/gpu/util/data_order_traits.hpp"

#include <device_launch_parameters.h>
#include <host_defines.h>

#include "cub/util_debug.cuh"

namespace rd
{
namespace gpu
{

/**
 * @brief Copy @p n elements of type @tparam T from @p src to @p dst
 * @param src
 * @param dst
 * @param n - number of elements to copy
 * @param dim - if dim equals one, we are copying linear memory segment
 * @param stride - offset between consecutive coordinates of elements
 */
template <typename T>
__device__ __forceinline__ 
void ctaCopyTable(
	T const * __restrict__ src,
	T * dst,
	int n,
	rowMajorOrderTag)
{

	const int tid = threadIdx.x + threadIdx.y * blockDim.x;

	for (int x = tid; x < n; x += blockDim.x * blockDim.y)
	{
		dst[x] = src[x];
	}
}

template <typename T>
__device__ __forceinline__
 void ctaCopyTable(
 	T const * __restrict__ src,
	T * dst,
	int n,
	int dim,
	int stride,
	colMajorOrderTag)
{

	int tid = threadIdx.x + threadIdx.y * blockDim.x;

	// TODO try to optimize copying of data
	for (int x = tid; x < n; x += blockDim.x * blockDim.y) {
		for (int d = 0; d < dim; ++d) {
			dst[d * stride + x] = src[d * stride + x];
		}
	}
}

template <typename T>
__device__ __forceinline__ 
void ctaCopyTable(
	T const * __restrict__ src,
	int srcStride,
	T *dst,
	int dstStride,
	int n,
	int dim)
{

	int tid = threadIdx.x + threadIdx.y * blockDim.x;

	// TODO try to optimize copying of data
	// if srcStride == 1 then read it as row major
	// the same to dstStride
	for (int x = tid; x < n; x += blockDim.x * blockDim.y) {
		for (int d = 0; d < dim; ++d) {
			dst[d * dstStride + x] = src[d * srcStride + x];
		}
	}
}

} // end namspace gpu
} // end namspace rd

#endif /* CTA_COPY_TABLE_CUH_ */
