/**
 * @file samples_generator.cuh
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is conducted under the supervision of prof. dr hab. inż. Marek
 * Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */

#ifndef GPU_SAMPLES_SET_CUH_
#define GPU_SAMPLES_SET_CUH_

#include <helper_cuda.h>

#include "rd/utils/memory.h"
#include "rd/gpu/util/dev_math.cuh"
#include "cub/util_type.cuh"


template <typename T, rd::DataMemoryLayout SAMPLES_ORDER>
static __global__ void __circle(int n, T x0, T y0, T r, T sigma, T *samples) 
{

	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < n) 
	{
		// we want to have a sigma standard deviation of samples so we have to 
		// scale it down, because distance is growing with sqr(dim)
		sigma *= rsqrt(T(2.f));
		T phi;
		curandState localState;
		curand_init(1234, tid, 0, &localState);
		phi = rd::gpu::getUniformDist<T>(&localState);

		// scale phi to range 0 to 2pi
		phi *= T(2.f)*rd::gpu::getPi<T>();
		T x, y;
		x = x0 + r * rd::gpu::getCos<T>(phi);
		y = y0 + r * rd::gpu::getSin<T>(phi);
		typename rd::gpu::vector2<T>::type rn =
				rd::gpu::getNormalDist<typename rd::gpu::vector2<T>::type>(&localState);

		if (SAMPLES_ORDER == rd::ROW_MAJOR)
		{
			samples[tid*2  ] = fma(sigma, rn.x, x);
			samples[tid*2+1] = fma(sigma, rn.y, y);
		}
		else
		{
			// SAMPLES_ORDER == COL_MAJOR
			samples[tid    ] = fma(sigma, rn.x, x);
			samples[tid + n] = fma(sigma, rn.y, y);
		}
	}
}

/*****************************************************************************************/

template <typename T, rd::DataMemoryLayout SAMPLES_ORDER>
static __global__ void __sphere(int n, T x0, T y0, T z0, T r, T sigma, T *samples) 
{

	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < n) 
	{
		// we want to have a sigma standard deviation of samples so we have to 
		// scale it down, because distance is growing with sqr(dim)
		sigma *= rsqrt(T(3.f));
		T phi, theta;
		curandState localState;
		curand_init(1234, tid, 0, &localState);
		phi = rd::gpu::getUniformDist<T>(&localState);
		theta = rd::gpu::getUniformDist<T>(&localState);

		// scale phi to range 0 to pi
		phi *= rd::gpu::getPi<T>();
		// scale theta to range 0 to 2pi
		theta *= T(2.f)*rd::gpu::getPi<T>();
		T x, y, z;
		x = x0 + r * rd::gpu::getCos<T>(theta) * rd::gpu::getSin<T>(phi);
		y = y0 + r * rd::gpu::getSin<T>(theta) * rd::gpu::getSin<T>(phi);
		z = z0 + r * rd::gpu::getCos<T>(phi);
		typename rd::gpu::vector2<T>::type rn_xy =
				rd::gpu::getNormalDist<typename rd::gpu::vector2<T>::type>(&localState);
		T rn_z = rd::gpu::getNormalDist<T>(&localState);

		if (SAMPLES_ORDER ==  rd::ROW_MAJOR)
		{
			samples[tid*3  ] = fma(sigma, rn_xy.x, x);
			samples[tid*3+1] = fma(sigma, rn_xy.y, y);
			samples[tid*3+2] = fma(sigma, rn_z, z);
		}
		else
		{
			// SAMPLES_ORDER == rd::COL_MAJOR
			samples[tid    ] = fma(sigma, rn_xy.x, x);
			samples[tid+  n] = fma(sigma, rn_xy.y, y);
			samples[tid+2*n] = fma(sigma, rn_z, z);
		}
	}
}

/*****************************************************************************************/
template <typename T, rd::DataMemoryLayout SAMPLES_ORDER>
static __global__ void __spiral2D(int n, T a, T b, T sigma, T *samples) 
{

	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < n) 
	{
		// we want to have a sigma standard deviation of samples so we have to 
		// scale it down, because distance is growing with sqr(dim)
		sigma *= rsqrt(T(2.f));
		T phi;
		curandState localState;
		curand_init(1234, tid, 0, &localState);
		phi = rd::gpu::getUniformDist<T>(&localState);

		// scale phi to range 0 to a
		phi *= a;
		T s1, s2;
		s1 = b * phi * rd::gpu::getCos<T>(phi);
		s2 = b * phi * rd::gpu::getSin<T>(phi);
		typename rd::gpu::vector2<T>::type rn =
				rd::gpu::getNormalDist<typename rd::gpu::vector2<T>::type>(&localState);

		if (SAMPLES_ORDER == rd::ROW_MAJOR)
		{
			samples[tid*2  ] = fma(sigma, rn.x, s1);
			samples[tid*2+1] = fma(sigma, rn.y, s2);
		}
		else
		{
			// SAMPLES_ORDER == COL_MAJOR
			samples[tid    ] = fma(sigma, rn.x, s1);
			samples[tid + n] = fma(sigma, rn.y, s2);
		}
	}
}

/*****************************************************************************************/

template <typename T, rd::DataMemoryLayout SAMPLES_ORDER>
static __global__ void __spiral3D(int n, T a, T b, T sigma, T *samples) 
{

	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < n) 
	{
		// we want to have a sigma standard deviation of samples so we have to 
		// scale it down, because distance is growing with sqr(dim)
		sigma *= rsqrt(T(3.f));
		T phi;
		curandState localState;
		curand_init(1234, tid, 0, &localState);
		phi = rd::gpu::getUniformDist<T>(&localState);

		// scale phi to range 0 to a
		phi *= a;
		T s1, s2, s3;
		s1 = b * phi * rd::gpu::getCos<T>(phi);
		s2 = b * phi * rd::gpu::getSin<T>(phi);
		s3 = b * phi;
		typename rd::gpu::vector2<T>::type rn_xy =
				rd::gpu::getNormalDist<typename rd::gpu::vector2<T>::type>(&localState);
		T rn_z = rd::gpu::getNormalDist<T>(&localState);

		if (SAMPLES_ORDER ==  rd::ROW_MAJOR)
		{
			samples[tid*3  ] = fma(sigma, rn_xy.x, s1);
			samples[tid*3+1] = fma(sigma, rn_xy.y, s2);
			samples[tid*3+2] = fma(sigma, rn_z, s3);
		}
		else
		{
			// SAMPLES_ORDER == rd::COL_MAJOR
			samples[tid    ] = fma(sigma, rn_xy.x, s1);
			samples[tid+  n] = fma(sigma, rn_xy.y, s2);
			samples[tid+2*n] = fma(sigma, rn_z, s3);
		}
	}
}


/*****************************************************************************************/

template <typename T>
static __global__ void __segmentND(int n, int dim, T sigma, T length, T *samples,
	cub::Int2Type<rd::ROW_MAJOR>, int) 
{

	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	typename rd::gpu::vector2<T>::type phi;
	curandState localState;
	curand_init(1234, tid, 0, &localState);
	// we want to have a sigma standard deviation of samples so we have to 
	// scale it down, because distance is growing with sqr(dim)
	sigma *= rsqrt(T(dim));

	for (int x = tid*2; x < n * dim; x += 2*gridDim.x * blockDim.x) 
	{
		typename rd::gpu::vector2<T>::type rn_xy =
				rd::gpu::getNormalDist<typename rd::gpu::vector2<T>::type>(&localState);

		phi.x = rd::gpu::getUniformDist<T>(&localState);
		phi.y = rd::gpu::getUniformDist<T>(&localState);
		
		/**
		 * if we want to generate a noisy (normal distribution, with zero mean,
		 * and sigma standard deviation) samples along segment we need to perturb segment
		 * with all but x dimension set to zero
		 */
		phi.x = ((x % dim) == 0) ? phi.x * length : 0;
		phi.y = (((x+1) % dim) == 0) ? phi.y * length : 0;

		phi.x = fma(sigma, rn_xy.x, phi.x);
		phi.y = fma(sigma, rn_xy.y, phi.y);

		samples[x] = phi.x;
		if (x + 1 < n*dim) 
		{
			samples[x+1] = phi.y;
		}
	}
}

template <typename T>
static __global__ void __segmentND(int n, int dim, T sigma, T length, T *samples,
	cub::Int2Type<rd::COL_MAJOR>, int stride) 
{

	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	typename rd::gpu::vector2<T>::type rn_xy;
	typename rd::gpu::vector2<T>::type phi;
	curandState localState;
	curand_init(1234, tid, 0, &localState);
	// we want to have a sigma standard deviation of samples so we have to 
	// scale it down, because distance is growing with sqr(dim)
	sigma *= rsqrt(T(dim));

	// generate first two coordinates
	if (tid < n)
	{
		rn_xy = rd::gpu::getNormalDist<typename rd::gpu::vector2<T>::type>(&localState);
		phi.x = rd::gpu::getUniformDist<T>(&localState) * length;
		phi.x = fma(sigma, rn_xy.x, phi.x);
		samples[tid] = phi.x;

		if (dim >= 2) 
		{
		/**
		 * If we want to generate a noisy (normal distribution, with zero mean,
		 * and sigma standard deviation) samples along segment we need to perturb segment
		 * with all but x dimension set to zero
		 */
			phi.y = sigma * rn_xy.y;
			samples[stride + tid] = phi.y;
		}

		// generate rest of coordinates
		for (int d = 2; d < dim; d += 2)
		{

			rn_xy = rd::gpu::getNormalDist<typename rd::gpu::vector2<T>::type>(&localState);
			phi.x = sigma * rn_xy.x;
			samples[d * stride + tid] = phi.x;

			if (d + 1 < dim) 
			{
				phi.y = sigma * rn_xy.y;
				samples[(d + 1) * stride + tid] = phi.y;
			}
		}
	}
}

/*****************************************************************************************/

namespace rd 
{

namespace gpu 
{

template <typename T>
struct SamplesGenerator 
{

	static const int blockThreads = 512;

	template <rd::DataMemoryLayout SAMPLES_ORDER = rd::ROW_MAJOR>
	static void spiral2D(size_t n, T a, T b, T sigma, T *dSamples) 
	{

		dim3 dimBlock(blockThreads);
		dim3 dimGrid((n+blockThreads-1)/blockThreads);
		__spiral2D<T, SAMPLES_ORDER><<<dimGrid, dimBlock>>>(n, a, b, sigma, dSamples);
		checkCudaErrors(cudaGetLastError());
	}

	template <rd::DataMemoryLayout SAMPLES_ORDER = rd::ROW_MAJOR>
	static void spiral3D(size_t n, T a, T b, T sigma, T *dSamples) 
	{

		dim3 dimBlock(blockThreads);
		dim3 dimGrid((n+blockThreads-1)/blockThreads);
		__spiral3D<T, SAMPLES_ORDER><<<dimGrid, dimBlock>>>(n, a, b, sigma, dSamples);
		checkCudaErrors(cudaGetLastError());

	}

	static void spiral2D(size_t n, T a, T b, T sigma, T **dSamples) 
	{

		dim3 dimBlock(blockThreads);
		dim3 dimGrid((n+blockThreads-1)/blockThreads);
		__spiral2D<T><<<dimGrid, dimBlock>>>(n, a, b, sigma, dSamples);
		checkCudaErrors(cudaGetLastError());
	}

	static void spiral3D(size_t n, T a, T b, T sigma, T **dSamples) 
	{

		dim3 dimBlock(blockThreads);
		dim3 dimGrid((n+blockThreads-1)/blockThreads);
		__spiral3D<T><<<dimGrid, dimBlock>>>(n, a, b, sigma, dSamples);
		checkCudaErrors(cudaGetLastError());

	}

	template <rd::DataMemoryLayout SAMPLES_ORDER = rd::ROW_MAJOR>
	static void segmentND(size_t n, size_t dim, T sigma, T length, T *dSamples, 
		int stride = 0) 
	{
		dim3 dimBlock(blockThreads);
		dim3 dimGrid((n+blockThreads-1)/blockThreads);
		__segmentND<T><<<dimGrid, dimBlock>>>(n, dim, sigma, length, dSamples,
			cub::Int2Type<SAMPLES_ORDER>(), stride);
		checkCudaErrors(cudaGetLastError());

	}

	static void segmentND(size_t n, size_t dim, T sigma, T length, T **dSamples) 
	{

		dim3 dimBlock(blockThreads);
		dim3 dimGrid((n+blockThreads-1)/blockThreads);
		__segmentND<T><<<dimGrid, dimBlock>>>(n, dim, sigma, length, dSamples);
		checkCudaErrors(cudaGetLastError());

	}

	template <rd::DataMemoryLayout SAMPLES_ORDER = rd::ROW_MAJOR>
	static void circle(size_t n, T x0, T y0, T r, T sigma, T *dSamples) 
	{

		dim3 dimBlock(blockThreads);
		dim3 dimGrid((n+blockThreads-1)/blockThreads);
		__circle<T, SAMPLES_ORDER><<<dimGrid, dimBlock>>>(n, x0, y0, r, sigma, dSamples);
		checkCudaErrors(cudaGetLastError());
	}

	template <rd::DataMemoryLayout SAMPLES_ORDER = rd::ROW_MAJOR>
	static void sphere(size_t n, T x0, T y0, T z0, T r, T sigma, T *dSamples) 
	{

		dim3 dimBlock(blockThreads);
		dim3 dimGrid((n+blockThreads-1)/blockThreads);
		__sphere<T, SAMPLES_ORDER><<<dimGrid, dimBlock>>>(n, x0, y0, z0, r, sigma, dSamples);
		checkCudaErrors(cudaGetLastError());
	}

};

}  // namespace gpu
}  // namespace rd

#endif /* GPU_SAMPLES_SET_CUH_ */
