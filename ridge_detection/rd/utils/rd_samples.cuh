/**
 * @file rd_sample.cuh
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

#ifndef RD_SAMPLE_CUH_
#define RD_SAMPLE_CUH_

#include "rd/utils/memory.h"

#include <helper_cuda.h>
#include <common_functions.h>

namespace rd 
{

template <
	typename 		T, 
	int 			DIM, 
	DataMemoryLayout 	order = ROW_MAJOR>
class rdDeviceSamples 
{

public:

	int size;
	// data in device memory
	T *dSamples;

	rdDeviceSamples() 
	{
		dSamples = nullptr;
	};
	rdDeviceSamples(int n) : size(n) 
	{
		checkCudaErrors(cudaMalloc((void**)& dSamples, n * DIM * sizeof(T*)));
	}

	virtual ~rdDeviceSamples() 
	{
		if (dSamples) 
		{
			checkCudaErrors(cudaFree(dSamples));
		}
	}

	void copyTo(T *dst) 
	{
		checkCudaErrors((cudaMemcpy(dst, dSamples,
				DIM * size * sizeof(T), cudaMemcpyDeviceToHost)));
	}

	void copyFrom(T *src) 
	{
		checkCudaErrors((cudaMemcpy(dSamples, src,
				DIM * size * sizeof(T), cudaMemcpyHostToDevice)));
	}

};

template <
	typename 	T, 
	int 		DIM>
class rdDeviceSamples<T, DIM, COL_MAJOR> 
{

public:

	int size;
	// contains pointers to device memory
	T *hDSamples[DIM];
	// data in device memory
	T **dSamples;

	rdDeviceSamples();
	rdDeviceSamples(int n) : size(n) 
	{
		checkCudaErrors(cudaMalloc((void **) &dSamples, DIM * sizeof(T*)));

		for (int k = 0; k < DIM; ++k) 
		{
			checkCudaErrors(cudaMalloc((void **) &hDSamples[k], n * sizeof(T)));
		}

		checkCudaErrors(cudaMemcpy(dSamples, hDSamples, DIM * sizeof(T*),
				cudaMemcpyHostToDevice));
	}

	virtual ~rdDeviceSamples() 
	{
		if (dSamples) checkCudaErrors(cudaFree(dSamples));
		if (hDSamples) 
		{
			for (T* ptr : hDSamples) 
			{
				checkCudaErrors(cudaFree(ptr));
			}
		}
	}

	void copy(rdDeviceSamples<T, DIM, COL_MAJOR> const &src)
	{
		if (src.size != size)
		{
			for (int k = 0; k < DIM; ++k)
				checkCudaErrors(cudaFree(hDSamples[k]));

			size = src.size;

			for (int k = 0; k < DIM; ++k) 
				checkCudaErrors(cudaMalloc((void **) &hDSamples[k], size * sizeof(T)));

			checkCudaErrors(cudaMemcpy(dSamples, hDSamples, DIM * sizeof(T*),
				cudaMemcpyHostToDevice));
		}

		T *ptr;
		for (int d = 0; d < DIM; ++d) 		
		{
			ptr = hDSamples[d];
			checkCudaErrors((cudaMemcpy(ptr, src.hDSamples[d], size * sizeof(T),
					cudaMemcpyDeviceToDevice)));
		}
	}

	void copyTo(T **dst) 
	{
		T *ptr;
		for (int d = 0; d < DIM; ++d) 
		{
			ptr = hDSamples[d];
			checkCudaErrors((cudaMemcpy(dst[d], ptr, size * sizeof(T),
					cudaMemcpyDeviceToHost)));
		}
	}

	void copyFrom(T **src) 
	{
		T *ptr;
		for (int d = 0; d < DIM; ++d) 
		{
			ptr = hDSamples[d];
			checkCudaErrors((cudaMemcpy(ptr, src[d], size * sizeof(T),
					cudaMemcpyHostToDevice)));
		}
	}

	/**
	 * @brief Transfer data from host to device.
	 * 
	 * @param src
	 * @param stride [in] - distance between coordinates
	 *
	 * @note Data at @p src should have column-major order.
	 */
	void copyFromContinuousData(T const * const src, int stride) 
	{
		T *ptr;
		for (int d = 0; d < DIM; ++d) 
		{
			ptr = hDSamples[d];
			checkCudaErrors((cudaMemcpy(ptr, src + d * stride, 
				size * sizeof(T), cudaMemcpyHostToDevice)));
		}
	}

	/**
	 * @brief Transfers data from Device to Host
	 * 
	 * @param dst [in|out]
	 *
	 * @note After this operation, data stored at @p dst will be 
	 * 		in column-major order.
	 */
	void copyToContinuousData(T *dst) 
	{
		T *ptr;
		for (int d = 0; d < DIM; ++d) 
		{
			ptr = hDSamples[d];
			checkCudaErrors((cudaMemcpy(dst + d * size, ptr,
				 size * sizeof(T), cudaMemcpyDeviceToHost)));
		}
	}

};


// template aliases c++11!

template <typename T, int DIM> using RowMajorDeviceSamples = rdDeviceSamples<T, DIM, ROW_MAJOR>;
template <typename T, int DIM> using ColMajorDeviceSamples = rdDeviceSamples<T, DIM, COL_MAJOR>;



}  // namespace rd

#endif /* RD_SAMPLE_CUH_ */
