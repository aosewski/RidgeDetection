/**
 * @file dev_memcpy.cuh
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


#ifndef DEV_MEMCPY_CUH
#define DEV_MEMCPY_CUH

#include "rd/utils/memory.h"
#include "rd/utils/utilities.hpp"
#include "cub/util_type.cuh"

#include <cstddef>
#include <helper_cuda.h>
#ifdef RD_USE_OPENMP
#include <omp.h>
#endif

namespace rd 
{

namespace gpu
{

namespace detail
{

template <
    DataMemoryLayout    DST,
    DataMemoryLayout    SRC,
    typename            T>
struct RdTransposeEngine
{
    static void transpose(
        T * src, 
        int width, 
        int height,
        size_t dstStride,
        size_t srcStride)
    {
        T * aux = new T[height * srcStride];
        rd::transposeMatrix_omp(src, aux, width, height, srcStride, dstStride);
        rd::copyTable_omp(aux, src, height * srcStride);
        delete[] aux;
    } 
};

template <
    DataMemoryLayout    LAYOUT,
    typename            T>
struct RdTransposeEngine<LAYOUT, LAYOUT, T>
{
    static void transpose(
        T * src, 
        int width, 
        int height,
        size_t dstStride,
        size_t srcStride)
    {
    }
};


template <
    DataMemoryLayout    DST,
    DataMemoryLayout    SRC,
    typename            T>
struct RdMemcpyEnginge
{
    static void doMemcpy(
        T * dst,
        T const * src,
        size_t width,
        size_t height,
        size_t dstStride,          /// number of elements in row
        size_t srcStride,
        cub::Int2Type<cudaMemcpyHostToDevice>)
    {
        T * copyBuff = new T[height * srcStride];
        copyTable(src, copyBuff, height * srcStride);
        RdTransposeEngine<DST, SRC, T>::transpose(copyBuff, width, height, dstStride, srcStride);
        checkCudaErrors(cudaMemcpy(dst, copyBuff, height * srcStride * sizeof(T), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
        delete[] copyBuff;
    }

    static void doMemcpy(
        T * dst, 
        T const * src, 
        size_t width,
        size_t height,
        size_t dstStride,          /// number of elements in row
        size_t srcStride,
        cub::Int2Type<cudaMemcpyDeviceToHost>)
    {
        checkCudaErrors(cudaMemcpy(dst, src, height * srcStride * sizeof(T), 
            cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());
        RdTransposeEngine<DST, SRC, T>::transpose(dst, width, height, dstStride, srcStride);
    }

    static void doMemcpy2D(
        T * dst, 
        T const * src, 
        size_t width,
        size_t height,
        size_t dpitch,
        size_t spitch, 
        cub::Int2Type<cudaMemcpyHostToDevice>)
    {
        T * copyBuff = new T[height * spitch / sizeof(T)];
        copyTable_omp(src, copyBuff, height * spitch / sizeof(T));
        RdTransposeEngine<DST, SRC, T>::transpose(copyBuff, width, height, height, spitch/sizeof(T));
        checkCudaErrors(cudaMemcpy2D(dst, dpitch, copyBuff, height * sizeof(T), height * sizeof(T), 
            width, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
        delete[] copyBuff;
    }

    static void doMemcpy2D(
        T * dst, 
        T const * src, 
        size_t width,
        size_t height,
        size_t dpitch,
        size_t spitch,
        cub::Int2Type<cudaMemcpyDeviceToHost>)
    {
    	T * auxBuff = new T[height * width];
        checkCudaErrors(cudaMemcpy2D(auxBuff, width * sizeof(T), src, spitch, width * sizeof(T), 
            height, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());
        RdTransposeEngine<DST, SRC, T>::transpose(auxBuff, width, height, dpitch/sizeof(T), width);
        copyTable_omp(auxBuff, dst, height * width);
    }
};

template <
    DataMemoryLayout    LAYOUT,
    typename            T>
struct RdMemcpyEnginge<LAYOUT, LAYOUT, T>
{
    static void doMemcpy(
        T * dst, 
        T const * src, 
        size_t ,
        size_t height,
        size_t , 
        size_t srcStride,
        cub::Int2Type<cudaMemcpyDeviceToDevice>)
    {
        checkCudaErrors(cudaMemcpy(dst, src, height * srcStride * sizeof(T), 
            cudaMemcpyDeviceToDevice));
    }

    static void doMemcpy(
        T * dst, 
        T const * src, 
        size_t ,
        size_t height, 
        size_t , 
        size_t srcStride,
        cub::Int2Type<cudaMemcpyHostToDevice>)
    {
        checkCudaErrors(cudaMemcpy(dst, src, height * srcStride * sizeof(T), cudaMemcpyHostToDevice));
    }

    static void doMemcpy(
        T * dst, 
        T const * src, 
        size_t ,
        size_t height,
        size_t ,
        size_t srcStride,
        cub::Int2Type<cudaMemcpyDeviceToHost>)
    {
        checkCudaErrors(cudaMemcpy(dst, src, height * srcStride * sizeof(T), cudaMemcpyDeviceToHost));
    }

    static void doMemcpy2D(
        T * dst, 
        T const * src, 
        size_t width,
        size_t height,
        size_t dpitch,
        size_t spitch, 
        cub::Int2Type<cudaMemcpyDeviceToDevice>)
    {
        checkCudaErrors(cudaMemcpy2D(dst, dpitch, src, spitch, width * sizeof(T), height, 
            cudaMemcpyDeviceToDevice));
    }

    static void doMemcpy2D(
        T * dst, 
        T const * src, 
        size_t width,
        size_t height,
        size_t dpitch,
        size_t spitch, 
        cub::Int2Type<cudaMemcpyHostToDevice>)
    {
        checkCudaErrors(cudaMemcpy2D(dst, dpitch, src, spitch, width * sizeof(T), height, 
            cudaMemcpyHostToDevice));
    }

    static void doMemcpy2D(
        T * dst, 
        T const * src, 
        size_t width,
        size_t height,
        size_t dpitch,
        size_t spitch,
        cub::Int2Type<cudaMemcpyDeviceToHost>)
    {
        checkCudaErrors(cudaMemcpy2D(dst, dpitch, src, spitch, width * sizeof(T), height, 
            cudaMemcpyDeviceToHost));
    }
};

} // end namespace detail

/**
 * @note stride is a number of elements in matrix row
 *
 */
template <
    DataMemoryLayout    DST,
    DataMemoryLayout    SRC,
    cudaMemcpyKind      KIND,
    typename            T>
void rdMemcpy(
    T * dst, 
    T const * src, 
    size_t width,
    size_t height,
    size_t dstStride,
    size_t srcStride)
{
    if (width * height > 0)
    {
        detail::RdMemcpyEnginge<DST, SRC, T>::doMemcpy(
            dst, src, width, height, dstStride, srcStride, cub::Int2Type<KIND>());
    }
}

/**
 * param[in] width Width of input matrix
 */
template <
    DataMemoryLayout    DST,
    DataMemoryLayout    SRC,
    cudaMemcpyKind      KIND,
    typename            T>
void rdMemcpy2D(
    T * dst, 
    T const * src, 
    size_t width,
    size_t height,
    size_t dpitch,
    size_t spitch)
{
	if (width * height > 0)
	{
		detail::RdMemcpyEnginge<DST, SRC, T>::doMemcpy2D(
			dst, src, width, height, dpitch, spitch, cub::Int2Type<KIND>());
	}
}

}   // end namespace gpu
}   // end namespace rd

#endif // DEV_MEMCPY_CUH
