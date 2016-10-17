/**
 * @file dev_utilities.cuh
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

#ifndef DEV_UTILITIES_CUH_
#define DEV_UTILITIES_CUH_

#include <type_traits>
// #ifndef RD_DEBUG
// #define NDEBUG      // for disabling assert macro
// #endif 
#include <assert.h>

#include "rd/gpu/util/data_order_traits.hpp"
#include "rd/utils/memory.h"

#include "cub/util_type.cuh"
#include "cub/util_debug.cuh"


namespace rd 
{
namespace gpu
{
	
/**
 * @brief Macro for error checking in device code.     
 */
#ifndef rdDevCheckCall
	#define rdDevCheckCall(e) if ( CubDebug((e)) ) { assert(0); }
#endif	

/**
 * Register modifier for pointer-types (for inlining PTX assembly)
 */
#if defined(_WIN64) || defined(__LP64__)
    #define __CUB_LP64__ 1
    // 64-bit register modifier for inlined asm
    #define _CUB_ASM_PTR_ "l"
    #define _CUB_ASM_PTR_SIZE_ "u64"
#else
    #define __CUB_LP64__ 0
    // 32-bit register modifier for inlined asm
    #define _CUB_ASM_PTR_ "r"
    #define _CUB_ASM_PTR_SIZE_ "u32"
#endif

template <class T1>
__device__ __forceinline__ T1 memFetch(T1 const *data, int i, int next)
{
    if (/*(threadIdx.x & 31) == 0 &&*/ next >= 0)
    {
        asm volatile ("prefetch.global.L2 [%0];" : : _CUB_ASM_PTR_(data + next));
    }
    return data[i];
}

/************************************************
 *			device runtime memory allocation utils
 ************************************************/

/**
 * @brief      Allocates pitched memory on the device
 *
 * Allocates at least @p width (in bytes) * @p height bytes of linear memory
 * on the device and returns in @p *devPtr a pointer to the allocated memory.
 * The function may pad the allocation to ensure that corresponding pointers
 * in any given row will continue to meet the alignment requirements for
 * coalescing as the address is updated from row to row. The pitch returned in
 * @p *pitch by ::cudaMallocPitch() is the width in bytes of the allocation.
 *
 * @param      devPtr  Pointer to allocated pitched device memory
 * @param      pitch   Pitch for allocation
 * @param[in]  width   Requested pitched allocation width (in bytes)
 * @param[in]  height  Requested pitched allocation height
 *
 * @tparam     T       Allocated data type.
 *
 * @return     cudaSuccess or cudaErrorMemoryAllocation
 */
template <typename T>
static __device__ __forceinline__ cudaError_t rdDevAllocPitchedMem(
  T      **devPtr,
  size_t  *pitch,
  size_t   width,
  size_t   height)
{
    const size_t ALIGN_BYTES   = 256;
    const size_t ALIGN_MASK    = ~(ALIGN_BYTES - 1);

    *pitch = (width + ALIGN_BYTES - 1) & ALIGN_MASK;
    *devPtr = (T*) new char[*pitch * height];

    if (*devPtr == nullptr)
    {
        return cudaErrorMemoryAllocation;
    }
    else
    {
        return cudaSuccess;
    }
}

/**
 * @brief      Allocates device memory for requested @p size, p@ dim dimensional points.
 *
 * @param      devPtr     Pointer to allocated device memory
 * @param      stride     Distance between consecutive coordinates (nr of elements in a row)
 * @param[in]  dim        The points dimension
 * @param[in]  size       The requested size of allocation
 *
 * @tparam     T          Data type of point single coordinate
 * @tparam     K          Data type for allocated memory region size description.
 *
 * @return     cudaSuccess or cudaErrorMemoryAllocation if allocation fails.
 */
template <
    typename T, 
    typename K,
    typename std::enable_if<
        std::is_integral<K>::value, int>::type = 0>
static __device__ __forceinline__ cudaError_t rdDevAllocMem(
    T **devPtr,
    K *stride,
    K dim,
    K size,
    cub::Int2Type<ROW_MAJOR>)
{
    *devPtr = new T[dim * size];
    /*
     * XXX: Ugly as hell... however, in this ridge-detection project, we're not aligning 
     * memory in ROW_MAJOR order at the moment.
     */
    *stride = dim;
    if (*devPtr == nullptr)
    {
        return cudaErrorMemoryAllocation;
    }
    else
    {
        return cudaSuccess;
    }
}

template <
    typename T, 
    typename K,
    typename std::enable_if<
        std::is_integral<K>::value, int>::type = 0>
static __device__ __forceinline__ cudaError_t rdDevAllocMem(
    T **devPtr,
    K *stride,
    K dim,
    K size,
    cub::Int2Type<COL_MAJOR>)
{
    size_t pitch;
    rdDevAllocPitchedMem(devPtr, &pitch, size_t(size * sizeof(T)), size_t(dim));
    *stride = pitch / sizeof(T);

    // #ifdef RD_DEBUG
    // printf("rdDevAllocMem<COL_MAJOR> dim: %d, size: %d, stride(is): %d, stride(should be) %d\n",
    //     dim, size, *stride, int(pitch / sizeof(T)));
    // #endif

    if (*devPtr == nullptr)
    {
        return cudaErrorMemoryAllocation;
    }
    else
    {
        return cudaSuccess;
    }
}

/*****************************************************************************************/
template <typename T>
__device__ __forceinline__ void printTable(T const * src, int size, const char * name,
		rowMajorOrderTag) {

	int tid = threadIdx.x + threadIdx.y * blockDim.x;

	if (threadIdx.x == 0) {
		printf("--------------------------------\n");
		printf("%s: \n", name);
	}
	__syncthreads();
	int step = 1;
	step *= blockDim.x * blockDim.y > 32 ? 32 : blockDim.x * blockDim.y;

	if (threadIdx.x < 32) {
		for (int x = tid; x < size; x += step) {
			printf("%s[%d]: %f\n", name, x, src[x]);
		}
	}

	__syncthreads();
	if (threadIdx.x == 0) {
		printf("--------------------------------\n");
	}
}

template <typename T>
__device__ __forceinline__ void printTable(T const * src, int n, int dim, int stride,
		const char * name, colMajorOrderTag ) {

	int tid = threadIdx.x + threadIdx.y * blockDim.x;

	if (threadIdx.x == 0) {
		printf("--------------------------------\n");
		printf("%s: \n", name);
	}
	__syncthreads();
	int step = 1;
	step *= blockDim.x * blockDim.y > 32 ? 32 : blockDim.x * blockDim.y;

	if (threadIdx.x < 32) {

		for (int x = tid; x < n; x += step) {
			for (int d = 0; d < dim; ++d) {
				printf("%s[%d]: %f\n", name, d * stride + x, src[d * stride + x]);
			}
		}
	}

	__syncthreads();
	if (threadIdx.x == 0) {
		printf("--------------------------------\n");
	}
}

/**
 * @brief Version supporting SOA data layout
 * @param src
 * @param n
 * @param dim
 * @param stride
 * @param name
 * @param
 */
template <typename T>
__device__ __forceinline__ void printTable(T const * const * const src, int n, int dim,
		const char * name) {

	int tid = threadIdx.x + threadIdx.y * blockDim.x;

	if (threadIdx.x == 0) {
		printf("--------------------------------\n");
		printf("%s: \n", name);
	}
	__syncthreads();
	int step = 1;
	step *= blockDim.x * blockDim.y > 32 ? 32 : blockDim.x * blockDim.y;

	if (threadIdx.x < 32) {

		for (int d = 0; d < dim; ++d) {
			for (int x = tid; x < n; x += step) {
				printf("%s[%d][%d]: %f\n", name, d, x, src[d][x]);
			}
		}
	}

	__syncthreads();
	if (threadIdx.x == 0) {
		printf("--------------------------------\n");
	}
}

} // end namespace gpu
} // end namespace rd
#endif /* DEV_UTILITIES_CUH_ */
