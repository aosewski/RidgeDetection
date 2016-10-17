/**
 * @file decimate_brute_force.cuh
 * @author Adam Rogowiec
 */

#ifndef DECIMATE_BRUTE_FORCE_CUH_
#define DECIMATE_BRUTE_FORCE_CUH_

#include "rd/utils/memory.h"

#include "rd/gpu/block/cta_count_neighbour_points.cuh"
#include "rd/gpu/warp/warp_functions.cuh"
#include "rd/gpu/util/dev_math.cuh"
#include "rd/gpu/util/data_order_traits.hpp"

#include "cub/util_type.cuh"

/*****************************************************************************************/

#if defined(__cplusplus) && defined(__CUDACC__)

/**
 * @brief Kernel version for SoA data layout
 *
 * @param S
 * @param ns
 * @param r
 * @param dim
 * @param
 */
template <
    int         BLOCK_SIZE,
	typename 	T>
__global__ void __decimate_kernel_v1(
	T * const * const S,
	int *ns,
	T r,
	int dim)
{
	T r2 = r*r;
	int l = 0;
	int nss = *ns;
	int neighbours = 0;

	while (l != nss && nss > 3)
	{
		l = nss;
		for (int i = 0; i < nss;)
		{
			neighbours = rd::gpu::ctaCountNeighbouringPoints<T, BLOCK_SIZE>(
                S, nss, S, i, dim, r2, 4);
			__syncthreads();
			if (neighbours == 1)
			{
				nss--;
				/*
				 * I know that it is run with only one block of threads.
				 */
				// round up to closest multiply of BLOCK_SIZE
				T tmp;
				const int count = (nss-i);
				int k = (count + BLOCK_SIZE ) / BLOCK_SIZE * BLOCK_SIZE;
				for (int d = 0; d < dim; ++d)
				{
					for (int x = threadIdx.x; x < k; x += BLOCK_SIZE)
					{
						if (x < count)
							tmp = S[d][i + 1 + x];
						__syncthreads();
						if (x < count)
							S[d][i + x] = tmp;
					}
				}
				
				if (nss < 3)
					break;
				continue;
			}
	
			neighbours = rd::gpu::ctaCountNeighbouringPoints<T, BLOCK_SIZE>(
                S, nss, S, i, dim, T(4)*r2, 3);
			__syncthreads();
			if (neighbours == 0)
			{
				nss--;

				/*
				 * I know that it is run with only one block of threads.
				 */
				// round up to closest multiply of BLOCK_SIZE
				T tmp;
				const int count = (nss-i);
				int k = (count + BLOCK_SIZE ) / BLOCK_SIZE * BLOCK_SIZE;
				for (int d = 0; d < dim; ++d)
				{
					for (int x = threadIdx.x; x < k; x += BLOCK_SIZE)
					{
						if (x < count)
							tmp = S[d][i + 1 + x];
						__syncthreads();
						if (x < count)
							S[d][i + x] = tmp;
					}
				}
				if (nss < 3)
					break;
			} else i++;
		}
	}

	if (threadIdx.x == 0)
		*ns = nss;
}


namespace rd
{
namespace gpu
{
namespace bruteForce
{

template <int _BLOCK_SIZE>
struct BlockDecimatePolicy
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
    typename            T,
    int                 DIM,
    int                 BLOCK_SIZE,
    DataMemoryLayout    MEM_LAYOUT  = ROW_MAJOR>
class BlockDecimate
{

private:

    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /// Decimate helper
    template <DataMemoryLayout _MEM_LAYOUT, int DUMMY>
    struct DecimateInternal;

    template <int DUMMY>
    struct DecimateInternal<ROW_MAJOR, DUMMY>
    {
        /// Constructor
        __device__ __forceinline__ DecimateInternal()
        {}

        /**
         * @param S
         * @param ns
         * @param r
         * @param <no name>     Stride in @p S data. Unused because this specialization 
         *                      assumes stride equal 1. Just for API consistency.
         */
        __device__ __forceinline__ void decimate(
            T *S,
            int *ns,
            T r,
            int )
        {

            T r2 = r*r;
            int l = 0;
            int nss = *ns;
            int neighbours = 0;

            while (l != nss && nss > 3)
            {
                l = nss;
                for (int i = 0; i < nss;)
                {
                    neighbours = ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(
                                    S, nss, S + i*DIM, r2, 4, rowMajorOrderTag());
                    __syncthreads();
                    if (neighbours >= 4)
                    {
                        nss--;

                        /*
                         * I know that it is run with only one block of threads.
                         */
                        // round up to closest multiply of BLOCK_SIZE
                        T * src = S + (i+1)*DIM;
                        T * dst = S + i * DIM;
                        T tmp;
                        const int count = (nss-i)*DIM;
                        int k = (count + BLOCK_SIZE ) / BLOCK_SIZE * BLOCK_SIZE;
                        for (int x = threadIdx.x; x < k; x += BLOCK_SIZE)
                        {
                            if (x < count)
                                tmp = src[x];
                            __syncthreads();
                            if (x < count)
                                dst[x] = tmp;
                        }

                        if (nss < 3)
                            break;
                        continue;
                    }

                    neighbours = ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(S, nss, 
                        S + i*DIM, 4.f*r2, 3, rowMajorOrderTag());
                    __syncthreads();
                    if (neighbours <= 2)
                    {
                        nss--;
                        /*
                         * I know that it is run with only one block of threads.
                         */
                        // round up to closest multiply of BLOCK_SIZE
                        T * src = S + (i+1)*DIM;
                        T * dst = S + i * DIM;
                        T tmp;
                        const int count = (nss-i)*DIM;
                        int k = (count + BLOCK_SIZE ) / BLOCK_SIZE * BLOCK_SIZE;
                        for (int x = threadIdx.x; x < k; x += BLOCK_SIZE)
                        {
                            if (x < count)
                                tmp = src[x];
                            __syncthreads();
                            if (x < count)
                                dst[x] = tmp;
                        }

                        if (nss < 3)
                            break;
                    } else i++;
                }
            }
            if (threadIdx.x == 0)
                *ns = nss;
        }
    };

    template <int DUMMY>
    struct DecimateInternal<COL_MAJOR, DUMMY>
    {
        /// Constructor
        __device__ __forceinline__ DecimateInternal()
        {}

        __device__ __forceinline__ void decimate(
            T *S,
            int *ns,
            T r,
            int stride)
        {
            T r2 = r*r;
            int l = 0;
            int nss = *ns;
            int neighbours = 0;

            while (l != nss && nss > 3)
            {
                l = nss;
                for (int i = 0; i < nss;)
                {
                    neighbours = ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(S, nss, stride, S + i,
                            stride, r2, 4, colMajorOrderTag());
                    __syncthreads();        
                    if (neighbours >= 4)
                    {
                        nss--;
                        /*
                         * I know that it is run with only one block of threads.
                         */
                        // move the rest of data in array
                        // round up to closest multiply of BLOCK_SIZE
                        T * src = S + (i+1);
                        T * dst = S + i;
                        T tmp;
                        const int count = (nss-i);
                        int k = (count + BLOCK_SIZE ) / BLOCK_SIZE * BLOCK_SIZE;
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                        {
                            for (int x = threadIdx.x; x < k; x += BLOCK_SIZE)
                            {
                                if (x < count)
                                    tmp = src[d * stride + x];
                                __syncthreads();
                                if (x < count)
                                    dst[d * stride + x] = tmp;
                            }
                        }
                        
                        if (nss < 3)
                            break;
                        continue;
                    }

                    neighbours = ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(S, nss, stride, S + i,
                            stride, 4.f*r2, 3, colMajorOrderTag());
                    __syncthreads();
                    if (neighbours <= 2)
                    {
                        nss--;
                        /*
                         * I know that it is run with only one block of threads.
                         */
                        // move the rest of data in array
                        // round up to closest multiply of BLOCK_SIZE
                        T * src = S + (i+1);
                        T * dst = S + i;
                        T tmp;
                        const int count = (nss-i);
                        int k = (count + BLOCK_SIZE ) / BLOCK_SIZE * BLOCK_SIZE;
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                        {
                            for (int x = threadIdx.x; x < k; x += BLOCK_SIZE)
                            {
                                if (x < count)
                                    tmp = src[d * stride + x];
                                __syncthreads();
                                if (x < count)
                                    dst[d * stride + x] = tmp;
                            }
                        }
                        
                        if (nss < 3)
                            break;
                    } else i++;
                }
            }
            if (threadIdx.x == 0)
                *ns = nss;
        }
    };

    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Internal choose implementation to use
    typedef DecimateInternal<MEM_LAYOUT, 0> InternalDecimate;

public:

    /******************************************************************************
     * Collective constructors
     ******************************************************************************/

     __device__ __forceinline__ BlockDecimate()
     {}

    /******************************************************************************
     * Decimate
     ******************************************************************************/

     __device__ __forceinline__ void decimate(
        T *S,
        int *ns,
        T r,
        int stride)
     {
        InternalDecimate().decimate(S, ns, r, stride);
     }

};


}   // end namespace bruteForce
}   // end namespace gpu
}   // end namespace rd

#endif // defined(__cplusplus) && defined(__CUDACC__)

#endif /* DECIMATE_BRUTE_FORCE_CUH_ */
