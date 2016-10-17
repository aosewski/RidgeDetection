/**
 * @file choose.cuh
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

#ifndef CHOOSE_CUH_
#define CHOOSE_CUH_


#include "rd/gpu/block/cta_count_neighbour_points.cuh"
#include "rd/gpu/block/cta_copy_table.cuh"
#include "rd/gpu/util/data_order_traits.hpp"
#include "rd/utils/memory.h"

#include "cub/util_type.cuh"

/*****************************************************************************************/
/**
 *	\brief Chose initial set S of the path's nodes.
 *
 *  @paragraph The function choses subset of P_ set of points, where each two
 *	of them are R-separated. This means that thera are
 *	no two different points closer than R.
 *
 *	@note Points are chosen in the order they appear in samples set P.
 */
template <typename T, int BLOCK_SIZE>
static __global__
void __choose_kernel_v1(T const * __restrict__ P,
						T *S,
						int np,
						T r,
						int *ns,
						int dim,
						rd::gpu::rowMajorOrderTag ordTag)
{

	int count = 1;
	/*
	 * Get the first point from set and start choosing from it.
	 */
	T rSqr = r * r;
	rd::gpu::ctaCopyTable(P, S, dim, ordTag);
	P += dim;

	for (int k = 1; k < np; ++k)
	{
		if (!rd::gpu::ctaCountNeighbouringPoints<T, BLOCK_SIZE>(S, count, P, dim, rSqr,
				1, ordTag))
		{
			rd::gpu::ctaCopyTable(P, S + dim * count++, dim, ordTag);
		}
		P += dim;
	}
	if (threadIdx.x == 0)
		*ns = count;
}

/**
 * @brief This version assumes that both P and S have column major order
 */
template <typename T, int BLOCK_SIZE>
static __global__
void __choose_kernel_v1(T const * __restrict__ P,
						T *S,
						int np,
						T r,
						int *ns,
						int dim,
						rd::gpu::colMajorOrderTag ordTag)
{

	int count = 1;
	/*
	 * Get the first point from set and start choosing from it.
	 */
	rd::gpu::ctaCopyTable(P, S, 1, dim, np, ordTag);
	P++;
	T rSqr = r * r;

	for (int k = 1; k < np; ++k)
	{
		if (!rd::gpu::ctaCountNeighbouringPoints<T, BLOCK_SIZE>(
				S, count, np, P, dim, rSqr, 1, ordTag))
		{
			rd::gpu::ctaCopyTable(P, S + count++, 1, dim, np, ordTag);
		}
		P++;
	}
	if (threadIdx.x == 0)
		*ns = count;
}


/**
 * @brief This version assumes that P has column major order and
 * S has row major order data layout.
 */
template <typename T, int BLOCK_SIZE>
static __global__
void __choose_kernel_v1(T const * __restrict__ P,
						T *S,
						int np,
						T r,
						int *ns,
						int dim)
{

	int count = 1;
	/*
	 * Get the first point from set and start choosing from it.
	 */
	rd::gpu::ctaCopyTable(P, np, S, 1, 1, dim);
	T rSqr = r * r;
	P++;

	for (int k = 1; k < np; ++k) {
		if (!rd::gpu::ctaCountNeighbouringPoints<T, BLOCK_SIZE>(S, count, 1, P, np,
				dim, rSqr, 1))
		{
			rd::gpu::ctaCopyTable(P, np, S + count++ * dim, 1, 1, dim);
		}
		P++;
	}
	if (threadIdx.x == 0)
		*ns = count;
}


/*****************************************************************************************/

/**
 *	\brief Chose initial set S of the path's nodes.
 *
 *  @paragraph The function choses subset of P_ set of points, where each two
 *	of them are R-separated. This means that thera are
 *	no two different points closer than R.
 *
 *	@note Points are choosen in the order they appear in samples set P.
 *	@note This function assumes SoA data layout!
 */
template <typename T, int BLOCK_SIZE>
static __global__
void __choose_kernel_v1(T const * const * const __restrict__ P,
						T * const * const S,
						int np,
						T r,
						int *ns,
						int dim)
{

	int count = 0;
	/*
	 * Get the first point from set and start choosing from it.
	 */
	T rSqr = r * r;
	rd::gpu::ctaCopyTable(P, 0, S, count++, 1, dim);

	for (int k = 1; k < np; ++k)
	{
		if (!rd::gpu::ctaCountNeighbouringPoints<T, BLOCK_SIZE>(S, count, P, k, dim, rSqr, 1))
		{
			rd::gpu::ctaCopyTable(P, k, S, count++, 1, dim);
		}
	}
	if (threadIdx.x == 0)
		*ns = count;
}


/*****************************************************************************************/

namespace rd 
{
namespace gpu
{
namespace bruteForce
{

template <int _BLOCK_SIZE>
struct BlockChoosePolicy
{
	enum
	{
		BLOCK_SIZE = _BLOCK_SIZE
	};
};

template <
	typename 			T,
	int 				DIM,
	int 				BLOCK_SIZE,
	DataMemoryLayout	INPUT_MEM_LAYOUT 	= COL_MAJOR,
	DataMemoryLayout	OUTPUT_MEM_LAYOUT	= ROW_MAJOR>
class BlockChoose
{

private:
	
    /******************************************************************************
     * Constants and typed definitions
     ******************************************************************************/

    /// Constants
    enum
    {
    	INSERT_PADDING		= (INPUT_MEM_LAYOUT == COL_MAJOR) ? 1 : 0,
    	PADDING_ITEMS		= (INSERT_PADDING) ? 1 : 0,
    	TMP_STORAGE_STRIDE	= BLOCK_SIZE + PADDING_ITEMS
    };

    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

	/// Choose helper
	template <
     	DataMemoryLayout _IN_MEM_LAYOUT,
     	DataMemoryLayout _OUT_MEM_LAYOUT,
     	int DUMMY>
	struct ChooseInternal;

	template <int DUMMY>
	struct ChooseInternal<ROW_MAJOR, ROW_MAJOR, DUMMY>
	{
        /// Shared memory storage layout type
		typedef T _TempStorage[BLOCK_SIZE * DIM];

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : cub::Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &tempStorage;

        /// thread private point buffer
	    T buff[DIM];

	    __device__ __forceinline__ ChooseInternal(
    		TempStorage &temp_storage) : tempStorage(temp_storage.Alias())
	    {}

     	__device__ __forceinline__ void choose(
			T const * __restrict__ 	P,
			T *	S,
			int np,
			int *ns,
			T r,
			int, 
			int)
		{
			#pragma unroll
			for (int d = 0; d < DIM; ++d)
			{
				buff[d] = 0;
			}

			int step = (np < BLOCK_SIZE) ? np : BLOCK_SIZE;
			#pragma unroll
			for (int d = 0; d < DIM; ++d)
			{
				buff[d] = (threadIdx.x < np) ? P[d * step + threadIdx.x] : 0;
			}

			int count = 0;
			int m = (np + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
			int m2 = m - BLOCK_SIZE;
			int fullTiles = m2 / BLOCK_SIZE;
			T rSqr = r * r;
			int x, t, neighbours = 0;

			for (x = threadIdx.x, t = 0; t < fullTiles; x += BLOCK_SIZE, t++)
			{
				__syncthreads();
				#pragma unroll
				for (int d = 0; d < DIM; ++d)
				{
					tempStorage[d * BLOCK_SIZE + threadIdx.x] = buff[d];
				}

				__syncthreads();
				// start load next tile
				// handle spacial case: calc step when loading last partial tile
				int nextX = x + BLOCK_SIZE;
				int nextTileOffset = (t + 1) * BLOCK_SIZE * DIM;
				step = (nextX >= m2) ? np - m2 : BLOCK_SIZE;
				#pragma unroll
				for (int d = 0; d < DIM; ++d)
				{
					buff[d] = (nextX < np) ? P[nextTileOffset + d * step + threadIdx.x] : 0;
				}

				for (int k = 0; k < BLOCK_SIZE; ++k)
				{
					neighbours = ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(
							S, count, tempStorage+k*DIM, rSqr, 1, rowMajorOrderTag());
					if (neighbours == 0)
					{
						ctaCopyTable(tempStorage + k*DIM, S + DIM * count++, DIM, 
							rowMajorOrderTag());
					}
				}
			}
			
			// process last portion of data
			__syncthreads();
			if (x < np)
			{
				#pragma unroll
				for (int d = 0; d < DIM; ++d)
				{
					tempStorage[d * step + threadIdx.x] = buff[d];
				}
			}
			__syncthreads();

			for (int k = 0; k < np - m2; ++k)
			{
				neighbours = ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(
						S, count, tempStorage+k*DIM, rSqr, 1, rowMajorOrderTag());
				if (neighbours == 0)
				{
					ctaCopyTable(tempStorage + k*DIM, S + DIM * count++, DIM, 
						rowMajorOrderTag());
				}
			}

			if (threadIdx.x == 0)
			{
				*ns = count;
			}
		}
    };


	template <int DUMMY>
	struct ChooseInternal<COL_MAJOR, COL_MAJOR, DUMMY>
	{
        /// Shared memory storage layout type
		typedef T _TempStorage[TMP_STORAGE_STRIDE * DIM];

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : cub::Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &tempStorage;

        /// thread private point buffer
	    T buff[DIM];

	    __device__ __forceinline__ ChooseInternal(
    		TempStorage &temp_storage) : tempStorage(temp_storage.Alias())
	    {}

     	__device__ __forceinline__ void choose(
			T const * __restrict__ 	P,
			T *	S,
			int np,
			int *ns,
			T r,
			int pStride,
			int sStride)
		{
			#pragma unroll
			for (int d = 0; d < DIM; ++d)
			{
				buff[d] = 0;
			}

			#pragma unroll
			for (int d = 0; d < DIM; ++d)
			{
				buff[d] = (threadIdx.x < np) ? P[d * pStride + threadIdx.x] : 0;
			}

			int count = 0;
			int m = (np + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
			int m2 = m - BLOCK_SIZE;
			T rSqr = r * r;
			int x, neighbours = 0;

			for (x = threadIdx.x; x < m2; x += BLOCK_SIZE)
			{
				__syncthreads();
				#pragma unroll
				for (int d = 0; d < DIM; ++d)
				{
					tempStorage[d * TMP_STORAGE_STRIDE + threadIdx.x] = buff[d];
				}

				__syncthreads();
				// load next tile
				#pragma unroll
				for (int d = 0; d < DIM; ++d)
				{
					buff[d] = (x + BLOCK_SIZE < np) ? P[d * pStride + x + BLOCK_SIZE] : 0;
				}

				for (int k = 0; k < BLOCK_SIZE; ++k)
				{
					neighbours = ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(
							S, count, sStride, tempStorage+k, TMP_STORAGE_STRIDE, rSqr, 1,
							colMajorOrderTag());
					if (neighbours == 0)
					{
						if (threadIdx.x < DIM)
						{
							S[sStride * threadIdx.x + count] = tempStorage[k + 
								threadIdx.x * TMP_STORAGE_STRIDE];
						}
						count++;
					}
				}
			}
			
			// process last portion of data
			__syncthreads();
			if (x < np)
			{
				#pragma unroll
				for (int d = 0; d < DIM; ++d)
				{
					tempStorage[d * TMP_STORAGE_STRIDE + threadIdx.x] = buff[d];
				}
			}
			__syncthreads();

			for (int k = 0; k < BLOCK_SIZE && m2 + k < np; ++k)
			{
				neighbours = ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(
						S, count, sStride, tempStorage+k, TMP_STORAGE_STRIDE, rSqr, 1,
						colMajorOrderTag());
				if (neighbours == 0)
				{
					if (threadIdx.x < DIM)
					{
						S[sStride * threadIdx.x + count] = tempStorage[k + 
							threadIdx.x * TMP_STORAGE_STRIDE];
					}
					count++;
				}
			}

			if (threadIdx.x == 0)
			{
				*ns = count;
			}
		}
	};

	template <int DUMMY>
	struct ChooseInternal<COL_MAJOR, ROW_MAJOR, DUMMY>
	{

        /// Shared memory storage layout type
		typedef T _TempStorage[(TMP_STORAGE_STRIDE) * DIM];

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : cub::Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &tempStorage;

        /// thread private point buffer
	    T buff[DIM];

	    __device__ __forceinline__ ChooseInternal(
    		TempStorage &temp_storage) : tempStorage(temp_storage.Alias())
	    {}

     	__device__ __forceinline__ void choose(
			T const * __restrict__ 	P,
			T *	S,
			int np,
			int *ns,
			T r,
			int pStride,
			int)
		{
			#pragma unroll
			for (int d = 0; d < DIM; ++d)
			{
				buff[d] = 0;
			}

			#pragma unroll
			for (int d = 0; d < DIM; ++d)
				buff[d] = (threadIdx.x < np) ? P[d * pStride + threadIdx.x] : 0;

			int count = 0;
			int m = (np + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
			int m2 = m - BLOCK_SIZE;
			T rSqr = r * r;
			int x, neighbours = 0;

			for (x = threadIdx.x; x < m2; x += BLOCK_SIZE)
			{
				__syncthreads();
				#pragma unroll
				for (int d = 0; d < DIM; ++d)
				{
					tempStorage[d * TMP_STORAGE_STRIDE + threadIdx.x] = buff[d];
				}

				__syncthreads();
				// load next tile
				#pragma unroll
				for (int d = 0; d < DIM; ++d)
				{
					buff[d] = (x + BLOCK_SIZE < np) ? P[d * pStride + x + BLOCK_SIZE] : 0;
				}

				for (int k = 0; k < BLOCK_SIZE; ++k)
				{
					neighbours = ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(
							S, count, tempStorage+k, TMP_STORAGE_STRIDE, rSqr, 1, rowMajorOrderTag());
					if (neighbours == 0)
					{
						if (threadIdx.x < DIM)
						{
							S[DIM * count + threadIdx.x] = tempStorage[k + 
								threadIdx.x * TMP_STORAGE_STRIDE];
						}
						count++;
					}
				}
			}
			
			// process last portion of data
			__syncthreads();
			if (x < np)
			{
				#pragma unroll
				for (int d = 0; d < DIM; ++d)
				{
					tempStorage[d * TMP_STORAGE_STRIDE + threadIdx.x] = buff[d];
				}
			}
			__syncthreads();

			for (int k = 0; k < BLOCK_SIZE && m2 + k < np; ++k)
			{
				neighbours = ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(
						S, count, tempStorage+k, TMP_STORAGE_STRIDE, rSqr, 1, rowMajorOrderTag());
				if (neighbours == 0)
				{
					if (threadIdx.x < DIM)
					{
						S[DIM * count + threadIdx.x] = tempStorage[k + 
							threadIdx.x * TMP_STORAGE_STRIDE];
					}
					count++;
				}
			}

			if (threadIdx.x == 0)
			{
				*ns = count;
			}
		}
	};

    /******************************************************************************
     * Type definitions
     ******************************************************************************/

	/// Internal choose implementation to use
	typedef ChooseInternal<INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0> InternalChoose;

    /// Shared memory storage layout type
    typedef typename InternalChoose::TempStorage _TempStorage;

    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Thread reference to shared storage
    _TempStorage &tempStorage;

    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }

public:

    /// \smemstorage{Choose}
    struct TempStorage : cub::Uninitialized<_TempStorage> {};

    /******************************************************************************
     * Collective constructors
     ******************************************************************************/

     __device__ __forceinline__ BlockChoose()
     :	
     	tempStorage(PrivateStorage())
     {}


	/******************************************************************************
     * Choose samples
     ******************************************************************************/

 	__device__ __forceinline__ void choose(
		T const * __restrict__ 	P,
		T *	S,
		int np,
		int *ns,
		T r,
		int pStride = 0,
		int sStride = 0)
 	{
 		InternalChoose(tempStorage).choose(P, S, np, ns, r, pStride, sStride);
 	}

};

}	// bruteForce namespace
}	// gpu namespace
}	// rd namespace

#endif /* CHOOSE_CUH_ */
