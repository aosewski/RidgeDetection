/**
 * @file decimate2.cuh
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

#ifndef DECIMATE2_CUH_
#define DECIMATE2_CUH_ 

#include "rd/gpu/util/dev_math.cuh"
#include "rd/gpu/warp/warp_shfl_functions.cuh"
#include "rd/gpu/thread/thread_sqr_euclidean_dist.cuh"

#ifdef DEBUG
#include "cub/util_debug.cuh"
#endif

#include <device_launch_parameters.h>
#include <device_functions.h>

__device__ unsigned int blockCount = 0;

/**
 * @brief         Removes redundant points from set S.
 *
 * @par The function removes points satisfying at least one of the two
 *     following conditions:
 * @li            point has at least 4 neighbours in 2R neighbourhood
 * @li            point has at most 2 neighbours in 4R neighbourhood
 *
 * The algorithm stops when there is no more than 3 points left, or when there
 * are no points satisfying specified criteria.
 *
 * @oaram[in|out] S     Samples set to reduce.
 * @param[in|out] ns    Pointer to device global memory, containing
 *                      number of point in @p S set.
 * @param[in]     r     Algorithm parameter. The radius at which we
 *                      search neighbours.
 *
 * @tparam        T                Data type
 * @tparam        BLOCK_SIZE       Number of threads in block.
 * @tparam        VALS_PER_THREAD  Number of values to process by individual
 *                                 thread.
 * @tparam        DIM              Dimensionality of samples
 */
template <typename T,
         int BLOCK_SIZE, 
         int VALS_PER_THREAD, 
         int DIM>
__global__ void __inner_decimate(volatile T *S,
                                     int *ns,
                                     T r)
{

    T rSqr = r*r;
    int ns_ = *ns;
    int m = (ns_ + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    int deleteSample[VALS_PER_THREAD]{-1};

    __shared__ T shmem[BLOCK_SIZE * DIM * VALS_PER_THREAD];
    T inS[DIM * VALS_PER_THREAD];

    // while (oldCount != ns_ && ns_ > 3)
    // {
    for (int bOffset = blockDim.x * blockIdx.x * VALS_PER_THREAD;
         bOffset < m;
         bOffset += gridDim.x * blockDim.x * VALS_PER_THREAD)
    {
        // load data to shmem
        #pragma unroll
        for (int k = 0; k < VALS_PER_THREAD; ++k)
        {
            const int kOffset = k * BLOCK_SIZE * DIM;
            const int skOffset = bOffset * DIM + kOffset;
            int idx = 0;
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                idx = BLOCK_SIZE * d + threadIdx.x;
                shmem[kOffset + idx] = (idx + skOffset < ns_ * DIM) ? 
                    S[skOffset + idx] : 0.;
            }
        }
        __syncthreads();

        // load samples egzamined by thread
        #pragma unroll
        for (int k = 0; k < VALS_PER_THREAD; ++k)
        {
            // sample number in S table of values loaded by this thread
            deleteSample[k] = bOffset + k * BLOCK_SIZE + threadIdx.x;

            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                // FIXME: bank conflicts!
                inS[k * DIM + d] = 
                    shmem[k * BLOCK_SIZE * DIM + threadIdx.x * DIM + d];
            #ifdef DEBUG
            if (threadIdx.x == 16)
            {
                _CubLog("load to reg index: %d\n",
                     k * BLOCK_SIZE * DIM + threadIdx.x * DIM + d);
            }
            #endif
            }
        }

        __syncthreads();
        
        int shouldDelete = 0;
        unsigned char n1[VALS_PER_THREAD]{0};
        unsigned char n2[VALS_PER_THREAD]{0};

        // iterate through samples in S and check whether my points
        // should be deleted
        for (int x = 0; x < ns_; x += BLOCK_SIZE * VALS_PER_THREAD)
        {
            // load data to shmem
            #pragma unroll
            for (int k = 0; k < VALS_PER_THREAD; ++k)
            {
                const int kOffset = k * BLOCK_SIZE * DIM;
                const int skOffset = x * DIM + kOffset;
                int idx = 0;
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    idx = BLOCK_SIZE * d + threadIdx.x;
                    shmem[kOffset + idx] = (idx + skOffset < ns_ * DIM) ? 
                        S[skOffset + idx] : 0.;
                }
            }
            __syncthreads();

            for (int i = 0; i < BLOCK_SIZE * VALS_PER_THREAD && x + i < ns_; ++i)
            {
                T dist = 0;
                T currS[DIM];
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    currS[d] = shmem[i * DIM + d];
                }

                #ifdef DEBUG
                if (isnan(currS[0]))
                    if (threadIdx.x == 0)
                        printf("[%d]---Have a NaN--- n: %d, val: %f\n", threadIdx.x, x + i, currS[0]);
                #endif

                #pragma unroll
                for (int k = 0; k < VALS_PER_THREAD; ++k)
                {
                    // if (!isnan(shmem[i * DIM]) && !(shouldDelete & (1 << k)))
                    if (isnan(currS[0]) || (shouldDelete & (1 << k)))
                        continue;
                    else
                    {
                        // dist = threadSqrEuclideanDistanceRowMajor(inS + k * DIM, 
                        //     shmem + i * DIM, DIM);
                        dist = threadSqrEuclideanDistanceRowMajor(inS + k * DIM, 
                            currS, DIM);
                        if (dist < 4.f * rSqr)
                            n1[k]++;
                        if (dist < 16.f * rSqr)
                            n2[k]++;
                        if (n1[k] >= 4)
                            shouldDelete |= (1 << k);
                    }
                }
            }
        }

        #pragma unroll
        for (int k = 0; k < VALS_PER_THREAD; ++k)
        {
            if (n2[k] <= 2)
            {
                shouldDelete |= (1 << k);
                _CubLog("2nd delete rule: n2:%d, n:%d\n", n2[k], deleteSample[k]);
            }
        }

        // mark points to delete
        #pragma unroll
        for (int k = 0; k < VALS_PER_THREAD; ++k)
        {
            if (shouldDelete & (1 << k))
            {
                S[deleteSample[k] * DIM] = GetNaN<T>::value();
                #ifdef DEBUG
                    printf("[%d]---Write a NaN--- n: %d\n", threadIdx.x, deleteSample[k]);
                #endif
            }
        }
        __threadfence();
    }

    // __shared__ bool isLastBlockDone;
    // if (threadIdx.x == 0)
    // {
    //     unsigned int value = atomicInc(&blockCount, gridDim.x);
    //     isLastBlockDone = (value == (gridDim.x - 1));
    // }
    // __syncthreads();

    // // last block removes NaN from S table.
    // if (isLastBlockDone) 
    // {
    //     __shared__ int SCount;
    //     if (threadIdx.x == 0)
    //     {
    //         blockCount = 0;
    //         SCount = 0;
    //     }
    //     __syncthreads();

    //     int nanVoting = 0;
    //     int nanCount = 0;
    //     int warpOffset = 0;

    //     for (int x = threadIdx.x; x < m; x += BLOCK_SIZE)
    //     {
    //         inS[0] = S[x * DIM];
    //         nanVoting = __ballot(!isnan(inS[0]));
    //         nanCount = __popc(nanVoting);

    //         if ((threadIdx.x & 0x1f) == 0)
    //         {
    //             warpOffset = atomicAdd(&SCount, nanCount);
    //         }
    //         broadcast(warpOffset, 0);

    //         // calculate thread offset. Store to now unused register
    //         nanCount = __popc(nanVoting & ((1 << threadIdx.x) -1));

    //         // threads not having nan write down their value
    //         if (!isnan(inS[0]))
    //         {
    //             #pragma unroll
    //             for (int d = 1; d < DIM; ++d)
    //             {
    //                 inS[d] = S[x * DIM + d];
    //             }
    //             #pragma unroll
    //             for (int d = 0; d < DIM; ++d)
    //             {
    //                 S[SCount * DIM + nanCount * DIM + d] = inS[d];
    //             }
    //         }
    //     }

    //     __syncthreads();
    //     if (threadIdx.x == 0)
    //     {
    //         *ns = SCount;
    //     }
    // }
}


#endif /* DECIMATE2_CUH_ */
