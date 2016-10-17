/**
 * @file evolve.cuh
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

#ifndef __BRUTE_FORCE_EVOLVE_CUH__
#define __BRUTE_FORCE_EVOLVE_CUH__ 

#include "rd/gpu/device/brute_force/rd_globals.cuh"
#include "rd/gpu/util/dev_math.cuh"
#include "cub/util_type.cuh"

#include "rd/utils/memory.h"

namespace rd
{
namespace gpu
{
namespace bruteForce
{
    
/******************************************************************************
 * CALCULATE CLOSEST SPHERE CENTER 
 ******************************************************************************/

template <int _BLOCK_SIZE, int _ITEMS_PER_THREAD>
struct AgentClosestSpherePolicy
{
    enum 
    { 
        BLOCK_SIZE =        _BLOCK_SIZE,
        ITEMS_PER_THREAD =  _ITEMS_PER_THREAD
    };
};


template <
    typename            T,
    typename            AgentClosestSpherePolicyT,
    int                 DIM,
    DataMemoryLayout    INPUT_MEM_LAYOUT    = COL_MAJOR,
    DataMemoryLayout    OUTPUT_MEM_LAYOUT   = ROW_MAJOR>
class AgentClosestSphere
{

private:
    
    /******************************************************************************
     * Constants 
     ******************************************************************************/

    /// Constants
    enum
    {
        BLOCK_SIZE          = AgentClosestSpherePolicyT::BLOCK_SIZE,
        ITEMS_PER_THREAD    = AgentClosestSpherePolicyT::ITEMS_PER_THREAD
    };

    /******************************************************************************
     * Calculate closest sphere center
     ******************************************************************************/

    /// helper struct
    template <
        DataMemoryLayout _IN_MEM_LAYOUT,
        DataMemoryLayout _OUT_MEM_LAYOUT,
        int DUMMY>
    struct ClosestSphereCenterInternal;

    template <int DUMMY>
    struct ClosestSphereCenterInternal<ROW_MAJOR, ROW_MAJOR, DUMMY>
    {
        /// Constants
        enum
        {
            ITEMS_PER_BLOCK     = ITEMS_PER_THREAD * BLOCK_SIZE,
            PTS_PER_TILE        = BLOCK_SIZE / DIM
        };

        /// Shared memory storage type
        typedef T _TempStorage[BLOCK_SIZE];

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : cub::Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &tempStorage;

        //---------------------------------------------------------------------
        // Per-thread fields
        //---------------------------------------------------------------------

        T minSquareDist[ITEMS_PER_THREAD];
        T sqDist;
        int minSIdx[ITEMS_PER_THREAD];
        T dist[DIM];
        T point[ITEMS_PER_THREAD * DIM];

        __device__ __forceinline__ ClosestSphereCenterInternal(
            TempStorage &temp_storage) : tempStorage(temp_storage.Alias())
        {}

        __device__ void calc(
            T const * __restrict__ P,
            T const * __restrict__ S, 
            T * cordSums,
            int * spherePointCount,
            int np,
            int ns,
            T r,
            int,                        // pStride
            int,                        // csStride
            int)                        // sStride
        {
            int i;
            const int nn = (np + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK * ITEMS_PER_BLOCK;
            const int nTiles = (ns + PTS_PER_TILE - 1) / PTS_PER_TILE;

            for ( i = ITEMS_PER_BLOCK * blockIdx.x + threadIdx.x;
                  i < nn - ITEMS_PER_BLOCK;
                  i += ITEMS_PER_BLOCK * gridDim.x ) 
            {
                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    minSquareDist[j] = getMaxValue<T>();
                    int index = i + j * BLOCK_SIZE;
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        point[j * DIM + d] = P[index * DIM + d];
                }

                // through all choosen samples
                for (int k = 0; k < nTiles; ++k) 
                {
                    tempStorage[threadIdx.x] = (threadIdx.x + k * PTS_PER_TILE * DIM < ns * DIM)
                            ? S[threadIdx.x + k * PTS_PER_TILE * DIM] : 0;
                    __syncthreads();

                    for (int p = 0; p < PTS_PER_TILE && k * PTS_PER_TILE + p < ns; ++p) 
                    {
                        #pragma unroll
                        for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                        {
                            sqDist = 0;
                            #pragma unroll
                            for (int d = 0; d < DIM; ++d)
                                dist[d] = tempStorage[p*DIM + d] - point[j * DIM + d];

                            #pragma unroll
                            for (int d = 0; d < DIM; ++d)
                                sqDist += dist[d] * dist[d];

                            if (sqDist < minSquareDist[j]) 
                            {
                                minSIdx[j] = k * PTS_PER_TILE + p;
                                minSquareDist[j] = sqDist;
                            }
                        }
                    }
                    __syncthreads();
                }

                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    if (minSquareDist[j] <= r * r) 
                    {
                        (void) atomicAdd(spherePointCount + minSIdx[j], 1);
                        // sum sample coordinates for later mass center calculation
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            (void) rdAtomicAdd(cordSums + minSIdx[j] * DIM + d, point[j * DIM + d]);
                    }
                }
            }

            // To avoid the (index<N) conditional in the inner loop,
            // we left off some work at the end for the last block.
            // Only one block should enter here!
            if (i < nn) 
            {

                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    minSquareDist[j] = getMaxValue<T>();
                    int index = i + j * BLOCK_SIZE;
                    if (index < np) 
                    {
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            point[j * DIM + d] = P[index * DIM + d];
                    }
                }

                // through all choosen samples
                for (int k = 0; k < nTiles; ++k) 
                {
                      tempStorage[threadIdx.x] = (threadIdx.x + k * PTS_PER_TILE * DIM < ns * DIM)
                            ? S[threadIdx.x + k * PTS_PER_TILE * DIM] : 0;
                    __syncthreads();

                    for (int p = 0; p < PTS_PER_TILE && k * PTS_PER_TILE + p < ns; ++p) 
                    {
                        #pragma unroll
                        for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                        {
                            int index = i + j * BLOCK_SIZE;
                            if (index < np) 
                            {
                                sqDist = 0;
                                #pragma unroll
                                for (int d = 0; d < DIM; ++d)
                                    dist[d] = tempStorage[p*DIM + d] - point[j * DIM + d];

                                #pragma unroll
                                for (int d = 0; d < DIM; ++d)
                                    sqDist += dist[d] * dist[d];

                                if (sqDist < minSquareDist[j]) 
                                {
                                    minSIdx[j] = k * PTS_PER_TILE + p;
                                    minSquareDist[j] = sqDist;
                                }
                            }
                        }
                    }
                    __syncthreads();
                }

                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    int index = i + j * BLOCK_SIZE;
                    if (index < np) 
                    {
                        if (minSquareDist[j] <= r * r) 
                        {
                            (void) atomicAdd(spherePointCount + minSIdx[j], 1);
                            // sum sample coordinates for later mass center calculation
                            #pragma unroll
                            for (int d = 0; d < DIM; ++d)
                                (void) rdAtomicAdd(cordSums + minSIdx[j] * DIM + d, point[j * DIM + d]);
                        }
                    }
                }
            }
        }
    };

    template <int DUMMY>
    struct ClosestSphereCenterInternal<COL_MAJOR, ROW_MAJOR, DUMMY>
    {
        /// Constants
        enum
        {
            ITEMS_PER_BLOCK     = ITEMS_PER_THREAD * BLOCK_SIZE,
            PTS_PER_TILE        = BLOCK_SIZE / DIM
        };

        /// Shared memory storage layout type
        typedef T _TempStorage[BLOCK_SIZE];

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : cub::Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &tempStorage;

        //---------------------------------------------------------------------
        // Per-thread fields
        //---------------------------------------------------------------------

        T minSquareDist[ITEMS_PER_THREAD];
        T sqDist;
        int minSIdx[ITEMS_PER_THREAD];
        T dist[DIM];
        T point[ITEMS_PER_THREAD * DIM];

        __device__ __forceinline__ ClosestSphereCenterInternal(
            TempStorage &temp_storage) : tempStorage(temp_storage.Alias())
        {}

        __device__  void calc(
            T const * __restrict__ P,
            T const * __restrict__ S, 
            T * cordSums,
            int * spherePointCount,
            int np,
            int ns,
            T r,
            int pStride,
            int csStride,
            int) 
        {
            int i;
            // round up samples number to the nearest multiply of ITEMS_PER_BLOCK
            const int nn = (np + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK * ITEMS_PER_BLOCK;
            const int nTiles = (ns + PTS_PER_TILE - 1) / PTS_PER_TILE;

            for ( i = ITEMS_PER_BLOCK * blockIdx.x + threadIdx.x;
                  i < nn-ITEMS_PER_BLOCK;
                  i += ITEMS_PER_BLOCK * gridDim.x ) 
            {
                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    minSquareDist[j] = getMaxValue<T>();
                    int index = i + j * BLOCK_SIZE;
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        point[j * DIM + d] = P[index + d * pStride];
                }

                // through all choosen samples
                for (int k = 0; k < nTiles; ++k) 
                {
                    tempStorage[threadIdx.x] = (threadIdx.x + k * PTS_PER_TILE * DIM < ns * DIM)
                            ? S[threadIdx.x + k * PTS_PER_TILE * DIM] : 0;
                    __syncthreads();

                    for (int p = 0; p < PTS_PER_TILE && k * PTS_PER_TILE + p < ns; ++p) 
                    {
                        #pragma unroll
                        for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                        {
                            sqDist = 0;
                            #pragma unroll
                            for (int d = 0; d < DIM; ++d)
                                dist[d] = tempStorage[p*DIM + d] - point[j * DIM + d];

                            #pragma unroll
                            for (int d = 0; d < DIM; ++d)
                                sqDist += dist[d] * dist[d];

                            if (sqDist < minSquareDist[j]) 
                            {
                                minSIdx[j] = k * PTS_PER_TILE + p;
                                minSquareDist[j] = sqDist;
                            }
                        }
                    }
                    __syncthreads();
                }

                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    if (minSquareDist[j] <= r * r) 
                    {
                        (void) atomicAdd(spherePointCount + minSIdx[j], 1);
                        // sum sample coordinates for later mass center calculation
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            (void) rdAtomicAdd(cordSums + minSIdx[j] + d * csStride, 
                                point[j*DIM + d]);
                    }
                }
            }

            // To avoid the (index<N) conditional in the inner loop,
            // we left off some work at the end for the last block.
            // Only one block should enter here!
            if (i < nn) 
            {
                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    minSquareDist[j] = getMaxValue<T>();
                    int index = i + j * BLOCK_SIZE;
                    if (index < np) 
                    {
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            point[j * DIM + d] = P[index + d * pStride];
                    }
                }

                // through all choosen samples
                for (int k = 0; k < nTiles; ++k) 
                {
                    tempStorage[threadIdx.x] = (threadIdx.x + k * PTS_PER_TILE * DIM < ns * DIM)
                            ? S[threadIdx.x + k * PTS_PER_TILE * DIM] : 0;
                    __syncthreads();

                    for (int p = 0; p < PTS_PER_TILE && k * PTS_PER_TILE + p < ns; ++p) 
                    {
                        #pragma unroll
                        for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                        {
                            int index = i + j * BLOCK_SIZE;
                            if (index < np) 
                            {
                                sqDist = 0;
                                #pragma unroll
                                for (int d = 0; d < DIM; ++d)
                                    dist[d] = tempStorage[p*DIM + d] - point[j * DIM + d];

                                #pragma unroll
                                for (int d = 0; d < DIM; ++d)
                                    sqDist += dist[d] * dist[d];

                                if (sqDist < minSquareDist[j]) 
                                {
                                    minSIdx[j] = k*PTS_PER_TILE + p;
                                    minSquareDist[j] = sqDist;
                                }
                            }
                        }
                    }
                    __syncthreads();
                }

                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    int index = i+j*BLOCK_SIZE;
                    if (index < np) 
                    {
                        if (minSquareDist[j] <= r * r) 
                        {
                            (void) atomicAdd(spherePointCount + minSIdx[j], 1);
                            // sum sample coordinates for later mass center calculation
                            #pragma unroll
                            for (int d = 0; d < DIM; ++d)
                                (void) rdAtomicAdd(cordSums + minSIdx[j] + d * csStride, 
                                    point[j*DIM + d]);
                        }
                    }
                }
            }
        }
    };

    template <int DUMMY>
    struct ClosestSphereCenterInternal<COL_MAJOR, COL_MAJOR, DUMMY>
    {
        /// Constants
        enum
        {
            ITEMS_PER_BLOCK     = ITEMS_PER_THREAD * BLOCK_SIZE,
            PTS_PER_TILE        = BLOCK_SIZE / DIM
        };

        /// Shared memory storage layout type
        typedef T _TempStorage[BLOCK_SIZE];

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : cub::Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &tempStorage;

        //---------------------------------------------------------------------
        // Per-thread fields
        //---------------------------------------------------------------------

        T minSquareDist[ITEMS_PER_THREAD];
        T sqDist;
        int minSIdx[ITEMS_PER_THREAD];
        T dist[DIM];
        T point[ITEMS_PER_THREAD * DIM];

        __device__ __forceinline__ ClosestSphereCenterInternal(
            TempStorage &temp_storage) : tempStorage(temp_storage.Alias())
        {}

        __device__ void calc(
            T const * __restrict__ P,
            T const * __restrict__ S, 
            T * cordSums,
            int * spherePointCount,
            int np,
            int ns,
            T r,
            int pStride,
            int csStride,
            int sStride) 
        {
            int i;
            // round up samples number to the nearest multiply of ITEMS_PER_BLOCK
            const int nn = (np + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK * ITEMS_PER_BLOCK;
            const int ns1 = (ns + PTS_PER_TILE - 1) / PTS_PER_TILE * PTS_PER_TILE;

            for ( i = ITEMS_PER_BLOCK * blockIdx.x + threadIdx.x;
                  i < nn - ITEMS_PER_BLOCK;
                  i += ITEMS_PER_BLOCK * gridDim.x ) 
            {
                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    minSquareDist[j] = getMaxValue<T>();
                    int index = i + j * BLOCK_SIZE;
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        point[j * DIM + d] = P[index + d * pStride];
                }

                // through all choosen samples
                for (int k = 0; k < ns1; k += PTS_PER_TILE) 
                {
                    tempStorage[threadIdx.x] = 
                            (k + (threadIdx.x % PTS_PER_TILE) < ns) && 
                            threadIdx.x < DIM * PTS_PER_TILE ?
                        S[k + (threadIdx.x % PTS_PER_TILE) + 
                            (threadIdx.x / PTS_PER_TILE) * sStride] : 0;
                    __syncthreads();

                    for (int p = 0; p < PTS_PER_TILE && k + p < ns; ++p) 
                    {
                        #pragma unroll
                        for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                        {
                            sqDist = 0;
                            #pragma unroll
                            for (int d = 0; d < DIM; ++d)
                                dist[d] = tempStorage[p + d * PTS_PER_TILE] - point[j * DIM + d];

                            #pragma unroll
                            for (int d = 0; d < DIM; ++d)
                                sqDist += dist[d] * dist[d];

                            if (sqDist < minSquareDist[j]) 
                            {
                                minSIdx[j] = k + p;
                                minSquareDist[j] = sqDist;
                            }
                        }
                    }
                    __syncthreads();
                }

                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    if (minSquareDist[j] <= r * r) 
                    {
                        (void) atomicAdd(spherePointCount + minSIdx[j], 1);
                        // sum sample coordinates for later mass center calculation
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            (void) rdAtomicAdd(cordSums + minSIdx[j] + d * csStride, 
                                point[j*DIM + d]);
                    }
                }
            }

            // To avoid the (index<N) conditional in the inner loop,
            // we left off some work at the end for the last block.
            // Only one block should enter here!
            if (i < nn) 
            {
                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    minSquareDist[j] = getMaxValue<T>();
                    int index = i + j * BLOCK_SIZE;
                    if (index < np) 
                    {
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            point[j * DIM + d] = P[index + d * pStride];
                    }
                }

                // through all choosen samples
                for (int k = 0; k < ns1; k += PTS_PER_TILE) 
                {
                    tempStorage[threadIdx.x] = 
                            (k + (threadIdx.x % PTS_PER_TILE) < ns) && 
                            threadIdx.x < DIM * PTS_PER_TILE ?
                        S[k + (threadIdx.x % PTS_PER_TILE) + 
                            (threadIdx.x / PTS_PER_TILE) * sStride] : 0;
                    __syncthreads();

                    for (int p = 0; p < PTS_PER_TILE && k + p < ns; ++p) 
                    {
                        #pragma unroll
                        for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                        {
                            int index = i + j * BLOCK_SIZE;
                            if (index < np) 
                            {
                                sqDist = 0;
                                #pragma unroll
                                for (int d = 0; d < DIM; ++d)
                                    dist[d] = 
                                        tempStorage[p + d * PTS_PER_TILE] - point[j * DIM + d];

                                #pragma unroll
                                for (int d = 0; d < DIM; ++d)
                                    sqDist += dist[d] * dist[d];

                                if (sqDist < minSquareDist[j]) 
                                {
                                    minSIdx[j] = k + p;
                                    minSquareDist[j] = sqDist;
                                }
                            }
                        }
                    }
                    __syncthreads();
                }

                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    int index = i+j*BLOCK_SIZE;
                    if (index < np) 
                    {
                        if (minSquareDist[j] <= r * r) 
                        {
                            (void) atomicAdd(spherePointCount + minSIdx[j], 1);
                            // sum sample coordinates for later mass center calculation
                            #pragma unroll
                            for (int d = 0; d < DIM; ++d)
                                (void) rdAtomicAdd(cordSums + minSIdx[j] + d * csStride, 
                                    point[j*DIM + d]);
                        }
                    }
                }
            }
        }
    };

    /******************************************************************************
     * Type definitions
     ******************************************************************************/

     /// Type of internal implementation to use
     typedef ClosestSphereCenterInternal<INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0> InternalImpl;

     /// shared memory storage layout type
     typedef typename InternalImpl::TempStorage _TempStorage;

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

    /// smemstorage
    struct TempStorage : cub::Uninitialized<_TempStorage> {};

    /******************************************************************************
     * Collective constructors
     ******************************************************************************/

     __device__ __forceinline__ AgentClosestSphere()
     :  
        tempStorage(PrivateStorage())
     {}


    /******************************************************************************
     * Interface
     ******************************************************************************/
 
    __device__ __forceinline__ void calc(
        T const * __restrict__ P,
        T const * __restrict__ S, 
        T * cordSums,
        int * spherePointCount,
        int np,
        int ns,
        T r,
        int pStride,
        int csStride,
        int sStride)
    {
        InternalImpl(tempStorage).calc(P, S, cordSums, spherePointCount, np, ns, r,
            pStride, csStride, sStride);
    } 

};


/******************************************************************************
 *
 *
 *
 *              SHIFT TOWARD MASS CENTER 
 *              
 *              
 *              
 ******************************************************************************/


template <int _BLOCK_SIZE, int _ITEMS_PER_THREAD>
struct AgentShiftSpherePolicy
{
    enum
    {
        ITEMS_PER_THREAD = _ITEMS_PER_THREAD,
        BLOCK_SIZE       = _BLOCK_SIZE       
    };
};

template <
    typename            T,
    typename            AgentShiftSpherePolicyT,
    int                 DIM,
    DataMemoryLayout    INPUT_MEM_LAYOUT    = COL_MAJOR,
    DataMemoryLayout    OUTPUT_MEM_LAYOUT   = ROW_MAJOR>
class AgentShiftSphere
{

private:

    /******************************************************************************
     * Constants 
     ******************************************************************************/

    /// Constants
    enum
    {
        ITEMS_PER_THREAD    = AgentShiftSpherePolicyT::ITEMS_PER_THREAD,
        BLOCK_SIZE          = AgentShiftSpherePolicyT::BLOCK_SIZE
    };

    /******************************************************************************
     * Shift towards mass center
     ******************************************************************************/

    /// helper struct
    template <
        DataMemoryLayout _IN_MEM_LAYOUT,
        DataMemoryLayout _OUT_MEM_LAYOUT,
        int DUMMY>
    struct ShiftSphereInternal;

    template <int DUMMY>
    struct ShiftSphereInternal<ROW_MAJOR, ROW_MAJOR, DUMMY>
    {
        /// Constants
        enum
        {
            ITEMS_PER_BLOCK     = ITEMS_PER_THREAD * BLOCK_SIZE
        };


        //---------------------------------------------------------------------
        // Per-thread fields
        //---------------------------------------------------------------------

        T cs[ITEMS_PER_THREAD * DIM];
        T mc[ITEMS_PER_THREAD * DIM];
        int spc[ITEMS_PER_THREAD];

        __device__ __forceinline__ ShiftSphereInternal()
        {}

        __device__  void shift(
            T * S,
            T const * __restrict__ cordSums,
            int const * __restrict__ spherePointCount,
            int ns,
            int,    // csStride 
            int )   // sStride
        {
            int i;
            // round up to the nearest multiply of items per block
            const int nn = (ns + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK * ITEMS_PER_BLOCK;

            for ( i = ITEMS_PER_BLOCK * blockIdx.x + threadIdx.x;
                  i < nn - ITEMS_PER_BLOCK;
                  i += ITEMS_PER_BLOCK * gridDim.x ) 
            {

                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    int index = i + j * BLOCK_SIZE;
                    spc[j] = spherePointCount[index];

                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        cs[j * DIM + d] = cordSums[index * DIM + d];
                }

                #pragma unroll
                for ( int j = 0; j < ITEMS_PER_THREAD; ++j ) 
                {
                    int index = i + j * BLOCK_SIZE;

                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        mc[j * DIM + d] = cs[j * DIM + d] / T(spc[j]);

                    for (int d = 0; d < DIM; ++d)
                    {
                        if (fabs(mc[j * DIM + d] - S[index * DIM + d]) > 
                            T(2.f) * fabs(mc[j * DIM + d]) * T(spc[j]) * getEpsilon<T>())
                        {
                            (void) atomicExch(&rdContFlag, 1);
                            #pragma unroll
                            for (int k = 0; k < DIM; ++k)
                                S[index * DIM + k] = mc[j * DIM + k];
                            break;
                        }
                    }
                }
            }

            // To avoid the (index<N) conditional in the inner loop,
            // we left off some work at the end for the last block.
            // Only one block should enter here!
            if (i < nn) 
            {
                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    int index = i + j * BLOCK_SIZE;
                    if (index < ns) 
                    {
                        spc[j] = spherePointCount[index];
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            cs[j * DIM + d] = cordSums[index * DIM + d];
                    }
                }

                #pragma unroll
                for ( int j = 0; j < ITEMS_PER_THREAD; ++j ) 
                {
                    int index = i + j * BLOCK_SIZE;
                    if (index < ns) 
                    {
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            mc[j * DIM + d] = cs[j * DIM + d] / T(spc[j]);

                        for (int d = 0; d < DIM; ++d)
                        {
                            if (fabs(mc[j * DIM + d] - S[index * DIM + d]) > 
                                T(2.f) * fabs(mc[j * DIM + d]) * T(spc[j]) * getEpsilon<T>())
                            {
                                (void) atomicExch(&rdContFlag, 1);
                                for (int k = 0; k < DIM; ++k)
                                    S[index * DIM + k] = mc[j * DIM + k];
                                break;
                            }
                        }
                    }
                }
            }
        }
    };

    template <int DUMMY>
    struct ShiftSphereInternal<COL_MAJOR, ROW_MAJOR, DUMMY>
    {
        /// Constants
        enum
        {
            ITEMS_PER_BLOCK     = ITEMS_PER_THREAD * BLOCK_SIZE
        };


        //---------------------------------------------------------------------
        // Per-thread fields
        //---------------------------------------------------------------------

        T cs[ITEMS_PER_THREAD * DIM];
        T mc[ITEMS_PER_THREAD * DIM];
        int spc[ITEMS_PER_THREAD];

        __device__ __forceinline__ ShiftSphereInternal()
        {}

        __device__  void shift(
            T * S,
            T const * __restrict__ cordSums,
            int const * __restrict__ spherePointCount,
            int ns,
            int csStride, 
            int )           // sStride
        {
            int i;
            // round up to the nearest multiply of items per block
            const int nn = (ns + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK * ITEMS_PER_BLOCK;

            for ( i = ITEMS_PER_BLOCK * blockIdx.x + threadIdx.x;
                  i < nn - ITEMS_PER_BLOCK;
                  i += ITEMS_PER_BLOCK * gridDim.x ) 
            {

                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    int index = i + j * BLOCK_SIZE;
                    spc[j] = spherePointCount[index];

                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        cs[j * DIM + d] = cordSums[index + d * csStride];
                }

                #pragma unroll
                for ( int j = 0; j < ITEMS_PER_THREAD; ++j ) 
                {
                    int index = i + j * BLOCK_SIZE;

                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        mc[j * DIM + d] = cs[j * DIM + d] / T(spc[j]);

                    for (int d = 0; d < DIM; ++d)
                    {
                        if (fabs(mc[j * DIM + d] - S[index * DIM + d]) > 
                            T(2.f) * fabs(mc[j * DIM + d]) * T(spc[j]) * getEpsilon<T>())
                        {
                            (void) atomicExch(&rdContFlag, 1);
                            #pragma unroll
                            for (int k = 0; k < DIM; ++k)
                                S[index * DIM + k] = mc[j * DIM + k];
                            break;
                        }
                    }
                }
            }

            // To avoid the (index<N) conditional in the inner loop,
            // we left off some work at the end for the last block.
            // Only one block should enter here!
            if (i < nn) 
            {
                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    int index = i + j * BLOCK_SIZE;
                    if (index < ns) 
                    {
                        spc[j] = spherePointCount[index];
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            cs[j * DIM + d] = cordSums[index + d * csStride];
                    }
                }

                #pragma unroll
                for ( int j = 0; j < ITEMS_PER_THREAD; ++j ) 
                {
                    int index = i + j * BLOCK_SIZE;
                    if (index < ns) 
                    {
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            mc[j * DIM + d] = cs[j * DIM + d] / T(spc[j]);

                        for (int d = 0; d < DIM; ++d)
                        {
                            if (fabs(mc[j * DIM + d] - S[index * DIM + d]) > 
                                T(2.f) * fabs(mc[j * DIM + d]) * T(spc[j]) * getEpsilon<T>())
                            {
                                (void) atomicExch(&rdContFlag, 1);
                                #pragma unroll
                                for (int k = 0; k < DIM; ++k)
                                    S[index * DIM + k] = mc[j * DIM + k];
                                break;
                            }
                        }
                    }
                }
            }
        }
    };

    template <int DUMMY>
    struct ShiftSphereInternal<COL_MAJOR, COL_MAJOR, DUMMY>
    {
        /// Constants
        enum
        {
            ITEMS_PER_BLOCK     = ITEMS_PER_THREAD * BLOCK_SIZE
        };

        //---------------------------------------------------------------------
        // Per-thread fields
        //---------------------------------------------------------------------

        T cs[ITEMS_PER_THREAD * DIM];
        T mc[ITEMS_PER_THREAD * DIM];
        int spc[ITEMS_PER_THREAD];

        __device__ __forceinline__ ShiftSphereInternal()
        {}

        __device__  void shift(
            T * S,
            T const * __restrict__ cordSums,
            int const * __restrict__ spherePointCount,
            int ns,
            int csStride,
            int sStride)
        {
            int i;
            // round up to the nearest multiply of items per block
            const int nn = (ns + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK * ITEMS_PER_BLOCK;

            for ( i = ITEMS_PER_BLOCK * blockIdx.x + threadIdx.x;
                  i < nn - ITEMS_PER_BLOCK;
                  i += ITEMS_PER_BLOCK * gridDim.x ) 
            {

                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    int index = i + j * BLOCK_SIZE;
                    spc[j] = spherePointCount[index];

                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        cs[j * DIM + d] = cordSums[index + d * csStride];
                }

                #pragma unroll
                for ( int j = 0; j < ITEMS_PER_THREAD; ++j ) 
                {
                    int index = i + j * BLOCK_SIZE;

                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        mc[j * DIM + d] = cs[j * DIM + d] / T(spc[j]);

                    for (int d = 0; d < DIM; ++d)
                    {
                        if (fabs(mc[j * DIM + d] - S[index + d * sStride]) > 
                            T(2.f) * fabs(mc[j * DIM + d]) * T(spc[j]) * getEpsilon<T>())
                        {
                            (void) atomicExch(&rdContFlag, 1);
                            #pragma unroll
                            for (int k = 0; k < DIM; ++k)
                                S[index + k * sStride] = mc[j * DIM + k];
                            break;
                        }
                    }
                }
            }

            // To avoid the (index<N) conditional in the inner loop,
            // we left off some work at the end for the last block.
            // Only one block should enter here!
            if (i < nn) 
            {
                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) 
                {
                    int index = i + j * BLOCK_SIZE;
                    if (index < ns) 
                    {
                        spc[j] = spherePointCount[index];
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            cs[j * DIM + d] = cordSums[index + d * csStride];
                    }
                }

                #pragma unroll
                for ( int j = 0; j < ITEMS_PER_THREAD; ++j ) 
                {
                    int index = i + j * BLOCK_SIZE;
                    if (index < ns) 
                    {
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            mc[j * DIM + d] = cs[j * DIM + d] / T(spc[j]);

                        for (int d = 0; d < DIM; ++d)
                        {
                            if (fabs(mc[j * DIM + d] - S[index + d * sStride]) > 
                                T(2.f) * fabs(mc[j * DIM + d]) * T(spc[j]) * getEpsilon<T>())
                            {
                                (void) atomicExch(&rdContFlag, 1);
                                #pragma unroll
                                for (int k = 0; k < DIM; ++k)
                                    S[index + k * sStride] = mc[j * DIM + k];
                                break;
                            }
                        }
                    }
                }
            }
        }
    };

    /******************************************************************************
     * Type definitions
     ******************************************************************************/

     /// Type of internal implementation to use
     typedef ShiftSphereInternal<INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0> InternalImpl;

public:


    /******************************************************************************
     * Collective constructors
     ******************************************************************************/


    __device__ __forceinline__ AgentShiftSphere()
    {}


    /******************************************************************************
     * Interface
     ******************************************************************************/

    __device__ __forceinline__ void shift(
        T * S,
        T const * __restrict__ cordSums,
        int const * __restrict__ spherePointCount,
        int ns,
        int csStride,
        int sStride)
    {
        InternalImpl().shift(S, cordSums, spherePointCount, ns, csStride, sStride);
    }

};


}   // end namespace bruteForce
}   // end namespace gpu
}   // end namespace rd

#endif