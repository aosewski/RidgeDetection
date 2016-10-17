/**
 * @file bounding_box.cuh
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

#pragma once

#include "rd/gpu/util/dev_math.cuh"
#include "rd/gpu/device/device_find_bounds.cuh"
#include "rd/utils/memory.h"

#include "cub/util_debug.cuh"
#include "cub/util_type.cuh"
#include "cub/util_device.cuh"
#include "cub/util_arch.cuh"

// #ifndef RD_DEBUG
// #define NDEBUG      // for disabling assert macro
// #endif 
#include <assert.h>
#include <common_functions.h>

namespace rd
{
namespace gpu
{


/**
 * @brief      Represents bounding box of multidimensional samples set.
 *
 * @tparam     DIM   Bounding box dimension
 * @tparam     SampleT     Data type.
 */
template <
    int         DIM,
    typename    SampleT>
struct BoundingBox
{

    typedef SampleT PointT[DIM];

    SampleT bbox[DIM * 2];
    /// Distance between min and max
    SampleT dist[DIM];

    /**
     * @brief      Finds the bounding box (min,max values) for the given @p d data set.
     *
     * @param      d_inputSamples  Pointer to device data set.
     * @param[in]  pointsCnt       Number of input points.
     * @param[in]  stride          Distance between consecutive point's coordinates
     * @param[in]  stream          Stream id to run computations with.
     *
     * @tparam     MEM_LAYOUT      Input data memory layout ROW/COL_MAJOR.
     *
     * @note       Within bbox data have following layout: [min_1,max_1,...,min_n,max_n]
     */
    template <DataMemoryLayout MEM_LAYOUT>
    CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t findBounds(
        SampleT const *         d_inputSamples, 
        int                     pointsCnt,
        int                     stride = 1,
        cudaStream_t            stream = 0,
        bool                    debugSynchronous = false)
    {
        cudaError_t error = cudaSuccess;
        SampleT *d_bboxMin = nullptr, *d_bboxMax = nullptr;
        void *d_tempStorage = nullptr;

        if ( CubDebug(error = cudaMalloc(&d_bboxMin, DIM * sizeof(SampleT))) ) return error;
        if ( CubDebug(error = cudaMalloc(&d_bboxMax, DIM * sizeof(SampleT))) ) return error;
        assert(d_bboxMin != nullptr);
        assert(d_bboxMax != nullptr);

        do
        {
            size_t tempStorageBytes = 0;

            // acquire temp storage requirements
            error = DeviceFindBounds::findBounds<DIM, MEM_LAYOUT>(
                d_tempStorage,
                tempStorageBytes,
                d_inputSamples,
                d_bboxMin,
                d_bboxMax,
                pointsCnt,
                stride, 
                stream,
                debugSynchronous);

            if( CubDebug(error = cudaPeekAtLastError()) ) break;
            if( CubDebug(error = cudaMalloc(&d_tempStorage, tempStorageBytes)) ) break;
            assert(d_tempStorage != nullptr);

            error = DeviceFindBounds::findBounds<DIM, MEM_LAYOUT>(
                d_tempStorage,
                tempStorageBytes,
                d_inputSamples,
                d_bboxMin,
                d_bboxMax,
                pointsCnt,
                stride, 
                stream,
                debugSynchronous);

            if( CubDebug(error = cudaPeekAtLastError()) ) break;
            if( CubDebug(error = cub::SyncStream(stream)) ) break;

            #if __CUDA_ARCH__ > 0
                for (int d = 0; d < DIM; ++d)
                {
                    bbox[2 * d]       = d_bboxMin[d];
                    bbox[2 * d + 1]   = d_bboxMax[d];
                }
            #else
                SampleT * h_bboxMin = new SampleT[DIM];
                SampleT * h_bboxMax = new SampleT[DIM];

                if ( CubDebug(error = cudaMemcpy(h_bboxMin, d_bboxMin, DIM * sizeof(SampleT),
                    cudaMemcpyDeviceToHost)) ) break;
                if ( CubDebug(error = cudaMemcpy(h_bboxMax, d_bboxMax, DIM * sizeof(SampleT),
                    cudaMemcpyDeviceToHost)) ) break;
                if ( CubDebug(error = cudaDeviceSynchronize())) break;

                for (int d = 0; d < DIM; ++d)
                {
                    bbox[2 * d]       = h_bboxMin[d];
                    bbox[2 * d + 1]   = h_bboxMax[d];
                }

                delete[] h_bboxMin;
                delete[] h_bboxMax;
            #endif
        } while (0);

        if (d_tempStorage != nullptr) CubDebug(cudaFree(d_tempStorage));
        if (d_bboxMin != nullptr) CubDebug(cudaFree(d_bboxMin));
        if (d_bboxMax != nullptr) CubDebug(cudaFree(d_bboxMax));

        return error;
    }


    /**
     * @brief      Calculates distance between min and max value in each
     *             dimension.
     */
    __host__  __device__ __forceinline__ void calcDistances()
    {
        // calc distances
        #if __CUDA_ARCH__ > 0
        #pragma unroll
        #endif
        for (int k = 0; k < DIM; ++k)
        {
            dist[k] = abs(bbox[2*k+1] - bbox[2*k]);
            if (dist[k] <= getEpsilon<SampleT>())
                dist[k] = 0;
        }
    }

    /**
     * @brief      Returns lower bound for @p idx dimension
     *
     * @param[in]  idx   Requested dimension.
     *
     * @return     Const reference to bound value.
     */
    __host__ __device__ __forceinline__ SampleT const & min(
        const int idx) const
    {
        assert(idx < DIM && idx >= 0);
        return bbox[2*idx];
    }

    /**
     * @brief      Returns lower bound for @p idx dimension.
     *
     * @param[in]  idx   Requested dimension.
     *
     * @return     Reference to bound value.
     */
    __host__ __device__ __forceinline__ SampleT & min(
        const int idx)
    {
        assert(idx < DIM && idx >= 0);
        return bbox[2*idx];
    }

    /**
     * @brief      Returns upper bound for @p idx dimension.
     *
     * @param[in]  idx   Requested dimension.
     *
     * @return     Const reference to bound value.
     */
    __host__ __device__ __forceinline__ SampleT const & max(
        const int idx) const
    {
        assert(idx < DIM && idx >= 0);
        return bbox[2*idx + 1];
    }

    /**
     * @brief      Returns upper bound for @p idx dimension.
     *             
     * @param[in]  idx   Requested dimension.
     *
     * @return     Const reference to bound value.
     */
    __host__ __device__ __forceinline__ SampleT & max(
        const int idx)
    {
        assert(idx < DIM && idx >= 0);
        return bbox[2*idx + 1];
    }

    __host__ __device__ __forceinline__ void print() const
    {
        #if __CUDA_ARCH__ > 0
            printf("\n");
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                printf("[bid: %d] bounds: DIM[%d], min: %8.6f, max %8.6f \n",
                   blockIdx.x, d, bbox[2*d], bbox[2*d+1]);
            }
            for (int d = 0; d < DIM; ++d)
            {
                printf("[bid: %d] distances: DIM[%d] %8.6f \n",
                    blockIdx.x, d, dist[d]);
            }
        #else 
            printf("--------------------------\nbounds:\n");
            for (int d = 0; d < DIM; ++d)
            {
                printf("DIM: %d, min: %8.6f, max %8.6f \n", d, bbox[2*d], bbox[2*d+1]);
            }
            printf("distances: [");
            for (int d = 0; d < DIM; ++d)
                printf("%8.6f, ", dist[d]);
            printf("]\n---------------------------------\n");
        #endif
    }

    /**
     * @brief      Counts how many spheres fits inside region described by @p bb
     *
     * @param[in]  radius  Sphere radius.
     *
     * @return     the number of spheres that fits in @p bb region.
     */
    __host__ __device__ __forceinline__ int countSpheresInside(
        SampleT radius) const
    {
        int cnt = 1;
        #if __CUDA_ARCH__ > 0
        #pragma unroll
        #endif
        for (int d = 0; d < DIM; ++d)
        {
            int dcnt= static_cast<int>(ceil(dist[d] / radius));
            cnt *= (dcnt) ? dcnt + 1 : 1;
        }

        return cnt;
    }

    /**
     * @brief      Checks whether @p point lies inside bounds
     *
     * @note       Inside bounds is meant to be: min <= x < max
     *
     * @param      point  Pointer to point coordinates.
     *
     * @return     True if @p point is inside this bounds
     */
    __host__ __device__ __forceinline__ bool isInside(
        SampleT const *point) const
    {
        #if __CUDA_ARCH__ > 0
        #pragma unroll
        #endif
        for (int d = 0; d < DIM; ++d)
        {
            if (point[d] < min(d) || point[d] >= max(d))
                return false;
        }
        return true;
    }

    /**
     * @brief      Chekcs wheather @p point lies inside extended bounding box.
     *
     *             Precisely it checks wheather @p point lies in area of (ext
     *             bounding box) minus area of this (bounding box). It is a
     *             neighbourhood of this bounding box.
     *
     * @param      point         Pointer to point coordinates.
     * @param[in]  extendRadius  Amount by which we enlarge this bounding box
     *                           area for searching neighbours.
     *
     * @return     True if @p point lies nearby this bonding box.
     */
    __host__ __device__ __forceinline__ bool isNearby(
        SampleT const *point, 
        SampleT extendRadius) const
    {
        #if __CUDA_ARCH__ > 0
        #pragma unroll
        #endif
        for (int d = 0; d < DIM; ++d)
        {
            // is outside ext bbox
            if ((point[d] < min(d) - extendRadius) || (point[d] >= max(d) + extendRadius))
                return false;
        }

        if (!isInside(point))
        {
            return true;
        }

        return false;
    }

    /**
     * @brief      Checks whether @p other overlaps with this bounding box
     *
     * @param      other  Examined bounding box
     *
     * @return     True if two bounding boxes overlap
     */
    __host__ __device__ __forceinline__ bool overlap(
        BoundingBox<DIM, SampleT> const &other) const
    {
        bool result;

        // there must be overlap in all dimensions
        #if __CUDA_ARCH__ > 0
        #pragma unroll
        #endif
        for (int d = 0; d < DIM; ++d)
        {
            /**
             * this code doesn't work properly - it doesn't handle all cases, 
             * however leave it for future - maybe it'll be useful.
             */
            /*  
                // check current DIM center distance
             
                SampleT m1, m2, half1, half2;
                m1 = (min(d) + max(d)) / 2;
                m2 = (other.min(d) + other.max(d)) / 2;
                half1 = dist[d] / 2;
                half2 = other.dist[d] / 2;
                result = (half1 + half2) - std::abs(m1 - m2) >= 0;
            */
            // if two segments overlap? 
            result = false;
            result = result || (other.min(d) >= min(d) && other.min(d) <= max(d));
            result = result || (other.max(d) >= min(d) && other.max(d) <= max(d));
            result = result || (min(d) >= other.min(d) && min(d) <= other.max(d));
            result = result || (max(d) >= other.min(d) && max(d) <= other.max(d));
            if (!result)
                return false;
        }

        return true;
    }
};


} // end namespace gpu
} // end namespace rd
