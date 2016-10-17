/**
 * @file device_decimate.cuh
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

#pragma once

#include "rd/gpu/device/dispatch/dispatch_decimate.cuh"
#include "rd/gpu/device/dispatch/dispatch_decimate_dist_mtx.cuh"

namespace rd
{
namespace gpu
{
namespace bruteForce
{

struct DeviceDecimate
{

    /**
     * @brief      Removes redundant points from set @p d_S
     *
     * @param      d_chosenPoints     Pointer to input chosen points
     * @param      chosenPointsNum    Number of chosen points
     * @param[in]  r                  Ridge detection paramter. Radius within which we search for
     *                                neighbours.
     * @param[in]  stride             Input data stride. Distance beetween point subsequent
     *                                coordinates.
     * @param[in]  stream             The stream we should run computations with.
     * @param[in]  debug_synchronous  Wheather or not to synchronize the stream after each kernel
     *                                launch to check for errors.
     *
     * @tparam     DIM                Input points dimension
     * @tparam     MEM_LAYOUT         Input data memory layout (ROW/COL_MAJOR)
     * @tparam     T                  Point coordinate data type.
     *
     * @return     Error if any or cudaSuccess.
     */
    template <
        int                 DIM,
        DataMemoryLayout    MEM_LAYOUT,
        typename            T>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t decimate(
            T *          d_chosenPoints,
            int *        chosenPointsNum,
            T            r, 
            int          stride,
            cudaStream_t stream = nullptr,
            bool         debug_synchronous = false)
    {
        return DispatchDecimate<T, DIM, MEM_LAYOUT>::dispatch(
            d_chosenPoints,
            chosenPointsNum,
            r,
            stride,
            stream,
            debug_synchronous);
    }

    /**
     * @brief      Removes redundant points from set @p d_S
     *
     * @param      d_chosenPoints     Pointer to input chosen points
     * @param      chosenPointsNum    Number of chosen points
     * @param[in]  r                  Ridge detection paramter. Radius within which we search for
     *                                neighbours.
     * @param[in]  stride             Input data stride. Distance beetween point subsequent
     *                                coordinates.
     * @param[in]  stream             The stream we should run computations with.
     * @param[in]  debug_synchronous  Wheather or not to synchronize the stream after each kernel
     *                                launch to check for errors.
     *
     * @tparam     DIM                Input points dimension
     * @tparam     MEM_LAYOUT         Input data memory layout (ROW/COL_MAJOR)
     * @tparam     T                  Point coordinate data type.
     *
     * @return     Error if any or cudaSuccess.
     */
    template <
        int                 DIM,
        DataMemoryLayout    MEM_LAYOUT,
        typename            T>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t decimateDistMtx(
            T *          d_chosenPoints,
            int *        chosenPointsNum,
            int          chPtsStride,
            T *          d_distMtx,
            int          distMtxStride,
            char *       d_pointsMask,
            T            r, 
            cudaStream_t stream = nullptr,
            bool         debug_synchronous = false)
    {
        return DispatchDecimateDistMtx<T, DIM, MEM_LAYOUT>::dispatch(
            d_chosenPoints,
            chosenPointsNum,
            chPtsStride,
            d_distMtx,
            distMtxStride,
            d_pointsMask,
            r,
            stream,
            debug_synchronous);
    }

    template <
        int                 DIM,
        DataMemoryLayout    MEM_LAYOUT,
        typename            T>
    static __host__ cudaError_t setDecimateCacheConfig()
    {
        return DispatchDecimate<T, DIM, MEM_LAYOUT>::setCacheConfig();
    }

    template <
        int                 DIM,
        DataMemoryLayout    MEM_LAYOUT,
        typename            T>
    static __host__ cudaError_t setDecimateDistMtxCacheConfig()
    {
        return DispatchDecimateDistMtx<T, DIM, MEM_LAYOUT>::setCacheConfig();
    }
};


} // bruteForce namespace
} // gpu namespace
} // rd namespace

