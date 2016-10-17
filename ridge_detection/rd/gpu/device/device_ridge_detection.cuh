/**
 * @file device_ridge_detection.cuh
 * @author Adam Rogowiec
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

#pragma once

#include "rd/gpu/device/dispatch/dispatch_ridge_detection.cuh"

#include "cub/util_type.cuh"

namespace rd 
{
namespace gpu
{
namespace tiled
{

//-----------------------------------------------------------------------------
// Name traits
//-----------------------------------------------------------------------------

template <RidgeDetectionAlgorithm ALGORITHM>
struct RidgeDetectionAlgorithmNameTraits{};

template <>
struct RidgeDetectionAlgorithmNameTraits<RD_BRUTE_FORCE> 
{
    static constexpr const char* name = "RD_BRUTE_FORCE";
    static constexpr const char* shortName = "BF";
};

template <>
struct RidgeDetectionAlgorithmNameTraits<RD_KD_TREE> 
{
    static constexpr const char* name = "RD_KD_TREE";
    static constexpr const char* shortName = "KDT";
};

template <>
struct RidgeDetectionAlgorithmNameTraits<RD_BVH_TREE> 
{
    static constexpr const char* name = "RD_BVH_TREE";
    static constexpr const char* shortName = "BVHT";
};

/**
 * @brief      Dr Marek Rupniewski's ridge detection algorithm parallel implementation on GPU.
 */
struct DeviceRidgeDetection
{

    /**
     * @brief      Ridge detection algorithm on single tile's points set.
     *
     * @param      d_inputPoints          The tile's input points set
     * @param      d_neighbourPoints      The tile's neighbouring points set
     * @param      d_chosenPoints         Pointer to memory for chosen points.
     * @param[in]  inPointsNum            Number of input points
     * @param[in]  neighbourPointsNum     Number of neighbouring points
     * @param[out] d_chosenPointsNum      Pointer to output result chosen points number.
     * @param[in]  r1                     Ridge detection algorithm parameter. Radius used for
     *                                    chosing points and in evolution.
     * @param[in]  r2                     Ridge detection algorithm parameter. Radius used for
     *                                    searching neighbours in decimation phase.
     * @param[in]  inPointsStride         Distance between point's subsequent coordinates (in terms
     *                                    of values count)
     * @param[in]  neighbourPointsStride  Distance between point's subsequent coordinates (in terms
     *                                    of values count)
     * @param[in]  chosenPointsStride     Distance between point's subsequent coordinates (in terms
     *                                    of values count)
     * @param[in]  debugSynchronous       Wheather or not to synchronize after each kernel launch
     *                                    for error checking. Also entails printing launch
     *                                    configuration informations before each kernel launch.
     *
     * @tparam     DIM                    Point dimension
     * @tparam     IN_MEM_LAYOUT          Input data (point cloud) memory layout (ROW/COL_MAJOR)
     * @tparam     OUT_MEM_LAYOUT         Output data (chosen points) memory layout (ROW/COL_MAJOR)
     * @tparam     RD_ALGORITHM           Algorithm to use for evolution phase.
     * @tparam     T                      Point coordinate data type
     *
     * @return     Cuda runtime error, if any, otherwise cudaSuccess
     */
    template <
        int                     DIM,
        DataMemoryLayout        IN_MEM_LAYOUT,
        DataMemoryLayout        OUT_MEM_LAYOUT,
        RidgeDetectionAlgorithm RD_ALGORITHM,
        typename                T>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t approximate(
        T const *   d_inputPoints,
        T const *   d_neighbourPoints,
        T *         d_chosenPoints,
        int         inPointsNum,
        int         neighbourPointsNum,
        int *       d_chosenPointsNum,
        T           r1,
        T           r2,
        int         inPointsStride,
        int         neighbourPointsStride,
        int         chosenPointsStride,
        bool        debugSynchronous = false)
    {
        return doApproximate<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
            d_inputPoints, d_neighbourPoints, d_chosenPoints, inPointsNum, neighbourPointsNum, 
            d_chosenPointsNum, r1, r2, inPointsStride, neighbourPointsStride, chosenPointsStride, 
            debugSynchronous, cub::Int2Type<RD_ALGORITHM>());
    }

};

} // end namespace tiled
} // end namespace gpu
} // end namespace rd
  