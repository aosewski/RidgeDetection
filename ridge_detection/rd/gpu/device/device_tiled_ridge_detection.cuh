/**
 * @file device_tiled_ridge_detection.cuh
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

#include <iostream>

#include "rd/gpu/device/dispatch/dispatch_tiled_ridge_detection.cuh"
#include "rd/gpu/device/bounding_box.cuh"
#include "rd/utils/memory.h"

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


template <TileType TILE_TYPE>
struct TileTypeNameTraits{};

template <>
struct TileTypeNameTraits<RD_EXTENDED_TILE>
{
    static constexpr const char* name = "RD_EXTENDED_TILE";
    static constexpr const char* shortName = "EXTT";
};

template <TiledRidgeDetectionPolicy POLICY>
struct TiledRidgeDetectionPolicyNameTraits{};

template <>
struct TiledRidgeDetectionPolicyNameTraits<RD_LOCAL> 
{
    static constexpr const char* name = "RD_LOCAL";
    static constexpr const char* shortName = "LOCAL";
};

template <>
struct TiledRidgeDetectionPolicyNameTraits<RD_MIXED> 
{
    static constexpr const char* name = "RD_MIXED";
    static constexpr const char* shortName = "MIXED";
};


/**
 * @brief      Dr Marek Rupniewski's ridge detection algorithm parallel implementation on GPU.
 */
struct DeviceTiledRidgeDetection
{

    /**
     * @brief      Ridge detection algorithm using data partitioning on tiles technique to speed-up
     *             computations.
     *
     * @param[in]  d_inputPoints       The input points set
     * @param[out] d_chosenPoints      Placeholder for pointer to result chosen points. Memory is
     *                                 allocated internally, however user is responsible for
     *                                 deallocating it.
     * @param[in]  inPointsNum         Number of points inside input points set @p d_inputPoints
     * @param[out] d_chosenPointsNum   Pointer to output number of chosen points.
     * @param[in]  r1                  Ridge detection algorithm parameter. Radius used for chosing
     *                                 points and in evolution.
     * @param[in]  r2                  Ridge detection algorithm parameter. Radius used for
     *                                 searching neighbours in decimation phase.
     * @param[in]  inPointsStride      Distance between point's subsequent coordinates (in terms of
     *                                 values count)
     * @param[in]  chosenPointsStride  Distance between point's subsequent coordinates (in terms of
     *                                 values count)
     * @param[in]  maxTileCapacity     The maximum number of points which tile can contain.
     * @param      tilesPerDim         Number of initial tiles in respective dimension.
     * @param[in]  endPhaseRefinement  Wheather or not to perform results refinement at the end. It
     *                                 involves few additional, global evolve and decimation phases.
     * @param      d_rdTimers          Pointer to structure for storing results of ridge detection's
     *                                 specific parts timing.
     * @param[in]  debugSynchronous    Wheather or not to synchronize after each kernel launch for
     *                                 error checking. Also entails printing launch configuration
     *                                 informations before each kernel launch.
     *
     * @tparam     DIM                 Point dimension
     * @tparam     IN_MEM_LAYOUT       Input data (point cloud) memory layout (ROW/COL_MAJOR)
     * @tparam     OUT_MEM_LAYOUT      Output data (chosen points) memory layout (ROW/COL_MAJOR)
     * @tparam     RD_TILE_ALGORITHM   The algorithm to use when calculating ridge detection within
     *                                 tile.
     * @tparam     RD_TILE_POLICY      The policy in respect to tiles communication which will be
     *                                 used.
     * @tparam     RD_TILE_TYPE        Tile type which is used when partitioning input samples.
     * @tparam     T                   Point coordinate data type
     *
     * @par Overview
     *     This version uses Cuda Dynamic parallelism to build some kind of a tree structure, which
     *     will distribute data onto spatial tiles. Each tile contain points laying inside bounded
     *     volume. Next, in parallel, within each tile ridge detection is performed.
     */
    template <
        int                             DIM,
        DataMemoryLayout                IN_MEM_LAYOUT,
        DataMemoryLayout                OUT_MEM_LAYOUT,
        RidgeDetectionAlgorithm         RD_TILE_ALGORITHM,
        TiledRidgeDetectionPolicy       RD_TILE_POLICY,
        TileType                        RD_TILE_TYPE,
        typename                        T>
    static void approximate(
        T const *   d_inputPoints,
        T *&        d_chosenPoints,
        int         inPointsNum,
        int *       d_chosenPointsNum,
        T           r1,
        T           r2,
        int         inPointsStride,
        int &       chosenPointsStride,
        int         maxTileCapacity,
        int         tilesPerDim[DIM],
        bool        endPhaseRefinement,
        detail::TiledRidgeDetectionTimers * d_rdTimers,
        bool        debugSynchronous = false)
    {
        
        DispatchTiledRidgeDetection<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, RD_TILE_ALGORITHM,
            RD_TILE_POLICY, RD_TILE_TYPE, T>::dispatch(
                d_inputPoints,
                d_chosenPoints,
                inPointsNum,
                d_chosenPointsNum,
                r1,
                r2,
                inPointsStride,
                chosenPointsStride,
                maxTileCapacity,
                tilesPerDim,
                endPhaseRefinement,
                d_rdTimers,
                debugSynchronous);
    }
};



} // end namespace tiled
} // end namespace gpu
} // end namespace rd


