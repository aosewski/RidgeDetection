/**
 * @file device_tiled_decimate.cuh
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

#include "rd/gpu/device/dispatch/dispatch_tiled_decimate.cuh"
#include "rd/gpu/device/tiled/tiled_tree_node.cuh"

namespace rd
{
namespace gpu
{
namespace tiled
{

struct DeviceDecimate
{

    /**
     * @brief      Removes redundant points from leaf nodes
     *
     * @param      d_leafNodes        Table of pointers to tree's leaf nodes.
     * @param[in]  leafNum            Number of leaf nodes.
     * @param      d_chosenPointsNum  Pointer to global chosen points counter.
     * @param[in]  r                  Ridge detection paramter. Radius within which we search for
     *                                neighbours.
     * @param[in]  stream             The stream we should run computations with.
     * @param[in]  debug_synchronous  Wheather or not to synchronize the stream after each kernel
     *                                launch to check for errors.
     *
     * @tparam     DIM                Input points dimension
     * @tparam     MEM_LAYOUT         Input data memory layout (ROW/COL_MAJOR)
     * @tparam     T                  Point coordinate data type.
     */
    template <
        int                 DIM,
        DataMemoryLayout    MEM_LAYOUT,
        typename            T>
    __device__ __forceinline__
    static void globalDecimate(
        TiledTreeNode<DIM, T> **    d_leafNodes,
        int                         leafNum,
        int *                       d_chosenPointsNum,
        T                           r,
        cudaStream_t                stream,
        bool                        debug_synchronous = false)
    {
        DispatchDecimate<T, DIM, MEM_LAYOUT>::dispatch(
            d_leafNodes, leafNum, d_chosenPointsNum, r, stream, debug_synchronous);
    }

    template <
        int                 DIM,
        DataMemoryLayout    MEM_LAYOUT,
        typename            T>
    static __host__ cudaError_t setDecimateCacheConfig()
    {
        return DispatchDecimate<T, DIM, MEM_LAYOUT>::setCacheConfig();
    }
};

} // tiled namespace
} // gpu namespace
} // rd namespace
