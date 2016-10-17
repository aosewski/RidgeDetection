/**
 * @file dispatch_tiled_decimate.cuh
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

#include "rd/gpu/device/tiled/tiled_tree_node.cuh"
#include "rd/gpu/device/tiled/decimate.cuh"

#include "rd/gpu/util/dev_utilities.cuh"
#include "rd/utils/memory.h"

#include "cub/util_debug.cuh"
#include "cub/util_device.cuh"

namespace rd
{
namespace gpu
{
namespace tiled
{

namespace detail
{

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

template <
    int                 DIM,
    int                 BLOCK_SIZE,
    DataMemoryLayout    MEM_LAYOUT,
    typename            T>
__launch_bounds__ (BLOCK_SIZE)
__global__ void deviceDecimateKernel(
    TiledTreeNode<DIM, T> **    d_leafNodes,
    int                         leafNum,
    int *                       d_chosenPointsNum,
    T                           r)
{
    typedef BlockDecimate<DIM, BLOCK_SIZE, MEM_LAYOUT, T> BlockDecimateT;
    BlockDecimateT().globalDecimate(d_leafNodes, leafNum, d_chosenPointsNum, r);
}

} // end namespace detail

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceDecimate
 */
template <
    typename            T,
    int                 DIM,
    DataMemoryLayout    MEM_LAYOUT>
struct DispatchDecimate
{
    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------

    enum 
    {  
        // maximum available smem storage in bytes (assumes appropriate
        // cache configuration) per block of threads
        MAX_SMEM_USAGE = 48 * 1024,
    };


    /// SM500
    template <
        typename            _T,
        int                 DUMMY>
    struct Policy500
    {
        enum 
        { 
            PTX_ARCH                        = 500,
            NOMINAL_BLOCK_SIZE              = 512,
            NOMINAL_COUNT_PTS_SMEM_USAGE    = NOMINAL_BLOCK_SIZE * DIM * sizeof(T) + sizeof(int),
            NOMINAL_SMEM_USAGE              = NOMINAL_COUNT_PTS_SMEM_USAGE,
            
            SCALED_BLOCK_SIZE               = (MAX_SMEM_USAGE - sizeof(int)) / (DIM * sizeof(T)),
            SCALED_BLOCK_SIZE2              = CUB_MAX(32, SCALED_BLOCK_SIZE / CUB_WARP_THREADS(
                                                PTX_ARCH) * CUB_WARP_THREADS(PTX_ARCH)),

            EXCEED_MAX_SMEM                 = (NOMINAL_SMEM_USAGE > MAX_SMEM_USAGE) ? 1 : 0,

            // if necessary use scaled down block size
            BLOCK_SIZE                      = (EXCEED_MAX_SMEM == 1) ? SCALED_BLOCK_SIZE2 :
                                                 NOMINAL_BLOCK_SIZE,

        };
    };

    /// SM350
    template <
        typename            _T,
        int                 DUMMY>
    struct Policy350
    {
        enum 
        { 
            PTX_ARCH                        = 350,
            NOMINAL_BLOCK_SIZE              = 512,
            NOMINAL_COUNT_PTS_SMEM_USAGE    = NOMINAL_BLOCK_SIZE * DIM * sizeof(T) + sizeof(int),
            NOMINAL_SMEM_USAGE              = NOMINAL_COUNT_PTS_SMEM_USAGE,
            
            SCALED_BLOCK_SIZE               = (MAX_SMEM_USAGE - sizeof(int)) / (DIM * sizeof(T)),
            SCALED_BLOCK_SIZE2              = CUB_MAX(32, SCALED_BLOCK_SIZE / CUB_WARP_THREADS(
                                                PTX_ARCH) * CUB_WARP_THREADS(PTX_ARCH)),

            EXCEED_MAX_SMEM                 = (NOMINAL_SMEM_USAGE > MAX_SMEM_USAGE) ? 1 : 0,

            // if necessary use scaled down block size
            BLOCK_SIZE                      = (EXCEED_MAX_SMEM == 1) ? SCALED_BLOCK_SIZE2 :
                                                 NOMINAL_BLOCK_SIZE,
        };
    };

    // XXX: Untested configuration
    /// SM300
    template <
        typename            _T,
        int                 DUMMY>
    struct Policy300
    {
        enum 
        { 
            BLOCK_SIZE = 512
        };
    };

    // XXX: Untested configuration
    /// SM210
    template <
        typename            _T,
        int                 DUMMY>
    struct Policy210
    {
        enum 
        { 
            BLOCK_SIZE = 512
        };
    };

    
    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

    #if (CUB_PTX_ARCH >= 500)
        typedef Policy500<T, 0> PtxPolicy;
    #elif (CUB_PTX_ARCH >= 350)
        typedef Policy350<T, 0> PtxPolicy;
    #elif (CUB_PTX_ARCH >= 300)
        typedef Policy300<T, 0> PtxPolicy;
    #else
        typedef Policy210<T, 0> PtxPolicy;
    #endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxAgentDecimatePolicy : PtxPolicy {};

    //---------------------------------------------------------------------
    // Dispatch entrypoints
    //---------------------------------------------------------------------

    /**
     * Internal dispatch routine for computing a block-wide decimate using the
     * specified kernel functions.
     */

    __device__ __forceinline__
    static void dispatch(
        TiledTreeNode<DIM, T> **    d_leafNodes,
        int                         leafNum,
        int *                       d_chosenPointsNum,
        T                           r,
        cudaStream_t                stream,
        bool                        debugSynchronous)
    {
        // Log decimateKernel configuration
        if (debugSynchronous)
        {
         _CubLog("Invoke decimateKernel<<<1, %d, 0, %p>>>(), leafNum: %d, d_chosenPointsNum: %d\n",
            PtxAgentDecimatePolicy::BLOCK_SIZE, stream, leafNum, *d_chosenPointsNum);
        }

        // Invoke decimateKernel
        detail::deviceDecimateKernel<DIM, PtxAgentDecimatePolicy::BLOCK_SIZE, MEM_LAYOUT>
            <<<1, PtxAgentDecimatePolicy::BLOCK_SIZE, 0, stream>>>(
                d_leafNodes, 
                leafNum, 
                d_chosenPointsNum, 
                r);

        // Check for failure to launch
        rdDevCheckCall(cudaPeekAtLastError());

        // Sync the stream if specified to flush runtime errors
        if (debugSynchronous)
        {
            rdDevCheckCall(cudaDeviceSynchronize());
        } 
    }

    static __host__ cudaError_t setCacheConfig()
    {
        cudaError error = cudaSuccess;
        do
        {
            int ptxVersion = 0;
            if (CubDebug(error = cub::PtxVersion(ptxVersion))) break;

            if (CubDebug(error = cudaFuncSetCacheConfig(detail::deviceDecimateKernel<DIM, 
                PtxAgentDecimatePolicy::BLOCK_SIZE, MEM_LAYOUT, T>, 
                cudaFuncCachePreferShared))) break;
        }
        while (0);

        return error;
    }
};

} // end namespace tiled
} // end namespace gpu
} // end namespace rd
