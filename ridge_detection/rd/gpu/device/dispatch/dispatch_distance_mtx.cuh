/**
 * @file dispatch_distance_mtx.cuh
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

#include "rd/utils/memory.h"
#include "rd/gpu/agent/agent_dist_mtx.cuh"

#include "cub/util_debug.cuh"
#include "cub/util_device.cuh"
#include "cub/util_arch.cuh"
#include "cub/util_type.cuh"


namespace rd
{
namespace gpu
{
namespace detail
{


/******************************************************************************
 * Symmetric distance matrix entry point
 *****************************************************************************/

template<
    typename            SymDistMtxPolicyT,
    DataMemoryLayout    MEM_LAYOUT,
    typename            T>
__launch_bounds__ (int(SymDistMtxPolicyT::BLOCK_H * SymDistMtxPolicyT::BLOCK_W))
__global__ void symDistMtxKernel(
    T const * __restrict__  d_in,
    T *                     d_out,
    int                     width,
    int                     height,
    int                     inStride,
    int                     outStride)
{
    typedef AgentDistMtx<
        SymDistMtxPolicyT::BLOCK_W, 
        SymDistMtxPolicyT::BLOCK_H, 
        MEM_LAYOUT, 
        T> AgentT;

    typedef typename AgentT::TempStorage TempStorageT;
    __shared__ TempStorageT smem;

    AgentT(smem).symDist(d_in, d_out, width, height, inStride, outStride);
}

} // end namespace detail
  
template <
    DataMemoryLayout    MEM_LAYOUT,
    typename            T>
struct DispatchDistanceMtx
{
    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------

    struct Policy500
    {
        enum 
        { 
            BLOCK_H = 4,
            BLOCK_W = 32,
        };
    };
    struct Policy350
    {
        enum 
        { 
            BLOCK_H = 4,
            BLOCK_W = 32,
        };
    };
    struct Policy210
    {
        enum 
        { 
            BLOCK_H = 4,
            BLOCK_W = 32,
        };
    };


    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

    #if (CUB_PTX_ARCH >= 500)
        typedef Policy500 PtxPolicy;
    #elif (CUB_PTX_ARCH >= 350)
        typedef Policy350 PtxPolicy;
    #else
        typedef Policy210 PtxPolicy;
    #endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxDistanceMtxPolicy : PtxPolicy {};
 
    //---------------------------------------------------------------------
    // Utilities
    //---------------------------------------------------------------------

    /**
     * Initialize kernel dispatch configurations with the policies corresponding 
     * to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptxVersion,
        KernelConfig    &distMtxConfig)
    {
        #if (CUB_PTX_ARCH > 0)

            // We're on the device, so initialize the kernel dispatch configurations with the 
            // current PTX policy
            distMtxConfig.template init<PtxDistanceMtxPolicy>();
        #else
            // We're on the host, so lookup and initialize the kernel dispatch configurations 
            // with the policies that match the device's PTX version
            if (ptxVersion >= 500)
            {
                distMtxConfig.template init<Policy500>();
            }
            else if (ptxVersion >= 350)
            {
                distMtxConfig.template init<Policy350>();
            }
            else
            {
                distMtxConfig.template init<Policy210>();
            }
        #endif
    }

    /**
     * Kernel kernel dispatch configuration.
     */
    struct KernelConfig
    {
        int blockThreadsW;
        int blockThreadsH;
        int blockThreads;

        template <typename PolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        void init()
        {
            blockThreadsW       = PolicyT::BLOCK_W;
            blockThreadsH       = PolicyT::BLOCK_H;
            blockThreads        = blockThreadsH * blockThreadsW;
        }
    };


    //---------------------------------------------------------------------
    // invocation 
    //---------------------------------------------------------------------
    
    template <typename    SymDistMtxKernelPtrT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t invoke(
        T const *       d_in,
        T *             d_out,
        const int       width,
        const int       height,
        const int       inStride,
        const int       outStride,
        cudaStream_t    stream,
        bool            debugSynchronous,
        SymDistMtxKernelPtrT kernelPtr,
        KernelConfig    kernelConfig,
        int             ptxVersion)
    {
        #ifndef CUB_RUNTIME_ENABLED
            // Kernel launch not supported from this device
            return CubDebug(cudaErrorNotSupported);
        #else

        cudaError error = cudaSuccess;
        do
        {           
            // Get device ordinal
            int deviceOrdinal;
            if (CubDebug(error = cudaGetDevice(&deviceOrdinal))) break;

            // Get SM count
            int smCount;
            if (CubDebug(error = cudaDeviceGetAttribute(&smCount, 
                cudaDevAttrMultiProcessorCount, deviceOrdinal))) break;

            // get SM occupancy
            int smOccupancy;
            if (CubDebug(error = cub::MaxSmOccupancy(
                smOccupancy,
                kernelPtr,
                kernelConfig.blockThreads)
            )) break;

            int blockCount = smCount * smOccupancy * CUB_SUBSCRIPTION_FACTOR(ptxVersion);
            float root = sqrtf(float(blockCount));
            int rc = static_cast<int>(ceilf(root) + 1);
            int rf = static_cast<int>(floorf(root) + 1);

            dim3 dimGrid(1), dimBlock(1);
            dimGrid.x = rc;
            dimGrid.y = rf;
            dimBlock.x = kernelConfig.blockThreadsW;
            dimBlock.y = kernelConfig.blockThreadsH;

            if (debugSynchronous)
            {
                _CubLog("Invoking DistanceMtxKernel<<<(%d, %d), (%d, %d), 0, %p>>>\n",
                    dimGrid.y, dimGrid.x, dimBlock.y, dimBlock.x, stream);
            }

            kernelPtr<<<dimGrid, dimBlock, 0, stream>>>(
                d_in, d_out, width, height, inStride, outStride);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;
            // Sync the stream if specified to flush runtime errors
            if (debugSynchronous && (CubDebug(error = cub::SyncStream(stream)))) break;

        } while (0);

        return error;

        #endif // CUB_RUNTIME_ENABLED
    }

    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t dispatch(
        T const *       d_in,
        T *             d_out,
        const int       width,
        const int       height,
        const int       inStride,
        const int       outStride,
        cudaStream_t    stream,
        bool            debugSynchronous)
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptxVersion;
            #if (CUB_PTX_ARCH == 0)
                if (CubDebug(error = cub::PtxVersion(ptxVersion))) break;
            #else
                ptxVersion = CUB_PTX_ARCH;
            #endif

            KernelConfig distanceMtxConfig_;

            // Get kernel dispatch configurations
            InitConfigs(ptxVersion, distanceMtxConfig_);

            if (CubDebug(error = invoke(
                d_in,
                d_out,
                width,
                height, 
                inStride,
                outStride,
                stream,
                debugSynchronous,
                detail::symDistMtxKernel<PtxDistanceMtxPolicy, MEM_LAYOUT, T>,
                distanceMtxConfig_,
                ptxVersion ))) break;
        }
        while (0);

        return error;
    }

    static __host__ cudaError_t setCacheConfig()
    {
        cudaError error = cudaSuccess;
        do
        {
            if (CubDebug(error = cudaFuncSetCacheConfig(detail::symDistMtxKernel<
                PtxDistanceMtxPolicy, MEM_LAYOUT, T>, 
                cudaFuncCachePreferL1))) break;
        }
        while (0);

        return error;
    }

};
  
} // end namespace gpu
} // end namespace rd