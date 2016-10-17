/**
 * @file dispatch_decimate.cuh
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

#ifndef __DISPATCH_DECIMATE_CUH__
#define __DISPATCH_DECIMATE_CUH__ 

#include "cub/util_debug.cuh"
#include "cub/util_device.cuh"
#include "rd/gpu/device/brute_force/decimate.cuh"

namespace rd
{
namespace gpu
{
namespace bruteForce
{

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

template <
    typename            DecimatePolicyT,
    typename            T,
    int                 DIM,
    DataMemoryLayout    MEM_LAYOUT>
__launch_bounds__ (int(DecimatePolicyT::BLOCK_SIZE))
__global__ void DeviceDecimateKernel(
        T *     S,
        int *   ns,
        T       r,
        int     stride)
{
    typedef BlockDecimate<T, DIM, DecimatePolicyT::BLOCK_SIZE, MEM_LAYOUT> BlockDecimateT;
    BlockDecimateT().decimate(S, ns, r, stride);
}


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

    /// SM500
    template <
        typename            _T,
        DataMemoryLayout    _MEM_LAYOUT,
        int                 DUMMY>
    struct Policy500
    {
        enum 
        { 
            BLOCK_SIZE = 1024,
        };

        typedef BlockDecimatePolicy<BLOCK_SIZE> DecimatePolicyT;
    };

    /// SM350
    template <
        typename            _T,
        DataMemoryLayout    _MEM_LAYOUT,
        int                 DUMMY>
    struct Policy350
    {
        enum 
        { 
            BLOCK_SIZE = 1024,
        };

        typedef BlockDecimatePolicy<BLOCK_SIZE> DecimatePolicyT;
    };

    /// SM210
    template <
        typename            _T,
        DataMemoryLayout    _MEM_LAYOUT,
        int                 DUMMY>
    struct Policy210
    {
        enum 
        { 
            BLOCK_SIZE = 1024,
        };

        typedef BlockDecimatePolicy<BLOCK_SIZE> DecimatePolicyT;
    };


    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

    #if (CUB_PTX_ARCH >= 500)
        typedef Policy500<T, MEM_LAYOUT, 0> PtxPolicy;
    #elif (CUB_PTX_ARCH >= 350)
        typedef Policy350<T, MEM_LAYOUT, 0> PtxPolicy;
    #else
        typedef Policy210<T, MEM_LAYOUT, 0> PtxPolicy;
    #endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxAgentDecimatePolicy : PtxPolicy::DecimatePolicyT {};


    //---------------------------------------------------------------------
    // Utilities
    //---------------------------------------------------------------------

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitDecimateConfigs(
        int             ptxVersion,
        KernelConfig    &decimateConfig)
    {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations 
        // with the current PTX policy
        decimateConfig.template Init<PtxAgentDecimatePolicy>();
    #else
        // We're on the host, so lookup and initialize the kernel dispatch configurations 
        // with the policies that match the device's PTX version
        if (ptxVersion >= 500)
        {
            decimateConfig.template Init<typename Policy500<T, MEM_LAYOUT, 0>
                ::DecimatePolicyT>();
        }
        else if (ptxVersion >= 350)
        {
            decimateConfig.template Init<typename Policy350<T, MEM_LAYOUT, 0>
                ::DecimatePolicyT>();
        }
        else 
        {
            decimateConfig.template Init<typename Policy210<T, MEM_LAYOUT, 0>
                ::DecimatePolicyT>();
        }
    #endif
    }

    /**
     * Kernel kernel dispatch configuration.
     */
    struct KernelConfig
    {
        int blockThreads;

        template <typename PolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            blockThreads       = PolicyT::BLOCK_SIZE;
        }
    };

    //---------------------------------------------------------------------
    // Dispatch entrypoints
    //---------------------------------------------------------------------

    /**
     * Internal dispatch routine for computing a device-wide decimate using the
     * specified kernel functions.
     */

    template <
        typename            DecimateKernelPtrT>    
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        T *                 d_S,
        int *               ns,
        T                   r, 
        int                 stride,
        cudaStream_t        stream,              
        bool                debugSynchronous,    
        DecimateKernelPtrT  decimateKernel,      
        KernelConfig        decimateConfig)      
    {

        #ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

        #else
        cudaError error = cudaSuccess;
        do
        {
            // Get grid size for scanning tiles
            dim3 deviceGridSize(1);

            // Log decimateKernel configuration
            if (debugSynchronous) 
            {
                _CubLog("Invoke decimateKernel<<<%d, %d, 0, %p>>>(d_S: %p, ns: %p, r: %f, "
                    "stride: %d)\n",
                    deviceGridSize.x, decimateConfig.blockThreads, stream, d_S, ns, r, stride);
            }

            // Invoke decimateKernel
            decimateKernel<<<deviceGridSize, decimateConfig.blockThreads, 0, stream>>>(
                d_S,
                ns,
                r, 
                stride);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debugSynchronous && (CubDebug(error = cub::SyncStream(stream)))) break;
        }
        while (0);

        return error;

        #endif  // CUB_RUNTIME_ENABLED
    }

    /**
     * Internal dispatch routine
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t dispatch(
        T *             d_S,
        int *           ns,
        T               r, 
        int             stride,
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

            // Get kernel kernel dispatch configurations
            KernelConfig decimateConfig;
            InitDecimateConfigs(ptxVersion, decimateConfig);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_S,
                ns,
                r,
                stride,
                stream,
                debugSynchronous,
                DeviceDecimateKernel<PtxAgentDecimatePolicy, T, DIM, MEM_LAYOUT>,
                decimateConfig))) break;
        }
        while (0);

        return error;
    }

    static __host__ cudaError_t setCacheConfig()
    {
        cudaError error = cudaSuccess;
        do
        {
            int ptxVersion;
            if (CubDebug(error = cub::PtxVersion(ptxVersion))) break;

            if (CubDebug(error = cudaFuncSetCacheConfig(DeviceDecimateKernel<
                PtxAgentDecimatePolicy, T, DIM, MEM_LAYOUT>, 
                cudaFuncCachePreferShared))) break;
        }
        while (0);

        return error;
    }
};

}   // end namespace bruteForce
}   // end namespace gpu
}   // end namespace rd


#endif // __DISPATCH_DECIMATE_CUH__