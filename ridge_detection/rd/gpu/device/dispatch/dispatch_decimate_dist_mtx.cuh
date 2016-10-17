/**
 * @file dispatch_decimate_dist_mtx.cuh
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

#include <stdexcept>

#include "cub/util_debug.cuh"
#include "cub/util_device.cuh"
#include "rd/gpu/device/brute_force/decimate_dist_mtx.cuh"

namespace rd
{
namespace gpu
{
namespace bruteForce
{

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceDecimateDistMtx
 */
template <
    typename            T,
    int                 DIM,
    DataMemoryLayout    MEM_LAYOUT>
struct DispatchDecimateDistMtx
{
    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------

    /// SM500
    template <
        typename            _T,
        int                 DUMMY>
    struct Policy500
    {
        enum 
        { 
            BLOCK_SIZE = 1024,
        };

        typedef DecimateDistMtxPolicy<BLOCK_SIZE> DecimateDistMtxPolicyT;
    };

    /// SM350
    template <
        typename            _T,
        int                 DUMMY>
    struct Policy350
    {
        enum 
        { 
            BLOCK_SIZE = 1024,
        };
        typedef DecimateDistMtxPolicy<BLOCK_SIZE> DecimateDistMtxPolicyT;
    };

    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

    #if (CUB_PTX_ARCH >= 500)
        typedef Policy500<T, 0> PtxPolicy;
    #else /*(CUB_PTX_ARCH >= 350)*/
        typedef Policy350<T, 0> PtxPolicy;
    #endif

    struct PtxDecimateDistMtxPolicy : PtxPolicy::DecimateDistMtxPolicyT {};


    //---------------------------------------------------------------------
    // Utilities
    //---------------------------------------------------------------------

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptxVersion,
        KernelConfig    &decimateConfig)
    {
    #if (CUB_PTX_ARCH >= 350)

        // We're on the device, so initialize the kernel dispatch configurations with 
        // the current PTX policy
        decimateConfig.template Init<PtxPolicy>();
    #elif (CUB_PTX_ARCH == 0)
        // We're on the host, so lookup and initialize the kernel dispatch configurations 
        // with the policies that match the device's PTX version
        if (ptxVersion >= 500)
        {
            decimateConfig.template Init<typename Policy500<T, 0>
                ::DecimateDistMtxPolicyT>();
        }
        else if (ptxVersion >= 350)
        {
            decimateConfig.template Init<typename Policy350<T, 0>
                ::DecimateDistMtxPolicyT>();
        }
        else
        {
            throw std::runtime_error("Unsupported cuda architecture!");
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
     * @brief      { function_description }
     *
     * @param      d_chosenPoints      The d chosen points
     * @param      chosenPointsNum     The chosen points number
     * @param[in]  chPtsStride         The ch points stride
     * @param      d_distMtx           The pointer to distance mtx
     * @param[in]  distMtxStride       The distance mtx stride
     * @param      d_pointsMask        The d points mask
     * @param[in]  r                   { parameter_description }
     * @param[in]  stream              stream CUDA stream to launch kernels within.  Default is
     *                                 stream<sub>0</sub>.
     * @param[in]  debugSynchronous    Whether or not to synchronize the stream after every kernel
     *                                 launch to check for errors.  Also causes launch
     *                                 configurations to be printed to the console.  Default is @p
     *                                 false.
     * @param[in]  decimateKernel      Kernel function pointer to parameterization of rd::decimateDistMtx
     * @param[in]  decimateConfig      Dispatch parameters that match the policy that @p decimateKernel was compiled for
     *
     * @tparam     DecimateKernelPtrT  { description }
     *
     * @return     
     */
    template <
        typename            DecimateKernelPtrT>    
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        T *                 d_chosenPoints,
        int *               chosenPointsNum,
        int                 chPtsStride,
        T *                 d_distMtx,
        int                 distMtxStride,
        char *              d_pointsMask,
        T                   r, 
        cudaStream_t        stream,              ///< [in] 
        bool                debugSynchronous,    ///< [in] 
        DecimateKernelPtrT  decimateKernel,      ///< [in] 
        KernelConfig        decimateConfig)      ///< [in] 
    {

        cudaError_t error = cudaSuccess;
        #if defined(CUB_RUNTIME_ENABLED) && (CUB_PTX_ARCH >= 350)
        do
        {
            // Get grid size for scanning tiles
            dim3 deviceGridSize(1);

            // Log decimateKernel configuration
            if (debugSynchronous) 
            {
                _CubLog("Invoke decimateDistMtx<<<%d, %d, 0, %p>>>(chPts: %p, chPtsNum: %p, "
                    "chPtsStride: %d, distMtx: %p, distMtxStride: %d, ptsMask: %p, r: %f)\n",
                    deviceGridSize.x, decimateConfig.blockThreads, stream, d_chosenPoints,
                    chosenPointsNum, chPtsStride, d_distMtx, distMtxStride, d_pointsMask, r);
            }

            // Invoke decimateKernel
            decimateKernel<<<deviceGridSize, decimateConfig.blockThreads, 0, stream>>>(
                d_chosenPoints, chosenPointsNum, chPtsStride, d_distMtx, distMtxStride, 
                d_pointsMask, r);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debugSynchronous && (CubDebug(error = cub::SyncStream(stream)))) break;

        }
        while (0);
        #else
            // Kernel launch not supported from this device
            error = cudaErrorNotSupported;
        #endif  // CUB_RUNTIME_ENABLED
        return error;
    }


   /**
     * Dispatch device decimate distance matrix kernel
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t dispatch(
        T *          d_chosenPoints,
        int *        chosenPointsNum,
        int          chPtsStride,
        T *          d_distMtx,
        int          distMtxStride,
        char *       d_pointsMask,
        T            r, 
        cudaStream_t stream,
        bool         debugSynchronous)
    {
        cudaError_t error = cudaSuccess;
        // enable only on devices with CC>=3.5 and with device runtime api enabled
        #if defined(CUB_RUNTIME_ENABLED) && (CUB_PTX_ARCH >= 350)
            
            // Get kernel kernel dispatch configurations
            KernelConfig decimateConfig;
            InitConfigs(CUB_PTX_ARCH, decimateConfig);

            // Dispatch
            CubDebug(error = Dispatch(
                d_chosenPoints,
                chosenPointsNum,
                chPtsStride,
                d_distMtx,
                distMtxStride,
                d_pointsMask,
                r, 
                stream,
                debugSynchronous,
                decimateDistMtx<DIM, 
                    PtxDecimateDistMtxPolicy::BLOCK_SIZE, MEM_LAYOUT, T>,
                decimateConfig));

        #else
            // Kernel launch not supported from this device
            error = cudaErrorNotSupported;
        #endif
        return error;
    }

    static __host__ cudaError_t setCacheConfig()
    {
        cudaError error = cudaSuccess;
        do
        {
            int ptxVersion;
            if (CubDebug(error = cub::PtxVersion(ptxVersion))) break;

            if (ptxVersion >= 500)
            {
                if (CubDebug(error = cudaFuncSetCacheConfig(decimateDistMtx<DIM, 
                    Policy500<T, 0>::DecimateDistMtxPolicyT::BLOCK_SIZE, MEM_LAYOUT, T>, 
                    cudaFuncCachePreferL1))) break;

                error = DeviceDistanceMtx::setCacheConfig<MEM_LAYOUT, T>();
                if (CubDebug(error)) break;
            }
            else if (ptxVersion >= 350)
            {
                if (CubDebug(error = cudaFuncSetCacheConfig(decimateDistMtx<DIM, 
                    Policy350<T, 0>::DecimateDistMtxPolicyT::BLOCK_SIZE, MEM_LAYOUT, T>, 
                    cudaFuncCachePreferL1))) break;

                error = DeviceDistanceMtx::setCacheConfig<MEM_LAYOUT, T>();
                if (CubDebug(error)) break;
            }
            else
            {
                error = cudaErrorNotSupported;
            }

        }
        while (0);

        return error;
    }
};

}   // end namespace bruteForce
}   // end namespace gpu
}   // end namespace rd


