/**
 * @file dispatch_choose.cuh
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

#ifndef __DISPATCH_CHOOSE_CUH__
#define __DISPATCH_CHOOSE_CUH__ 

#include "cub/util_debug.cuh"
#include "cub/util_device.cuh"
#include "rd/gpu/device/brute_force/choose.cuh"

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
    typename          ChoosePolicyT,
    typename          T,
    int               DIM,
    DataMemoryLayout  INPUT_MEM_LAYOUT,
    DataMemoryLayout  OUTPUT_MEM_LAYOUT>
__launch_bounds__ (int(ChoosePolicyT::BLOCK_SIZE))
__global__ void DeviceChooseKernel(
        T const * __restrict__ P,
        T * S,
        int np,
        int *ns,
        T r,
        int pStride,
        int sStride)
{

    typedef BlockChoose<
        T, 
        DIM, 
        ChoosePolicyT::BLOCK_SIZE, 
        INPUT_MEM_LAYOUT, 
        OUTPUT_MEM_LAYOUT> BlockChooseT;

    BlockChooseT().choose(P, S, np, ns, r, pStride, sStride);
}



/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceChoose
 */
template <
    typename    T,
    int         DIM,
    DataMemoryLayout  INPUT_MEM_LAYOUT,
    DataMemoryLayout  OUTPUT_MEM_LAYOUT>
struct DispatchChoose
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
        DataMemoryLayout    _IN_MEM_LAYOUT,
        DataMemoryLayout    _OUT_MEM_LAYOUT,
        int                 DUMMY>
    struct Policy500
    {
        enum 
        { 
            PTX_ARCH                        = 500,
            NOMINAL_BLOCK_SIZE              = 512,
            NOMINAL_CHOOSE_SMEM_USAGE       = (_IN_MEM_LAYOUT == COL_MAJOR) ? 
                                                (NOMINAL_BLOCK_SIZE + 1) * DIM * sizeof(T) : 
                                                 NOMINAL_BLOCK_SIZE * DIM * sizeof(T),
            NOMINAL_COUNT_PTS_SMEM_USAGE    = NOMINAL_BLOCK_SIZE * DIM * sizeof(T) + sizeof(int),
            NOMINAL_SMEM_USAGE              = NOMINAL_CHOOSE_SMEM_USAGE + NOMINAL_COUNT_PTS_SMEM_USAGE,
            
            SCALED_BLOCK_SIZE               = (_IN_MEM_LAYOUT == COL_MAJOR) ?
                                                (MAX_SMEM_USAGE - DIM * sizeof(T) - sizeof(int)) /
                                                (DIM * sizeof(T) * 2) :
                                                (MAX_SMEM_USAGE - sizeof(int)) / (DIM * sizeof(T) * 2),
            SCALED_BLOCK_SIZE2              = SCALED_BLOCK_SIZE / CUB_WARP_THREADS(PTX_ARCH) * 
                                                CUB_WARP_THREADS(PTX_ARCH),

            // SCALED_BS_SMEM_USAGE            = (SCALED_BLOCK_SIZE2+1)*DIM*sizeof(T)+SCALED_BLOCK_SIZE2*DIM*sizeof(T),

            // if necessary use scaled down block size
            EXCEED_MAX_SMEM                 = (NOMINAL_SMEM_USAGE > MAX_SMEM_USAGE) ? 1 : 0,
            CHOOSE_BLOCK_SIZE               = (EXCEED_MAX_SMEM == 1) ? SCALED_BLOCK_SIZE2 : NOMINAL_BLOCK_SIZE,

        };

        typedef BlockChoosePolicy<CHOOSE_BLOCK_SIZE> ChoosePolicyT;
    };

    /// SM350
    template <
        typename            _T,
        DataMemoryLayout    _IN_MEM_LAYOUT,
        DataMemoryLayout    _OUT_MEM_LAYOUT,
        int                 DUMMY>
    struct Policy350
    {
        enum 
        { 
            PTX_ARCH                        = 350,
            NOMINAL_BLOCK_SIZE              = 512,
            NOMINAL_CHOOSE_SMEM_USAGE       = (_IN_MEM_LAYOUT == COL_MAJOR) ? 
                                                (NOMINAL_BLOCK_SIZE + 1) * DIM * sizeof(T) : 
                                                 NOMINAL_BLOCK_SIZE * DIM * sizeof(T),
            NOMINAL_COUNT_PTS_SMEM_USAGE    = NOMINAL_BLOCK_SIZE * DIM * sizeof(T) + sizeof(int),
            NOMINAL_SMEM_USAGE              = NOMINAL_CHOOSE_SMEM_USAGE + NOMINAL_COUNT_PTS_SMEM_USAGE,
            
            SCALED_BLOCK_SIZE               = (_IN_MEM_LAYOUT == COL_MAJOR) ?
                                                (MAX_SMEM_USAGE - DIM * sizeof(T) - sizeof(int)) /
                                                (DIM * sizeof(T) * 2) :
                                                (MAX_SMEM_USAGE - sizeof(int)) / (DIM * sizeof(T) * 2),
            SCALED_BLOCK_SIZE2              = CUB_MAX(32, SCALED_BLOCK_SIZE / CUB_WARP_THREADS(PTX_ARCH) * 
                                                CUB_WARP_THREADS(PTX_ARCH)),

            // SCALED_BS_SMEM_USAGE            = (SCALED_BLOCK_SIZE2+1)*DIM*sizeof(T)+SCALED_BLOCK_SIZE2*DIM*sizeof(T),

            EXCEED_MAX_SMEM                 = (NOMINAL_SMEM_USAGE > MAX_SMEM_USAGE) ? 1 : 0,
            CHOOSE_BLOCK_SIZE               = (EXCEED_MAX_SMEM == 1) ? SCALED_BLOCK_SIZE2 : NOMINAL_BLOCK_SIZE,
        };

        typedef BlockChoosePolicy<CHOOSE_BLOCK_SIZE> ChoosePolicyT;
    };

    /// SM210
    template <
        typename            _T,
        DataMemoryLayout    _IN_MEM_LAYOUT,
        DataMemoryLayout    _OUT_MEM_LAYOUT,
        int                 DUMMY>
    struct Policy210
    {
        enum 
        { 
            PTX_ARCH                        = 210,
            NOMINAL_BLOCK_SIZE              = 256,
            NOMINAL_CHOOSE_SMEM_USAGE       = (_IN_MEM_LAYOUT == COL_MAJOR) ? 
                                                (NOMINAL_BLOCK_SIZE + 1) * DIM * sizeof(T) : 
                                                 NOMINAL_BLOCK_SIZE * DIM * sizeof(T),
            NOMINAL_COUNT_PTS_SMEM_USAGE    = NOMINAL_BLOCK_SIZE * DIM * sizeof(T) + sizeof(int),
            NOMINAL_SMEM_USAGE              = NOMINAL_CHOOSE_SMEM_USAGE + NOMINAL_COUNT_PTS_SMEM_USAGE,
            
            SCALED_BLOCK_SIZE               = (_IN_MEM_LAYOUT == COL_MAJOR) ?
                                                (MAX_SMEM_USAGE - DIM * sizeof(T) - sizeof(int)) /
                                                (DIM * sizeof(T) * 2) :
                                                (MAX_SMEM_USAGE - sizeof(int)) / (DIM * sizeof(T) * 2),
            SCALED_BLOCK_SIZE2              = CUB_MAX(32, SCALED_BLOCK_SIZE / CUB_WARP_THREADS(PTX_ARCH) * 
                                                CUB_WARP_THREADS(PTX_ARCH)),

            // SCALED_BS_SMEM_USAGE            = (SCALED_BLOCK_SIZE2+1)*DIM*sizeof(T)+SCALED_BLOCK_SIZE2*DIM*sizeof(T),

            EXCEED_MAX_SMEM                 = (NOMINAL_SMEM_USAGE > MAX_SMEM_USAGE) ? 1 : 0,
            CHOOSE_BLOCK_SIZE               = (EXCEED_MAX_SMEM == 1) ? SCALED_BLOCK_SIZE2 : NOMINAL_BLOCK_SIZE,
        };

        typedef BlockChoosePolicy<CHOOSE_BLOCK_SIZE> ChoosePolicyT;
    };

    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

    #if (CUB_PTX_ARCH >= 500)
        typedef Policy500<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0> PtxPolicy;

    #elif (CUB_PTX_ARCH >= 350)
        typedef Policy350<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0> PtxPolicy;

    #else
        typedef Policy210<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0> PtxPolicy;
    #endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxAgentChoosePolicy : PtxPolicy::ChoosePolicyT {};


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
        KernelConfig    &chooseConfig)
    {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        chooseConfig.template Init<PtxAgentChoosePolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptxVersion >= 500)
        {
            chooseConfig.template Init<typename Policy500<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0>::ChoosePolicyT>();
        }
        else if (ptxVersion >= 350)
        {
            chooseConfig.template Init<typename Policy350<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0>::ChoosePolicyT>();
        }
        else 
        {
            chooseConfig.template Init<typename Policy210<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0>::ChoosePolicyT>();
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
     * Internal dispatch routine for computing a device-wide choose using the
     * specified kernel functions.
     */
    template <
        typename            ChooseKernelPtrT>    ///< Function type of rd::DeviceChooseKernelPtrT
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t invoke(
        T const *           d_P,
        T *                 d_S,
        int                 np,
        int *               d_ns,
        T                   r, 
        int                 pStride,
        int                 sStride,
        cudaStream_t        stream,                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debugSynchronous,      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        ChooseKernelPtrT    chooseKernel,      ///< [in] Kernel function pointer to parameterization of cub::DeviceChooseKernel
        KernelConfig        chooseConfig)      ///< [in] Dispatch parameters that match the policy that \p chooseKernel was compiled for
    {

        #ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

        #else
        cudaError error = cudaSuccess;
        do
        {
            // Get grid size for scanning tiles
            dim3 chooseGridSize(1);

            // Log chooseKernel configuration
            if (debugSynchronous)
            {
                _CubLog("Invoke chooseKernel<<<{%d,%d,%d}, %d, 0, %p>>> d_P: [%p], d_S[%p]"
                    " np: %d, *d_ns: %p, pStride: %d, sStride: %d\n",
                chooseGridSize.x, chooseGridSize.y, chooseGridSize.z, chooseConfig.blockThreads, 
                stream, d_P, d_S, np, d_ns, pStride, sStride);
            } 

            // Invoke chooseKernel
            chooseKernel<<<chooseGridSize, chooseConfig.blockThreads, 0, stream>>>(
                d_P,
                d_S,
                np,
                d_ns,
                r, 
                pStride,
                sStride);

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
        T const *       d_P,
        T *             d_S,
        int             np,
        int *           d_ns,
        T               r, 
        int             pStride,
        int             sStride,
        cudaStream_t    stream,
        bool            debugSynchronous)      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.)
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
            KernelConfig chooseConfig;
            InitConfigs(ptxVersion, chooseConfig);

            // Dispatch
            if (CubDebug(error = invoke(
                d_P,
                d_S,
                np,
                d_ns,
                r,
                pStride,
                sStride,
                stream,
                debugSynchronous,
                DeviceChooseKernel<PtxAgentChoosePolicy, T, DIM, INPUT_MEM_LAYOUT, 
                    OUTPUT_MEM_LAYOUT>,
                chooseConfig))) break;
        }
        while (0);

        return error;
    }

    static __host__ cudaError_t setCacheConfig()
    {
        cudaError error = cudaSuccess;
        do
        {
            if (CubDebug(error = cudaFuncSetCacheConfig(DeviceChooseKernel<
                PtxAgentChoosePolicy, T, DIM, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT>, 
                cudaFuncCachePreferShared))) break;
        }
        while (0);

        return error;
    }

};

} // namespace bruteForce
} // namespace gpu
} // namespace rd

#endif // __DISPATCH_CHOOSE_CUH__
