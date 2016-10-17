/**
 * @file dispatch_find_bounds.cuh
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

#include "cub/util_debug.cuh"
#include "cub/util_device.cuh"
#include "cub/util_arch.cuh"
#include "cub/util_type.cuh"

#include "rd/gpu/agent/agent_find_bounds.cuh"

namespace rd
{
namespace gpu
{
namespace detail
{

/******************************************************************************
 * Find bounds kernel entry points
 *****************************************************************************/
/**
 * Find bounds kernel entry point (multi-block) . Computes privatized bounds, one per thread block.
 */
template <
    typename AgentFindBoundsPolicyT,
    int DIM,
    DataMemoryLayout INPUT_MEMORY_LAYOUT,
    typename SampleT,
    typename OffsetT>
__launch_bounds__ (int(AgentFindBoundsPolicyT::BLOCK_THREADS))
__global__ void deviceFindBoundsKernelFirstPass(
    SampleT const *                  d_in,
    cub::ArrayWrapper<SampleT*, DIM> d_outMin,           ///< pointers to block-private intermediate output bounds
    cub::ArrayWrapper<SampleT*, DIM> d_outMax,           ///< pointers to block-private intermediate output bounds
    OffsetT                          numItems,
    OffsetT                          stride)
{
    typedef AgentFindBoundsFirstPass<
        AgentFindBoundsPolicyT,
        INPUT_MEMORY_LAYOUT,
        DIM,
        SampleT,
        OffsetT> AgentFindBoundsT;

    // Shared memory storage
    __shared__ typename AgentFindBoundsT::TempStorage tempStorage;

    AgentFindBoundsT agent(tempStorage, d_in, d_outMin.array, d_outMax.array);
    agent.consumeTiles(numItems, stride);
}

/**
 * @brief      Reduces block privatized bounds. Uses 2 blocks per single dimension. Evenly indexed blocks perform max
 *             reduction and odd indexed blocks perform min reduction.
 *
 * @param[in]  d_inMin                 Table of pointers to memory containing intermediate min reduction results. Each
 *                                     region has @p numItems elements.
 * @param[in]  d_inMax                 Table of pointers to memory containing intermediate max reduction results. Each
 *                                     region has @p numItems elements.
 * @param      d_outMin                Pointer to memory for min bounds. It has capacity for @p DIM elements.
 * @param      d_outMax                Pointer to memory for max bounds. It has capacity for @p DIM elements.
 * @param[in]  numItems                Number of intermediate results elements to reduce. It equals the number of
 *                                     dispatched blocks in deviceFindBoundsKernelFirstPass.
 *
 * @tparam     AgentFindBoundsPolicyT  Policy for AgentFindBounds. The same as for deviceFindBoundsKernelFirstPass.
 * @tparam     DIM                     Dimensionality of created bounding box.
 * @tparam     SampleT                 Type of point coordinate element.
 * @tparam     OffsetT                 Integer type for offsets.
 */
template <
    typename AgentFindBoundsPolicyT,
    int DIM,
    typename SampleT,
    typename OffsetT>
__launch_bounds__ (int(AgentFindBoundsPolicyT::BLOCK_THREADS))
__global__ void deviceFindBoundsKernelSecondPass(
    cub::ArrayWrapper<SampleT *, DIM>       d_inMin,
    cub::ArrayWrapper<SampleT *, DIM>       d_inMax,
    SampleT *                               d_outMin,
    SampleT *                               d_outMax,
    OffsetT                                 numItems)
{
    typedef AgentFindBoundsSecondPass<
        AgentFindBoundsPolicyT,
        DIM,
        SampleT,
        OffsetT> AgentFindBoundsT;

    // Shared memory storage
    __shared__ typename AgentFindBoundsT::TempStorage tempStorage;

    AgentFindBoundsT agent(tempStorage, d_inMin, d_inMax, d_outMin, d_outMax);
    agent.consumeTiles(numItems);
}


} // end namespace detail


/******************************************************************************
 * Dispatch
 ******************************************************************************/

template <
    typename            SampleT,
    typename            OffsetT,
    int                 DIM,
    DataMemoryLayout    INPUT_MEM_LAYOUT>
struct DispatchFindBounds
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
            BLOCK_THREADS           = DIM < 5 ? 128 : 64,
            ITEMS_PER_THREAD        = 8,
        };

        typedef detail::AgentFindBoundsPolicy<
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            cub::BLOCK_REDUCE_WARP_REDUCTIONS,
            cub::LOAD_LDG,
            IO_BACKEND_CUB>
        AgentFindBoundsPolicyT;
    };
     /// SM350
    template <
        typename            _T,
        int                 DUMMY>
    struct Policy350
    {
        enum 
        { 
            BLOCK_THREADS           = INPUT_MEM_LAYOUT == ROW_MAJOR ? 128 : 64,
            ITEMS_PER_THREAD        = INPUT_MEM_LAYOUT == ROW_MAJOR ? 10 : 
                                        DIM < 3 ? 11 :
                                            DIM < 6 ? 10 : 6,
        };

        typedef detail::AgentFindBoundsPolicy<
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            cub::BLOCK_REDUCE_WARP_REDUCTIONS,
            cub::LOAD_LDG,
            IO_BACKEND_TROVE>
        AgentFindBoundsPolicyT;
    };
     /// SM300
    template <
        typename            _T,
        int                 DUMMY>
    struct Policy300
    {
        typedef detail::AgentFindBoundsPolicy<
            256,
            4,
            cub::BLOCK_REDUCE_WARP_REDUCTIONS,
            cub::LOAD_DEFAULT,
            IO_BACKEND_CUB>
        AgentFindBoundsPolicyT;
    };
     /// SM200
    template <
        typename            _T,
        int                 DUMMY>
    struct Policy200
    {
        typedef detail::AgentFindBoundsPolicy<
            128,
            2,
            cub::BLOCK_REDUCE_RAKING,
            cub::LOAD_DEFAULT,
            IO_BACKEND_CUB>
        AgentFindBoundsPolicyT;
    };
    

    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

    #if (CUB_PTX_ARCH >= 500)
        typedef Policy500<SampleT, 0> PtxPolicy;
    #elif (CUB_PTX_ARCH >= 350)
        typedef Policy350<SampleT, 0> PtxPolicy;
    #elif (CUB_PTX_ARCH >= 300)
        typedef Policy300<SampleT, 0> PtxPolicy;
    #else
        typedef Policy200<SampleT, 0> PtxPolicy;
    #endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxAgentFindBoundsPolicy : PtxPolicy::AgentFindBoundsPolicyT {};
 
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
        KernelConfig    &findBoundsConfig)
    {
        #if (CUB_PTX_ARCH > 0)

            // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
            findBoundsConfig.template Init<PtxAgentFindBoundsPolicy>();
        #else
            // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
            if (ptxVersion >= 500)
            {
                findBoundsConfig.template Init<typename Policy500<SampleT, 0>::AgentFindBoundsPolicyT>();
            }
            else if (ptxVersion >= 350)
            {
                findBoundsConfig.template Init<typename Policy350<SampleT, 0>::AgentFindBoundsPolicyT>();
            }
            else if (ptxVersion >= 300)
            {
                findBoundsConfig.template Init<typename Policy300<SampleT, 0>::AgentFindBoundsPolicyT>();
            }
            else 
            {
                findBoundsConfig.template Init<typename Policy200<SampleT, 0>::AgentFindBoundsPolicyT>();
            }
        #endif
    }


    /**
     * Kernel kernel dispatch configuration.
     */
    struct KernelConfig
    {
        int blockThreads;
        int itemsPerThread;

        template <typename PolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            blockThreads       = PolicyT::BLOCK_THREADS;
            itemsPerThread     = PolicyT::POINTS_PER_THREAD;
        }
    };

    //---------------------------------------------------------------------
    // invocation (two-pass)
    //---------------------------------------------------------------------
    
    template <
        typename    FindBoundsKernelFirstPassPtrT,
        typename    FindBoundsKernelSecondPassPtrT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t invoke(
        void *                  d_tempStorage,               ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p tempStorageBytes and no work is done.
        size_t &                tempStorageBytes,            ///< [in,out] Reference to size in bytes of \p d_tempStorage allocation
        SampleT const *         d_in,                        ///< [in] Pointer to the input sequence of data items
        SampleT *               d_outMin,                    ///< [out] Pointer to the output min bounds
        SampleT *               d_outMax,                    ///< [out] Pointer to the output max bounds
        OffsetT                 numItems,                    ///< [in] Total number of input items (i.e., length of \p d_in)
        OffsetT                 stride,
        cudaStream_t            stream,                      ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debugSynchronous,            ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.)
        FindBoundsKernelFirstPassPtrT    firstPassKernel,
        FindBoundsKernelSecondPassPtrT   secondPassKernel,
        KernelConfig            kernelConfig,
        int                     ptxVersion)
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
            if (CubDebug(error = cudaDeviceGetAttribute (&smCount, cudaDevAttrMultiProcessorCount, deviceOrdinal))) break;

            // get SM occupancy
            int smOccupancy;
            if (CubDebug(error = cub::MaxSmOccupancy(
                smOccupancy,
                firstPassKernel,
                kernelConfig.blockThreads)
            )) break;

            dim3 boundsGridSize(1);
            boundsGridSize.x = smCount * smOccupancy * CUB_SUBSCRIPTION_FACTOR(ptxVersion);

            // Alias allocations for blocks' output bounds
            cub::ArrayWrapper<SampleT *, DIM> d_blocksOutMin;
            cub::ArrayWrapper<SampleT *, DIM> d_blocksOutMax;

            // Temporary storage allocation requirements
            void* allocations[2 * DIM]{nullptr};
            size_t allocationSizes[2 * DIM];

            for (int d = 0; d < DIM * 2; ++d)
            {
                allocationSizes[d] = boundsGridSize.x * sizeof(SampleT);
            }

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            if (CubDebug(error = cub::AliasTemporaries(d_tempStorage, tempStorageBytes,
                allocations, allocationSizes))) break;
            if (d_tempStorage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            for (int d = 0; d < DIM; ++d)
            {
                d_blocksOutMin.array[d] = (SampleT*) allocations[d * 2];
                d_blocksOutMax.array[d] = (SampleT*) allocations[d * 2 + 1];
            }

            if (debugSynchronous)
            {
                _CubLog("Invoking firstPassKernel<<<%d, %d, 0, %lld>>>\n",
                 boundsGridSize.x, kernelConfig.blockThreads, (long long)stream);
            }

            // Invoke findBoundsKernelFirstPass
            firstPassKernel<<<boundsGridSize, kernelConfig.blockThreads, 0, stream>>>(
                d_in,
                d_blocksOutMin,
                d_blocksOutMax,
                numItems,
                stride);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;
            // Sync the stream if specified to flush runtime errors
            if (debugSynchronous && (CubDebug(error = cub::SyncStream(stream)))) break;

            int intermediateResultsCount = boundsGridSize.x;
            boundsGridSize.x = DIM * 2;

            if (debugSynchronous)
            {
                _CubLog("Invoking secondPassKernel<<<%d, %d, 0, %lld>>> "
                    "intermediateResultsCount: %d\n",
                    boundsGridSize.x, kernelConfig.blockThreads, (long long)stream,
                    intermediateResultsCount);
            }

            secondPassKernel<<<boundsGridSize, kernelConfig.blockThreads, 0, stream>>>(
                d_blocksOutMin,
                d_blocksOutMax,
                d_outMin,
                d_outMax,
                intermediateResultsCount);

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
        void                *d_tempStorage,               ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p tempStorageBytes and no work is done.
        size_t              &tempStorageBytes,            ///< [in,out] Reference to size in bytes of \p d_tempStorage allocation
        SampleT const *     d_in,                         ///< [in] Pointer to the input sequence of data items
        SampleT *           d_outMin,                     ///< [out] Pointer to the output min bounds
        SampleT *           d_outMax,                     ///< [out] Pointer to the output max bounds
        OffsetT             numItems,                     ///< [in] Total number of input items (i.e., length of \p d_in)
        OffsetT             stride,
        cudaStream_t        stream,                       ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debugSynchronous)             ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.)
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

            KernelConfig findBoundsConfig_;

            // Get kernel dispatch configurations
            InitConfigs(ptxVersion, findBoundsConfig_);

            if (CubDebug(error = invoke(
                d_tempStorage,
                tempStorageBytes,
                d_in,
                d_outMin,
                d_outMax,
                numItems,
                stride,
                stream,
                debugSynchronous,
                detail::deviceFindBoundsKernelFirstPass<
                    PtxAgentFindBoundsPolicy, DIM, INPUT_MEM_LAYOUT, SampleT, OffsetT>,
                detail::deviceFindBoundsKernelSecondPass<
                    PtxAgentFindBoundsPolicy, DIM, SampleT, OffsetT>,
                findBoundsConfig_ ,
                ptxVersion))) break;

        }
        while (0);

        return error;
    }
};

} // end namespace gpu
} // end namespace rd
