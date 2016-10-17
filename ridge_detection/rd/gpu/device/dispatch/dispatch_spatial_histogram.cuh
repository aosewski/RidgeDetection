/**
 * @file dispatch_spatial_histogram.cuh
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

#include "rd/gpu/agent/agent_spatial_histogram.cuh"
#include "rd/gpu/device/bounding_box.cuh"
#include "rd/gpu/util/dev_math.cuh"

namespace rd
{
namespace gpu
{

namespace detail
{
    
/******************************************************************************
 * Histogram kernel entry points
 *****************************************************************************/

template<
    typename            AgentSpatialHistogramPolicyT,
    DataMemoryLayout    INPUT_MEMORY_LAYOUT,
    int                 DIM,
    typename            SampleT,
    typename            CounterT,
    typename            OffsetT,
    typename            PointDecodeOpT,
    bool                CUSTOM_DECODE_OP = false,
    bool                USE_GMEM_PRIV_HIST = true>
__launch_bounds__ (int(AgentSpatialHistogramPolicyT::BLOCK_THREADS))
__global__ void deviceSpatialHistogramKernel(
    SampleT const *         d_in,
    int                     numPoints,
    CounterT *              d_outHistogram,
    CounterT *              d_privatizedHistogram,
    int                     numBins,
    OffsetT                 stride,
    PointDecodeOpT          pointDecodeOp)
{
    typedef AgentSpatialHistogram<
        AgentSpatialHistogramPolicyT,
        INPUT_MEMORY_LAYOUT,
        DIM,
        SampleT,
        CounterT,
        OffsetT,
        PointDecodeOpT,
        CUSTOM_DECODE_OP,
        USE_GMEM_PRIV_HIST> AgentSpatialHistogramT;

    AgentSpatialHistogramT(
        d_in,
        numBins,
        d_outHistogram,
        d_privatizedHistogram,
        pointDecodeOp).consumeRange(numPoints, stride);
}

} // end namespace detail


/******************************************************************************
 * Dispatch
 ******************************************************************************/
template <
    typename            SampleT,
    typename            CounterT,
    typename            OffsetT,
    int                 DIM,
    DataMemoryLayout    INPUT_MEMORY_LAYOUT>
struct DispatchSpatialHistogram
{
    //---------------------------------------------------------------------
    // Transform operation
    //---------------------------------------------------------------------
    struct Transform
    {
        int binsCnt[DIM];
        BoundingBox<DIM, SampleT> bb;

        __host__ __device__ Transform (
            int const                   binsCnt[DIM],
            BoundingBox<DIM, SampleT> const & bb)
        :   
            bb(bb)
        {
            for (int d = 0; d < DIM; ++d)
            {
                this->binsCnt[d] = binsCnt[d];
            }
        }

        // Converts spatial point cooridinates to linear bin idx.
        __device__ int operator()(
            SampleT   point[DIM])
        {
            int bin = 0;
            int binIdx;            
            for (int i = 0; i < DIM; ++i)
            {
                // get sample's bin [x,y,z...n] idx
                /*
                 * translate each sample coordinate to the common origin (by subtracting minimum)
                 * then divide shifted coordinate by current dimension bin width and get the 
                 * floor of this value (counting from zero!) which is our bin idx we search for.
                 */
                if (bb.dist[i] < getEpsilon<SampleT>())
                {
                    binIdx = 0;
                }
                else
                {
                    SampleT normCord = abs(point[i] - bb.min(i));
                    SampleT step = bb.dist[i] / static_cast<SampleT>(binsCnt[i]);

                    // XXX: second condition is only a hack to cope with strange behaviour, 
                    // when normCord is greater than respective dist, which ofcourse shouldn't 
                    // take place!
                    // if (abs(normCord - bb.dist[i]) <= getEpsilon<SampleT>() ||
                        // normCord - bb.dist[i] > 0)
                    if (almostEqual(normCord, bb.dist[i]) || normCord > bb.dist[i])
                    {
                        binIdx = binsCnt[i]-1;
                    }
                    else
                    {
                        binIdx = floor(normCord / step);
                    }
                }

                /*
                 * Calculate global idx value linearizing bin idx
                 * idx = k_1 + sum_{i=2}^{dim}{k_i mul_{j=i-1}^{1}bDim_j}
                 */
                int tmp = 1;
                for (int j = i - 1; j >= 0; --j)
                {
                    tmp *= binsCnt[j];
                }
                bin += binIdx*tmp;
            }
            return bin;
        }   
    };

    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------

    /// SM500
    struct Policy500
    {
        enum
        {
            NOMINAL_4B_ITEMS = 12,
            ITEMS_PER_THREAD = CUB_MAX(1, (NOMINAL_4B_ITEMS * 4) / (DIM * sizeof(SampleT))),
        };

        typedef detail::AgentSpatialHistogramPolicy<
            384,
            ITEMS_PER_THREAD,
            cub::LOAD_LDG,
            IO_BACKEND_CUB>
        AgentSpatialHistogramPolicyT;
    };
    /// SM350
    struct Policy350
    {
        enum
        {
            BLOCK_THREADS       = DIM < 4 ? 128 : 512,
            ITEMS_PER_THREAD    = DIM < 9 ? 1 :
                                    DIM < 11 ? 3 : 1,
        };

        typedef detail::AgentSpatialHistogramPolicy<
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            cub::LOAD_LDG,
            IO_BACKEND_CUB>
        AgentSpatialHistogramPolicyT;
    };

    // XXX: untested configuration
    /// SM300
    struct Policy300
    {
        typedef detail::AgentSpatialHistogramPolicy<
            512,
            1,
            cub::LOAD_DEFAULT,
            IO_BACKEND_CUB>
        AgentSpatialHistogramPolicyT;
    };
    // XXX: untested configuration
    /// SM200
    struct Policy200
    {
        typedef detail::AgentSpatialHistogramPolicy<
            512,
            1,
            cub::LOAD_DEFAULT,
            IO_BACKEND_CUB>
        AgentSpatialHistogramPolicyT;
    };
    

    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

    #if (CUB_PTX_ARCH >= 500)
        typedef Policy500 PtxPolicy;
    #elif (CUB_PTX_ARCH >= 350)
        typedef Policy350 PtxPolicy;
    #elif (CUB_PTX_ARCH >= 300)
        typedef Policy300 PtxPolicy;
    #else
        typedef Policy200 PtxPolicy;
    #endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxAgentSpatialHistogramPolicy : PtxPolicy::AgentSpatialHistogramPolicyT {};
 
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
        KernelConfig    &histogramConfig)
    {
        #if (CUB_PTX_ARCH > 0)
            // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
            histogramConfig.template Init<PtxAgentSpatialHistogramPolicy>();
        #else
            // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
            if (ptxVersion >= 500)
            {
                histogramConfig.template Init<typename Policy500::AgentSpatialHistogramPolicyT>();
            }
            else if (ptxVersion >= 350)
            {
                histogramConfig.template Init<typename Policy350::AgentSpatialHistogramPolicyT>();
            }
            else if (ptxVersion >= 300)
            {
                histogramConfig.template Init<typename Policy300::AgentSpatialHistogramPolicyT>();
            }
            else 
            {
                histogramConfig.template Init<typename Policy200::AgentSpatialHistogramPolicyT>();
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
    // invocation 
    //---------------------------------------------------------------------
    
    template <
        typename    PointDecodeOpT,
        typename    HistogramKernelPtrT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t invoke(
        void *                  d_tempStorage,               ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p tempStorageBytes and no work is done.
        size_t &                tempStorageBytes,            ///< [in,out] Reference to size in bytes of \p d_tempStorage allocation
        SampleT const *         d_in,
        int                     numPoints,
        CounterT *              d_outHistogram,
        int                     numBins,
        OffsetT                 stride,
        PointDecodeOpT &        pointDecodeOp,
        bool                    aggregateHist,              ///< Controls wheter or not to zeroize @p d_outHistogram in case we want to aggregate results of few histograms into one.
        bool                    useGmemPrivHist,            ///< Whether or not to use gmem private histograms
        cudaStream_t            stream,
        bool                    debugSynchronous,
        HistogramKernelPtrT     kernelPtr,
        KernelConfig            kernelConfig)
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
            if (CubDebug(error = cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 
                deviceOrdinal))) break;

            // get SM occupancy
            int smOccupancy;
            if (CubDebug(error = cub::MaxSmOccupancy(
                smOccupancy,
                kernelPtr,
                kernelConfig.blockThreads)
            )) break;

            dim3 boundsGridSize(1);
            // We're not multiplying it by CUB_SUBSCRIPTION_FACTOR, because, of memory usage. Each 
            // block of threads is assigned private histogram, which become extremely large with 
            // increasing number of dimensions. Eg. 4 bins per dim in 10 dims yields total more 
            // than 1 million bins!
            boundsGridSize.x = smCount * smOccupancy;

            // Temporary storage allocation requirements
            void* allocations[1] {nullptr};
            size_t allocationSizes[1] {boundsGridSize.x * numBins * sizeof(CounterT)};

            if (useGmemPrivHist)
            {
                // Alias the temporary allocations from the single storage blob (or compute 
                // the necessary size of the blob)
                if (CubDebug(error = cub::AliasTemporaries(d_tempStorage, tempStorageBytes, 
                    allocations, allocationSizes))) break;
                if (d_tempStorage == NULL)
                {
                    // Return if the caller is simply requesting the size of the storage allocation
                    return cudaSuccess;
                }
            }

            // set output histogram counters to zero if necessary
            if (!aggregateHist)
            {
                if (CubDebug(error = cudaMemsetAsync(d_outHistogram, 0, numBins * sizeof(CounterT), 
                    stream))) break;
            }

            if (debugSynchronous)
            {
                printf("Invoking deviceSpatialHistogramKernel<<<%d, %d, 0, %p>>> numBins: %d, "
                    " numPoints: %d, itemsPerThread: %d\n",
                    boundsGridSize.x, kernelConfig.blockThreads, stream, numBins, 
                    numPoints, kernelConfig.itemsPerThread);
            }

            kernelPtr<<<boundsGridSize.x, kernelConfig.blockThreads, 0, stream>>>(
                d_in,
                numPoints,
                d_outHistogram,
                (CounterT*)allocations[0],
                numBins,
                stride,
                pointDecodeOp);

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
        void *                      d_tempStorage,               ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p tempStorageBytes and no work is done.
        size_t &                    tempStorageBytes,            ///< [in,out] Reference to size in bytes of \p d_tempStorage allocation
        SampleT const *             d_in,
        int                         numPoints,
        CounterT *                  d_outHistogram,
        OffsetT                     stride,
        int const                   (&binsCnt)[DIM],
        BoundingBox<DIM, SampleT> const & bbox,
        bool                        useGmemPrivHist,            ///< Whether or not to use gmem private histograms
        cudaStream_t                stream,
        bool                        debugSynchronous)
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

            int numBins = 1;
            for (int d = 0; d < DIM; ++d)
            {
                numBins *= binsCnt[d];
            }

            typedef Transform PointDecodeOpT;
            PointDecodeOpT pointDecodeOp(binsCnt, bbox);

            KernelConfig histogramConfig;

            auto kernelPtr = (useGmemPrivHist) ? 
                detail::deviceSpatialHistogramKernel<PtxAgentSpatialHistogramPolicy,
                    INPUT_MEMORY_LAYOUT, DIM, SampleT, CounterT, OffsetT, PointDecodeOpT,
                    false, true> :
                detail::deviceSpatialHistogramKernel<PtxAgentSpatialHistogramPolicy,
                    INPUT_MEMORY_LAYOUT, DIM, SampleT, CounterT, OffsetT, PointDecodeOpT,
                    false, false>;

            // Get kernel dispatch configurations
            InitConfigs(ptxVersion, histogramConfig);

            if (CubDebug(error = invoke(
                d_tempStorage,
                tempStorageBytes,
                d_in,
                numPoints,
                d_outHistogram,
                numBins,
                stride,
                pointDecodeOp,
                false,
                useGmemPrivHist,
                stream,
                debugSynchronous,
                kernelPtr,
                histogramConfig))) break;

        }
        while (0);

        return error;
    }

    template <typename PointDecodeOpT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t dispatch(
        void *                      d_tempStorage,              ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p tempStorageBytes and no work is done.
        size_t &                    tempStorageBytes,           ///< [in,out] Reference to size in bytes of \p d_tempStorage allocation
        SampleT const *             d_in,
        int                         numPoints,
        CounterT *                  d_outHistogram,
        int                         numBins,
        PointDecodeOpT &            pointDecodeOp,              ///< Functor object providing decode operation for determining linear bin id.
        bool                        aggregateHist,              ///< Controls wheter or not to zeroize @p d_outHistogram in case we want to aggregate results of few histograms into one.
        bool                        useGmemPrivHist,            ///< Whether or not to use gmem private histograms
        OffsetT                     stride,
        cudaStream_t                stream,
        bool                        debugSynchronous)
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

            auto kernelPtr = (useGmemPrivHist) ? 
                detail::deviceSpatialHistogramKernel<PtxAgentSpatialHistogramPolicy,
                    INPUT_MEMORY_LAYOUT, DIM, SampleT, CounterT, OffsetT, PointDecodeOpT,
                    true, true> :
                detail::deviceSpatialHistogramKernel<PtxAgentSpatialHistogramPolicy,
                    INPUT_MEMORY_LAYOUT, DIM, SampleT, CounterT, OffsetT, PointDecodeOpT,
                    true, false>;

            KernelConfig histogramConfig;

            // Get kernel dispatch configurations
            InitConfigs(ptxVersion, histogramConfig);

            if (CubDebug(error = invoke(
                d_tempStorage,
                tempStorageBytes,
                d_in,
                numPoints,
                d_outHistogram,
                numBins,
                stride,
                pointDecodeOp,
                aggregateHist,
                useGmemPrivHist,
                stream,
                debugSynchronous,
                kernelPtr,
                histogramConfig))) break;

        }
        while (0);

        return error;
    }

};

} // end namespace gpu
} // end namespace rd
