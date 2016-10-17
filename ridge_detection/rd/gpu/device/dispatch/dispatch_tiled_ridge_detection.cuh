/**
 * @file dispatch_tiled_ridge_detection.cuh
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

#include <helper_cuda.h>

#include "rd/gpu/agent/agent_tiled_ridge_detection.cuh"
#include "rd/gpu/device/device_evolve.cuh"
#include "rd/gpu/device/device_decimate.cuh"
#include "rd/gpu/device/device_choose.cuh"
#include "rd/gpu/device/tiled/tree_drawer.cuh"

namespace rd 
{
namespace gpu
{
namespace tiled
{

//-----------------------------------------------------------------------------
//  Tiled ridge detection kernel entry points
//-----------------------------------------------------------------------------

namespace detail
{

#ifdef RD_DRAW_TREE_TILES
template <
    typename                        AgentTiledRidgeDetectionPolicy,
    int                             DIM,
    DataMemoryLayout                IN_MEM_LAYOUT,
    DataMemoryLayout                OUT_MEM_LAYOUT,
    RidgeDetectionAlgorithm         RD_TILE_ALGORITHM,
    TiledRidgeDetectionPolicy       RD_TILE_POLICY,
    TileType                        RD_TILE_TYPE,
    typename                        TiledTreeT,
    typename                        T>
__launch_bounds__ (1)
__global__ void deviceTiledRidgeDetectionKernel(
    TiledTreeT *                d_tree,
    T const *                   d_inputPoints,
    T *                         d_chosenPoints,
    int                         inPointsNum,
    int *                       d_chosenPointsNum,
    T                           r1,
    T                           r2,
    int                         inPointsStride,
    int                         chosenPointsStride,
    BoundingBox<DIM, T> *       d_globalBBox,        
    int                         maxTileCapacity,
    cub::ArrayWrapper<int,DIM>  dimTiles,
    bool                        endPhaseRefinement,
    TiledRidgeDetectionTimers * d_rdTimers,
    bool                        debugSynchronous)
{
    typedef AgentTiledRidgeDetection<
                AgentTiledRidgeDetectionPolicy,
                DIM,
                IN_MEM_LAYOUT,
                OUT_MEM_LAYOUT,
                RD_TILE_ALGORITHM,
                RD_TILE_POLICY,
                RD_TILE_TYPE,
                T>
        AgentTiledRidgeDetectionT;

    #ifdef RD_INNER_KERNEL_TIMING
    long long int startRdTime = clock64();
    #endif

    AgentTiledRidgeDetectionT agent(maxTileCapacity, r1, debugSynchronous);
    agent.approximate(
        d_inputPoints, d_chosenPoints, inPointsNum, d_chosenPointsNum, r1, r2, inPointsStride, 
        chosenPointsStride, d_globalBBox, dimTiles, endPhaseRefinement, d_rdTimers, 
        debugSynchronous);

    #ifdef RD_INNER_KERNEL_TIMING
    long long int endRdTime = clock64();
    d_rdTimers->wholeTime = endRdTime - startRdTime;
    #endif

    new(d_tree) TiledTreeT(agent.getTree());
}
#else

template <
    typename                        AgentTiledRidgeDetectionPolicy,
    int                             DIM,
    DataMemoryLayout                IN_MEM_LAYOUT,
    DataMemoryLayout                OUT_MEM_LAYOUT,
    RidgeDetectionAlgorithm         RD_TILE_ALGORITHM,
    TiledRidgeDetectionPolicy       RD_TILE_POLICY,
    TileType                        RD_TILE_TYPE,
    typename                        T>
__launch_bounds__ (1)
__global__ void deviceTiledRidgeDetectionKernel(
    T const *                   d_inputPoints,
    T *                         d_chosenPoints,
    int                         inPointsNum,
    int *                       d_chosenPointsNum,
    T                           r1,
    T                           r2,
    int                         inPointsStride,
    int                         chosenPointsStride,
    BoundingBox<DIM, T> *       d_globalBBox,        
    int                         maxTileCapacity,
    cub::ArrayWrapper<int,DIM>  dimTiles,
    bool                        endPhaseRefinement,
    TiledRidgeDetectionTimers * d_rdTimers,
    bool                        debugSynchronous)
{
    typedef AgentTiledRidgeDetection<
                AgentTiledRidgeDetectionPolicy,
                DIM,
                IN_MEM_LAYOUT,
                OUT_MEM_LAYOUT,
                RD_TILE_ALGORITHM,
                RD_TILE_POLICY,
                RD_TILE_TYPE,
                T>
        AgentTiledRidgeDetectionT;

    #ifdef RD_INNER_KERNEL_TIMING
    long long int startRdTime = clock64();
    #endif
    AgentTiledRidgeDetectionT(maxTileCapacity, r1, debugSynchronous).approximate(
        d_inputPoints, d_chosenPoints, inPointsNum, d_chosenPointsNum, r1, r2, inPointsStride, 
        chosenPointsStride, d_globalBBox, dimTiles, endPhaseRefinement, d_rdTimers, 
        debugSynchronous);

    #ifdef RD_INNER_KERNEL_TIMING
    long long int endRdTime = clock64();
    d_rdTimers->wholeTime = endRdTime - startRdTime;
    #endif
}
#endif
}   // end namespace detail


//-----------------------------------------------------------------------------
//  Helper structure to choose optimal launch configuration
//-----------------------------------------------------------------------------

template <
    int                             DIM,
    DataMemoryLayout                IN_MEM_LAYOUT,
    DataMemoryLayout                OUT_MEM_LAYOUT,
    RidgeDetectionAlgorithm         RD_TILE_ALGORITHM,
    TiledRidgeDetectionPolicy       RD_TILE_POLICY,
    TileType                        RD_TILE_TYPE,
    typename                        T>
struct DispatchTiledRidgeDetection
{

    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------

    /// SM500
    struct Policy500
    {
        enum
        {
            BLOCK_THREADS       = 64,
            NOMINAL_4B_ITEMS    = DIM < 5 ? 18 : 64,
            ITEMS_PER_THREAD    = DIM < 3 ? 12 :
                                    CUB_MAX(1, (NOMINAL_4B_ITEMS * 4) / (DIM * sizeof(T))),
        };

        typedef AgentTiledRidgeDetectionPolicy<
            BLOCK_THREADS,
            ITEMS_PER_THREAD>
        AgentTiledRDPolicyT;
    };
    /// SM350
    struct Policy350
    {
        enum
        {
            BLOCK_THREADS       = DIM < 3 ? 128 : 192,
            NOMINAL_4B_ITEMS    = 10,
            ITEMS_PER_THREAD    = CUB_MAX(1, (NOMINAL_4B_ITEMS * 4) / (DIM * sizeof(T))),
        };

        typedef AgentTiledRidgeDetectionPolicy<
            BLOCK_THREADS,
            ITEMS_PER_THREAD>
        AgentTiledRDPolicyT;
    };

    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

    #if (CUB_PTX_ARCH >= 500)
        typedef Policy500 PtxPolicy;
    #else /*(CUB_PTX_ARCH >= 350)*/
        typedef Policy350 PtxPolicy;
    #endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxAgentTiledRDPolicy : PtxPolicy::AgentTiledRDPolicyT {};
 
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
        KernelConfig    &histogramConfig)
    {
        #if (CUB_PTX_ARCH > 0)
            // We're on the device, so initialize the kernel dispatch configurations 
            // with the current PTX policy
            histogramConfig.template Init<PtxAgentTiledRDPolicy>();
        #else
            // We're on the host, so lookup and initialize the kernel dispatch configurations 
            // with the policies that match the device's PTX version
            if (ptxVersion >= 500)
            {
                histogramConfig.template Init<typename Policy500::AgentTiledRDPolicyT>();
            }
            else /*if (ptxVersion >= 350)*/
            {
                histogramConfig.template Init<typename Policy350::AgentTiledRDPolicyT>();
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
            itemsPerThread     = PolicyT::ITEMS_PER_THREAD;
        }
    };

    //---------------------------------------------------------------------
    // invocation 
    //---------------------------------------------------------------------
    
    #ifndef RD_DRAW_TREE_TILES
    template<
        typename    TiledRDKernelPtrT>
    #else
    template<
        typename    TiledRDKernelPtrT,
        typename    TiledTreeT>
    #endif
    static void invoke(
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
        TiledRDKernelPtrT   kernelPtr,
        KernelConfig        config,
    #ifndef RD_DRAW_TREE_TILES
        bool        debugSynchronous = false)
    #else
        TiledTreeT  d_tree,
        bool        debugSynchronous = false)
    #endif
    {
        // set used kernels cache configurations
        rd::gpu::bruteForce::DeviceChoose::
            setCacheConfig<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, T>();
        rd::gpu::bruteForce::DeviceEvolve::
            setCacheConfig<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, T>();
        // set cache configuration for decimate distance matrix
        rd::gpu::bruteForce::DeviceDecimate::
            setDecimateDistMtxCacheConfig<DIM, OUT_MEM_LAYOUT, T>();
        // set cache configuration for global (tiled) decimate
        rd::gpu::tiled::DeviceDecimate::
            setDecimateCacheConfig<DIM, OUT_MEM_LAYOUT, T>();
        // cache configuration for block select if
        checkCudaErrors(cudaFuncSetCacheConfig(kernelPtr, cudaFuncCachePreferL1));

        BoundingBox<DIM, T> globalBBox;
        checkCudaErrors(globalBBox.template findBounds<IN_MEM_LAYOUT>(
            d_inputPoints, inPointsNum, inPointsStride));
        globalBBox.calcDistances();

        #ifdef RD_DEBUG
            std::cout << "DeviceTiledRidgeDetection::approximate() " << std::endl;
            globalBBox.print();
        #endif

        // allocate & copy memory for device global bounding box
        BoundingBox<DIM, T> *d_globalBBox;
        checkCudaErrors(cudaMalloc(&d_globalBBox, sizeof(BoundingBox<DIM,T>)));
        checkCudaErrors(cudaMemcpy(d_globalBBox, &globalBBox, sizeof(BoundingBox<DIM,T>), 
            cudaMemcpyHostToDevice));

        // assess needed memory storage for chosen points
        if (d_chosenPoints == nullptr)
        {
            chosenPointsStride = DIM;
            int maxInitChosenPtsNum = globalBBox.countSpheresInside(r1);
            // in high dimensions or when we have sparse data,
            // above result may be much bigger than inPointsNum,
            // so in order to not wasting device memory, we allocate as small as it is
            // really needed
            maxInitChosenPtsNum = min(maxInitChosenPtsNum, inPointsNum);
            if (OUT_MEM_LAYOUT == COL_MAJOR)
            {
                size_t chPtsPitch = 0;
                checkCudaErrors(cudaMallocPitch(&d_chosenPoints, &chPtsPitch, 
                    maxInitChosenPtsNum * sizeof(T), DIM));
                chosenPointsStride = chPtsPitch / sizeof(T);
            }
            else
            {
                checkCudaErrors(cudaMalloc(&d_chosenPoints, 
                    maxInitChosenPtsNum * DIM * sizeof(T)));
            }
        }

        if (debugSynchronous)
        {
            std::cout << "Invoking detail::deviceTiledRidgeDetectionKernel<<<1,1>>>"
                << " blockThreads: " << config.blockThreads
                << ", itemsPerThread: " << config.itemsPerThread << std::endl;
        }

        cub::ArrayWrapper<int, DIM> dimTiles;
        for (int d = 0; d < DIM; ++d)
        {
            dimTiles.array[d] = tilesPerDim[d];
        }

        #ifndef RD_DRAW_TREE_TILES
        kernelPtr<<<1,1>>>(
            d_inputPoints, d_chosenPoints, inPointsNum, d_chosenPointsNum, r1, r2, 
            inPointsStride, chosenPointsStride, d_globalBBox, maxTileCapacity, 
            dimTiles, endPhaseRefinement, d_rdTimers, debugSynchronous);
        #else
        kernelPtr<<<1,1>>>(
            d_tree, d_inputPoints, d_chosenPoints, inPointsNum, d_chosenPointsNum, r1, r2, 
            inPointsStride, chosenPointsStride, d_globalBBox, maxTileCapacity, 
            dimTiles, endPhaseRefinement, d_rdTimers, debugSynchronous);
        #endif

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(d_globalBBox));
    }

    static void dispatch(
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

        // Get PTX version
        int ptxVersion = 0;
        checkCudaErrors(cub::PtxVersion(ptxVersion));

        KernelConfig config;
        // Get kernel dispatch configurations
        InitConfigs(ptxVersion, config);

        #ifndef RD_DRAW_TREE_TILES
        auto kernelPtr = detail::deviceTiledRidgeDetectionKernel<PtxAgentTiledRDPolicy, DIM, 
            IN_MEM_LAYOUT, OUT_MEM_LAYOUT, RD_TILE_ALGORITHM, RD_TILE_POLICY, RD_TILE_TYPE, T>;
        #else
        typedef TiledTreePolicy<
            PtxAgentTiledRDPolicy::BLOCK_THREADS,
            PtxAgentTiledRDPolicy::ITEMS_PER_THREAD,
            cub::LOAD_LDG,
            IO_BACKEND_CUB>
        TiledTreePolicyT;

        typedef TiledTree<
                TiledTreePolicyT,
                DIM,
                IN_MEM_LAYOUT,
                T>
            TiledTreeT;

        TiledTreeT * d_tree;
        checkCudaErrors(cudaMalloc(&d_tree, sizeof(TiledTreeT)));

        auto kernelPtr = detail::deviceTiledRidgeDetectionKernel<PtxAgentTiledRDPolicy, DIM, 
            IN_MEM_LAYOUT, OUT_MEM_LAYOUT, RD_TILE_ALGORITHM, RD_TILE_POLICY, RD_TILE_TYPE, TiledTreeT, T>;
        #endif

        invoke(
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
            kernelPtr,
            config,
        #ifndef RD_DRAW_TREE_TILES
            debugSynchronous);
        #else
            d_tree,
            debugSynchronous);

        checkCudaErrors(cudaDeviceSynchronize());

        util::TreeDrawer<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, TiledTreeT, T> treeDrawer(d_tree,
            d_inputPoints, inPointsNum, inPointsStride);
        treeDrawer.drawBounds();
        treeDrawer.drawEachTile();
        treeDrawer.releaseTree();

        checkCudaErrors(cudaFree(d_tree));
        #endif
    }


};

} // end namespace tiled
} // end namespace gpu
} // end namespace rd


