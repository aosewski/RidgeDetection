/**
 * @file dispatch_evolve.cuh
 * @author     Adam Rogowiec
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

#ifndef __DISPATCH_EVOLVE_CUH__
#define __DISPATCH_EVOLVE_CUH__

#include <helper_cuda.h>

#include "rd/gpu/device/brute_force/evolve.cuh"
#include "rd/gpu/device/brute_force/rd_globals.cuh"
#include "rd/gpu/util/dev_utilities.cuh"

#include "cub/util_debug.cuh"
#include "cub/util_device.cuh"
#include "cub/util_arch.cuh"

namespace rd
{
namespace gpu
{
namespace bruteForce
{
namespace detail
{

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

template <
    typename            AgentClosestSpherePolicyT,
    int                 DIM,
    DataMemoryLayout    INPUT_MEM_LAYOUT,
    DataMemoryLayout    OUTPUT_MEM_LAYOUT,
    typename            T>
__launch_bounds__ (int(AgentClosestSpherePolicyT::BLOCK_SIZE))
static __global__ void DeviceClosestSphereKernel(
        T const * __restrict__ P,
        T const * __restrict__ S, 
        T * cordSums,
        int * spherePointCount,
        int np,
        int ns,
        T r,
        int pStride,
        int csStride,
        int sStride)
{
    typedef AgentClosestSphere<T, AgentClosestSpherePolicyT, DIM, INPUT_MEM_LAYOUT, 
        OUTPUT_MEM_LAYOUT> AgentClosestSphereT;
    AgentClosestSphereT().calc(P, S, cordSums, spherePointCount, np, ns, r, pStride, csStride,
        sStride);
}

template <
    typename            AgentShiftSpherePolicyT,
    int                 DIM,
    DataMemoryLayout    INPUT_MEM_LAYOUT,
    DataMemoryLayout    OUTPUT_MEM_LAYOUT,
    typename            T>
__launch_bounds__ (int(AgentShiftSpherePolicyT::BLOCK_SIZE))
static __global__ void DeviceShiftSphereKernel(
        T * S,
        T const * __restrict__ cordSums,
        int const * __restrict__ spherePointCount,
        int ns,
        int csStride,
        int sStride)
{
    typedef AgentShiftSphere<T, AgentShiftSpherePolicyT, DIM, INPUT_MEM_LAYOUT, 
        OUTPUT_MEM_LAYOUT> AgentShiftSphereT;
    AgentShiftSphereT().shift(S, cordSums, spherePointCount, ns, csStride, sStride);
}

//-----------------------------------------------------------
//  Main evolve routine kernel
//-----------------------------------------------------------

template <
    int         DIM,
    typename    AgentClosestSpherePolicyT,
    typename    AgentShiftSpherePolicyT,
    typename    closestSphereKernelPtrT,
    typename    shiftSphereKernelPtrT,
    typename    T>
static __global__ void deviceEvolveKernel(
    T const *       d_P,
    T *             d_S,
    T *             d_cordSums, 
    int *           d_spherePointCount,
    int             np,
    int             ns,
    T               r,
    int             pStride,
    int             sStride,
    int             csStride,
    cudaStream_t    stream,
    bool            debugSynchronous,
    closestSphereKernelPtrT closestSphereKernelPtr,
    shiftSphereKernelPtrT   shiftSphereKernelPtr)
{
    dim3 csGridSize(1), ssGridSize(1);

    #ifndef CUB_RUNTIME_ENABLED
        // Kernel launch not supported from this device
        rdDevCheckCall(cudaErrorNotSupported);
    #else
    // Get device ordinal
    int deviceOrdinal;
    rdDevCheckCall(cudaGetDevice(&deviceOrdinal));

    // Get SM count
    int smCount;
    rdDevCheckCall(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 
        deviceOrdinal));

    // Get SM occupancy for closestSphereKernel
    int calcClosestSphereSmOccupancy;
    rdDevCheckCall(cub::MaxSmOccupancy(
        calcClosestSphereSmOccupancy,            // out
        closestSphereKernelPtr,
        AgentClosestSpherePolicyT::BLOCK_SIZE));

    csGridSize.x = smCount * calcClosestSphereSmOccupancy * CUB_PTX_SUBSCRIPTION_FACTOR;

    // Get SM occupancy for shiftSphereKernel
    int shiftSphereSmOccupancy;
    rdDevCheckCall(cub::MaxSmOccupancy(
        shiftSphereSmOccupancy,            // out
        shiftSphereKernelPtr,
        AgentShiftSpherePolicyT::BLOCK_SIZE));

    ssGridSize.x = smCount * shiftSphereSmOccupancy * CUB_PTX_SUBSCRIPTION_FACTOR;

    rdContFlag = 1;
    while (rdContFlag)
    {
        rdContFlag = 0;
        // ugly hack...
        if (csStride != DIM)
        {
            rdDevCheckCall(cudaMemset2DAsync(d_cordSums, csStride * sizeof(T), 0, ns * sizeof(T), 
                DIM, stream));
        }
        else
        {
            rdDevCheckCall(cudaMemsetAsync(d_cordSums, 0, ns * DIM * sizeof(T), stream));
        }
        rdDevCheckCall(cudaMemsetAsync(d_spherePointCount, 0, ns * sizeof(int), stream));
        
        if (debugSynchronous)
        {
            _CubLog("Invoke closestSphereKernel<<<%d, %d, 0, %p>>> for input points"
                " pStride: %d, csStride: %d, sStride: %d\n",
                csGridSize.x, AgentClosestSpherePolicyT::BLOCK_SIZE, nullptr,
                pStride, csStride, sStride);
        }

        closestSphereKernelPtr<<<csGridSize, AgentClosestSpherePolicyT::BLOCK_SIZE, 
            0, stream>>>(
                d_P, d_S, d_cordSums, d_spherePointCount, np, ns, r, pStride, csStride, sStride); 

        // Check for failure to launch
        rdDevCheckCall(cudaPeekAtLastError());
        // Sync the stream if specified to flush runtime errors
        if (debugSynchronous)
        {
            rdDevCheckCall(cub::SyncStream(stream));
            _CubLog("Invoke shiftSphereKernel<<<%d, %d, 0, %p>>>"
                " csStride: %d, sStride: %d\n\n",
                ssGridSize.x, AgentShiftSpherePolicyT::BLOCK_SIZE, nullptr,
                csStride, sStride);
        } 
        
        shiftSphereKernelPtr<<<ssGridSize, AgentShiftSpherePolicyT::BLOCK_SIZE, 0, stream>>>(
            d_S, d_cordSums, d_spherePointCount, ns, csStride, sStride); 
        
        // Check for failure to launch
        rdDevCheckCall(cudaPeekAtLastError());
        rdDevCheckCall(cudaDeviceSynchronize());
    }
    #endif
}


template <
    int         DIM,
    typename    AgentClosestSpherePolicyT,
    typename    AgentShiftSpherePolicyT,
    typename    closestSphereKernelPtrT,
    typename    shiftSphereKernelPtrT,
    typename    T>
static __global__ void deviceEvolveKernel(
    T const *       d_inputPoints,
    T const *       d_neighbourPoints,
    T *             d_chosenPoints,
    T *             d_cordSums, 
    int *           d_spherePointCount,
    int             inPointsNum,
    int             neighbourPointsNum,
    int             chosenPointsNum,
    T               r,
    int             inPointsStride,
    int             neighbourPointsStride,
    int             chosenPointsStride,
    int             cordSumsStride,
    bool            debugSynchronous,
    closestSphereKernelPtrT closestSphereKernelPtr,
    shiftSphereKernelPtrT   shiftSphereKernelPtr)
{
    dim3 csGridSize(1), ssGridSize(1);

    #if defined(CUB_RUNTIME_ENABLED) && (CUB_PTX_ARCH >= 350)
    // Get device ordinal
    int deviceOrdinal;
    rdDevCheckCall(cudaGetDevice(&deviceOrdinal));

    // Get SM count
    int smCount;
    rdDevCheckCall(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 
        deviceOrdinal));

    // Get SM occupancy for closestSphereKernel
    int calcClosestSphereSmOccupancy;
    rdDevCheckCall(cub::MaxSmOccupancy(
        calcClosestSphereSmOccupancy,            // out
        closestSphereKernelPtr,
        AgentClosestSpherePolicyT::BLOCK_SIZE));

    csGridSize.x = smCount * calcClosestSphereSmOccupancy * CUB_PTX_SUBSCRIPTION_FACTOR;

    // Get SM occupancy for shiftSphereKernel
    int shiftSphereSmOccupancy;
    rdDevCheckCall(cub::MaxSmOccupancy(
        shiftSphereSmOccupancy,            // out
        shiftSphereKernelPtr,
        AgentShiftSpherePolicyT::BLOCK_SIZE));

    ssGridSize.x = smCount * shiftSphereSmOccupancy * CUB_PTX_SUBSCRIPTION_FACTOR;

    rdContFlag = 1;
    while (rdContFlag)
    {
        rdContFlag = 0;

        // ugly hack... :(
        if (cordSumsStride != DIM)
        {
            rdDevCheckCall(cudaMemset2DAsync(d_cordSums, cordSumsStride * sizeof(T), 0, 
                chosenPointsNum * sizeof(T), DIM, nullptr));
        }
        else
        {
            rdDevCheckCall(cudaMemsetAsync(d_cordSums, 0, chosenPointsNum * DIM * sizeof(T), 
                nullptr));
        }
        rdDevCheckCall(cudaMemsetAsync(d_spherePointCount, 0, chosenPointsNum * sizeof(int),
            nullptr));
        
        if (debugSynchronous)
        {
            _CubLog("Invoke closestSphereKernel<<<%d, %d, 0, %p>>> for input points"
                " inPStride: %d, csStride: %d, sStride: %d\n",
                csGridSize.x, AgentClosestSpherePolicyT::BLOCK_SIZE, nullptr,
                inPointsStride, cordSumsStride, chosenPointsStride);
        }

        closestSphereKernelPtr<<<csGridSize, AgentClosestSpherePolicyT::BLOCK_SIZE>>>(
                d_inputPoints, d_chosenPoints, d_cordSums, d_spherePointCount, inPointsNum, 
                chosenPointsNum, r, inPointsStride, cordSumsStride, chosenPointsStride); 
        // Check for failure to launch
        rdDevCheckCall(cudaPeekAtLastError());
        // Sync the stream if specified to flush runtime errors
        if (debugSynchronous)
        {
            rdDevCheckCall(cub::SyncStream(nullptr));
            _CubLog("Invoke closestSphereKernel<<<%d, %d, 0, %p>>> for neighbour points"
                " npStride: %d, csStride: %d, sStride: %d\n",
                csGridSize.x, AgentClosestSpherePolicyT::BLOCK_SIZE, nullptr,
                neighbourPointsStride, cordSumsStride, chosenPointsStride);
        }
        
        closestSphereKernelPtr<<<csGridSize, AgentClosestSpherePolicyT::BLOCK_SIZE>>>(
                d_neighbourPoints, d_chosenPoints, d_cordSums, d_spherePointCount, 
                neighbourPointsNum, chosenPointsNum, r, neighbourPointsStride, cordSumsStride,
                chosenPointsStride); 
        // Check for failure to launch
        rdDevCheckCall(cudaPeekAtLastError());
        // Sync the stream if specified to flush runtime errors
        if (debugSynchronous)
        {
            rdDevCheckCall(cub::SyncStream(nullptr));
            _CubLog("Invoke shiftSphereKernel<<<%d, %d, 0, %p>>>"
                " csStride: %d, sStride: %d\n\n",
                ssGridSize.x, AgentShiftSpherePolicyT::BLOCK_SIZE, nullptr,
                cordSumsStride, chosenPointsStride);
        }

        shiftSphereKernelPtr<<<ssGridSize, AgentShiftSpherePolicyT::BLOCK_SIZE>>>(
            d_chosenPoints, d_cordSums, d_spherePointCount, chosenPointsNum, cordSumsStride,
            chosenPointsStride); 
        
        // Check for failure to launch
        rdDevCheckCall(cudaPeekAtLastError());
        rdDevCheckCall(cudaDeviceSynchronize());
    }
    #else
    rdDevCheckCall(cudaErrorNotSupported);
    #endif
}


} // end namespace detail

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceEvolve
 */
template <
    typename            T,
    int                 DIM,
    DataMemoryLayout    INPUT_MEM_LAYOUT,
    DataMemoryLayout    OUTPUT_MEM_LAYOUT>
struct DispatchEvolve
{

    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------

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
            CS_NOMINAL_4B_BUFFER_ITEMS = 30,
            SS_NOMINAL_4B_BUFFER_ITEMS = 16,

            CS_BLOCK_SIZE = 64,        /// closest sphere center kernel
            SS_BLOCK_SIZE = 256,        /// shift sphere center kernel
            CS_ITEMS_PER_THREAD = CUB_MAX(1, (CS_NOMINAL_4B_BUFFER_ITEMS * 4) / (DIM * sizeof(T))),
            SS_ITEMS_PER_THREAD = CUB_MAX(1, (SS_NOMINAL_4B_BUFFER_ITEMS * 4) / (DIM * sizeof(T))),
        };

        typedef AgentClosestSpherePolicy<CS_BLOCK_SIZE, CS_ITEMS_PER_THREAD> ClosestSpherePolicyT;
        typedef AgentShiftSpherePolicy<SS_BLOCK_SIZE, SS_ITEMS_PER_THREAD> ShiftSpherePolicyT;
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
            CS_NOMINAL_4B_BUFFER_ITEMS = 30,

            CS_BLOCK_SIZE = 64,        /// closest sphere center kernel
            SS_BLOCK_SIZE = 512,        /// shift sphere center kernel
            CS_ITEMS_PER_THREAD = CUB_MAX(1, (CS_NOMINAL_4B_BUFFER_ITEMS * 4) / (DIM * sizeof(T))),
            SS_ITEMS_PER_THREAD = 1,
        };

        typedef AgentClosestSpherePolicy<CS_BLOCK_SIZE, CS_ITEMS_PER_THREAD> ClosestSpherePolicyT;
        typedef AgentShiftSpherePolicy<SS_BLOCK_SIZE, SS_ITEMS_PER_THREAD> ShiftSpherePolicyT;
    };
    
    /// SM300
    template <
        typename            _T,
        DataMemoryLayout    _IN_MEM_LAYOUT,
        DataMemoryLayout    _OUT_MEM_LAYOUT,
        int                 DUMMY>
    struct Policy300
    {
        enum 
        { 
            CS_BLOCK_SIZE = 256,        /// closest sphere center kernel
            SS_BLOCK_SIZE = 256,        /// shift sphere center kernel
            CS_ITEMS_PER_THREAD = 7,
            SS_ITEMS_PER_THREAD = 2
        };

        typedef AgentClosestSpherePolicy<CS_BLOCK_SIZE, CS_ITEMS_PER_THREAD> ClosestSpherePolicyT;
        typedef AgentShiftSpherePolicy<SS_BLOCK_SIZE, SS_ITEMS_PER_THREAD> ShiftSpherePolicyT;
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
            CS_BLOCK_SIZE = 256,        /// closest sphere center kernel
            SS_BLOCK_SIZE = 256,        /// shift sphere center kernel
            CS_ITEMS_PER_THREAD = 4,
            SS_ITEMS_PER_THREAD = 2
        };

        typedef AgentClosestSpherePolicy<CS_BLOCK_SIZE, CS_ITEMS_PER_THREAD> ClosestSpherePolicyT;
        typedef AgentShiftSpherePolicy<SS_BLOCK_SIZE, SS_ITEMS_PER_THREAD> ShiftSpherePolicyT;
    };

    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

    #if (CUB_PTX_ARCH >= 500)
        typedef Policy500<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0> PtxPolicy;

    #elif (CUB_PTX_ARCH >= 350)
        typedef Policy350<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0> PtxPolicy;

    #elif (CUB_PTX_ARCH >= 300)
        typedef Policy300<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0> PtxPolicy;

    #else
        typedef Policy210<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0> PtxPolicy;
    #endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxAgentClosestSpherePolicy : PtxPolicy::ClosestSpherePolicyT {};
    struct PtxAgentShiftSpherePolicy : PtxPolicy::ShiftSpherePolicyT {};

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
        KernelConfig    &closestSphereConfig,
        KernelConfig    &shiftSphereConfig)
    {
        #if (CUB_PTX_ARCH > 0)

            // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
            closestSphereConfig.template Init<PtxAgentClosestSpherePolicy>();
            shiftSphereConfig.template Init<PtxAgentShiftSpherePolicy>();

        #else

            // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
            if (ptxVersion >= 500)
            {
                closestSphereConfig.template Init<typename Policy500<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0>::ClosestSpherePolicyT>();
                shiftSphereConfig.template Init<typename Policy500<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0>::ShiftSpherePolicyT>();
            }
            else if (ptxVersion >= 350)
            {
                closestSphereConfig.template Init<typename Policy350<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0>::ClosestSpherePolicyT>();
                shiftSphereConfig.template Init<typename Policy350<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0>::ShiftSpherePolicyT>();
            }
            else if (ptxVersion >= 300)
            {
                closestSphereConfig.template Init<typename Policy300<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0>::ClosestSpherePolicyT>();
                shiftSphereConfig.template Init<typename Policy300<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0>::ShiftSpherePolicyT>();
            }
            else 
            {
                closestSphereConfig.template Init<typename Policy210<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0>::ClosestSpherePolicyT>();
                shiftSphereConfig.template Init<typename Policy210<T, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, 0>::ShiftSpherePolicyT>();
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
            blockThreads       = PolicyT::BLOCK_SIZE;
            itemsPerThread     = PolicyT::ITEMS_PER_THREAD;
        }
    };

    //---------------------------------------------------------------------
    // Dispatch entrypoints
    //---------------------------------------------------------------------

    template <
        typename            closestSphereKernelPtrT,
        typename            shiftSphereKernelPtrT>
    CUB_RUNTIME_FUNCTION __forceinline__ 
    static cudaError_t DispatchEvolveKernels(
        T const *       d_P,
        T *             d_S,
        T *             d_cordSums, 
        int *           d_spherePointCount,
        int             np,
        int             ns,
        T               r,
        int             pStride,
        int             sStride,
        int             csStride,
        cudaStream_t    stream,
        bool            debugSynchronous,
        closestSphereKernelPtrT closestSphereKernelPtr,
        shiftSphereKernelPtrT   shiftSphereKernelPtr,
        KernelConfig            closestSphereConfig,
        KernelConfig            shiftSphereConfig,
        int             ptxVersion)
    {

        cudaError_t error = cudaSuccess;
        // #ifndef CUB_RUNTIME_ENABLED
        #if (CUB_PTX_ARCH == 0)
            dim3 csGridSize(1), ssGridSize(1);
            do
            {
                // Get device ordinal
                int deviceOrdinal;
                if (CubDebug(error = cudaGetDevice(&deviceOrdinal))) break;

                // Get SM count
                int smCount;
                if (CubDebug(error = cudaDeviceGetAttribute(&smCount, 
                        cudaDevAttrMultiProcessorCount, deviceOrdinal))) break;

                // Get SM occupancy for closestSphereKernel
                int calcClosestSphereSmOccupancy;
                if (CubDebug(error = cub::MaxSmOccupancy(
                    calcClosestSphereSmOccupancy,            // out
                    closestSphereKernelPtr,
                    closestSphereConfig.blockThreads))) break;

                csGridSize.x = smCount * calcClosestSphereSmOccupancy * 
                    CUB_SUBSCRIPTION_FACTOR(ptxVersion);

                // Get SM occupancy for shiftSphereKernel
                int shiftSphereSmOccupancy;
                if (CubDebug(error = cub::MaxSmOccupancy(
                    shiftSphereSmOccupancy,            // out
                    shiftSphereKernelPtr,
                    shiftSphereConfig.blockThreads))) break;

                ssGridSize.x = smCount * shiftSphereSmOccupancy * 
                    CUB_SUBSCRIPTION_FACTOR(ptxVersion);

                // launching kernels from host
                
                int hContFlag = 1;
                while (hContFlag)
                {
                    hContFlag = 0;
                    checkCudaErrors(cudaMemcpyToSymbolAsync(rdContFlag, &hContFlag, sizeof(int), 
                        0, cudaMemcpyHostToDevice, stream));
                    // ugly hack...
                    if (csStride != DIM)
                    {
                        rdDevCheckCall(cudaMemset2DAsync(d_cordSums, csStride * sizeof(T), 0, 
                            ns * sizeof(T), DIM, stream));
                    }
                    else
                    {
                        rdDevCheckCall(cudaMemsetAsync(d_cordSums, 0, ns * DIM * sizeof(T), 
                            stream));
                    }
                    checkCudaErrors(cudaMemsetAsync(d_cordSums, 0, ns * DIM * sizeof(T), stream));
                    checkCudaErrors(cudaMemsetAsync(d_spherePointCount, 0, ns * sizeof(int), 
                        stream));
                    
                    if (debugSynchronous)
                    {
                        _CubLog("Invoke closestSphereKernel<<<%d, %d, 0, %p>>> np: %d, ns: %d"
                            " pStride: %d, csStride: %d, sStride: %d\n",
                            csGridSize.x, closestSphereConfig.blockThreads, stream, np, ns,
                            pStride, csStride, sStride);
                    }

                    closestSphereKernelPtr<<<csGridSize, closestSphereConfig.blockThreads, 
                        0, stream>>>(
                            d_P, d_S, d_cordSums, d_spherePointCount, np, ns, r, pStride, 
                            csStride, sStride);
                    
                    checkCudaErrors(cudaPeekAtLastError());
                    // Sync the stream if specified to flush runtime errors
                    if (debugSynchronous)
                    {
                        checkCudaErrors(cub::SyncStream(stream));

                        _CubLog("Invoke shiftSphereKernel<<<%d, %d, 0, %p>>> ns: %d"
                            " csStride: %d, sStride: %d\n",
                            ssGridSize.x, shiftSphereConfig.blockThreads, stream, ns,
                            csStride, sStride);
                    }

                    shiftSphereKernelPtr<<<ssGridSize, shiftSphereConfig.blockThreads, 0, stream>>>(
                        d_S, d_cordSums, d_spherePointCount, ns, csStride, sStride); 
                    
                    checkCudaErrors(cudaPeekAtLastError());
                    // Sync the stream if specified to flush runtime errors
                    if (debugSynchronous)
                    {
                        checkCudaErrors(cudaDeviceSynchronize());
                    }
                    checkCudaErrors(cudaMemcpyFromSymbolAsync(&hContFlag, rdContFlag, sizeof(int), 
                        0, cudaMemcpyDeviceToHost, stream));
                    checkCudaErrors(cudaDeviceSynchronize());
                }
            } while(0);
        #else // (CUB_PTX_ARCH == 0)
            // launching kernels from device
            detail::deviceEvolveKernel<DIM, PtxAgentClosestSpherePolicy, PtxAgentShiftSpherePolicy,
                closestSphereKernelPtrT, shiftSphereKernelPtrT><<<1,1,0,stream>>>(
                    d_P, d_S, d_cordSums, d_spherePointCount, np, ns, r, pStride, sStride, csStride,
                    stream, debugSynchronous, closestSphereKernelPtr, shiftSphereKernelPtr);

            rdDevCheckCall(cudaPeekAtLastError());
            if (debugSynchronous)
            {
                rdDevCheckCall(cub::SyncStream(stream));
            }
            // supress warnings
            if (closestSphereConfig.itemsPerThread == 0) {};
            if (shiftSphereConfig.itemsPerThread   == 0) {};
        #endif // (CUB_PTX_ARCH != 0)

        return error;
    }

    template <
        typename            closestSphereKernelPtrT,
        typename            shiftSphereKernelPtrT>
    CUB_RUNTIME_FUNCTION __forceinline__ 
    static cudaError_t DispatchEvolveKernels(
        T const *               d_inputPoints,
        T const *               d_neighbourPoints,
        T *                     d_chosenPoints,
        T *                     d_cordSums, 
        int *                   d_spherePointCount,
        int                     inPointsNum,
        int                     neighbourPointsNum,
        int                     chosenPointsNum,
        T                       r,
        int                     inPointsStride,
        int                     neighbourPointsStride,
        int                     chosenPointsStride,
        int                     cordSumsStride,
        cudaStream_t            stream,
        bool                    debugSynchronous,
        closestSphereKernelPtrT closestSphereKernelPtr,
        shiftSphereKernelPtrT   shiftSphereKernelPtr)
    {
        cudaError_t error = cudaSuccess;
        // enable only on devices with CC>=3.5 and with device runtime api enabled
        #if defined(CUB_RUNTIME_ENABLED) && (CUB_PTX_ARCH >= 350)
        
        if (debugSynchronous)
        {
            _CubLog("Invoke deviceEvolveKernel<<<1, 1, 0, %p>>> inPointsNum: %d, "
                "neighbourPointsNum: %d pStride: %d, nStride: %d, sStride: %d, csStride: %d\n",
                 stream, inPointsNum, neighbourPointsNum, inPointsStride, neighbourPointsStride,
                 chosenPointsStride, cordSumsStride);
        }

        // launching kernels from device
        detail::deviceEvolveKernel<DIM, PtxAgentClosestSpherePolicy, PtxAgentShiftSpherePolicy,
            closestSphereKernelPtrT, shiftSphereKernelPtrT>
            <<<1,1,0,stream>>>(
                d_inputPoints, d_neighbourPoints, d_chosenPoints, d_cordSums, d_spherePointCount, 
                inPointsNum, neighbourPointsNum, chosenPointsNum, r, inPointsStride, 
                neighbourPointsStride, chosenPointsStride, cordSumsStride, debugSynchronous, 
                closestSphereKernelPtr, shiftSphereKernelPtr);

        rdDevCheckCall(cudaPeekAtLastError());
        if (debugSynchronous)
        {
            rdDevCheckCall(cub::SyncStream(stream));
        }

        #else
        error = cudaErrorNotSupported;
        #endif

        return error;
    }

    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        T const *       d_P,
        T *             d_S,
        T *             d_cordSums, 
        int *           d_spherePointCount,
        int             np,
        int             ns,
        T               r,
        int             pStride,
        int             sStride,
        int             csStride,
        cudaStream_t    stream,
        bool            debugSynchronous)
    {
        cudaError_t error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptxVersion;
            #if (CUB_PTX_ARCH == 0)
                if (CubDebug(error = cub::PtxVersion(ptxVersion))) break;
            #else
                ptxVersion = CUB_PTX_ARCH;
            #endif

            KernelConfig closestSphereConfig_, shiftSphereConfig_;

            // Get kernel kernel dispatch configurations
            InitConfigs(ptxVersion, closestSphereConfig_, shiftSphereConfig_);

            if (CubDebug(error = DispatchEvolveKernels(
                d_P,
                d_S,
                d_cordSums,
                d_spherePointCount,
                np,
                ns,
                r,
                pStride,
                sStride,
                csStride,
                stream,
                debugSynchronous,
                detail::DeviceClosestSphereKernel<PtxAgentClosestSpherePolicy, DIM, 
                    INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, T>,
                detail::DeviceShiftSphereKernel<PtxAgentShiftSpherePolicy, DIM, INPUT_MEM_LAYOUT, 
                    OUTPUT_MEM_LAYOUT, T>,
                closestSphereConfig_,
                shiftSphereConfig_ ,
                ptxVersion ))) break;

        }
        while (0);

        return error;
    }

    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        T const *       d_inputPoints,
        T const *       d_neighbourPoints,
        T *             d_chosenPoints,
        T *             d_cordSums, 
        int *           d_spherePointCount,
        int             inPointsNum,
        int             neighbourPointsNum,
        int             chosenPointsNum,
        T               r,
        int             inPointsStride,
        int             neighbourPointsStride,
        int             chosenPointsStride,
        int             cordSumsStride,
        cudaStream_t    stream,
        bool            debugSynchronous)
    {
        cudaError_t error = cudaSuccess;
        // enable only on devices with CC>=3.5 and with device runtime api enabled
        #if defined(CUB_RUNTIME_ENABLED) && (CUB_PTX_ARCH >= 350)
        do
        {
            if (CubDebug(error = DispatchEvolveKernels(
                d_inputPoints,
                d_neighbourPoints,
                d_chosenPoints,
                d_cordSums,
                d_spherePointCount,
                inPointsNum,
                neighbourPointsNum,
                chosenPointsNum,
                r,
                inPointsStride,
                neighbourPointsStride,
                chosenPointsStride,
                cordSumsStride,
                stream,
                debugSynchronous,
                detail::DeviceClosestSphereKernel<PtxAgentClosestSpherePolicy, DIM, 
                    INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, T>,
                detail::DeviceShiftSphereKernel<PtxAgentShiftSpherePolicy, DIM, INPUT_MEM_LAYOUT, 
                    OUTPUT_MEM_LAYOUT, T>))) break;

        }
        while (0);
        #else
        error = cudaErrorNotSupported;
        #endif

        return error;
    }

    static __host__ cudaError_t setCacheConfig()
    {
        cudaError error = cudaSuccess;
        do
        {
            if (CubDebug(error = cudaFuncSetCacheConfig(detail::DeviceClosestSphereKernel<
                PtxAgentClosestSpherePolicy, DIM, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, T>, 
                cudaFuncCachePreferL1))) break;

            if (CubDebug(error = cudaFuncSetCacheConfig(detail::DeviceShiftSphereKernel<
                PtxAgentShiftSpherePolicy, DIM, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT, T>, 
                cudaFuncCachePreferL1))) break;
        }
        while (0);

        return error;
    }
};

} // end namespace bruteForce
} // end namespace gpu
} // end namespace rd

#endif // __DISPATCH_EVOLVE_CUH__