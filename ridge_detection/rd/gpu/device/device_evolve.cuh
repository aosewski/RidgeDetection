/**
 * @file device_evolve.cuh
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

#include "rd/gpu/device/dispatch/dispatch_evolve.cuh"

namespace rd
{
namespace gpu
{
namespace bruteForce
{

struct DeviceEvolve
{
    template <
        int         DIM,
        DataMemoryLayout  INPUT_MEM_LAYOUT,
        DataMemoryLayout  OUTPUT_MEM_LAYOUT,
        typename    T>
    CUB_RUNTIME_FUNCTION __forceinline__ 
    static cudaError_t evolve(
        T const *       d_inputPoints,
        T *             d_chosenSamples,
        T *             d_cordSums, 
        int *           d_spherePointCount,
        int             inputPointsNum,
        int             chosenPointsNum,
        T               r,
        int             inPtsStride,
        int             chosenPtsStride,
        int             cordSumsStride,
        cudaStream_t    stream = nullptr,
        bool            debugSynchronous = false)
    {
        return DispatchEvolve<T, DIM, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT>::Dispatch(
            d_inputPoints,
            d_chosenSamples,
            d_cordSums,
            d_spherePointCount,
            inputPointsNum,
            chosenPointsNum,
            r,
            inPtsStride,
            chosenPtsStride,
            cordSumsStride,
            stream,
            debugSynchronous);
    }

    template <
        int                 DIM,
        DataMemoryLayout    INPUT_MEM_LAYOUT,
        DataMemoryLayout    OUTPUT_MEM_LAYOUT,
        typename            T>
    CUB_RUNTIME_FUNCTION __forceinline__ 
    static cudaError_t evolve(
        T const *       d_inputPoints,
        T const *       d_neighbourPoints,
        T *             d_chosenSamples,
        T *             d_cordSums, 
        int *           d_spherePointCount,
        int             inputPointsNum,
        int             neighbourPointsNum,
        int             chosenPointsNum,
        T               r,
        int             inPtsStride,
        int             neighbourPtsStride,
        int             chosenPtsStride,
        int             cordSumsStride,
        cudaStream_t    stream = nullptr,
        bool            debugSynchronous = false)
    {
        // enable only on devices with CC>=3.5 and with device runtime api enabled
        #if defined(CUB_RUNTIME_ENABLED) && (CUB_PTX_ARCH >= 350)

        return DispatchEvolve<T, DIM, INPUT_MEM_LAYOUT, OUTPUT_MEM_LAYOUT>::Dispatch(
            d_inputPoints, d_neighbourPoints, d_chosenSamples, d_cordSums, d_spherePointCount,
            inputPointsNum, neighbourPointsNum, chosenPointsNum, r, inPtsStride, neighbourPtsStride,
            chosenPtsStride, cordSumsStride, stream, debugSynchronous);
        #else 
        return cudaErrorNotSupported;
        #endif
    }

    template <
        int               DIM,
        DataMemoryLayout  IN_MEM_LAYOUT,
        DataMemoryLayout  OUT_MEM_LAYOUT,
        typename    T>
    static __host__ cudaError_t setCacheConfig()
    {
        return DispatchEvolve<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>::setCacheConfig();
    }

};

}   // end namespace bruteForce
}   // end namespace gpu
}   // end namespace rd
