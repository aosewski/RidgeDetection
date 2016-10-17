/**
 * @file dispatch_ridge_detection.cuh
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

#include "rd/gpu/util/dev_utilities.cuh"
#include "rd/gpu/device/device_choose.cuh"
#include "rd/gpu/device/device_evolve.cuh"
#include "rd/gpu/device/device_decimate.cuh"

#include "cub/util_type.cuh"
#include "cub/util_arch.cuh"
#include "cub/util_debug.cuh"

#include <assert.h>

namespace rd 
{
namespace gpu
{
namespace tiled
{

/**
 * @brief      Describes algorithm used in ridge detection's evolution phase.
 */
enum RidgeDetectionAlgorithm
{
    /**
     * @par Overview
     *     This is the simplest version which checks each point in input set with each chosen
     *     point to determine its closest neighbour.
     *
     * @par Performance Considerations
     *     This version performs cloud points times chosen points comparisons. It will be
     *     the slowest one, due to the amount of calculations needed to find the closest chosen
     *     point for each input point.
     */
    RD_BRUTE_FORCE,

    /**
     * @par Overview
     *     This version uses kd-tree to speed-up process of finding the closest chosen point for
     *     each input point.
     *
     * @par Performance Considerations
     *     Algorithm efficiency may drastically decrease after exceeding certain point dimension
     *     threashold.
     */
    RD_KD_TREE,

    /**
     * @par Overview
     *     This version uses bounding volume hierarchy to speed-up process of finding the closest
     *     chosen point for each input point.
     *     
     * @par Performance Considerations
     *     Algorithm efficiency may drastically decrease after exceeding certain point dimension
     *     threashold.
     */
    RD_BVH_TREE,
};

//-------------------------------------------------------------------------------
//  Algorithm variants implementations
//-------------------------------------------------------------------------------

template <
    int                     DIM,
    DataMemoryLayout        IN_MEM_LAYOUT,
    DataMemoryLayout        OUT_MEM_LAYOUT,
    typename                T>
CUB_RUNTIME_FUNCTION __forceinline__
static cudaError_t doApproximate(
    T const *   d_inputPoints,
    T const *   d_neighbourPoints,
    T *         d_chosenPoints,
    int         inPointsNum,
    int         neighbourPointsNum,
    int *       d_chosenPointsNum,
    T           r1,
    T           r2,
    int         inPointsStride,
    int         neighbourPointsStride,
    int         chosenPointsStride,
    bool        debugSynchronous,
    cub::Int2Type<RD_BRUTE_FORCE> )
{
    #ifndef CUB_RUNTIME_ENABLED
        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);
    #else 
        cudaError_t error = cudaSuccess;

        if (debugSynchronous)
        {
            _CubLog("Dispatch choose, inPtsNum: %d\n", inPointsNum);
        }

        error = bruteForce::DeviceChoose::choose<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
            d_inputPoints, d_chosenPoints, inPointsNum, d_chosenPointsNum, r1, 
            inPointsStride, chosenPointsStride);
        rdDevCheckCall(error);
        // wait for choose to end
        rdDevCheckCall(cudaDeviceSynchronize());

        int chosenPointsNum = *d_chosenPointsNum;
         int cordSumsStride;
        int distMtxStride;
        
        T * d_cordSums;
        T * d_distMtx = nullptr;
        
        char * d_pointsMask = new char[chosenPointsNum];
        int * d_spherePointCnt = new int[chosenPointsNum];
        // check allocations
        assert(d_pointsMask != nullptr);
        assert(d_spherePointCnt != nullptr);
        
        rdDevCheckCall(rdDevAllocMem(&d_cordSums, &cordSumsStride, DIM, chosenPointsNum, 
            cub::Int2Type<IN_MEM_LAYOUT>()));
        rdDevCheckCall(rdDevAllocMem(&d_distMtx, &distMtxStride, chosenPointsNum, chosenPointsNum, 
            cub::Int2Type<COL_MAJOR>()));


        int oldCount = 0;

        /*
         *  Repeat untill the count of chosen samples won't
         *  change in two consecutive iterations.
         */
        while (oldCount != chosenPointsNum) 
        {
            oldCount = chosenPointsNum;

            if (debugSynchronous)
            {
                _CubLog("Dispatch evolve, chosenPointsNum: %d\n", chosenPointsNum);
            }

            error = bruteForce::DeviceEvolve::evolve<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
                d_inputPoints, d_neighbourPoints, d_chosenPoints, d_cordSums, d_spherePointCnt, 
                inPointsNum, neighbourPointsNum, chosenPointsNum, r1, inPointsStride, 
                neighbourPointsStride, chosenPointsStride, cordSumsStride);
            rdDevCheckCall(error);

            if (debugSynchronous)
            {
                _CubLog("Dispatch decimate, chosenPointsNum: %d\n", chosenPointsNum);
            }

            error = bruteForce::DeviceDecimate::decimateDistMtx<DIM, OUT_MEM_LAYOUT>(
                d_chosenPoints, d_chosenPointsNum, chosenPointsStride, d_distMtx, 
                distMtxStride, d_pointsMask, r2, nullptr, debugSynchronous);
            rdDevCheckCall(error);
            rdDevCheckCall(cudaDeviceSynchronize());

            chosenPointsNum = *d_chosenPointsNum;
        }

        delete[] d_cordSums;
        delete[] d_spherePointCnt;
        delete[] d_distMtx;
        delete[] d_pointsMask;

        return error;
    #endif
}


} // end namespace tiled
} // end namespace gpu
} // end namespace rd
