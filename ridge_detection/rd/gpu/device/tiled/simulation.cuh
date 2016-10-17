/**
 * @file simulation.cuh
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
#include <stdexcept>

#include "rd/gpu/device/device_tiled_ridge_detection.cuh"
#include "rd/gpu/util/dev_memcpy.cuh"

#include "rd/utils/memory.h"

namespace rd 
{
namespace gpu
{
namespace tiled
{

template <
    int                             DIM,
    DataMemoryLayout                IN_MEM_LAYOUT,
    DataMemoryLayout                OUT_MEM_LAYOUT,
    RidgeDetectionAlgorithm         RD_TILE_ALGORITHM,
    TiledRidgeDetectionPolicy       RD_TILE_POLICY,
    TileType                        RD_TILE_TYPE,
    typename                        T>
class RidgeDetection
{
private:
    int pointsNum_;
    int inPtsStride_;
    int chosenPointsNum_;
    int chosenPointsStride_;
    bool debugSynchronous_;
    bool enableTiming_;

    T *d_inputPoints_;
    T *d_chosenPoints_;
    int *d_chosenPointsNum_;

    detail::TiledRidgeDetectionTimers *d_rdTimers;
    detail::TiledRidgeDetectionTimers h_rdTimers;

public:
    RidgeDetection(
        int pointsNum,
        T const * inputPoints,
        bool enableTiming = false,
        bool debugSynchronous = false)
    :
        pointsNum_(pointsNum),
        inPtsStride_(DIM),
        chosenPointsNum_(0),
        chosenPointsStride_(0),
        enableTiming_(enableTiming),
        debugSynchronous_(debugSynchronous),
        d_chosenPoints_(nullptr),
        d_rdTimers(nullptr)
    {
        if (sizeof(T) >= 8)
        {
            checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
        }

        if (IN_MEM_LAYOUT == COL_MAJOR)
        {
            size_t pitch;
            checkCudaErrors(cudaMallocPitch(&d_inputPoints_, &pitch, 
                pointsNum_ * sizeof(T), DIM));
            inPtsStride_ = pitch / sizeof(T);
        }
        else
        {
            checkCudaErrors(cudaMalloc(&d_inputPoints_, pointsNum_ * DIM * sizeof(T)));
        }
        checkCudaErrors(cudaMalloc(&d_chosenPointsNum_, sizeof(int)));

        h_rdTimers.wholeTime = 0;
        h_rdTimers.rdTilesTime = 0;
        h_rdTimers.refinementTime = 0;

        if (enableTiming_)
        {
            checkCudaErrors(cudaMalloc(&d_rdTimers, sizeof(detail::TiledRidgeDetectionTimers)));
        }

        // copy and if necessary transpose input data to gpu device
        // input data are always in ROW_MAJOR order
        if (IN_MEM_LAYOUT == COL_MAJOR)
        {
            rdMemcpy2D<COL_MAJOR, ROW_MAJOR, cudaMemcpyHostToDevice>(
                d_inputPoints_, inputPoints, DIM, pointsNum_, inPtsStride_ * sizeof(T), 
                DIM * sizeof(T));
        }
        else if (IN_MEM_LAYOUT == ROW_MAJOR)
        {
            rdMemcpy<ROW_MAJOR, ROW_MAJOR, cudaMemcpyHostToDevice>(
                d_inputPoints_, inputPoints, DIM, pointsNum_, DIM, DIM);
        }
        else 
        {
            throw std::runtime_error("Unsupported memory layout!");
        }
        checkCudaErrors(cudaDeviceSynchronize());
    }

    ~RidgeDetection()
    {
        if (d_inputPoints_ != nullptr)
        {
            checkCudaErrors(cudaFree(d_inputPoints_));
        }
        if (d_chosenPoints_ != nullptr)
        {
            checkCudaErrors(cudaFree(d_chosenPoints_));
        }
        if (d_chosenPointsNum_ != nullptr)
        {
            checkCudaErrors(cudaFree(d_chosenPointsNum_));
        }
        if (d_rdTimers != nullptr)
        {
            checkCudaErrors(cudaFree(d_rdTimers));
        }
    }

    void operator()(
        T       r1, 
        T       r2,
        int     maxTileCapacity,
        int     tilesPerDim[DIM],
        bool    endPhaseRefinement)
    {
        approximate(r1, r2, maxTileCapacity, tilesPerDim, endPhaseRefinement);
    }

    void ridgeDetection(
        T       r1, 
        T       r2,
        int     maxTileCapacity,
        int     tilesPerDim[DIM],
        bool    endPhaseRefinement)
    {
        approximate(r1, r2, maxTileCapacity, tilesPerDim, endPhaseRefinement);
    }

    int getChosenPointsNum() 
    {
        if (debugSynchronous_)
        {
            std::cout << "RidgeDetection::getChosenPointsNum()" << std::endl;
        }

        checkCudaErrors(cudaMemcpy(&chosenPointsNum_, d_chosenPointsNum_, sizeof(int), 
            cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());
        return chosenPointsNum_;
    }

    void getChosenPoints(T * chosenPoints)
    {

        if (debugSynchronous_)
        {
            std::cout << "RidgeDetection::getChosenPoints() " << std::endl;
            std::cout << "dst: " << chosenPoints 
                    << ", src: " << d_chosenPoints_ 
                    << ", numElem: " << chosenPointsNum_
                    << ", dst stride: " << DIM
                    << ", src stride " << chosenPointsStride_
                    << std::endl;
        }

        // copy back to host results and if necessary transpose them to ROW_MAJOR order
        if (OUT_MEM_LAYOUT == ROW_MAJOR)
        {
            rdMemcpy<ROW_MAJOR, ROW_MAJOR, cudaMemcpyDeviceToHost>(
                chosenPoints, d_chosenPoints_, DIM, chosenPointsNum_, DIM, DIM);
        }
        else if (OUT_MEM_LAYOUT == COL_MAJOR)
        {
            rdMemcpy2D<ROW_MAJOR, COL_MAJOR, cudaMemcpyDeviceToHost>(
                chosenPoints, d_chosenPoints_, chosenPointsNum_, DIM, DIM * sizeof(T), 
                chosenPointsStride_ * sizeof(T));
        }
        else 
        {
            throw std::runtime_error("Unsupported memory layout!");
        }
        checkCudaErrors(cudaDeviceSynchronize());
    }

    detail::TiledRidgeDetectionTimers getTimers()
    {
        if (enableTiming_)
        {
            checkCudaErrors(cudaMemcpy(&h_rdTimers, d_rdTimers, 
                sizeof(detail::TiledRidgeDetectionTimers), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaDeviceSynchronize());
        }
        return h_rdTimers;
    }

private:

    void approximate(
        T       r1, 
        T       r2,
        int     maxTileCapacity,
        int     tilesPerDim[DIM],
        bool    endPhaseRefinement)
    {

        if (d_chosenPoints_ != nullptr)
        {
            checkCudaErrors(cudaFree(d_chosenPoints_));
            d_chosenPoints_ = nullptr;
        }

        DeviceTiledRidgeDetection::approximate<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, 
                RD_TILE_ALGORITHM, RD_TILE_POLICY, RD_TILE_TYPE>(
            d_inputPoints_, d_chosenPoints_, pointsNum_, d_chosenPointsNum_, r1, r2,
            inPtsStride_, chosenPointsStride_, maxTileCapacity,
            tilesPerDim, endPhaseRefinement, d_rdTimers, debugSynchronous_);
    }

};

} // namespace tiled
} // namespace gpu
} // namespace rd