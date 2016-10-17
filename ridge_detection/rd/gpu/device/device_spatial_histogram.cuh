/**
 * @file device_spatial_histogram.cuh
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

#include "rd/gpu/device/dispatch/dispatch_spatial_histogram.cuh"
#include "rd/gpu/device/bounding_box.cuh"

namespace rd
{
namespace gpu
{

struct DeviceHistogram
{
    /**
     * @brief      Device-wide calculation of input data spatial histogram.
     *
     * @param      d_tempStorage      Device-accessible allocation of temporary storage.  When NULL,
     *                                the required allocation size is written to @p tempStorageBytes
     *                                and no work is done.
     * @param      tempStorageBytes   Reference to size in bytes of @p d_tempStorage allocation
     * @param[in]  d_samples          Input points set.
     * @param      d_outputHistogram  Pointer to output data histogram.
     * @param[in]  samplesCnt         Number of points in @p d_samples set.
     * @param      binsCnt            Number of bins in respective dimension we divide data onto.
     * @param      bbox               Bounding box of input data set.
     * @param[in]  stride             Input data stride. (Distance between consecutive point
     *                                coordinates)
     * @param[in]  useGmemPrivHist    Whether or not to use private global memory histograms
     * @param[in]  stream             Device stream to run computation with. If zero use default
     *                                stream.
     * @param[in]  debugSynchronous   Wheather or not synchronize after each kernel and print log
     *                                messages.
     *
     * @tparam     DIM                Input points dimension.
     * @tparam     INPUT_MEM_LAYOUT   Input data memory layout. (ROW/COL-MAJOR)
     * @tparam     SampleT            Pointer to input data.
     * @tparam     CounterT           Integer type for histogram counters.
     *
     * @return     cudaSucces or cudaError if any arise.
     */
    template <
        int                 DIM,
        DataMemoryLayout    INPUT_MEM_LAYOUT,
        typename            SampleT,
        typename            CounterT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t spatialHistogram(
        void *                      d_tempStorage,             
        size_t &                    tempStorageBytes,         
        SampleT const *             d_samples,
        CounterT *                  d_outputHistogram,
        int                         samplesCnt,
        int const                   (&binsCnt)[DIM],
        BoundingBox<DIM, SampleT> const & bbox,
        int                         stride,
        bool                        useGmemPrivHist = true,
        cudaStream_t                stream = nullptr,
        bool                        debugSynchronous = false)
    {
        typedef int OffsetT; 

        return DispatchSpatialHistogram<
            SampleT, CounterT, OffsetT, DIM, INPUT_MEM_LAYOUT>::dispatch(
                d_tempStorage,
                tempStorageBytes,
                d_samples,
                samplesCnt,
                d_outputHistogram,
                stride,
                binsCnt,
                bbox,
                useGmemPrivHist,
                stream,
                debugSynchronous);
    }

    /**
     * @brief      Device-wide calculation of input data spatial histogram.
     *
     * @param      d_tempStorage      Device-accessible allocation of temporary storage.  When NULL,
     *                                the required allocation size is written to @p tempStorageBytes
     *                                and no work is done.
     * @param      tempStorageBytes   Reference to size in bytes of @p d_tempStorage allocation
     * @param[in]  d_samples          Input points set.
     * @param      d_outputHistogram  Pointer to output data histogram.
     * @param[in]  pointsCnt          Number of points in @p d_samples set.
     * @param[in]  numBins            Overall number of bins we divide data onto.
     * @param[in]  decodeOp           User-provided functor obj for custom mapping input points on
     *                                bin indexes.
     * @param[in]  aggregateHist      Controls wheather or not to zeroize @p d_outputHistogram befor
     *                                calculations. Enable aggregating few histograms into one.
     * @param[in]  stride             Input data stride. (Distance between consecutive point
     *                                coordinates)
     * @param[in]  stream             Device stream to run computation with. If zero use default
     *                                stream.
     * @param[in]  useGmemPrivHist    Whether or not to use private global memory histograms
     * @param[in]  debugSynchronous   Wheather or not synchronize after each kernel and print log
     *                                messages.
     *
     * @tparam     DIM                Input points dimension.
     * @tparam     INPUT_MEM_LAYOUT   Input data memory layout. (ROW/COL-MAJOR)
     * @tparam     SampleT            Pointer to input data.
     * @tparam     CounterT           Integer type for histogram counters.
     * @tparam     PointDecodeOpT     Functor structure for mapping input points on bin indexes.
     *
     * @return     cudaSuccess or cudaError
     */
    template <
        int                 DIM,
        DataMemoryLayout    INPUT_MEM_LAYOUT,
        typename            SampleT,
        typename            CounterT,
        typename            PointDecodeOpT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t spatialHistogram(
        void *                      d_tempStorage,             
        size_t &                    tempStorageBytes,         
        SampleT const *             d_samples,
        CounterT *                  d_outputHistogram,
        int                         pointsCnt,
        int                         numBins,
        PointDecodeOpT              decodeOp,
        int                         stride,
        bool                        aggregateHist = false,
        bool                        useGmemPrivHist = true,
        cudaStream_t                stream = 0,
        bool                        debugSynchronous = false)
    {
        typedef int OffsetT; 

        return DispatchSpatialHistogram<
            SampleT, CounterT, OffsetT, DIM, INPUT_MEM_LAYOUT>::dispatch(
                d_tempStorage,
                tempStorageBytes,
                d_samples,
                pointsCnt,
                d_outputHistogram,
                numBins,
                decodeOp,
                aggregateHist,
                useGmemPrivHist,
                stride,
                stream,
                debugSynchronous);
    }

    

};

} // end namespace gpu
} // end namespace rd