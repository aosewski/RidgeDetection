/**
 * @file device_find_bounds.cuh
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

#include "rd/gpu/device/dispatch/dispatch_find_bounds.cuh"

namespace rd
{
namespace gpu
{

struct DeviceFindBounds
{

    template <
        int               DIM,
        DataMemoryLayout  INPUT_MEM_LAYOUT,
        typename          SampleT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t findBounds(
        void *                  d_temp_storage,             ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t &                temp_storage_bytes,         ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SampleT const *         d_samples,
        SampleT *               d_bbox_min,
        SampleT *               d_bbox_max,
        int                     samplesCnt,
        int                     stride,
        cudaStream_t            stream = 0,
        bool                    debugSynchronous = false)
    {
        typedef int OffsetT; 

        return DispatchFindBounds<SampleT, OffsetT, DIM, INPUT_MEM_LAYOUT>::dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_bbox_min,
            d_bbox_max,
            samplesCnt,
            stride,
            stream,
            debugSynchronous);
    }

};

} // end namespace gpu
} // end namespace rd
