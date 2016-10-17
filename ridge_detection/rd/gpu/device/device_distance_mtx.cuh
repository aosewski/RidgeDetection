/**
 * @file device_distance_mtx.cuh
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

#include "rd/utils/memory.h"
#include "rd/gpu/device/dispatch/dispatch_distance_mtx.cuh"

namespace rd
{
namespace gpu
{

struct DeviceDistanceMtx
{

    /**
     * @brief      Calculates symmetric Euclidean distance matrix from input matrix A.
     *
     * Input matrix A may have row/col-major memory layout, whereas output matrix is always 
     * row-major.
     *
     * @param      d_in              Pointer to (device) input data
     * @param      d_out             Pointer to (device) output matrix
     * @param[in]  width             The input matrix width - number of points dimensions
     * @param[in]  height            The input matrix height - number of points
     * @param[in]  inStride          The input matrix stride - nr of elements in leading dimension
     * @param[in]  outStride         The output matrix stride - nr of elements in a row
     * @param[in]  stream            The stream to run computations with
     * @param[in]  debugSynchronous  The debug synchronous - wheather or not to synchronize after
     *                               each kernel call to check for errors.
     *
     * @tparam     MEM_LAYOUT        Input matrix memory layout [COL/ROW-MAJOR]
     * @tparam     T                 Input data type.
     *
     * @return     Cuda error if any occur.
     */
    template <
        DataMemoryLayout MEM_LAYOUT,
        typename T>
    CUB_RUNTIME_FUNCTION __forceinline__ 
    static cudaError_t symmetricDistMtx(
        T const *       d_in,
        T *             d_out,
        const int       width,
        const int       height,
        const int       inStride,
        const int       outStride,
        cudaStream_t    stream = 0,
        bool            debugSynchronous = false)
    {
        return DispatchDistanceMtx<MEM_LAYOUT, T>::dispatch(
            d_in, d_out, width, height, inStride, outStride, stream, debugSynchronous);
    }

    template <
        DataMemoryLayout MEM_LAYOUT,
        typename T>
    static __host__ cudaError_t setCacheConfig()
    {
        return DispatchDistanceMtx<MEM_LAYOUT, T>::setCacheConfig();
    }

};

} // end namesapce gpu
} // end namesapce rd
