/**
 * @file device_choose.cuh
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

#ifndef __DEVICE_CHOOSE_CUH__
#define __DEVICE_CHOOSE_CUH__

#include "rd/gpu/device/dispatch/dispatch_choose.cuh"

namespace rd
{
namespace gpu
{
namespace bruteForce
{

struct DeviceChoose
{

    /**
     * @brief      Chose initial set of chosen points for ridge detection.
     *
     * @param      d_inputPoints      Pointer to input points set.
     * @param      d_chosenPoints     Pointer to storage for chosen points.
     * @param[in]  inputPointsNum     Number of input points
     * @param      d_chosenPointsNum  Pointer to device accessible number of chosen points.
     * @param[in]  r                  Ridge detection paramter. Radius used to seek for neighbours.
     * @param[in]  inPtsStride        Input points set stride.
     * @param[in]  chosenPtsStride    Chosen points set stride.
     * @param[in]  stream             The stream we run computations with.
     * @param[in]  debugSynchronous   Wheather or not to synchronize after each kernel launch to
     *                                check for errors.
     *
     * @tparam     DIM                Points dimension.
     * @tparam     IN_MEM_LAYOUT      @p d_inputPoints data memory layout (ROW/COL_MAJOR)
     * @tparam     OUT_MEM_LAYOUT     @p d_chosenPoints data memory layout (ROW/COL_MAJOR)
     * @tparam     T                  Point coordinate data type.
     *
     * @return     Errors if eny, otherwise cudaSuccess.
     *
     * @paragraph The function choses subset of @p d_inputPoints set of points, where each two of
     * them are @p r -separated. This means that there are no two different points closer than R.
     */
    template <
        int         DIM,
        DataMemoryLayout  IN_MEM_LAYOUT,
        DataMemoryLayout  OUT_MEM_LAYOUT,
        typename    T>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t choose(
            T const *    d_inputPoints,
            T *          d_chosenPoints,
            int          inputPointsNum,
            int *        d_chosenPointsNum,
            T            r, 
            int          inPtsStride,
            int          chosenPtsStride,
            cudaStream_t stream = nullptr,
            bool         debugSynchronous = false)      
    {
        return DispatchChoose<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>::dispatch(
            d_inputPoints,
            d_chosenPoints,
            inputPointsNum,
            d_chosenPointsNum,
            r,
            inPtsStride, 
            chosenPtsStride,
            stream,
            debugSynchronous);
    }

    template <
        int         DIM,
        DataMemoryLayout  IN_MEM_LAYOUT,
        DataMemoryLayout  OUT_MEM_LAYOUT,
        typename    T>
    static __host__ cudaError_t setCacheConfig()
    {
        return DispatchChoose<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>::setCacheConfig();
    }
};

} // bruteForce namespace
} // gpu namespace
} // rd namespace

#endif  // __DEVICE_CHOOSE_CUH__