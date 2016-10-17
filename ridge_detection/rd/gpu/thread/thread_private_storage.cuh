/**
 * @file thread_private_storage.cuh
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

#pragma once

#include "rd/utils/memory.h"
#include "rd/utils/macro.h"

#include "cub/util_type.cuh"

namespace rd
{
namespace gpu
{

// specialiation for ROW_MAJOR data layout
template <
    int                 ITEMS_PER_THREAD,
    int                 DIM,
    DataMemoryLayout    MEM_LAYOUT,
    typename            SampleT>
struct RD_MEM_ALIGN(16) ThreadPrivatePointsAligned 
{
    typedef cub::ArrayWrapper<SampleT, DIM> PointT;
    typedef SampleT AliasedSamples[ITEMS_PER_THREAD * DIM];
    typedef PointT AliasedPoints[ITEMS_PER_THREAD];

    SampleT data[ITEMS_PER_THREAD][DIM];

    __device__ __forceinline__ SampleT * samplesPtr() 
    {
        return data;
    }

    __device__ __forceinline__ SampleT const * samplesCPtr() const
    {
        return data;
    }

    __device__ __forceinline__ PointT * pointsPtr() 
    {
        return reinterpret_cast<PointT*>(data);
    }

    __device__ __forceinline__ PointT const * pointsCPtr() const
    {
        return reinterpret_cast<PointT const *>(data);
    }

    __device__ __forceinline__ AliasedSamples & samplesRef() 
    {
        return reinterpret_cast<AliasedSamples &>(data);
    }

    __device__ __forceinline__ AliasedSamples const & samplesCRef() const
    {
        return reinterpret_cast<AliasedSamples &>(data);
    }

    __device__ __forceinline__ AliasedPoints & pointsRef() 
    {
        return reinterpret_cast<AliasedPoints &>(data);
    }

    __device__ __forceinline__ AliasedPoints const & pointsCRef() const
    {
        return reinterpret_cast<AliasedPoints &>(data);
    }
};

// specialization for COL_MAJOR data layout
template <
    int         ITEMS_PER_THREAD,
    int         DIM,
    typename    SampleT>
struct RD_MEM_ALIGN(16) ThreadPrivatePointsAligned<ITEMS_PER_THREAD, DIM, COL_MAJOR, SampleT> 
{
    typedef SampleT AliasedSamples[ITEMS_PER_THREAD * DIM];

    SampleT data[DIM][ITEMS_PER_THREAD];

    __device__ __forceinline__ SampleT * samplesPtr() 
    {
        return data;
    }

    __device__ __forceinline__ SampleT const * samplesCPtr() const
    {
        return data;
    }

    __device__ __forceinline__ AliasedSamples & samplesRef() 
    {
        return reinterpret_cast<AliasedSamples &>(data);
    }

    __device__ __forceinline__ AliasedSamples const & samplesCRef() const
    {
        return reinterpret_cast<AliasedSamples &>(data);
    }

};

} // end namespace gpu
} // end namespace rd