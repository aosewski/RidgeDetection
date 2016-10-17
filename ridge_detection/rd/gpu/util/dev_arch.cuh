/**
 * @file dev_arch.h
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and Information
 * Technology Warsaw University of Technology 2016
 * 
 */

#pragma once

#include "cub/util_arch.cuh"

namespace rd
{
namespace gpu
{
    
// maximum number of resident grids per device
#ifndef MAX_CONCURRENT_KERNELS
    #define MAX_CONCURRENT_KERNELS(arch)    \
        ((arch >= 530) ?                    \
            (16) :                          \
            ((arch >= 350) ?                \
                (32) :                      \
                ((arch >= 320) ?            \
                    (4) :                   \
                    (16))))
    #define RD_PTX_MAX_CONCURRENT_KERNELS   MAX_CONCURRENT_KERNELS(CUB_PTX_ARCH)
#endif

} // end namespace gpu
} // end namespace rd