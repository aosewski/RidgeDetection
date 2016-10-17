/**
 * @file dev_static_for.cuh
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

#include <type_traits>

namespace rd 
{
namespace gpu 
{

/*
 * if c++14 would be available we could use generic lambdas 
 * and call this like that:
 * 
 * StaticFor<0, 8>()([](auto size) {
 *      Vector<decltype(size)::value> var;
 * });
 */

template <int COUNT, int MAX, typename Lambda>
struct StaticFor 
{
    template <typename... Args>
    static __device__ void impl(Args&&... args) 
    {
        Lambda::impl(std::integral_constant<int, COUNT>{}, std::forward<Args>(args)...);
        StaticFor<COUNT + 1, MAX, Lambda>::impl(std::forward<Args>(args)...);
    }
};

template <int N, typename Lambda>
struct StaticFor<N, N, Lambda> 
{
    template <typename... Args>
    static __device__ void impl(Args&&... args) 
    {
    }
};

} // end namespace gpu
} // end namespace rd