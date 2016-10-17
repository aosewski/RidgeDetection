/**
 * @file test_evolve3_1.cu
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

#ifndef TEST_EVOLVE_GOLD
#define TEST_EVOLVE_GOLD

#include "rd/utils/utilities.hpp"
#include "rd/cpu/brute_force/evolve.hpp"

#include <limits>
#include <cstddef>
#include <cmath>

template <typename T>
void ccsc_gold(
    T const *P,
    T const *S,
    T *cordSums,
    int *spherePointCount,
    T r,
    size_t const np,
    size_t const ns,
    size_t const dim)
{
    #ifndef RD_USE_OPENMP
        rd::detail::calculateClosestSphereCenter(
            P, S, cordSums, spherePointCount, np, ns, dim, r*r);
    #else
        int numThr = omp_get_num_procs();
        #pragma omp parallel num_threads(numThr)
        {
            rd::detail::calculateClosestSphereCenter_omp(
                P, S, cordSums, spherePointCount, np, ns, dim, r*r);
        }
    #endif
}

template <typename T>
void stmc_gold(
    T *S,
    T const *cordSums,
    int const *spherePointCount,
    size_t ns, 
    size_t dim)
{
    int contFlag = 0;
    #ifndef RD_USE_OPENMP
        rd::detail::shiftTowardMassCenter(
            S, cordSums, spherePointCount, ns, dim, contFlag);
    #else
        int numThr = omp_get_num_procs();
        rd::detail::shiftTowardMassCenter_omp(
            S, cordSums, spherePointCount, ns, dim, contFlag, numThr);
    #endif
}

template <typename T>
void evolve_gold(
    T const *P,
    T *S,
    T r,
    size_t const np,
    size_t const ns,
    size_t const dim)
{
    #ifndef RD_USE_OPENMP
        rd::evolve(P, S, np, ns, dim, r);
    #else
        rd::evolve_omp(P,S, np, ns, dim, r, omp_get_num_procs());
    #endif
}

#endif
