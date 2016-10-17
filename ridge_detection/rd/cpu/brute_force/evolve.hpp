/**
 * @file evolve.hpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering, Faculty of Electronics and Information
 * Technology, Warsaw University of Technology 2016
 */

#ifndef CPU_BRUTE_FORCE_EVOLVE_HPP
#define CPU_BRUTE_FORCE_EVOLVE_HPP

#include "rd/cpu/brute_force/rd_inner.hpp"
#include "rd/utils/utilities.hpp"

#include <vector>
#include <limits>
#include <cmath>
#include <utility>

namespace rd
{
namespace cpu
{
namespace brute_force
{

namespace detail
{

template <typename T>
void calculateClosestSphereCenter(
    T const *   samples,
    T const *   chosenSamples,
    T *         cordSums,
    int *       spherePointCount,
    size_t      sCnt,
    size_t      csCnt,
    size_t      dim,
    T           rSqr)
{
    for (size_t n = 0; n < sCnt; ++n)
    {
        T const *sPtr = chosenSamples;
        T minSquareDist = std::numeric_limits<T>::max();
        T sqDist;
        int minSIndex = -1;
        T dist = 0;

        // through all chosen points
        for (size_t k = 0; k < csCnt; ++k) 
        {
            sqDist = 0;
            // through point dimensions
            for (size_t d = 0; d < dim; ++d, sPtr++) 
            {
                dist = *sPtr - samples[n * dim + d];
                sqDist += dist * dist;
            }
            if (sqDist < minSquareDist) 
            {
                minSIndex = (int)k;
                minSquareDist = sqDist;
            }
        }
        if (minSquareDist <= rSqr) 
        {
            spherePointCount[minSIndex]++;
            // sum point coordinates for later mass center calculation
            for (size_t d = 0; d < dim; ++d) 
            {
                cordSums[minSIndex * dim + d] += samples[n * dim + d];
            }
        }
    }
}

template <typename T>
void calculateClosestSphereCenter_omp(
    T const *   samples,
    T const *   chosenSamples,
    T *         cordSums,
    int *       spherePointCount,
    size_t      sCnt,
    size_t      csCnt,
    size_t      dim,
    T           rSqr)
{
    #pragma omp for schedule(static)
    for (size_t n = 0; n < sCnt; ++n)
    {
        T const *sPtr = chosenSamples;
        T minSquareDist = std::numeric_limits<T>::max();
        T sqDist;
        int minSIndex = -1;
        T dist = 0;

        // through all chosen points
        for (size_t k = 0; k < csCnt; ++k) 
        {
            sqDist = 0;
            // through point dimensions
            for (size_t d = 0; d < dim; ++d, sPtr++) 
            {
                dist = *sPtr - samples[n * dim + d];
                sqDist += dist * dist;
            }
            if (sqDist < minSquareDist) 
            {
                minSIndex = (int)k;
                minSquareDist = sqDist;
            }
        }
        if (minSquareDist <= rSqr) 
        {
            #pragma omp critical
            {
                spherePointCount[minSIndex]++;
                for (size_t d = 0; d < dim; ++d) 
                {
                    cordSums[minSIndex * dim + d] += samples[n * dim + d];
                }
            }
        }
    }
}

template <typename T>
void shiftTowardMassCenter(
    T *         chosenSamples,
    T const *   cordSums,
    int const * spherePointCount,
    size_t      csCnt,
    size_t      dim,
    int &       contFlag,
    int         ompNumThreads = 1,
    bool        verbose = false)
{
    if (verbose) {};
    if (ompNumThreads) {};  // suppress warnings
                            
    for (size_t k = 0; k < csCnt; ++k) 
    {
        for (size_t d = 0; d < dim; ++d) 
        {
            if ((T)spherePointCount[k]) 
            {
                T massCenter = cordSums[k * dim + d] / (T)spherePointCount[k];
                // if distance from mass center is numerically distinguishable
                if (std::fabs(massCenter - chosenSamples[k * dim + d])
                        > 2.f * std::fabs(massCenter) * spherePointCount[k]
                         * std::numeric_limits<T>::epsilon()) 
                {
                    // if (verbose) 
                    // {
                        /*
                         *   #if defined(RD_USE_OPENMP)
                         *   #pragma omp critical
                         *    {
                         *   #endif
                         *       std::cout << "duza zmiana: " << ", k: " << std::left << std::setw(4)
                         *        << k << ", massCenter: " << std::right << std::setw(15)
                         *        << massCenter << ", chosenSamples[" << k * dim + d << "]:" << std::left
                         *        << std::setw(15) << chosenSamples[k * dim + d] << ", roznica: "
                         *        << std::left << std::setw(15)
                         *        << fabs(massCenter - chosenSamples[k * dim + d]) << std::endl;
                         *   #if defined(RD_USE_OPENMP)
                         *    }
                         *   #endif
                         */
                    // }
                    contFlag = 1;
                    chosenSamples[k * dim + d] = massCenter;
                }
            }
        }
    }
}

template <typename T>
void shiftTowardMassCenter_omp(
    T *         chosenSamples,
    T const *   cordSums,
    int const * spherePointCount,
    size_t      csCnt,
    size_t      dim,
    int &       contFlag,
    int         ompNumThreads = 1,
    bool        verbose = false)
{
    if (verbose) {};    // suppres warnings

    #pragma omp parallel for num_threads(ompNumThreads), schedule(static)
    for (size_t k = 0; k < csCnt; ++k) 
    {
        for (size_t d = 0; d < dim; ++d) 
        {
            if ((T)spherePointCount[k]) 
            {
                T massCenter = cordSums[k * dim + d] / (T)spherePointCount[k];
                // if distance from mass center is numerically distinguishable
                if (std::fabs(massCenter - chosenSamples[k * dim + d])
                        > 2.f * std::fabs(massCenter) * spherePointCount[k]
                         * std::numeric_limits<T>::epsilon()) 
                {
                    // if (verbose) 
                    // {
                        /*
                         *   #if defined(RD_USE_OPENMP)
                         *   #pragma omp critical
                         *    {
                         *   #endif
                         *       std::cout << "duza zmiana: " << ", k: " << std::left << std::setw(4)
                         *        << k << ", massCenter: " << std::right << std::setw(15)
                         *        << massCenter << ", chosenSamples[" << k * dim + d << "]:" << std::left
                         *        << std::setw(15) << chosenSamples[k * dim + d] << ", roznica: "
                         *        << std::left << std::setw(15)
                         *        << fabs(massCenter - chosenSamples[k * dim + d]) << std::endl;
                         *   #if defined(RD_USE_OPENMP)
                         *    }
                         *   #endif
                         */
                    // }
                    #pragma omp atomic write
                    contFlag = 1;

                    chosenSamples[k * dim + d] = massCenter;
                }
            }
        }
    }
}

}   // end namespace detail

template <typename T>
void evolve_omp(
    T const *   samples,
    T *         chosenSamples,
    size_t      sCnt,
    size_t      csCnt,
    size_t      dim,
    T           r,
    int         ompNumThreads = 1,
    bool        verbose = false)
{
    T rSqr = r * r;
    int contFlag = 1;

    // Sum of coordinates of points falling into the intersection of sphere with
    // Voronoi cell. Used to calculating mass center.
    T * cordSums = new T[csCnt * dim];
    // Table containing points count which falls into the intersection of
    // sphere centered in one of the chosen points, with radius r1_ and a respective Voronoi cell
    int * spherePointCount = new int[csCnt];

    while (contFlag)
    {
        #pragma omp parallel num_threads(ompNumThreads)
        {
        fillTable_omp(cordSums, T(0), csCnt * dim);
        fillTable_omp(spherePointCount, int(0), csCnt);
        detail::calculateClosestSphereCenter_omp(
            samples,
            chosenSamples,
            cordSums,
            spherePointCount,
            sCnt,
            csCnt,
            dim,
            rSqr);
        }   // end omp parallel

        contFlag = 0;

        detail::shiftTowardMassCenter_omp(
            chosenSamples,
            cordSums,
            spherePointCount,
            csCnt,
            dim,
            contFlag,
            ompNumThreads,
            verbose);
    }

    delete[] spherePointCount;
    delete[] cordSums;
}

template <typename T>
void evolve(
    T const *   samples,
    T *         chosenSamples,
    size_t      sCnt,
    size_t      csCnt,
    size_t      dim,
    T           r,
    int         ompNumThreads = 1,
    bool        verbose = false)
{
    T rSqr = r * r;
    int contFlag = 1;

    // Sum of coordinates of points falling into the intersection of sphere with
    // Voronoi cell. Used to calculating mass center.
    T * cordSums = new T[csCnt * dim];
    // Table containing points count which falls into the intersection of
    // sphere centered in one of the chosen points, with radius r1_ and a respective Voronoi cell
    int * spherePointCount = new int[csCnt];

    while (contFlag)
    {
        fillTable(cordSums, T(0), csCnt * dim);
        fillTable(spherePointCount, int(0), csCnt);
        detail::calculateClosestSphereCenter(
            samples,
            chosenSamples,
            cordSums,
            spherePointCount,
            sCnt,
            csCnt,
            dim,
            rSqr);

        contFlag = 0;

        detail::shiftTowardMassCenter(
            chosenSamples,
            cordSums,
            spherePointCount,
            csCnt,
            dim,
            contFlag,
            ompNumThreads,
            verbose);
    }

    delete[] spherePointCount;
    delete[] cordSums;
}

template <typename T>
void evolve_neighbours(
    std::vector<std::pair<T const*, size_t>> samples,
    T *                             chosenSamples,
    size_t                          csCnt,
    size_t                          dim,
    T                               r,
    int                             ompNumThreads = 1,
    bool                            verbose = false)
{
    T rSqr = r * r;
    int contFlag = 1;

    // Sum of coordinates of points falling into the intersection of sphere with
    // Voronoi cell. Used to calculating mass center.
    T * cordSums = new T[csCnt * dim];
    // Table containing points count which falls into the intersection of
    // sphere centered in one of the chosen points, with radius r1_ and a respective Voronoi cell
    int * spherePointCount = new int[csCnt];

    /*
     * Because this evolve version is called only in rd_tiled, we don't define omp pragmas here.
     */
    while (contFlag) 
    {
        fillTable(cordSums, T(0), csCnt * dim);
        fillTable(spherePointCount, int(0), csCnt);

        for (size_t i = 0; i < samples.size(); ++i)
        {
            detail::calculateClosestSphereCenter(
                samples[i].first,
                chosenSamples,
                cordSums,
                spherePointCount,
                samples[i].second,
                csCnt,
                dim,
                rSqr);
        }
        contFlag = 0;

        detail::shiftTowardMassCenter(
            chosenSamples,
            cordSums,
            spherePointCount,
            csCnt,
            dim,
            contFlag,
            ompNumThreads,
            verbose);
    }

    delete[] spherePointCount;
    delete[] cordSums;
}

template <typename T>
void evolve_neighbours_omp(
    std::vector<std::pair<T const*, size_t>> samples,
    T *                             chosenSamples,
    size_t                          csCnt,
    size_t                          dim,
    T                               r,
    int                             ompNumThreads = 1,
    bool                            verbose = false)
{
    T rSqr = r * r;
    int contFlag = 1;

    // Sum of coordinates of points falling into the intersection of sphere with
    // Voronoi cell. Used to calculating mass center.
    T * cordSums = new T[csCnt * dim];
    // Table containing points count which falls into the intersection of
    // sphere centered in one of the chosen points, with radius r1_ and a respective Voronoi cell
    int * spherePointCount = new int[csCnt];

    /*
     * Because this evolve version is called only in rd_tiled, we don't define omp pragmas here.
     */
    while (contFlag) 
    {
        #pragma omp parallel num_threads(ompNumThreads)
        {
            fillTable_omp(cordSums, T(0), csCnt * dim);
            fillTable_omp(spherePointCount, int(0), csCnt);
        }
        #pragma omp parallel for num_threads(ompNumThreads), schedule(dynamic)
        for (size_t i = 0; i < samples.size(); ++i)
        {
            detail::calculateClosestSphereCenter(
                samples[i].first,
                chosenSamples,
                cordSums,
                spherePointCount,
                samples[i].second,
                csCnt,
                dim,
                rSqr);
        }
        contFlag = 0;

        detail::shiftTowardMassCenter_omp(
            chosenSamples,
            cordSums,
            spherePointCount,
            csCnt,
            dim,
            contFlag,
            ompNumThreads,
            verbose);
    }

    delete[] spherePointCount;
    delete[] cordSums;
}

} // end namespace brute_force
} // end namespace cpu
} // end namespace rd

#endif // CPU_BRUTE_FORCE_EVOLVE_HPP