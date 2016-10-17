/**
 * @file samples_generator.hpp
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

#include "rd/utils/utilities.hpp"
#if __cplusplus >= 201103L
    #include <random>
#else
    #include "rd/utils/box_muller.hpp"
#endif

#include <cmath>

namespace rd
{

    /**
     * @brief      Generates some nice parametric spiral.
     *
     * @param[in]  n          Number of samples to generate.
     * @param[in]  a          Spiral parameter max value.
     * @param[in]  b          Scaling width parameter
     * @param[in]  sigma      Standard deviation.
     * @param      container  Pointer to storage memory for samples
     *
     * @tparam     T          Point's coordinate data type.
     */
    template <typename T>
    static void genSpiral2D(size_t n, T a, T b, T sigma, T * container)
    {
        // we want to have a sigma standard deviation of samples so we have to 
        // scale it down, because distance is growing with sqr(dim)
        sigma *= T(1.f) / std::sqrt(T(2.f));
        size_t dim = 2;
        size_t size = n;
        fillRandomDataTable(container, size, T(0), a);
        for (int i = (int) (size - 1); i >= 0; --i)
        {
            container[i*dim+1] = b*container[i] * std::sin(container[i]);
            container[i*dim  ] = b*container[i] * std::cos(container[i]);
        }

        #if __cplusplus >= 201103L
        std::random_device rd;
        std::mt19937 e2(rd());
        std::normal_distribution<> dist(0, sigma);

        for (size_t i = 0; i < n*dim; ++i) {
            container[i] += dist(e2);
        }
        #else
        for (size_t i = 0; i < size*dim; ++i) {
            container[i] += rand_normal(T(0), sigma);
        }
        #endif
    }

    /**
     * @brief      Generates some nice parametric spiral.
     *
     * @param[in]  n      Number of samples to generate.
     * @param[in]  a      Spiral parameter max value.
     * @param[in]  b      Scaling parameter.
     * @param[in]  sigma  Standard deviation.
     *
     * @return     Pointer to generated samples.
     */
    template <typename T>
    static void genSpiral3D(size_t n, T a, T b, T sigma, T * container)
    {
        // we want to have a sigma standard deviation of samples so we have to 
        // scale it down, because distance is growing with sqr(dim)
        sigma *= T(1.f) / std::sqrt(T(3.f));
        size_t dim = 3;
        size_t size = n;
        fillRandomDataTable(container, size, T(0), a);
        for (int i = (int)size-1; i >= 0; --i)
        {
            container[i*dim+2] = b*container[i];
            container[i*dim+1] = b*container[i] * std::sin(container[i]);
            container[i*dim  ] = b*container[i] * std::cos(container[i]);
        }

        #if __cplusplus >= 201103L
        std::random_device rd;
        std::mt19937 e2(rd());
        std::normal_distribution<> dist(0, sigma);

        for (size_t i = 0; i < size*dim; ++i) {
            container[i] += dist(e2);
        }
        #else
        for (size_t i = 0; i < size*dim; ++i) {
            container[i] += rand_normal(T(0), sigma);
        }
        #endif
    }

    /*-------------------------------------------------------------------------------------------------------------*//**
     * @brief      Generates Dim-dimensional set of N samples with normally distributed noise.
     *
     * @param      n          number of samples
     * @param      dim        dimension
     * @param      sigma      standard deviation
     * @param      container  Pointer to memory storage for samples
     * @param[in]  length     The length of generated segment and range of generated values.
     *
     * @tparam     T          Point's coordinate data type.
     */
    template <
        typename T,
        typename std::enable_if<std::is_integral<T>::value, T>::type* = nullptr>
    static void genSegmentND(size_t n, size_t dim, T sigma, T * container, size_t length = 1)
    {
        #if __cplusplus >= 201103L

        // we want to have a sigma standard deviation of samples so we have to 
        // scale it down, because distance is growing with sqr(dim)
        sigma *= T(1.f) / std::sqrt(T(dim));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> udist(0, int(length));
        std::normal_distribution<> ndist(0, sigma);
    
        for (size_t i = 0; i < n; ++i) 
        {
            container[i * dim] = udist(gen);
            container[i * dim] += ndist(gen);
            for (size_t d = 1; d < dim; ++d)
            {
                container[i * dim + d] = ndist(gen);
            }
        }
        #else
        #error "Please use c++11 capable compiler version!"
        #endif
    }

    template <
        typename T,
        typename std::enable_if<std::is_floating_point<T>::value, T>::type* = nullptr>
    static void genSegmentND(size_t n, size_t dim, T sigma, T * container, size_t length = 1)
    {
        #if __cplusplus >= 201103L
        // we want to have a sigma standard deviation of samples so we have to 
        // scale it down, because distance is growing with sqr(dim)
        sigma *= T(1.f) / std::sqrt(T(dim));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> udist(0, T(length));
        std::normal_distribution<> ndist(0, sigma);

        for (size_t i = 0; i < n; ++i) 
        {
            container[i * dim] = udist(gen);
            container[i * dim] += ndist(gen);
            for (size_t d = 1; d < dim; ++d)
            {
                container[i * dim + d] = ndist(gen);
            }
        }
        #else
        #error "Please use c++11 capable compiler version!"
        #endif
    }


    /**
     * @brief      Generates 2D circle centered at (x0, y0)
     *
     * @param[in]  n          Number of points 
     * @param[in]  x0         The x0
     * @param[in]  y0         The y0
     * @param[in]  r          circle radius
     * @param[in]  sigma      Standard deviation.
     * @param      container  The container for generated data
     *
     * @tparam     T          Data type
     */
    template <typename T>
    static void genCircle2D(size_t n, T x0, T y0, T r, T sigma, T * container)
    {
        // we want to have a sigma standard deviation of samples so we have to 
        // scale it down, because distance is growing with sqr(dim)
        sigma *= T(1.f) / std::sqrt(T(2.f));
        T pi = 3.14159265359;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> udist(0, T(2.0) * pi);

        for (size_t k = 0; k < n; ++k)
        {
            T phi = udist(gen);
            T x,y;
            x = x0 + r * std::cos(phi);
            y = y0 + r * std::sin(phi);

            container[k * 2] = x;
            container[k * 2 + 1] = y;
        }

        if (sigma > 0)
        {
            std::mt19937 e2(rd());
            std::normal_distribution<> ndist(0, sigma);

            for (size_t k = 0; k < n*2; ++k)
            {
                container[k] += ndist(e2);
            }
        }
    }

 } // end namespace rd