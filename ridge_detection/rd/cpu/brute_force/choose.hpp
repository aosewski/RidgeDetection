/**
 * @file choose.hpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is supervised by prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */

#ifndef CPU_BRUTE_FORCE_CHOOSE_HPP
#define CPU_BRUTE_FORCE_CHOOSE_HPP

#include "rd/cpu/brute_force/rd_inner.hpp"
#include "rd/utils/utilities.hpp"

#include <list>

namespace rd
{

/**
 * @brief      Chose initial set S of the path's nodes.
 *
 *             The function choses subset of P_ set of points, where each
 *             two of them are R-separated. This means that thera are no two
 *             different points closer than R.
 *
 * @note       Points are chosen in the order they appear in samples set P.

 */
template <typename T>
void choose(
    T const *       samples,
    T *             chosenSamples,
    std::list<T*> & csList,
    size_t          sCnt,
    size_t &        csCnt,
    size_t          dim,
    T               r)
{
    size_t count = 0;
    T rSqr = r * r;
    T const *point = samples;

    // copy first point from samples to chosenSamples
    copyTable(samples, chosenSamples, dim);
    count++;
    point += dim;
    csList.clear();
    csList.push_back(chosenSamples);

    while (--sCnt > 0) {
        /*
         * check whether there is no points in chosen from which the squared
         * distance to 'point' is lower than rSqr
         */
        if (!countNeighbouringPoints(chosenSamples, count, point, dim, rSqr, 1))
        {
            copyTable(point, chosenSamples + dim * count, dim);
            csList.push_back(chosenSamples + dim * count++);
        }
        point += dim;
    }
    csCnt = count;
}

}   // end namespace rd

#endif // CPU_BRUTE_FORCE_CHOOSE_HPP