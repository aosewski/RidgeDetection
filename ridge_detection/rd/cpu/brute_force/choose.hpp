/**
 * @file choose.hpp
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

#ifndef CPU_BRUTE_FORCE_CHOOSE_HPP
#define CPU_BRUTE_FORCE_CHOOSE_HPP

#include "rd/cpu/brute_force/rd_inner.hpp"
#include "rd/utils/utilities.hpp"

#include <list>

namespace rd
{
namespace cpu
{
namespace brute_force
{

/**
 * @brief      Chose initial chosenPoints set of the reconstructed path's nodes.
 *
 *             The function choses subset of input set of points, where each two of them are
 *             R-separated. This means that there are no two different points closer than R.
 *
 * @note       Points are chosen in the order they appear in the input points set.
 *
 * @param      inPoints        The input points set.
 * @param      chosenPoints  The chosen points set.
 * @param      cpList        The chosen points pointers list.
 * @param      pCnt          The points count.
 * @param      cpCnt         The chosen points count.
 * @param      dim           The point's dimensionality.
 * @param      r             The minimum distance between any two points from @p chosenPoints set.
 *
 * @tparam     T             Point coordinate data type.
 */
template <typename T>
void choose(
    T const *       inPoints,
    T *             chosenPoints,
    std::list<T*> & cpList,
    size_t          pCnt,
    size_t &        cpCnt,
    size_t          dim,
    T               r)
{
    size_t count = 0;
    T rSqr = r * r;
    T const *point = inPoints;

    // copy first point from points to chosenPoints
    copyTable(inPoints, chosenPoints, dim);
    count++;
    point += dim;
    cpList.clear();
    cpList.push_back(chosenPoints);

    while (--pCnt > 0) {
        /*
         * check whether there is no points in chosen from which the squared
         * distance to 'point' is lower than rSqr
         */
        if (!countNeighbouringPoints(chosenPoints, count, point, dim, rSqr, 1))
        {
            copyTable(point, chosenPoints + dim * count, dim);
            cpList.push_back(chosenPoints + dim * count++);
        }
        point += dim;
    }
    cpCnt = count;
}

}   // end namespace brute_force
}   // end namespace cpu
}   // end namespace rd

#endif // CPU_BRUTE_FORCE_CHOOSE_HPP
