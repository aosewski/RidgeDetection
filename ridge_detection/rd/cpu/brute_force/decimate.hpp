/**
 * @file decimate.hpp
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

#ifndef CPU_BRUTE_FORCE_DECIMATE_HPP
#define CPU_BRUTE_FORCE_DECIMATE_HPP

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
 * @brief      Removes redundant points from @p chosenPoints set.
 *
 *             The function removes points satisfying at least one of the two following conditions:
 * @li         point has at least 4 neighbours in R2 neighbourhood
 * @li         point has at most 2 neighbours in 2R2 neighbourhood
 *
 * @note       The algorithm stops when there is no more than 3 points left, or when there are no
 *             points satisfying specified criteria.
 *
 * @param      chosenPoints  The chosen points
 * @param      cpList        The chosen points pointers list
 * @param      cpCnt         The chosen points count
 * @param[in]  dim           The points dimensionality
 * @param[in]  r             Second RD algorithm parameter used for checking wheather or not to
 *                           remove particular point
 *
 * @tparam     T             Point's coordinate data type.
 */
template <typename T>
void decimate(
    T *             chosenPoints,
    std::list<T*> & cpList,
    size_t &        cpCnt,
    size_t          dim,
    T               r)
{
    T rSqr = r * r;
    // number of chosen points in previous iteration
    size_t prevCnt = 0;

    /*
     * Erases only pointers from list thus don't have to move 
     * large amount of memory with each removed point. 
     * There is only one copy performed in the end.
     */
    while (prevCnt != cpCnt && cpCnt > 3) 
    {
        prevCnt = cpCnt;
        for (auto it = cpList.begin(); it != cpList.end();)
        {
            if (countNeighbouringPoints(cpList, *it, dim, rSqr, 4) ||
                    !countNeighbouringPoints(cpList, *it, dim, 4.f*rSqr, 3)) 
            {
                cpCnt--;
                it = cpList.erase(it);
                if (cpCnt < 3) break;
            } else it++;
        }
    }

    T* dstAddr = chosenPoints;
    // copy real data
    for (auto it = cpList.begin(); it != cpList.end(); dstAddr += dim, it++)
    {
        if (*it != dstAddr)
        {
            copyTable<T>(*it, dstAddr, dim);
            *it = dstAddr;
        }
    }
}


/**
 * @brief      Removes redundant points from @p chosenPoints set.
 *
 *             The function removes points satisfying at least one of the two following conditions:
 * @li         point has at least 4 neighbours in R2 neighbourhood
 * @li         point has at most 2 neighbours in 2R2 neighbourhood
 *
 * @note       The algorithm stops when there is no more than 3 points left, or when there are no
 *             points satisfying specified criteria. Do not perform data copy in the end.
 *
 * @param      cpList  List of independent tile's list of chosen samples pointers.
 * @param      cpCnt   The overall (global, from all tiles) chosen points count
 * @param[in]  dim     The points dimensionality
 * @param[in]  r       Second RD algorithm parameter used for checking wheather or not to remove
 *                     particular point
 *
 * @tparam     T       The point's coordinate data type.
 */
template <typename T>
void decimateNoCopy(
    std::list<std::list<T*>*> & cpList,
    size_t &        cpCnt,
    size_t          dim,
    T               r)
{
    T rSqr = r * r;
    // number of chosen points in previous iteration
    size_t prevCnt = 0;

    /*
     * Erases only pointers from list thus don't have to move 
     * large amount of memory with each removed point. 
     */
    while (prevCnt != cpCnt && cpCnt > 3) 
    {
        prevCnt = cpCnt;
        for (auto lit = cpList.begin(); lit != cpList.end(); lit++)
        {
            std::list<T*> *curList = *lit;
            curList->remove_if([&cpCnt, dim, rSqr, &cpList](T* sptr)
            {
                if (countNeighbouringPoints(cpList, sptr, dim, rSqr, 4) ||
                    !countNeighbouringPoints(cpList, sptr, dim, 4.f*rSqr, 3)) 
                {
                    if (cpCnt - 1 < 3)
                    {
                        return false;
                    }
                    else
                    {
                        cpCnt--;
                        return true;
                    }
                }
                else
                {
                    return false;
                }
            });
        }
    }
}

}   // end namespace brute_force
}   // end namespace cpu
}   // end namespace rd

#endif // CPU_BRUTE_FORCE_DECIMATE_HPP
