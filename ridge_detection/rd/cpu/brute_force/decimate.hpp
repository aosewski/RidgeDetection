/**
 * @file decimate.hpp
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

#ifndef CPU_BRUTE_FORCE_DECIMATE_HPP
#define CPU_BRUTE_FORCE_DECIMATE_HPP

#include "rd/cpu/brute_force/rd_inner.hpp"
#include "rd/utils/utilities.hpp"

#include <list>

namespace rd
{
    

/**
 * @brief      Removes redundant points from set S.
 *
 *             The function removes points satisfying at least one of the
 *             two following conditions:
 * @li         point has at least 4 neighbours in R2 neighbourhood
 * @li         point has at most 2 neighbours in 2R2 neighbourhood
 *
 * The algorithm stops when there is no more than 3 points left, or when
 * there are no points satisfying specified criteria.
 *
 * @return     pointer to the changed S point set.
 */
template <typename T>
void decimate(
    T *             chosenSamples,
    std::list<T*> & csList,
    size_t &        csCnt,
    size_t          dim,
    T               r)
{
    T rSqr = r * r;
    size_t left = 0;

    /*
     * Erases only pointers from list thus don't have to move 
     * large amount of memory with each removed point.There is only one copy performed
     * in the end.
     */
    while (left != csCnt && csCnt > 3) 
    {
        left = csCnt;
        for (auto it = csList.begin(); it != csList.end();)
        {
            if (countNeighbouringPoints(csList, *it, dim, rSqr, 4) ||
                    !countNeighbouringPoints(csList, *it, dim, 4.f*rSqr, 3)) 
            {
                csCnt--;
                it = csList.erase(it);
                if (csCnt < 3) break;
            } else it++;
        }
    }

    T* dstAddr = chosenSamples;
    // copy real data
    for (auto it = csList.begin(); it != csList.end(); dstAddr += dim, it++)
    {
        if (*it != dstAddr)
        {
            copyTable<T>(*it, dstAddr, dim);
            *it = dstAddr;
        }
    }
}


template <typename T>
void decimateNoCopy(
    std::list<std::list<T*>*> & csList,
    size_t &        csCnt,
    size_t          dim,
    T               r)
{
    T rSqr = r * r;
    size_t left = 0;

    /*
     * Erases only pointers from list thus don't have to move 
     * large amount of memory with each removed point.There is only one copy performed
     * in the end.
     */
    while (left != csCnt && csCnt > 3) 
    {
        left = csCnt;
        for (auto lit = csList.begin(); lit != csList.end(); lit++)
        {
            std::list<T*> *curList = *lit;
            curList->remove_if([&csCnt, dim, rSqr, &csList](T* sptr)
            {
                if (countNeighbouringPoints(csList, sptr, dim, rSqr, 4) ||
                    !countNeighbouringPoints(csList, sptr, dim, 4.f*rSqr, 3)) 
                {
                    if (csCnt - 1 < 3)
                    {
                        return false;
                    }
                    else
                    {
                        csCnt--;
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

}   // end namespace rd

#endif // CPU_BRUTE_FORCE_DECIMATE_HPP