/**
 * @file order.hpp
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

#pragma once

#include <cstddef>
#include <deque>
#include <list>
#include <iterator>

namespace rd
{

/**
 * @brief      Orders points in @a in set, so as each two consecutive points meets the @a cond
 *             function condition.
 *
 * @param      in             Input data set.
 * @param[in]  dim            Dimensionality of samples.
 * @param[in]  size           Number of samples.
 * @param[in]  cond           Function object which take four arguments: pointers to two different
 *                            points in @a in set, pointer to currently closest point found, points
 *                            dimensionality and return boolean value: bool operator()(PointType
 *                            const *p1, PointType const *p2, PointType const *pBest, int dim)
 *
 * @tparam     PointType      Samples data type.
 * @tparam     ConditionFunc  Condition function object type.
 *
 * @return     List of coherent chains.
 */
template <
    typename PointType,
    typename ConditionFunc>
std::list<std::deque<PointType const*>> orderSamples(
                PointType const *in,
                std::size_t dim,
                std::size_t size,
                ConditionFunc cond)
{

    typedef std::deque<PointType const*> ChainType;
    typedef std::list<ChainType> ChainListType;
    typedef std::list<PointType const *> ListType;

    ChainListType orderedLists;
    ListType points;
    PointType const *inPtr = in;
    for (std::size_t i = 0; i < size; ++i, inPtr += dim)
    {
        points.push_back(inPtr);
    }

    for (auto pointIt = points.begin();
         pointIt != points.end();
         ++pointIt)
    {

        orderedLists.emplace_back(1, *pointIt);
        ChainType &buildedChain = orderedLists.back();

        bool repeat = true;
        while (repeat)
        {
            repeat = false;

            for (auto currPointIt = std::next(pointIt);
                 currPointIt != points.end();)
            {
                if (cond(*currPointIt, buildedChain.front(), dim)) 
                {
                    buildedChain.push_front(*currPointIt);
                    points.erase(currPointIt);
                    repeat = true;
                    break;
                }
                else if (cond(*currPointIt, buildedChain.back(), dim))
                {
                    buildedChain.push_back(*currPointIt);
                    points.erase(currPointIt);
                    repeat = true;
                    break;
                }
                else
                    ++currPointIt;
            }

            /*
             * [17.05.16] Wydawało mi sie, że powyższa implementacja jest błędna, gdyż wcześniej otrzymywałem 
             * poprzecinane linie. Z drugiej strony znajduje ona tylko PIERWSZY punkt spełniający kryterium, a nie
             * najbliższy. Niestety poniższa próba wyszukiwania najbliższego punktu jest błędna. Może kiedyś zechce
             * mi się ją poprawić..
             */

            // auto bestMatchFrontIt = points.end();
            // auto bestMatchBackIt = points.end();
            
            // for (auto currPointIt = std::next(pointIt); 
            //      currPointIt != points.end();
            //      ++currPointIt)
            // {
                
            //     if (cond(*currPointIt, buildedChain.front(),
            //          (bestMatchFrontIt != points.end()) ? *bestMatchFrontIt : nullptr, dim))
            //     {
            //         bestMatchFrontIt = currPointIt;
            //         continue;
            //     }
            //     else if(cond(*currPointIt, buildedChain.back(),
            //         (bestMatchBackIt != points.end()) ? *bestMatchBackIt : nullptr, dim)) 
            //     {
            //         bestMatchBackIt = currPointIt;
            //     }
            // }

            // if (bestMatchFrontIt != points.end())
            // {
            //     buildedChain.push_front(*bestMatchFrontIt);
            //     points.erase(bestMatchFrontIt);
            //     repeat = true;
            // }
            // if (bestMatchBackIt != points.end() && bestMatchBackIt != bestMatchFrontIt)
            // {
            //     buildedChain.push_back(*bestMatchBackIt);
            //     points.erase(bestMatchBackIt);
            //     repeat = true;
            // }
        }
    }

    return orderedLists;
}

} // end namespace rd
