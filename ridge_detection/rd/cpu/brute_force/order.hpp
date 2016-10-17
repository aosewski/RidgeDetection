/**
 * @file order.hpp
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

#pragma once

#include <cstddef>
#include <deque>
#include <list>
#include <iterator>

namespace rd
{
namespace cpu
{
namespace brute_force
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
             * XXX: [17.05.16] It seemed to me that above order implementation gives wrong results,
             * because during earlier tests i got cut through segments. On the other side above 
             * impl. finds ONLY FIRST point satisfing criteria, while it should find the closest 
             * one. Unfortunately following attempt to correct this behaviour ended with failure.
             * Maybe I will fix it some day.
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

} // end namespace brute_force
} // end namespace cpu
} // end namespace rd
