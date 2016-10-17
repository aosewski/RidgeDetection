/**
 * @file rd_inner.hpp
 * @author: Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering, Faculty of Electronics and Information
 * Technology, Warsaw University of Technology 2016
 */

#ifndef RD_INNER_HPP
#define RD_INNER_HPP

#include <list>

namespace rd
{
namespace cpu
{
namespace brute_force
{

////////////////////////////////////////////////////////////////////////////
//
// Inner ridge detection algorihtm routines
//
////////////////////////////////////////////////////////////////////////////


/********************************************************************************/

/**
 * @brief      Calculates square euclidean distance between two points @p p1 and @p p2
 *
 * @param      p1    - first point
 * @param      p2    - second point
 * @param      dim   - point's dimension
 *
 * @tparam     T     Point coordinate data type
 *
 * @return     distance
 *
 * @note       Assumes row-major data order
 */
template <typename T>
static T squareEuclideanDistance(T const * p1, T const * p2, size_t dim) {
    T dist = 0, t;
    while(dim-- > 0) {
        t = *(p1++) - *(p2++);
        dist += t*t;
    }
    return dist;
}


/********************************************************************************/

/**
 * @brief      checks whether point @p src_p has at least @p threshold neighbours within @p
 *             sqrt(rSqr) distance
 *
 * @param      points     - points set
 * @param      np         - number of points
 * @param      src_p      - source point
 * @param      dim        - point's dimension
 * @param      rSqr       - search distance square radius
 * @param      threshold  - number of points to search
 *
 * @tparam     T          Point coordinate data type
 *
 * @return     True if found at least threshold points, else false
 */
template <typename T>
static bool countNeighbouringPoints(
    T const * points,
    size_t    np,
    T const * src_p,
    size_t    dim,
    T         rSqr,
    int       threshold) 
{
    T const * p = points;
    for (size_t i = 0; i < np; ++i) 
    {
        if (squareEuclideanDistance(src_p, p, dim) <= rSqr) 
        {
            if (--threshold <= 0)
                return true;
        }
        p += dim;
    }
    return false;
}

template <typename T>
static bool countNeighbouringPoints(
    T * const * points,
    size_t      np,
    T * const * src_p,
    size_t      dim,
    T           rSqr,
    int         threshold) 
{
    T * const * p = points;
    for (size_t i = 0; i < np; ++i) 
    {
        if (squareEuclideanDistance(*src_p, *p, dim) <= rSqr) 
        {
            if (--threshold <= 0)
            {
                return true;
            }
        }
        p++;
    }
    return false;
}

template <typename T>
static bool countNeighbouringPoints(
    std::list<T*> const &   points,
    T const *               src_p,
    size_t                  dim, 
    T                       rSqr, 
    int                     threshold) 
{
    for (auto it = points.cbegin();
         it != points.cend();
         it++) 
    {
        if (squareEuclideanDistance(src_p, *it, dim) <= rSqr) 
        {
            if (--threshold <= 0)
            {
                return true;
            }
        }
    }
    return false;
}

template <typename T>
static bool countNeighbouringPoints(
    std::list<std::list<T*>*> const &   points,
    T const *               src_p,
    size_t                  dim, 
    T                       rSqr, 
    int                     threshold) 
{
    for (auto lit = points.cbegin(); lit != points.cend(); lit++)     
    {
        for (auto it = (*lit)->cbegin(); it != (*lit)->cend(); it++) 
        {
            if (squareEuclideanDistance(src_p, *it, dim) <= rSqr) 
            {
                if (--threshold <= 0)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

} // end namespace brute_force
} // end namespace cpu
} // end namespace rd
  
#endif  // RD_INNER_HPP