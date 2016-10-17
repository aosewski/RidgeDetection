/**
 * @file assessment_quality.hpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is conducted under the supervision of prof. dr hab. inż. Marka
 * Nałęcza.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */


#pragma once

#include "rd/cpu/samples_generator.hpp"

#include <vector>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <cmath>

#ifdef RD_USE_OPENMP
#include <omp.h>
#endif

namespace rd
{
    
template <typename T>
class RDAssessmentQuality
{
protected:
    size_t pointCnt;
    size_t dim;

public:

    RDAssessmentQuality(
        size_t dim)
    :
        pointCnt(0),
        dim(dim)
    {}

    RDAssessmentQuality(
        size_t pointCnt,
        size_t dim)
    :
        pointCnt(pointCnt),
        dim(dim)
    {
        idealPattern.resize(pointCnt * dim);
    }

    virtual ~RDAssessmentQuality() {};
    
    void clean() {
        idealPattern.clear();
    }

    virtual void genIdealPattern(
        size_t pointCnt_,
        size_t dim_) 
    {
        pointCnt = pointCnt_;
        dim = dim_;
        clean();
        idealPattern.resize(pointCnt * dim);
    }

    virtual void genIdealPattern(
        size_t dim_) 
    {
        pointCnt = 0;
        dim = dim_;
        clean();
    }

    /**
     * @brief      Calculates Hausdorff metric from idealPattern.
     *
     * @param      testSet  Set for which we want to calculate Hausdorff metric.
     *
     * @return     Hausdorff distance from idealPattern.
     */
    virtual T hausdorffDistance(
        std::vector<T> const & testSet) const 
    {
        if (testSet.empty())
        {
            #ifdef RD_DEBUG
                std::cout << ">>>>>     hausdorffDistance on empty set!!       <<<<<<<" << std::endl;
            #endif
            return -1;
        }

        T distAB = setDistance(testSet, idealPattern);
        T distBA = setDistance(idealPattern, testSet);

        #ifdef RD_DEBUG
        std::cout << ">>>>>>>>> dist(set, ideal): " << distAB << ", dist(ideal, set): " << distBA << "<<<<<<<<<<<\n";
        #endif
        // return std::max(setDistance(testSet, idealPattern),
        //                 setDistance(idealPattern, testSet));
        return std::max(distAB, distBA);
                        
    }

    /**
     * @brief      Calculates range of @p srcSet distance form idealPattern statistics
     *
     * @param      srcSet      Set to examine.
     * @param      medianDist  
     * @param      avgDist     
     * @param      minDist     
     * @param      maxDist     
     */
    void setDistanceStats(
        std::vector<T> const &  srcSet,
        T &                     medianDist,
        T &                     avgDist,
        T &                     minDist,
        T &                     maxDist) const
    {
        std::vector<T>&& distances = calcDistances(srcSet, idealPattern);
        std::sort(distances.begin(), distances.end());
        
        minDist = distances.front();
        maxDist = distances.back();
        medianDist = distances[distances.size() / 2];

        T avg = 0;
        for (T const &d : distances)
        {
            avg += d;
        }
        avgDist = avg /= distances.size();
    }

protected:
    std::vector<T> idealPattern;

    /**
     * @brief      Calculates distance from nearest point from @p set to @srcPoint
     *
     * @param      srcPoint  Pointer to reference point coordinates
     * @param[in]  set       Vector containing points to search.
     *
     * @return     Distance from nearest point from @p set to @srcPoint
     */
    virtual T distance(
        T const *               srcPoint,
        std::vector<T> const &  set) const
    {
        T nearestDist = std::numeric_limits<T>::max();
        T const * data = set.data();

        #ifdef RD_USE_OPENMP
        #pragma omp parallel for schedule(static) reduction(min:nearestDist)
        #endif
        for (size_t k = 0; k < set.size(); k += dim)
        {
            T dist = squareEuclideanDistance(srcPoint, data + k, dim);
            if (dist < nearestDist)
            {
                nearestDist = dist;
            }
        }

        return std::sqrt(nearestDist);
    }

    /**
     * @brief      Calculate distance from each @p setA point to @p setB
     *
     * @param      setA  Point set for which points we calculate distances
     * @param      setB  Point set from which we calculate distances
     *
     * @return     Vector containing distances from each @p setA point to @p setB
     */
    std::vector<T> calcDistances(
        std::vector<T> const & setA,
        std::vector<T> const & setB) const
    {
        std::vector<T> distances;
        T const * srcPoints = setA.data();
        for (size_t k = 0; k < setA.size(); k += dim)
        {
            distances.push_back(distance(srcPoints + k, setB));
        }
        return distances;
    }

    /**
     * @brief      Calculate distance from @p setA to @p setB
     *
     * @param      setA  
     * @param      setB  
     *
     * @return     distance
     */
    T setDistance(
        std::vector<T> const & setA,
        std::vector<T> const & setB) const
    {
        std::vector<T>&& distances = calcDistances(setA, setB);
        std::sort(distances.begin(), distances.end());

        return distances.back();
    }

    /**
     * @brief      Calculate square euclidean distance
     *
     * @param      p1    First point
     * @param      p2    Second point
     *
     * @return     square euclidean distance
     */
    T squareEuclideanDistance(T const * p1, T const * p2, size_t pDim) const
    {
        T dist = 0, t;
        while(pDim-- > 0) {
            t = *(p1++) - *(p2++);
            dist += t*t;
        }
        return dist;
    }

};

template <typename T>
class RDSpiralAssessmentQuality : public RDAssessmentQuality<T>
{
    typedef RDAssessmentQuality<T> BaseT;

public:
    using BaseT::genIdealPattern;

    RDSpiralAssessmentQuality(
        size_t pointCnt,
        size_t dim,
        T length,
        T step) 
    :
        BaseT(pointCnt, dim)
    {
        if (dim < 2 || dim > 3)
        {
            throw std::logic_error("Unsupported spiral dimension !");
        }

        if (dim == 2)
        {
            genSpiral2D(pointCnt, length, step, T(0), this->idealPattern.data());
        }
        else 
        {
            genSpiral3D(pointCnt, length, step, T(0), this->idealPattern.data());
        }
    }

    void genIdealPattern(
        size_t pointCnt,
        size_t dim,
        T length,
        T step)
    {
        BaseT::genIdealPattern(pointCnt, dim);
        if (dim < 2 || dim > 3)
        {
            throw std::logic_error("Unsupported spiral dimension !");
        }

        if (dim == 2)
        {
            genSpiral2D(pointCnt, length, step, T(0), this->idealPattern.data());
        }
        else 
        {
            genSpiral3D(pointCnt, length, step, T(0), this->idealPattern.data());
        }
    }

    virtual ~RDSpiralAssessmentQuality() {}; 
};
  

template <typename T>
class RDSegmentAssessmentQuality : public RDAssessmentQuality<T>
{
    typedef RDAssessmentQuality<T> BaseT;

public:
    using BaseT::genIdealPattern;

    /**
     * The ideal pattern for segment is a segment with all but first coordinates set to zero.
     */

    RDSegmentAssessmentQuality(
    	size_t pointCnt,
        size_t dim)
    :
        BaseT(pointCnt, dim)
    {
    }

    virtual void genIdealPattern(
    	size_t pointCnt,
        size_t dim) override
    {
        BaseT::genIdealPattern(pointCnt, dim);
    }

    virtual ~RDSegmentAssessmentQuality() {}; 

    /**
     * @brief      Calculates Hausdorff metric from idealPattern.
     *
     * @param      testSet  Set for which we want to calculate Hausdorff metric.
     *
     * @return     Hausdorff distance from idealPattern.
     */
    virtual T hausdorffDistance(
        std::vector<T> const & testSet) const
    {
        if (testSet.empty())
        {
            #ifdef RD_DEBUG
                std::cout << ">>>>>     hausdorffDistance on empty set!!       <<<<<<<" << std::endl;
            #endif
            return -1;
        }

        T distAB = this->setDistance(testSet, this->idealPattern);

        #ifdef RD_DEBUG
            std::cout << ">>>>>>>>> dist(set, ideal): " << distAB << "<<<<<<<<<<<\n";
        #endif

        return distAB;
                        
    }

protected:
    /**
     * @brief      Calculates distance from nearest point from @p set to @srcPoint
     *
     * @param      srcPoint  Pointer to reference point coordinates
     * @param[in]  set       Vector containing points to search.
     *
     * @return     Distance from nearest point from @p set to @srcPoint
     */
    virtual T distance(
        T const *               srcPoint,
        std::vector<T> const &  set) const
    {
        set.empty();    // suppress warnings
        /**
         * The ideal pattern for segment is a segment with all but first coordinates set to zero
         */
        T nearestDist = 0;
        for (size_t k = 1; k < this->dim; ++k)
        {
            nearestDist += srcPoint[k] * srcPoint[k];
        }

        return (nearestDist > 0) ? std::sqrt(nearestDist) : 0;
    }
};

}   // end namespace rd
    
