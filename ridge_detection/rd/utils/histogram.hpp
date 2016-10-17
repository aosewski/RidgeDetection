/**
 * @file histogram.hpp
 * @author Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of 
 *  estimation of multidimensional random variable density function ridge
 *  detection algorithm.",
 * which is supervised by prof. dr hab. inż. Marek Nałęcz.
 * 
 * Institute of Control and Computation Engineering
 * Faculty of Electronics and Information Technology
 * Warsaw University of Technology 2016
 */

#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <vector>
#include <algorithm>
#include <functional>
#include <cstddef>

#include <iostream>
#include <stdexcept>

namespace rd
{

/**
 * @brief      Class representing histogram of given data samples.
 *
 * @tparam     T     Samples data type.
 */
template <typename T>
struct Histogram
{
    size_t dim;
    std::vector<size_t> binCnt;
    std::vector<size_t> hist;

    Histogram()
    {
        dim = 0;
    }

    Histogram(size_t d) : dim(d)
    {
        binCnt.resize(d);
    }

    /**
     * @brief      Set number of bins in each dimension
     *
     * @param      bc    Vector containing number of bins for respective 
     *                   dimensions
     */
    void setBinCnt(std::vector<size_t> const &bc)
    {
        dim = bc.size();
        binCnt.clear();
        binCnt.insert(binCnt.begin(), bc.begin(), bc.end());
        hist.resize(std::accumulate(binCnt.begin(), binCnt.end(),
                         1, std::multiplies<size_t>()));
    }

    /**
     * @brief      Runs histogram calculation.
     *
     * @see      getHist
     */
    template <typename MapFunc>
    void operator()(T const * data, size_t n, MapFunc f)
    {
        getHist(data, n, f);
    }

    /**
     * @brief      Runs histogram calculation.
     *
     * @param      data     Pointer to samples.
     * @param[in]  n        Number of points
     * @param[in]  f        Function mapping sample values to histogram bin idx.
     *
     * @tparam     MapFunc  Functor class for mapping values to histogram bins.
     */
    template <typename MapFunc>
    void getHist(T const * data, size_t n, MapFunc f)
    {
        for (size_t k = 0; k < n; ++k)
        {
        	size_t idx = f(data, binCnt);
            #ifdef RD_DEBUG
                if (idx >= hist.size())
                {
                    std::cout << "ERROR!!! idx: " << idx << ", hist.size(): " << hist.size() << std::endl << std::flush;
                    std::string errstring = std::string("Bad idx value! at: ") +  __FILE__ + std::string("(")  + std::to_string(__LINE__) + std::string(")");
                    throw std::logic_error(errstring.c_str());
                }
            #endif
            data += dim;
            hist[idx]++;
        }
    }

    size_t operator[](size_t idx) const
    {
        return hist[idx];
    }

    void print() const 
    {
        std::cout << "Histogram: \n [";
        for (size_t k = 0; k < hist.size(); ++k)
        {
            std::cout << hist[k] << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Hist's bin cnt: \n [";
        for (size_t k = 0; k < binCnt.size(); ++k)
        {
            std::cout << binCnt[k] << ", ";
        }
        std::cout << "]" << std::endl;
    }
};

} // namespace rd


#endif // HISTOGRAM_HPP
