/** 
 * @file ridge_detection.inl
 * @author Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering, Faculty of Electronics and Information
 * Technology, Warsaw University of Technology 2016
 */

#include "rd/cpu/brute_force/choose.hpp"
#include "rd/cpu/brute_force/evolve.hpp"
#include "rd/cpu/brute_force/decimate.hpp"
#include "rd/cpu/brute_force/order.hpp"

#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <typeinfo>
#include <cstring>

namespace rd
{
namespace cpu
{
namespace brute_force
{ 

template <typename T>
RidgeDetection<T>::RidgeDetection() 
{
    ompNumThreads_ = 0;
    verbose_ = false;
    order_ = false;
    noOMP_ = false;
    #if defined(RD_USE_OPENMP)
        ompNumThreads_ = omp_get_num_procs();
        omp_set_num_threads(ompNumThreads_);
        #ifdef RD_DEBUG
        std::cout << "Default using " << ompNumThreads_ << " CPU threads" << std::endl;
        #endif
    #endif
}

template <typename T>
void RidgeDetection<T>::ompSetNumThreads(int t)
{
    ompNumThreads_ = t;
    #if defined(RD_USE_OPENMP)
    omp_set_num_threads(t);
        #ifdef RD_DEBUG
        std::cout << "Set to use " << t << " CPU threads" << "\n";
        #endif
    #endif
}


template <typename T>
void RidgeDetection<T>::noOMP()
{
    ompNumThreads_ = 0;
    noOMP_ = true;
}

template <typename T>
void RidgeDetection<T>::ridgeDetection(T const *P, size_t np, T *S, T r1, T r2, size_t dim) 
{
    // List containing chosen points
    std::list<T*> SList_;

    choose(P, S, SList_, np, ns_, dim, r1);
    if (verbose_ && dim <= 3) 
    {
        rd::GraphDrawer<T> gDrawer;
        std::cout << "Wybrano reprezentantów, liczność zbioru S: " << ns_ << std::endl;
        std::cout << HLINE << std::endl;

        std::ostringstream os;
        os << typeid(T).name();
        os << "_";
        os << dim;
        os << "D_cpu_choosen_set";
        gDrawer.showPoints(os.str(), S, ns_, dim);
    }

    size_t oldCount = 0;
    int iter = 0;

    /*
     * Iterate until points count won't change 
     */
    while (oldCount != ns_) 
    {
        oldCount = ns_;

        if (noOMP_)
        {
            evolve(P, S, np, ns_, dim, r1, ompNumThreads_, verbose_);
        }
        else
        {
            evolve_omp(P, S, np, ns_, dim, r1, ompNumThreads_, verbose_);
        }
        if (verbose_ && dim <= 3) 
        {
            rd::GraphDrawer<T> gDrawer;
            std::cout << "Ewolucja nr: " << iter << std::endl;

            std::ostringstream os;
            os << typeid(T).name();
            os << "_cpu_";
            os << dim;
            os << "D_iter_";
            os << iter;
            os << "_a_evolution";
            if (np > 10000) {
                gDrawer.showCircles(os.str(), 0, np, S, ns_, r1, dim);
            } else {
                gDrawer.showCircles(os.str(), P, np, S, ns_, r1, dim);
            }
        }

        decimate(S, SList_, ns_, dim, r2);
        if (verbose_ && dim <= 3)
        {
            rd::GraphDrawer<T> gDrawer;
            std::cout << "Decymacja nr: " << iter << ", ns: " << ns_ << std::endl;

            std::ostringstream os;
            os << typeid(T).name();
            os << "_cpu_";
            os << dim;
            os << "D_iter_";
            os << iter;
            os << "_decimation";
            if (np > 10000) {
                gDrawer.showCircles(os.str(), 0, np, S, ns_, r2, dim);
            } else {
                gDrawer.showCircles(os.str(), P, np, S, ns_, r2, dim);
            }
        }
        iter++;
    }

    if (order_)
    {
        T r2Sqr = r2 * r2;
        chainList_ = orderSamples<T>(S, dim, ns_, [&r2Sqr](
            T const *p1,
            T const *p2,
            // T const *pBest,
            size_t dim) -> bool
        {
            // T dist = squareEuclideanDistance(p1, p2, dim);
            // if (pBest == nullptr)
            // {
            //     return dist <= r2Sqr;
            // }
            // else 
            // {
            //     return dist < squareEuclideanDistance(p1, pBest, dim);
            // }
            return squareEuclideanDistance(p1, p2, dim) <= r2Sqr;
        });
    }
}

template <typename T>
void RidgeDetection<T>::ridgeDetection(
    std::vector<std::pair<T const *, size_t>> samples, T *S, T r1, T r2, size_t dim)
{
    // List containing chosen points
    std::list<T*> SList_;

    choose(samples[0].first, S, SList_, samples[0].second, ns_, dim, r1);
    if (verbose_ && dim <= 3) 
    {
        std::cout << "Wybrano reprezentantów, liczność zbioru S: " << ns_ << std::endl;
        std::cout << HLINE << std::endl;

        std::ostringstream os;
        os << typeid(T).name();
        os << "_";
        os << dim;
        os << "D_cpu_choosen_set";

        rd::GraphDrawer<T> gDrawer;
        gDrawer.startGraph(os.str(), dim);
        // background all points including neighbours
        for (auto p : samples)
        {
            gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
                p.first, rd::GraphDrawer<T>::POINTS, p.second);
        }

        // draw this tile chosen samples 
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#000099' ps 1.3 ",
             S, rd::GraphDrawer<T>::POINTS, ns_);
        gDrawer.endGraph();
    }

    size_t oldCount = 0;
    int iter = 0;

    /*
     * Iterate until points count won't change in two consecutive
     * iterations.
     */
    while (oldCount != ns_) 
    {
        oldCount = ns_;

        if (noOMP_)
        {
            evolve_neighbours(samples, S, ns_, dim, r1, ompNumThreads_, verbose_);
        }
        else 
        {
            evolve_neighbours_omp(samples, S, ns_, dim, r1, ompNumThreads_, verbose_);
        }
        if (verbose_ && dim <= 3) 
        {
            rd::GraphDrawer<T> gDrawer;
            std::cout << "Ewolucja nr: " << iter << std::endl;

            std::ostringstream os;
            os << typeid(T).name();
            os << "_cpu_";
            os << dim;
            os << "D_iter_";
            os << iter;
            os << "_a_evolution";
            if (samples[0].second > 10000) {
                gDrawer.showCircles(os.str(), 0, samples[0].second, S, ns_, r1, dim);
            } else {
                gDrawer.showCircles(os.str(), samples[0].first, samples[0].second, S, ns_, r1, dim);
            }
        }

        decimate(S, SList_, ns_, dim, r2);
        if (verbose_ && dim <= 3)
        {
            rd::GraphDrawer<T> gDrawer;
            std::cout << "Decymacja nr: " << iter << ", ns: " << ns_ << std::endl;

            std::ostringstream os;
            os << typeid(T).name();
            os << "_cpu_";
            os << dim;
            os << "D_iter_";
            os << iter;
            os << "_decimation";
            if (samples[0].second > 10000) {
                gDrawer.showCircles(os.str(), 0, samples[0].second, S, ns_, r2, dim);
            } else {
                gDrawer.showCircles(os.str(), samples[0].first, samples[0].second, S, ns_, r2, dim);
            }
        }
        iter++;
    }

    if (order_)
    {
        T r2Sqr = r2 * r2;
        chainList_ = orderSamples<T>(S, dim, ns_, [&r2Sqr](
            T const *p1,
            T const *p2,
            // T const *pBest,
            size_t dim) -> bool
        {
            // T dist = squareEuclideanDistance(p1, p2, dim);
            // if (pBest == nullptr)
            // {
            //     return dist <= r2Sqr;
            // }
            // else 
            // {
            //     return dist < squareEuclideanDistance(p1, pBest, dim);
            // }
            return squareEuclideanDistance(p1, p2, dim) <= r2Sqr;
        });
    }
}

} // end namespace brute_force
} // end namespace cpu
} // end namespace rd
