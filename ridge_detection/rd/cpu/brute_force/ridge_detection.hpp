/**
 * @file ridge_detection.hpp
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


#ifndef CPU_BRUTE_FORCE_RIDGE_DETECTION_HPP
#define CPU_BRUTE_FORCE_RIDGE_DETECTION_HPP

#include "rd/utils/utilities.hpp"
#include "rd/utils/graph_drawer.hpp"

#include <cstdlib>
#include <list>
#include <deque>
#include <vector>
#include <utility>

#if defined(RD_USE_OPENMP)
 	#include <omp.h>
#endif

namespace rd 
{

/**
 * @class      RidgeDetection
 * @brief      CPU version of ridge detection algorithm.
 * @note       Capable of using multiple threads with OpenMP.
 *
 * @paragraph Short algorithm description: The algorithm first choose some set of
 * points, wchich are called 'chosen points' or 'sphere centers'. Every two
 * points in this set are R separated, wchis means the distance between them is
 * at least R. After operation of choosing the main loop of the algorithm takes
 * place. It consists of two operations: evolution and decimation. During
 * evolution each 'chosen point' is shifted towards the mass center of all
 * samples wchich falls into the intersection of Voronoi cell (constructed for
 * all 'chosen points') of this point and the sphere with radius R (centered in
 * this point). During decimation excess points are removed from 'chosen set'.
 * Each point which has more than two other points in his 2R-neighbourhood or
 * has less than two other points in his 4R-neighbourhood is discarded. These
 * operations are repeted untill there will be no change in number of chosen
 * points in two consecutive iterations.
 *
 *
 * @tparam     T     Samples data type
 */
template<typename T>
class RidgeDetection {

public:
	/// cardinality of initial choosen points set and final number of chosen points
	size_t ns_;

	/// List with coherent point chains.
	std::list<std::deque<T const*>> chainList_;

	bool verbose_;
	bool order_;

	RidgeDetection();

	/**
	 * @brief - Sets the number of threads to use.
	 * @param t - number of threds to run
	 */
	void ompSetNumThreads(int t);

	void noOMP();

	/**
	 * @brief      Starts ridge detection algorithm.
	 *
	 * @param      P     table containing samples
	 * @param      np    cardinality of points set
	 * @param      S     table for choosen points
	 * @param      r1    Algorithm parameter. Radius used for choosing points
	 *                   and in evolve phase.
	 * @param      r2    Algorithm parameter. Radius used for decimation phase.
	 * @param      dim   points dimension
	 */
	void ridgeDetection(T const *P, size_t np, T *S, T r1, T r2, size_t dim);

	/**
	 * @brief      Starts ridge detection algorithm.
	 *
	 * @param      samples  Vector with tables containing samples.
	 * @param      np       cardinality of points set
	 * @param      S        table for choosen points
	 * @param      r1       Algorithm parameter. Radius used for choosing points
	 *                      and in evolve phase.
	 * @param      r2       Algorithm parameter. Radius used for decimation phase.
	 * @param      dim      points dimension
	 */
	void ridgeDetection(std::vector<std::pair<T const *, size_t>> samples,
		 T *S, T r1, T r2, size_t dim);

protected:

	/// number of threads used to parallelize calculations
	int ompNumThreads_;
	// wheather or not to use omp multithreaded version
	bool noOMP_;

private:
	RidgeDetection(RidgeDetection<T> const &rd) {}
	void operator=(RidgeDetection<T> const &rd) {}
};

}  // namespace ridge_detection


#include "ridge_detection.inl"


#endif /* CPU_BRUTE_FORCE_RIDGE_DETECTION_HPP */
