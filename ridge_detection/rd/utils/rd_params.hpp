/**
 *  @file rd_params.hpp
 *  @author: Adam Rogowiec
 * 
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering, Faculty of Electronics and Information
 * Technology, Warsaw University of Technology 2016
 */

#ifndef RD_PARAMS_HPP_
#define RD_PARAMS_HPP_

#include <cstdlib>
#include <string>
#include <vector>

namespace  rd 
{

/**
 * @brief      Ridge Detection algorithm parameters.
 *
 * @tparam     T     Point coordinates data type.
 */
template <typename T>
struct RDParams {
	/// samples dimension
	size_t dim;
	/// number of samples in initial set
	size_t np;
	/// number of chosen samples for main part of algorithm
	size_t ns;
	/// the radius used for choosing samples
	T r1;
	/// the radius used in decimation phase
	T r2;
	/// flag for verbose output
	bool verbose;
	/// simulation version name
	std::string name;
	int version;
	/// (GPU) device Id to use
	int devId;
	/// whether to order samples in the end of algorithm.
	bool order;

	RDParams() 
	: 
		dim(0), np(0), ns(0), r1(T(0)), r2(T(0)),
		verbose(false), name(""), version(0), devId(0),
		order(false) 
	{
	}
};

/**
 * @brief      Ridge detection tiled version algorithm parameters.
 *
 * @tparam     T     Point coordinates data type.
 */
template <typename T>
struct RDTiledParams : public RDParams<T>
{
	/// maximum number of samples which tile can contain
	size_t maxTileCapacity;
	/// initial number of tiles in respective dimensions
	std::vector<size_t> nTilesPerDim;
	/// tile extension factor
	T extTileFactor;
	/// whether to perform global refinement at the end of detection
	bool endPhaseRefinement;

	RDTiledParams() 
	: 
		RDParams<T>(),
		maxTileCapacity(0),
		nTilesPerDim(),
		extTileFactor(0),
		endPhaseRefinement(false)
	{
	}
};


/**
 * @brief      Helper structure holding parameters for generating spirals
 *
 * @tparam     T     Point coordinates data type.
 */
template <typename T>
struct RDSpiralParams {
	/// spiral param a (determines spiral length)
	T a;
	/// spiral param b (determines spiral distance between rolls)
	T b;
	/// determines amount of noise we add to points
	T sigma;
	/// flag indicating whether load samples from file or generate them
	bool loadFromFile;
	/// input samples file
	std::string file;

	RDSpiralParams() 
	: 
		a(T(0)), b(T(0)), sigma(T(0)), 
		loadFromFile(false), file(""), files()
	{
	}
};

}  // end namespace rd

#endif /* RD_PARAMS_HPP_ */
