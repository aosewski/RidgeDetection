/**
 * @file bounding_box.hpp
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


#ifndef BOUNDING_BOX_HPP
#define BOUNDING_BOX_HPP 
 
#include <stdexcept>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <limits>

#ifdef RD_USE_OPENMP
#include <omp.h>
#endif

namespace rd
{

/**
 * @brief      Represents bounding box of multidimensional samples set.
 *
 * @tparam     T     Data type.
 */
template <typename T>
struct BoundingBox
{

	T *bbox;
	size_t dim;
	/// Distance between min and max
	T *dist;

	BoundingBox() : bbox(nullptr), dim(0), dist(nullptr)
	{
	};


	BoundingBox(T const * const d, size_t n, size_t aDim) 
	: 
		bbox(nullptr), dim(0), dist(nullptr)
	{
		findBounds(d, n, aDim);
	}

	BoundingBox(BoundingBox const & rhs)
	{
		clone(rhs);
	}

	BoundingBox(BoundingBox && rhs)
	:
		bbox(rhs.bbox),
		dim(rhs.dim),
		dist(rhs.dist)
	{
		rhs.bbox = nullptr;
		rhs.dist = nullptr;
		rhs.dim = 0;
	}

	~BoundingBox()
	{
		if (bbox != nullptr)
		{
			delete[] bbox;
		}
		if (dist != nullptr)
		{
			delete[] dist;
		}
	}

	BoundingBox & operator=(BoundingBox const &rhs)
	{
		clone(rhs);
		return *this;
	}

	BoundingBox & operator=(BoundingBox && rhs)
	{
		if (bbox != nullptr)
		{
			delete[] bbox;
		}
		if (dist != nullptr)
		{
			delete[] dist;
		}

		bbox = rhs.bbox;
		dist = rhs.dist;
		dim = rhs.dim;

		rhs.bbox = nullptr;
		rhs.dist = nullptr;
		rhs.dim = 0;
		
		return *this;
	}

	/**
	 * @brief      Finds the bounding box (min,max values) for the given @p d
	 *             data set.
	 * @note       Data must be in row-major order.
	 *
	 * @param      d     [in] - input data set
	 * @param      n     [in] - number of points in @p d data set
	 * @param      aDim  [in] - dimension of points
	 *
	 * @note       Within bbox data have following layout:
	 *             [min_1,max_1,...,min_n,max_n]
	 */
	void findBounds(T const * const d, size_t n, size_t aDim)
	{
		if (dim != aDim)
		{
			if (bbox != nullptr)
			{
				delete[] bbox;
			}
			bbox = new T[2 * aDim];
		}
		if (bbox == nullptr)
		{
			bbox = new T[2 * aDim];
		}
		dim = aDim;


		//init bbox
		for (size_t k = 0; k < dim; ++k)
		{
			bbox[2*k] = std::numeric_limits<T>::max();
			bbox[2*k + 1] = std::numeric_limits<T>::lowest();
		}

	    #if defined(RD_USE_OPENMP)
	    {
	    int threadsNum = omp_get_num_procs();
    	T * privMin;
    	T * privMax;
	    #pragma omp parallel num_threads(threadsNum) shared(n) private(privMin, privMax)
	    {
			//init private bounds
			privMin = new T[dim];
			privMax = new T[dim];
			for (size_t k = 0; k < dim; ++k)
			{
				privMin[k] = std::numeric_limits<T>::max();
				privMax[k] = std::numeric_limits<T>::lowest();
			}

		    #pragma omp for schedule(static) 
		    for (size_t k = 0; k < n; ++k)
			{
				for (size_t i = 0; i < dim; ++i)
				{
					T val = d[dim*k + i];
					privMin[i] = (val < privMin[i]) ? val : privMin[i];
					privMax[i] = (val > privMax[i]) ? val : privMax[i];
				}
			}
			// reduce private results
			#pragma omp critical
			{
				for (size_t i = 0; i < dim; ++i)
				{
					T val = min(i);
					min(i) = (val < privMin[i]) ? val : privMin[i];
					val = max(i);
					max(i) = (val > privMax[i]) ? val : privMax[i];
				}
			}
			delete[] privMin;
			delete[] privMax;
		}	// end omp parallel
		}		
	    #else // defined(RD_USE_OPENMP)
		for (size_t k = 0; k < n; ++k)
		{
			for (size_t i = 0; i < dim; ++i)
			{
				T val = d[dim*k + i];
				min(i) = (val < min(i)) ? val : min(i);
				max(i) = (val > max(i)) ? val : max(i);
			}
		}
	    #endif
	}


	/**
	 * @brief      Calculates distance between min and max value in each
	 *             dimension.
	 */
	void calcDistances()
	{
		if (bbox == nullptr)
			return;

		if (dist == nullptr)
			dist = new T[dim];

		// calc distances
		for (size_t k = 0; k < dim; ++k)
		{
			dist[k] = std::abs(bbox[2*k+1] - bbox[2*k]);
			if (dist[k] <= std::numeric_limits<T>::epsilon())
			{
				dist[k] = 0;
			}
		}
	}

	/**
	 * @brief      Returns lower bound for @p idx dimension
	 *
	 * @param[in]  idx   Requested dimension.
	 *
	 * @return     Const reference to bound value.
	 */
	T const & min(const size_t idx) const
	{
		if (idx < dim)
			return bbox[2*idx];
		else 
			throw std::out_of_range("Requested idx exceeds" 
				" bounding box dimensions!");
	}

	/**
	 * @brief      Returns lower bound for @p idx dimension.
	 *
	 * @param[in]  idx   Requested dimension.
	 *
	 * @return     Reference to bound value.
	 */
	T & min(const size_t idx)
	{
		if (idx < dim)
			return bbox[2*idx];
		else 
			throw std::out_of_range("Requested idx exceeds" 
				" bounding box dimensions!");
	}

	/**
	 * @brief      Returns upper bound for @p idx dimension.
	 *
	 * @param[in]  idx   Requested dimension.
	 *
	 * @return     Const reference to bound value.
	 */
	T const & max(const size_t idx) const
	{
		if (idx < dim)
			return bbox[2*idx + 1];
		else 
			throw std::out_of_range("Requested idx exceeds" 
				" bounding box dimensions!");
	}

	/**
	 * @brief      Returns upper bound for @p idx dimension.
	 *             
	 * @param[in]  idx   Requested dimension.
	 *
	 * @return     Const reference to bound value.
	 */
	T & max(const size_t idx)
	{
		if (idx < dim)
			return bbox[2*idx + 1];
		else 
			throw std::out_of_range("Requested idx exceeds" 
				" bounding box dimensions!");
	}

	void print() const
	{
		std::cout << "bounds: " << std::endl;
		for (size_t d = 0; d < dim; ++d)
		{
			std::cout << "dim: " << d << ", min: " << bbox[2*d] <<
				", max: " << bbox[2*d+1] << "\n";
		}
		std::cout  << "distances: [";
		for (size_t d = 0; d < dim; ++d)
			std::cout << dist[d] << ", ";
		std::cout << "]\n";
	}

	/**
	 * @brief      Counts how many spheres fits inside region described by @p bb
	 *
	 * @param[in]  radius  Sphere radius.
	 *
	 * @return     the number of spheres that fits in @p bb region.
	 */
	inline size_t countSpheresInside(T radius)
	{
	    size_t cnt = 1;
	    for (size_t d = 0; d < dim; ++d)
	    {
	    	size_t dcnt= static_cast<size_t>(std::ceil(dist[d] / radius));
	    	cnt *= (dcnt) ? dcnt + 1 : 1;
	    }

	    return cnt;
	}

	/**
	 * @brief      Checks whether @p point lies inside bounds
	 *
	 * @note       Inside bounds is meant to be: min <= x < max
	 *
	 * @param      point  Pointer to point coordinates.
	 *
	 * @return     True if @p point is inside this bounds
	 */
	inline bool isInside(T const *point)
	{
		for (size_t d = 0; d < dim; ++d)
		{
			if (point[d] < min(d) || point[d] >= max(d))
				return false;
		}

		return true;
	}


	/**
	 * @brief      Chekcs wheather @p point lies inside extended bounding box.
	 *
	 *             Precisely it checks wheather @p point lies in area of (ext
	 *             bounding box) minus area of this (bounding box). It is a
	 *             neighbourhood of this bounding box.
	 *
	 * @param      point         Pointer to point coordinates.
	 * @param[in]  extendRadius  Amount by which we enlarge this bounding box
	 *                           area for searching neighbours.
	 *
	 * @return     True if @p point lies nearby this bonding box.
	 */
	inline bool isNearby(T const *point, T extendRadius)
	{
		for (size_t d = 0; d < dim; ++d)
		{
			// is outside ext bbox
			if ((point[d] < min(d) - extendRadius) || (point[d] >= max(d) + extendRadius))
				return false;
		}

		if (!isInside(point))
		{
			return true;
		}

		return false;
	}

	/**
	 * @brief      Checks whether @p other overlaps with this bounding box
	 *
	 * @param      other  Examined bounding box
	 *
	 * @return     True if two bounding boxes overlap
	 */
	inline bool overlap(BoundingBox<T> const &other)
	{

		if (dim != other.dim)
			return false;

		bool result;

		// there must be overlap in all dimensions
		for (size_t d = 0; d < dim; ++d)
		{
			/*	
				// check current dim center distance
			  	// XXX: this code doesn't work properly - it doesn't handle all cases, 
			  	// however leave it for future - maybe idea will be useful.
			 
			 	T m1, m2, half1, half2;
				m1 = (min(d) + max(d)) / 2;
				m2 = (other.min(d) + other.max(d)) / 2;
				half1 = dist[d] / 2;
				half2 = other.dist[d] / 2;
				result = (half1 + half2) - std::abs(m1 - m2) >= 0;
			*/
			// if two segments overlap? 
			result = false;
			result = result || (other.min(d) >= min(d) && other.min(d) <= max(d));
			result = result || (other.max(d) >= min(d) && other.max(d) <= max(d));
			result = result || (min(d) >= other.min(d) && min(d) <= other.max(d));
			result = result || (max(d) >= other.min(d) && max(d) <= other.max(d));
			if (!result)
				return false;
		}

		return true;
	}

private:

	void clone(BoundingBox<T> const & rhs)
	{
		if (bbox != nullptr)
		{
			delete[] bbox;
		}
		if (dist != nullptr)
		{
			delete[] dist;
		}

		dim = rhs.dim;

		bbox = new T[2 * dim];
		dist = new T[dim];

		for (int i = 0; i < dim; ++i)
		{
			bbox[2*i] = rhs.bbox[2*i];
			bbox[2*i + 1] = rhs.bbox[2*i + 1];
			dist[i] = rhs.dist[i];
		}
	}

};

} // end namespace rd

#endif	// BOUNDING_BOX_HPP
