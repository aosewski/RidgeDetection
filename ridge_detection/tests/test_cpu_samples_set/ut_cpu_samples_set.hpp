/**
 * @file ut_cpu_samples_set.hpp
 * @date 30-03-2015
 * @author Adam Rogowiec
 */

#ifndef UT_CPU_SAMPLES_SET_HPP_
#define UT_CPU_SAMPLES_SET_HPP_

#include "rd/utils/graph_drawer.hpp"

#include "rd/utils/utilities.hpp"
#include "rd/utils/samples_set.hpp"

#include <iostream>
#include <typeinfo>
#include <sstream>

template <typename T>
class CpuSamplesSetUnitTests {

public:

	size_t n_;
	T a_;
	T b_;
	T sigma_;

	CpuSamplesSetUnitTests(size_t n, T a, T b, T sigma) :
		n_(n), a_(a), b_(b), sigma_(sigma) {

	}

	virtual ~CpuSamplesSetUnitTests() {}

	void testGenSpiral2D();
	void testGenSpiral3D();
	void testGenSegmentND(size_t dim);
	
private:
	rd::GraphDrawer<T> gDrawer_;
};

template<typename T>
void CpuSamplesSetUnitTests<T>::testGenSpiral2D() {

	std::cout << HLINE << std::endl;
	std::cout << "testGenSpiral2D: " << std::endl;

	rd::Samples<T> samples;
	T *spiral = samples.genSpiral2D(n_, a_, b_, sigma_);

	std::ostringstream ss;
	ss << typeid(T).name() << "_test_cpu_spiral_points2D";
	gDrawer_.showPoints(ss.str(), spiral, n_, 2);
}

template<typename T>
void CpuSamplesSetUnitTests<T>::testGenSpiral3D() {

	std::cout << HLINE << std::endl;
	std::cout << "testGenSpiral3D: " << std::endl;

	rd::Samples<T> samples;
	T *spiral = samples.genSpiral3D(n_, a_, b_, sigma_);

	std::ostringstream ss;
	ss << typeid(T).name() << "_test_cpu_spiral_points3D";
	gDrawer_.showPoints(ss.str(), spiral, n_, 3);
}

template<typename T>
void CpuSamplesSetUnitTests<T>::testGenSegmentND(size_t dim) {

	std::cout << HLINE << std::endl;
	std::cout << "testGenSegmentND: " << std::endl;

	rd::Samples<T> samples;
	T *spiral = samples.genSegmentND(n_, dim, sigma_);

	// XXX: weryfikacja?
	if(!checkValues(spiral, n_ * dim)) {
		std::cout << "Some values are almost for sure INCORRECT!" << std::endl;
	}
}

#endif /* UT_CPU_SAMPLES_SET_HPP_ */
