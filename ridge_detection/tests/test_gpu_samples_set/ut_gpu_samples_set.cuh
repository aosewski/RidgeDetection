/**
 * @file ut_gpu_samples_set.cuh
 * @date 30-03-2015
 * @author Adam Rogowiec
 */

#ifndef UT_GPU_SAMPLES_SET_CUH_
#define UT_GPU_SAMPLES_SET_CUH_

#include "rd/utils/graph_drawer.hpp"

#include "rd/utils/utilities.hpp"
#include "rd/gpu/device/samples_generator.cuh"

#include <iostream>


template <typename T>
class GpuSamplesSetUnitTests {

public:

	size_t n_;
	T a_;
	T b_;
	T sigma_;

	GpuSamplesSetUnitTests(size_t n, T a, T b, T sigma) :
		n_(n), a_(a), b_(b), sigma_(sigma) {
	}

	virtual ~GpuSamplesSetUnitTests() {}

	void testGenSpiral2D();
	void testGenSpiral3D();
	void testGenSegmentND(size_t dim);

private:
	rd::GraphDrawer<T> gDrawer_;
};

template<typename T>
void GpuSamplesSetUnitTests<T>::testGenSpiral2D() {

	std::cout << rd::HLINE << std::endl;
	std::cout << "testGpuGenSpiral2D: " << std::endl;

	T *dSamples, *gpu_spiral;
	gpu_spiral = new T[n_ * 2];

	checkCudaErrors(cudaMalloc((void**)&dSamples, n_ * 2 *sizeof(T)));
	rd::gpu::SamplesGenerator<T>::spiral2D(n_, a_, b_, sigma_, dSamples);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(gpu_spiral, dSamples, n_ * 2 * sizeof(T),
		cudaMemcpyDeviceToHost));

#ifdef DEBUG
	printTable(gpu_spiral, n_ * 2, "gpuSpiral2D", 2);
#endif

	gDrawer_.showPoints("test_gpu_spiral_points2D", gpu_spiral, n_, 2);

	delete[] gpu_spiral;
	checkCudaErrors(cudaFree(dSamples));
}

template<typename T>
void GpuSamplesSetUnitTests<T>::testGenSpiral3D() {

	std::cout << rd::HLINE << std::endl;
	std::cout << "testGpuGenSpiral3D: " << std::endl;

	T *dSamples, *gpu_spiral;
	gpu_spiral = new T[n_ * 3];

	checkCudaErrors(cudaMalloc((void**)&dSamples, n_ * 3 *sizeof(T)));
	rd::gpu::SamplesGenerator<T>::spiral3D(n_, a_, b_, sigma_, dSamples);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(gpu_spiral, dSamples, n_ * 3 * sizeof(T),
		cudaMemcpyDeviceToHost));

#ifdef DEBUG
	printTable(gpu_spiral, n_ * 3, "gpuSpiral3D", 3);
#endif
	gDrawer_.showPoints("test_gpu_spiral_points3D", gpu_spiral, n_, 3);

	delete[] gpu_spiral;
	checkCudaErrors(cudaFree(dSamples));
}

template<typename T>
void GpuSamplesSetUnitTests<T>::testGenSegmentND(size_t dim) {

	std::cout << rd::HLINE << std::endl;
	std::cout << "testGpuGenSegmentND: " << std::endl;

	T *dSamples, *gpu_spiral;
	gpu_spiral = new T[n_ * dim];

	checkCudaErrors(cudaMalloc((void**)&dSamples, n_ * dim * sizeof(T)));
	rd::gpu::SamplesGenerator<T>::segmentND(n_, dim, sigma_, T(1), dSamples);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(gpu_spiral, dSamples, n_ * dim * sizeof(T),
		cudaMemcpyDeviceToHost));

	// XXX: weryfikacja?
#ifdef DEBUG
	printTable(gpu_spiral, n_ * dim, "gpuSegmentND: ", dim);
#endif
	if(!rd::checkValues(gpu_spiral, n_ * dim)) {
		std::cout << "Some values are almost for sure INCORRECT!" << std::endl;
	}

	delete[] gpu_spiral;
	checkCudaErrors(cudaFree(dSamples));
}

#endif /* UT_GPU_SAMPLES_SET_CUH_ */
