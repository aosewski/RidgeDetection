/**
 * @file rd_dp.cuh
 * @author Adam Rogowiec
 * 
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and Information
 * Technology Warsaw University of Technology 2016
 */

#ifndef RD_DP_BRUTE_FORCE_CUH_
#define RD_DP_BRUTE_FORCE_CUH_

#include "cub/util_debug.cuh"

#include "rd/gpu/util/dev_utilities.cuh"

#include "rd/gpu/device/device_decimate.cuh"
#include "rd/gpu/device/device_choose.cuh"
#include "rd/gpu/device/device_evolve.cuh"
#include "rd/gpu/device/brute_force/rd_globals.cuh"

namespace rd
{
namespace gpu
{
namespace bruteForce
{

template <
    typename 			T,
    int 				DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT>
static __global__ void __rd_simulation_dp(
		T const *	P,
		T *			S,
		T *			cordSums,
		int * 		spherePointCount,
		T * 		distMtx,
		char * 		ptsMask,
		T 			r1,
		T 			r2,
		int 		np,
		int 		pStride,
		int 		sStride,
		int 		csStride,
		int 		distMtxStride) 
{

	cudaError_t error = cudaSuccess;

	/***********************************************************
	 * 					START ALGORITHM
	 ***********************************************************/
	int ns = rdBruteForceNs;
	int oldCount = 0;

	/*
	 * 	Repeat untill the count of chosen samples won't
	 * 	change in two consecutive iterations.
	 */
	while (oldCount != ns) 
	{
		oldCount = ns;
	    error = DeviceEvolve::evolve<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
	        P,
	        S,
	        cordSums,
	        spherePointCount,
	        np,
	        ns,
	        r1,
	        pStride,
	        sStride,
	        csStride);
    	rdDevCheckCall(error);

		// error = DeviceDecimate::decimate<DIM, OUT_MEM_LAYOUT>(
		// 	S,
		// 	&rdBruteForceNs,
		// 	r2,
		// 	sStride);
	    error = DeviceDecimate::decimateDistMtx<DIM, OUT_MEM_LAYOUT>(
	        S, &rdBruteForceNs, sStride, distMtx, distMtxStride, 
	        ptsMask, r2);
		rdDevCheckCall(error);
		rdDevCheckCall(cudaDeviceSynchronize());

		ns = rdBruteForceNs;
	}

}


}  // namespace bruteForce
}  // namespace gpu
}  // namespace rd


#endif	// RD_DP_BRUTE_FORCE_CUH_
