/**
 * @file rd_brute_force_globals.cuh
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

#ifndef RD_BRUTE_FORCE_GLOBALS_CUH_
#define RD_BRUTE_FORCE_GLOBALS_CUH_


namespace rd
{
namespace gpu
{

/// number of choosen samples
static __device__ int rdBruteForceNs = 0;
/// flag indicating we have numerically visible change when shifting sphere centers
static __device__ int rdContFlag = 1;

} // end namspace gpu
} // end namspace rd

/*****************************************************************************************/

#endif /* RD_BRUTE_FORCE_GLOBALS_CUH_ */
