/**
 * @file rd_brute_force_globals.cuh
 * @date 12-04-2015
 * @author Adam Rogowiec
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
