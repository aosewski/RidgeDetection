/**
 * @file warp_functions.cuh
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and Information
 * Technology Warsaw University of Technology 2016
 */

#ifndef WARP_FUNCTIONS_CUH_
#define WARP_FUNCTIONS_CUH_ 

#include <device_launch_parameters.h>
#include <device_functions.h>

namespace rd
{
namespace gpu
{
    
#if defined(__cplusplus) && defined(__CUDACC__)

#define RD_WARP_SZ         32
#define RD_WARP_MASK       31
#define RD_LOG2_WARP_SZ    5
#define RD_WARP_CONVERGED  0xffffffff

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300

template <typename T>
__device__ __forceinline__ void warpReduce(T &value)
{
    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i >>= 1)
        value += __shfl_xor(value, i, warpSize);
}


template <typename T>
__device__ __forceinline__ void broadcast(T &arg, const int srcLaneId)
{
    arg = __shfl(arg, srcLaneId);
}

#endif  // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300

/**
 * @return Thread inner-warp index.
 */
__device__ __forceinline__ int laneId() 
{ 

    return threadIdx.x & RD_WARP_MASK; 
}

__device__ __forceinline__ int warpId()
{
    return threadIdx.x >> RD_LOG2_WARP_SZ;
}

__device__ __forceinline__ bool isWarpConverged()
{
    return (__ballot(true) == RD_WARP_CONVERGED);
}

__device__ __forceinline__ int warpActiveThreads()
{
    return __popc(__ballot(true));
}


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300
/**
 * @brief      Finds peers sharing the same @p key value
 *
 * @param[in]  key   Parameter value
 *
 * @tparam     T     Key data type.
 *
 * @return     Mask with peer lanes bit set.
 */
template <typename T>
__device__ __forceinline__ uint warpGetPeers(T const key) 
{
    uint peers = 0;
    bool is_peer;

    // in the beginning, all lanes are available
    uint unclaimed = 0xffffffff;

    do 
    {
        // fetch key of first unclaimed lane and compare with this key
        is_peer = (key == __shfl(key, __ffs(unclaimed) - 1));

        // determine which lanes had a match
        peers = __ballot(is_peer);

        // remove lanes with matching keys from the pool
        unclaimed ^= peers;

        // quit if we had a match
    } while (!is_peer);

    return peers;
}

/**
 * @brief      Finds peers sharing the same @p key value
 *
 * @param[in]  key   Parameter value
 *
 * @tparam     T     Key data type.
 *
 * @return     Mask with peer lanes bit set.
 */
template <typename T>
__device__ __forceinline__ uint warpGetPeers2(T const key) 
{
    uint warpMask = 0;
    uint myPeers = 0;
    bool is_peer;

    // in the beginning, all lanes are available
    uint unclaimed = 0xffffffff;

    do 
    {
        // fetch key of first unclaimed lane and compare with this key
        is_peer = (key == __shfl(key, __ffs(unclaimed) - 1));

        // determine which lanes had a match
        warpMask = __ballot(is_peer);

        if (is_peer)
        {
            myPeers = warpMask;
        }

        // remove lanes with matching keys from the pool
        unclaimed ^= warpMask;

        // quit if we had a match
    } while (unclaimed != 0);

    return myPeers;
}


/**
 * @brief      Reduces values among peer threads.
 *
 * @param[in]  peers  Bit mask indicating peer threads to reduce.
 * @param      value  Thread value.
 *
 * @tparam     T      Values data type
 *
 * @return     Reduced value.
 */
template <typename T>
__device__ __forceinline__ T warpReducePeers(uint peers, T &value) 
{

    int lane = laneId();

    // find the peer with lowest lane index
    int first = __ffs(peers)-1;

    // calculate own relative position among peers
    int rel_pos = __popc(peers << (RD_WARP_SZ - lane));

    // ignore peers with lower (or same) lane index
    peers &= (0xfffffffe << lane);

    while(__any(peers)) 
    {
        // find next-highest remaining peer
        int next = __ffs(peers);

        // __shfl() only works if both threads participate, so we always do.
        T t = __shfl(value, next - 1);

        // only add if there was anything to add
        if (next)
        {
            value += t;  
        } 

        // all lanes with their least significant index bit set are done
        uint done = rel_pos & 1;

        // remove all peers that are already done
        peers &= ~__ballot(done);

        // abuse relative position as iteration counter
        rel_pos >>= 1;
    }

    // distribute final result to all peers (optional)
    T res = __shfl(value, first);

    return value;
}

/**
 * @brief         Warp-aggregated atomic increment
 *
 * @param[in|out] ctr   Pointer to counter.
 */
__device__ __forceinline__ int atomicWarpAggInc(int *ctr) 
{
    // select active threads
    int mask = __ballot(1);
    // select the leader
    int leader = __ffs(mask) - 1;
    // leader does the update
    int res;
    if(laneId() == leader)
    {
      res = atomicAdd(ctr, __popc(mask));
    }
    // broadcast result
    broadcast(res, leader);
    // each thread computes its own value
    return res + __popc(mask & ((1 << laneId()) - 1));
} 


#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300

#undef RD_WARP_SZ         
#undef RD_WARP_MASK       
#undef RD_LOG2_WARP_SZ    
#undef RD_WARP_CONVERGED

#endif // defined(__cplusplus) && defined(__CUDACC__)

} // end namespace gpu
} // end namespace rd

#endif /* WARP_FUNCTIONS_CUH_ */
