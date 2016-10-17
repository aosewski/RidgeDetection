/**
 * @file decimate.cuh
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
#pragma once

#include "rd/gpu/block/cta_count_neighbour_points.cuh"
#include "rd/gpu/device/tiled/tiled_tree_node.cuh"
#include "rd/gpu/util/dev_utilities.cuh"
#include "rd/utils/memory.h"

#include "cub/util_debug.cuh"
#include "cub/util_device.cuh"

namespace rd
{
namespace gpu
{
namespace tiled
{

template <
    int                 DIM,
    int                 BLOCK_SIZE,
    DataMemoryLayout    MEM_LAYOUT,
    typename            T>
class BlockDecimate
{

private:

    typedef TiledTreeNode<DIM, T> NodeT;

    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /// Decimate helper
    template <DataMemoryLayout _MEM_LAYOUT, int DUMMY>
    struct DecimateInternal;

    template <int DUMMY>
    struct DecimateInternal<ROW_MAJOR, DUMMY>
    {
        /// thread-private fields
        int         globalChosenPointsNum;
        T           rSqr;
        NodeT **    d_leafNodes;
        int         leafNum;

        /// Constructor
        __device__ __forceinline__ DecimateInternal(
            NodeT **    d_leafNodes,
            int         leafNum,
            T r)
        :
            globalChosenPointsNum(0),
            d_leafNodes(d_leafNodes),
            leafNum(leafNum),
            rSqr(r*r)
        {}

        __device__ __forceinline__ void decimate(
            int *       d_chosenPointsNum) 
        {
            int prevIterPtsNum = 0;
            globalChosenPointsNum = *d_chosenPointsNum;

            // iterate untill nothing changes in subsequent iterations or we reach
            // threshold points number
            while (prevIterPtsNum != globalChosenPointsNum && globalChosenPointsNum > 3)
            {

    #ifdef RD_DEBUG
    if (threadIdx.x == 0)
    {
        printf("--------\t[globalDecimate] prevIterPtsNum: %d, globalChosenPointsNum: %d\n",
            prevIterPtsNum, globalChosenPointsNum);
    }
    #endif
                prevIterPtsNum = globalChosenPointsNum;
                // iterate over all leafs
                for (int ln = 0; ln < leafNum; ++ln)
                {
                    NodeT * node = d_leafNodes[ln];
                    // #ifdef RD_DEBUG
                    // if (threadIdx.x == 0)
                    // {
                    //     printf("###### checkNode id: %d, nodeChosenPointsCnt: %d\n", 
                    //         node->id, node->chosenPointsCnt);
                    // }
                    // #endif

                    // iterate over leaf's all chosen points
                    for (int i = 0; i < node->chosenPointsCnt;)
                    {
                        // check with every other leaf
                        if(checkPoint(node, i))
                        {
                            i++;
                        }
                        else
                        {
                            // checked point has been removed so this leaf needs one more evolution 
                            // iteration
                            node->needEvolve = 1;
                        }

                        if (globalChosenPointsNum < 3)
                        {
                    // #ifdef RD_DEBUG
                    // if (threadIdx.x == 0)
                    // {
                    //     printf("###### [checkNode] globalChosenPointsNum < 3! ---- END-----\n");
                    // }
                    // #endif
                            // stop checking other nodes
                            if (threadIdx.x == 0)
                            {
                                *d_chosenPointsNum = globalChosenPointsNum;
                            }
                            return;
                        }
                    }
                }
            }

    #ifdef RD_DEBUG
    if (threadIdx.x == 0)
    {
        printf("--------\t[globalDecimate] End! globalChosenPointsNum: %d\n", 
            globalChosenPointsNum);
    }
    #endif

            if (threadIdx.x == 0)
            {
                *d_chosenPointsNum = globalChosenPointsNum;
            }
        }

        __device__ __forceinline__ int checkPoint(
            NodeT *     checkedNode,
            int         ptIdx)
        {
            int neighbours_r = 0;
            int neighbours_2r = 0;

            // check with other leafs chosen points
            for (int n = 0; n < leafNum; ++n)
            {
                NodeT const * node = d_leafNodes[n];

    // #ifdef RD_DEBUG
    // if (threadIdx.x == 0)
    // {
    //     printf(">>>>>>> [checkPoint] ptIdx: %d, check with node id: %d, nodePtsCnt: %d\n",
    //         ptIdx, node->id, node->chosenPointsCnt);
    // }
    // #endif
                neighbours_r += ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(
                    node->chosenSamples, node->chosenPointsCnt, 
                    checkedNode->chosenSamples + ptIdx * DIM, rSqr, 4, rowMajorOrderTag());
                __syncthreads();
                
                // we found at least 4 points in currently checked point neighbourhood
                if (neighbours_r >= 4)
                {
                    // decrement counters
                    globalChosenPointsNum--;
                    checkedNode->chosenPointsCnt--;
                    /*
                     * I know that it is run with only one block of threads.
                     */
                    // round up to closest multiply of BLOCK_SIZE
                    T * src = checkedNode->chosenSamples + (ptIdx + 1) * DIM;
                    T * dst = checkedNode->chosenSamples + ptIdx * DIM;
                    T tmp;
                    const int count = (checkedNode->chosenPointsCnt - ptIdx) * DIM;
                    int k = (count + BLOCK_SIZE ) / BLOCK_SIZE * BLOCK_SIZE;
                    for (int x = threadIdx.x; x < k; x += BLOCK_SIZE)
                    {
                        if (x < count)
                            tmp = src[x];
                        __syncthreads();
                        if (x < count)
                            dst[x] = tmp;
                    }

    // #ifdef RD_DEBUG
    // if (threadIdx.x == 0)
    // {
    //     printf(" -----> too many neighbours! (1)\n");
    // }
    // #endif
                    return 0;
                }

                if (neighbours_2r < 2)
                {
                    neighbours_2r += ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(
                        node->chosenSamples, node->chosenPointsCnt, 
                        checkedNode->chosenSamples + ptIdx * DIM, 4.f*rSqr, 3, rowMajorOrderTag());
                    __syncthreads();
                }
            }

            // we haven't found at least 3 points in 4R neighbourhood, so
            // this point has at most 2 neighbours in 4R neighbourhood
            if (neighbours_2r <= 2)
            {
                globalChosenPointsNum--;
                checkedNode->chosenPointsCnt--;
                /*
                 * I know that it is run with only one block of threads.
                 */
                // round up to closest multiply of BLOCK_SIZE
                T * src = checkedNode->chosenSamples + (ptIdx + 1) * DIM;
                T * dst = checkedNode->chosenSamples + ptIdx * DIM;
                T tmp;
                const int count = (checkedNode->chosenPointsCnt - ptIdx) * DIM;
                int k = (count + BLOCK_SIZE ) / BLOCK_SIZE * BLOCK_SIZE;
                for (int x = threadIdx.x; x < k; x += BLOCK_SIZE)
                {
                    if (x < count)
                        tmp = src[x];
                    __syncthreads();
                    if (x < count)
                        dst[x] = tmp;
                }

// #ifdef RD_DEBUG
// if (threadIdx.x == 0)
// {
//     printf(" -----> not enough neighbours! (2)\n");
// }
// #endif
                return 0;
            } 


    // #ifdef RD_DEBUG
    // if (threadIdx.x == 0)
    // {
    //     printf("\n");
    // }
    // #endif
            // if currently checked chosen point has correct number of neighbours, 
            // proceed to next one
            return 1;
        }
    };

    template <int DUMMY>
    struct DecimateInternal<COL_MAJOR, DUMMY>
    {
        /// Constructor
        /// thread-private fields
        int         globalChosenPointsNum;
        T           rSqr;
        NodeT **    d_leafNodes;
        int         leafNum;

        /// Constructor
        __device__ __forceinline__ DecimateInternal(
            NodeT **    d_leafNodes,
            int         leafNum,
            T r)
        :
            globalChosenPointsNum(0),
            d_leafNodes(d_leafNodes),
            leafNum(leafNum),
            rSqr(r*r)
        {}

        __device__ __forceinline__ void decimate(
            int *       d_chosenPointsNum)
        {
            int prevIterPtsNum = 0;
            globalChosenPointsNum = *d_chosenPointsNum;

            // iterate untill nothing changes in subsequent iterations or we reach
            // threshold points number
            while (prevIterPtsNum != globalChosenPointsNum && globalChosenPointsNum > 3)
            {

    // #ifdef RD_DEBUG
    // if (threadIdx.x == 0)
    // {
    //     printf("--------\t\t[globalDecimate] prevIterPtsNum: %d, globalChosenPointsNum: %d\n",
    //         prevIterPtsNum, globalChosenPointsNum);
    // }
    // #endif
                prevIterPtsNum = globalChosenPointsNum;
                // iterate over all leafs
                for (int ln = 0; ln < leafNum; ++ln)
                {
                    NodeT * node = d_leafNodes[ln];
                    // #ifdef RD_DEBUG
                    // if (threadIdx.x == 0)
                    // {
                    //     printf("###### checkNode id: %d, nodeChosenPointsCnt: %d\n", 
                    //         node->id, node->chosenPointsCnt);
                    // }
                    // #endif

                    // iterate over leaf's all chosen points
                    for (int i = 0; i < node->chosenPointsCnt;)
                    {
                        // check with every other leaf
                        if(checkPoint(node, i))
                        {
                            i++;
                        }
                        else
                        {
                            // checked point has been removed so this leaf needs one more evolution 
                            // iteration
                            node->needEvolve = 1;
                        }

                        if (globalChosenPointsNum < 3)
                        {
                    // #ifdef RD_DEBUG
                    // if (threadIdx.x == 0)
                    // {
                    //     printf("###### [checkNode] globalChosenPointsNum < 3! ---- END-----\n");
                    // }
                    // #endif
                            // stop checking other nodes
                            if (threadIdx.x == 0)
                            {
                                *d_chosenPointsNum = globalChosenPointsNum;
                            }
                            return;
                        }
                    }
                }
            }

    // #ifdef RD_DEBUG
    // if (threadIdx.x == 0)
    // {
    //     printf("--------\t\t[globalDecimate] End! globalChosenPointsNum: %d\n", 
    //         globalChosenPointsNum);
    // }
    // #endif

            if (threadIdx.x == 0)
            {
                *d_chosenPointsNum = globalChosenPointsNum;
            }
        }

        __device__ __forceinline__ int checkPoint(
            NodeT *     checkedNode,
            int         ptIdx) 
        {
            int neighbours_r = 0;
            int neighbours_2r = 0;

            // check with other leafs chosen points
            for (int n = 0; n < leafNum; ++n)
            {
                NodeT * node = d_leafNodes[n];

                neighbours_r += ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(
                    node->chosenSamples, node->chosenPointsCnt, node->chosenSamplesStride,
                    checkedNode->chosenSamples + ptIdx, checkedNode->chosenSamplesStride, 
                    rSqr, 4, colMajorOrderTag());
                __syncthreads();        
                
                // we found at least 4 points in currently checked point neighbourhood
                if (neighbours_r >= 4)
                {
                    globalChosenPointsNum--;
                    checkedNode->chosenPointsCnt--;
                    /*
                     * I know that it is run with only one block of threads.
                     */
                    // move the rest of data in array
                    // round up to closest multiply of BLOCK_SIZE
                    T * src = checkedNode->chosenSamples + (ptIdx+1);
                    T * dst = checkedNode->chosenSamples + ptIdx;
                    T tmp;
                    const int count = checkedNode->chosenPointsCnt - ptIdx;
                    int k = (count + BLOCK_SIZE ) / BLOCK_SIZE * BLOCK_SIZE;
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                    {
                        for (int x = threadIdx.x; x < k; x += BLOCK_SIZE)
                        {
                            if (x < count)
                                tmp = src[d * checkedNode->chosenSamplesStride + x];
                            __syncthreads();
                            if (x < count)
                                dst[d * checkedNode->chosenSamplesStride + x] = tmp;
                        }
                    }
                    
                    return 0;
                }

                if (neighbours_2r < 2)
                {
                    neighbours_2r += ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(
                        node->chosenSamples, node->chosenPointsCnt, node->chosenSamplesStride,
                        checkedNode->chosenSamples + ptIdx, checkedNode->chosenSamplesStride, 
                        4.f * rSqr, 3, colMajorOrderTag());
                    __syncthreads();
                }

            }

            // we haven't found at least 3 points in 4R neighbourhood, so
            // this point has at most 2 neighbours in 4R neighbourhood
            if (neighbours_2r <= 2)
            {
                globalChosenPointsNum--;
                checkedNode->chosenPointsCnt--;
                /*
                 * I know that it is run with only one block of threads.
                 */
                // move the rest of data in array
                // round up to closest multiply of BLOCK_SIZE
                T * src = checkedNode->chosenSamples + (ptIdx+1);
                T * dst = checkedNode->chosenSamples + ptIdx;
                T tmp;
                const int count = checkedNode->chosenPointsCnt - ptIdx;
                int k = (count + BLOCK_SIZE ) / BLOCK_SIZE * BLOCK_SIZE;
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    for (int x = threadIdx.x; x < k; x += BLOCK_SIZE)
                    {
                        if (x < count)
                            tmp = src[d * checkedNode->chosenSamplesStride + x];
                        __syncthreads();
                        if (x < count)
                            dst[d * checkedNode->chosenSamplesStride + x] = tmp;
                    }
                }
                
                return 0;
            } 
            // if currently checked chosen point has correct number of neighbours, 
            // proceed to next one
            return 1;
        }
    };

    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Internal choose implementation to use
    typedef DecimateInternal<MEM_LAYOUT, 0> InternalDecimate;

public:

    /******************************************************************************
     * Collective constructors
     ******************************************************************************/

     __device__ __forceinline__ BlockDecimate()
     {}

    /******************************************************************************
     * Decimate
     ******************************************************************************/

     __device__ __forceinline__ void globalDecimate(
        TiledTreeNode<DIM, T> **    d_leafNodes,
        int                         leafNum,
        int *                       d_chosenPointsNum,
        T                           r)
     {
        InternalDecimate(d_leafNodes, leafNum, r).decimate(d_chosenPointsNum);
     }

};


} // end namespace tiled
} // end namespace gpu
} // end namespace rd
