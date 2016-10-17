/**
 * @file tiled_tree_node.cuh
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

#include "rd/gpu/device/bounding_box.cuh"
#include "rd/gpu/agent/agent_memcpy.cuh"
#include "rd/utils/memory.h"

#include "cub/util_type.cuh"

#include <utility>

namespace rd
{
namespace gpu
{
namespace tiled
{

namespace nodeDetail
{

template <
    int                     DIM,
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename                T>
__launch_bounds__ (128)
static __global__ void dataCopyKernel(
    T const * d_in,
    T * d_out,
    int numPoints,
    int startOffset,
    int inStride,
    int outStride)
{
    typedef rd::gpu::BlockTileLoadPolicy<
            128,
            8,
            cub::LOAD_CS>
        BlockTileLoadPolicyT;

    typedef rd::gpu::BlockTileStorePolicy<
            128,
            8,
            cub::STORE_DEFAULT>
        BlockTileStorePolicyT;

    typedef rd::gpu::AgentMemcpy<
            BlockTileLoadPolicyT,
            BlockTileStorePolicyT,
            DIM,
            MEM_LAYOUT,
            MEM_LAYOUT,
            rd::gpu::IO_BACKEND_CUB,
            int,
            T>
        AgentMemcpyT;

    #ifdef RD_DEBUG
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("dataCopyKernel: startOffset: %d, numPoints: %d, inStride: %d, outStride: %d\n",
            startOffset, numPoints, inStride, outStride);
    }
    #endif

    AgentMemcpyT agent(d_in, d_out);
    agent.copyRange(startOffset, numPoints, inStride, outStride);
}

}   // end namespace nodeDetail

static __device__ int tiledTreeNodeGlobalObjCounter = 0;
static __device__ int tiledTreeNodeIdCounter = 0;

/**
 * @brief      Tree node data structure. Holds all necessary data to perform tiled ridge detection.
 *
 * @tparam     DIM   Point's dimension
 * @tparam     T     Point's coordinate data type
 *
 * @note       Allows to build binary tree.
 */
template <
    int         DIM,
    typename    T>
struct TiledTreeNode
{
    typedef TiledTreeNode<DIM, T>   NodeT;
    typedef BoundingBox<DIM, T>     BBoxT;
    
    //----------------------------------------------------------------
    // fields
    //----------------------------------------------------------------

    int             id;                                 ///< Tile id
    #ifdef RD_DEBUG
        int         refCounter;
    #endif
    int             treeLevel;                          ///< The tree level this tile is on

    #ifdef RD_DEBUG
    NodeT *         parent;                             ///< Pointer to parent node
    #else
    NodeT const *   parent;                             ///< Pointer to parent node
    #endif
    NodeT *         left;                               ///< Pointer to left child node
    NodeT *         right;                              ///< Pointer to right child node

    int         pointsCnt;                              ///< Number of points inside this tile.
    int         neighboursCnt;                          ///< Number of points neigbouring with this tile
    int         chosenPointsCnt;                        ///< Number of so called "chosen points" by RD algorithm
    int         chosenPointsCapacity;                   ///< Maximum number of "chosen points" which can be inside this tile.
    
    T *         samples;                                ///< Pointer to tile points cooridnates
    T *         neighbours;                             ///< Pointer to tile neighbouring points
    T *         chosenSamples;                          ///< Pointer to tile chosen samples

    T *         cordSums;
    int *       spherePointCnt;

    int         samplesStride;                          ///< Distance between point consecutive coordinates
    int         neighboursStride;
    int         chosenSamplesStride;
    int         cordSumsStride;

    BBoxT       bounds;                                 ///< Bounds of this tile
    int         needEvolve;                             ///< Whether or not this tile need yet another pass of evolution
    
    //----------------------------------------------------------------
    // Constructors
    //----------------------------------------------------------------

    __device__ __forceinline__ TiledTreeNode()
    :
        id(atomicAdd(&tiledTreeNodeIdCounter,1)),
        treeLevel(0),
        parent(nullptr),
        left(nullptr),
        right(nullptr),
        pointsCnt(0),
        neighboursCnt(0),
        chosenPointsCnt(0),
        chosenPointsCapacity(0),
        samples(nullptr),
        neighbours(nullptr),
        chosenSamples(nullptr),
        cordSums(nullptr),
        spherePointCnt(nullptr),
        samplesStride(0),
        neighboursStride(0),
        chosenSamplesStride(0),
        cordSumsStride(0),
        needEvolve(0)
    {
        #ifdef RD_DEBUG
            refCounter = 0;
            int objCnt = atomicAdd(&tiledTreeNodeGlobalObjCounter, 1);
            printf("TiledTreeNode() id: %d,  objCounter: %d\n", id, objCnt + 1);
        #endif
    }

    #ifdef RD_DEBUG
    __device__ __forceinline__ TiledTreeNode(NodeT       * parentNode)
    #else
    __device__ __forceinline__ TiledTreeNode(NodeT const * parentNode)
    #endif
    :
        id(atomicAdd(&tiledTreeNodeIdCounter,1)),
        treeLevel(parentNode->treeLevel+1),
        parent(parentNode),
        left(nullptr),
        right(nullptr),
        pointsCnt(0),
        neighboursCnt(0),
        chosenPointsCnt(0),
        chosenPointsCapacity(0),
        samples(nullptr),
        neighbours(nullptr),
        chosenSamples(nullptr),
        cordSums(nullptr),
        spherePointCnt(nullptr),
        samplesStride(0),
        neighboursStride(0),
        chosenSamplesStride(0),
        cordSumsStride(0),
        needEvolve(0)
    {
        #ifdef RD_DEBUG
            refCounter = 0;
            parent->refCounter++;
            int objCnt = atomicAdd(&tiledTreeNodeGlobalObjCounter,1);
            printf("TiledTreeNode(NodeT const * p) id: %d, parent id: %d, treeLevel: %d, "
                "objCounter: %d\n",
                 id, parent->id, treeLevel, objCnt + 1);
        #endif
    }

    /*
     *  Make exact, shallow copy
     */

    __device__ __forceinline__ TiledTreeNode(TiledTreeNode const & rhs)
    {
        #ifdef RD_DEBUG
        printf("TiledTreeNode::TiledTreeNode(const &) >>>> id: %d\n", id);
        #endif
        shallowCopy(rhs);
    }

    __device__ __forceinline__ TiledTreeNode & operator=(TiledTreeNode const & rhs)
    {
        #ifdef RD_DEBUG
        printf("TiledTreeNode::operator=(const &) >>>> id: %d\n", id);
        #endif
        shallowCopy(rhs);
        return *this;
    }

    __device__ __forceinline__ void shallowCopy(TiledTreeNode const & rhs)
    {
        id = rhs.id;
        treeLevel = rhs.treeLevel;
        parent = rhs.parent;
        left = rhs.left;
        right = rhs.right;
        pointsCnt = rhs.pointsCnt;
        neighboursCnt = rhs.neighboursCnt;
        chosenPointsCnt = rhs.chosenPointsCnt;
        chosenPointsCapacity = rhs.chosenPointsCapacity;
        samples = rhs.samples;
        neighbours = rhs.neighbours;
        chosenSamples = rhs.chosenSamples;
        cordSums = rhs.cordSums;
        spherePointCnt = rhs.spherePointCnt;
        samplesStride = rhs.samplesStride;
        neighboursStride = rhs.neighboursStride;
        chosenSamplesStride = rhs.chosenSamplesStride;
        cordSumsStride = rhs.cordSumsStride;
        needEvolve = rhs.needEvolve;
        bounds = rhs.bounds;

        #ifdef RD_DEBUG
            refCounter = rhs.refCounter;
            if(parent != nullptr)
            {
                parent->refCounter++;
            }
        #endif
    }

    /*
     * Move object possesion 
     */

    __device__ __forceinline__ TiledTreeNode(TiledTreeNode && rhs)
    {
        #ifdef RD_DEBUG
        printf("TiledTreeNode::TiledTreeNode(&&) >>>> id: %d\n", id);
        #endif
        moveObjPosession(std::move(rhs));
    }

    __device__ __forceinline__ TiledTreeNode & operator=(TiledTreeNode && rhs)
    {
        #ifdef RD_DEBUG
        printf("TiledTreeNode::operator=(&&) >>>> id: %d\n", id);
        #endif
        moveObjPosession(std::move(rhs));
        return *this;
    }

    __device__ __forceinline__ void moveObjPosession(TiledTreeNode && rhs)
    {
        id = rhs.id;
        treeLevel = rhs.treeLevel;
        parent = rhs.parent;
        left = rhs.left;
        right = rhs.right;
        pointsCnt = rhs.pointsCnt;
        neighboursCnt = rhs.neighboursCnt;
        chosenPointsCnt = rhs.chosenPointsCnt;
        chosenPointsCapacity = rhs.chosenPointsCapacity;
        samples = rhs.samples;
        neighbours = rhs.neighbours;
        chosenSamples = rhs.chosenSamples;
        cordSums = rhs.cordSums;
        spherePointCnt = rhs.spherePointCnt;
        samplesStride = rhs.samplesStride;
        neighboursStride = rhs.neighboursStride;
        chosenSamplesStride = rhs.chosenSamplesStride;
        cordSumsStride = rhs.cordSumsStride;
        needEvolve = rhs.needEvolve;
        bounds = rhs.bounds;

        #ifdef RD_DEBUG
            refCounter = rhs.refCounter;
            rhs.refCounter = 0;
        #endif

        rhs.id = 0;
        rhs.treeLevel = -1;
        rhs.parent = nullptr;
        rhs.left = nullptr;
        rhs.right = nullptr;
        rhs.pointsCnt = 0;
        rhs.neighboursCnt = 0;
        rhs.chosenPointsCnt = 0;
        rhs.chosenPointsCapacity = 0;
        rhs.samples = nullptr;
        rhs.neighbours = nullptr;
        rhs.chosenSamples = nullptr;
        rhs.cordSums = nullptr;
        rhs.spherePointCnt = nullptr;
        rhs.samplesStride = 0;
        rhs.neighboursStride = 0;
        rhs.chosenSamplesStride = 0;
        rhs.cordSumsStride = 0;
        rhs.needEvolve = 0;
    }

    /**
     * @brief      Creates a new instance of the object with same properties than original.
     *
     * @return     Copy of this object.
     */
    template <
        DataMemoryLayout IN_MEM_LAYOUT,
        DataMemoryLayout OUT_MEM_LAYOUT>
    __device__ __forceinline__ TiledTreeNode* clone(NodeT* nodeClone = nullptr) const 
    {
        #ifdef RD_DEBUG
        printf("TiledTreeNode::clone()\n");
        #endif

        if (empty())
        {
            if (nodeClone == nullptr)
            {
                nodeClone = new NodeT();
            }

            return nodeClone;
        }

        if (nodeClone == nullptr)
        {
            nodeClone = new NodeT();
        }

        nodeClone->treeLevel = treeLevel;
        nodeClone->parent = parent;
        #ifdef RD_DEBUG
        if (parent != nullptr)
        {
            parent->refCounter++;
        }
        #endif
        nodeClone->neighboursCnt = neighboursCnt;
        nodeClone->chosenPointsCnt = chosenPointsCnt;
        nodeClone->chosenPointsCapacity = chosenPointsCapacity;
        nodeClone->samplesStride = samplesStride;
        nodeClone->neighboursStride = neighboursStride;
        nodeClone->chosenSamplesStride = chosenSamplesStride;
        nodeClone->cordSumsStride = cordSumsStride;
        nodeClone->needEvolve = needEvolve;
        nodeClone->bounds = bounds;
        nodeClone->pointsCnt = pointsCnt;
        // not copying this containers as they are created per rd alg run.
        nodeClone->cordSums = nullptr;
        nodeClone->spherePointCnt = nullptr;
        
        nodeClone->chosenSamples = nullptr;
        nodeClone->samples = nullptr;
        nodeClone->neighbours = nullptr;
        nodeClone->left = nullptr;
        nodeClone->right = nullptr;
        

        // copy data
        // 
        
        cudaStream_t copySamplesStream, copyNeighboursStream, copyChosenSamplesStream;
        rdDevCheckCall(cudaStreamCreateWithFlags(&copySamplesStream, cudaStreamNonBlocking));
        rdDevCheckCall(cudaStreamCreateWithFlags(&copyNeighboursStream, cudaStreamNonBlocking));
        rdDevCheckCall(cudaStreamCreateWithFlags(&copyChosenSamplesStream, cudaStreamNonBlocking));

        if (chosenSamples != nullptr && chosenPointsCnt > 0)
        {
            #ifdef RD_DEBUG
            printf("TiledTreeNode::clone() >>>> id: %d, copy chosenSamples(%d)\n", id, chosenPointsCnt);
            #endif
            rdDevCheckCall(rdDevAllocMem(&nodeClone->chosenSamples, 
                &nodeClone->chosenSamplesStride, DIM, chosenPointsCnt, 
                cub::Int2Type<OUT_MEM_LAYOUT>()));
            
            dim3 copyChosenSamplesGrid(1);
            copyChosenSamplesGrid.x = (((chosenPointsCnt + 7) / 8) + 127) / 128;

            nodeDetail::dataCopyKernel<DIM, OUT_MEM_LAYOUT><<<copyChosenSamplesGrid, 128, 0, 
                    copyChosenSamplesStream>>>(
                chosenSamples, nodeClone->chosenSamples, chosenPointsCnt, 0, chosenSamplesStride, 
                nodeClone->chosenSamplesStride);
            rdDevCheckCall(cudaPeekAtLastError());
            #ifdef RD_DEBUG
                rdDevCheckCall(cudaDeviceSynchronize());
            #endif
        }
        if (samples != nullptr && pointsCnt > 0)
        {
            #ifdef RD_DEBUG
            printf("TiledTreeNode::clone() >>>> id: %d, copy samples(%d)\n", id, pointsCnt);
            #endif
            rdDevCheckCall(rdDevAllocMem(&nodeClone->samples, &nodeClone->samplesStride, 
                DIM, pointsCnt, cub::Int2Type<IN_MEM_LAYOUT>()));

            dim3 copySamplesGrid(1);
            copySamplesGrid.x = (((pointsCnt + 7) / 8) + 127) / 128;

            nodeDetail::dataCopyKernel<DIM, IN_MEM_LAYOUT><<<copySamplesGrid, 128, 0, 
                    copySamplesStream>>>(
                samples, nodeClone->samples, pointsCnt, 0, samplesStride, nodeClone->samplesStride);
            rdDevCheckCall(cudaPeekAtLastError());
            #ifdef RD_DEBUG
                rdDevCheckCall(cudaDeviceSynchronize());
            #endif
        }
        if (neighbours != nullptr && neighboursCnt > 0)
        {
            #ifdef RD_DEBUG
            printf("TiledTreeNode::clone() >>>> id: %d, copy neighbours(%d)\n", id, neighboursCnt);
            #endif
            rdDevCheckCall(rdDevAllocMem(&nodeClone->neighbours, &nodeClone->neighboursStride, 
                DIM, neighboursCnt, cub::Int2Type<IN_MEM_LAYOUT>()));

            dim3 copyNeighboursGrid(1);
            copyNeighboursGrid.x = (((neighboursCnt + 7) / 8) + 127) / 128;

            nodeDetail::dataCopyKernel<DIM, IN_MEM_LAYOUT><<<copyNeighboursGrid, 128, 0, 
                    copyNeighboursStream>>>(
                neighbours, nodeClone->neighbours, neighboursCnt, 0, neighboursStride, 
                nodeClone->neighboursStride);
            rdDevCheckCall(cudaPeekAtLastError());
            #ifdef RD_DEBUG
                rdDevCheckCall(cudaDeviceSynchronize());
            #endif
        }

        #ifdef RD_DEBUG
            rdDevCheckCall(cudaDeviceSynchronize());
        #endif

        if (left != nullptr)
        {
            #ifdef RD_DEBUG
            printf("TiledTreeNode::clone() >>>> id: %d, clone left child\n", id);
            #endif
            nodeClone->left = left->clone<IN_MEM_LAYOUT, OUT_MEM_LAYOUT>();
        }
        if (right != nullptr)
        {
            #ifdef RD_DEBUG
            printf("TiledTreeNode::clone() >>>> id: %d, clone right child\n", id);
            #endif
            nodeClone->right = right->clone<IN_MEM_LAYOUT, OUT_MEM_LAYOUT>();
        }

        rdDevCheckCall(cudaDeviceSynchronize());

        rdDevCheckCall(cudaStreamDestroy(copySamplesStream));
        rdDevCheckCall(cudaStreamDestroy(copyNeighboursStream));
        rdDevCheckCall(cudaStreamDestroy(copyChosenSamplesStream));

        return nodeClone;
    }

    //----------------------------------------------------------------
    // Bounds initialization
    //----------------------------------------------------------------

    /**
     * @brief      Calculates node tile bounds.
     *
     * @param[in]  initTileCntPerDim  Array with initial tiles count per subsequent dimensions
     * @param[in]  initTileCnt        Overall initial tiles count
     * @param      parentBBox         The parent node bounding box
     */
     __device__ __forceinline__ void initTileBounds(
        int                                 linearTid,
        cub::ArrayWrapper<int, DIM> const   initTileCntPerDim,
        int                                 initTileCnt,
        BBoxT const &                       parentBBox)
    {
        T boundStep;
        int tileCnt = initTileCnt;
        int tileSpatialCoord;
        
        /*----------------------------------------------------------------------------------------- 
         *  translation of linear tid into DIM-dimensional coordinates: 
         *  require a number of unit bins (product of all dimension sizes)
         *  start from last coordinate (that is least likely to be chanching) and repeat
         *   1) divide the reminder of previous division by product of all dim sizes lower
         *      than this one
         *   2) the integer part is the current dim idx
         *  First dim idx value is the reminder of previous division
         *  example:
         *  
         *  4-5-6 (dim sizes)
         *  (3,4,1); (point coordinates, so its linear id is 39)
         *  3D) 39 / (4*5) = 1; reminder 19
         *  2D) 19 / 4 = 4, reminder 3;
         *  1D) 3;
         *----------------------------------------------------------------------------------------*/
        
        #pragma unroll
        for (int d = DIM - 1; d > 0; --d)
        {
            int nBins = initTileCntPerDim.array[d];
            boundStep = parentBBox.dist[d] / static_cast<T>(nBins);
            tileCnt /= nBins;   // product of lower dimensions
            tileSpatialCoord = linearTid / tileCnt;
            linearTid -= tileSpatialCoord * tileCnt;

            bounds.min(d) = parentBBox.min(d) + T(tileSpatialCoord) * boundStep;
            bounds.max(d) = parentBBox.min(d) + T(tileSpatialCoord + 1) * boundStep;
            // if we are the last tile in respective dimension, set max as a parent max. This
            // will result in slightly larger tile than other (in this dim).
            // This enable to correctly cover all space. Otherwise, (due to the floating point
            // rounding error we would miss some points, because sum of tile dist won't be equal
            // to parent dist in respective dimension)
            if (tileSpatialCoord == (nBins - 1))
            {
                bounds.max(d) = parentBBox.max(d);
            }
        }

        boundStep = parentBBox.dist[0] / static_cast<T>(initTileCntPerDim.array[0]);
        tileSpatialCoord = linearTid;
        bounds.min(0) = parentBBox.min(0) + T(tileSpatialCoord) * boundStep;
        bounds.max(0) = parentBBox.min(0) + T(tileSpatialCoord + 1) * boundStep;
        if (tileSpatialCoord == (initTileCntPerDim.array[0] - 1))
        {
            bounds.max(0) = parentBBox.max(0);
        }
        bounds.calcDistances();
    }
    
    //----------------------------------------------------------------
    // Releasing memory, destructor, emptyness
    //----------------------------------------------------------------

    __device__ __forceinline__ ~TiledTreeNode()
    {
        #ifdef RD_DEBUG
            printf("~TiledTreeNode() id: %d refCounter: %d\n", id, refCounter);
        #endif
        clear();
        atomicSub(&tiledTreeNodeGlobalObjCounter, 1);
    }

    __device__ __forceinline__ void releaseChild(int childNodeId)
    {
        #ifdef RD_DEBUG
        _CubLog("TreeNode::releaseChild id: %d, parent id: %d\n",
            childNodeId, id);
        #endif
        if (left->id == childNodeId)
        {
            delete left;
            left = nullptr;
        }
        if (right->id == childNodeId)
        {
            delete right;
            right = nullptr;
        }
    }

    __device__ __forceinline__ bool empty() const
    {
        return !haveChildren() &&
            samples == nullptr &&
            chosenSamples == nullptr &&
            neighbours == nullptr &&
            cordSums == nullptr &&
            spherePointCnt == nullptr;
    }

    __device__ __forceinline__ bool haveChildren() const
    {
        return left != nullptr || right != nullptr;
    }

    __device__ __forceinline__ void clearData() 
    {
        if (samples != nullptr)         { delete[] samples;         samples = nullptr;}
        if (chosenSamples != nullptr)   { delete[] chosenSamples;   chosenSamples = nullptr;}
        if (neighbours != nullptr)      { delete[] neighbours;      neighbours = nullptr;}
        if (cordSums != nullptr)        { delete[] cordSums;        cordSums = nullptr;}
        if (spherePointCnt != nullptr)  { delete[] spherePointCnt;  spherePointCnt = nullptr;}
    }

    __device__ __forceinline__ void clear()
    {
        #ifdef RD_DEBUG
            printf("TiledTreeNode::clear() id: %d, level: %d, withChildren: %d, refCounter: %d\n",
                id, treeLevel, haveChildren(), refCounter);
        #endif

        clearData();
        if (left != nullptr)            { delete left;  left = nullptr;}
        if (right != nullptr)           { delete right; right = nullptr;}
        #ifdef RD_DEBUG
            if (parent != nullptr)      { parent->refCounter--; }
        #endif
    }

    
    //----------------------------------------------------------------
    // Debug printing
    //----------------------------------------------------------------

    __device__ __forceinline__ void print() const
    {
        #ifdef RD_DEBUG
        _CubLog("Node id %d, pointsCnt: %d, chosenPointsCnt %d, objCnt: %d, refCnt: %d"
            " treeLevel: %d\n",
            id, pointsCnt, chosenPointsCnt, tiledTreeNodeGlobalObjCounter, refCounter,
            treeLevel);
        #else 
        _CubLog("Node id %d, pointsCnt: %d, chosenPointsCnt %d, objCnt: %d, treeLevel: %d\n",
            id, pointsCnt, chosenPointsCnt, tiledTreeNodeGlobalObjCounter, treeLevel);
        #endif
        bounds.print();
    }

    __device__ __forceinline__ void printRecursive() const
    {
        char * offsetStr = new char[treeLevel+1];
        for (int i = 0; i < treeLevel; ++i)
        {
            offsetStr[i] = '>';
        }
        offsetStr[treeLevel] = '\0';

        #ifdef RD_DEBUG
        printf("%s Node id %d, pointsCnt: %d, chosenPointsCnt %d, objCnt: %d, refCnt: %d"
            " treeLevel: %d\n",
            offsetStr, id, pointsCnt, chosenPointsCnt, tiledTreeNodeGlobalObjCounter, refCounter,
            treeLevel);
        #else 
        printf("%s Node id %d, pointsCnt: %d, chosenPointsCnt %d, objCnt: %d, treeLevel: %d\n",
            offsetStr, id, pointsCnt, chosenPointsCnt, tiledTreeNodeGlobalObjCounter, treeLevel);
        #endif

        if (left != nullptr)
        {
            left->printRecursive();
        }
        if (right != nullptr)
        {
            right->printRecursive();
        }

        delete offsetStr;
    }
};



} // end namespace tiled
} // end namespace gpu
} // end namespace rd