/**
 * @file tree_drawer.cuh
 * @author Adam Rogowiec
 * 
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is conducted under the supervision of prof. dr hab. inż. Marek
 * Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */


#pragma once

#include <iostream>
#include <stdexcept>
#include <type_traits> 
#include <helper_cuda.h>

#ifdef RD_USE_OPENMP
#include <omp.h>
#endif

#include "rd/gpu/device/bounding_box.cuh"
#include "rd/gpu/agent/agent_memcpy.cuh"
#include "rd/gpu/util/dev_memcpy.cuh"
#include "rd/gpu/util/dev_utilities.cuh"

#include "rd/utils/memory.h"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/name_traits.hpp"
#include "tests/test_util.hpp"

#include "cub/util_type.cuh"

namespace rd 
{
namespace gpu
{
namespace tiled
{
namespace util
{

//------------------------------------------------------------------------
// Drawing graphs utilities
//------------------------------------------------------------------------

template <typename T>
void collectTileBounds(
    std::vector<T> &           tbound,
    BoundingBox<2,T> const &   bounds)
{
    // upper left
    tbound.push_back(bounds.min(0));
    tbound.push_back(bounds.max(1));
    // upper right
    tbound.push_back(bounds.max(0));
    tbound.push_back(bounds.max(1));
    // bottom right
    tbound.push_back(bounds.max(0));
    tbound.push_back(bounds.min(1));
    // bottom left
    tbound.push_back(bounds.min(0));
    tbound.push_back(bounds.min(1));
    // upper left
    tbound.push_back(bounds.min(0));
    tbound.push_back(bounds.max(1));
};

template <typename T>
void collectTileBounds(
    std::vector<T> &           tbound,
    BoundingBox<3,T> const &   bounds)
{
    /*
              5_____6
              /|   /|
             / |  / |
           8/__4_/7 |
            | /--|-/ 3
            |/___|/
           1     2 
     */

    // 1
    tbound.push_back(bounds.min(0));
    tbound.push_back(bounds.max(1));
    tbound.push_back(bounds.min(2));
    // 2
    tbound.push_back(bounds.max(0));
    tbound.push_back(bounds.max(1));
    tbound.push_back(bounds.min(2));
    // 3
    tbound.push_back(bounds.max(0));
    tbound.push_back(bounds.max(1));
    tbound.push_back(bounds.max(2));
    // 4
    tbound.push_back(bounds.min(0));
    tbound.push_back(bounds.max(1));
    tbound.push_back(bounds.max(2));
    // 5
    tbound.push_back(bounds.min(0));
    tbound.push_back(bounds.min(1));
    tbound.push_back(bounds.max(2));
    // 6
    tbound.push_back(bounds.max(0));
    tbound.push_back(bounds.min(1));
    tbound.push_back(bounds.max(2));
    // 7
    tbound.push_back(bounds.max(0));
    tbound.push_back(bounds.min(1));
    tbound.push_back(bounds.min(2));
    // 8
    tbound.push_back(bounds.min(0));
    tbound.push_back(bounds.min(1));
    tbound.push_back(bounds.min(2));
    // 5
    tbound.push_back(bounds.min(0));
    tbound.push_back(bounds.min(1));
    tbound.push_back(bounds.max(2));
    // 4
    tbound.push_back(bounds.min(0));
    tbound.push_back(bounds.max(1));
    tbound.push_back(bounds.max(2));
    // 1
    tbound.push_back(bounds.min(0));
    tbound.push_back(bounds.max(1));
    tbound.push_back(bounds.min(2));
    // 8
    tbound.push_back(bounds.min(0));
    tbound.push_back(bounds.min(1));
    tbound.push_back(bounds.min(2));
    // 7
    tbound.push_back(bounds.max(0));
    tbound.push_back(bounds.min(1));
    tbound.push_back(bounds.min(2));
    // 2
    tbound.push_back(bounds.max(0));
    tbound.push_back(bounds.max(1));
    tbound.push_back(bounds.min(2));
    // 3
    tbound.push_back(bounds.max(0));
    tbound.push_back(bounds.max(1));
    tbound.push_back(bounds.max(2));
    // 6
    tbound.push_back(bounds.max(0));
    tbound.push_back(bounds.min(1));
    tbound.push_back(bounds.max(2));
};

template <
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    int DIM, 
    typename T>
void doDrawAllTileBounds(
    int allPointsNum,
    int treeLeafsNum,
    BoundingBox<DIM, T> *tileBounds,
    int *tilePointsNum,
    T **tilePoints,
    std::string devName)
{
    rd::GraphDrawer<T> gDrawer;
    std::ostringstream graphName;
    std::vector<std::vector<T>> bounds;

    graphName << typeid(T).name() << "_" 
        << DIM << "D_"
        << getCurrDateAndTime() << "_"
        << allPointsNum << "-pt_"
        << devName << "_"
        << DataMemoryLayoutNameTraits<IN_MEM_LAYOUT>::name << "_"
        << DataMemoryLayoutNameTraits<OUT_MEM_LAYOUT>::name << "_"
        << "_bounds";
    gDrawer.startGraph(graphName.str(), DIM);
    if (DIM == 3)
    {
        gDrawer.setGraph3DConf();
    }

    // draw each tile points
    for (int i = 0; i < treeLeafsNum; ++i)
    {
        std::vector<T> tb;
        collectTileBounds(tb, tileBounds[i]);
        bounds.push_back(tb);

        if (tilePoints[i] != nullptr && tilePointsNum[i] > 0)
        {
            gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.8 ",
                tilePoints[i], rd::GraphDrawer<T>::POINTS, tilePointsNum[i]);
        }
    }

    // draw each tile bounds
    for (int i = 0; i < treeLeafsNum; ++i)
    {
        gDrawer.addPlotCmd("'-' w l lt 8 lc rgb '#CB1E1E' lw 1.5 ",
            bounds[i].data(), rd::GraphDrawer<T>::LINE, (DIM == 2) ? 5 : 16);
    }
    gDrawer.endGraph();
}


template <
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    typename T>
void drawAllTileBounds(
    int allPointsNum,
    int treeLeafsNum,
    BoundingBox<2, T> *tileBounds,
    int *tilePointsNum,
    T **tilePoints,
    std::string devName)
{
    doDrawAllTileBounds<IN_MEM_LAYOUT, OUT_MEM_LAYOUT, 2>(
        allPointsNum, treeLeafsNum, tileBounds, tilePointsNum, tilePoints, devName);
}

template <
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    typename T>
void drawAllTileBounds(
    int allPointsNum,
    int treeLeafsNum,
    BoundingBox<3, T> *tileBounds,
    int *tilePointsNum,
    T **tilePoints,
    std::string devName)
{
    doDrawAllTileBounds<IN_MEM_LAYOUT, OUT_MEM_LAYOUT, 3>(allPointsNum, treeLeafsNum, tileBounds, tilePointsNum,
        tilePoints, devName);
}

template <
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    int DIM, 
    typename T>
void drawAllTileBounds(
    int ,
    int ,
    BoundingBox<DIM, T> *,
    int *,
    T **,
    std::string )
{
}

template <
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    int DIM, 
    typename T>
void doDrawTile(
    int allPointsNum,
    T const * allPoints,
    BoundingBox<DIM, T> tileBounds,
    int tilePointsNum,
    T const * tilePoints, 
    int tileNeighboursNum,
    T const * tileNeighbours,
    int tileChosenPointsNum,
    T const * tileChosenPoints,
    int tileId,
    std::string devName)
{
    rd::GraphDrawer<T> gDrawer;
    std::ostringstream graphName;
    std::vector<T> bounds;

    collectTileBounds(bounds, tileBounds);

    graphName << typeid(T).name() << "_"
        << DIM << "D_" 
        << getCurrDateAndTime() << "_"
        << allPointsNum << "-pt_"
        << devName << "_" 
        << rd::DataMemoryLayoutNameTraits<IN_MEM_LAYOUT>::name << "_"
        << rd::DataMemoryLayoutNameTraits<OUT_MEM_LAYOUT>::name << "_"
        << tileId << "_tile";
    gDrawer.startGraph(graphName.str(), DIM);
    if (DIM == 3)
    {   
        gDrawer.setGraph3DConf();
    }
    if (allPoints != nullptr && allPointsNum > 0)
    {
        gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
            allPoints, rd::GraphDrawer<T>::POINTS, allPointsNum);
    }
    if (tileNeighbours != nullptr && tileNeighboursNum > 0)
    {
        gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#B2D88C' ps 1 ",
            tileNeighbours, rd::GraphDrawer<T>::POINTS, tileNeighboursNum);
    }
    if (tilePoints != nullptr && tilePointsNum > 0)
    {
        gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#99CCFF' ps 1 ",
            tilePoints, rd::GraphDrawer<T>::POINTS, tilePointsNum);
    }
    if (tileChosenPoints != nullptr && tileChosenPointsNum > 0)
    {
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#000000' ps 1.2 ",
            tileChosenPoints, rd::GraphDrawer<T>::POINTS, tileChosenPointsNum);
    }
    gDrawer.addPlotCmd("'-' w l lt 8 lc rgb '#CB1E1E' lw 1.5 ",
        bounds.data(), rd::GraphDrawer<T>::LINE, (DIM == 2) ? 5 : 16);
    gDrawer.endGraph();
}

template <
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    typename T>
void drawTile(
    int allPointsNum,
    T const * allPoints,
    BoundingBox<2, T> tileBounds,
    int tilePointsNum,
    T const * tilePoints, 
    int tileNeighboursNum,
    T const * tileNeighbours,
    int tileChosenPointsNum,
    T const * tileChosenPoints,
    int tileId,
    std::string devName)
{
    doDrawTile<IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(allPointsNum, allPoints, tileBounds, 
        tilePointsNum, tilePoints, tileNeighboursNum, tileNeighbours, tileChosenPointsNum, 
        tileChosenPoints, tileId, devName);
}

template <
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    typename T>
void drawTile(
    int allPointsNum,
    T const * allPoints,
    BoundingBox<3, T> tileBounds,
    int tilePointsNum,
    T const * tilePoints, 
    int tileNeighboursNum,
    T const * tileNeighbours,
    int tileChosenPointsNum,
    T const * tileChosenPoints,
    int tileId,
    std::string devName)
{
    doDrawTile<IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(allPointsNum, allPoints, tileBounds, 
        tilePointsNum, tilePoints, tileNeighboursNum, tileNeighbours, tileChosenPointsNum, 
        tileChosenPoints, tileId, devName);
}

template <
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    int DIM,
    typename T>
void drawTile(
    int ,
    T const * ,
    BoundingBox<DIM, T> ,
    int ,
    T const * , 
    int ,
    T const * ,
    int ,
    T const * ,
    int ,
    std::string )
{
}

//-------------------------------------------------------------------------------------------
//
//-------------------------------------------------------------------------------------------

template <
    typename TiledTreeT>
__launch_bounds__ (1)
static __global__ void getTreeLeafsNum(
    TiledTreeT *    tree,
    int *           leafsNum)
{
    typedef typename TiledTreeT::NodeT NodeT;
    int lnum = *tree->d_leafCount;

    if (lnum == 0)
    {
        tree->forEachNodePreorder([&lnum](NodeT const * node) {
            if (!node->haveChildren() && !node->empty())
            {
                lnum++;
            }
        });
        
    }
    *leafsNum = lnum;
}

template <typename TiledTreeT>
__launch_bounds__ (1)
static __global__ void collectTreeLeafsPtsCountersKernel(
    TiledTreeT *    d_tree, 
    int             h_treeLeafsNum, 
    int *           d_tilePointsNum, 
    int *           d_tileNeighboursNum, 
    int *           d_tileChosenPointsNum, 
    int *           d_tileSamplesStride, 
    int *           d_tileNeighboursStride,
    int *           d_tileChosenSamplesStride)
{
    typedef typename TiledTreeT::NodeT NodeT;
    int leafCounter = 0;

    d_tree->forEachNodePreorder([&](NodeT const * node){
        // am I a leaf?
        if (!node->haveChildren() && !node->empty())
        {
            assert((leafCounter+1) <= h_treeLeafsNum);
            d_tilePointsNum[leafCounter] = node->pointsCnt;
            d_tileNeighboursNum[leafCounter] = node->neighboursCnt;
            d_tileChosenPointsNum[leafCounter] = node->chosenPointsCnt;
            d_tileSamplesStride[leafCounter] = node->samplesStride;
            d_tileNeighboursStride[leafCounter] = node->neighboursStride;
            d_tileChosenSamplesStride[leafCounter] = node->chosenSamplesStride;
            leafCounter++;
        }
    });
}

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

    // #ifdef RD_DEBUG
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    // {
    //     printf("dataCopyKernel: startOffset: %d, numPoints: %d, inStride: %d, outStride: %d\n",
    //         startOffset, numPoints, inStride, outStride);
    // }
    // #endif

    AgentMemcpyT agent(d_in, d_out);
    agent.copyRange(startOffset, numPoints, inStride, outStride);
}


template <
    int                     DIM,
    rd::DataMemoryLayout    IN_MEM_LAYOUT,
    rd::DataMemoryLayout    OUT_MEM_LAYOUT,
    typename                TiledTreeT,
    typename                T>
__launch_bounds__ (1)
static __global__ void collectTreeLeafsDataKernel(
    TiledTreeT *                    d_tree, 
    int                             h_treeLeafsNum, 
    T **                            d_tileSamplesPtrs, 
    T **                            d_tileNeighboursPtrs, 
    T **                            d_tileChosenSamplesPtrs, 
    rd::gpu::BoundingBox<DIM, T> *  d_tileBounds, 
    int *                           d_tileIds)
{
    typedef typename TiledTreeT::NodeT NodeT;
    int leafCounter = 0;

    cudaStream_t copySamplesStream, copyNeighboursStream, copyChosenSamplesStream;
    rdDevCheckCall(cudaStreamCreateWithFlags(&copySamplesStream, cudaStreamNonBlocking));
    rdDevCheckCall(cudaStreamCreateWithFlags(&copyNeighboursStream, cudaStreamNonBlocking));
    rdDevCheckCall(cudaStreamCreateWithFlags(&copyChosenSamplesStream, cudaStreamNonBlocking));

    d_tree->forEachNodePreorder([&](NodeT const * node)
    {
        // am I a leaf?
        if (!node->haveChildren() && !node->empty())
        {
            assert((leafCounter+1) <= h_treeLeafsNum);

            dim3 copySamplesGrid(1), copyNeighboursGrid(1), copyChosenSamplesGrid(1);
            copySamplesGrid.x = (((node->pointsCnt + 7) / 8) + 127) / 128;
            copyNeighboursGrid.x = (((node->neighboursCnt + 7) / 8) + 127) / 128;
            copyChosenSamplesGrid.x = (((node->chosenPointsCnt + 7) / 8) + 127) / 128;

            // printf(">>>> [node id: %d] Invoke copy samples<<<%d, 128,0,%p>>>\n",
            //         node->id, copySamplesGrid.x, copySamplesStream);
            if (node->pointsCnt > 0)
            {
                dataCopyKernel<DIM, IN_MEM_LAYOUT><<<copySamplesGrid, 128, 0, copySamplesStream>>>(
                    node->samples, d_tileSamplesPtrs[leafCounter], node->pointsCnt, 0, 
                    node->samplesStride, 
                    // we're copying to a flattened device memory region, without aligned
                    // memory, with minimal capacity able to contain data
                    (IN_MEM_LAYOUT == COL_MAJOR) ? node->pointsCnt : DIM);
                rdDevCheckCall(cudaPeekAtLastError());
                #ifdef RD_DEBUG
                    rdDevCheckCall(cudaDeviceSynchronize());
                #endif
            }

            // printf(">>>> [node id: %d] Invoke copy neighbours<<<%d, 128,0,%p>>>\n",
            //         node->id, copyNeighboursGrid.x, copyNeighboursStream);
            if (node->neighboursCnt > 0)
            {
                dataCopyKernel<DIM, IN_MEM_LAYOUT><<<copyNeighboursGrid, 128, 0, 
                        copyNeighboursStream>>>(
                    node->neighbours, d_tileNeighboursPtrs[leafCounter], node->neighboursCnt, 0, 
                    node->neighboursStride, 
                    (IN_MEM_LAYOUT == COL_MAJOR) ? node->neighboursCnt : DIM);
                rdDevCheckCall(cudaPeekAtLastError());
                #ifdef RD_DEBUG
                    rdDevCheckCall(cudaDeviceSynchronize());
                #endif
            }

            // printf(">>>> [node id: %d] Invoke copy chosen samples<<<%d, 128,0,%p>>>\n",
            //         node->id, copyChosenSamplesGrid.x, copyChosenSamplesStream);
            if (node->chosenPointsCnt > 0)
            {
                dataCopyKernel<DIM, OUT_MEM_LAYOUT><<<copyChosenSamplesGrid, 128, 0, 
                        copyChosenSamplesStream>>>(
                    node->chosenSamples, d_tileChosenSamplesPtrs[leafCounter], 
                    node->chosenPointsCnt, 0, node->chosenSamplesStride, 
                    (OUT_MEM_LAYOUT == COL_MAJOR) ? node->chosenPointsCnt : DIM);
                rdDevCheckCall(cudaPeekAtLastError());
                #ifdef RD_DEBUG
                    rdDevCheckCall(cudaDeviceSynchronize());
                #endif
            }

            d_tileBounds[leafCounter] = node->bounds;
            d_tileIds[leafCounter] = node->id;

            leafCounter++;
        }
    });

    rdDevCheckCall(cudaDeviceSynchronize());

    rdDevCheckCall(cudaStreamDestroy(copySamplesStream));
    rdDevCheckCall(cudaStreamDestroy(copyNeighboursStream));
    rdDevCheckCall(cudaStreamDestroy(copyChosenSamplesStream));
}

template <
    typename TiledTreeT>
__launch_bounds__ (1)
static __global__ void deleteTiledTree(
    TiledTreeT *          tree)
{
    tree->~TiledTreeT();
}


//------------------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------------------
template <
    int                 DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT,
    typename            TiledTreeT,
    typename            T,
    class               EnableT = void>
class TreeDrawer
{
public:
    TreeDrawer(
        TiledTreeT * ,
        T const * ,
        int,
        int)
    {}

public:
    void collectTreeData()
    {
    }

    void drawBounds()
    {
    }

    void drawEachTile()
    {
    }
    
    void releaseTree()
    {
    }
};


template <
    int                 DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT,
    typename            TiledTreeT,
    typename            T>
class TreeDrawer<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, TiledTreeT, T,
    typename std::enable_if<
        std::is_same<
            typename std::conditional<DIM == 2 || DIM == 3, std::true_type, std::false_type>::type,
            std::true_type>::value
        >::type>
{
    std::string devName;

    T * h_inputPoints;
    T const * d_inputPoints;
    int pointsNum;

    TiledTreeT * d_tree;
    int h_treeLeafsNum;
    int * d_treeLeafsNum;

    int *d_tilePointsNum = nullptr;
    int *d_tileNeighboursNum = nullptr;
    int *d_tileSamplesStride = nullptr;
    int *d_tileNeighboursStride = nullptr;
    int *d_tileChosenPointsNum = nullptr;
    int *d_tileChosenSamplesStride = nullptr;

    int *h_tilePointsNum = nullptr;
    int *h_tileNeighboursNum = nullptr;
    int *h_tileSamplesStride = nullptr;
    int *h_tileNeighboursStride = nullptr;
    int *h_tileChosenPointsNum = nullptr;
    int *h_tileChosenSamplesStride = nullptr;

    int *d_tileIds = nullptr;
    int *h_tileIds = nullptr;

    T **d_tileSamplesPtrs = nullptr;
    T **d_tileNeighboursPtrs = nullptr;
    T **d_tileChosenSamplesPtrs = nullptr;
    T **hd_tileNeighboursPtrs = nullptr;
    T **hd_tileSamplesPtrs = nullptr;
    T **hd_tileChosenSamplesPtrs = nullptr;
    T **h_tileSamplesPtrs = nullptr;
    T **h_tileNeighboursPtrs = nullptr;
    T **h_tileChosenSamplesPtrs = nullptr;

    rd::gpu::BoundingBox<DIM,T> *d_tileBounds = nullptr;
    rd::gpu::BoundingBox<DIM,T> *h_tileBounds = nullptr;

public:
    TreeDrawer(
        TiledTreeT * tree,
        T const * d_inputPoints,
        int pointsNum,
        int stride)
    :
        d_tree(tree),
        d_inputPoints(d_inputPoints),
        pointsNum(pointsNum)
    {
        int devId;
        cudaGetDevice(&devId);
        // set device name for logging and drawing purposes
        cudaDeviceProp devProp;
        checkCudaErrors(cudaGetDeviceProperties(&devProp, devId));
        devName = devProp.name;

        checkCudaErrors(cudaMalloc(&d_treeLeafsNum, sizeof(int)));
        h_inputPoints = new T[pointsNum * DIM];

        if (IN_MEM_LAYOUT == ROW_MAJOR)
        {
            rdMemcpy<ROW_MAJOR, ROW_MAJOR, cudaMemcpyDeviceToHost>(
                h_inputPoints, d_inputPoints, DIM, pointsNum, stride, DIM);
        }
        else if (IN_MEM_LAYOUT == COL_MAJOR)
        {
            rd::gpu::rdMemcpy2D<ROW_MAJOR, COL_MAJOR, cudaMemcpyDeviceToHost>(
                h_inputPoints, d_inputPoints, pointsNum, DIM, DIM * sizeof(T),
                stride * sizeof(T));
        }
        else
        {
            throw std::runtime_error("Unsupported memory layout!");
        }
        checkCudaErrors(cudaDeviceSynchronize());

        h_treeLeafsNum = 0;

        collectTreeData();
        checkCudaErrors(cudaDeviceSynchronize());
    }

    ~TreeDrawer()
    {
        clear();
        if (h_inputPoints != nullptr) delete[] h_inputPoints;
        if (d_treeLeafsNum != nullptr) checkCudaErrors(cudaFree(d_treeLeafsNum));
    }
    
private:
    void clearDataContainers()
    {
        for (int i = 0; i < h_treeLeafsNum; ++i)
        {
            if (hd_tileSamplesPtrs != nullptr && hd_tileSamplesPtrs[i] != nullptr) checkCudaErrors(cudaFree(hd_tileSamplesPtrs[i]));
            if (hd_tileNeighboursPtrs != nullptr && hd_tileNeighboursPtrs[i] != nullptr) checkCudaErrors(cudaFree(hd_tileNeighboursPtrs[i]));
            if (hd_tileChosenSamplesPtrs != nullptr && hd_tileChosenSamplesPtrs[i] != nullptr) checkCudaErrors(cudaFree(hd_tileChosenSamplesPtrs[i]));
            if (h_tileSamplesPtrs != nullptr && h_tileSamplesPtrs[i] != nullptr) delete[] h_tileSamplesPtrs[i];
            if (h_tileNeighboursPtrs != nullptr && h_tileNeighboursPtrs[i] != nullptr) delete[] h_tileNeighboursPtrs[i];
            if (h_tileChosenSamplesPtrs != nullptr && h_tileChosenSamplesPtrs[i] != nullptr) delete[] h_tileChosenSamplesPtrs[i];
        }

        if (hd_tileSamplesPtrs != nullptr)  delete[] hd_tileSamplesPtrs;
        if (hd_tileNeighboursPtrs != nullptr)  delete[] hd_tileNeighboursPtrs;
        if (hd_tileChosenSamplesPtrs != nullptr)  delete[] hd_tileChosenSamplesPtrs;
        if (h_tileSamplesPtrs != nullptr)  delete[] h_tileSamplesPtrs;
        if (h_tileNeighboursPtrs != nullptr)  delete[] h_tileNeighboursPtrs;
        if (h_tileChosenSamplesPtrs != nullptr)  delete[] h_tileChosenSamplesPtrs;
        if (d_tileSamplesPtrs != nullptr)  checkCudaErrors(cudaFree(d_tileSamplesPtrs));
        if (d_tileNeighboursPtrs != nullptr)  checkCudaErrors(cudaFree(d_tileNeighboursPtrs));
        if (d_tileChosenSamplesPtrs != nullptr)  checkCudaErrors(cudaFree(d_tileChosenSamplesPtrs));
    }

    void clearCounterContainers()
    {
        if (d_tilePointsNum != nullptr) checkCudaErrors(cudaFree(d_tilePointsNum));
        if (d_tileNeighboursNum != nullptr) checkCudaErrors(cudaFree(d_tileNeighboursNum));
        if (d_tileChosenPointsNum != nullptr) checkCudaErrors(cudaFree(d_tileChosenPointsNum));
        if (d_tileSamplesStride != nullptr) checkCudaErrors(cudaFree(d_tileSamplesStride));
        if (d_tileNeighboursStride != nullptr) checkCudaErrors(cudaFree(d_tileNeighboursStride));
        if (d_tileChosenSamplesStride != nullptr) checkCudaErrors(cudaFree(d_tileChosenSamplesStride));

        if (h_tilePointsNum != nullptr) delete[] h_tilePointsNum;
        if (h_tileNeighboursNum != nullptr) delete[] h_tileNeighboursNum;
        if (h_tileChosenPointsNum != nullptr) delete[] h_tileChosenPointsNum;
        if (h_tileSamplesStride != nullptr) delete[] h_tileSamplesStride;
        if (h_tileNeighboursStride != nullptr) delete[] h_tileNeighboursStride;
        if (h_tileChosenSamplesStride != nullptr) delete[] h_tileChosenSamplesStride;
        
        if (d_tileBounds != nullptr) checkCudaErrors(cudaFree(d_tileBounds));
        if (h_tileBounds != nullptr) delete[] h_tileBounds;

        if (d_tileIds != nullptr) checkCudaErrors(cudaFree(d_tileIds));
        if (h_tileIds != nullptr) delete[] h_tileIds;
    }

    void clear()
    {
        clearDataContainers();
        clearCounterContainers();
    }

public:
    void collectTreeData()
    {
        std::cout << "Invoking getTreeLeafsNum kernel" << std::endl;
    
        getTreeLeafsNum<TiledTreeT><<<1, 1>>>(d_tree, d_treeLeafsNum);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        int newLeafNum = 0;
        checkCudaErrors(cudaMemcpy(&newLeafNum, d_treeLeafsNum, sizeof(int),
            cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        if (newLeafNum != h_treeLeafsNum && newLeafNum > 0)
        {
            h_treeLeafsNum = newLeafNum;
            std::cout << "Got [" << h_treeLeafsNum << "] tree leafs!" << std::endl;

            clearCounterContainers();
            checkCudaErrors(cudaMalloc(&d_tileBounds, 
            h_treeLeafsNum * sizeof(rd::gpu::BoundingBox<DIM,T>)));
            h_tileBounds = new rd::gpu::BoundingBox<DIM, T>[h_treeLeafsNum];

            checkCudaErrors(cudaMalloc(&d_tilePointsNum, h_treeLeafsNum * sizeof(int)));
            checkCudaErrors(cudaMalloc(&d_tileNeighboursNum, h_treeLeafsNum * sizeof(int)));
            checkCudaErrors(cudaMalloc(&d_tileChosenPointsNum, h_treeLeafsNum * sizeof(int)));
            
            checkCudaErrors(cudaMalloc(&d_tileSamplesStride, h_treeLeafsNum * sizeof(int)));
            checkCudaErrors(cudaMalloc(&d_tileNeighboursStride, h_treeLeafsNum * sizeof(int)));
            checkCudaErrors(cudaMalloc(&d_tileChosenSamplesStride, h_treeLeafsNum * sizeof(int)));

            checkCudaErrors(cudaMalloc(&d_tileIds, h_treeLeafsNum * sizeof(int)));

            h_tilePointsNum = new int[h_treeLeafsNum];
            h_tileNeighboursNum = new int[h_treeLeafsNum];
            h_tileChosenPointsNum = new int[h_treeLeafsNum];

            h_tileSamplesStride = new int[h_treeLeafsNum];
            h_tileNeighboursStride = new int[h_treeLeafsNum];
            h_tileChosenSamplesStride = new int[h_treeLeafsNum];

            h_tileIds = new int[h_treeLeafsNum];
        }

        std::cout << "Invoking collectTreeLeafsPtsCountersKernel" << std::endl;
        collectTreeLeafsPtsCountersKernel<TiledTreeT><<<1,1>>>(d_tree, h_treeLeafsNum,
            d_tilePointsNum, d_tileNeighboursNum, d_tileChosenPointsNum, d_tileSamplesStride, 
            d_tileNeighboursStride, d_tileChosenSamplesStride);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        std::cout << "Copying device pts counters to host..." << std::endl;
        // copy from device pts counters
        checkCudaErrors(cudaMemcpy(h_tilePointsNum, d_tilePointsNum, 
            h_treeLeafsNum * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_tileNeighboursNum, d_tileNeighboursNum, 
            h_treeLeafsNum * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_tileChosenPointsNum, d_tileChosenPointsNum, 
            h_treeLeafsNum * sizeof(int), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaMemcpy(h_tileSamplesStride, d_tileSamplesStride, 
            h_treeLeafsNum * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_tileNeighboursStride, d_tileNeighboursStride, 
            h_treeLeafsNum * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_tileChosenSamplesStride, d_tileChosenSamplesStride, 
            h_treeLeafsNum * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        // allocate host and device containers
        clearDataContainers();
        
        checkCudaErrors(cudaMalloc(&d_tileSamplesPtrs, h_treeLeafsNum * sizeof(T*)));
        checkCudaErrors(cudaMalloc(&d_tileNeighboursPtrs, h_treeLeafsNum * sizeof(T*)));
        checkCudaErrors(cudaMalloc(&d_tileChosenSamplesPtrs, h_treeLeafsNum * sizeof(T*)));

        hd_tileSamplesPtrs = new T*[h_treeLeafsNum];
        hd_tileNeighboursPtrs = new T*[h_treeLeafsNum];
        hd_tileChosenSamplesPtrs = new T*[h_treeLeafsNum];
        h_tileSamplesPtrs = new T*[h_treeLeafsNum];
        h_tileNeighboursPtrs = new T*[h_treeLeafsNum];
        h_tileChosenSamplesPtrs = new T*[h_treeLeafsNum];

        for (int i = 0; i < h_treeLeafsNum; ++i)
        {
            checkCudaErrors(cudaMalloc(hd_tileSamplesPtrs+i, 
                h_tilePointsNum[i] * DIM * sizeof(T)));
            h_tileSamplesPtrs[i] = new T[h_tilePointsNum[i] * DIM];

            checkCudaErrors(cudaMalloc(hd_tileNeighboursPtrs+i, 
                h_tileNeighboursNum[i] * DIM * sizeof(T)));
            h_tileNeighboursPtrs[i] = new T[h_tileNeighboursNum[i] * DIM];

            checkCudaErrors(cudaMalloc(hd_tileChosenSamplesPtrs+i, 
                h_tileChosenPointsNum[i] * DIM * sizeof(T)));
            h_tileChosenSamplesPtrs[i] = new T[h_tileChosenPointsNum[i] * DIM];
        }

        checkCudaErrors(cudaMemcpy(d_tileSamplesPtrs, hd_tileSamplesPtrs, 
            h_treeLeafsNum * sizeof(T*), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_tileNeighboursPtrs, hd_tileNeighboursPtrs, 
            h_treeLeafsNum * sizeof(T*), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_tileChosenSamplesPtrs, hd_tileChosenSamplesPtrs, 
            h_treeLeafsNum * sizeof(T*), cudaMemcpyHostToDevice));

        // collect data on device
        std::cout << "Invoking collectTreeLeafsDataKernel" << std::endl;
        collectTreeLeafsDataKernel<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, TiledTreeT><<<1,1>>>(
            d_tree, h_treeLeafsNum, d_tileSamplesPtrs,  d_tileNeighboursPtrs, 
            d_tileChosenSamplesPtrs, d_tileBounds, d_tileIds);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // copy data back to host
        std::cout << "Copying tile data to host.." << std::endl;
        checkCudaErrors(cudaMemcpy(h_tileBounds, d_tileBounds, 
            h_treeLeafsNum * sizeof(rd::gpu::BoundingBox<DIM,T>), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_tileIds, d_tileIds, h_treeLeafsNum * sizeof(int),
            cudaMemcpyDeviceToHost));

        for (int i = 0; i < h_treeLeafsNum; ++i)
        {
            checkCudaErrors(cudaMemcpy(h_tileSamplesPtrs[i], hd_tileSamplesPtrs[i], 
                h_tilePointsNum[i] * DIM * sizeof(T), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_tileNeighboursPtrs[i], hd_tileNeighboursPtrs[i],
                h_tileNeighboursNum[i] * DIM * sizeof(T), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_tileChosenSamplesPtrs[i], hd_tileChosenSamplesPtrs[i],
                h_tileChosenPointsNum[i] * DIM * sizeof(T), cudaMemcpyDeviceToHost));
        }

        // transpose if needed
        if (IN_MEM_LAYOUT == COL_MAJOR)
        {
            std::cout << "Transpose input tile data... " << std::endl;
            #ifdef RD_USE_OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int i = 0; i < h_treeLeafsNum; ++i)
            {
                if(h_tilePointsNum[i] > 0)
                {
                    rd::transposeInPlace(h_tileSamplesPtrs[i], h_tileSamplesPtrs[i] +
                        h_tilePointsNum[i]*DIM, h_tilePointsNum[i]);
                }

                if(h_tileNeighboursNum[i] > 0)
                {
                    rd::transposeInPlace(h_tileNeighboursPtrs[i], h_tileNeighboursPtrs[i] + 
                        h_tileNeighboursNum[i]*DIM, h_tileNeighboursNum[i]);
                }
            }
        }

        if (OUT_MEM_LAYOUT == COL_MAJOR)
        {
            std::cout << "Transpose output tile data... " << std::endl;
            #ifdef RD_USE_OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int i = 0; i < h_treeLeafsNum; ++i)
            {
                if (h_tileChosenPointsNum[i] > 0)
                {
                    rd::transposeInPlace(h_tileChosenSamplesPtrs[i], 
                        h_tileChosenSamplesPtrs[i] + h_tileChosenPointsNum[i]*DIM, 
                        h_tileChosenPointsNum[i]);
                }
            }
        }
    }

    void drawBounds()
    {
        std::cout << "Drawing all tile bounds... " << std::endl;

        drawAllTileBounds<IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(pointsNum, h_treeLeafsNum, h_tileBounds,
            h_tilePointsNum, h_tileSamplesPtrs, devName);
    }

    void drawEachTile()
    {
        std::cout << "Drawing each tile with input points... " << std::endl;

        #ifdef RD_USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < h_treeLeafsNum; ++i)
        {
            drawTile<IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(pointsNum, h_inputPoints, h_tileBounds[i], 
                h_tilePointsNum[i], h_tileSamplesPtrs[i], h_tileNeighboursNum[i], 
                h_tileNeighboursPtrs[i], h_tileChosenPointsNum[i], h_tileChosenSamplesPtrs[i],
                h_tileIds[i], devName);
        }
    }

    void releaseTree()
    {
        std::cout << "Invoking deleteTiledTree kernel" << std::endl;
        deleteTiledTree<TiledTreeT><<<1, 1>>>(d_tree);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
};

} // namespace util
} // namespace tiled
} // namespace gpu
} // namespace rd
