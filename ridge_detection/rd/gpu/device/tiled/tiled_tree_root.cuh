/**
 * @file tiled_tree_root.cuh
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

#include "rd/utils/memory.h"

#include <utility>

namespace rd
{
namespace gpu
{
namespace tiled
{

/**
 * @brief      Tiled tree root data structure.
 *
 * @tparam     NodeT  Data type of root children structure
 *
 * @paragraph Root can have many children (nodes).
 */
template <typename    NodeT>
struct TiledTreeRoot
{
    int childrenCount;
    NodeT *children;

    #ifdef RD_DRAW_TREE_TILES
    unsigned int * d_referenceCount;
    #endif

    __device__ __forceinline__ TiledTreeRoot()
    :
        childrenCount(0),
        children(nullptr)
    {
        #ifdef RD_DRAW_TREE_TILES
            d_referenceCount = new unsigned int();
            assert(d_referenceCount != nullptr);
            *d_referenceCount = 1;
        #endif
        #ifdef RD_DEBUG
            printf("TiledTreeRoot(): childrenCount : %d\n", childrenCount);
        #endif
    }

    __device__ __forceinline__ TiledTreeRoot(int chCnt)
    :
        childrenCount(chCnt)
    {
        children = new NodeT[childrenCount];
        assert(children != nullptr);
    }

    __device__ __forceinline__ void init(int childrenNum)
    {
        clear();
        childrenCount = childrenNum;
        children = new NodeT[childrenCount];
        assert(children != nullptr);
    }

    __device__ __forceinline__ TiledTreeRoot(TiledTreeRoot const & rhs) 
    :
        childrenCount(rhs.childrenCount),
        children(rhs.children)
    {
        #ifdef RD_DEBUG
        printf("TiledTreeRoot::TiledTreeRoot(const &)\n");
        #endif
    #ifdef RD_DRAW_TREE_TILES
        d_referenceCount = rhs.d_referenceCount;
        (*d_referenceCount)++;
        #ifdef RD_DEBUG
            printf("TiledTreeRoot[copy constr], *d_referenceCount: %d\n", *d_referenceCount);
        #endif
    #else
        #ifdef RD_DEBUG
            printf("TiledTreeRoot[copy constr], \n");
        #endif
    #endif
    }

    __device__ __forceinline__ TiledTreeRoot & operator=(TiledTreeRoot const & rhs) 
    {
        #ifdef RD_DEBUG
        printf("TiledTreeRoot::operator=(const &)\n");
        #endif
        childrenCount = rhs.childrenCount;
        children = rhs.children;
    #ifdef RD_DRAW_TREE_TILES
        d_referenceCount = rhs.d_referenceCount;
        (*d_referenceCount)++;
        #ifdef RD_DEBUG
            printf("TiledTreeRoot[assing op], *d_referenceCount: %d\n", *d_referenceCount);
        #endif
    #else
        #ifdef RD_DEBUG
            printf("TiledTreeRoot[assing op], \n");
        #endif
    #endif
        return *this;
    }

    __device__ __forceinline__ TiledTreeRoot(TiledTreeRoot && rhs)
    :
        childrenCount(rhs.childrenCount),
        children(rhs.children)
    #ifdef RD_DRAW_TREE_TILES
        ,
        d_referenceCount(rhs.d_referenceCount)
    #endif
    {
        #ifdef RD_DEBUG
        printf("TiledTreeRoot::TiledTreeRoot(&&)\n");
        #endif

        rhs.children = nullptr;
        rhs.childrenCount = 0;
        #ifdef RD_DRAW_TREE_TILES
            // (*d_referenceCount)++;
            rhs.d_referenceCount = nullptr;
        #endif
    }

    __device__ __forceinline__ TiledTreeRoot & operator=(TiledTreeRoot && rhs)
    {
        #ifdef RD_DEBUG
        printf("TiledTreeRoot::operator=(&&)\n");
        #endif
        children = rhs.children;
        childrenCount = rhs.childrenCount;
        #ifdef RD_DRAW_TREE_TILES
            d_referenceCount = rhs.d_referenceCount;
            // (*d_referenceCount)++;
            rhs.d_referenceCount = nullptr;
        #endif
        rhs.children = nullptr;
        rhs.childrenCount = 0;

        return *this;
    }

    template <
        DataMemoryLayout IN_MEM_LAYOUT,
        DataMemoryLayout OUT_MEM_LAYOUT>
    __device__ __forceinline__ TiledTreeRoot clone() const
    {
        #ifdef RD_DEBUG
        printf("TiledTreeRoot::clone()\n");
        #endif
        if (empty())
        {
            return TiledTreeRoot();
        }

        TiledTreeRoot rootClone(childrenCount);
        #ifdef RD_DRAW_TREE_TILES
            (rootClone.d_referenceCount)++;
        #endif

        for (int i = 0; i < childrenCount; ++i)
        {
            #ifdef RD_DEBUG
            printf("TiledTreeRoot::clone() -> clone children %d\n", i);
            #endif
            children[i].clone<IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(rootClone.children + i);
        }

        // for (int i = 0; i < childrenCount; ++i)
        // {
        //     rootClone.children[i].printRecursive();
        // }
        // #ifdef RD_DEBUG
        //     printf("TiledTreeRoot::clone() -> End!\n");
        // #endif

        return rootClone;
    }

    __device__ __forceinline__ ~TiledTreeRoot()
    {
        #if defined(RD_DRAW_TREE_TILES)
        if (d_referenceCount != nullptr)
        {
            (*d_referenceCount)--;
            #ifdef RD_DEBUG
            printf("~TiledTreeRoot(), childrenCount: %d, *d_referenceCount: %d\n",
                childrenCount, *d_referenceCount);
            #endif
            if (*d_referenceCount == 0)
            {
                delete d_referenceCount;
                clear();
            }
        }
        #ifdef RD_DEBUG
        printf("~TiledTreeRoot(), childrenCount: %d, d_referenceCount: %p\n",
            childrenCount, d_referenceCount);
        #endif
        #else
        #ifdef RD_DEBUG
            printf("~TiledTreeRoot(), childrenCount: %d\n", childrenCount);
        #endif
        clear();
        #endif
            
    }

    __device__ __forceinline__ bool empty() const
    {
        return childrenCount == 0 && children == nullptr;
    }

    __device__ __forceinline__ void clear()
    {
        #ifdef RD_DEBUG
            printf("TiledTreeRoot::clear() \n", childrenCount);
        #endif

        if (!empty())
        {
            delete[] children;
            children = nullptr;
        }
    }

};


} // end namespace tiled
} // end namespace gpu
} // end namespace rd