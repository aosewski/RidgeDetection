/**
 * @file grouped_tile_tree.inl
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is supervised by prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */

#include "rd/utils/graph_drawer.hpp"
#include <sstream>
#include <stdexcept>

#include <cmath>
#include <algorithm>
#include <functional>

#include <iostream>
#include <utility>

#ifdef RD_USE_OPENMP
    #include <omp.h>
#endif

namespace rd
{
namespace tiled
{ 

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>  
TiledGroupTreeBase<BUILD_POLICY, T>::TiledGroupTreeBase()
    : maxTileCapacity(0), sphereRadius(0), cpuThreads(0)
{
    #if defined(RD_USE_OPENMP)    
        cpuThreads = omp_get_num_procs();
        omp_set_num_threads(cpuThreads);
    #endif
}

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>  
TiledGroupTreeBase<BUILD_POLICY, T>::~TiledGroupTreeBase()
{
    leafs_.clear();
    clear(root_);
}

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>  
void TiledGroupTreeBase<BUILD_POLICY, T>::initTreeRoot()
{
    if (root_.use_count() == 0)
    {
        #ifdef RD_DEBUG
            std::cout << ">>>> allocate root_ and assign sentinel_" << std::endl;
        #endif
        root_ = std::make_shared<Node>();
        sentinel_.root = root_;
        #ifdef RD_DEBUG
            std::cout << "<<<<<" << std::endl;
        #endif
    }
    else
    {
        #ifdef RD_DEBUG
            std::cout << ">>>> reset root_ and assign sentinel_" << std::endl;
        #endif
        root_.reset(new Node());
        sentinel_.root.reset();
        sentinel_.root = root_;
        #ifdef RD_DEBUG
            std::cout << "<<<<<" << std::endl;
        #endif
    }
}

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>
void TiledGroupTreeBase<BUILD_POLICY, T>::buildTree(
    T const *                   samples,
    size_t                      pointCnt,
    size_t                      dim,
    std::vector<size_t> const & initTileCnt)
{
    typedef std::vector<std::pair<T const *, size_t>> pairVec;

    initTreeRoot();
    BoundingBox<T> globalBbox(samples, pointCnt, dim);
    globalBbox.calcDistances();

    std::vector<Tile*> tiles;
    createTiles(pairVec{std::make_pair(samples, pointCnt)}, 
        dim, 
        tiles, 
        initTileCnt, 
        globalBbox);

    // create tiles
    for (Tile *t : tiles)
    {
        addNode(root_, t, 1);
    }

    collectLeafs();
}

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>
void TiledGroupTreeBase<BUILD_POLICY, T>::createTiles(
    std::vector<std::pair<T const *, size_t>> const &   samples,
    size_t                                              dim,
    std::vector<Tile*> &                                outTtiles,
    std::vector<size_t> const &                         tileCnt,
    BoundingBox<T> const &                              inSamplesBbox)
{
    allocTiles(dim, outTtiles, tileCnt, inSamplesBbox);
    partitionSamples(samples, dim, outTtiles);
    reduceTilesAndAllocChosenSamples(outTtiles);
}


/**
 * @brief      Allocate tile objects and initializes its bounds.
 *
 * @param[in]  dim            Tile dimension.
 * @param[out] outTiles       Destination vector for storing tiles.
 * @param      tileCnt        Vector with @p dim elements, each denoting initial tile count in respective dimension
 * @param[in]  inSamplesBbox  Bounding box of region we divide into tiles.
 *
 * @tparam     BUILD_POLICY   Partition, find neighbours algorithm
 * @tparam     T              Point's coordinates data type
 */
template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>  
void TiledGroupTreeBase<BUILD_POLICY, T>::allocTiles(
    size_t                          dim,
    std::vector<Tile*> &            outTiles,
    std::vector<size_t> const &     tileCnt,
    BoundingBox<T> const &          inSamplesBbox)
{
    const size_t nTiles = std::accumulate(tileCnt.begin(), tileCnt.end(),
                         1, std::multiplies<size_t>());
    
    std::vector<T> boundStep(dim);
    std::vector<int> tileCoords(dim, -1);

    auto nextSpatialIdx = [&tileCnt, &tileCoords]()
    {
        // Iterate through tile coordinates and every time this function is called
        // increment each value (modulo number of tiles in respective dimension)
        for (size_t i = 0; i < tileCnt.size(); ++i)
        {
            tileCoords[i]++;
            if ((tileCoords[i] % tileCnt[i]) == 0)
            {
                tileCoords[i] = 0;
            }
            else
                break;
        }
    };

    auto calcTileBounds = [&boundStep, &tileCoords, &inSamplesBbox, tileCnt](BoundingBox<T> &b)
    {
        b.dim = boundStep.size();
        b.bbox = new T[2 * b.dim];
        b.dist = new T[b.dim];
        for (size_t i = 0; i < b.dim; ++i)
        {
            b.min(i) = inSamplesBbox.min(i) + tileCoords[i] * boundStep[i];
            b.max(i) = inSamplesBbox.min(i) + (tileCoords[i] + 1) * boundStep[i];
            // if we are the last tile in respective dimension, set max as a parent max. This
            // will result in slightly larger tile than other (in this dim).
            // This enable to correctly cover all space. Otherwise, (due to the floating point
            // rounding error we would miss some points, because sum of tile dist won't be equal
            // to parent dist in respective dimension)
            if (tileCoords[i] == (int)(tileCnt[i]-1))
            {
                b.max(i) = inSamplesBbox.max(i);
            }
        }
        b.calcDistances();
    };

    // calculate tiles' bounds step for each dim
    for (size_t k = 0; k < dim; ++k)
        boundStep[k] = inSamplesBbox.dist[k] / static_cast<T>(tileCnt[k]);

    outTiles.clear();
    // create tiles
    for (size_t i = 0; i < nTiles; ++i)
    {
        Tile *t = new Tile();
        t->dim = dim;
        nextSpatialIdx();
        calcTileBounds(t->bounds);
        outTiles.push_back(t);
    }             
}

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>  
void TiledGroupTreeBase<BUILD_POLICY, T>::reduceTilesAndAllocChosenSamples(
    std::vector<Tile*> & outTiles)
{
    // remove empty tiles and with only one sample as they introduce little information
    for (auto it = outTiles.begin(); it != outTiles.end(); )
    {
        /*
         *  TODO: mechanizm łączenia kafelków w przypadku zbyt małej ilości próbek wewnątrz
         */
        if ((*it)->pointsCnt == 0 || (*it)->pointsCnt == 1)
        {
            delete (*it);
            it = outTiles.erase(it);
        }
        else 
        {
            // calculate bounds for each tile
            // and alocate memory for chosen samples
            if ((*it)->pointsCnt < maxTileCapacity)
            {
                (*it)->chosenPointsCapacity = (*it)->bounds.countSpheresInside(sphereRadius);
                (*it)->chosenSamples.resize((*it)->chosenPointsCapacity * (*it)->dim);
            }
            it++;
        }
    }
}

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>  
void TiledGroupTreeBase<BUILD_POLICY, T>::addNode(
        SptrNode &  parent,
        Tile *      tile,
        size_t      treeLevel)
{
    #ifdef RD_DEBUG
        std::cout << "------ add tile: ------------" << std::endl;
        tile->print();
        std::cout << "-----------------------------" << std::endl;
    #endif

    if (tile->pointsCnt < maxTileCapacity)
    {
        #ifdef RD_DEBUG
            std::cout << std::string(treeLevel*4, '>') << "addNode()" << std::endl;
        #endif
        SptrNode node = std::make_shared<Node>();
        node->parent = parent;
        node->data.reset(tile);
        #ifdef RD_DEBUG
            node->treeLevel = treeLevel;
        #endif    
        parent->children.push_back(node);
        #ifdef RD_DEBUG
            node->print();
            std::cout << std::string(treeLevel*4, '<') << "addNode()" << std::endl;
        #endif
    }
    else
    {
        #ifdef RD_DEBUG
            std::cout << std::string(treeLevel*4, '>') << "addNode() [subdivided]" << std::endl;
        #endif
        SptrNode node = std::make_shared<Node>();
        node->parent = parent;
        node->data.reset(tile);
        #ifdef RD_DEBUG
        node->treeLevel = treeLevel;
        #endif
        parent->children.push_back(node);

        std::vector<Tile*> subTiles;
        subdivideTile(tile, subTiles, treeLevel);

        node->data->samples.clear();
        node->data->chosenSamples.clear();

        for (Tile *t : subTiles)
        {
            addNode(node, t, treeLevel+1);
        }
        #ifdef RD_DEBUG
            node->print();
            std::cout << std::string(treeLevel*4, '<') << "addNode() [subdivided]" << std::endl;
        #endif
    }
}

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>
void TiledGroupTreeBase<BUILD_POLICY, T>::subdivideTile(
        Tile const *            tile,
        std::vector<Tile*> &    subTiles,
        size_t                  treeLevel)
{
    /*
     * XXX: As for now, I'm using simple solution, that is half splitting.
     * In subsequent versions this should be parameterized
     */

    // halft split tile
    // so we have 2 tiles in (tile->treeLevel%dim) dimension,
    // and 1 in other dims
    std::vector<size_t> tileCnt(tile->dim,1);
    tileCnt[(treeLevel)%tile->dim] = 2;
    
    createTiles(tile->getAllSamples(),
        tile->dim, 
        subTiles, 
        tileCnt, 
        tile->bounds);
}

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>  
    template <typename UnaryFunction>
void TiledGroupTreeBase<BUILD_POLICY, T>::forEachNodePreorder(
    UnaryFunction const &f)
{
    visitNode(root_, f);
}

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>  
    template <typename UnaryFunction>
void TiledGroupTreeBase<BUILD_POLICY, T>::visitNode(
    SptrNode node,
    UnaryFunction const &f)
{
    f(node);
    for (SptrNode n : node->children)
        visitNode(n, f);
}

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>  
    template <typename UnaryFunction>
void TiledGroupTreeBase<BUILD_POLICY, T>::forEachLeaf(UnaryFunction const &f)
{
    #if defined(RD_USE_OPENMP)
    #pragma omp parallel for num_threads(cpuThreads), schedule(static)
    #endif
    for (size_t k = 0; k < leafs_.size(); ++k)
    {
        // #if defined(RD_USE_OPENMP) && defined(RD_DEBUG)
        //     printf("omp_tid: %d\n", omp_get_thread_num());
        // #endif
        f(leafs_[k]);
    }
}

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>  
void TiledGroupTreeBase<BUILD_POLICY, T>::clear(SptrNode &node)
{
     if (node)
     {
        node->clear();
     }
}

template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>  
void TiledGroupTreeBase<BUILD_POLICY, T>::print() const
{
    root_->printRecursive();
}


template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>  
void TiledGroupTreeBase<BUILD_POLICY, T>::collectLeafs()
{
    forEachNodePreorder([this](SptrNode n)
    {
        // am I a leaf?
        if (n->children.empty())
        {
            this->leafs_.push_back(n->data);
        }
    });

    #if defined(RD_DEBUG)
    printf("<<<<<       Num collected leafs: %lu\n", leafs_.size());

    printf("-------------------------------------------------\n");
    for(SptrTile t : leafs_)
    {
        t->print();
    }
    printf("-------------------------------------------------\n");

    #endif
}

//===============================================================
//      specializations of TiledGroupTree for EXTENDED_TILE
//===============================================================

template <typename T>  
TiledGroupTree<EXTENDED_TILE, T>::TiledGroupTree()
    : BaseType()
{
}

template <typename T>  
TiledGroupTree<EXTENDED_TILE, T>::~TiledGroupTree()
{
}

template <typename T>
void TiledGroupTree<EXTENDED_TILE, T>::buildTree(
    T const *samples,
    size_t cnt,
    size_t dim,
    std::vector<size_t> const &initTileCnt)
{
    BaseType::buildTree(samples, cnt, dim, initTileCnt);
}

template <typename T>  
void TiledGroupTree<EXTENDED_TILE, T>::partitionSamples(
    std::vector<std::pair<T const *, size_t>> const & vSamples,
    size_t                  dim,
    std::vector<Tile*> &    tiles)
{
    /*
     *  partition data into tiles
     *  
     *  first element in vector are samples inside this tile, others
     *  are this tile neighbouring samples
     */
    for (size_t k = 0; k < vSamples.size(); ++k)
    {
        T const * samples = vSamples[k].first;
        size_t pointsCnt =  vSamples[k].second;

        #ifdef RD_USE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (size_t i = 0; i < pointsCnt; ++i) 
        {
            T const *ptr = samples + i*dim;

            for (Tile *t : tiles)
            {
                if (t->bounds.isNearby(ptr, this->sphereRadius))
                {
                    #ifdef RD_USE_OPENMP
                        #pragma omp critical 
                        {
                    #endif
                    for (size_t d = 0; d < dim; ++d)
                    {
                        t->neighbours.push_back(ptr[d]);
                    }
                    #ifdef RD_USE_OPENMP
                        }
                    #endif
                }
            }
            if (k == 0)
            {
                for (Tile *t : tiles)
                {
                    if (t->bounds.isInside(ptr))
                    {
                        #ifdef RD_USE_OPENMP
                            #pragma omp critical 
                            {
                        #endif
                        t->pointsCnt++;
                        for (size_t d = 0; d < dim; ++d)
                        {
                            t->samples.push_back(ptr[d]);
                        }
                        #ifdef RD_USE_OPENMP
                            }
                        #endif
                        break;
                    }
                }
            }
        }
    }

}


//===============================================================
//      specializations of TiledGroupTree for FIND_NEIGHBOURS
//===============================================================

template <typename T>  
TiledGroupTree<FIND_NEIGHBOURS, T>::TiledGroupTree()
    : BaseType()
{
}

template <typename T>  
TiledGroupTree<FIND_NEIGHBOURS, T>::~TiledGroupTree()
{
}

template <typename T>
void TiledGroupTree<FIND_NEIGHBOURS, T>::buildTree(
    T const *samples,
    size_t cnt,
    size_t dim,
    std::vector<size_t> const &initTileCnt)
{
    BaseType::buildTree(samples, cnt, dim, initTileCnt);

    // find neighbours for each leaf
    this->forEachNodePreorder([this](SptrNode node)
    {
        if (node->children.empty())
        {
            for (SptrNode n : this->root_->children)
            {
                findNeighbourTiles(n, node);
            }
        }
    });
}

template <typename T>  
void TiledGroupTree<FIND_NEIGHBOURS, T>::partitionSamples(
    std::vector<std::pair<T const *, size_t>> const &vSamples,
    size_t dim,
    std::vector<Tile*> &tiles)
{
    /*
     *  partition data into tiles
     */

    T const * samples = vSamples[0].first;
    size_t pointsCnt = vSamples[0].second;

    #ifdef RD_USE_OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < pointsCnt; ++i) 
    {
        T const *ptr = samples + i*dim;

        for (Tile *t : tiles)
        {
            if (t->bounds.isInside(ptr))
            {
                #ifdef RD_USE_OPENMP
                    #pragma omp critical 
                    {
                #endif
                t->pointsCnt++;
                for (size_t d = 0; d < dim; ++d)
                {
                    t->samples.push_back(*ptr++);
                }
                #ifdef RD_USE_OPENMP
                    }
                #endif
                break;
            }
        }
    }
}

template <typename T>  
void TiledGroupTree<FIND_NEIGHBOURS, T>::findNeighbourTiles(
    SptrNode const currNode,
    SptrNode refNode)
{
    typedef std::shared_ptr<Tile> tSptr;

    tSptr currTile(currNode->data);
    tSptr refTile(refNode->data);

    if (currTile->id == refTile->id)
    {
        return;
    }

    if (currNode->children.empty())
    {
        if (refTile->bounds.overlap(currTile->bounds))
        {
            refTile->neighbours.push_back(currTile);
        }
    }
    else
    {
        if (currTile->bounds.overlap(refTile->bounds))
        {
            for (SptrNode n : currNode->children)
            {
                findNeighbourTiles(n, refNode);
            }
        }
    }
}

}   // namespace tiled
}   // namespace rd
