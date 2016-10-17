/**
 * @file local_tile_tree.inl
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering, Faculty of Electronics and Information
 * Technology, Warsaw University of Technology 2016
 */

#include <cmath>
#include <cstring>
#include <algorithm>
#include <functional>
#include <limits>
#include <iostream>

#if defined(RD_USE_OPENMP)
    #include <omp.h>
#endif

namespace rd
{
namespace cpu
{
namespace tiled
{ 

template <typename T>
TiledLocalTree<T>::TiledLocalTree()
	: maxTileCapacity(0), sphereRadius(0), cpuThreads(0)
{
    #if defined(RD_USE_OPENMP)    
        cpuThreads = omp_get_num_procs();
        omp_set_num_threads(cpuThreads);
    #endif
}

template <typename T>
TiledLocalTree<T>::~TiledLocalTree()
{
    leafs_.clear();
    clear(root_);
}

template <typename T>
void TiledLocalTree<T>::buildTree(
	T const *samples,
	size_t pointCnt,
	size_t dim,
	std::vector<size_t> const &initTileCnt)
{
    if (root_.use_count() == 0)
    {
        root_ = std::make_shared<Node>();
        sentinel_.root = root_;
    }
    else
    {
        root_.reset(new Node());
        sentinel_.root.reset();
        sentinel_.root = root_;
    }

    std::vector<Tile*> tiles;
    createTiles(samples, pointCnt, dim, tiles, initTileCnt);

    // create tiles
    for (Tile *t : tiles)
    {
        addNode(root_, t, 1);
    }
    collectLeafs();
}


template <typename T>
void TiledLocalTree<T>::createTiles(
	T const *                      samples,
	size_t                         pointCnt,
	size_t                         dim,
	std::vector<Tile*> &           outTiles,
	std::vector<size_t> const &    tileCnt)
{
    BoundingBox<T> globalBbox(samples, pointCnt, dim);
    globalBbox.calcDistances();

    Histogram<T> hist;
    hist.setBinCnt(tileCnt);
    hist.getHist(samples, pointCnt, [&globalBbox, this](
        T const *point,
        std::vector<size_t> const &bc)
    {
        return histogramMapFunc(point, bc, globalBbox);
    });

    createTiles(samples, pointCnt, dim, hist, outTiles,
                 tileCnt, globalBbox);
}

template <typename T>
void TiledLocalTree<T>::createTiles(
	T const *                       samples,
    size_t                          pointsCnt,
    size_t                          dim,
    Histogram<T> const &            inSamplesHist,
    std::vector<Tile*> &            outTiles,
    std::vector<size_t> const &     tileCnt,
    BoundingBox<T> const &          inSamplesBbox)
{
    
    const size_t nTiles = std::accumulate(tileCnt.begin(), tileCnt.end(),
                         1, std::multiplies<size_t>());

    outTiles.clear();
    // create tiles
    for (size_t i = 0; i < nTiles; ++i)
    {
        Tile *t = new Tile();
        t->pointsCnt = inSamplesHist[i];
        t->dim = dim;
        if (t->pointsCnt)
        {
            t->samples.resize(inSamplesHist[i]*dim);
        }
        outTiles.push_back(t);
    }

    // copy data samples to respective tiles
    // vector of pointers to respective bins end() - 
    // a position where to store next bin point
    std::vector<T*> iters(outTiles.size());
    for (size_t k = 0; k < outTiles.size(); ++k)
    {
        iters[k] = outTiles[k]->samples.data();
    }

    T const *ptr = samples;
    T *tilePtr;
    size_t idx;

    for (size_t k = 0; k < pointsCnt; ++k)
    {
        idx = histogramMapFunc(ptr, tileCnt, inSamplesBbox);
        tilePtr = iters[idx];
        std::memcpy(tilePtr, ptr, dim * sizeof(T));
        iters[idx] = tilePtr + dim;
        ptr += dim;
    }

    // remove empty tiles and with only one sample as they introduce little information
    for (auto it = outTiles.begin(); it != outTiles.end(); )
    {
        /*
         *  TODO: Some mechanism merging tiles with small number of points inside
         */
        if ((*it)->pointsCnt == 0 || (*it)->pointsCnt == 1)
        {
            delete (*it);
            it = outTiles.erase(it);
        }
        else 
        {
            it++;
        }
    }

    // calculate bounds for each tile
    // and alocate memory for chosen samples
    for (Tile *t : outTiles)
    {
		t->bounds.findBounds(t->samples.data(), t->pointsCnt, t->dim);
        t->bounds.calcDistances();
        if (t->pointsCnt < maxTileCapacity)
        {
            t->chosenPointsCapacity = t->bounds.countSpheresInside(sphereRadius);
            t->chosenSamples.resize(t->chosenPointsCapacity * t->dim);
        }
    }
}

/**
 * @brief      Maps point's coordinates to histogram bin idx.
 *
 * @param      sample   Pointer to memory containing mapped point's data.
 * @param      binsCnt  Vector with number of bins in subsequent dimensions.
 * @param      bb       Bounding box for volume containing mapped point.
 *
 * @tparam     T        Point coordinate data type.
 *
 * @return     Histogram bin linear idx .
 */
template <typename T>
size_t TiledLocalTree<T>::histogramMapFunc(
  T const *                     sample,
  std::vector<size_t> const &   binsCnt,
  BoundingBox<T> const &        bb) const
{
    std::vector<size_t> binIdx(bb.dim, 0);
    
    // get sample's bin [x,y,z...n] idx
    for (size_t i = 0; i < bb.dim; ++i, sample++)
    {
        /*
         * translate each sample coordinate to the common origin (by distracting minimum)
         * then divide shifted coordinate by current dimension bin width and get the 
         * floor of this value (counting from zero!) which is our bin idx we search for.
         */
        if (bb.dist[i] < std::numeric_limits<T>::epsilon())
        {
            binIdx[i] = 0;
            continue;
        }

        T normCord = std::abs(*sample - bb.min(i));
        T step = bb.dist[i] / T(binsCnt[i]);

        if (std::abs(normCord - bb.dist[i]) <= std::numeric_limits<T>::epsilon() ||
            normCord - bb.dist[i] > 0)
        {
            binIdx[i] = binsCnt[i]-1;
            // #if defined(RD_DEBUG)
                if (normCord - bb.dist[i] > 0)
                {
                    std::cout << "/////////////////////////////////////////////////\n";
                    std::cout << "normCord - bb.dist["<<i<<"]: " << normCord - bb.dist[i] 
                                << "\t*smaple: " << *sample
                                << "\tbb.min("<<i<<"): " << bb.min(i)
                                << "\tbb.max("<<i<<"): " << bb.max(i)
                                << "\n";
                    std::cout << "/////////////////////////////////////////////////\n";
                }
            // #endif
        }
        else
        {
            binIdx[i] = std::floor(normCord / step);

            // #if defined(RD_DEBUG)
                if (binIdx[i] >= binsCnt[i])
                {
                    std::cout << "/////////////////////////////////////////////////\n";
                    std::cout << "normCord: " << normCord 
                              << "\t*sample: " << *sample
                              << "\tbb.min(" << i << "): " << bb.min(i)
                              << "\tbinsCnt[" << i << "]: " << binsCnt[i]
                             << "\n" ;
                    std::cout << "normCord / step: " << normCord / step 
                            << "\tbinIdx["<<i<<"]: " << binIdx[i]
                            << "\n" ;
                    std::cout << "/////////////////////////////////////////////////\n";
                }
            // #endif
        }
    }

    /*
     * Calculate global idx value linearizing bin idx
     * idx = k_0 + sum_{i=2}^{dim}{k_i mul_{j=i-1}^{1}bDim_j}
     */
    size_t idx = binIdx[0];
    size_t tmp;
    for (size_t i = 1; i < bb.dim; ++i)
    {
        tmp = 1;
        for (int j = (int)i - 1; j >= 0; --j)
        {
            tmp *= binsCnt[j];
        }
        idx += binIdx[i]*tmp;
    }

    return idx;
}

template <typename T>
void TiledLocalTree<T>::addNode(
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
        #ifdef RD_DEBUG
        node->treeLevel = treeLevel;
        #endif
        parent->children.push_back(node);

        // std::cout << "Subdividing tile! ..\n" << std::flush;
        // tile->print();

        std::vector<Tile*> subTiles;
        subdivideTile(tile, subTiles, treeLevel);

        delete tile;

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

template <typename T>
void TiledLocalTree<T>::subdivideTile(
		Tile const *            tile,
        std::vector<Tile*> &    outSubTiles,
        size_t                  treeLevel)
{
    /*
     * XXX: As for now, I'm using simple solution, that is half splitting.
     * In subsequent versions this should be parameterized
     */

    // halft split tile
    // so we have 2 tiles in (tile->treeLevel+1%dim) dimension,
    // and 1 in other dims
    std::vector<size_t> tileCnt(tile->dim,1);
    tileCnt[(treeLevel)%tile->dim] = 2;
    Histogram<T> hist;
    hist.setBinCnt(tileCnt);
    hist.getHist(tile->samples.data(), tile->pointsCnt, 
                    [tile, this](T const *s, std::vector<size_t> &bc)
    {
        return histogramMapFunc(s, bc, tile->bounds);
    });

    createTiles(tile->samples.data(), tile->pointsCnt, tile->dim, 
                hist, outSubTiles, tileCnt, tile->bounds);
}

template <typename T>
    template <typename UnaryFunction>
void TiledLocalTree<T>::forEachNodePreorder(UnaryFunction const &f)
{
    visitNode(root_, f);
}

template <typename T>
    template <typename UnaryFunction>
void TiledLocalTree<T>::visitNode(SptrNode node, UnaryFunction const &f)
{
    f(node);
    for (SptrNode n : node->children)
        visitNode(n, f);
}

template <typename T>
    template <typename UnaryFunction>
void TiledLocalTree<T>::forEachLeaf(UnaryFunction const &f)
{
    #if defined(RD_USE_OPENMP)
    #pragma omp parallel for num_threads(cpuThreads) schedule(static)
    #endif
    for (size_t k = 0; k < leafs_.size(); ++k)
    {
        // #if defined(RD_USE_OPENMP) && defined(RD_DEBUG)
        //     printf("omp_tid: %d\n", omp_get_thread_num());
        // #endif
        f(leafs_[k]);
    }
}


template <typename T>
void TiledLocalTree<T>::clear(SptrNode &node)
{
     if (node)
     {
        node->clear();
     }
}

template <typename T>
void TiledLocalTree<T>::print() const
{
    root_->printRecursive();
}

template <typename T>
void TiledLocalTree<T>::collectLeafs()
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

}   // end namespace tiled
}   // end namespace cpu
}   // end namespace rd
