/*
 * @file rd_tiled.hpp
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


#ifndef RD_TILED_HPP
#define RD_TILED_HPP

#include "rd/utils/utilities.hpp"
#include "rd/utils/graph_drawer.hpp"
#include "tests/test_util.hpp"
#include "rd/cpu/brute_force/ridge_detection.hpp"
#include "rd/cpu/brute_force/choose.hpp"
#include "rd/cpu/brute_force/evolve.hpp"
#include "rd/cpu/brute_force/decimate.hpp"
#include "rd/cpu/tiled/local_tile_tree.hpp"
#include "rd/cpu/tiled/grouped_tile_tree.hpp"

#include <cstdlib>
#include <list>
#include <deque>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <tuple>

#if defined(RD_USE_OPENMP)
    #include <omp.h>
#endif

namespace rd 
{
namespace tiled
{


//==========================================================================
//  Data structures and enumerations
//==========================================================================

template <typename T>
struct rdTiledData
{
    /// samples dimension
    size_t dim;
    /// cardinality of samples set
    size_t np;
    /// cardinality of initial choosen samples set
    size_t ns;
    /// max number of samples inside single tile
    size_t maxTileCapacity;
    /// initial tiles count for respective dimension
    std::vector<size_t> nTilesPerDim;
    /**
     * @var r1_ Algorithm parameter. Radius used for choosing samples and in
     * evolve phase.
     */
    T r1;
    /**
     * @var r2_ Algorithm parameter. Radius used for decimation phase.
     */
    T r2;
    /// generated samples parameters (s - standard deviation)
    T a, b, s;
    /// table containing samples
    T const * P;
    /// table containing choosen samples
    std::vector<T> S;

    // /// List with coherent point chains.
    // std::list<std::deque<T const*>> chainList_;
    
    rdTiledData()
    :
        dim(0),
        np(0),
        ns(0),
        maxTileCapacity(0),
        nTilesPerDim(),
        r1(0), r2(0),
        a(0), b(0), s(0),
        P(nullptr), S()
    {
    }  
    
};

enum TiledRDAlgorithm
{
    TILED_LOCAL_TREE_MIXED_RD,
    TILED_GROUPED_TREE_MIXED_RD,
    TILED_LOCAL_TREE_LOCAL_RD,
    TILED_LOCAL_LIST,
    TILED_GROUPED_TREE_LOCAL_RD,
    EXT_TILE_TREE_MIXED_RD
};

//==========================================================================
//  Algorithm traits only for drawing graphs purposes
//==========================================================================

template <TiledRDAlgorithm Alg>
struct TiledRDAlgorithmNameTraits
{
};

template <>
struct TiledRDAlgorithmNameTraits<TILED_LOCAL_TREE_LOCAL_RD>
{
    static constexpr const char* name = "_LOCAL_TREE_LOCAL_RD";
    static constexpr const char* shortName = "LT_LR";
};
template <>
struct TiledRDAlgorithmNameTraits<TILED_LOCAL_TREE_MIXED_RD>
{
    static constexpr const char* name = "_LOCAL_TREE_MIXED_RD";
    static constexpr const char* shortName = "LT_MR";
};
template <>
struct TiledRDAlgorithmNameTraits<TILED_LOCAL_LIST>
{
    static constexpr const char* name = "_LOCAL_LIST";
    static constexpr const char* shortName = "LL";
};
template <>
struct TiledRDAlgorithmNameTraits<TILED_GROUPED_TREE_LOCAL_RD>
{
    static constexpr const char* name = "_GROUPED_TREE_LOCAL_RD";
    static constexpr const char* shortName = "GT_LR";
};
template <>
struct TiledRDAlgorithmNameTraits<TILED_GROUPED_TREE_MIXED_RD>
{
    static constexpr const char* name = "_GROUPED_TREE_MIXED_RD";
    static constexpr const char* shortName = "GT_MR";
};
template <>
struct TiledRDAlgorithmNameTraits<EXT_TILE_TREE_MIXED_RD>
{
    static constexpr const char* name = "_EXT_TILE_TREE_MIXED_RD";
    static constexpr const char* shortName = "ETT_MR";
};

namespace detail
{

//=========================================================================
//      Drawing methods
//=========================================================================


template <typename T>
void collectTileBounds2D(
    std::vector<T> &        tbound,
    BoundingBox<T> const &  bounds)
{
    /**
     * can generate bounds from Gray codes
     * dim-bit Gray code sequence is a sequence of
     * our bounding box corners
     */

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
void collectTileBounds3D(
    std::vector<T> &        tbound,
    BoundingBox<T> const &  bounds)
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

template <typename T>
void collectTileBounds(
    std::vector<T> &        tbound,
    BoundingBox<T> const &  bounds)
{
    if (bounds.dim == 2)
    {
        collectTileBounds2D(tbound, bounds);
    }
    else if (bounds.dim == 3)
    {
        collectTileBounds3D(tbound, bounds);
    }
    else
    {
        throw std::runtime_error(__FILE__ + 
            std::string(" ") + std::to_string(__LINE__) + 
            std::string(" [collectTileBounds]: Unsupported dimension!"));
    }
}

template <
    TiledRDAlgorithm ALG,
    typename T>
void getGraphNamePrefix(
    rdTiledData<T> const & dataPack,
    std::ostringstream & s)
{
    s << typeid(T).name() << "_" << getCurrDateAndTime() << "_"
        << rd::tiled::TiledRDAlgorithmNameTraits<ALG>::shortName
        << "_mtc-" << dataPack.maxTileCapacity 
        << "_ntpd-" << rdToString(dataPack.nTilesPerDim) 
        << "_np-" << dataPack.np; 
        // << "_r1-" << dataPack.r1 
        // << "_r2-" << dataPack.r2 
        // << "_a-" << dataPack.a 
        // << "_b-" << dataPack.b 
        // << "_s-" << dataPack.s;
}

template <
    TiledRDAlgorithm ALG,
    typename Tree,
    typename T>
void drawEachLocalTileWithBoundsAndChosenSamples(
    rdTiledData<T> const & dataPack,
    Tree & tree)
{
    typedef std::shared_ptr<typename Tree::Tile> TileSptr;

    if (dataPack.dim > 3)
    {
        return;
    }

    rd::GraphDrawer<T> gDrawer;
    std::ostringstream s;
    std::vector<std::vector<T>> bounds;

    getGraphNamePrefix<ALG>(dataPack, s);
    s << "_tiles";
    std::string filePath = rd::findPath("img/", s.str());

    gDrawer.startGraph(filePath, dataPack.dim);
    if (dataPack.dim == 3)
    {
        gDrawer.setGraph3DConf();
    }

        
    gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#B8E186' ps 0.5 ",
         dataPack.P, rd::GraphDrawer<T>::POINTS, dataPack.np);

    tree.forEachNodePreorder([&dataPack, &gDrawer, &s, &bounds](
        typename Tree::SptrNode const &n)
    {
        if (n->children.empty())
        {
            TileSptr t(n->data);
            gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#171212' ps 1.6 ",
                 t->chosenSamples.data(), rd::GraphDrawer<T>::POINTS,
                  t->chosenPointsCnt);
            std::vector<T> tbound;
            collectTileBounds(tbound, t->bounds);
            bounds.push_back(tbound);
       }
    });
    for (std::vector<T> &v : bounds)
    {
        gDrawer.addPlotCmd("'-' w l lt 8 lc rgb '#d64f4f' lw 1.3 ",
            v.data(), rd::GraphDrawer<T>::LINE, v.size() / dataPack.dim);
    }

    gDrawer.endGraph();
}

template <
    TiledRDAlgorithm ALG,
    typename T>
void drawPointCloudAndChosenSamples(
    rdTiledData<T> const & dataPack,
    std::string graphName)
{
    if (dataPack.dim > 3)
    {
        return;
    }

    rd::GraphDrawer<T> gDrawer;
    std::ostringstream s;

    getGraphNamePrefix<ALG>(dataPack, s);
    s << graphName;
    std::string filePath = rd::findPath("img/", s.str());
    gDrawer.startGraph(filePath, dataPack.dim);
    if (dataPack.dim == 3)
    {
        gDrawer.setGraph3DConf();
    }
    gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#B8E186' ps 0.5 ",
         dataPack.P, rd::GraphDrawer<T>::POINTS, dataPack.np);
    gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#D73027' ps 1.3 ",
         dataPack.S.data(), rd::GraphDrawer<T>::POINTS, dataPack.ns);
    gDrawer.endGraph();
}

template <
    TiledRDAlgorithm ALG,
    typename Tree,
    typename T>
void drawAllTilesWithBounds(
    rdTiledData<T> const & dataPack,
    Tree & tree)
{
    /************************************
     * collect bounds 
     ************************************/
    std::vector<std::vector<T>> bounds;
    tree.forEachLeaf([&bounds](typename Tree::SptrTile t)
    {
        std::vector<T> tbound;
        collectTileBounds(tbound, t->bounds);
        #if defined(RD_USE_OPENMP)
            #pragma omp critical
            {
            bounds.push_back(tbound);
            }    
        #else
        bounds.push_back(tbound);
        #endif
    });

    /************************************
     * all tiles with bounds 
     ************************************/
    rd::GraphDrawer<T> gDrawer;
    std::ostringstream s;
    getGraphNamePrefix<ALG>(dataPack, s);
    s << "_tiled_tree";
    std::string filePath = rd::findPath("img/", s.str());

    gDrawer.startGraph(filePath, dataPack.dim);
    if (dataPack.dim == 3)
    {
        gDrawer.setGraph3DConf();
    }

    // gDrawer.sendCmd("set pm3d depthorder");
    // gDrawer.sendCmd("set hidden3d front");
    // splot f(x,y) with pm3d, ..
    // http://gnuplot.sourceforge.net/demo/hidden2.html

    gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
     dataPack.P, rd::GraphDrawer<T>::POINTS, dataPack.np);
    for (std::vector<T> &v : bounds)
    {
        gDrawer.addPlotCmd("'-' w l lt 8 lc rgb '#107a00' lw 1.5 ",
            v.data(), rd::GraphDrawer<T>::LINE, v.size() / dataPack.dim);
    }
    gDrawer.endGraph();
}

template <
    TiledRDAlgorithm ALG,
    typename Tree,
    typename T>
void drawEachGroupedTileWithBoundsAndChosenSamples(
    rdTiledData<T> const & dataPack,
    Tree & tree)
{
    typedef std::shared_ptr<typename Tree::Tile> TileSptr;

    if (dataPack.dim > 3)
    {
        return;
    }

    /************************************
     * collect all bounds 
     ************************************/
    std::vector<std::vector<T>> bounds;
    tree.forEachNodePreorder([&bounds](typename Tree::SptrNode const &n)
    {
        if (n->children.empty())
        {
            TileSptr t(n->data);
            std::vector<T> tbound;
            collectTileBounds(tbound, t->bounds);
            bounds.push_back(tbound);
        }
    });

    std::ostringstream s;
    rd::GraphDrawer<T> gDrawer;

    /************************************
     * tiles with neighbours
     ************************************/
    tree.forEachNodePreorder([&](typename Tree::SptrNode const &n)
    {
        // am I a leaf?
        if (n->children.empty())
        {
            TileSptr t(n->data);
            std::vector<std::vector<T>> neighbourBounds;
            getGraphNamePrefix<ALG>(dataPack, s);
            s << "_tid_" << t->id;
            std::string filePath = rd::findPath("img/", s.str());
            gDrawer.startGraph(filePath, dataPack.dim);
            if (dataPack.dim == 3)
            {
                gDrawer.setGraph3DConf();
            }
            // draw all points as background
            gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
                 dataPack.P, rd::GraphDrawer<T>::POINTS, dataPack.np);

            for (TileSptr neighbour : t->neighbours)
            {
                // mark each neighbour tile points 
                gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#B2D88C' ps 0.8 ",
                     neighbour->samples.data(), rd::GraphDrawer<T>::POINTS, neighbour->pointsCnt);

                std::vector<T> tbound;
                collectTileBounds(tbound, neighbour->bounds);
                neighbourBounds.push_back(tbound);
            }

            // draw all bounds
            for (std::vector<T> &v : bounds)
            {
                gDrawer.addPlotCmd("'-' w l lt 8 lc rgb '#107a00' lw 1.3 ",
                    v.data(), rd::GraphDrawer<T>::LINE, v.size() / dataPack.dim);
            }

            // mark neighbour bounds
            for (std::vector<T> &v : neighbourBounds)
            {
                gDrawer.addPlotCmd("'-' w l lt 8 lc rgb '#99CCFF' lw 1.5 ",
                    v.data(), rd::GraphDrawer<T>::LINE, v.size() / dataPack.dim);
            }

            // mark this tile points 
            gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#99CCFF' ps 0.8 ",
                 t->samples.data(), rd::GraphDrawer<T>::POINTS, t->pointsCnt);

            // draw this tile chosen samples 
            gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#A50000' ps 1.3 ",
                 t->chosenSamples.data(), rd::GraphDrawer<T>::POINTS, t->chosenPointsCnt);

            std::vector<T> tbound;
            collectTileBounds(tbound, t->bounds);

            // mark this tile bounds
            gDrawer.addPlotCmd("'-' w l lt 8 lc rgb '#CB1E1E' lw 1.5 ",
                tbound.data(), rd::GraphDrawer<T>::LINE, tbound.size() / dataPack.dim);

            gDrawer.endGraph();

            s.clear();
            s.str(std::string());
       }
    });
}

template <
    typename Tree,
    typename T>
void drawEachGroupedTileWithBoundsAndChosenSamples(
    rdTiledData<T> const & dataPack,
    Tree & tree,
    Int2Type<EXT_TILE_TREE_MIXED_RD>)
{
    typedef std::shared_ptr<typename Tree::Tile> TileSptr;

    if (dataPack.dim > 3)
    {
        return;
    }

    /************************************
     * collect all bounds 
     ************************************/
    std::vector<std::vector<T>> bounds;
    tree.forEachNodePreorder([&bounds](typename Tree::SptrNode const &n)
    {
        if (n->children.empty())
        {
            TileSptr t(n->data);
            std::vector<T> tbound;
            collectTileBounds(tbound, t->bounds);
            bounds.push_back(tbound);
        }
    });

    std::ostringstream s;
    rd::GraphDrawer<T> gDrawer;

    /************************************
     * tiles with neighbours
     ************************************/
    tree.forEachNodePreorder([&](typename Tree::SptrNode const &n)
    {
        // am I a leaf?
        if (n->children.empty())
        {
            TileSptr t(n->data);
            std::vector<std::vector<T>> neighbourBounds;

            getGraphNamePrefix<EXT_TILE_TREE_MIXED_RD>(dataPack, s);
            s << "_tid_" << t->id;
            std::string filePath = rd::findPath("img/", s.str());
            gDrawer.startGraph(filePath, dataPack.dim);
            if (dataPack.dim == 3)
            {
                gDrawer.setGraph3DConf();
            }
            // draw all points as background
            gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
                 dataPack.P, rd::GraphDrawer<T>::POINTS, dataPack.np);

            // mark neighbouring points 
            gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#B2D88C' ps 0.8 ",
                 t->neighbours.data(), rd::GraphDrawer<T>::POINTS, t->neighbours.size() / dataPack.dim);

            // draw all bounds
            for (std::vector<T> &v : bounds)
            {
                gDrawer.addPlotCmd("'-' w l lt 8 lc rgb '#107a00' lw 1.3 ",
                    v.data(), rd::GraphDrawer<T>::LINE, v.size() / dataPack.dim);
            }

            // mark this tile points 
            gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#99CCFF' ps 0.8 ",
                 t->samples.data(), rd::GraphDrawer<T>::POINTS, t->pointsCnt);

            // draw this tile chosen samples 
            gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#A50000' ps 1.3 ",
                 t->chosenSamples.data(), rd::GraphDrawer<T>::POINTS, t->chosenPointsCnt);

            std::vector<T> tbound;
            collectTileBounds(tbound, t->bounds);

            // mark this tile bounds
            gDrawer.addPlotCmd("'-' w l lt 8 lc rgb '#CB1E1E' lw 1.5 ",
                tbound.data(), rd::GraphDrawer<T>::LINE, tbound.size() / dataPack.dim);

            gDrawer.endGraph();

            s.clear();
            s.str(std::string());
       }
    });    
}

template <
    TiledRDAlgorithm ALG,
    typename Tree,
    typename T>
void drawAllTilesWithBoundsAndChosenSamples(
    rdTiledData<T> const & dataPack,
    Tree & tree)
{
    typedef std::shared_ptr<typename Tree::Tile> TileSptr;

    if (dataPack.dim > 3)
    {
        return;
    }

    /************************************
     * collect all bounds
     ************************************/
    std::vector<std::vector<T>> bounds;
    tree.forEachNodePreorder([&bounds](typename Tree::SptrNode const &n)
    {
        if (n->children.empty())
        {
            TileSptr t(n->data);
            std::vector<T> tbound;
            collectTileBounds(tbound, t->bounds);
            bounds.push_back(tbound);
        }
    });

    std::ostringstream s;
    rd::GraphDrawer<T> gDrawer;

    /************************************
     * tiles with chosen samples
     ************************************/

    getGraphNamePrefix<ALG>(dataPack, s);
    s << "_bounds_w_chosen_smpl";
    std::string filePath = rd::findPath("img/", s.str());
    gDrawer.startGraph(filePath, dataPack.dim);
    if (dataPack.dim == 3)
    {
        gDrawer.setGraph3DConf();
    }
    // draw all points as background
    gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#A8DDB5' ps 0.5 ",
         dataPack.P, rd::GraphDrawer<T>::POINTS, dataPack.np);
    // draw all bounds
    for (std::vector<T> &v : bounds)
    {
        gDrawer.addPlotCmd("'-' w l lt 8 lc rgb '#FDBB84' lw 1.1 ",
            v.data(), rd::GraphDrawer<T>::LINE, v.size() / dataPack.dim);
    }

    // std::ostringstream circlesCmd;
    // circlesCmd << "'-' u 1:2:(" << dataPack.r1 << ") w circles lc rgb '#D7301F' fs transparent solid 0.15 noborder"; 

    tree.forEachNodePreorder([&](typename Tree::SptrNode const &n)
    {
        // am I a leaf?
        if (n->children.empty())
        {
            TileSptr t(n->data);
            // draw this tile chosen samples
            // gDrawer.addPlotCmd(circlesCmd.str(),
            //      t->chosenSamples.data(), rd::GraphDrawer<T>::CIRCLES, t->chosenPointsCnt);
            gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#084594' ps 1.5 ",
                 t->chosenSamples.data(), rd::GraphDrawer<T>::POINTS, t->chosenPointsCnt);
       }
    });
    gDrawer.endGraph();

    s.clear();
    s.str(std::string());
}

//==========================================================================
//  Ridge detection algorithm specializations
//==========================================================================

template <
    TiledRDAlgorithm ALG,
    typename Tree,
    typename T>
void buildTree(
    Tree &              tree,
    rdTiledData<T> &    dataPack,
    bool                verbose = false)
{
    tree.maxTileCapacity = dataPack.maxTileCapacity;
    tree.sphereRadius = dataPack.r1;
    dataPack.ns = 0;

    /************************************
     *      build tree
     ************************************/

    tree.buildTree(dataPack.P, dataPack.np, dataPack.dim, dataPack.nTilesPerDim);

    if (verbose && (dataPack.dim == 2 || dataPack.dim == 3))
    {
        drawAllTilesWithBounds<ALG, Tree>(dataPack, tree);
    }
}

template <typename T>
std::tuple<float, float, float, float> doRidgeDetection(
    rdTiledData<T> &    dataPack,
    Int2Type<TILED_LOCAL_TREE_MIXED_RD>,
    int                 cpuThreads = 1,
    bool                refinement = false,
    bool                verbose = false,
    bool                drawTiles = false)
{
    typedef TiledLocalTree<T> TiledTree;

    // in case of verbose output
    rd::CpuTimer treeTimer, rdTilesTimer;
    rd::CpuTimer endPhaseTimer, wholeTimer;

    TiledTree tree;
    tree.cpuThreads = cpuThreads;

    wholeTimer.start();
    treeTimer.start();

    /************************************
     *      build tree
     ************************************/
    buildTree<TILED_LOCAL_TREE_MIXED_RD>(tree, dataPack, verbose);

    treeTimer.stop();
    rdTilesTimer.start();

    /************************************
     *      process tiles
     ************************************/

    /*
     * Since in mixed ridge detection version each decimation phase is performed
     * globally I need to somehow track from which tile which chosen points were 
     * removed. To tackle this problem each tile stores its own list of chosen points.
     * It is simply a list of pointers to chosen points in chosenSamples table.
     * Now global decimation iterates through these lists and removes redundant points 
     * from them. Thus preserving information from which tile given point has been removed.
     * The last thing to do, after global decimation is to compact chosen points table. 
     * That is to reduce empty cells after decimation.
     */

    std::list<std::list<T*>*> gCPList;

    /************************************
     *      choose points in tiles
     ************************************/
    tree.forEachLeaf([&dataPack, &gCPList](typename TiledTree::SptrTile t)
        {
            choose(t->samples.data(),
                t->chosenSamples.data(),
                t->cpList,
                t->pointsCnt,
                t->chosenPointsCnt,
                t->dim,
                dataPack.r1);

            #if defined(RD_USE_OPENMP)
            #pragma omp atomic update
            #endif
            dataPack.ns += t->chosenPointsCnt;
            #if defined(RD_USE_OPENMP)
                #pragma omp critical
                {
            #endif
            gCPList.push_back(&(t->cpList));
            #if defined(RD_USE_OPENMP)
                }
            #endif
            #ifdef RD_DEBUG                
            if (t->chosenPointsCnt > t->chosenPointsCapacity)
            {
                std::cout << ">>>>>>>> Exceeding chosenPointsCapacity! chosen cnt:(" <<
                    t->chosenPointsCnt << "), capacity: (" <<
                    t->chosenPointsCapacity << ")" << std::endl;
                t->print();
                std::cout.flush();
                throw std::logic_error("Exceeding chosenPointsCapacity!");
            }
            #endif
        }
    );

    /************************************
     *      ridge detection main part
     ************************************/
    size_t oldCount = 0;

    while(oldCount != dataPack.ns)
    {
        oldCount = dataPack.ns;
        //  evolve chosen samples locally in every tile
        tree.forEachLeaf([&dataPack, cpuThreads, verbose](typename TiledTree::SptrTile t)
            {
                evolve(t->samples.data(),
                    t->chosenSamples.data(),
                    t->pointsCnt,
                    t->chosenPointsCnt,
                    t->dim,
                    dataPack.r1,
                    cpuThreads,
                    verbose);
            }
        );
        // decimate chosen samples globally in all tiles
        decimateNoCopy(gCPList, dataPack.ns, dataPack.dim, dataPack.r2);
        // update chosen samples information in every tile and move data
        tree.forEachLeaf([](typename TiledTree::SptrTile t)
            {
                t->chosenPointsCnt = t->cpList.size();
                T* dstAddr = t->chosenSamples.data();
                // copy real data
                for (auto it = t->cpList.begin();
                          it != t->cpList.end();
                     dstAddr += t->dim, it++)
                {
                    /*
                     *  Starting from first addresses discrepancy the copyTable will be carried out
                     *  at every iteration. Thus shifting left all subsequent cells to fill empyt 
                     *  space after removed redundant point.
                     */
                    if (*it != dstAddr)
                    {
                        copyTable<T>(*it, dstAddr, t->dim);
                        *it = dstAddr;
                    }
                }
            }
        );
    }

    // collect chosen points
    tree.forEachNodePreorder([&dataPack](
        typename TiledTree::SptrNode n)
        {
            if (n->children.empty())
            {
                typename TiledTree::SptrTile t(n->data);
                dataPack.S.insert(
                            dataPack.S.end(), 
                            t->chosenSamples.data(),
                            t->chosenSamples.data() + t->chosenPointsCnt * t->dim);
            }
        }
    );


    rdTilesTimer.stop();

    if (verbose && drawTiles)
    {
        drawEachLocalTileWithBoundsAndChosenSamples<TILED_LOCAL_TREE_MIXED_RD, TiledTree>(
            dataPack, tree);
    } 

    /************************************
     *      end phase refinement
     ************************************/

    endPhaseTimer.start();

    if (refinement)
    {
        if (verbose)
        {
            drawPointCloudAndChosenSamples<TILED_LOCAL_TREE_MIXED_RD>(
                dataPack, "_before_refinement");
        }

        std::list<T*> cpList;        
        T * ptr = dataPack.S.data();
        for (size_t k = 0; k < dataPack.ns; ++k)
        {
            cpList.push_back(ptr);
            ptr += dataPack.dim;
        }

        decimate(dataPack.S.data(),
            cpList, 
            dataPack.ns,
            dataPack.dim,
            dataPack.r2);

        evolve_omp(dataPack.P,
            dataPack.S.data(),
            dataPack.np,
            dataPack.ns,
            dataPack.dim,
            dataPack.r1,
            cpuThreads);
    }

    endPhaseTimer.stop();
    wholeTimer.stop();
    
    float buildTreeTime = treeTimer.elapsedMillis();
    float rdTilesTime = rdTilesTimer.elapsedMillis();
    float refinementTime = endPhaseTimer.elapsedMillis();
    float wholeTime = wholeTimer.elapsedMillis();
    
    if (verbose)
    {
        std::cout << "build tree: " << buildTreeTime << "\n";
        std::cout << "ridge detection tiles: " << rdTilesTime << "\n";
        std::cout << "end phase refinement: " << refinementTime << "\n";
        std::cout << "whole: " << wholeTime << "\n";
    }
    return std::make_tuple(wholeTime, buildTreeTime, rdTilesTime, refinementTime);
}

template <typename T>
std::tuple<float, float, float, float> doRidgeDetection(
    rdTiledData<T> &    dataPack,
    Int2Type<TILED_GROUPED_TREE_MIXED_RD>,
    int                 cpuThreads = 1,
    bool                refinement = false,
    bool                verbose = false,
    bool                drawTiles = false)
{
    typedef TiledGroupTree<FIND_NEIGHBOURS, T> TiledTree;

    // in case of verbose output
    rd::CpuTimer treeTimer, rdTilesTimer;
    rd::CpuTimer endPhaseTimer, wholeTimer;

    TiledTree tree;
    tree.cpuThreads = cpuThreads;

    wholeTimer.start();
    treeTimer.start();

    /************************************
     *      build tree
     ************************************/
    buildTree<TILED_GROUPED_TREE_MIXED_RD>(tree, dataPack, verbose);

    treeTimer.stop();
    rdTilesTimer.start();

    /************************************
     *      process tiles
     ************************************/

    std::list<std::list<T*>*> gCPList;

    /************************************
     *      choose samples in tiles
     ************************************/
    tree.forEachLeaf([&dataPack, &gCPList](typename TiledTree::SptrTile t)
        {
            choose(t->samples.data(),
                t->chosenSamples.data(),
                t->cpList,
                t->pointsCnt,
                t->chosenPointsCnt,
                t->dim,
                dataPack.r1);

            #if defined(RD_USE_OPENMP)
            #pragma omp atomic update
            #endif
            dataPack.ns += t->chosenPointsCnt;
            #if defined(RD_USE_OPENMP)
                #pragma omp critical
                {
            #endif
            gCPList.push_back(&(t->cpList));
            #if defined(RD_USE_OPENMP)
                }
            #endif
            #ifdef RD_DEBUG
            if (t->chosenPointsCnt > t->chosenPointsCapacity)
            {
                std::cout << ">>>>>>>> Exceeding chosenPointsCapacity! chosen cnt:(" <<
                    t->chosenPointsCnt << "), capacity: (" <<
                    t->chosenPointsCapacity << ")" << std::endl;
                t->print();
                std::cout.flush();
                throw std::logic_error("Exceeding chosenPointsCapacity!");
            }
            #endif
        }
    );


    if (verbose && drawTiles)
    {
    	// collect chosen points
        tree.forEachNodePreorder([&dataPack](typename TiledTree::SptrNode n)
            {
                if (n->children.empty())
                {
                    typename TiledTree::SptrTile t(n->data);
                    dataPack.S.insert(
                                dataPack.S.end(), 
                                t->chosenSamples.data(),
                                t->chosenSamples.data() + t->chosenPointsCnt * t->dim);
                }
            }
        );   

    	drawPointCloudAndChosenSamples<TILED_GROUPED_TREE_MIXED_RD>(
            dataPack, "_after_choose");

        drawAllTilesWithBoundsAndChosenSamples<TILED_GROUPED_TREE_MIXED_RD, TiledTree>(
            dataPack, tree);
    	dataPack.S.clear();
    }

    /************************************
     *      ridge detection main part
     ************************************/
    size_t oldCount = 0;

    while(oldCount != dataPack.ns)
    {
        oldCount = dataPack.ns;
        //  evolve chosen samples locally in every tile
        tree.forEachLeaf([&dataPack, cpuThreads, verbose](typename TiledTree::SptrTile t)
            {
                evolve_neighbours(t->getAllSamples(),
                    t->chosenSamples.data(),
                    t->chosenPointsCnt,
                    t->dim,
                    dataPack.r1,
                    cpuThreads,
                    verbose);
            }
        );

        #ifdef RD_DEBUG
        if (verbose && drawTiles)
        {
            drawEachGroupedTileWithBoundsAndChosenSamples<TILED_GROUPED_TREE_MIXED_RD, TiledTree>(
                dataPack, tree);
            drawAllTilesWithBoundsAndChosenSamples<TILED_GROUPED_TREE_MIXED_RD, TiledTree>(
                dataPack, tree);
        }
        #endif

        // decimate chosen samples globally in all tiles
        decimateNoCopy(gCPList, dataPack.ns, dataPack.dim, dataPack.r2);
        // update chosen samples information in every tile and move data
        tree.forEachLeaf([](typename TiledTree::SptrTile t)
            {
                t->chosenPointsCnt = t->cpList.size();
                T* dstAddr = t->chosenSamples.data();
                // copy real data
                for (auto it = t->cpList.begin();
                          it != t->cpList.end();
                     dstAddr += t->dim, it++)
                {
                    if (*it != dstAddr)
                    {
                        copyTable<T>(*it, dstAddr, t->dim);
                        *it = dstAddr;
                    }
                }
                // XXX: this only for grouped tile version!!!
                // now can erase unnecesary elements at the end of vector.
                // t->chosenSamples.resize(t->chosenPointsCnt);
                //  OR:
                // t->chosenSamples.erase(t->chosenSamples.begin()+t->chosenPointsCnt,
                //     t->chosenSamples.end());
            }
        );

        #ifdef RD_DEBUG
        if (verbose && drawTiles)
        {
            drawEachGroupedTileWithBoundsAndChosenSamples<TILED_GROUPED_TREE_MIXED_RD, TiledTree>(
                dataPack, tree);
            drawAllTilesWithBoundsAndChosenSamples<TILED_GROUPED_TREE_MIXED_RD, TiledTree>(
                dataPack, tree);
        }
        #endif
    }

    // collect chosen points
    tree.forEachNodePreorder([&dataPack](typename TiledTree::SptrNode n)
        {
            if (n->children.empty())
            {
                typename TiledTree::SptrTile t(n->data);
                dataPack.S.insert(
                            dataPack.S.end(), 
                            t->chosenSamples.data(),
                            t->chosenSamples.data() + t->chosenPointsCnt * t->dim);
            }
        }
    );

    rdTilesTimer.stop();


    if (verbose && drawTiles)
    {
        drawEachGroupedTileWithBoundsAndChosenSamples<TILED_GROUPED_TREE_MIXED_RD, TiledTree>(
            dataPack, tree);
        drawAllTilesWithBoundsAndChosenSamples<TILED_GROUPED_TREE_MIXED_RD, TiledTree>(
            dataPack, tree);
    }

    /************************************
     *      end phase refinement
     ************************************/

    endPhaseTimer.start();

    if (refinement)
    {
        if (verbose)
        {
            drawPointCloudAndChosenSamples<TILED_GROUPED_TREE_MIXED_RD>(
                dataPack, "_before_refinement");
        }

        std::list<T*> cpList;        
        T * ptr = dataPack.S.data();
        for (size_t k = 0; k < dataPack.ns; ++k)
        {
            cpList.push_back(ptr);
            ptr += dataPack.dim;
        }

        decimate(dataPack.S.data(),
            cpList, 
            dataPack.ns,
            dataPack.dim,
            dataPack.r2);

        evolve_omp(dataPack.P,
            dataPack.S.data(),
            dataPack.np,
            dataPack.ns,
            dataPack.dim,
            dataPack.r1,
            cpuThreads);
    }

    endPhaseTimer.stop();
    wholeTimer.stop();
    
    float buildTreeTime = treeTimer.elapsedMillis();
    float rdTilesTime = rdTilesTimer.elapsedMillis();
    float refinementTime = endPhaseTimer.elapsedMillis();
    float wholeTime = wholeTimer.elapsedMillis();
    
    if (verbose)
    {
        std::cout << "build tree: " << buildTreeTime << "\n";
        std::cout << "ridge detection tiles: " << rdTilesTime << "\n";
        std::cout << "end phase refinement: " << refinementTime << "\n";
        std::cout << "whole: " << wholeTime << "\n";
    }
    return std::make_tuple(wholeTime, buildTreeTime, rdTilesTime, refinementTime);
}


template <typename T>
std::tuple<float, float, float, float> doRidgeDetection(
    rdTiledData<T> &    dataPack,
    Int2Type<EXT_TILE_TREE_MIXED_RD>,
    int                 cpuThreads = 1,
    bool                refinement = false,
    bool                verbose = false,
    bool                drawTiles = false)
{
    typedef TiledGroupTree<EXTENDED_TILE, T> TiledTree;

    // in case of verbose output
    rd::CpuTimer treeTimer, rdTilesTimer;
    rd::CpuTimer endPhaseTimer, wholeTimer;

    TiledTree tree;
    tree.cpuThreads = cpuThreads;

    wholeTimer.start();
    treeTimer.start();

    /************************************
     *      build tree
     ************************************/
    buildTree<EXT_TILE_TREE_MIXED_RD>(tree, dataPack, verbose);

    treeTimer.stop();
    rdTilesTimer.start();

    /************************************
     *      process tiles
     ************************************/

    std::list<std::list<T*>*> gCPList;

    /************************************
     *      choose samples in tiles
     ************************************/
    tree.forEachLeaf([&dataPack, &gCPList](typename TiledTree::SptrTile t)
        {
            choose(t->samples.data(),
                t->chosenSamples.data(),
                t->cpList,
                t->pointsCnt,
                t->chosenPointsCnt,
                t->dim,
                dataPack.r1);

            #if defined(RD_USE_OPENMP)
            #   pragma omp critical
            {
            #endif
            gCPList.push_back(&(t->cpList));
            #if defined(RD_USE_OPENMP)
            }
            #   pragma omp atomic update
            #endif
            dataPack.ns += t->chosenPointsCnt;
            
            #ifdef RD_DEBUG
            if (t->chosenPointsCnt > t->chosenPointsCapacity)
            {
                std::cout << ">>>>>>>> Exceeding chosenPointsCapacity! chosen cnt:(" <<
                    t->chosenPointsCnt << "), capacity: (" <<
                    t->chosenPointsCapacity << ")" << std::endl;
                t->print();
                std::cout.flush();
                throw std::logic_error("Exceeding chosenPointsCapacity!");
            }
            #endif
        }
    );

    if (verbose && drawTiles)
    {
        // collect chosen points
        tree.forEachNodePreorder([&dataPack](typename TiledTree::SptrNode n)
            {
                if (n->children.empty())
                {
                    typename TiledTree::SptrTile t(n->data);
                    dataPack.S.insert(
                                dataPack.S.end(), 
                                t->chosenSamples.data(),
                                t->chosenSamples.data() + t->chosenPointsCnt * t->dim);
                }
            }
        );   

        drawAllTilesWithBoundsAndChosenSamples<EXT_TILE_TREE_MIXED_RD, TiledTree>(
            dataPack, tree);
        
        drawPointCloudAndChosenSamples<EXT_TILE_TREE_MIXED_RD>(
            dataPack, "_after_choose");
        dataPack.S.clear();
    }

    /************************************
     *      ridge detection main part
     ************************************/
    size_t oldCount = 0;

    while(oldCount != dataPack.ns)
    {
        oldCount = dataPack.ns;
        //  evolve chosen samples locally in every tile
        tree.forEachLeaf([&dataPack, cpuThreads, verbose](typename TiledTree::SptrTile t)
            {
                evolve_neighbours(t->getAllSamples(),
                    t->chosenSamples.data(),
                    t->chosenPointsCnt,
                    t->dim,
                    dataPack.r1,
                    cpuThreads,
                    verbose);
            }
        );
        // decimate chosen samples globally in all tiles
        decimateNoCopy(gCPList, dataPack.ns, dataPack.dim, dataPack.r2);
        // update chosen samples information in every tile and move data
        tree.forEachLeaf([](typename TiledTree::SptrTile t)
            {
                t->chosenPointsCnt = t->cpList.size();
                T* dstAddr = t->chosenSamples.data();
                // copy real data
                for (auto it = t->cpList.begin();
                          it != t->cpList.end();
                     dstAddr += t->dim, it++)
                {
                    if (*it != dstAddr)
                    {
                        copyTable<T>(*it, dstAddr, t->dim);
                        *it = dstAddr;
                    }
                }
                // XXX: this only for grouped tile version!
                // now can erase unnecesary elements at the end of vector. (however this unnecessarily takes time)
                // t->chosenSamples.resize(t->chosenPointsCnt);
                //  OR:
                // t->chosenSamples.erase(t->chosenSamples.begin()+t->chosenPointsCnt,
                //     t->chosenSamples.end());
            }
        );
    }

    // collect chosen points
    tree.forEachNodePreorder([&dataPack](typename TiledTree::SptrNode n)
        {
            if (n->children.empty())
            {
                typename TiledTree::SptrTile t(n->data);
                dataPack.S.insert(
                            dataPack.S.end(), 
                            t->chosenSamples.data(),
                            t->chosenSamples.data() + t->chosenPointsCnt * t->dim);
            }
        }
    );

    rdTilesTimer.stop();


    if (verbose && drawTiles)
    {
        drawEachGroupedTileWithBoundsAndChosenSamples<TiledTree>(
            dataPack, tree, Int2Type<EXT_TILE_TREE_MIXED_RD>());
    }

    /************************************
     *      end phase refinement
     ************************************/

    endPhaseTimer.start();

    if (refinement)
    {
        if (verbose)
        {
            drawPointCloudAndChosenSamples<EXT_TILE_TREE_MIXED_RD>(
                dataPack, "_before_refinement");
        }

        std::list<T*> cpList;        
        T * ptr = dataPack.S.data();
        for (size_t k = 0; k < dataPack.ns; ++k)
        {
            cpList.push_back(ptr);
            ptr += dataPack.dim;
        }

        decimate(dataPack.S.data(),
            cpList, 
            dataPack.ns,
            dataPack.dim,
            dataPack.r2);

        evolve_omp(dataPack.P,
            dataPack.S.data(),
            dataPack.np,
            dataPack.ns,
            dataPack.dim,
            dataPack.r1,
            cpuThreads);
    }

    endPhaseTimer.stop();
    wholeTimer.stop();

    float buildTreeTime = treeTimer.elapsedMillis();
    float rdTilesTime = rdTilesTimer.elapsedMillis();
    float refinementTime = endPhaseTimer.elapsedMillis();
    float wholeTime = wholeTimer.elapsedMillis();
    
    if (verbose)
    {
        std::cout << "build tree: " << buildTreeTime << "\n";
        std::cout << "ridge detection tiles: " << rdTilesTime << "\n";
        std::cout << "end phase refinement: " << refinementTime << "\n";
        std::cout << "whole: " << wholeTime << "\n";
    }
    return std::make_tuple(wholeTime, buildTreeTime, rdTilesTime, refinementTime);
}


template <typename T>
std::tuple<float, float, float, float> doRidgeDetection(
    rdTiledData<T> &    dataPack,
    Int2Type<TILED_LOCAL_TREE_LOCAL_RD>,
    int                 cpuThreads = 1,
    bool                refinement = false,
    bool                verbose = false,
    bool                drawTiles = false)
{
    typedef TiledLocalTree<T> TiledTree;

    // in case of verbose output
    rd::CpuTimer treeTimer, rdTilesTimer;
    rd::CpuTimer endPhaseTimer, wholeTimer;

    TiledTree tree;
    tree.cpuThreads = cpuThreads;

    wholeTimer.start();
    treeTimer.start();

    /************************************
     *      build tree
     ************************************/
    buildTree<TILED_LOCAL_TREE_LOCAL_RD>(tree, dataPack, verbose);

    treeTimer.stop();
    rdTilesTimer.start();

    /************************************
     *      process tiles
     ************************************/
    
    tree.forEachLeaf([&dataPack](typename TiledTree::SptrTile t)
    {
        RidgeDetection<T> rdCpu;
        rdCpu.noOMP();
        
        rdCpu.ridgeDetection(
            t->samples.data(),
            t->pointsCnt, 
            t->chosenSamples.data(),
            dataPack.r1,
            dataPack.r2,
            dataPack.dim);
        t->chosenPointsCnt = rdCpu.ns_;

        #ifdef RD_DEBUG
        if (t->chosenPointsCnt > t->chosenPointsCapacity)
        {
            std::cout << ">>>>>>>> Exceeding chosenPointsCapacity! chosen cnt:(" <<
                t->chosenPointsCnt << "), capacity: (" <<
                t->chosenPointsCapacity << ")" << std::endl;
            t->print();
            std::cout.flush();
            throw std::logic_error("Exceeding chosenPointsCapacity!");
        }
        #endif
    });

    // collect chosen points
    tree.forEachNodePreorder([&dataPack](typename TiledTree::SptrNode n)
        {
            if (n->children.empty())
            {
                typename TiledTree::SptrTile t(n->data);
                dataPack.ns += t->chosenPointsCnt;
                dataPack.S.insert(
                            dataPack.S.end(), 
                            t->chosenSamples.data(),
                            t->chosenSamples.data() + t->chosenPointsCnt * t->dim);
            }
        }
    );

    rdTilesTimer.stop();

    if (verbose && drawTiles)
    {
        drawEachLocalTileWithBoundsAndChosenSamples<TILED_LOCAL_TREE_LOCAL_RD, TiledTree>(
            dataPack, tree);
    } 

    /************************************
     *      end phase refinement
     ************************************/

    endPhaseTimer.start();

    if (refinement)
    {
        if (verbose)
        {
            drawPointCloudAndChosenSamples<TILED_LOCAL_TREE_LOCAL_RD>(
                dataPack, "_before_refinement");
        }

        std::list<T*> cpList;        
        T * ptr = dataPack.S.data();
        for (size_t k = 0; k < dataPack.ns; ++k)
        {
            cpList.push_back(ptr);
            ptr += dataPack.dim;
        }

        decimate(dataPack.S.data(),
            cpList, 
            dataPack.ns,
            dataPack.dim,
            dataPack.r2);

        evolve_omp(dataPack.P,
            dataPack.S.data(),
            dataPack.np,
            dataPack.ns,
            dataPack.dim,
            dataPack.r1,
            cpuThreads);
        
    }

    endPhaseTimer.stop();
    wholeTimer.stop();

    float buildTreeTime = treeTimer.elapsedMillis();
    float rdTilesTime = rdTilesTimer.elapsedMillis();
    float refinementTime = endPhaseTimer.elapsedMillis();
    float wholeTime = wholeTimer.elapsedMillis();
    
    if (verbose)
    {
        std::cout << "build tree: " << buildTreeTime << "\n";
        std::cout << "ridge detection tiles: " << rdTilesTime << "\n";
        std::cout << "end phase refinement: " << refinementTime << "\n";
        std::cout << "whole: " << wholeTime << "\n";
    }
    return std::make_tuple(wholeTime, buildTreeTime, rdTilesTime, refinementTime);
}

template <typename T>
std::tuple<float, float, float, float> doRidgeDetection(
    rdTiledData<T> &    dataPack,
    Int2Type<TILED_LOCAL_LIST>,
    int                 cpuThreads = 1,
    bool                refinement = false,
    bool                verbose = false,
    bool                drawTiles = false)
{
    return std::make_tuple(0.f, 0.f, 0.f, 0.f);
}

template <typename T>
std::tuple<float, float, float, float> doRidgeDetection(
    rdTiledData<T> &    dataPack,
    Int2Type<TILED_GROUPED_TREE_LOCAL_RD>,
    int                 cpuThreads = 1,
    bool                refinement = false,
    bool                verbose = false,
    bool                drawTiles = false)
{
   
    typedef TiledGroupTree<FIND_NEIGHBOURS, T> TiledTree;

    // in case of verbose output
    rd::CpuTimer treeTimer, rdTilesTimer;
    rd::CpuTimer endPhaseTimer, wholeTimer;

    TiledTree tree;
    tree.cpuThreads = cpuThreads;

    wholeTimer.start();
    treeTimer.start();

    /************************************
     *      build tree
     ************************************/
    buildTree<TILED_GROUPED_TREE_LOCAL_RD>(tree, dataPack, verbose);

    treeTimer.stop();
    rdTilesTimer.start();

    /************************************
     *      process tiles
     ************************************/

    tree.forEachLeaf([&dataPack, verbose](typename TiledTree::SptrTile t)
    {
        RidgeDetection<T> rdCpu;
        // #ifdef RD_DEBUG
        //     rdCpu.verbose_ = verbose;
        // #endif
        rdCpu.noOMP();
        
        rdCpu.ridgeDetection(
            t->getAllSamples(),
            t->chosenSamples.data(),
            dataPack.r1,
            dataPack.r2,
            dataPack.dim);
        t->chosenPointsCnt = rdCpu.ns_;
        
        #ifdef RD_DEBUG
        if (t->chosenPointsCnt > t->chosenPointsCapacity)
        {
            std::cout << ">>>>>>>> Exceeding chosenPointsCapacity! chosen cnt:(" <<
                t->chosenPointsCnt << "), capacity: (" <<
                t->chosenPointsCapacity << ")" << std::endl;
            t->print();
            std::cout.flush();
            throw std::logic_error("Exceeding chosenPointsCapacity!");
        }
        #endif
    });

    // collect chosen points
    tree.forEachNodePreorder([&dataPack](typename TiledTree::SptrNode n)
        {
            if (n->children.empty())
            {
                typename TiledTree::SptrTile t(n->data);
                dataPack.ns += t->chosenPointsCnt;
                dataPack.S.insert(
                            dataPack.S.end(), 
                            t->chosenSamples.data(),
                            t->chosenSamples.data() + t->chosenPointsCnt * t->dim);
            }
        }
    );

    rdTilesTimer.stop();

    if (verbose && drawTiles)
    {
        drawEachGroupedTileWithBoundsAndChosenSamples<TILED_GROUPED_TREE_LOCAL_RD, TiledTree>(
            dataPack, tree);
    }

    /************************************
     *      end phase refinement
     ************************************/

    endPhaseTimer.start();

    if (refinement)
    {
        if (verbose)
        {
            drawPointCloudAndChosenSamples<TILED_GROUPED_TREE_LOCAL_RD>(
                dataPack, "_before_refinement");
        }

        std::list<T*> cpList;        
        T * ptr = dataPack.S.data();
        for (size_t k = 0; k < dataPack.ns; ++k)
        {
            cpList.push_back(ptr);
            ptr += dataPack.dim;
        }
        
        decimate(dataPack.S.data(),
            cpList, 
            dataPack.ns,
            dataPack.dim,
            dataPack.r2);

        evolve_omp(dataPack.P,
            dataPack.S.data(),
            dataPack.np,
            dataPack.ns,
            dataPack.dim,
            dataPack.r1,
            cpuThreads);
    }

    endPhaseTimer.stop();
    wholeTimer.stop();

    float buildTreeTime = treeTimer.elapsedMillis();
    float rdTilesTime = rdTilesTimer.elapsedMillis();
    float refinementTime = endPhaseTimer.elapsedMillis();
    float wholeTime = wholeTimer.elapsedMillis();
    
    if (verbose)
    {
        std::cout << "build tree: " << buildTreeTime << "\n";
        std::cout << "ridge detection tiles: " << rdTilesTime << "\n";
        std::cout << "end phase refinement: " << refinementTime << "\n";
        std::cout << "whole: " << wholeTime << "\n";
    }

    return std::make_tuple(wholeTime, buildTreeTime, rdTilesTime, refinementTime);
}


}   // end namespace detail

//==========================================================================
//  User api
//==========================================================================

/**
 * @class      RidgeDetection
 * @brief      CPU version of ridge detection algorithm.
 * @note       Capable of using multiple threads with OpenMP.
 *
 * @paragraph Short algorithm description: The algorithm first choose some set of
 * points, wchich are called 'chosen points' or 'sphere centers'. Every two
 * points in this set are R separated, wchis means the distance between them is
 * at least R. After operation of choosing the main loop of the algorithm takes
 * place. It consists of two operations: evolution and decimation. During
 * evolution each 'chosen point' is shifted towards the mass center of all
 * samples wchich falls into the intersection of Voronoi cell (constructed for
 * all 'chosen points') of this point and the sphere with radius R (centered in
 * this point). During decimation excess points are removed from 'chosen set'.
 * Each point which has more than two other points in his 2R-neighbourhood or
 * has less than two other points in his 4R-neighbourhood is discarded. These
 * operations are repeted untill there will be no change in number of chosen
 * points in two consecutive iterations.
 *
 *
 * @tparam     T     Samples data type
 */
template <
    TiledRDAlgorithm    ALGORITHM,
    typename                        T>
std::tuple<float, float, float, float> tiledRidgeDetection(
    rdTiledData<T> &    dataPack,
    int                 cpuThreads = 1,
    bool                endPhaseRefinement = false,
    bool                verbose = false,
    bool                drawTiles = false)
{
    std::tuple<float, float, float, float> performance = detail::doRidgeDetection(
        dataPack,
        Int2Type<ALGORITHM>(),
        cpuThreads,
        endPhaseRefinement,
        verbose,
        drawTiles);

    /************************************
     * all tiles with chosen samples marked
     ************************************/
    if (verbose)
    {
        if (dataPack.dim <= 3)
        {
            detail::drawPointCloudAndChosenSamples<ALGORITHM>(
                dataPack, "_final_chosen_samples");
        }
        else
        {
            printTable(dataPack.S.data(), dataPack.dim, dataPack.S.size(), "chosen samples: ");
        }
    }
    
    return performance;
}

}   // end namespace tiled

}  // namespace rd

#endif /* RD_TILED_HPP */
