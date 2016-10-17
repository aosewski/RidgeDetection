/**
 * @file local_tile_tree.hpp
 * @author Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of 
 *  estimation of multidimensional random variable density function ridge
 *  detection algorithm.",
 * which is supervised by prof. dr hab. inż. Marek Nałęcz.
 * 
 * Institute of Control and Computation Engineering
 * Faculty of Electronics and Information Technology
 * Warsaw University of Technology 2016
 */


#ifndef LOCAL_TILE_TREE_HPP
#define LOCAL_TILE_TREE_HPP

#include "rd/utils/bounding_box.hpp"
#include "rd/utils/histogram.hpp"
#include "rd/cpu/tiled/tile.hpp"
#include "rd/cpu/tiled/tree_node.hpp"

#include <vector>
#include <memory>
#include <cstddef>
#include <string>

namespace rd
{
namespace tiled
{

template <typename T>             
class TiledLocalTree
{

public:

    typedef LocalSamplesTile<T>         Tile;
    typedef TreeNode<Tile>              Node;
    typedef std::shared_ptr<Node>       SptrNode;
    typedef std::shared_ptr<Tile>       SptrTile;

    size_t maxTileCapacity;
    // Ridge detection choose phase parameter
    // Needed for chosen samples count estimation.
    T sphereRadius;
    // number of threads to use when running forEachLeaf function
    int cpuThreads;

    TiledLocalTree();
    virtual ~TiledLocalTree();
    

    /**
     * @brief      Build a Tiled Local Tree.
     *
     * @param[in]  samples          Pointer to source data.
     * @param[in]  cnt              Number of points in @p samples set.
     * @param[in]  dim              Points dimension
     * @param[in]  initTileCnt      Initial number of tiles.
     * @param[in]  maxTileCapacity  Max samples count in one tile.
     */
    void buildTree(T const *samples,
                    size_t cnt,
                    size_t dim,
                    std::vector<size_t> const &initTileCnt);

    /**
     * @brief      Visit each tree node and perform @p f function on it.
     *
     * @param      f              Functor object defining function to perform.
     *
     * @tparam     UnaryFunction  Functor class with defined function to perform
     * on nodes' data.
     */
    template <typename UnaryFunction>
    void forEachNodePreorder(UnaryFunction const &f);

    /**
     * @brief      Perform @p f function on each trees' leaf.
     *
     * @param      f              Functor object with defined operator() to perform on leaf.
     *
     * @tparam     UnaryFunction  Functor class with defined function to perform on leafs' data.
     */
    template <typename UnaryFunction>
    void forEachLeaf(UnaryFunction const &f);

    void print() const;

protected:

    TreeSentinel<Node> sentinel_;
    SptrNode root_;

    std::vector<SptrTile> leafs_;

    /**
     * @brief      Removes all children and data connected to @p node.
     *
     * @param      node  The subtree root which we clear.
     */
    void clear(SptrNode &node);

    /**
     * @brief      Creates tiles containing given samples set.
     *
     * @param      samples     Ptr to samples data.
     * @param[in]  pointsCnt   Number of points in @p samples set.
     * @param[in]  dim         Points dimension
     * @param[out] outTiles    Vector to store created tiles.
     * @param[in]  tileCnt     Number of tiles in each dimension.
     */
    void createTiles(T const *samples,
                    size_t pointsCnt,
                    size_t dim,
                    std::vector<Tile*> &outTiles,
                    std::vector<size_t> const &tileCnt);

    /**
     * @brief      Creates tiles containing given samples set.
     *
     * @param      samples      Ptr to samples data.
     * @param[in]  pointsCnt    Number of points in @p samples set.
     * @param[in]  dim          Points dimension
     * @param      hist         Histogram created for given @p samples and 
     *             @p tileCnt bins.
     * @param[out] outTiles        Vector to store created tiles.
     * @param[in]  tileCnt      Number of tiles in each dimension.
     * @param      inSamplesBbox  Bounding box describing @p samples.
     */
    void createTiles(T const *samples,
                    size_t pointsCnt,
                    size_t dim,
                    Histogram<T> const &hist,
                    std::vector<Tile*> &outTiles,
                    std::vector<size_t> const &tileCnt,
                    BoundingBox<T> const &inSamplesBbox);

    /**
     * @brief      Adds new node @p tile to the @p parent tile
     *
     * @param[in]  parent     Tile to which we add new one.
     * @param[in]  tile       The tile we hook to @p parent.
     * @param[in]  treeLevel  Tree level.
     */
    void addNode(SptrNode &parent, Tile *tile, size_t treeLevel);

    /**
     * @brief      Subdivides @p tile into @p subTiles
     *
     * @param[in]  tile      The tile to subdivide.
     * @param[out] outSubTiles  Storage for subtiles.
     */
    void subdivideTile(Tile const *tile,
    					std::vector<Tile*> &outSubTiles,
                        size_t treeLevel);

    /**
     * @brief      Function mapping points coordinates to histogram bin idx.
     *
     * @param      sample   Pointer to point's coordinates.
     * @param[in]  binsCnt  Number of bins in respective dimensions.
     * @param[in]  bb       Bounding box describing set containing @sample.
     *
     * @return     Returns idx in row-major order.
     */
    size_t histogramMapFunc(T const *sample,
                            std::vector<size_t> const &binsCnt,
                            BoundingBox<T>const &bb) const;

    /**
     * @brief      Visits @p node and all of its children and performs @p f
     *              function on each data.
     *
     * @param[in]  node           Node we want to visit.
     * @param      f              Functor object with function to perform.
     *
     * @tparam     UnaryFunction  Functor object with function which gets
     *              SptrNode.
     */
    template <typename UnaryFunction>
    void visitNode(SptrNode node, UnaryFunction const &f);


    /**
     * @brief      Traverse through entire tree and adds all leafs to private container.
     */
    void collectLeafs();
};

}   // end namespace tiled    
}   // namespace rd

#include "local_tile_tree.inl"

#endif // LOCAL_TILE_TREE_HPP
