/**
 * @file grouped_tile_tree.hpp
 * @author Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering, Faculty of Electronics and Information
 * Technology, Warsaw University of Technology 2016
 */


#ifndef GROUPED_TILE_TREE_HPP
#define GROUPED_TILE_TREE_HPP

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
namespace cpu
{
namespace tiled
{

/**
 * @brief      Base class representing tree with grouped tiles.
 *
 * @tparam     BUILD_POLICY  Describes scheme used during tree building.
 * @tparam     T             Input points' coordinates data type.
 */
template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename T>             
class TiledGroupTreeBase
{

public:
    /// Defines Tile type stored by this tree.
    typedef GroupedSamplesTile<BUILD_POLICY, T> Tile;
    /// Defines Node typed used by this tree.
    typedef TreeNode<Tile>              Node;
    /// Shared pointer Node type
    typedef std::shared_ptr<Node>       SptrNode;
    /// Shared pointer Tile type
    typedef std::shared_ptr<Tile>       SptrTile;

    /// Maximum number of points that single tile can contain.
    size_t maxTileCapacity;
    /// Ridge detection choose phase parameter. Needed for chosen samples count estimation.
    T sphereRadius;
    /// number of threads to use when running forEachLeaf function
    int cpuThreads;

    TiledGroupTreeBase();
    virtual ~TiledGroupTreeBase();
    
    /**
     * @brief      Visit each tree node and perform @p f function on it.
     *
     * @param      f              Functor object defining function to perform.
     *
     * @tparam     UnaryFunction  Functor class with defined function to perform on nodes' data.
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

    /**
     * @brief      Prints informations about this tree's nodes and tiles.
     */
    void print() const;

protected:
    TreeSentinel<Node> sentinel_;
    SptrNode root_;

    /// Vector containing tree's leaf node's tiles.
    std::vector<SptrTile> leafs_;

    /**
     * @brief      Build a Tree.
     *
     * @param[in]  samples      Pointer to source data.
     * @param[in]  cnt          Number of points in @p samples set.
     * @param[in]  dim          Points dimension.
     * @param[in]  initTileCnt  Initial number of tiles.
     * @param      func  Functor with data partition function.
     *
     * @tparam     PartitionFunc  Functor implementing partition function
     */
    virtual void buildTree(T const *samples,
                    size_t cnt,
                    size_t dim,
                    std::vector<size_t> const &initTileCnt) = 0;

    /**
     * @brief      Initialize tree's root and sentinel
     */
    void initTreeRoot();

    /**
     * @brief      Creates tiles containing given samples set.
     *
     * @param      vSamples     Vector with samples and their cardinality.
     * @param[in]  dim          Samples dimension.
     * @param[out] tiles        Vector to store created tiles.
     * @param[in]  tileCnt      Number of tiles in each dimension.
     * @param      samplesBbox  Bounding box describing @p samples.
     */
    void createTiles(std::vector<std::pair<T const *, size_t>> const &vSamples,
                     size_t dim,
                     std::vector<Tile*> &tiles,
                     std::vector<size_t> const &tileCnt,
                     BoundingBox<T> const &samplesBbox);

    /**
     * @brief      Allocate tiles and initializes their bounds
     *
     * @param[in]  dim          Samples dimension
     * @param      tiles        Vector for tiles
     * @param      tileCnt      Number of tile in each dimension.
     * @param      samplesBbox  Point cloud bounding box.
     */
    void allocTiles(size_t dim,
                    std::vector<Tile*> &tiles,
                    std::vector<size_t> const &tileCnt,
                    BoundingBox<T> const &samplesBbox);

    /**
     * @brief      Partition @p samples onto @p tiles
     *
     * @param      vSamples    Vector containing samples and their cardinality.
     * @param[in]  dim         Samples dimension
     * @param      tiles       Vector containing tiles to partition data onto.
     */
    virtual void partitionSamples(
                    std::vector<std::pair<T const *, size_t>> const &vSamples,
                    size_t dim,
                    std::vector<Tile*> &tiles) = 0;

    /**
     * @brief      Removes empty tiles and ones with only one sample and
     *             allocate storage for chosen samples.
     *
     * @param      tiles  Tiles to reduce and allocate storage for.
     */
    void reduceTilesAndAllocChosenSamples(std::vector<Tile*> &tiles);

    /**
     * @brief      Adds new node @p tile to the @p parent tile
     *
     * @param[in]  parent     Tile to which we add new one.
     * @param[in]  tile       The tile we hook to @p parent.
     * @param[in]  treeLevel  Tree level.
     */
    void addNode(SptrNode &parent,
                 Tile *tile, 
                 size_t treeLevel);

    /**
     * @brief      Subdivides @p tile into @p subTiles
     *
     * @param[in]  tile      The tile to subdivide.
     * @param[out] subTiles  Storage for subtiles.
     */
    void subdivideTile(Tile const *tile,
                        std::vector<Tile*> &subTiles,
                        size_t treeLevel);

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
     * @brief      Removes all children and data connected to @p node.
     *
     * @param      node  The subtree root which we clear.
     */
    void clear(SptrNode &node);

    /**
     * @brief      Traverse through entire tree and adds all leafs to private container.
     */
    void collectLeafs();
};


//===============================================================
//      specializations of TiledGroupTree
//===============================================================


/**
 * @brief      Class representing tree with grouped tiles, that is with tiles having some 
 * knowledge about their neighbourhood.
 *
 * @tparam     BUILD_POLICY  Tree build policy, also defining used tile type.
 * @tparam     T             Point coordinates data type.
 */
template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename T>             
class TiledGroupTree : public TiledGroupTreeBase<BUILD_POLICY, T>
{

};

template <typename T>             
class TiledGroupTree<EXTENDED_TILE, T> : public TiledGroupTreeBase<EXTENDED_TILE, T>
{
public:
    typedef TiledGroupTreeBase<EXTENDED_TILE, T>    BaseType;
    typedef typename BaseType::Tile                 Tile;
    typedef typename BaseType::Node                 Node;
    typedef typename BaseType::SptrNode             SptrNode;

    TiledGroupTree();
    virtual ~TiledGroupTree();

    /**
     * @brief      Build a Tree.
     *
     * @param[in]  samples      Pointer to source data.
     * @param[in]  cnt          Number of points in @p samples set.
     * @param[in]  dim          Points dimension
     * @param[in]  initTileCnt  Initial number of tiles.
     */
    void buildTree(T const *samples,
                    size_t cnt,
                    size_t dim,
                    std::vector<size_t> const &initTileCnt) override;

private:
    /**
     * @brief      Partition @p samples onto @p tiles
     *
     * @param      vSamples    Vector containing samples and their cardinality.
     * @param[in]  dim         Samples dimension
     * @param      tiles       Vector containing tiles to partition data onto.
     */
    void partitionSamples(std::vector<std::pair<T const *, size_t>> const &vSamples,
                          size_t dim,
                          std::vector<Tile*> &tiles) override;
};

template <typename T>             
class TiledGroupTree<FIND_NEIGHBOURS, T> : public TiledGroupTreeBase<FIND_NEIGHBOURS, T>
{
public:
    typedef TiledGroupTreeBase<FIND_NEIGHBOURS, T>  BaseType;
    typedef typename BaseType::Tile                 Tile;
    typedef typename BaseType::Node                 Node;
    typedef typename BaseType::SptrNode             SptrNode;

    TiledGroupTree();
    virtual ~TiledGroupTree();

    /**
     * @brief      Build a Tree.
     *
     * @param[in]  samples        Pointer to source data.
     * @param[in]  cnt            Number of points in @p samples set.
     * @param[in]  dim            Points dimension
     * @param[in]  initTileCnt    Initial number of tiles.
     */
    void buildTree(T const *samples,
                    size_t cnt,
                    size_t dim,
                    std::vector<size_t> const &initTileCnt) override;

private:

    /**
     * @brief      Partition @p samples onto @p tiles
     *
     * @param      vSamples    Vector containing samples and their cardinality.
     * @param[in]  dim         Samples dimension
     * @param      tiles       Vector containing tiles to partition data onto.
     */
    void partitionSamples(std::vector<std::pair<T const *, size_t>> const &vSamples,
                          size_t dim,
                          std::vector<Tile*> &tiles) override;

    /**
     * @brief         Searches for tiles neighbouring with tile connected to @p
     *                node and connect them.
     *
     * @param[in]     currNode  Currently examined node.
     * @param[in|out] refNode   Node with tile we search neighbours for.
     */
    void findNeighbourTiles(SptrNode const currNode, SptrNode refNode);

};

}   // end namespace tiled    
}   // end namespace cpu    
}   // namespace rd

#include "grouped_tile_tree.inl"

#endif // GROUPED_TILE_TREE_HPP
