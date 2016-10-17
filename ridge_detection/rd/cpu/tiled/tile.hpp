/**
 * @file tile.hpp
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

#ifndef TILE_HPP
#define TILE_HPP

#include "rd/utils/bounding_box.hpp"

#include <vector>
#include <utility>
#include <memory>
#include <iostream>
#include <cstddef>
#include <list>

namespace rd
{
namespace cpu
{
namespace tiled
{

/**
 * @brief      A tile containing points.
 *
 * @paragraph LocalSamplesTileDescription (Description) This structure represents a n-dimensional
 * cuboid space segment with points lying within its boundaries.
 *
 * @tparam     T     Point coordinates data type.
 */
template <typename T>
struct LocalSamplesTile
{

    #ifdef RD_DEBUG
        // needed only for debugging purposes
        static int objCounter;
    #endif
    /// Global tile counter.
    static int idCounter;
    /// Unique tile id. Used while drawing tiles and during finding neigbours as well as for 
    /// identification during debugging.
    int id;

    /// Number of points residing within tile boundaries.
    size_t pointsCnt;
    /// Number of chosen points connected with this tile.
    size_t chosenPointsCnt;
    /// Maximum number of chosen points that this tile can contain. (It's restricted by its
    ///  volume).
    size_t chosenPointsCapacity;
    /// Tile points dimensionality.
    size_t dim;
    /// List of pointers to chosen points. Ridge detection decimation phase works on pointers in 
    /// order to not perform expensive memory copying in each iteration.
    std::list<T*> cpList;
    /// Vector of points lying within this tile boundaries.
    std::vector<T> samples;
    /// Vector of chosen points.
    std::vector<T> chosenSamples;
    /// This tile boundaries.
    BoundingBox<T> bounds;

    LocalSamplesTile() 
    : 
        pointsCnt(0), chosenPointsCnt(0), chosenPointsCapacity(0), dim(0)
    {
        id = idCounter++;
        #ifdef RD_DEBUG
            objCounter++;
            std::cout << "LocalSamplesTile() id: " << id 
                    << " objCounter: " << objCounter 
                    << std::endl;
        #endif
    }

    LocalSamplesTile(LocalSamplesTile const &) = default;
    LocalSamplesTile(LocalSamplesTile &&) = default;
    LocalSamplesTile & operator=(LocalSamplesTile const &) = default;
    LocalSamplesTile & operator=(LocalSamplesTile &&) = default;

    #ifdef RD_DEBUG
    virtual ~LocalSamplesTile()
    {
            objCounter--;
            std::cout << "~LocalSamplesTile() id: " << id
                    << ", pointsCnt: " << pointsCnt
                    << ", chosenPointsCnt: " << chosenPointsCnt
                    << ", TileCounter: " << objCounter << std::endl;
            bounds.print();
    }
    #else
    virtual ~LocalSamplesTile() = default;
    #endif

    void print()
    {
        std::cout << "-------------------------------\n"
        #ifdef RD_DEBUG
            << ", TileCounter: " << objCounter
        #endif 
            << "\nLocalSamplesTile, id: " << id
            << ", pointsCnt: " << pointsCnt
            << ", chosenPointsCnt: " << chosenPointsCnt
            << ", chosenPointsCapacity: " << chosenPointsCapacity
            << "\n";
        bounds.print();
    }
};

#ifdef RD_DEBUG
    template <typename T>
    int LocalSamplesTile<T>::objCounter = 0;
#endif
template <typename T>
int LocalSamplesTile<T>::idCounter = 0;


/**
 * @brief      Describes scheme used while building grouped tiled tree
 */
enum GroupTreeBuildPolicy
{
    /**
     * With this scheme after all tiles have been built, there is additional neighbour search
     * in the end of building tree process. Each tile search for its neighbours. In this way
     * each tiles acquires information how approximated curve behaves in its neighbourhood which 
     * helps to get better approximation.
     */
    FIND_NEIGHBOURS,
    /**
     * With this scheme during point cloud partitioning into respective tiles, each tile not only 
     * collects points falling into its boundaries, but also collects points lying in its 'near
     * neighbourhood'. This enables to acquire more global information about reconstructed curve
     * in this part of the space. In contrast to the FIND_NEIGHBOURS policy, this tile type
     * possess data only relevant to its region, thus minimizing amount of work.
     */
    EXTENDED_TILE
};

/**
 * @brief      Tile containing samples and knowing it's neighbours.
 *
 * @tparam     BUILD_POLICY  Building tile policy.
 * @tparam     T             Point's coordinates data type.
 */
template <
    GroupTreeBuildPolicy    BUILD_POLICY,
    typename                T>
struct GroupedSamplesTile : public LocalSamplesTile<T>
{
};

template <typename T>
struct GroupedSamplesTile<FIND_NEIGHBOURS, T> : public LocalSamplesTile<T>
{
    /**
     * Vector of shared pointers to neighbouring tiles. Storing information in this way prevents
     * redundation.
     */
    std::vector<std::shared_ptr<GroupedSamplesTile<FIND_NEIGHBOURS, T>>> neighbours;

    GroupedSamplesTile()    
    :   LocalSamplesTile<T>()
    {
    }

    /**
     * @brief      Get all points connected with this tile. That is points lying within this tile's
     *             boundaries as well as its neighbour tiles points.
     *
     * @return     Returns vector of pairs which first element is pointer to samples and second
     *             element is number of points.
     */
    std::vector<std::pair<T const *, size_t>> getAllSamples() const
    {
        std::vector<std::pair<T const *, size_t>> nSamples;
        nSamples.push_back(std::make_pair(this->samples.data(), this->pointsCnt));
        for (auto const &n : neighbours)
        {
            nSamples.push_back(std::make_pair(n->samples.data(), n->pointsCnt));
        }
        return nSamples;
    }
};

template <typename T>
struct GroupedSamplesTile<EXTENDED_TILE, T> : public LocalSamplesTile<T>
{
    /**
     * Vector of points lying close to this tile boundaries.
     */
    std::vector<T> neighbours;

    GroupedSamplesTile()    
    :   LocalSamplesTile<T>()
    {
    }

    /**
     * @brief      Get all points connected with this tile. That is points lying within this tile's
     *             boundaries, as well as points lying nearby.
     *
     * @return     Returns vector of pairs which first element is pointer to samples and second
     *             element is number of points.
     */
    std::vector<std::pair<T const *, size_t>> getAllSamples() const
    {
        std::vector<std::pair<T const *, size_t>> nSamples;
        nSamples.push_back(std::make_pair(this->samples.data(), this->pointsCnt));
        nSamples.push_back(std::make_pair(neighbours.data(), neighbours.size() / this->dim));

        return nSamples;
    }
};



}   // end namespace tiled
}   // end namespace cpu
}   // end namespace rd

#endif  // TILE_HPP
