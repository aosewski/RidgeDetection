/**
 * @file tile.hpp
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
namespace tiled
{

/**
 * @brief      Tile containing points.
 *
 * @tparam     T     Samples data type.
 */
template <typename T>
struct LocalSamplesTile
{

    #ifdef RD_DEBUG
        static int objCounter;
    #endif
    static int idCounter;
    int id;

    size_t pointsCnt;
    size_t chosenPointsCnt;
    size_t chosenPointsCapacity;
    size_t dim;
    std::list<T*> cpList;
    std::vector<T> samples;
    std::vector<T> chosenSamples;
    BoundingBox<T> bounds;

    LocalSamplesTile() 
    : 
        pointsCnt(0), chosenPointsCnt(0), chosenPointsCapacity(0), dim(0)
    {
        id = idCounter++;
        #ifdef RD_DEBUG
            objCounter++;
            std::cout << "LocalSamplesTile() id: " << id << " objCounter: " << objCounter << std::endl;
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


enum GroupTreeBuildPolicy
{
    FIND_NEIGHBOURS,
    EXTENDED_TILE
};

/**
 * @brief      Tile containing samples and knowing it's neighbours.
 *
 * @tparam     T     Samples data type.
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
    std::vector<std::shared_ptr<GroupedSamplesTile<FIND_NEIGHBOURS, T>>> neighbours;

    GroupedSamplesTile()    
    :   LocalSamplesTile<T>()
    {
    }

    /**
     * @brief      returns vector of pairs which first element is pointer to samples and second element is number of
     *             points.
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
    std::vector<T> neighbours;

    GroupedSamplesTile()    
    :   LocalSamplesTile<T>()
    {
    }

    /**
     * @brief      returns vector of pairs which first element is pointer to samples and second element is number of
     *             points.
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
}   // end namespace rd

#endif  // TILE_HPP
