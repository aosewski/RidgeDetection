/**
 * @file sim_state_desc.hpp
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

#ifndef SIM_STATE_DESC_HPP
#define SIM_STATE_DESC_HPP

#include <stdexcept>

namespace rd
{
    
namespace vis
{


struct SimPhaseTag {};
struct ChosePhaseTag : SimPhaseTag {};
struct EvolvePhaseTag : SimPhaseTag {};
struct EvolveSphereCellsDiagramPhaseTag : EvolvePhaseTag {};
struct EvolveMassCentersShiftPhaseTag : EvolvePhaseTag {};
struct DecimatePhaseTag : SimPhaseTag {};


/**
 * @struct SimStateDesc
 * @brief      Describes simulation state.
 */
struct SimStateDesc
{
    // simulation is running until there's no change in 
    // chosen count in two consecutive iterations
    size_t chosenCount;
    size_t prevChosenCount;

    virtual SimPhaseTag* getCurrentPhaseTag() const = 0;

    SimStateDesc() : chosenCount(0), prevChosenCount(0)
    {
    }

    virtual ~SimStateDesc() 
    {
    }
};

/**
 * @struct ChosePhaseDesc
 * @brief   Describes choose phase state.
 */
template <typename T>
struct ChosePhaseDesc : SimStateDesc
{

    /// Position in samples array which we are currently
    /// processing
    size_t idx;
    /// Pointer to currently processed point in samples 
    /// table
    T *currPoint;
    // min neighbourhood distance squared
    T r2;

    ChosePhaseDesc() : SimStateDesc(), idx(0), currPoint(nullptr) 
    {
    }

    SimPhaseTag* getCurrentPhaseTag() const
    {
        return ChosePhaseTag();
    }

};

/**
 * @struct EvolvePhaseDesc
 * @brief Describes evolve phase state.
 */
template <typename T>
struct EvolvePhaseDesc : SimStateDesc
{
    /// container for points coordinates sums which are falling 
    /// into respective sphere cell 
    T *cordSums;
    int *spherePointCount;
    /// whether there was a numerically visible change in mass center
    /// shift during current iteration
    bool contFlag;
    // sphere radius squared
    T r2;
    /// stores idx to chosen samples array for each point in samples set
    int *sphereCellIdx;

    enum EVOLVE_PART {
        SPHERE_CELLS_DIAGRAM,
        MASS_CENTERS_SHIFT
    } part;

    EvolvePhaseDesc() : SimStateDesc(), cordSums(nullptr), 
            spherePointCount(nullptr), contFlag(true), r2(0), 
            sphereCellIdx(nullptr), part(SPHERE_CELLS_DIAGRAM)
    {   
    }

    virtual ~EvolvePhaseDesc()
    {
        if (cordSums != nullptr)            delete[] cordSums;
        if (spherePointCount != nullptr)    delete[] spherePointCount;
        if (sphereCellIdx != nullptr)       delete[] sphereCellIdx;
    }

    SimPhaseTag* getCurrentPhaseTag() const
    {
        switch (part)
        {
            case SPHERE_CELLS_DIAGRAM:
                return EvolveSphereCellsDiagramPhaseTag();
                break;
            case MASS_CENTERS_SHIFT:
                return EvolveMassCentersShiftPhaseTag();
                break;
            default:
                throw std::invalid_argument("Unsupported phase tag!");
        }
    }
};

/**
 * @struct DecimatePhaseDesc
 * @brief Describes simulation decimation phase state.
 */
template <typename T>
struct DecimatePhaseDesc : SimStateDesc
{
    // sphere radius squared
    T r2;
    // number of chosen samples left after previous iteration
    size_t left;
    // stores indexes of chosen samples marked to delete
    int *decimateIdx;
    // index of currently checked point
    int currIdx;
    // counter of points makred to delete
    int deleteCount;

    DecimatePhaseDesc() : SimStateDesc(), r2(0), left(0), decimateIdx(nullptr),
        currIdx(0), deleteCount(0)
    {
    }

    virtual ~DecimatePhaseDesc()
    {
        if (decimateIdx != nullptr) delete[] decimateIdx;
    }

    SimPhaseTag* getCurrentPhaseTag() const
    {
        return DecimatePhaseTag();
    }
};


} // end namespace vis
  
} // end namespace rd

#endif // SIM_STATE_DESC_HPP
