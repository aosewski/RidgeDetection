/**
 * @file rd_simulation_vis.hpp
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


#ifndef CPU_SIMULATION_STEP_HPP
#define CPU_SIMULATION_STEP_HPP

#include "../../utils/sim_state_desc.hpp"

#include <list>

#if defined(RD_USE_OPENMP)
    #include <omp.h>
#endif

namespace rd
{
    
namespace vis
{

/**
 * @class      RDSimVis
 * @brief      CPU Ridge detection algorithm version with API designed for
 *             algorithm simulation visualisation.
 *
 * @tparam     T     Samples data type.
 *
 * @see        RidgeDetection
 */
template <typename T>
class RDSimVis
{
public:

    /// samples dimension
    size_t dim_;
    /// cardinality of samples set
    size_t np_;
    /// cardinality of initial choosen samples set
    size_t ns_;
    /**
     * @var r1_
     * Algorithm parameter. Radius used for choosing samples
     * and in evolve phase.
     */
    T r1_;
    /**
     * @var r2_
     * Algorithm parameter. Radius used for decimation phase.
     */
    T r2_;
    /// table containing samples
    T *P_;
    /// table containing choosen samples
    T *S_;

    RDSimVis();
    virtual ~RDSimVis();


    /**
     * @brief - Sets the number of threads to use.
     * @param t - number of threds to run
     */
    void ompSetNumThreads(int t);
        
    /**
     * @brief      Start choose phase of ridge detection algorithm.
     */
    void choose();
    /**
     * @brief      Start evolve phase of ridge detection algorithm.
     */
    void evolve();
    /**
     * @brief      Start decimate phase of ridge detection algorithm.
     */
    void decimate();
    /**
     * @brief      Plays whole algorithm simulation.
     */
    void run();
    /**
     * @brief      Performs one step of ridge detection algorithm.
     * 
     * The step depends on the algorithm current phase.
     */
    void step();

    /**
     * @brief      Runs the algorithm from current state till the end.
     */
    void finish();

    /**
     * @brief      Runs current phase till the end.
     */
    void finishPhase();

    /**
     * @brief      Informs about current phase of the algorithm.
     *
     * @return     Pointer to current algorithm phase descriptor.
     */
    SimPhaseTag const * getCurrentPhaseDesc() const
    {
        return stateDesc;
    }

protected:

    // List containing chosen points
    std::list<T*> SList_;
    /// number of threads used to parallelize calculations
    int ompNumThreads_;


private:

    SimStateDesc *stateDesc;


    /**
     * @brief      Performs one step of choose algorithm phase
     *
     * @param      desc  Phase state descriptor
     *
     * @return     True if phase is still running, false otherwise
     */
    bool doStep(ChosePhaseDesc *desc);

    /**
     * @brief      Performs one step of evolve algorithm phase
     *
     * @param      desc  Phase state descriptor
     *
     * @return     True if phase is still running, false otherwise
     */
    bool doStep(EvolvePhaseDesc *desc);

    /**
     * @brief      Performs one step of decimate algorithm phase
     *
     * @param      desc  Phase state descriptor
     *
     * @return     True if phase is still running, false otherwise
     */
    bool doStep(DecimatePhaseDesc *desc);

    enum SIM_PHASE 
    {
        STOP,
        CHOSE,
        EVOLVE,
        DECIMATE
        // ORDER .... 
    } phase;
};

} // end namespace vis
  
} // end namespace rd

#include "rd_simulation_vis.inl"

#endif  // CPU_SIMULATION_STEP_HPP