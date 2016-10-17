/**
 * @file simulation.cuh
 * @author Adam Rogowiec
 * 
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is conducted under the supervision of prof. dr hab. inż. Marek
 * Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */

#ifndef BRUTE_FORCE_GPU_SIMULATION_CUH_
#define BRUTE_FORCE_GPU_SIMULATION_CUH_

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/memory.h"

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <string>

namespace rd 
{
namespace gpu
{
namespace bruteForce
{


template <
    typename            T,
    int                 DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT>
class RidgeDetection
{

public:

    /// cardinality of samples set
    size_t np_;
    /// cardinality of initial choosen samples set
    size_t ns_;
    /// device pointer to 'ns_' variable
    int *dNs_;
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
    /**
     * @brief table containing samples in device memory
     */
    T *dP_;
    // @var pPtich_ In case of COL_MAJOR input memory layout, we allocate 2D memory region with 
    // aligned rows, otherwise is meaningless, equal zero.
    size_t pPitch_;
    /// table containing choosen samples
    T *dS_;
    // in case of COL_MAJOR output mem layout this is a dS_ mem pitch 
    size_t sPitch_;

    /**
     * Table containing temporal cordinate sums for calculating
     * mass center. Resides in device memory
     */
    T *dCordSums_;
    size_t csPitch_;
    /**
     * Table containing counters of points which fall into respective spheres.
     */
    int *dSpherePointCnt_;

    // table used in decimation phase to store computed distances
    T * dDistMtx_;
    size_t distMtxPitch_;
    // mask used in decimation phase
    char * dPtsMask_;

    /// flag verbose output
    bool verbose_;

    RidgeDetection(size_t np, T r1, T r2, bool verbose = false);
    virtual ~RidgeDetection();

    void ridgeDetection();
    void getChosenSamplesCount();

private:

    cudaError_t err_;

    void init();
    void doChoose();
    void doEvolve();
    void doDecimate();

    void freeTemporaries();
    void allocTemporaries();

};

} // end namespace bruteForce
} // end namespace gpu
} // end namespace rd

#include "simulation.inl"

#endif /* BRUTE_FORCE_GPU_SIMULATION_CUH_ */
