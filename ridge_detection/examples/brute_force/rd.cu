/**
 *  @file rd.cu
 *  @author Adam Rogowiec
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

#include "rd/utils/rd_params.hpp"
#include "cub/test_util.h"
#include "rd/gpu/brute_force/simulation3.cuh"
#include "rd/gpu/samples_generator.cuh"
#include "rd/utils/utilities.hpp"
#include "rd/utils/flags.h"

#include <helper_cuda.h>
#include <sstream>
#include <string>
#include <stdexcept>
#include <typeinfo>

template <
    typename    T,
    int         DIM>
void ridgeDetection(T const *h_pointCloud,
                    T **h_detectedRidge,
                    rd::RDParams<T> &rdp,
                    rd::RDSpiralParams<T> sp,
                    int devId)
{
    checkCudaErrors(deviceInit(devId));

    const int ITEMS_PER_THREAD = 2;

    rd::RDGPUBruteForce3<T, DIM, ITEMS_PER_THREAD> *rdGpu = 
      new rd::RDGPUBruteForce3<T, DIM, ITEMS_PER_THREAD>(rdp.np, rdp.r1,
         rdp.r2, rdp.verbose);

    if (!sp.loadFromFile)
    {
        switch (DIM)
        {
            case 3:
                rd::SamplesGenerator<T>::
                    template spiral3D<rd::COL_MAJOR>(
                        rdp.np, sp.a, sp.b, sp.sigma, rdGpu->dP_);
                break;
            case 2:
                rd::SamplesGenerator<T>::
                    template spiral2D<rd::COL_MAJOR>(
                        rdp.np, sp.a, sp.b, sp.sigma, rdGpu->dP_);
                break;
            default:
                throw std::logic_error("Unsupported samples dimension!");
        }

        checkCudaErrors(cudaDeviceSynchronize());
        rdGpu->copySamplesTo(rdGpu->hP_);
        checkCudaErrors(cudaDeviceSynchronize());
        rdGpu->samplesTransferredToHostMem_ = true;

    } 
    // in case of loading from file and pipeline mode 
    else
    {
        rdGpu->copySamplesFrom(h_pointCloud);
        copyTable(h_pointCloud, rdGpu->hP_, rdp.np * rdp.dim);
        rdGpu->samplesTransferredToHostMem_ = true;
    }

    if (rdp.verbose)
    {
        rd::GraphDrawer<T> gDrawer;
        std::ostringstream os;
        os << typeid(T).name() << "_" << DIM;
        os << "D_initial_samples_set_";
        gDrawer.showPoints(os.str(), rdGpu->hP_, rdp.np, DIM);
        os.clear();
        os.str(std::string());
    }

    rdGpu->ridgeDetection();

    *h_detectedRidge = new T[rdGpu->ns_ * DIM];
    rdp.ns = rdGpu->ns_;

    rdGpu->copyChosenSamplesTo(*h_detectedRidge);
    checkCudaErrors(cudaDeviceSynchronize());

    delete rdGpu;

    checkCudaErrors(deviceReset());
}

/**
 * @brief      Detects ridge of 2D function.
 *
 * @param      h_pointCloud     Pointer to input data.
 * @param      h_detectedRidge  Addres of pointer to which detected ridge is
 *                              saved.
 * @param[in]  rdp              Parameters of simulation.
 * @param[in]  sp               Parameters of generated (or not) spiral.
 *
 */
void ridgeDetection2D(float const *h_pointCloud,
                      float **h_detectedRidge,
                      rd::RDParams<float> &rdp,
                      rd::RDSpiralParams<float> sp,
                      int devId)
{
    ridgeDetection<float, 2>(h_pointCloud, h_detectedRidge, rdp, sp, devId);
}

/**
 * @brief      Detects ridge of 2D function.
 *
 * @param      h_pointCloud     Pointer to input data.
 * @param      h_detectedRidge  Addres of pointer to which detected ridge is
 *                              saved.
 * @param[in]  rdp              Parameters of simulation.
 * @param[in]  sp               Parameters of generated (or not) spiral.
 *
 */
void ridgeDetection2D(double const *h_pointCloud,
                      double **h_detectedRidge,
                      rd::RDParams<double> &rdp,
                      rd::RDSpiralParams<double> sp,
                      int devId)
{
    ridgeDetection<double, 2>(h_pointCloud, h_detectedRidge, rdp, sp, devId);
}

/**
 * @brief      Detects ridge of 3D function.
 *
 * @param      h_pointCloud     Pointer to input data.
 * @param      h_detectedRidge  Addres of pointer to which detected ridge is
 *                              saved.
 * @param[in]  rdp              Parameters of simulation.
 * @param[in]  sp               Parameters of generated (or not) spiral.
 *
 */
void ridgeDetection3D(float const *h_pointCloud,
                      float **h_detectedRidge,
                      rd::RDParams<float> &rdp,
                      rd::RDSpiralParams<float> sp,
                      int devId)
{
    ridgeDetection<float, 3>(h_pointCloud, h_detectedRidge, rdp, sp, devId);
}

/**
 * @brief      Detects ridge of 3D function.
 *
 * @param      h_pointCloud     Pointer to input data.
 * @param      h_detectedRidge  Addres of pointer to which detected ridge is
 *                              saved.
 * @param[in]  rdp              Parameters of simulation.
 * @param[in]  sp               Parameters of generated (or not) spiral.
 *
 */
void ridgeDetection3D(double const *h_pointCloud,
                      double **h_detectedRidge,
                      rd::RDParams<double> &rdp,
                      rd::RDSpiralParams<double> sp,
                      int devId)
{
    ridgeDetection<double, 3>(h_pointCloud, h_detectedRidge, rdp, sp, devId);
}
