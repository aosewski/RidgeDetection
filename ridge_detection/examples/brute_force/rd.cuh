/**
 *  @file rd.cuh
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

#ifndef RD_HPP
#define RD_HPP 

#include "rd/utils/rd_params.hpp"

/**
 * @brief      Detects ridge of 2D function.
 *
 * @param      h_pointCloud     Pointer to input data.
 * @param      h_detectedRidge  Addres of pointer to which detected ridge is
 *                              saved.
 * @param[in]  rdp              Parameters of simulation.
 * @param[in]  sp               Parameters of generated (or not) spiral.
 * @param[in]  devId            Device id on which to run algorithm.
 */
void ridgeDetection2D(float const *h_pointCloud,
                      float **h_detectedRidge,
                      rd::RDParams<float> &rdp,
                      rd::RDSpiralParams<float> sp,
                      int devId);

/**
 * @brief      Detects ridge of 2D function.
 *
 * @param      h_pointCloud     Pointer to input data.
 * @param      h_detectedRidge  Addres of pointer to which detected ridge is
 *                              saved.
 * @param[in]  rdp              Parameters of simulation.
 * @param[in]  sp               Parameters of generated (or not) spiral.
 * @param[in]  devId            Device id on which to run algorithm.
 */
void ridgeDetection2D(double const *h_pointCloud,
                      double **h_detectedRidge,
                      rd::RDParams<double> &rdp,
                      rd::RDSpiralParams<double> sp,
                      int devId);
/**
 * @brief      Detects ridge of 3D function.
 *
 * @param      h_pointCloud     Pointer to input data.
 * @param      h_detectedRidge  Addres of pointer to which detected ridge is
 *                              saved.
 * @param[in]  rdp              Parameters of simulation.
 * @param[in]  sp               Parameters of generated (or not) spiral.
 * @param[in]  devId            Device id on which to run algorithm.
 */
void ridgeDetection3D(float const *h_pointCloud,
                      float **h_detectedRidge,
                      rd::RDParams<float> &rdp,
                      rd::RDSpiralParams<float> sp,
                      int devId);

/**
 * @brief      Detects ridge of 3D function.
 *
 * @param      h_pointCloud     Pointer to input data.
 * @param      h_detectedRidge  Addres of pointer to which detected ridge is
 *                              saved.
 * @param[in]  rdp              Parameters of simulation.
 * @param[in]  sp               Parameters of generated (or not) spiral.
 * @param[in]  devId            Device id on which to run algorithm.
 */
void ridgeDetection3D(double const *h_pointCloud,
                      double **h_detectedRidge,
                      rd::RDParams<double> &rdp,
                      rd::RDSpiralParams<double> sp,
                      int devId);

#endif  // RD_HPP