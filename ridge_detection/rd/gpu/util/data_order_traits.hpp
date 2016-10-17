/**
 * @file data_order_traits.hpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is conducted under the supervision of prof. dr hab. inż. Marka
 * Nałęcza.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */

#ifndef DATA_ORDER_TRAITS_HPP_
#define DATA_ORDER_TRAITS_HPP_

namespace rd 
{
namespace gpu
{

struct rowMajorOrderTag {};
struct colMajorOrderTag {};

} /* end namespace gpu */
} /* end namespace rd */

#endif /* DATA_ORDER_TRAITS_HPP_ */
