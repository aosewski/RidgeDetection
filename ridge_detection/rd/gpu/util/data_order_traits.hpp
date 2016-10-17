/**
 * @file data_order_traits.hpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and Information
 * Technology Warsaw University of Technology 2016
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
