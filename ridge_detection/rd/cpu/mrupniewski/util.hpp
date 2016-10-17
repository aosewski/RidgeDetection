/**
 * @file util.hpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering, Faculty of Electronics and Information
 * Technology, Warsaw University of Technology 2016
 * 
 * @note Slightly modified version of original Marek Rupniewski's
 * ridge detection algorithm. Changes are limited to making c++
 * function template version.
 * 
 */

#ifndef CPU_MRUP_UTIL_HPP
#define CPU_MRUP_UTIL_HPP

namespace rd
{
namespace cpu
{
namespace mrup
{

template <typename T>
inline T odl2(T *p1, T *p2, int dim) {
  T s=0, t;
  int d=dim;
  while (d-- > 0) {
    t = *(p1++) - *(p2++);
    s += t*t;
  }
  return s;
}

template <typename T>
inline int wiecejBliskich(T *punkty, int num, T *punkt, T r2, int prog, int dim) {
  // funkcja zwraca 1 (w p.p. 0) jeśli istnieje przynajmniej prog punktów (po
  // dim współrzędnych każdy) spod adresu punkty, których
  // kwadrat odległości od punkt jest mniejszy bądź równy r2
  while (num-- > 0) {
    if (odl2(punkt, punkty, dim) <= r2) 
      if (--prog <= 0) return 1;
    punkty += dim;
  }
  return 0;
}

} // end namespace mrup
} // end namespace cpu
} // end namespace rd

#endif //CPU_MRUP_UTIL_HPP