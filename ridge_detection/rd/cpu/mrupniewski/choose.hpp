/**
 * @file choose.hpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: "Design of the parallel version of
 * the ridge detection algorithm for the multidimensional random variable density function and its
 * implementation in CUDA technology", which is conducted under the supervision of prof. dr hab.
 * inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and Information
 * Technology Warsaw University of Technology 2016
 *
 * @note       Slightly modified version of original Marek Rupniewski's ridge detection algorithm.
 *             Changes are limited to making c++ function template version.
 */

#ifndef CPU_MRUP_CHOOSE_HPP
#define CPU_MRUP_CHOOSE_HPP

namespace rd
{
namespace cpu
{
namespace mrup
{

// Wybranie maksymalnego zbioru punktów, z których każde dwa są w odległości co najwyżej r.
// Punkty dobierane są w takiej kolejności, w jakiej występują w tablicy punkty.

#include "util.hpp"

#include <assert.h>
#include <string.h>

template <typename T>
int choose(
    T *punkty,//< początek tablicy ze współrzędnymi, każdy punkt reprezentowany
              //  jest przez dim współrzędnych
    int num,  //< liczba punktów (każdy po dim współrzędnych)
    T *wybrane,  //< początek tablicy na wybrańców
    T r, //< parametr funkcji choose (różne znaczenie w zależności od implementacji).
    int dim, //< wymiar danych
    void *pozostale   //< wskaźnik do struktury zawierającej pozostałe argumenty funkcji (w razie potrzeby)
)
{
  int liczba = 0;
  T r2 = r*r;
  T *punkt = punkty;
  assert(wybrane!=NULL);
  memcpy(wybrane, punkty, dim*sizeof(T));
  liczba++;
  punkt += dim;
  while (--num > 0) {
    if (!wiecejBliskich(wybrane, liczba, punkt, r2, 1, dim)) 
            memcpy(wybrane + dim * liczba++ , punkt, dim*sizeof(T));
    punkt += dim;
  }
  return liczba;
}

} // end namespace mrup
} // end namespace cpu
} // end namespace rd

#endif  // CPU_MRUP_CHOOSE_HPP

