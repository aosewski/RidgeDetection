/**
 * @file decimate.hpp
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
 * 
 * @note Slightly modified version of original Marek Rupniewski's
 * ridge detection algorithm. Changes are limited to making c++
 * function template version.
 * 
 */

#ifndef CPU_MRUP_DECIMATE_HPP
#define CPU_MRUP_DECIMATE_HPP

namespace rd
{
namespace cpu
{
namespace mrup
{

#include "util.hpp"

#include <string.h>
/* Usuwane są punkty mające 3 lub więcej sąsiadów w r otoczeniu lub mniej niż
 * 2 sąsiadów w 2*r otoczeniu. Zaprzestanie usuwania jeśli zostało mniej niż 3
 * punkty, lub nie ma punktów spełniających powyższe kryteria.
 * Funkcja zwraca ten sam adres tablicy, który dostała.
 */

template <typename T>
T *decimate(
    T *srodki,//< początek tablicy ze współrzędnymi
              //  każdy punkt reprezentowany jest przez dim współrzędnych,
    int *liczba, //< liczba punktów przed redukcją (wejście) oraz po redukji (wyjście)
    T r, //< parametr funkcji (różne znaczenie w zależności od implementacji).
    int dim, //< wymiar danych
    void *pozostale   //< wskaźnik do struktury zawierającej pozostałe argumenty funkcji (w razie potrzeby)
)
{
  T r2 = r*r;
  int l = 0;
  while (l != *liczba && *liczba > 2) {
    l = *liczba;
    for(int i = 0; i < *liczba;) {
      if (wiecejBliskich(srodki, *liczba, srodki + i*dim, r2, 4, dim) ||
         !wiecejBliskich(srodki, *liczba, srodki + i*dim, 4*r2, 3, dim) ) {
        (*liczba)--;
        memmove(srodki + i*dim, srodki + (i+1)*dim, (*liczba-i)*dim*sizeof(T));
        if (*liczba < 3) break;
      } else i++;
    }
  }
  return srodki;
}

} // end namespace mrup
} // end namespace cpu
} // end namespace rd

#endif // CPU_MRUP_DECIMATE_HPP
