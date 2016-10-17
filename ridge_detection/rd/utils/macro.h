/**
 * @file macro.h
 * @author     Adam Rogowiec
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
 * 
 */

#pragma once 

namespace rd
{

#if defined(__CUDACC__) // NVCC
   #define RD_MEM_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define RD_MEM_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define RD_MEM_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for RD_MEM_ALIGN macro for your host compiler!"
#endif

#define DO_PRAGMA(x) _Pragma (#x)
/* 
 * "#pragma message" TODO wrapper
 * 
 * prints message on compilation
 * 
 * example usage:         
 * TODO(Remember to fix this)
 */
#define TODO(x) DO_PRAGMA(message ("TODO - " #x))

} // end namespace rd

