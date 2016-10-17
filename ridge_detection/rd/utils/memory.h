/**
 * @file memory.h
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

#ifndef __MEMORY_H__
#define __MEMORY_H__

namespace rd
{
    
enum DataMemoryLayout
{
    ROW_MAJOR,
    COL_MAJOR,
    SOA
};

}

#endif