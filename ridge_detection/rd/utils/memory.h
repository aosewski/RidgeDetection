/**
 * @file memory.h
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering, Faculty of Electronics and Information
 * Technology, Warsaw University of Technology 2016
 */

#ifndef __MEMORY_H__
#define __MEMORY_H__

namespace rd
{
    
/**
 * @brief      Determines possible data layout in memory.
 */
enum DataMemoryLayout
{
    ROW_MAJOR,
    COL_MAJOR,
    SOA
};

}

#endif
