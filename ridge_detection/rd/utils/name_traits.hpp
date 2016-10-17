/**
 * @file name_traits.hpp
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
 */

#pragma once

#include "cub/block/block_load.cuh"
#include "cub/block/block_store.cuh"
#include "cub/thread/thread_load.cuh"
#include "cub/thread/thread_store.cuh"

#include "rd/utils/memory.h"

#if defined(BLOCK_TILE_LOAD_V1)
    #include "rd/gpu/block/block_tile_load_store.cuh"
// #elif defined(BLOCK_TILE_LOAD_V2)    
//     #include "rd/gpu/block/block_tile_load_store2.cuh"
// #elif defined(BLOCK_TILE_LOAD_V3)
//     #include "rd/gpu/block/block_tile_load_store3.cuh"
#elif defined(BLOCK_TILE_LOAD_V4)
    #include "rd/gpu/block/block_tile_load_store4.cuh"
#elif defined(BLOCK_TILE_LOAD_V5)
    #include "rd/gpu/block/block_tile_load_store5.cuh"
#endif

#if defined(BLOCK_TILE_LOAD_V1) || defined(BLOCK_TILE_LOAD_V4) || defined(BLOCK_TILE_LOAD_V5)
#define ENABLE_RD_IO_BACKEND_NAME_TRAITS
#endif

#include <string>
#include <stdexcept>

namespace rd
{

//------------------------------------------------------------
//  CUB LOAD CACHE MODIFIER
//------------------------------------------------------------
template <cub::CacheLoadModifier  LOAD_MODIFIER>
struct LoadModifierNameTraits
{
};
template <>
struct LoadModifierNameTraits<cub::LOAD_DEFAULT>
{
    static constexpr const char* name = "LOAD_DEFAULT";
};
template <>
struct LoadModifierNameTraits<cub::LOAD_CA>
{
    static constexpr const char* name = "LOAD_CA";
};
template <>
struct LoadModifierNameTraits<cub::LOAD_CG>
{
    static constexpr const char* name = "LOAD_CG";
};
template <>
struct LoadModifierNameTraits<cub::LOAD_CS>
{
    static constexpr const char* name = "LOAD_CS";
};
template <>
struct LoadModifierNameTraits<cub::LOAD_CV>
{
    static constexpr const char* name = "LOAD_CV";
};
template <>
struct LoadModifierNameTraits<cub::LOAD_LDG>
{
    static constexpr const char* name = "LOAD_LDG";
};
template <>
struct LoadModifierNameTraits<cub::LOAD_VOLATILE>
{
    static constexpr const char* name = "LOAD_VOLATILE";
};

std::string getLoadModifierName(cub::CacheLoadModifier  LOAD_MODIFIER)
{
    switch (LOAD_MODIFIER)
    {
        case cub::LOAD_DEFAULT:     return std::string("LOAD_DEFAULT");
        case cub::LOAD_CA:          return std::string("LOAD_CA");
        case cub::LOAD_CG:          return std::string("LOAD_CG");
        case cub::LOAD_CS:          return std::string("LOAD_CS");
        case cub::LOAD_CV:          return std::string("LOAD_CV");
        case cub::LOAD_LDG:         return std::string("LOAD_LDG");
        case cub::LOAD_VOLATILE:    return std::string("LOAD_VOLATILE");
        default: throw std::logic_error("unsupported load modifier!");
    }
}

//------------------------------------------------------------
//  CUB STORE CACHE MODIFIER
//------------------------------------------------------------
template <cub::CacheStoreModifier  STORE_MODIFIER>
struct StoreModifierNameTraits
{
};
template <>
struct StoreModifierNameTraits<cub::STORE_DEFAULT>
{
    static constexpr const char* name = "STORE_DEFAULT";
};
template <>
struct StoreModifierNameTraits<cub::STORE_WB>
{
    static constexpr const char* name = "STORE_WB";
};
template <>
struct StoreModifierNameTraits<cub::STORE_CG>
{
    static constexpr const char* name = "STORE_CG";
};
template <>
struct StoreModifierNameTraits<cub::STORE_CS>
{
    static constexpr const char* name = "STORE_CS";
};
template <>
struct StoreModifierNameTraits<cub::STORE_WT>
{
    static constexpr const char* name = "STORE_WT";
};
template <>
struct StoreModifierNameTraits<cub::STORE_VOLATILE>
{
    static constexpr const char* name = "STORE_VOLATILE";
};

std::string getStoreModifierName(cub::CacheStoreModifier  STORE_MODIFIER)
{
    switch (STORE_MODIFIER)
    {
        case cub::LOAD_DEFAULT:     return std::string("LOAD_DEFAULT");
        case cub::STORE_WB:         return std::string("STORE_WB");
        case cub::STORE_CG:         return std::string("STORE_CG");
        case cub::STORE_CS:         return std::string("STORE_CS");
        case cub::STORE_WT:         return std::string("STORE_WT");
        case cub::STORE_VOLATILE:   return std::string("STORE_VOLATILE");
        default: throw std::logic_error("unsupported store modifier!");
    }
}

//------------------------------------------------------------
//  CUB BLOCK STORE ALGORITHM
//------------------------------------------------------------
template <cub::BlockStoreAlgorithm STORE_ALGORITHM>
struct StoreAlgorithmNameTraits
{
};
template <>
struct StoreAlgorithmNameTraits<cub::BLOCK_STORE_DIRECT>
{ 
    static constexpr const char* name = "BLOCK_STORE_DIRECT";
};
template <>
struct StoreAlgorithmNameTraits<cub::BLOCK_STORE_VECTORIZE>
{
    static constexpr const char* name = "BLOCK_STORE_VECTORIZE";
};
template <>
struct StoreAlgorithmNameTraits<cub::BLOCK_STORE_TRANSPOSE>
{
    static constexpr const char* name = "BLOCK_STORE_TRANSPOSE";
};
template <>
struct StoreAlgorithmNameTraits<cub::BLOCK_STORE_WARP_TRANSPOSE>
{
    static constexpr const char* name = "BLOCK_STORE_WARP_TRANSPOSE";
};
template <>
struct StoreAlgorithmNameTraits<cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED>
{
    static constexpr const char* name = "BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED";
};


std::string getBlockStoreAlgName(cub::BlockStoreAlgorithm STORE_ALGORITHM)
{
    switch (STORE_ALGORITHM)
    {
        case cub::BLOCK_STORE_DIRECT:                       return std::string("BLOCK_STORE_DIRECT");
        case cub::BLOCK_STORE_VECTORIZE:                    return std::string("BLOCK_STORE_VECTORIZE");
        case cub::BLOCK_STORE_TRANSPOSE:                    return std::string("BLOCK_STORE_TRANSPOSE");
        case cub::BLOCK_STORE_WARP_TRANSPOSE:               return std::string("BLOCK_STORE_WARP_TRANSPOSE");
        case cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED:    return std::string("BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED");
        default: throw std::logic_error("unsupported block store algorithm!");
    }
}

//------------------------------------------------------------
//  CUB BLOCK LOAD ALGORITHM
//------------------------------------------------------------
template <cub::BlockLoadAlgorithm LOAD_ALGORITHM>
struct LoadAlgorithmNameTraits
{
};
template <>
struct LoadAlgorithmNameTraits<cub::BLOCK_LOAD_DIRECT>
{ 
    static constexpr const char* name = "BLOCK_LOAD_DIRECT";
};
template <>
struct LoadAlgorithmNameTraits<cub::BLOCK_LOAD_VECTORIZE>
{
    static constexpr const char* name = "BLOCK_LOAD_VECTORIZE";
};
template <>
struct LoadAlgorithmNameTraits<cub::BLOCK_LOAD_TRANSPOSE>
{
    static constexpr const char* name = "BLOCK_LOAD_TRANSPOSE";
};
template <>
struct LoadAlgorithmNameTraits<cub::BLOCK_LOAD_WARP_TRANSPOSE>
{
    static constexpr const char* name = "BLOCK_LOAD_WARP_TRANSPOSE";
};
template <>
struct LoadAlgorithmNameTraits<cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED>
{
    static constexpr const char* name = "BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED";
};


std::string getBlockLoadAlgName(cub::BlockLoadAlgorithm LOAD_ALGORITHM)
{
    switch (LOAD_ALGORITHM)
    {
        case cub::BLOCK_LOAD_DIRECT:                       return std::string("BLOCK_LOAD_DIRECT");
        case cub::BLOCK_LOAD_VECTORIZE:                    return std::string("BLOCK_LOAD_VECTORIZE");
        case cub::BLOCK_LOAD_TRANSPOSE:                    return std::string("BLOCK_LOAD_TRANSPOSE");
        case cub::BLOCK_LOAD_WARP_TRANSPOSE:               return std::string("BLOCK_LOAD_WARP_TRANSPOSE");
        case cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED:    return std::string("BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED");
        default: throw std::logic_error("unsupported block load algorithm!");
    }
}

//------------------------------------------------------------
//  RD DATA MEMORY LAYOUT
//------------------------------------------------------------
template <DataMemoryLayout    INPUT_MEM_LAYOUT>
struct DataMemoryLayoutNameTraits
{
};
template <>
struct DataMemoryLayoutNameTraits<ROW_MAJOR>
{
    static constexpr const char* name = "ROW_MAJOR";
    static constexpr const char* shortName = "R";
};
template <>
struct DataMemoryLayoutNameTraits<COL_MAJOR>
{
    static constexpr const char* name = "COL_MAJOR";
    static constexpr const char* shortName = "C";
};
template <>
struct DataMemoryLayoutNameTraits<SOA>
{
    static constexpr const char* name = "SoA";
    static constexpr const char* shortName = "SoA";
};


std::string getRDDataMemoryLayout(DataMemoryLayout    INPUT_MEM_LAYOUT)
{
    switch (INPUT_MEM_LAYOUT)
    {
        case ROW_MAJOR:     return std::string("ROW_MAJOR");
        case COL_MAJOR:     return std::string("COL_MAJOR");
        case SOA:           return std::string("SOA");
        default: throw std::logic_error("unsupported memory layout!");
    }
}

//------------------------------------------------------------
//  RD TILE IO BACKEND
//------------------------------------------------------------
#ifdef ENABLE_RD_IO_BACKEND_NAME_TRAITS

template <gpu::BlockTileIOBackend    IO_BACKEND>
struct BlockTileIONameTraits
{
};
template <>
struct BlockTileIONameTraits<gpu::IO_BACKEND_CUB>
{
    static constexpr const char* name = "IO_BACKEND_CUB";
};
template <>
struct BlockTileIONameTraits<gpu::IO_BACKEND_TROVE>
{
    static constexpr const char* name = "IO_BACKEND_TROVE";
};

std::string getRDTileIOBackend(gpu::BlockTileIOBackend    IO_BACKEND)
{
    switch (IO_BACKEND)
    {
        case gpu::IO_BACKEND_CUB:     return std::string("IO_BACKEND_CUB");
        case gpu::IO_BACKEND_TROVE:   return std::string("IO_BACKEND_TROVE");
        default: throw std::logic_error("unsupported io backend!");
    }
}

#endif

} // end namespace rd
  