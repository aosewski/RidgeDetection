/*-------------------------------------------------------------------------*//**
 * @file block_dynamic_vector.cuh
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

#include "rd/utils/memory.h"

#include "rd/gpu/warp/warp_functions.cuh"
#include "rd/gpu/block/block_tile_load_store4.cuh"
#include "rd/gpu/agent/agent_memcpy.cuh"
#include "rd/gpu/util/dev_math.cuh"

#include "cub/util_type.cuh"
#include "cub/util_ptx.cuh"

#include <initializer_list>
#include <type_traits>


// #ifndef RD_DEBUG
// #define NDEBUG      // for disabling assert macro
// #endif 
#include <assert.h>

namespace rd 
{
namespace gpu
{
namespace detail
{

/**
 * @class      BlockDynamicVectorBase
 *
 * @brief      Vector with dynamic storage allocation and continuous memory.
 *
 * @param      When  resizing vector capacity all data are copied to new larger, continuous memory region. It's a user
 *                   responsibility to detect when capacity is reached and call vector resize function.
 *
 * @tparam     BLOCK_THREADS     Number of threads within block. Used as copy engine's template parameter for moving
 *                               data when resizing.
 * @tparam     ITEMS_PER_THREAD  Number of items copied by one thread. It's a copy engine's template parameter used for
 *                               moving data when resizing.
 * @tparam     T                 Stored data type.
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    typename    T>
class BlockDynamicVectorBase
{

public:
    
    __device__ __forceinline__ BlockDynamicVectorBase()
    :
        m_sharedStorage(privateSharedStorage()),
        m_storage_(nullptr),
        // m_itemsCnt_(0),
        m_capacity_(0),
        m_enlargeFactor_(DEFAULT_ENLARGE_FACTOR)
    {
        if (threadIdx.x == 0)
        {
            m_sharedStorage.itemsCnt_ = 0;
        }
        __syncthreads();
    }

    __device__ __forceinline__ BlockDynamicVectorBase(
        unsigned int enlargeFactor)
    :
        m_sharedStorage(privateSharedStorage()),
        m_storage_(nullptr),
        // m_itemsCnt_(0),
        m_capacity_(0),
        m_enlargeFactor_(enlargeFactor)
    {
        if (threadIdx.x == 0)
        {
            m_sharedStorage.itemsCnt_ = 0;
        }        
        __syncthreads();
    }

    __device__ __forceinline__ BlockDynamicVectorBase(
        unsigned int size,
        unsigned int enlargeFactor)
    :
        m_sharedStorage(privateSharedStorage()),
        m_storage_(nullptr),
        // m_itemsCnt_(0),
        m_capacity_(size),
        m_enlargeFactor_(enlargeFactor)
    {
        if (threadIdx.x == 0)
        {
            m_sharedStorage.itemsCnt_ = 0;
        }        
        __syncthreads();
        resize(m_capacity_);
    }

    __device__ __forceinline__ BlockDynamicVectorBase(
        T *          d_storage,
        unsigned int size,
        unsigned int capacity,
        unsigned int enlargeFactor)
    :
        m_sharedStorage(privateSharedStorage()),
        m_storage_(d_storage),
        // m_itemsCnt_(size),
        m_capacity_(capacity),
        m_enlargeFactor_(enlargeFactor)
    {
        if (threadIdx.x == 0)
        {
            m_sharedStorage.itemsCnt_ = size;
        }        
        __syncthreads();
    }

    virtual __device__ __forceinline__ ~BlockDynamicVectorBase()
    {
    }

    __device__ __forceinline__ void clear()
    {
        if (threadIdx.x == 0 && m_storage_ != nullptr)
        {
            delete[] m_storage_;
        }
    }

    //------------------------------------------------------------------------------
    //      Accessors
    //------------------------------------------------------------------------------

    __device__ __forceinline__ T const * begin() const
    {
        return m_storage_;
    }

    __device__ __forceinline__ T * begin()
    {
        return m_storage_;
    }

    __device__ __forceinline__ T * end() const
    {
        return m_storage_ + size();
    }


    __device__ __forceinline__ unsigned int capacity() const 
    {
        return m_capacity_;
    }

    __device__ __forceinline__ unsigned int size() const
    {
        return m_sharedStorage.itemsCnt_;
    }


    //------------------------------------------------------------------------------
    //     Modifiers
    //------------------------------------------------------------------------------

    __device__ __forceinline__ void blockIncrementItemsCnt(int items)
    {
        if (threadIdx.x == 0)
        {
            m_sharedStorage.itemsCnt_ += items;
        }
        __threadfence_block();
    }

    __device__ __forceinline__ void warpIncrementItemsCnt(int items)
    {
        if (cub::LaneId() == 0)
        {
            atomicAdd(&m_sharedStorage.itemsCnt_, (unsigned int)items);
        }
        __threadfence_block();
    }

    __device__ __forceinline__ void threadIncrementItemsCnt(int items)
    {
        warpReduce(items);
        warpIncrementItemsCnt(items);
    }

    __device__ __forceinline__ void resize(unsigned int newCapacity)
    {
        // currently do nothing if requested newCapacity is lower than current
        if (newCapacity > m_capacity_)
        {
            // assert(isWarpConverged());
        #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                 _CubLog("<<<__resize__>>>> old_capacity_: %u new_capacity: %u\n", m_capacity_, newCapacity);
            }
        #endif
            expandStorage(newCapacity);
            // make sure new storage pointer value is visible to all threads in a block
            __syncthreads();
            T * newStorage = m_sharedStorage.newStoragePtr_;
            copyData(newStorage);
            // wait for copy to end to safely release old storage
            __syncthreads();
            clear();
            m_storage_ = newStorage;
            m_capacity_ = newCapacity;
        }
    }

    //------------------------------------------------------------------------------
    //      Debug facilities
    //------------------------------------------------------------------------------

    #ifdef RD_DEBUG
    __device__ __forceinline__ void print(char const * msg, int tid = -1)
    {
        size();
        if (tid >= 0)
        {
            if (threadIdx.x == tid)
            {
                _CubLog("[%s] size: %u, capacity: %u, step: %u, data: %p\n",
                 msg,  m_sharedStorage.itemsCnt_, m_capacity_, m_enlargeFactor_, m_storage_);
            }
        }
        else
        {
            _CubLog("[%s] size: %u, capacity: %u, step: %u, data: %p\n",
             msg,  m_sharedStorage.itemsCnt_, m_capacity_, m_enlargeFactor_, m_storage_);
        }
    }

    __device__ __forceinline__ void print() const 
    {
        unsigned int m = (m_capacity_ + blockDim.x - 1) / blockDim.x * blockDim.x;
        for (int i = threadIdx.x; i < m; i += blockDim.x)
        {
            if (i < m_sharedStorage.itemsCnt_)
            {
                _CubLog("v[%6d]: %10.7f\n", i, m_storage_[i]);
            }
            __syncthreads();
        }
    }
    #endif

private:

    //---------------------------------------------------------------------
    // types
    //---------------------------------------------------------------------

    typedef BlockTileLoadPolicy<
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        // cub::LOAD_LDG> BlockTileLoadPolicyT;
        cub::LOAD_CS> BlockTileLoadPolicyT;

    typedef BlockTileStorePolicy<
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        cub::STORE_DEFAULT> BlockTileStorePolicyT;

    typedef AgentMemcpy<
        BlockTileLoadPolicyT,
        BlockTileStorePolicyT,
        1,
        ROW_MAJOR,          // in/out mem layout
        ROW_MAJOR,          // private mem layout
        IO_BACKEND_TROVE,
        unsigned int,
        T> AgentMemcpyT;

    // shared memory type for this threadblock
    struct _SharedStorage
    {
        // used for broadcasting to all threads new storage pointer
        T * volatile newStoragePtr_;
        // used to determine index where this thread can store data
        volatile unsigned int itemsCnt_;
    };

    // Alias wrapper allowing storage to be unioned
    struct SharedStorage : cub::Uninitialized<_SharedStorage> {};

    //---------------------------------------------------------------------
    // per-thread fields
    //---------------------------------------------------------------------
protected:
    static constexpr unsigned int DEFAULT_ENLARGE_FACTOR = 2;
    unsigned int              m_enlargeFactor_;       /// Factor by which we expand vector's capacity.
private:
    _SharedStorage &    m_sharedStorage;              /// reference to shared storage
    T *                 m_storage_;                   /// Pointer to continuous memory storage. 
    unsigned int              m_capacity_;            /// Current vector capacity.
    
    //---------------------------------------------------------------------
    // inner implementation routines
    //---------------------------------------------------------------------

    /// Internal storage allocator
    __device__ __forceinline__ _SharedStorage& privateSharedStorage()
    {
        __shared__ _SharedStorage storage;
        return storage;
    }

    __device__ __forceinline__ void expandStorage(unsigned int newCapacity)
    {
        if (threadIdx.x == 0)
        {
            T * ptr = new T[newCapacity];
            assert(ptr != nullptr);
            m_sharedStorage.newStoragePtr_ = ptr;
        }
    }

    __device__ __forceinline__ void copyData(T * dst)
    {
        int itemsCnt = m_sharedStorage.itemsCnt_;
        // __threadfence_block();
        if (itemsCnt > 0)
        {
            // #ifdef RD_DEBUG
            // if (threadIdx.x == 0)
            // {
            //      _CubLog(">>>> DynamicVector::copyData(): dst: %p,\titemsCnt:\t%d\n", dst, itemsCnt);
            // }
            // #endif
            // assert(itemsCnt <= m_capacity_);

            // dynamic vector always copy data in single block version! (true = single block)
            AgentMemcpyT(m_storage_, dst).copyRange(0, itemsCnt, 1, true);
        }
    }

};

} // end namespace detail
  

/*-----------------------------------------------------------------------------------------------------------------*//**
 * @brief      Vector with dynamic storage allocation and continuous memory.
 *
 * @paragraph When resizing vector capacity all data are copied to new larger, continuous memory region. It's a user
 * responsibility to detect when capacity is reached and call vector resize function.
 *
 * @tparam     BLOCK_THREADS         Number of threads within block. Used as copy engine's template parameter for moving
 *                                   data when resizing.
 * @tparam     ITEMS_PER_THREAD      Number of items copied by one thread. It's a copy engine's template parameter used
 *                                   for moving data when resizing.
 * @tparam     RESIZE_FACTORS_COUNT  Number of resize factors. Vector may be resized using few defined capacity values
 *                                   or with few defined factors of a given max value.
 * @tparam     ResizeFactorT         Data type of resize factors.
 * @tparam     T                     Stored data type.
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    class       T,
    int         RESIZE_FACTORS_COUNT = 0,
    class       ResizeFactorT = cub::NullType,
    class       Enable = void>
class BlockDynamicVector
    : public detail::BlockDynamicVectorBase<BLOCK_THREADS, ITEMS_PER_THREAD, T>
{
private:
    typedef detail::BlockDynamicVectorBase<BLOCK_THREADS, ITEMS_PER_THREAD, T>  BaseT;

public:
    __device__ __forceinline__ BlockDynamicVector()
    :
        BaseT()
    {
    }

    __device__ __forceinline__ BlockDynamicVector(
        unsigned int enlargeFactor)
    :   
        BaseT(enlargeFactor)
    {
    }

    __device__ __forceinline__ BlockDynamicVector(
        unsigned int size,
        unsigned int enlargeFactor)
    :
        BaseT(size, enlargeFactor)
    {
    }

    __device__ __forceinline__ BlockDynamicVector(
        T *          d_storage,
        unsigned int size,
        unsigned int capacity,
        unsigned int enlargeFactor)
    :
        BaseT(d_storage, size, capacity, enlargeFactor)
    {
    }

    __device__ __forceinline__  ~BlockDynamicVector()
    {
    }

    __device__ __forceinline__ void resize(unsigned int requestedCapacity)
    {
        unsigned int newCapacity = (BaseT::capacity() > 0) ? BaseT::capacity() : requestedCapacity;

        #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                _CubLog("---resize()--- requestedCapacity %d\n", requestedCapacity);
            }
        #endif

        while (newCapacity < requestedCapacity)
        {
            newCapacity *= BaseT::m_enlargeFactor_;
        #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                _CubLog("---resize()--- newCapacity: %d\n", newCapacity);
            }
        #endif

        };

		BaseT::resize(newCapacity);
    }

};


/**
 * @brief      Class specialization with integral resize factors (capacity thresholds).
 *
 * @par Initial vector capacity is the value of the first resize factor. Subsequent resize factors are used as a
 *     subsequent capacity values. When the last resize value is reached the vector switches to default base class
 *     resize mechanizm. That is resize enlarge factor.
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    class       T,
    int         RESIZE_FACTORS_COUNT,
    class       ResizeFactorT>
class BlockDynamicVector<
    BLOCK_THREADS,
    ITEMS_PER_THREAD,
    T,
    RESIZE_FACTORS_COUNT,
    ResizeFactorT,
    typename std::enable_if<
                std::is_integral<ResizeFactorT>::value &&
                std::is_unsigned<ResizeFactorT>::value>::type>
    : public detail::BlockDynamicVectorBase<BLOCK_THREADS, ITEMS_PER_THREAD, T>
{

private: 
    typedef detail::BlockDynamicVectorBase<BLOCK_THREADS, ITEMS_PER_THREAD, T>  BaseT;

public:

    __device__ __forceinline__ BlockDynamicVector(
        std::initializer_list<ResizeFactorT> resizeFactors)
    :   
        BaseT(BaseT::DEFAULT_ENLARGE_FACTOR),
        m_sharedStorage(privateSharedStorage())
    {
        // need c++14 for size() to be a constexpr function
        // static_assert(resizeFactors.size() == RESIZE_FACTORS_COUNT,
        //      "Initializer list size must have equal value as class template parameter RESIZE_FACTORS_COUNT!");

        if (threadIdx.x == 0)
        {
            ResizeFactorT const *ptr = resizeFactors.begin();
            #pragma unroll
            for (int k = 0; k < RESIZE_FACTORS_COUNT; ++k)
            {
                m_sharedStorage.resizeFactors[k] = ptr[k];
            }
            m_sharedStorage.nextFactorIdx = 1;
        }
        __syncthreads();

        resize(*resizeFactors.begin());
    }

    virtual __device__ __forceinline__  ~BlockDynamicVector()
    {
    }


    //------------------------------------------------------------------------------
    //     Modifiers
    //------------------------------------------------------------------------------
    __device__ __forceinline__ void resize(unsigned int requestedCapacity)
    {
        unsigned int newCapacity = (BaseT::capacity() > 0) ? BaseT::capacity() : requestedCapacity;

        #ifdef RD_DEBUG
            if (threadIdx.x == 0)
            {
                _CubLog("---resize()--- requestedCapacity %d\n", requestedCapacity);
            }
        #endif

        for (int i = 0; i < RESIZE_FACTORS_COUNT; ++i)
        {
            if (newCapacity >= requestedCapacity)
            {
                break;
            }
            else
            {
                newCapacity = m_sharedStorage.resizeFactors[i];
            }
        }

		while (newCapacity < requestedCapacity)
		{
			newCapacity *= BaseT::m_enlargeFactor_;
		#ifdef RD_DEBUG
			if (threadIdx.x == 0)
			{
				_CubLog("---resize()--- newCapacity: %d\n", newCapacity);
			}
		#endif
		}

		BaseT::resize(newCapacity);
    }

private:

    struct _SharedResizeFactors
    {
        ResizeFactorT   resizeFactors[RESIZE_FACTORS_COUNT];
    };

    // Alias wrapper allowing storage to be unioned
    struct SharedResizeFactors : cub::Uninitialized<_SharedResizeFactors> {};

    _SharedResizeFactors &  m_sharedStorage;    /// shared storage for resize factors.


    /// Internal storage allocator
    __device__ __forceinline__ _SharedResizeFactors& privateSharedStorage()
    {
        __shared__ _SharedResizeFactors storage;
        return storage;
    }

};

/**
 * @brief      Class specialization for floating point max size enlarge factors.
 *
 * @par Initial vector capacity is the product of the first resize factor and maxCapacity. Subsequent resize factors are used to scale
 *     maxCapacity and results are used as capacity values. When the last resize value is reached the vector switches to
 *     default base class resize mechanizm. That is resize enlarge factor.
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    class       T,
    int         RESIZE_FACTORS_COUNT,
    class       ResizeFactorT>
class BlockDynamicVector<
    BLOCK_THREADS, 
    ITEMS_PER_THREAD,
    T,
    RESIZE_FACTORS_COUNT,
    ResizeFactorT,
    typename std::enable_if<std::is_floating_point<ResizeFactorT>::value>::type>
    : public detail::BlockDynamicVectorBase<BLOCK_THREADS, ITEMS_PER_THREAD, T>
{
private: 
    typedef detail::BlockDynamicVectorBase<BLOCK_THREADS, ITEMS_PER_THREAD, T>  BaseT;

public:
    __device__ __forceinline__ BlockDynamicVector(
        std::initializer_list<ResizeFactorT>    resizeFactors,
        unsigned int                            maxCapacity)
    :   
        BaseT(BaseT::DEFAULT_ENLARGE_FACTOR),
        m_sharedStorage(privateSharedStorage())
    {
        // need c++14 for size() to be a constexpr function
        // static_assert(resizeFactors.size() == RESIZE_FACTORS_COUNT,
        //      "Initializer list size must have equal value as class template parameter RESIZE_FACTORS_COUNT!");

        if (threadIdx.x == 0)
        {
            ResizeFactorT const *ptr = resizeFactors.begin();
            #pragma unroll
            for (int k = 0; k < RESIZE_FACTORS_COUNT; ++k)
            {
                // #ifdef RD_DEBUG
                //     _CubLog("---constructor({...}, uint)--- factor[%d]: %f\n", k, ptr[k]);
                // #endif
                m_sharedStorage.resizeFactors[k] = ptr[k];
            }
            m_sharedStorage.maxCapacity = maxCapacity;
        }
        __syncthreads();

        resize(*resizeFactors.begin() * maxCapacity);
    }

    virtual __device__ __forceinline__  ~BlockDynamicVector()
    {
    }


    //------------------------------------------------------------------------------
    //     Modifiers
    //------------------------------------------------------------------------------
    __device__ __forceinline__ void resize(unsigned int requestedCapacity)
    {
        unsigned int newCapacity = (BaseT::capacity() > 0) ? BaseT::capacity() : requestedCapacity;

        // #ifdef RD_DEBUG
        //     if (threadIdx.x == 0)
        //     {
        //         _CubLog("---resize()--- requestedCapacity %d\n", requestedCapacity);
        //     }
        // #endif

        unsigned int maxCapacity = m_sharedStorage.maxCapacity;

        for (int i = 0; i < RESIZE_FACTORS_COUNT; ++i)
        {
            if (newCapacity >= requestedCapacity)
            {
                break;
            }
            else
            {
                newCapacity = toUint(m_sharedStorage.resizeFactors[i] * maxCapacity);
            }
        }

        // in case we need more than maxCapacity
		while (newCapacity < requestedCapacity)
		{
			newCapacity *= BaseT::m_enlargeFactor_;
		// #ifdef RD_DEBUG
		// 	if (threadIdx.x == 0)
		// 	{
		// 		_CubLog("---resize()--- newCapacity: %d\n", newCapacity);
		// 	}
		// #endif
		}

		BaseT::resize(newCapacity);
    }

private:

    struct _SharedResizeFactors
    {
        ResizeFactorT   resizeFactors[RESIZE_FACTORS_COUNT];
        unsigned int    maxCapacity;
    };

    // Alias wrapper allowing storage to be unioned
    struct SharedResizeFactors : cub::Uninitialized<_SharedResizeFactors> {};

    _SharedResizeFactors &  m_sharedStorage;    /// shared storage for resize factors.


    /// Internal storage allocator
    __device__ __forceinline__ _SharedResizeFactors& privateSharedStorage()
    {
        __shared__ _SharedResizeFactors storage;
        return storage;
    }
};

} // end namespace gpu
} // end namespace rd

