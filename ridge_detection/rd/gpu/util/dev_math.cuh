/**
 * @file dev_math.cuh
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


#ifndef DEV_MATH_CUH_
#define DEV_MATH_CUH_

#include <vector_types.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <math_functions.hpp>
#include <device_functions.hpp>
#include <device_double_functions.hpp>
#include <device_atomic_functions.h>

#include <type_traits>
#include "cub/util_type.cuh"

namespace rd 
{
namespace gpu
{

/*****************************************************************************************
 *		 RNG FUNCTIONS
 *****************************************************************************************/

template <typename T>
__device__ __forceinline__ T getUniformDist(curandState *s) 
{
	return (T)curand_uniform(s);
}

template <>
__device__ __forceinline__ float getUniformDist(curandState *s) 
{
	return curand_uniform(s);
}

template <>
__device__ __forceinline__ double getUniformDist(curandState *s) 
{
	return curand_uniform_double(s);
}

/*****************************************************************************************/

template <typename T>
__device__ __forceinline__ T getNormalDist(curandState *s) 
{
	return (T)curand_normal(s);
}

template <>
__device__ __forceinline__ float getNormalDist(curandState *s) 
{
	return curand_normal(s);
}

template <>
__device__ __forceinline__ double getNormalDist(curandState *s) 
{
	return curand_normal_double(s);
}

template <>
__device__ __forceinline__ float2 getNormalDist(curandState *s) 
{
	return curand_normal2(s);
}

template <>
__device__ __forceinline__ double2 getNormalDist(curandState *s) 
{
	return curand_normal2_double(s);
}

/*****************************************************************************************
 *		 ATOMIC FUNCTIONS
 *****************************************************************************************/

template <typename T>
__device__ __forceinline__ T rdAtomicAdd(T *address, T val)
{
	return atomicAdd(address, val);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 120
template <>
__device__ __forceinline__ double rdAtomicAdd<double>(double *address, double val) 
{

	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do 
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
		__double_as_longlong(val + __longlong_as_double(assumed)));
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 120

/*****************************************************************************************
 *		 OTHER FUNCTIONS
 *****************************************************************************************/


template <typename T>
__device__ __forceinline__ T getMaxValue() 
{
	return 0;
}

template <>
__device__  __forceinline__ float getMaxValue<float>() 
{
	return 1.70141e+38f;
}

template <>
__device__  __forceinline__ double getMaxValue<double>() 
{
	return 8.98847e+307;
}

/*****************************************************************************************/

template <typename T>
__host__  __device__ __forceinline__ T getEpsilon() 
{
	return 0;
}

template <>
__host__  __device__ __forceinline__ float getEpsilon() 
{
	return 1.19209e-07f;
}

template <>
__host__  __device__ __forceinline__ double getEpsilon() 
{
	return 2.22045e-16;
}

/*****************************************************************************************/

template <typename T>
__device__ __forceinline__ T getPi() 
{
	return 3.14;
}

template <>
__device__ __forceinline__ float getPi() 
{
	return CUDART_PI_F;
}

template <>
__device__ __forceinline__ double getPi() 
{
	return CUDART_PI;
}

/*****************************************************************************************/

template <typename T>
__device__ __forceinline__ T getCos(T x) 
{
	return (T)cosf((float)x);
}

template <>
__device__ __forceinline__ float getCos(float x) 
{
	return cosf(x);
}

template <>
__device__ __forceinline__ double getCos(double x) 
{
	return cos(x);
}
/*****************************************************************************************/

template <typename T>
__device__ __forceinline__ T getSin(T x) 
{
	return (T)sinf((float)x);
}

template <>
__device__ __forceinline__ float getSin(float x) 
{
	return sinf(x);
}

template <>
__device__ __forceinline__ double getSin(double x) 
{
	return sin(x);
}

/*****************************************************************************************/

template <typename T>
__device__ __forceinline__ T getExp(T x) 
{
	return (T)expf((float)x);
}

template <>
__device__ __forceinline__ float getExp(float x) 
{
	return expf(x);
}

template <>
__device__ __forceinline__ double getExp(double x) 
{
	return exp(x);
}

/*****************************************************************************************/

template <typename T>
struct GetNaN
{
};

template <>
struct GetNaN<float>
{
	static __device__ __forceinline__ float value()
	{
		return nanf("");
	}
};

template <>
struct GetNaN<double>
{
	static __device__ __forceinline__ double value()
	{
		return nan("");
	}
};


/*****************************************************************************************/

template <typename T>
struct vector2 {
};

template <>
struct vector2<int> {
	typedef int2 type;
};

template <>
struct vector2<float> {
	typedef float2 type;
};

template <>
struct vector2<double> {
	typedef double2 type;
};

/*****************************************************************************************/

template <typename T>
struct vector3 {
};

template <>
struct vector3<int> {
	typedef int3 type;
};

template <>
struct vector3<float> {
	typedef float3 type;
};

template <>
struct vector3<double> {
	typedef double3 type;
};

/*****************************************************************************************/

__device__ __forceinline__ unsigned int toUint(float val, enum cudaRoundMode mode = cudaRoundNearest)
{
	return float2uint(val, mode);
}

__device__ __forceinline__ unsigned int toUint(double val, enum cudaRoundMode mode = cudaRoundNearest)
{
	return double2uint(val, mode);
}

/*****************************************************************************************/

// (1-t)*v0 + t*v1
template <typename T>
__device__ __forceinline__ T lerp(T v0, T v1, T t) {
    return fma(t, v1, fma(-t, v0, v0));
}

/*****************************************************************************************/

template <typename T>
__device__ __forceinline__ 
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
    almostEqual(T x, T y, int ulp = 2)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return abs(x-y) < cub::FpLimits<T>::Epsilon() * abs(x+y) * ulp
    // unless the result is subnormal
           || abs(x-y) < cub::FpLimits<T>::Min();

    /*
     * When there will be no problems with --expt-relaxed-constexpr flag we may use:
     * In CUDA 7.5 when compiling with debug flags there are some errors in <cmath>
     */
    // return abs(x-y) < std::numeric_limits<T>::epsilon() * abs(x+y) * ulp
    // || abs(x-y) < std::numeric_limits<T>::min();
}

template <typename T>
__device__ __forceinline__ 
typename std::enable_if<std::is_integral<T>::value, bool>::type
    almostEqual(T x, T y)
{
    return x == y;
}

} // end namespace gpu
} // end namespace rd
#endif /* DEV_MATH_CUH_ */
