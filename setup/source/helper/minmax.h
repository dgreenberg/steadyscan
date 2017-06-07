#ifndef MINMAX_H
#define MINMAX_H

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\includes.h"
#else
#include "../includes.h"
#endif


template<typename T>
__host__ __device__ inline T minValue() {
	return 0;
}
template<typename T>
__host__ __device__ inline T maxValue() {
	return 0;
}

template<>
__host__ __device__ inline double minValue<double>() { return DBL_MIN; }

template<>
__host__ __device__ inline float minValue<float>() { return FLT_MIN; }


template<>
__host__ __device__ inline double maxValue<double>() { return DBL_MAX; }

template<>
__host__ __device__ inline float maxValue<float>() { return FLT_MAX; }


#ifndef min
template<typename Ta, typename Tb>
__host__ __device__ inline Ta min(Ta a, Tb b) {
	return a<b?a:b;
}
#endif
#ifndef max
template<typename Ta, typename Tb>
__host__ __device__ inline Ta max(Ta a, Tb b) {
	return a>b?a:b;
}
#endif

#endif
