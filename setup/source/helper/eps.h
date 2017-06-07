#ifndef EPS_H_
#define EPS_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\includes.h"
#else
#include "../includes.h"
#endif

template<typename T>
__device__ __host__ inline T eps() {
	return -1;
}

template<>
__device__ __host__ inline float eps<float>() {
	return FLT_EPSILON;
}
template<>
__device__ __host__ inline double eps<double>() {
	return DBL_EPSILON;
}


#endif /* EPS_H_ */
