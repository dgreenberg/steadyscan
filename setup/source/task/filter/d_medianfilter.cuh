#ifndef GPUMEDIANFILTER_CUH_
#define GPUMEDIANFILTER_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\cudaincludes.h"
#else
#include "../../cudaincludes.h"
#endif

template<typename T>
void hd_gpumedianfilter( T* values, int nValues );

#endif
