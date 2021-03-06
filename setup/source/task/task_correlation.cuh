#ifndef TASK_CORRELATION_CUH_
#define TASK_CORRELATION_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\includes.h"
#include "..\\cudaincludes.h"
#include "..\\interface\\memory.h"
#include "..\\helper\\cudahelper\\cudahelper.cuh"
#else
#include "../includes.h"
#include "../cudaincludes.h"
#include "../interface/memory.h"
#include "../helper/cudahelper/cudahelper.cuh"
#endif

template<typename T>
void hd_globalCorrelation( DeviceMemory<T>& mem, T* corr_out );

template<typename T>
void hd_frameCorrelation( DeviceMemory<T>& mem );

#endif
