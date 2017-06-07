#ifndef TASK_MASKINACTIVE_CUH_
#define TASK_MASKINACTIVE_CUH_

#if defined( _WIN32 ) || defined( _WIn64 )
#include "..\\includes.h"
#include "..\\interface\\memory.h"
#include "..\\helper\\cudahelper\\cudahelper.cuh"
#else
#include "../includes.h"
#include "../interface/memory.h"
#include "../helper/cudahelper/cudahelper.cuh"
#endif

template<typename T>
void hd_maskInactiveGroups( DeviceMemory<T>& mem );

#endif
