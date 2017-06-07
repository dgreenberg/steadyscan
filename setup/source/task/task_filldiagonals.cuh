#ifndef TASK_FILLDIAGONALS_CUH_
#define TASK_FILLDIAGONALS_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\includes.h"
#include "..\\cudaincludes.h"
#include "..\\interface\\memory.h"
#include "..\\helper\\cudahelper\\cudahelper.cuh"
#else
#include "../helper/cudahelper/cudahelper.cuh"
#include "../includes.h"
#include "../cudaincludes.h"
#include "../interface/memory.h"
#endif

template<typename T>
void hd_fillDiagonals( DeviceMemory<T>& mem );

#endif
