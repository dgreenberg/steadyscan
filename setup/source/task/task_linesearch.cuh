#ifndef TASK_LINESEARCH_CUH_
#define TASK_LINESEARCH_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\cudaincludes.h"
#include "..\\interface\\memory.h"
#include "..\\helper\\cudahelper\\cudahelper.cuh"
#else
#include "../cudaincludes.h"
#include "../interface/memory.h"
#include "../helper/cudahelper/cudahelper.cuh"
#endif

template<typename T>
bool hd_linesearch(DeviceMemory<T>& d_Ptr, int shift);


#endif
