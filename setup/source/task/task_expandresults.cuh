#ifndef TASK_EXPANDRESULT_CUH_
#define TASK_EXPANDRESULT_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\cudaincludes.h"
#include "..\\interface\\memory.h"
#else
#include "../cudaincludes.h"
#include "../interface/memory.h"
#endif

template<typename T>
void hd_expandResults( DeviceMemory<T>& mem, int shift );

#endif
