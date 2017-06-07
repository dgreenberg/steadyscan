#ifndef TASK_ALLOCATE_CUH_
#define TASK_ALLOCATE_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\interface\\memory.h"
#else
#include "../interface/memory.h"
#endif


template<typename T>
int h_allocateDeviceMemory( PreAllocatedMemory<T>& rom, DeviceMemory<T>& mem );

template<typename T>
void h_freeDeviceMemory( DeviceMemory<T>& mem );

#endif
