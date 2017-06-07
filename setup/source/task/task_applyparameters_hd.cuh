#ifndef TASK_APPLYPARAMETERS_H_
#define TASK_APPLYPARAMETERS_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\cudaincludes.h"
#include "..\\interface\\memory.h"
#include "..\\helper\\eps.h"
#else
#include "../cudaincludes.h"
#include "../interface/memory.h"
#include "../helper/eps.h"
#endif

template<typename T>
void hd_applyParameters( DeviceMemory<T>& mem );

#endif
