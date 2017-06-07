#ifndef TASK_FILLJTR_H_
#define TASK_FILLJTR_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\includes.h"
#include "..\\cudaincludes.h"
#include "..\\interface\\memory.h"
#else
#include "../includes.h"
#include "../cudaincludes.h"
#include "../interface/memory.h"
#endif


template<typename T>
void hd_fillJtr( DeviceMemory<T>& mem );

#endif
