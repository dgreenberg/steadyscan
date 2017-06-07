#ifndef UPDATEGRADIENTS_CUH_
#define UPDATEGRADIENTS_CUH_

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
void hd_updateGradients( DeviceMemory<T>& mem );

#endif
