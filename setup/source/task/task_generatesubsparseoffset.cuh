#ifndef TASK_GENERATESUBSPARSEOFFSET_CUH_
#define TASK_GENERATESUBSPARSEOFFSET_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\cudaincludes.h"
#include "..\\interface\\memory.h"
#else
#include "../cudaincludes.h"
#include "../interface/memory.h"
#endif

template<typename T>
void hd_generateSubSparseOffsets( DeviceMemory<T>& mem );

#endif
