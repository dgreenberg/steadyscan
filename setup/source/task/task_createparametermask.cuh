#ifndef TASK_CREATEPARAMETERMASK_CUH_
#define TASK_CREATEPARAMETERMASK_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\interface\\memory.h"
#else
#include "../interface/memory.h"
#endif

//fframeW threads
template<typename T>
__device__ void d_createParameterMask( DeviceMemory<T>* mem, int idx ) {
	if( idx == 0 ) {
		mem->paramsUsed[idx] = mem->blocksPresent[idx];
	} else if( idx == mem->nParamPoints - 1 ) {
		mem->paramsUsed[idx] = mem->blocksPresent[idx - 1];
	} else {
		mem->paramsUsed[idx] = ( mem->blocksPresent[idx] | mem->blocksPresent[idx - 1] );
	}
}

#endif
