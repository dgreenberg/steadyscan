#ifndef TASK_APPLYGROUPPARAMETERS_CUH_
#define TASK_APPLYGROUPPARAMETERS_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\interface\\memory.h"
#include "..\\cudaincludes.h"
#include "..\\helper\\eps.h"
#else
#include "../interface/memory.h"
#include "../cudaincludes.h"
#include "../helper/eps.h"
#endif

template<typename T>
__device__ void d_applyParameters( DeviceMemory<T>* mem, int idx, T* p, bool* anyBlocksPresent ) {
	if( idx >= mem->fframeW ) return;

	int idxm = idx % mem->frameW;

	T px = p[idx];
	T px1 = p[idx + 1];
	T py = p[idx + mem->nParamPoints];
	T py1 = p[idx + mem->nParamPoints + 1];

	T templateW = static_cast< T >( mem->templateW - 1 );
	T templateH = static_cast< T >( mem->templateH - 1 );

	int pxCount = 0;
	int ipx = idx, ipxm = idxm;

	for( int row = 0; row < mem->frameH; row++ ) {
#ifdef FRAC_AS_MAT
		T frac = mem->frac[ipxm];
#else
		T frac = mem->frac[row];
#endif

		T x = mem->baseX[ipxm] + px * ( 1 - frac ) + px1 * frac;
		T y = mem->baseY[ipxm] + py * ( 1 - frac ) + py1 * frac;

		bool mask = mem->baseMask[ipxm]
			&& x >= 0 && x < templateW - eps<T>()
			&& y >= 0 && y < templateH - eps<T>();

		pxCount += mask;

		mem->mask[ipx] = mask;
		mem->x[ipx] = x;
		mem->y[ipx] = y;

		ipx += mem->fframeW;
		ipxm += mem->frameW;
	}

#ifdef MINPOINTSPERBLOCK_AS_VECTOR
	if( pxCount >= mem->minPointsPerBlock[idxm] ) {
#else
	if( pxCount >= mem->minPointsPerBlock ) {
#endif
		mem->blocksPresent[idx] = true;
		*anyBlocksPresent = true;
	}
}

#endif
