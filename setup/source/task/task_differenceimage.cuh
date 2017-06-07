#ifndef TASK_DIFFERENCEIMAGE_CUH_
#define TASK_DIFFERENCEIMAGE_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\cudaincludes.h"
#include "..\\interface\\memory.h"
#else
#include "../cudaincludes.h"
#include "../interface/memory.h"
#endif

template<typename T>
__device__ void d_createDifferenceImage( DeviceMemory<T>* mem, int idx ) {
	for( int row = 0; row < mem->frameH; row++ ) {
		int rid = IDX2R( row, idx, mem->fframeW );
		if( mem->mask[rid] ) {
			int iX = static_cast<int>( mem->x[rid] );
			int iY = static_cast<int>( mem->y[rid] );
			int iXY = IDX2R( iY, iX, mem->templateW );

			T xFrac = static_cast<T>( mem->x[rid] - iX );
			T yFrac = static_cast<T>( mem->y[rid] - iY );

			T wI = ( 1 - xFrac ) * ( ( 1 - yFrac ) * mem->templateImage[iXY] 	 + yFrac * mem->templateImage[iXY + mem->templateW] )
					   + xFrac   * ( ( 1 - yFrac ) * mem->templateImage[iXY + 1] + yFrac * mem->templateImage[iXY + mem->templateW + 1] );

			mem->differenceImage[rid] = mem->image[rid] - wI;
		} else {
			mem->differenceImage[rid] = 0;
		}
	}
}

#endif
