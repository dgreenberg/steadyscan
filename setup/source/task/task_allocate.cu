#include "task_allocate.cuh"

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\cudaincludes.h"
#include "..\\helper\\cudahelper\\cudahelper.cuh"
#include "..\\interface\\neutralmatrix\\neutralmatrix.h"
#include "..\\interface\\parameterorganizer\\parameterorganizer.h"
#include "..\\helper\\minmax.h"
#include "..\\mex\\reporter\\reporter.h"
#else
#include "../cudaincludes.h"
#include "../helper/cudahelper/cudahelper.cuh"
#include "../interface/neutralmatrix/neutralmatrix.h"
#include "../interface/parameterorganizer/parameterorganizer.h"
#include "../helper/minmax.h"
#include "../mex/reporter/reporter.h"
#endif

template<typename T>
void h_freeDeviceMemory( DeviceMemory<T>& mem ) {

#ifdef MINPOINTSPERBLOCK_AS_VECTOR
	zCudaFree( mem.minPointsPerBlock );
#endif

	CudaHelper<T>::freeCuSolver( mem );
	CudaHelper<T>::freeCuSparse( mem );

	zCudaFreeHost( mem.h_slvValues );
	zCudaFreeHost( mem.h_slvNewDeltaP );
	zCudaFreeHost( mem.h_slvJTr );
	zCudaFreeHost( mem.h_slvRowInd );
	zCudaFreeHost( mem.h_slvColInd );
	zCudaFreeHost( mem.h_penaltyBuffer );
	zCudaFreeHost( mem.h_paramsUsedBuffer );
	zCudaFreeHost( mem.h_subSparseOffsetBuffer );
	zCudaFreeHost( mem.h_errval );
	zCudaFreeHost( mem.h_rgParams );

	zCudaFree( mem.templateImage );
	zCudaFree( mem.xGradients );
	zCudaFree( mem.yGradients );
	zCudaFree( mem.frac );
	zCudaFree( mem.baseX );
	zCudaFree( mem.baseY );
	zCudaFree( mem.baseMask );
	zCudaFree( mem.image );
	zCudaFree( mem.p );
	zCudaFree( mem.pLast );
	zCudaFree( mem.groupActive );
	zCudaFree( mem.paramsUsed );
	zCudaFree( mem.blocksPresent );
	zCudaFree( mem.jtr );
	zCudaFree( mem.pnew );
	zCudaFree( mem.deltaP );
	zCudaFree( mem.newDeltaP );
	zCudaFree( mem.mask );
	zCudaFree( mem.corr );
	zCudaFree( mem.corrNew );
	zCudaFree( mem.x );
	zCudaFree( mem.y );
	zCudaFree( mem.wxgrad );
	zCudaFree( mem.wygrad );
	zCudaFree( mem.differenceImage );
	zCudaFree( mem.anyBlocksPresent );
	zCudaFree( mem.qcAssq );
	zCudaFree( mem.qcBssq );
	zCudaFree( mem.qcq );
	zCudaFree( mem.qcImg );
	zCudaFree( mem.errval );
	zCudaFree( mem.H.rowInd );
	zCudaFree( mem.H.colInd );
	zCudaFree( mem.H.values );
	zCudaFree( mem.H.intermediateBuffer );
	zCudaFree( mem.H.csrRowInd );
	zCudaFree( mem.diagonalSortedValues );
	zCudaFree( mem.diagonalSortedIndices );
	zCudaFree( mem.diagonalIndices );
	zCudaFree( mem.subSparseOffsets );

	zCudaFree( mem.d_mem );
}


/**
 * allocated memory that is used during execute
 */
template<typename T>
int h_allocateDeviceMemory( PreAllocatedMemory<T>& rom, DeviceMemory<T>& mem ) {
	cudaError_t rc;
	NeutralMatrix<T, int, RowMajor<int>> organizedParameters;
	memset( &mem, 0, sizeof( DeviceMemory<T> ) );
	*reinterpret_cast< ReadOnlyVariables<T>* >( &mem ) = *reinterpret_cast< ReadOnlyVariables<T>* >( &rom );

	//this library is only used when the cusolver solver is selected
	if( rom.solverID == 1 ) {
		if( !CudaHelper<T>::initCuSolver( mem ) ) {
			return -3;
		}
	} else mem.cuSolverHandle = nullptr;

	//this library is used when either the cusolver or cusparse solver is selected
	if( rom.solverID < 2 ) {
		if( !CudaHelper<T>::initCuSparse( mem ) ) {
			return -2;
		}
	} else mem.cuSparseHandle = nullptr;

	//only allocate there when host solver is enabled
	if( rom.solverID == 2 ) {
		if( !CudaHelper<T>::hAlloc( &mem.h_slvValues, rom.nParams * 6, cudaHostAllocDefault, "h_slvValues" ) ) goto CLEANUP;
		if( !CudaHelper<T>::hAlloc( &mem.h_slvNewDeltaP, rom.nParams, cudaHostAllocDefault, "h_slvNewDeltaP" ) ) goto CLEANUP;
		if( !CudaHelper<T>::hAlloc( &mem.h_slvJTr, rom.nParams, cudaHostAllocDefault, "h_slvJTr" ) ) goto CLEANUP;
		if( !CudaHelper<int>::hAlloc( &mem.h_slvRowInd, rom.nParams * 6, cudaHostAllocDefault, "h_slvRowInd" ) ) goto CLEANUP;
		if( !CudaHelper<int>::hAlloc( &mem.h_slvColInd, rom.nParams * 6, cudaHostAllocDefault, "h_slvColInd" ) ) goto CLEANUP;
	}
	if( !CudaHelper<T>::hAlloc( &mem.h_penaltyBuffer, rom.nParams, cudaHostAllocDefault, "h_penaltyBuffer" ) ) goto CLEANUP;
	if( !CudaHelper<bool>::hAlloc( &mem.h_paramsUsedBuffer, rom.nParamPoints, cudaHostAllocDefault, "h_paramsUsedBuffer" ) ) goto CLEANUP;
	if( !CudaHelper<int>::hAlloc( &mem.h_subSparseOffsetBuffer, rom.nParams, cudaHostAllocDefault, "h_subSparseOffsetBuffer" ) ) goto CLEANUP;
	if( !CudaHelper<T>::hAlloc( &mem.h_errval, rom.nFrames * 2, cudaHostAllocDefault, "h_errval" ) ) goto CLEANUP;

	if( !CudaHelper<T>::hAlloc( &mem.h_rgParams, rom.nParams * 2 * rom.maxIterations, cudaHostAllocDefault, "h_rgParams" ) ) goto CLEANUP;

	//support for 2D parameter input
	//x11x12x13y11y12y13 (initParameters)
	//x21x22x23y21y22y23 (initParameters)
	//and 2x 1D parameter input
	//x11x12x13x21x22x23 (initParameterX)
	//y11y12y13y21y22y23 (initParameterY)
	if( nullptr != rom.initParameters.data() ) {
		ParameterOrganizer<T>::organize2D1( rom.initParameters, organizedParameters, rom.nFrames, rom.frameW, rom.nParamPoints );
	} else {
		ParameterOrganizer<T>::organize2D1( rom.initParameterX, rom.initParameterY, organizedParameters, rom.nFrames, rom.frameW, rom.nParamPoints );
	}

	//copy matrices from host to device
	if( !CudaHelper<T>::h2d( &mem.templateImage, rom.templateImage, "templateImage" ) ) goto CLEANUP;
	if( !CudaHelper<T>::h2d( &mem.xGradients, rom.xGradients, "xGradients" ) ) goto CLEANUP;
	if( !CudaHelper<T>::h2d( &mem.yGradients, rom.yGradients, "yGradients" ) ) goto CLEANUP;
	if( !CudaHelper<T>::h2d( &mem.frac, rom.frac, "frac" ) ) goto CLEANUP;
	if( !CudaHelper<T>::h2d( &mem.baseX, rom.baseX, "baseX" ) ) goto CLEANUP;
	if( !CudaHelper<T>::h2d( &mem.baseY, rom.baseY, "baseY" ) ) goto CLEANUP;
	if( !CudaHelper<bool>::h2d( &mem.baseMask, rom.baseMask, "baseMask" ) ) goto CLEANUP;
	if( !CudaHelper<T>::h2d( &mem.image, rom.image, "image" ) ) goto CLEANUP;
	if( !CudaHelper<T>::h2d( &mem.p, organizedParameters, "parameters" ) ) goto CLEANUP;
#ifdef MINPOINTSPERBLOCK_AS_VECTOR
	if( !CudaHelper<int>::h2d( &mem.minPointsPerBlock, rom.minPointsPerBlock, "minPointsPerBlock" ) ) goto CLEANUP;
#endif

	if( !CudaHelper<T>::dAlloc( &mem.pLast, mem.nParams, "pLast" ) ) goto CLEANUP;
	if( !CudaHelper<bool>::dAlloc( &mem.groupActive, mem.nGroups * 2, "groupActive", true ) ) goto CLEANUP;
	if( !CudaHelper<bool>::dAlloc( &mem.paramsUsed, mem.nParamPoints, "paramsUsed" ) ) goto CLEANUP;
	if( !CudaHelper<bool>::dAlloc( &mem.blocksPresent, mem.nParamPoints, "blocksPresent" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.jtr, mem.nParams, "jtr" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.pnew, mem.nParams, "pnew", 1E6 ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.deltaP, mem.nParams, "deltaP", 0 ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.newDeltaP, mem.nParams, "newDeltaP" ) ) goto CLEANUP;
	if( !CudaHelper<bool>::dAlloc( &mem.mask, mem.frameH * mem.fframeW, "mask" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.corr, mem.nFrames, "corr" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.corrNew, mem.nFrames, "corrNew" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.x, mem.frameH * mem.fframeW, "x" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.y, mem.frameH * mem.fframeW, "y" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.wxgrad, mem.frameH * mem.fframeW, "wxgrad" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.wygrad, mem.frameH * mem.fframeW, "wygrad" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.differenceImage, mem.frameH * mem.fframeW, "differenceImage" ) ) goto CLEANUP;
	if( !CudaHelper<bool>::dAlloc( &mem.anyBlocksPresent, 1, "anyBlocksPresent" ) ) goto CLEANUP;

	if( !CudaHelper<T>::dAlloc( &mem.qcAssq, mem.fframeW, "qcAssq" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.qcBssq, mem.fframeW, "qcBssq" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.qcq, mem.fframeW, "qcq" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.qcImg, mem.fframeW * mem.frameH, "qcImg" ) ) goto CLEANUP;

	if( !CudaHelper<T>::dAlloc( &mem.errval, mem.nFrames * 2, "errval", 1E6 ) ) goto CLEANUP;

	if( !CudaHelper<int>::dAlloc( &mem.H.rowInd, mem.nParams * 6, "rowInd" ) ) goto CLEANUP;
	if( !CudaHelper<int>::dAlloc( &mem.H.colInd, mem.nParams * 6, "colInd" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.H.values, mem.nParams * 6, "values" ) ) goto CLEANUP;
	if( !CudaHelper<T>::dAlloc( &mem.H.intermediateBuffer, mem.nParams * 6, "intermediateBuffer" ) ) goto CLEANUP;
	if( !CudaHelper<int>::dAlloc( &mem.H.csrRowInd, mem.nParams * 6, "csrRowInd" ) ) goto CLEANUP;

	if( !CudaHelper<T>::dAlloc( &mem.diagonalSortedValues, mem.nParams * 6, "diagonalSortedValues" ) ) goto CLEANUP;
	if( !CudaHelper<int>::dAlloc( &mem.diagonalSortedIndices, mem.nParams * 6, "diagonalSortedIndices" ) ) goto CLEANUP;
	if( !CudaHelper<int>::dAlloc( &mem.diagonalIndices, mem.nParams * 6, "diagonalIndices" ) ) goto CLEANUP;
	if( !CudaHelper<int>::dAlloc( &mem.subSparseOffsets, mem.nParams, "subSparseOffsets" ) ) goto CLEANUP;
//####################DEBUG VARS#########################



//###################END ALLOCATION######################
	rc = cudaMalloc( &mem.d_mem, sizeof( DeviceMemory<T> ) );
	if( rc != cudaError::cudaSuccess ) {
		char buff[128];
		sprintf(buff, "error allocating %lu bytes for devicememory struct\n", sizeof( DeviceMemory<T> ) );
		Reporter::inform(buff);
		goto CLEANUP;
	}

	//copy pointer struct to gpu
	rc = cudaMemcpy( mem.d_mem, &mem, sizeof( DeviceMemory<T> ), cudaMemcpyHostToDevice );
	if( rc != cudaError::cudaSuccess ) goto CLEANUP;

	return 0;
CLEANUP:
	h_freeDeviceMemory( mem );
	return -1;
}


template void h_freeDeviceMemory( DeviceMemory<float>& mem );
template void h_freeDeviceMemory( DeviceMemory<double>& mem );

template int h_allocateDeviceMemory( PreAllocatedMemory<float>& rom, DeviceMemory<float>& mem );
template int h_allocateDeviceMemory( PreAllocatedMemory<double>& rom, DeviceMemory<double>& mem );
