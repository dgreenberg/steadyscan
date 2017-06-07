#include "task_generatesubsparseoffset.cuh"

template<typename T>
__global__ void d_setParamsUsed( DeviceMemory<T>* mem, int nParamsUsed ) {
	mem->nParamsUsed = nParamsUsed;
}

template<typename T>
void hd_generateSubSparseOffsets( DeviceMemory<T>& mem ) {
	int nParamsUsed = 0;
	int curOffset = 0;

	cudaMemcpy( mem.h_paramsUsedBuffer, mem.paramsUsed, sizeof( bool ) * mem.nParamPoints, cudaMemcpyDeviceToHost );

	for( int i = 0; i < mem.nParams; i++ ) {
		if( mem.h_paramsUsedBuffer[i % mem.nParamPoints] ) {
			nParamsUsed++;
		} else {
			curOffset++;
		}
		mem.h_subSparseOffsetBuffer[i] = curOffset;
	}
	mem.nParamsUsed = nParamsUsed;

	cudaMemcpy( mem.subSparseOffsets, mem.h_subSparseOffsetBuffer, sizeof( int ) * mem.nParams, cudaMemcpyHostToDevice );
	d_setParamsUsed<<< 1, 1 >>>( mem.d_mem, nParamsUsed );
}

template void hd_generateSubSparseOffsets( DeviceMemory<float>& mem );
template void hd_generateSubSparseOffsets( DeviceMemory<double>& mem );
