#include "task_maskinactive.cuh"

template<typename T>
__global__ void d_maskInactiveGroups( DeviceMemory<T>* mem ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= mem->fframeW ) return;

	int groupSize_2 = ( mem->groupSize >> 1 );

	int frame = idx / mem->frameW;
	if( mem->corr[frame] < mem->haltCorrelation ) {
		mem->groupActive[idx / mem->groupSize] = true;
		if( idx < groupSize_2 ) return;
		if( idx >= mem->fframeW - groupSize_2 ) return;

		mem->groupActive[( idx - groupSize_2 ) / mem->groupSize + mem->nGroups] = true;
	}
}

template<typename T>
void hd_maskInactiveGroups( DeviceMemory<T>& mem ) {
	CudaHelper<bool>::setArray(mem.groupActive, false, mem.nGroups * 2);
	int numBlocks = ( mem.fframeW + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
	d_maskInactiveGroups<T><<<numBlocks, THREADS_PER_BLOCK>>>( mem.d_mem );
}

template void hd_maskInactiveGroups( DeviceMemory<float>& mem );
template void hd_maskInactiveGroups( DeviceMemory<double>& mem );
