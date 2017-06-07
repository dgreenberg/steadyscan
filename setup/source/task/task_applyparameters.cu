#include "task_applyparameters_hd.cuh"
#include "task_applyparameters.cuh"

template<typename T>
__global__ void d_startApplyParameters( DeviceMemory<T>* mem ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= mem->nParamPoints) return;

	d_applyParameters( mem, idx, mem->p, mem->anyBlocksPresent );
}

template<typename T>
void hd_applyParameters( DeviceMemory<T>& mem ) {
	int nBlocks = ( mem.fframeW + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
	d_startApplyParameters<T><<< nBlocks, THREADS_PER_BLOCK >>>( mem.d_mem );
}

template void hd_applyParameters( DeviceMemory<float>& mem );
template void hd_applyParameters( DeviceMemory<double>& mem );
