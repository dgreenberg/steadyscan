#include "task_calcerrval.cuh"

#include "task_createparametermask.cuh"
#include "task_differenceimage.cuh"

template<typename T>
__device__ void d_calcBaseErrval( DeviceMemory<T>* mem, int idx, T* errvalParts, int* errvalPxCounts ) {
	T errval = 0;
	int pxCount = 0;

	for( int row = 0; row < mem->frameH; row++ ) {
		errval += mem->differenceImage[idx] * mem->differenceImage[idx];
		pxCount += mem->mask[idx]?1:0;
		idx += mem->fframeW;
	}

	errvalPxCounts[threadIdx.x] = pxCount;
	errvalParts[threadIdx.x] = errval;
}

template<typename T>
__global__ void d_calcAllGroupErrval( DeviceMemory<T>* mem, int shift ) {
	extern __shared__ char s_mem[];
	int* errvalPxCounts = reinterpret_cast<int*>( &s_mem[sizeof( T ) * mem->frameW] );
	T* errvalParts = reinterpret_cast<T*>( s_mem );

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int groupId = idx / mem->groupSize;

	if( !mem->groupActive[groupId + shift * mem->nGroups] ) return;

	d_createParameterMask<T>( mem, idx );
	__syncthreads();
	d_createDifferenceImage<T>( mem, idx );
	d_calcBaseErrval<T>( mem, idx, errvalParts, errvalPxCounts );
	__syncthreads();

	for( int stride = ( mem->frameW2 >> 1 ); stride; stride >>= 1 ) {
		if( threadIdx.x < stride ) {
			int ui = threadIdx.x + stride;
			if( ui < mem->frameW ) {
				errvalParts[threadIdx.x] += errvalParts[ui];
				errvalPxCounts[threadIdx.x] += errvalPxCounts[ui];
			}
		}
		__syncthreads();
	}

	mem->errval[shift * mem->nFrames + blockIdx.x] = errvalParts[0] / static_cast<T>( errvalPxCounts[0] );
}

template<typename T>
void hd_calculateGroupErrvals( DeviceMemory<T>& mem, int shift ) {
	d_calcAllGroupErrval<T><<< mem.nFrames, mem.frameW, ( sizeof( T ) + sizeof( int ) ) * mem.frameW >>>( mem.d_mem, shift );
}

template void hd_calculateGroupErrvals( DeviceMemory<float>& mem, int shift );
template void hd_calculateGroupErrvals( DeviceMemory<double>& mem, int shift );
