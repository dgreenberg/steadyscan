#include "task_linesearch.cuh"
#include "task_calcerrval.cu"
#include "task_applyparameters.cuh"

template<typename T>
__device__ inline void d_addDeltaP( DeviceMemory<T>* mem, int idx ) {
	mem->pnew[idx] = mem->p[idx] + mem->deltaP[idx];
	idx += mem->nParamPoints;
	mem->pnew[idx] = mem->p[idx] + mem->deltaP[idx];
}
template<typename T>
__device__ inline void d_reduceDeltaP( DeviceMemory<T>* mem, int idx ) {
	mem->deltaP[idx] *= mem->linesearchReductionMultiplier;
	idx += mem->nParamPoints;
	mem->deltaP[idx] *= mem->linesearchReductionMultiplier;
}
template<typename T>
__device__ inline void d_setNewP( DeviceMemory<T>* mem, int idx ) {
	mem->p[idx] = mem->pnew[idx];
	idx += mem->nParamPoints;
	mem->p[idx] = mem->pnew[idx];
}
template<typename T>
__device__ inline void d_checkMovement( DeviceMemory<T>* mem, int idx, bool* movementAboveThreshold ) {
	if( abs( mem->deltaP[idx] ) > mem->moveThreshold
	 || abs( mem->deltaP[idx + mem->nParamPoints] ) > mem->moveThreshold ) {
		*movementAboveThreshold = true;
	}
}

//only one frame per block
//allows up to 1024 parameters per frame
//nFrames blocks & frameW threads per Block
template<typename T>
__global__ void d_linesearch( DeviceMemory<T>* mem, int shift ) {
	extern __shared__ char s_mem[];
	__shared__ bool sharedResult;

	//identifies thread within ff
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//identifies group within all groups
	int groupId = idx / mem->groupSize;

	if( !mem->groupActive[groupId + shift * mem->nGroups] ) return;

	T* errvalParts = reinterpret_cast<T*>( s_mem );
	int* errvalPxCounts = reinterpret_cast<int*>( &s_mem[sizeof(T) * mem->frameW] );

	sharedResult = false;

	do {
		__syncthreads();
		do {
			d_addDeltaP<T>( mem, idx );
			d_applyParameters<T>( mem, idx, mem->pnew, &sharedResult );
			__syncthreads();
			if( sharedResult ) break;
			sharedResult = false;

			d_reduceDeltaP<T>( mem, idx );
			d_checkMovement<T>( mem, idx, &sharedResult );
			__syncthreads();
			if( !sharedResult ) return;
			sharedResult = false;
		} while( true );
		sharedResult = false;

		d_createDifferenceImage<T>( mem, idx );
		d_calcBaseErrval<T>( mem, idx, errvalParts, errvalPxCounts );

		//wait for threads to complete calcGroupErrval
		__syncthreads();

		//reduce block residuals
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

		//compare residuals and apply if lower
		T errvalNew = errvalParts[0] / static_cast<T>( errvalPxCounts[0] );

		if( errvalNew < mem->errval[shift * mem->nFrames + blockIdx.x] ) {
			d_setNewP<T>( mem, idx );
			if( idx == 0 ) {
				mem->errval[shift * mem->nFrames + blockIdx.x] = errvalNew;
			}

			return;
		}

		d_checkMovement<T>( mem, idx, &sharedResult );
		__syncthreads();

		if( !sharedResult ) return;
		sharedResult = false;

		d_reduceDeltaP<T>( mem, idx );
	} while( true );
}

template<typename T>
bool hd_linesearch( DeviceMemory<T>& mem, int shift ) {

	//run threads
	d_linesearch<T> <<<mem.nFrames, mem.frameW, ( sizeof(T) + sizeof(int) ) * mem.frameW>>>( mem.d_mem, shift );

	//copy errvals to host
	cudaMemcpy( mem.h_errval, mem.errval, sizeof(T) * mem.nFrames * 2, cudaMemcpyDeviceToHost );

	//subtract ans sum errvals
	T evSum = 0;
	T evSum1 = 0;
	for( int i = 0; i < mem.nFrames; i++ ) {
		evSum += mem.h_errval[i];
		evSum1 += mem.h_errval[i + mem.nFrames];
	}

	return ( shift && evSum > evSum1 ) || ( !shift && evSum < evSum1 );
}

template bool hd_linesearch( DeviceMemory<float>& mem, int shift );
template bool hd_linesearch( DeviceMemory<double>& mem, int shift );
