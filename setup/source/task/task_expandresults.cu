#include "task_expandresults.cuh"

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\helper\\minmax.h"
#else
#include "../helper/minmax.h"
#endif

template<typename T>
__global__ void d_expandResults( DeviceMemory<T>* mem, int shift ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= mem->nParams ) return;

	int idx2 = idx - mem->nParamPoints;

	int j = ( idx >= mem->nParamPoints ) ? ( idx2 ) : ( idx );

	T val[] = { 0, mem->deltaP[idx], mem->newDeltaP[idx - mem->subSparseOffsets[idx]] };

	bool isFixedParam = ( ( ( ( idx + 1 ) - ( shift * ( mem->groupSize >> 1 ) ) ) % mem->groupSize ) == 0 );

	switch( isFixedParam + 2 * mem->paramsUsed[j] ) {
	case 0: //not fixed + not used
		mem->deltaP[idx] = val[0];
		break;
	case 1: //fixed + not used
		mem->deltaP[idx] = val[1];
		break;
	case 2://not fixed + used
		mem->deltaP[idx] = val[2];
		break;
	case 3: //fixed + used (should never happen)
		mem->deltaP[idx] = val[1];
		break;
	}
}
template<typename T>
void hd_expandResults( DeviceMemory<T>& mem, int shift ) {
	int numBlocks = ( mem.nParams + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
	d_expandResults<T><<< numBlocks, THREADS_PER_BLOCK >>>( mem.d_mem, shift );
}

template void hd_expandResults( DeviceMemory<float>& mem, int shift );
template void hd_expandResults( DeviceMemory<double>& mem, int shift );
