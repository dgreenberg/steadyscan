#include "task_filljtr.cuh"

template<typename T>
__global__ void d_fillJTr( DeviceMemory<T>* mem ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= mem->nParamPoints ) return;
	if( !mem->paramsUsed[idx] ) return;

	T sumx = 0;
	T sumy = 0;
	T sumx2 = 0;
	T sumy2 = 0;

	int idxm = idx % mem->frameW;

	if( idx != mem->fframeW ) {
		for( int row = 0; row < mem->frameH; row++ ) {
			int idxy = IDX2R( row, idx, mem->fframeW );
#ifdef FRAC_AS_MAT
			T frac = mem->frac[IDX2R( row, idxm, mem->frameW )];
#else
			T frac = mem->frac[row];
#endif

			T fdi = mem->differenceImage[idxy] * ( 1 - frac ); 

			sumx += fdi * mem->wxgrad[idxy];
			sumy += fdi * mem->wygrad[idxy];
		}
	}

	__syncthreads();

	if( idx != 0 ) {
		if( idxm == 0 ) idxm = mem->frameW - 1;
		else idxm -= 1;
		
		for( int row = 0; row < mem->frameH; row++ ) {
			int idxy = IDX2R( row, idx - 1, mem->fframeW );
#ifdef FRAC_AS_MAT
			T frac = mem->frac[IDX2R( row, idxm, mem->frameW )];
#else
			T frac = mem->frac[row];
#endif

			T fdi = mem->differenceImage[idxy] * frac;

			sumx2 += fdi * mem->wxgrad[idxy];
			sumy2 += fdi * mem->wygrad[idxy];
		}
	}

	__syncthreads();

	mem->jtr[idx - mem->subSparseOffsets[idx]] =  sumx + sumx2;
	idx += mem->nParamPoints;
	mem->jtr[idx - mem->subSparseOffsets[idx]] = sumy + sumy2;
}

template<typename T>
void hd_fillJtr( DeviceMemory<T>& mem )  {
	int numBlocks = ( mem.nParamPoints + ( THREADS_PER_BLOCK - 1 ) ) / THREADS_PER_BLOCK;
	d_fillJTr<T><<< numBlocks, THREADS_PER_BLOCK >>>( mem.d_mem );
}

template void hd_fillJtr( DeviceMemory<float>& d_Ptr );
template void hd_fillJtr( DeviceMemory<double>& d_Ptr );
