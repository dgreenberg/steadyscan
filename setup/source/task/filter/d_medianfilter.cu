#include "d_medianfilter.cuh"

template<typename T>
__global__ void d_medianfilter(T* p, int nParams) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= nParams ) return;

	T vals[] = { p[idx], p[idx - 1], p[idx + 1]};

	int v = 1 * ( ( vals[1] >= vals[0] && vals[1] < vals[2] ) || ( vals[1] <= vals[0] && vals[1] > vals[2] ) )
		  + 2 * ( ( vals[2] >= vals[0] && vals[2] < vals[1] ) || ( vals[2] >= vals[1] && vals[2] < vals[0] ) );

	p[idx] = vals[v];
}

template<typename T>
void hd_gpumedianfilter( T* values, int nValues ) {
	int numBlocks = ( nValues + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
	d_medianfilter<<<numBlocks, THREADS_PER_BLOCK>>>( values, nValues );
}

template void hd_gpumedianfilter( float* values, int nValues );
template void hd_gpumedianfilter( double* values, int nValues );
