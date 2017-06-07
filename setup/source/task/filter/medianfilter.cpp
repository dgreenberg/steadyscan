#include "medianfilter.h"
#include "d_medianfilter.cuh"

template<class T>
MedianFilter<T>::MedianFilter() {
	return;
}

template<class T>
MedianFilter<T>::~MedianFilter() {
	return;
}

template<class T>
void MedianFilter<T>::apply( T* values, int nValues ) {
	T* d_dataptr;

	cudaMalloc( &d_dataptr, sizeof( T ) * nValues );
	cudaMemcpy( d_dataptr, values, sizeof( T ) * nValues, cudaMemcpyHostToDevice );

	hd_gpumedianfilter( d_dataptr, nValues );

	cudaMemcpy( values, d_dataptr, sizeof( T ) * nValues, cudaMemcpyDeviceToHost );

	cudaFree( d_dataptr );
}

template<class T>
void MedianFilter<T>::applyGPU( T* d_values, int nValues ) {
	hd_gpumedianfilter( d_values, nValues );
}

template class MedianFilter<float>;
template class MedianFilter<double>;
