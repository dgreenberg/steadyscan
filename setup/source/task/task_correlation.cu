#include "task_correlation.cuh"

template<typename T>
__device__ void d_blockReduce( T* arr, int numElements2 ) {
	for( int stride = ( numElements2 >> 1 ); stride; stride >>= 1 ) {
		if( threadIdx.x < stride ) {
			int ui = threadIdx.x + stride;
			if( ui < blockDim.x ) {
				arr[threadIdx.x] += arr[ui];
			}
		}
		__syncthreads();
	}
}

template<typename T>
__global__ void d_globalCorrelation( DeviceMemory<T>* mem, T divSumA, T divSumB, T* image ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= mem->fframeW ) return;

	T assq = 0;
	T bssq = 0;
	T q = 0;
	for( int i = 0, pxId = idx; i < mem->frameH; i++ ) {
		if(mem->mask[pxId]) {
			T va = ( mem->differenceImage[pxId] - divSumA ); //wI[pxId]
			assq += va * va;

			T vb = ( image[pxId] - divSumB );
			bssq += vb * vb;

			q += va * vb;
			pxId += mem->fframeW;
		}
	}

	mem->qcAssq[idx] = assq;
	mem->qcBssq[idx] = bssq;
	mem->qcq[idx] = q;
}

template<typename T>
__global__ void d_clearImage( bool* mask, T* image, T* dest, int N ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= N ) return;

	T iv[2] = { 0, image[idx] };
	dest[idx] = iv[mask[idx]];
}

template<typename T>
__global__ void d_swapWIDiff( DeviceMemory<T>* mem, int N ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= N ) return;

	if(!mem->mask[idx]) return;
	//differenceImage = image - wI;
	//wI = image - differenceImage
	mem->differenceImage[idx] = mem->image[idx] - mem->differenceImage[idx];
}

template<typename T>
void hd_globalCorrelation( DeviceMemory<T>& d_Ptr, T* corr ) {
	int numBlocks;
	int numPixel = d_Ptr.frameH * d_Ptr.fframeW;

	numBlocks = ( numPixel + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
	d_clearImage<T><<< numBlocks, THREADS_PER_BLOCK >>>( d_Ptr.mask, d_Ptr.image, d_Ptr.qcImg, numPixel );
	d_swapWIDiff<T><<< numBlocks, THREADS_PER_BLOCK >>>( d_Ptr.d_mem, numPixel );

	thrust::device_ptr<T> qcImg = thrust::device_pointer_cast( d_Ptr.image );
	thrust::device_ptr<T> qcWi = thrust::device_pointer_cast( d_Ptr.differenceImage ); //changed from d_Ptr.wI
	thrust::device_ptr<bool> qcMask = thrust::device_pointer_cast( d_Ptr.mask );

	cudaDeviceSynchronize();

	T divSumA = thrust::reduce( qcImg, qcImg + numPixel );

	T divSumB = thrust::reduce( qcWi, qcWi + numPixel );
	int n = thrust::count( qcMask, qcMask + numPixel, true );

	divSumA /= n;
	divSumB /= n;

	numBlocks = ( d_Ptr.fframeW + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
	d_globalCorrelation<T> <<< numBlocks, THREADS_PER_BLOCK >>>( d_Ptr.d_mem, divSumA, divSumB, d_Ptr.qcImg );

	thrust::device_ptr<T> dpAssq = thrust::device_pointer_cast( d_Ptr.qcAssq );
	thrust::device_ptr<T> dpBssq = thrust::device_pointer_cast( d_Ptr.qcBssq );
	thrust::device_ptr<T> dpQcq = thrust::device_pointer_cast( d_Ptr.qcq );

	T assq = thrust::reduce( dpAssq, dpAssq + d_Ptr.fframeW );
	T bssq = thrust::reduce( dpBssq, dpBssq + d_Ptr.fframeW );
	T q = thrust::reduce( dpQcq, dpQcq + d_Ptr.fframeW );

	*corr = q / sqrt( assq * bssq );

	d_swapWIDiff<T><<< numBlocks, THREADS_PER_BLOCK >>>( d_Ptr.d_mem, numPixel );
}

template<typename T>
__global__ void d_frameCorrelation( DeviceMemory<T>* mem ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ char s_mem[];

	int idpx = idx;

	T* buffA = reinterpret_cast<T*>( &s_mem[sizeof(T) * blockDim.x * 0] );
	T* buffB = reinterpret_cast<T*>( &s_mem[sizeof(T) * blockDim.x * 1] );
	int* buffC = reinterpret_cast<int*>( &s_mem[sizeof(T) * blockDim.x * 2] );
	T* tbuffC = reinterpret_cast<T*>( buffC );

	T blob[2]; blob[0] = 0;

	buffA[threadIdx.x] = 0;
	buffB[threadIdx.x] = 0;
	buffC[threadIdx.x] = 0;

	for( int i = 0; i < mem->frameH; i++ ) {
		int used = mem->mask[idpx];
		buffC[threadIdx.x] += used;

		blob[1] = mem->image[idpx];
		buffB[threadIdx.x] += blob[used];

		blob[1] = mem->differenceImage[idpx]; //changed from wI
		buffA[threadIdx.x] += blob[used];

		idpx += mem->fframeW;
	}

	::d_blockReduce<T>( buffA, mem->groupSize2 );
	::d_blockReduce<T>( buffB, mem->groupSize2 );
	::d_blockReduce<int>( buffC, mem->groupSize2 );

	//buffA is actually differenceImage not wI
	//differenceImage = image - wI;
	//sum(wI) = sum(image) - sum(differenceImage)
	T divSumA = buffA[0] / static_cast<T>( buffC[0] );
	T divSumB = buffB[0] / static_cast<T>( buffC[0] );

	T assq = 0;
	T bssq = 0;
	T q = 0;

	blob[1] = 1.0;
	idpx = idx;
	for( int i = 0; i < mem->frameH; i++ ) {

		T va = ( mem->differenceImage[idpx] - divSumA ) * blob[mem->mask[idpx]];
		assq += va * va;

		T vb = ( mem->image[idpx] - divSumB ) * blob[mem->mask[idpx]];
		bssq += vb * vb;

		q += va * vb;
		idpx += mem->fframeW;
	}

	buffA[threadIdx.x] = assq;
	buffB[threadIdx.x] = bssq;
	tbuffC[threadIdx.x] = q;

	__syncthreads();

	::d_blockReduce<T>( buffA, mem->groupSize2 );
	::d_blockReduce<T>( buffB, mem->groupSize2 );
	::d_blockReduce<T>( tbuffC, mem->groupSize2 );


	mem->corr[blockIdx.x] = tbuffC[0] / sqrt( buffA[0] * buffB[0] );
}

template<typename T>
void hd_frameCorrelation( DeviceMemory<T>& d_Ptr ) {
	int numPixel = d_Ptr.frameH * d_Ptr.fframeW;
	int numBlocks = (numPixel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	d_swapWIDiff<T><<< numBlocks, THREADS_PER_BLOCK >>>( d_Ptr.d_mem, numPixel );
	d_frameCorrelation<<< d_Ptr.nFrames, d_Ptr.frameW, d_Ptr.frameW * sizeof( T ) * 3 >>>( d_Ptr.d_mem );
	d_swapWIDiff<T><<< numBlocks, THREADS_PER_BLOCK >>>( d_Ptr.d_mem, numPixel );
}


template void hd_globalCorrelation( DeviceMemory<float>& d_Ptr, float* corr );
template void hd_globalCorrelation( DeviceMemory<double>& d_Ptr, double* corr );

template void hd_frameCorrelation( DeviceMemory<float>& mem );
template void hd_frameCorrelation( DeviceMemory<double>& mem );
