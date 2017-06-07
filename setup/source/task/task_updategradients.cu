#include "task_updategradients.cuh"

template<typename T>
__global__ void d_updateGradients( DeviceMemory<T>* mem ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= mem->fframeW ) return;

	int idxy = IDX2R( 0, idx, mem->fframeW );

	for(int row = 0; row < mem->frameH; row++) {
		if( mem->mask[idxy] ) {
			int iX = static_cast<int>( mem->x[idxy] );
			int iY = static_cast<int>( mem->y[idxy] );

			int tidxy = IDX2R( iY, iX, mem->templateW );
			T xFrac = mem->x[idxy] - static_cast<T>( iX );
			T yFrac = mem->y[idxy] - static_cast<T>( iY );

			mem->wxgrad[idxy] = ( 1 - yFrac ) * mem->xGradients[tidxy] + yFrac * mem->xGradients[tidxy + mem->templateW];
			mem->wygrad[idxy] = ( 1 - xFrac ) * mem->yGradients[tidxy] + xFrac * mem->yGradients[tidxy + 1];
		} else {
			mem->wxgrad[idxy] = 0;
			mem->wygrad[idxy] = 0;
		}

		idxy += mem->fframeW;
	}

}

template<typename T>
void hd_updateGradients( DeviceMemory<T>& mem ) {
	int numBlocks = ( mem.fframeW + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
	d_updateGradients<T><<< numBlocks, THREADS_PER_BLOCK >>>( mem.d_mem );
}

template void hd_updateGradients( DeviceMemory<float>& mem );
template void hd_updateGradients( DeviceMemory<double>& mem );

//template void hd_updateBlockGradients( DeviceMemory<float>& mem, int groupBase );
//template void hd_updateBlockGradients( DeviceMemory<double>& mem, int groupBase );
