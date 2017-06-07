#include "task_filldiagonals.cuh"


template<typename T>
__device__ inline void d_swap( T* a, T* b ) {
	T x = *a;
	*a = *b;
	*b = x;
}

template<typename T>
__device__ void d_ISort( DeviceMemory<T>* mem, int start, int end ) {
	for( int i = start + 1; i < end; i++ ) {
		int j = i;
		while( j > start && mem->H.colInd[j - 1] > mem->H.colInd[j] ) {
			d_swap( &mem->H.colInd[j - 1], &mem->H.colInd[j] );
			d_swap( &mem->H.values[j - 1], &mem->H.values[j] );
			j--;
		}
	}
}

template<typename T>
__global__ void d_sortColumnIndices( DeviceMemory<T>* mem ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= mem->H.nnz ) return;

	if( idx != 0 && mem->H.rowInd[idx - 1] == mem->H.rowInd[idx] ) return;

	int end = idx;
	while( mem->H.rowInd[end] == mem->H.rowInd[idx] ) end++;
	d_ISort( mem, idx, end );
}

template<typename T>
__device__ inline void d_insertValue( DeviceMemory<T>* mem, int nzId, T value, int row, int col ) {
	int mrow = row >= mem->nParamPoints ? row - mem->nParamPoints : row;
	int mcol = col >= mem->nParamPoints ? col - mem->nParamPoints : col;

	if( mem->paramsUsed[mrow] && mem->paramsUsed[mcol] ) {
		mem->H.values[nzId] = value;
		mem->H.rowInd[nzId] = row - mem->subSparseOffsets[row];
		mem->H.colInd[nzId] = col - mem->subSparseOffsets[col];
	}
}

template<typename T>
__global__ void d_fillDiagonals( DeviceMemory<T>* mem ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= mem->nParamPoints || !mem->paramsUsed[idx] ) return;

	//offdiagonals
	T offx, offy, offxy;
	//ondiagonals
	T diagx, diagy, diagxy;

	offx = offy = offxy = 0;
	diagx = diagy = diagxy = 0;

	bool lastParam = ( idx == mem->fframeW );
	bool firstParam = ( idx == 0 );

	int idxy = ( idx - lastParam );
	int idxy_1 = ( idx - 1 );

	int idxm = idx % mem->frameW;

	for( int row = 0; row < mem->frameH; row++ ) {
#ifdef FRAC_AS_MAT
		T frac = mem->frac[IDX2R( row, idxm, mem->frameW )];
#else
		T frac = mem->frac[row];
#endif

		T fracsq = frac * frac;
		T compsq = ( 1 - frac ) * ( 1 - frac );
		T fcprod = frac - fracsq;

		T wxgradsq = mem->wxgrad[idxy] * mem->wxgrad[idxy];
		T wygradsq = mem->wygrad[idxy] * mem->wygrad[idxy];
		T wxygrad = mem->wxgrad[idxy] * mem->wygrad[idxy];

		offx += wxgradsq * fcprod;
		offy += wygradsq * fcprod;
		offxy += wxygrad * fcprod;

		switch( static_cast<int>( firstParam ) + ( static_cast<int>( lastParam ) << 1 ) ) {
		case 0:
			diagx += wxgradsq * compsq + mem->wxgrad[idxy_1] * mem->wxgrad[idxy_1] * fracsq;
			diagy += wygradsq * compsq + mem->wygrad[idxy_1] * mem->wygrad[idxy_1] * fracsq;
			diagxy += wxygrad * compsq + mem->wxgrad[idxy_1] * mem->wygrad[idxy_1] * fracsq;
			break;
		case 1:
			diagx += wxgradsq * compsq;
			diagy += wygradsq * compsq;
			diagxy += wxygrad * compsq;
			break;
		case 2:
			diagx += wxgradsq * fracsq;
			diagy += wygradsq * fracsq;
			diagxy += wxygrad * fracsq;
			break;
		}

		idxy += mem->fframeW;
		idxy_1 += mem->fframeW;
	}

	if( !lastParam ) {
		int nnzId = idx;
		d_insertValue( mem, nnzId, diagx, idx, idx );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, offx, idx + 1, idx );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, offx, idx, idx + 1 );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, diagy, idx + mem->nParamPoints, idx + mem->nParamPoints );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, offy, idx + mem->nParamPoints + 1, idx + mem->nParamPoints );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, offy, idx + mem->nParamPoints, idx + mem->nParamPoints + 1 );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, diagxy, idx, idx + mem->nParamPoints );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, offxy, idx, idx + mem->nParamPoints + 1 );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, offxy, idx + 1, idx + mem->nParamPoints );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, diagxy, idx + mem->nParamPoints, idx );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, offxy, idx + mem->nParamPoints + 1, idx );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, offxy, idx + mem->nParamPoints, idx + 1 );
	} else {
		int nnzId = idx;

		d_insertValue( mem, nnzId, diagx, idx, idx );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, diagy, idx + mem->nParamPoints, idx + mem->nParamPoints );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, diagxy, idx + mem->nParamPoints, idx );
		nnzId += mem->nParamPoints;

		d_insertValue( mem, nnzId, diagxy, idx, idx + mem->nParamPoints );
	}
}

template<typename T>
__global__ void d_countNonZeros( DeviceMemory<T>* mem, int maskValue ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= mem->nParams * 6 || idx == 0 ) return;

	if( mem->H.rowInd[idx - 1] != maskValue && mem->H.rowInd[idx] == maskValue ) {
		mem->H.nnz = idx;
	}
}


//assumes INT_MAX is an invalid row index
template<typename T>
void hd_fillDiagonals( DeviceMemory<T>& mem ) {
	int nnz = mem.nParams * 6;
	//init row indices with INT_MAX so that sort moves unused indices to the end of the array
	CudaHelper<int>::setArray( mem.H.rowInd, INT_MAX, nnz );

	int numBlocks = ( mem.nParamPoints + ( THREADS_PER_BLOCK - 1 ) ) / THREADS_PER_BLOCK;
	d_fillDiagonals<T><<< numBlocks, THREADS_PER_BLOCK >>>( mem.d_mem );
	//cudaDeviceSynchronize();

	//eigen solver does not rely on sorted indices
	//nnzs are counted by the solver on insert
	if( mem.solverID != 2 ) {

		//move unused params to the end by sorting the array (unused index = MAX_INT)
		thrust::device_ptr<T> values = thrust::device_pointer_cast( mem.H.values );
		thrust::device_ptr<int> rowInd = thrust::device_pointer_cast( mem.H.rowInd );
		thrust::device_ptr<int> colInd = thrust::device_pointer_cast( mem.H.colInd );

		thrust::device_ptr<T> diagSortedValues = thrust::device_pointer_cast( mem.diagonalSortedValues );
		thrust::device_ptr<int> diagSortedIndices = thrust::device_pointer_cast( mem.diagonalSortedIndices );
		thrust::device_ptr<int> diagIndices = thrust::device_pointer_cast( mem.diagonalIndices );

		//generate index vector, [0, 1, 2....n-1]
		thrust::sequence( diagIndices, diagIndices + nnz );

		//sort the indices using row indices as key
		thrust::sort_by_key( rowInd, rowInd + nnz, diagIndices );

		//reorder other arrays by 'sorted' indices
		thrust::gather( thrust::device, diagIndices, diagIndices + nnz, values, diagSortedValues );
		thrust::copy( diagSortedValues, diagSortedValues + nnz, values );

		thrust::gather( thrust::device, diagIndices, diagIndices + nnz, colInd, diagSortedIndices );
		thrust::copy( diagSortedIndices, diagSortedIndices + nnz, colInd );

		numBlocks = ( nnz + ( THREADS_PER_BLOCK - 1 ) ) / THREADS_PER_BLOCK;
		//determine the actual number of nonzero elements
		d_countNonZeros<T><<< numBlocks, THREADS_PER_BLOCK >>>( mem.d_mem, INT_MAX );

		//sort nz elements with same row indice by column
		d_sortColumnIndices<T><<< numBlocks, THREADS_PER_BLOCK >>>( mem.d_mem );

	}

}

template void hd_fillDiagonals( DeviceMemory<float>& mem );
template void hd_fillDiagonals( DeviceMemory<double>& mem );
