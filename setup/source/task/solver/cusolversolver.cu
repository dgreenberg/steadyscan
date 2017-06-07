#include "cusolversolver.cuh"

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\cudaincludes.h"
#include "..\\..\\helper\\cudahelper\\cudahelper.cuh"
#include "..\\..\\helper\\cudahelper\\cusolvertemplates.cuh"
#include "..\\filter\\allfilter.h"
#else
#include "../../cudaincludes.h"
#include "../../helper/cudahelper/cudahelper.cuh"
#include "../../helper/cudahelper/cusolvertemplates.cuh"
#include "../filter/allfilter.h"
#endif

template<class T>
SolverCuSolver<T>::SolverCuSolver( IFilter<T>* filter ) : ISolver<T>() {
	if( filter == nullptr ) filter = NoFilter<T>::Static();
	this->filter = filter;
}

template<class T>
SolverCuSolver<T>::~SolverCuSolver() {
	if( nullptr == this->filter ) return;
	if( this->filter == NoFilter<T>::Static() ) return;

	delete this->filter;
}

template<class T>
IFilter<T>* SolverCuSolver<T>::setFilter( IFilter<T>* filter ) {
	if( nullptr == filter ) return filter;
	IFilter<T>* retv = this->filter;
	this->filter = filter;
	return retv;
}

template<class T>
void SolverCuSolver<T>::solve( DeviceMemory<T>& mem ) {
	DeviceMemory<T> d_mem;

	CUCHECKRC( cudaMemcpy( &d_mem, mem.d_mem, sizeof( DeviceMemory<T> ), cudaMemcpyDeviceToHost ) );

	int nnz = d_mem.H.nnz;
	int m = d_mem.nParamsUsed;

	//convert coo to csr
	int* d_csrRowInd = mem.H.csrRowInd;
	int* d_csrColInd = mem.H.colInd;
	T* d_csrVal = mem.H.values;
	T* d_x = mem.jtr;
	T* d_y = mem.newDeltaP;

	CUCHECKRC( cusparseXcoo2csr( mem.cuSparseHandle, mem.H.rowInd, nnz, m, d_csrRowInd, CUSPARSE_INDEX_BASE_ZERO ) );

	int singular;
	CUCHECKRC( cusolverSpTcsrlsvchol<T>( mem.cuSolverHandle, m, nnz, mem.cuSparseDescription, d_csrVal, d_csrRowInd, d_csrColInd, d_x, 1e-6, 1, d_y, &singular ) );

	this->filter->applyGPU( mem.newDeltaP, d_mem.nParamsUsed / 2 );
	this->filter->applyGPU( &mem.newDeltaP[d_mem.nParamsUsed / 2], d_mem.nParamsUsed / 2 );
}

template class SolverCuSolver<float>;
template class SolverCuSolver<double>;
