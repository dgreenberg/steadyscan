#include "eigensolver.cuh"

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\cudaincludes.h"
#include "..\\..\\helper\\cudahelper\\cudahelper.cuh"
#include "..\\..\\helper\\commonhelper.h"
#include "eigen\\eigensolve.h"
#include "..\\filter\\allfilter.h"
#else
#include "../../cudaincludes.h"
#include "../../helper/cudahelper/cudahelper.cuh"
#include "../../helper/commonhelper.h"
#include "eigen/eigensolve.h"
#include "../filter/allfilter.h"
#endif

template<class T>
SolverEigen<T>::SolverEigen( IFilter<T>* filter ) : ISolver<T>() {
	if( nullptr == filter ) filter = NoFilter<T>::Static();
	this->filter = filter;
}

template<class T>
SolverEigen<T>::~SolverEigen() {
	if( nullptr != this->filter && this->filter != NoFilter<T>::Static() ) {
		delete this->filter;
	}
}

template<class T>
IFilter<T>* SolverEigen<T>::setFilter( IFilter<T>* filter ) {
	if( nullptr == filter ) return filter;
	IFilter<T>* retv = this->filter;
	this->filter = filter;
	return retv;
}

template<class T>
void SolverEigen<T>::solve( DeviceMemory<T>& mem ) {
	DeviceMemory<T> d_mem;
	CUCHECKRC( cudaMemcpy( &d_mem, mem.d_mem, sizeof( DeviceMemory<T> ), cudaMemcpyDeviceToHost ) );

	CUCHECKRC( cudaMemcpy( mem.h_slvValues, mem.H.values, sizeof( T ) * d_mem.nParams * 6, cudaMemcpyDeviceToHost ) );
	CUCHECKRC( cudaMemcpy( mem.h_slvRowInd, mem.H.rowInd, sizeof( int ) * d_mem.nParams * 6, cudaMemcpyDeviceToHost ) );
	CUCHECKRC( cudaMemcpy( mem.h_slvColInd, mem.H.colInd, sizeof( int ) * d_mem.nParams * 6, cudaMemcpyDeviceToHost ) );
	CUCHECKRC( cudaMemcpy( mem.h_slvJTr, mem.jtr, sizeof( T ) * d_mem.nParamsUsed, cudaMemcpyDeviceToHost ) );

	int nnz = h_EigenSolve( mem.h_slvValues, mem.h_slvRowInd, mem.h_slvColInd, mem.h_slvJTr, d_mem.nParamsUsed, d_mem.nParams * 6, mem.h_slvNewDeltaP );

	//set nnz and copy to device
	d_mem.H.nnz = nnz;
	CUCHECKRC( cudaMemcpy( mem.d_mem, &d_mem, sizeof( DeviceMemory<T> ), cudaMemcpyHostToDevice ) );

	this->filter->apply( mem.h_slvNewDeltaP, d_mem.nParamsUsed / 2 );
	this->filter->apply( &mem.h_slvNewDeltaP[d_mem.nParamsUsed / 2], d_mem.nParamsUsed / 2 );

	CUCHECKRC( cudaMemcpy( mem.newDeltaP, mem.h_slvNewDeltaP, sizeof(T) * d_mem.nParamsUsed, cudaMemcpyHostToDevice ) );
}

template class SolverEigen<float>;
template class SolverEigen<double>;
