#include "smearingfilter.h"
#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\cudaincludes.h"
#else
#include "../../cudaincludes.h"
#endif

template<class T>
SmearingFilter<T>::SmearingFilter( T strength ) {
	this->strength = strength;
	this->hostBuffer = nullptr;
	this->hostBufferSize = 0;
}

template<class T>
SmearingFilter<T>::~SmearingFilter() {
	if( nullptr != this->hostBuffer ) {
		delete[] this->hostBuffer;
		this->hostBuffer = nullptr;
		this->hostBufferSize = 0;
	}
}

template<class T>
void SmearingFilter<T>::apply( T* values, int nValues ) {
	for( int i = 0; i < nValues; i++ ) {
		T p;
		if( i == 0 ) p = 0;
		else p = values[i - 1];

		values[i] = p * this->strength + values[i] * ( 1 - this->strength );
	}
}

template<class T>
void SmearingFilter<T>::applyGPU( T* d_values, int nValues ) {
	//allocate new host buffer if the current buffer can store
	//less elements than nValues
	if( nValues > this->hostBufferSize ) {
		delete[] this->hostBuffer;
		this->hostBuffer = new T[nValues];

		//skip filtering if allocation failed
		if( nullptr == this->hostBuffer ) {
			this->hostBufferSize = 0;
			return;
		}

		this->hostBufferSize = nValues;
	}

	//copy data from gpu to hostBuffer
	cudaMemcpy( this->hostBuffer, d_values, nValues * sizeof( T ), cudaMemcpyDeviceToHost );

	//filter data in hostBuffer
	this->apply( this->hostBuffer, nValues );

	//copy filtered data back to gpu
	cudaMemcpy( d_values, this->hostBuffer, nValues * sizeof( T ), cudaMemcpyHostToDevice );
}

template class SmearingFilter<float>;
template class SmearingFilter<double>;
