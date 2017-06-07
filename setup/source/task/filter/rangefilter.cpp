#include "rangefilter.h"
#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\cudaincludes.h"
#else
#include "../../cudaincludes.h"
#endif

template<class T>
RangeFilter<T>::RangeFilter( T strength ) {
    this->strength = strength;
    this->hostBuffer = nullptr;
    this->hostBufferSize = 0;
}

template<class T>
RangeFilter<T>::~RangeFilter() {
	if( nullptr != this->hostBuffer ) {
		delete[] this->hostBuffer;
		this->hostBuffer = nullptr;
		this->hostBufferSize = 0;
	}
}

template<class T>
void RangeFilter<T>::apply( T* values, int nValues ) {
    for(int i = 1; i < nValues - 1; i++) {
    	T tc = ( values[i - 1] + values[i + 1] ) / 2;
    	T diff = ( tc - values[i - 1] ) * this->strength;

    	if( values[i - 1] < values[i + 1] ) {
    		if(values[i] > values[i + 1] + diff || values[i] < values[i - 1] - diff) {
        		values[i] = tc;
    		}
    	} else {
    		if(values[i] < values[i + 1] + diff || values[i] > values[i - 1] - diff) {
        		values[i] = tc;
    		}
    	}
    }
}

template<class T>
void RangeFilter<T>::applyGPU( T* d_values, int nValues ) {
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

	this->apply( this->hostBuffer, nValues );


	//copy filtered data back to gpu
	cudaMemcpy( d_values, this->hostBuffer, nValues * sizeof( T ), cudaMemcpyHostToDevice );
}

template class RangeFilter<float>;
template class RangeFilter<double>;
