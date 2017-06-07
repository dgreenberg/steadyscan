#include "nofilter.h"

template<class T>
NoFilter<T>::NoFilter() {
	return;
}

template<class T>
NoFilter<T>::~NoFilter() {
	return;
}

template<class T>
NoFilter<T>* NoFilter<T>::Static() {
	static NoFilter<T> dfl;
	return &dfl;
}

template<class T>
void NoFilter<T>::apply( T* data, int nValues ) {
	return;
}

template<class T>
void NoFilter<T>::applyGPU( T* d_data, int nValues ) {
	return;
}

template class NoFilter<float>;
template class NoFilter<double>;
