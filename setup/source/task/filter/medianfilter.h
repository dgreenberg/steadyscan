#ifndef MEDIANFILTER_H_
#define MEDIANFILTER_H_

#include "ifilter.h"

template<class T>
class MedianFilter : public IFilter<T> {
public:
	MedianFilter();
	virtual ~MedianFilter();

	virtual void apply( T* values, int nValues );
	virtual void applyGPU( T* d_values, int nValues );
};

#endif
