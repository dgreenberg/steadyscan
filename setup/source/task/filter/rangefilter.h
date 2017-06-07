#ifndef RANGEFILTER_H_
#define RANGEFILTER_H_

#include "ifilter.h"

template<class T>
class RangeFilter : public IFilter<T> {
public:
    //strength max(T) ... -1
	//dfl: 1
	RangeFilter( T strength );
    virtual ~RangeFilter();

    virtual void apply( T* values, int nValues );
    virtual void applyGPU( T* d_values, int nValues );

private:
    T strength;
    T* hostBuffer;
    int hostBufferSize;
};

#endif
