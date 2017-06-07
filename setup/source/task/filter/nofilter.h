#ifndef NOFILTER_H_
#define NOFILTER_H_

#include "ifilter.h"

template<class T>
class NoFilter : public IFilter<T> {
private: //force the use of NoFilter::Static()
	NoFilter();
public:
	virtual ~NoFilter();

	static NoFilter<T>* Static();

    virtual void apply( T* data, int nValues );
    virtual void applyGPU( T* d_data, int nValues );
};

#endif
