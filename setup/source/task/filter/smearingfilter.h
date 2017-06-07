#ifndef SMEARINGFILTER_H_
#define SMEARINGFILTER_H_

#include "ifilter.h"

template<class T>
class SmearingFilter : public IFilter<T> {
public:
    //strength 0..1
    SmearingFilter( T strength );
    virtual ~SmearingFilter();

    virtual void apply( T* values, int nValues );
    virtual void applyGPU( T* d_values, int nValues );

private:
    T strength;

    T* hostBuffer;
    unsigned int hostBufferSize;
};

#endif
