#ifndef IFILTER_H_
#define IFILTER_H_

template<class T>
class IFilter {
public:
	virtual ~IFilter() {}

    virtual void apply( T* values, int nValues ) = 0;
    virtual void applyGPU( T* d_values, int nValues ) = 0;
};


#endif
