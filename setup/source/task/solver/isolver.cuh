#ifndef ISOLVER_CUH_
#define ISOLVER_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\filter\\ifilter.h"
#include "..\\..\\interface\\memory.h"
#else
#include "../filter/ifilter.h"
#include "../../interface/memory.h"
#endif

template<class T>
class ISolver {
public:
	virtual ~ISolver() {}

	virtual void solve( DeviceMemory<T>& mem ) = 0;
	virtual IFilter<T>* setFilter( IFilter<T>* filter ) = 0;
};

#endif
