#ifndef CUSPARSESOLVER_CUH_
#define CUSPARSESOLVER_CUH_

#include "isolver.cuh"

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\interface\\memory.h"
#include "..\\filter\\ifilter.h"
#else
#include "../../interface/memory.h"
#include "../filter/ifilter.h"
#endif

template<class T>
class SolverCuSparse : public ISolver<T> {
public:
  SolverCuSparse( IFilter<T>* filter = nullptr );
  virtual ~SolverCuSparse();

  virtual void solve( DeviceMemory<T>& mem );

  virtual IFilter<T>* setFilter( IFilter<T>* filter );

private:
  IFilter<T>* filter;
};

#endif
