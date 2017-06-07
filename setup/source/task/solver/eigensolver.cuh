#ifndef EIGENSOLVER_CUH_
#define EIGENSOLVER_CUH_

#include "isolver.cuh"

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\interface\\memory.h"
#include "..\\filter\\ifilter.h"
#else
#include "../../interface/memory.h"
#include "../filter/ifilter.h"
#endif

template<class T>
class SolverEigen : public ISolver<T> {
public:
  SolverEigen( IFilter<T>* filter = nullptr );
  virtual ~SolverEigen();

  virtual void solve( DeviceMemory<T>& d_Ptr );

  virtual IFilter<T>* setFilter( IFilter<T>* filter );
private:
  IFilter<T>* filter;
};

#endif
