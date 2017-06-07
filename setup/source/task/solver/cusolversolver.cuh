#ifndef CUSOLVERSOLVER_CUH_
#define CUSOLVERSOLVER_CUH_

#include "isolver.cuh"

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\interface\\memory.h"
#include "..\\filter\\ifilter.h"
#else
#include "../../interface/memory.h"
#include "../filter/ifilter.h"
#endif

template<class T>
class SolverCuSolver : public ISolver<T> {
public:
  SolverCuSolver( IFilter<T>* filter = nullptr);
  virtual ~SolverCuSolver();

  virtual void solve( DeviceMemory<T>& d_Ptr );

  virtual IFilter<T>* setFilter( IFilter<T>* filter );
private:
  IFilter<T>* filter;
};

#endif
