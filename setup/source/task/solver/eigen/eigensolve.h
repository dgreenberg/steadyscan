#ifndef EIGENSOLVE_H_
#define EIGENSOLVE_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\..\\includes.h"
#else
#include "../../../includes.h"
#endif

template<typename T>
int h_EigenSolve( T* values, int* rowInd, int* colInd, T* jtr, int nParamsUsed, int nValues, T* result );

#endif
