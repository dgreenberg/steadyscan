#ifndef MAINITERATION_H_
#define MAINITERATION_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\includes.h"
#include "..\\solver\\isolver.cuh"
#include "..\\..\\interface\\memory.h"
#include "..\\..\\interface\\taskparameters.h"
#include "..\\filter\\allfilter.h"
#include "..\\..\\mex\\configuration.h"
#include "..\\..\\mex\\reporter\\reporter.h"
#else
#include "../../includes.h"
#include "../solver/isolver.cuh"
#include "../../interface/memory.h"
#include "../../interface/taskparameters.h"
#include "../filter/allfilter.h"
#include "../../mex/configuration.h"
#include "../../mex/reporter/reporter.h"
#endif

//used for mexPrintf to print iteration when filter is disabled
#ifdef MEXINCLUDE
#include MEXINCLUDE
#endif

template<class T>
class MainIteration {
public:
	MainIteration( ISolver<T>* solver, IFilter<T>* filter2, Configuration* config );
	virtual ~MainIteration();

	virtual bool execute( DeviceMemory<T>& mem, int iteration );
	virtual void copyOutput( TaskOutput<T>& out );

	void disableFilter();

protected:
	Configuration* configuration;

	ISolver<T>* solver;
	IFilter<T>* filter;
};

#endif
