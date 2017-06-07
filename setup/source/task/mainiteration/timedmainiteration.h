#ifndef TIMEDMAINITERATION_H_
#define TIMEDMAINITERATION_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\solver\\isolver.cuh"
#include "..\\..\\interface\\memory.h"
#include "..\\..\\helper\\timer\\realtimer.h"
#include "..\\..\\mex\\reporter\\reporter.h"
#else
#include "../solver/isolver.cuh"
#include "../../interface/memory.h"
#include "../../helper/timer/realtimer.h"
#include "../../mex/reporter/reporter.h"
#endif

#include "mainiteration.h"

template<class T>
class TimedMainIteration : public MainIteration<T> {
public:
	TimedMainIteration( ISolver<T>* solver, IFilter<T>* filter2, Configuration* config );
	virtual ~TimedMainIteration();

	virtual bool execute( DeviceMemory<T>& mem, int iteration );

	virtual void copyOutput( TaskOutput<T>& out );

	static const int nTimedFunctions = 10;
private:
	RealTimer<T> timer;

	T* functionTimes;

};

#endif
