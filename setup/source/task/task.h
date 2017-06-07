#ifndef TASK_CUH_
#define TASK_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\includes.h"
#include "..\\cudaincludes.h"
#include "..\\interface\\memory.h"
#include "..\\interface\\taskparameters.h"
#include "..\\interface\\neutralmatrix\\neutralmatrix.h"
#include "..\\helper\\eps.h"
#include "..\\mex\\configuration.h"
#include "..\\helper\\timer\\realtimer.h"
#include "mainiteration\\allmainiterations.h"
#include "..\\interface\\stopcriterion.h"
#include "filter\\allfilter.h"
#else
#include "../includes.h"
#include "../cudaincludes.h"
#include "../interface/memory.h"
#include "../interface/taskparameters.h"
#include "../interface/neutralmatrix/neutralmatrix.h"
#include "../helper/eps.h"
#include "../mex/configuration.h"
#include "../helper/timer/realtimer.h"
#include "mainiteration/allmainiterations.h"
#include "../interface/stopcriterion.h"
#include "filter/allfilter.h"
#endif

template<class T>
class Task {
public:
	Task( Configuration* configuration );
	~Task();

	//allocates memory used during execute
	bool initialize( PreAllocatedMemory<T>& rom );

	//allocates memory for the output matrices
	void allocateOutput( TaskOutput<T>& output );

	//executes frame alignment
	StopCriterion alignFrames( TaskOutput<T>& output );

	//releases allocated memory
	//that was allocated during initialize
	void freePreAllocatedMemory( DeviceMemory<T>* mem );

	//copies the parameters to the output struct
	//after executing an endFilter
	void getParameters( TaskOutput<T>& output );
private:
	//contains ptr to device matrices etc.
	DeviceMemory<T> mem;

	Configuration* configuration;

	MainIteration<T>* mainIteration;

	IFilter<T>* endFilter;
};

#endif /* TASK_H_ */
