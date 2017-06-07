#ifndef MEXGATEWAY_H_
#define MEXGATEWAY_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\includes.h"
#else
#include "../includes.h"
#endif

#ifdef _MATLAB

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\interface\\memory.h"
#include "..\\interface\\neutralmatrix\\neutralmatrix.h"
#include "..\\task\\task.h"
#include "..\\helper\\mexhelper.h"
#include "..\\helper\\timer\\realtimer.h"
#include "mexsimulator\\mexsimulator.h"
#include "..\\interface\\stopcriterion.h"
#include "conversionvalidator\\allconversionvalidators.h"
#else
#include "../interface/memory.h"
#include "../interface/neutralmatrix/neutralmatrix.h"
#include "../task/task.h"
#include "../helper/mexhelper.h"
#include "../helper/timer/realtimer.h"
#include "mexsimulator/mexsimulator.h"
#include "../interface/stopcriterion.h"
#include "conversionvalidator/allconversionvalidators.h"
#endif
#include "configuration.h"
#include "mexconverter.h"

#ifdef MEXINCLUDE
#include MEXINCLUDE
#endif

//takes data from MATLAB
//converts it into the neutral format
//and passes it to a Task instance
template<class T>
class MexGateway {
public:
	MexGateway( Configuration* configuration );

	void execute( int nlhs, mxArray* plhs[], int nrhs, mxArray* prhs[] );
private:
	bool preparePersistentMemory( mxArray* prhs[], int nrhs );
	bool convertOutputArguments( TaskOutput<T>& src, mxArray* plhs[], int nlhs );
private:
	Configuration* configuration;
	IConversionValidator* conversionValidator;
	PreAllocatedMemory<T> rom;
	Task<T> task;
};



#endif /* _MATLAB */
#endif /* MEXGATEWAY_H_ */
