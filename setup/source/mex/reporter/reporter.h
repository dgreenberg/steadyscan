#ifndef REPORTER_H_
#define REPORTER_H_

#if defined(_WIN32 ) || defined( _WIN64 )
#include "..\\..\\includes.h"
#else
#include "../../includes.h"
#endif

enum ReportID {
	Unknown,
	WarningIsError,
	NotEnoughInputs,
	InvalidArgument,
	CudaWarning,
	CudaError,
	CuSparseSolver
};
const char* ReportID_ToString(ReportID id);

//reports back to matlab
class Reporter {
public:
	//if set to true, warn() calls error()
	static void treatWarningsAsErrors(bool treatAsErrors);

	//function does not return
	static void error(ReportID identifier, const char* msg);

	//prints warning message in matlab
	static void warn(ReportID identifier, const char* msg);

	//outputs text
	static void inform(const char* msg);
};

#endif
