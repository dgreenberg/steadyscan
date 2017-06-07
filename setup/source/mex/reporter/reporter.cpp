#include "reporter.h"

const char* ReportID_ToString( ReportID id ) {
	static const char* ReportIdentifiers[] = {
		"GPUMC:Unknown",
		"GPUMC:WarningIsError",
		"GPUMC:NotEnoughInputs",
		"GPUMC:InvalidArgument",
		"GPUMC:CudaWarning",
		"GPUMC:CudaError",
		"GPUMC:CuSparseSolver"
	};

	return ReportIdentifiers[id];
}

static bool treatWarningsAsErrors = false;
void Reporter::treatWarningsAsErrors( bool treatAsErrors ) {
	::treatWarningsAsErrors = treatAsErrors;
}

#ifdef MEXINCLUDE
#include MEXINCLUDE

void Reporter::error( ReportID ident, const char* msg ) {
	mexErrMsgIdAndTxt( ReportID_ToString( ident ), msg );
}

void Reporter::warn( ReportID ident, const char* msg ) {
	mexWarnMsgIdAndTxt( ReportID_ToString( ident ), msg );
	if( ::treatWarningsAsErrors ) Reporter::error( ReportID::WarningIsError, "Warning -> Error" );
}

void Reporter::inform( const char* msg ) {
	mexPrintf( "%s", msg );
	mexEvalString( "drawnow;" );
}
#else
void Reporter::error(ReportID ident, const char* msg) {
	std::cout << "---ERROR---" << std::endl;
	std::cout << "(" << ReportID_ToString(ident) << ")" << std::endl;
	std::cout << msg << std::endl;
}

void Reporter::warn(ReportID ident, const char* msg) {
	std::cout << "---Warning---" << std::endl;
	std::cout << "(" << ReportID_ToString(ident) << ")" << std::endl;
	std::cout << msg << std::endl;
}

void Reporter::inform(const char* msg) {
	std::cout << msg << std::endl;
}

#endif
