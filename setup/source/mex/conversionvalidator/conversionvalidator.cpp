#include "conversionvalidator.h"

#ifdef _MATLAB

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\reporter\\reporter.h"
#else
#include "../reporter/reporter.h"
#endif

ConversionValidator::ConversionValidator( bool treatWarningsAsErrors ) {
	this->treatWarningsAsErrors = treatWarningsAsErrors;
}

ConversionValidator::~ConversionValidator() {
	return;
}

void ConversionValidator::validate( ConversionResult result, mxArray* mx, int index, const char* varName ) {
	//first error value in result is (1<<16)
	//so every value below (1<<16) can only contain warnings
	if( this->treatWarningsAsErrors ) {
		if( result == ConversionResult::Success ) return;
		Reporter::error( ReportID::InvalidArgument, nullptr );
	}
}

#endif
