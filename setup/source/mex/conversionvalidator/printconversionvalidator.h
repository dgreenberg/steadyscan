#ifndef PRINTCONVERSIONVALIDATOR_H_
#define PRINTCONVERSIONVALIDATOR_H_

#include "iconversionvalidator.h"

#ifdef _MATLAB

//validates conversionresult
//prints statements to std::out when result indicates
//- warnings
//- errors
//returns:
//- true: no error(s)
//- false: error(s)
class PrintConversionValidator : public IConversionValidator {
public:
	PrintConversionValidator();
	virtual ~PrintConversionValidator();

	virtual void validate( ConversionResult result, mxArray* mx, int index, const char* varName = nullptr );
};

#endif

#endif
