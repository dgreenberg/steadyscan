#ifndef CONVERSIONVALIDATOR_H_
#define CONVERSIONVALIDATOR_H_

#include "iconversionvalidator.h"

#ifdef _MATLAB

//validates ConversionResult
//returnvalues:
//- true: no error(s)
//- false: error(s)
class ConversionValidator : public IConversionValidator {
public:
	ConversionValidator( bool treatWarningsAsErrors );
	virtual ~ConversionValidator();

	virtual void validate( ConversionResult result, mxArray* mx, int index, const char* varName = nullptr );

protected:
	bool treatWarningsAsErrors;
};

#endif

#endif
