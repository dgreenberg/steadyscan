#ifndef ICONVERSIONVALIDATOR_H_
#define ICONVERSIONVALIDATOR_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\includes.h"
#else
#include "../../includes.h"
#endif

#ifdef _MATLAB

#include "../mexconverter.h"

class IConversionValidator {
public:
	virtual ~IConversionValidator() {}

	virtual void validate( ConversionResult result, mxArray* mx, int index, const char* varName = nullptr ) = 0;
};

#endif

#endif
