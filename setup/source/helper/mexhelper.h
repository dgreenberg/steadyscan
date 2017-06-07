#ifndef MEXHELPER_H_
#define MEXHELPER_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\includes.h"
#else
#include "../includes.h"
#endif

#ifdef _MATLAB

#include MEXINCLUDE

// Returns the variable name of var as a char string mxArray
// If it can't be found, returns NULL
char* MxGetName( const mxArray *var );

#endif /* _MATLAB */
#endif /* MEXHELPER_H_ */
