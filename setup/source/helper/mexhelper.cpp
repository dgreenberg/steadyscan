#include "mexhelper.h"

#ifdef _MATLAB

// Returns the variable name of var as a char string mxArray
// If it can't be found, returns NULL
char* MxGetName( const mxArray *var ) {
	mxArray *who, *mx, *cell;
	int i, n;
	char* name = nullptr;

	if( var ) {
		mexCallMATLAB( 1, &who, 0, NULL, "who" );
		if( who ) {
			n = mxGetNumberOfElements( who );
			for( i = 0; i < n; i++ ) {
				cell = mxGetCell( who, i );
				name = mxArrayToString( cell );
				mx = const_cast<mxArray*>( mexGetVariablePtr( "caller", name ) );
				if( !mx ) mx = const_cast<mxArray*>( mexGetVariablePtr( "global", name ) );
				if( mx == var ) break;
			}
			mxDestroyArray( who );
		}
	}

	return name;
}
#endif
