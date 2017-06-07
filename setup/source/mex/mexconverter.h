#ifndef MEXCONVERTER_H_
#define MEXCONVERTER_H_


#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\includes.h"
#include "..\\interface\\neutralmatrix\\neutralmatrix.h"
#else
#include "../includes.h"
#include "../interface/neutralmatrix/neutralmatrix.h"
#endif

#ifdef _MATLAB
#ifdef MEXINCLUDE
#include MEXINCLUDE
#endif

//defines  conversionmodes as bitvalues
//multiple modes can be combined into one using the | operator
//when selecting opposing flags (e.g. ShareData and CopyData) the behaviour is undefined
enum ConversionMode {
	FailOnTypeMismatch = 1,
	ConvertTypesOnMismatch = 2,
	ShareData = 4,
	CopyData = 8,
	Default = ( FailOnTypeMismatch | ShareData )
};
inline ConversionMode operator|( ConversionMode a, ConversionMode b ) {
	return static_cast<ConversionMode>(static_cast<int>( a ) | static_cast<int>( b ));
}

//defines results of a conversion as bitvalues
//multiple results can be combined into one using the | operator
enum ConversionResult {
	ErrorNotSupported = ( 1 << 17 ),
	ErrorTypeMismatch = ( 1 << 16 ),
	WarningDataNotShared = ( 1 << 4 ),
	WarningDecimalMismatch = ( 1 << 3 ),
	Reserved = (1 << 2),
	WarningSignedMismatch = ( 1 << 1 ),
	WarningOutputSmallerThanInput = ( 1 << 0 ),
	Success = 0
};
inline ConversionResult operator|( ConversionResult a, ConversionResult b ) {
	return static_cast<ConversionResult>(static_cast<int>( a ) | static_cast<int>( b ));
}

//used to convert scalars and matrices from matlab
//into a interface format (NeutralMatrix<T>)
template<class T>
class MexConverter {
public:

	//c++ -> matlab
	//converts a value (T) into a 1x1 matrix mxArray*
	static ConversionResult mxFromScalar( T scalar, mxArray*& dest, mxClassID outputType = mxClassID::mxUNKNOWN_CLASS, ConversionMode mode = ConversionMode::Default );

	//c++ <- matlab
	//reads a scalar from a 1x1 matrix mxArray*
	//if mx points to a mxn matrix only the first element is converted
	static ConversionResult scalarFromMx( mxArray* mx, T& dest, ConversionMode mode = ConversionMode::Default );

	//c++ <- matlab
	//creates an interface matrix (NeutralMatrix<T>) from a matlab matrix (mxArray*)
	static ConversionResult neutralFromMx( mxArray* mx, NeutralMatrix<T>& dest, ConversionMode mode = ConversionMode::Default );
	//c++ <- matlab
	//creates an interface matrix (NeutralMatrix<T>) from a matlab matrix (mxArray*)
	static ConversionResult neutralFromMx( mxArray* mx, NeutralMatrix<T, int, RowMajor<int>>& dest, ConversionMode mode = ConversionMode::Default );

	//c++ -> matlab
	//creates a mxn matlab matrix (mxArray*) from an interface matrix (Neutralmatrix<T>)
	static ConversionResult mxFromNeutral( NeutralMatrix<T>& mat, mxArray*& dest, mxClassID outputType = mxClassID::mxUNKNOWN_CLASS, ConversionMode mode = ConversionMode::Default );
	//c++ -> matlab
	//creates a mxn matlab matrix (mxArray*) from an interface matrix (Neutralmatrix<T>)
	static ConversionResult mxFromNeutral( NeutralMatrix<T, int, RowMajor<int>>& mat, mxArray*& dest, mxClassID outputType = mxClassID::mxUNKNOWN_CLASS, ConversionMode mode = ConversionMode::Default );

	//c++ -> matlab
	//creates a 0x0 matlab matrix
	static mxArray* mxFromEmpty();
private:
	//returns the mxClassID according to the class template
	static mxClassID getMatlabClassIdFromType();
};


#endif /* _MATLAB */
#endif /* MEXCONVERTER_H_ */
