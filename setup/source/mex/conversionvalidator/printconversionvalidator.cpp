#include "printconversionvalidator.h"

#ifdef _MATLAB

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\helper\\mexhelper.h"
#include "..\\reporter\\reporter.h"
#else
#include "../../helper/mexhelper.h"
#include "../reporter/reporter.h"
#endif

PrintConversionValidator::PrintConversionValidator() {
}
PrintConversionValidator::~PrintConversionValidator() {
}

void PrintConversionValidator::validate( ConversionResult result, mxArray* mx, int index, const char* varName ) {
	bool nameIsAllocated = true;
	const char* name;
	char buff[128];

	if( result == ConversionResult::Success ) return;
	index++;

	if( nullptr == varName ) {
		name = MxGetName( mx );
		if( nullptr == name ) {
			name = "unknown";
			nameIsAllocated = false;
		}
	} else {
		name = varName;
		nameIsAllocated = false;
	}
	
	//process warnings
	if( result & ConversionResult::WarningOutputSmallerThanInput ) {
		sprintf(buff, "Argument %s (%i)::requested type is smaller than input type ( overflow may occur )", name, index);
		Reporter::warn( ReportID::InvalidArgument, buff );
	}
	if( result & ConversionResult::WarningSignedMismatch ) {
		sprintf( buff, "Argument %s (%i)::unsigned/signed mismatch", name, index );
		Reporter::warn( ReportID::InvalidArgument, buff );
	}
	if( result & ConversionResult::WarningDecimalMismatch ) {
		if( mxIsDouble( mx ) || mxIsSingle( mx ) ) {
			sprintf( buff, "Argument %s (%i)::fixpoint type requested, floatingpoint given( possible rounding errors )", name, index );
			Reporter::warn( ReportID::InvalidArgument, buff );
		} else {
			sprintf( buff, "Argument %s (%i)::floatingpoint type requested, fixpoint given ( possible rounding errors )", name, index );
			Reporter::warn( ReportID::InvalidArgument, buff );
		}
	}
	if( ( result & ConversionResult::WarningDataNotShared ) && !( result & ConversionResult::ErrorTypeMismatch ) ) {
		sprintf( buff, "Argument %s (%i)::unable to share data between matrices of different value types", name, index );
		Reporter::warn( ReportID::InvalidArgument, buff );
	}

	//process errors
	if( result & ConversionResult::ErrorNotSupported ) {
		sprintf( buff, "Argument %s (%i)::passed type is not supported by the converter", name, index );
		Reporter::error( ReportID::InvalidArgument, buff );
	}
	if( result & ConversionResult::ErrorTypeMismatch ) {
		sprintf( buff, "Argument %s (%i)::passed type does not match requested type", name, index );
		Reporter::error( ReportID::InvalidArgument, buff );
	}

	if( nameIsAllocated ) mxFree( const_cast<char*>( name ) );
}

#endif
