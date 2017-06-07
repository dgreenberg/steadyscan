#include "configuration.h"

#ifdef MEXINCLUDE
#include MEXINCLUDE
#endif

static const char* floatPrecisionNames[] = {
	"None",
	"Single",
	"Double"
};
const char* FloatPrecision_toString( FloatPrecision precision ) {
	return ::floatPrecisionNames[static_cast<int>(precision)];
}

FloatPrecision Configuration::precisionType() {
	return this->precision;
}

bool Configuration::subFunctionTimingEnabled() {
	return this->timeSubfunctions;
}
bool Configuration::executionTimingEnabled() {
	return this->timeExecution;
}

bool Configuration::parameterInformationPrintEnabled() {
	return this->showParameterInformation;
}

bool Configuration::saveIterationCorrelationsEnabled() {
	return this->saveIterationCorrelations;
}
bool Configuration::saveFrameCorrelationsEnabled() {
	return this->saveFrameCorrelations;
}
bool Configuration::saveArgumentsToFileEnabled() {
	return this->saveArgumentsToFile;
}

bool Configuration::showConversionWarningsEnabled() {
	return this->showConversionWarnings;
}

bool Configuration::showGpuMemoryUsageEnabled() {
	return this->showGpuMemoryUsage;
}

bool Configuration::showStopCriterionEnabled() {
	return this->showStopCriterion;
}

bool Configuration::maskInactiveGroupsEnabled() {
	return this->maskInactiveGroups;
}

bool Configuration::medianFilterAfterSolveEnabled() {
	return this->medianFilterAfterSolve;
}

bool Configuration::medianFilterBeforeReturnEnabled() {
	return this->medianFilterBeforeReturn;
}
bool Configuration::copyRawParametersEnabled() {
	return this->copyRawParameters;
}
bool Configuration::notifyOnFilterDisabledEnabled() {
	return this->notifyOnFilterDisabled;
}
bool Configuration::treatWarningsAsErrorsEnabled() {
	return this->treatWarningsAsErrors;
}
bool Configuration::versionPrintEnabled() {
	return this->versionPrint;
}
bool Configuration::frameOrderedParametersEnabled() {
	return this->frameOrderedParameters;
}
bool Configuration::saveIterationErrvalsEnabled() {
	return this->saveIterationErrvals;
}

void ConfigurationParser::parseFlags( const char* flags, Configuration* dest ) {
	memset( dest, 0, sizeof( Configuration ) );

	for( int i = 0; flags[i] != '\0'; i++ ) {
		switch( flags[i] ) {
		case 'D':
			dest->precision = FloatPrecision::Double;
			break;
		case 'S':
			dest->precision = FloatPrecision::Single;
			break;
		case 'T':
			dest->timeExecution = true;
			break;
		case 't':
			dest->timeSubfunctions = true;
			break;
		case 'I':
			dest->showParameterInformation = true;
			break;
		case 'C':
			dest->saveIterationCorrelations = true;
			break;
		case 'c':
			dest->saveFrameCorrelations = true;
			break;
		case 'F':
			dest->saveArgumentsToFile = true;
			break;
		case 'W':
			dest->showConversionWarnings = true;
			break;
		case 'w':
			dest->treatWarningsAsErrors = true;
			break;
		case 'M':
			dest->maskInactiveGroups = true;
			break;
		case 'G':
			dest->showGpuMemoryUsage = true;
			break;
		case 'E':
			dest->showStopCriterion = true;
			break;
		case 'R':
			dest->medianFilterAfterSolve = true;
			break;
		case 'r':
			dest->medianFilterBeforeReturn = true;
			break;
		case 'P':
			dest->copyRawParameters = true;
			break;
		case 'f':
			dest->notifyOnFilterDisabled = true;
			break;
		case 'V':
			dest->versionPrint = true;
			break;
		case 'p':
			dest->frameOrderedParameters = true;
			break;
		case 'e':
			dest->saveIterationErrvals = true;
			break;
		default:
			break;
		}
	}
}
