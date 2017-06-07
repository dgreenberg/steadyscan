#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\includes.h"
#else
#include "../includes.h"
#endif

//S: single precision
//D: double precision
//T: time execution
//t: time iterations / subfunctions
//M: mask inactive groups
//I: show parameter information
//C: save correlations after each iteration
//c: save all frame correlations
//F: save call arguments to file (./VERSION_rom.bin)
//W: show gateway conversion warnings/errors
//w: treat gateway warnings as errors
//G: show gpu memory usage
//E: show stop criterion
//R: enable 3-median filtering of parameters before linesearch
//r: enable 3-median filtering of the resulting parameter values
//P: output unformatted parameters
//f: notify when filter stage is deactivated
//V: print version
//p: use frame ordered Parameters (x1y1x2y2x3y3... instead of x1x2x3... and y1y2y3...)

enum FloatPrecision {
	None,
	Single,
	Double
};
const char* FloatPrecision_toString( FloatPrecision precision );

class Configuration;

class ConfigurationParser {
public:
    static void parseFlags( const char* flags, Configuration* dest );
};

class Configuration {
	friend class ConfigurationParser;

public:
	FloatPrecision precisionType();

	bool subFunctionTimingEnabled();
	bool executionTimingEnabled();

	bool maskInactiveGroupsEnabled();

	bool parameterInformationPrintEnabled();

	bool saveIterationCorrelationsEnabled();
	bool saveFrameCorrelationsEnabled();
	bool saveArgumentsToFileEnabled();

	bool showConversionWarningsEnabled();
	bool showGpuMemoryUsageEnabled();
	bool showStopCriterionEnabled();

	bool medianFilterAfterSolveEnabled();
	bool medianFilterBeforeReturnEnabled();

	bool copyRawParametersEnabled();

	bool notifyOnFilterDisabledEnabled();
	bool treatWarningsAsErrorsEnabled();

	bool frameOrderedParametersEnabled();
	bool saveIterationErrvalsEnabled();

	bool versionPrintEnabled();
private:
	FloatPrecision precision;
	bool timeExecution;
	bool timeSubfunctions;
	bool maskInactiveGroups;
	bool showParameterInformation;
	bool saveIterationCorrelations;
	bool saveFrameCorrelations;
	bool saveArgumentsToFile;
	bool showConversionWarnings;
	bool showGpuMemoryUsage;
	bool showStopCriterion;
	bool medianFilterAfterSolve;
	bool medianFilterBeforeReturn;
	bool copyRawParameters;
	bool notifyOnFilterDisabled;
	bool treatWarningsAsErrors;
	bool frameOrderedParameters;
	bool saveIterationErrvals;

	bool versionPrint;
};


#endif
