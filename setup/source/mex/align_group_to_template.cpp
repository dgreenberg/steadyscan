#include "mexgateway.h"

#ifdef _MATLAB

template<typename T>
static void ExecuteGateway( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[], Configuration* configuration ) {
	//create gateway
	MexGateway<T> gateway( configuration );

	//execute alignment
	gateway.execute( nlhs, plhs, nrhs, const_cast<mxArray**>( prhs ) );

	//destroy gateway
}

/*
 * MATLAB mex entry point
 * parses flag parameter to decide template type (precision float/double)
 * instantiates gateway and executes frame alignment
 */
void mexFunction( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) {
	RealTimer<float> timer;
	Configuration config;
	char* flags;

	//start timer in case execution timing is enabled
	timer.start();

	//read the flags parameter
	flags = mxArrayToString( prhs[0] );
	if( nullptr == flags ) {
		mexPrintf( "error reading flags parameter\n" );
		return;
	}

	ConfigurationParser::parseFlags( flags, &config );

	mxFree( flags );

	if( config.versionPrintEnabled() ) {
		mexPrintf( "align_group_to_template version: %s\n", VERSION );
		mexEvalString( "drawnow;" );
	}

	//print flags
	if( config.parameterInformationPrintEnabled() ) {
		mexPrintf( "parsed flags:\n" );
		mexPrintf( "precision:\t\t\t%s\n", ::FloatPrecision_toString( config.precisionType() ) );
		mexPrintf( "time execution:\t\t\t%s\n", config.executionTimingEnabled()?"true":"false" );
		mexPrintf( "time subfunctions:\t\t%s\n", config.subFunctionTimingEnabled()?"true":"false" );
		mexPrintf( "mask correlated groups:\t\t%s\n", config.maskInactiveGroupsEnabled()?"true":"false" );
		mexPrintf( "show parameter information:\t%s\n", config.parameterInformationPrintEnabled()?"true":"false" );
		mexPrintf( "save iteration correlations:\t%s\n", config.saveIterationCorrelationsEnabled()?"true":"false" );
		mexPrintf( "save all frame correlations:\t%s\n", config.saveFrameCorrelationsEnabled()?"true":"false" );
		mexPrintf( "save arguments to file:\t\t%s\n", config.saveArgumentsToFileEnabled()?"true":"false" );
		mexPrintf( "show gateway warnings:\t\t%s\n", config.showConversionWarningsEnabled()?"true":"false" );
		mexPrintf( "gateway warnings are errors:\t%s\n", config.treatWarningsAsErrorsEnabled()?"true":"false" );
		mexPrintf( "show gpu memory usage:\t\t%s\n", config.showGpuMemoryUsageEnabled()?"true":"false" );
		mexPrintf( "show stop criterion:\t\t%s\n", config.showStopCriterionEnabled()?"true":"false" );
		mexPrintf( "3-median parameterfilter:\t%s\n", config.medianFilterAfterSolveEnabled()?"true":"false" );
		mexPrintf( "3-median endFilter:\t\t%s\n", config.medianFilterBeforeReturnEnabled()?"true":"false" );
		mexPrintf( "copy raw parameters:\t\t%s\n", config.copyRawParametersEnabled()?"true":"false" );
		mexPrintf( "notify when disabling filter:\t%s\n", config.notifyOnFilterDisabledEnabled()?"true":"false" );
		mexPrintf( "\n" );
		mexEvalString( "drawnow;" );
	}

	//execute the algorithm according to the requested template type
	switch( config.precisionType() ) {
	case FloatPrecision::Single:
		::ExecuteGateway<float>( nlhs, plhs, nrhs, prhs, &config );
		break;
	case FloatPrecision::Double:
		::ExecuteGateway<double>( nlhs, plhs, nrhs, prhs, &config );
		break;
	default:
		mexPrintf( "invalid precision type!\nuse 'S' for single-precision and 'D' for double-precision.\n" );
		break;
	}

	//print time used in function
	if( config.executionTimingEnabled() ) {
		mexPrintf( "time in mexFunction: %g\n", timer.stop() );
	}
}

#endif /* _MATLAB */
