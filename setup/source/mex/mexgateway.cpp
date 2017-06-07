#include "mexgateway.h"

#ifdef _MATLAB

#if defined( _WIN32 ) || defined( _WIN64 )
#include "../helper/cudahelper/cudahelper.cuh"
#include "reporter\\reporter.h"
#else
#include "../helper/cudahelper/cudahelper.cuh"
#include "reporter/reporter.h"
#endif

template<class T>
MexGateway<T>::MexGateway( Configuration* configuration ) : task( configuration ) {
	memset( &rom, 0, sizeof( PreAllocatedMemory<T> ) );

	this->configuration = configuration;

	Reporter::treatWarningsAsErrors(this->configuration->treatWarningsAsErrorsEnabled());

	//select Conversionvalidator used in
	//preparePersistentMemory and convertOutputArguments
	if( this->configuration->showConversionWarningsEnabled() ) {
		this->conversionValidator = new PrintConversionValidator();
	} else {
		this->conversionValidator = new ConversionValidator( this->configuration->treatWarningsAsErrorsEnabled() );
	}
}

template<class T>
bool MexGateway<T>::preparePersistentMemory( mxArray* prhs[], int nrhs ) {
	bool retv;
	ConversionResult result;
	char buff[128];

	retv = true;

	if( nrhs < 26 ) {
		sprintf( buff, "not enough input arguments, expected 26 - received %i", nrhs );
		Reporter::error( ReportID::NotEnoughInputs, buff );
	}

	int paramIndex = 1;

	//template
	result = MexConverter<T>::neutralFromMx( prhs[paramIndex], rom.templateImage, ConversionMode::ConvertTypesOnMismatch | ConversionMode::ShareData );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//xGradients
	result = MexConverter<T>::neutralFromMx( prhs[paramIndex], rom.xGradients, ConversionMode::ConvertTypesOnMismatch | ConversionMode::ShareData );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//yGradients
	result = MexConverter<T>::neutralFromMx( prhs[paramIndex], rom.yGradients, ConversionMode::ConvertTypesOnMismatch | ConversionMode::ShareData );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//frac
	result = MexConverter<T>::neutralFromMx( prhs[paramIndex], rom.frac, ConversionMode::ConvertTypesOnMismatch | ConversionMode::ShareData );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//baseX
	result = MexConverter<T>::neutralFromMx( prhs[paramIndex], rom.baseX, ConversionMode::ConvertTypesOnMismatch | ConversionMode::ShareData );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//baseY
	result = MexConverter<T>::neutralFromMx( prhs[paramIndex], rom.baseY, ConversionMode::ConvertTypesOnMismatch | ConversionMode::ShareData );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//baseMask
	result = MexConverter<bool>::neutralFromMx( prhs[paramIndex], rom.baseMask, ConversionMode::ConvertTypesOnMismatch | ConversionMode::ShareData );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//image data
	result = MexConverter<T>::neutralFromMx( prhs[paramIndex], rom.image, ConversionMode::ConvertTypesOnMismatch | ConversionMode::ShareData );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//init param x
	result = MexConverter<T>::neutralFromMx( prhs[paramIndex], rom.initParameterX, ConversionMode::ConvertTypesOnMismatch | ConversionMode::ShareData );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//init param y
	result = MexConverter<T>::neutralFromMx( prhs[paramIndex], rom.initParameterY, ConversionMode::ConvertTypesOnMismatch | ConversionMode::ShareData );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//groupSize
	result = MexConverter<int>::scalarFromMx( prhs[paramIndex], rom.groupSize, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//nFrames
	result = MexConverter<int>::scalarFromMx( prhs[paramIndex], rom.nFrames, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//minPointsPerBlock
#ifdef MINPOINTSPERBLOCK_AS_VECTOR
	result = MexConverter<int>::neutralFromMx( prhs[paramIndex], rom.minPointsPerBlock, ConversionMode::ConvertTypesOnMismatch | ConversionMode::ShareData );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;
#else
	result = MexConverter<int>::scalarFromMx( prhs[paramIndex], rom.minPointsPerBlock, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;
#endif

	//maxIterations
	result = MexConverter<int>::scalarFromMx( prhs[paramIndex], rom.maxIterations, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//moveThreshold
	result = MexConverter<T>::scalarFromMx( prhs[paramIndex], rom.moveThreshold, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//haltCorrelation
	result = MexConverter<T>::scalarFromMx( prhs[paramIndex], rom.haltCorrelation, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//correlationIncreaseThreshold
	result = MexConverter<T>::scalarFromMx( prhs[paramIndex], rom.correlationIncreaseThreshold, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//movementPenalty
	result = MexConverter<T>::scalarFromMx( prhs[paramIndex], rom.filterStrength, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//nFilterIterations
	result = MexConverter<int>::scalarFromMx( prhs[paramIndex], rom.nFilterIterations, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//filterCorrelationThreshold
	result = MexConverter<T>::scalarFromMx( prhs[paramIndex], rom.filterCorrelationThreshold, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//lambda
	result = MexConverter<T>::scalarFromMx( prhs[paramIndex], rom.lambda, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//linesearchReductionMultiplier
	result = MexConverter<T>::scalarFromMx( prhs[paramIndex], rom.linesearchReductionMultiplier, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//solver
	result = MexConverter<int>::scalarFromMx( prhs[paramIndex], rom.solverID, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//filterID
	result = MexConverter<int>::scalarFromMx( prhs[paramIndex], rom.filterID, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	//Gpu Device ID
	result = MexConverter<int>::scalarFromMx( prhs[paramIndex], rom.gpuDeviceID, ConversionMode::ConvertTypesOnMismatch );
	this->conversionValidator->validate( result, prhs[paramIndex], paramIndex );
	paramIndex++;

	rom.frameH = static_cast< int >( mxGetM( prhs[5] ) );
	rom.frameW = static_cast< int >( mxGetN( prhs[5] ) );
	rom.fframeW = rom.frameW * rom.nFrames;

	rom.nParamPoints = rom.frameW * rom.nFrames + 1;
	rom.nParams = rom.nParamPoints * 2;

	rom.templateH = mxGetM( prhs[1] );
	rom.templateW = mxGetN( prhs[1] );

	if( rom.groupSize > 1024 ) {
		sprintf( buff, "groupsize is too big\nmax groupsize: %i\n", 1024 );
		Reporter::error( ReportID::InvalidArgument, buff );
		return false;
	}

	rom.groupSize2 = 1;
	while( rom.groupSize2 < rom.groupSize ) rom.groupSize2 <<= 1;

	rom.frameW2 = 1;
	while( rom.frameW2 < rom.frameW ) rom.frameW2 <<= 1;

	rom.nGroups = rom.fframeW / rom.groupSize;

	if( this->configuration->parameterInformationPrintEnabled() ) {
		mexPrintf( "given scalar arguments:\n" );
		mexPrintf( "frameH:\t\t\t%i\n", rom.frameH );
		mexPrintf( "frameW:\t\t\t%i\n", rom.frameW );
		mexPrintf( "frameW2:\t\t%i\n", rom.frameW2 );
		mexPrintf( "nFrames:\t\t%i\n", rom.nFrames );
		mexPrintf( "fframeW:\t\t%i\n", rom.fframeW );
		mexPrintf( "nParamPoints:\t\t%i\n", rom.nParamPoints );
		mexPrintf( "nParams:\t\t%i\n", rom.nParams );
		mexPrintf( "templateH:\t\t%i\n", rom.templateH );
		mexPrintf( "templateW:\t\t%i\n", rom.templateW );
		mexPrintf( "groupSize:\t\t%i\n", rom.groupSize );
		mexPrintf( "groupSize2:\t\t%i\n", rom.groupSize2 );
		mexPrintf( "nGroups:\t\t%i\n", rom.nGroups );
#ifndef MINPOINTSPERBLOCK_AS_VECTOR
		mexPrintf( "minPointsPerBlock:\t%i\n", rom.minPointsPerBlock );
#endif
		mexPrintf( "maxIterations:\t\t%i\n", rom.maxIterations );
		mexPrintf( "moveThreshold:\t\t%g\n", rom.moveThreshold );
		mexPrintf( "haltCorrelation\t\t%g\n", rom.haltCorrelation );
		mexPrintf( "filterID:\t\t%i\n", rom.filterID );
		mexPrintf( "filterStrength:\t\t%g\n", rom.filterStrength );
		mexPrintf( "nFilterIterations:\t%i\n", rom.nFilterIterations );
		mexPrintf( "filterCorrThreshold:\t%g\n", rom.filterCorrelationThreshold );
		mexPrintf( "CorrIncreaseThreshold:\t%g\n", rom.correlationIncreaseThreshold );
		mexPrintf( "lambda:\t\t\t%g\n", rom.lambda );
		mexPrintf( "linesearch reduction:\t%g\n", rom.linesearchReductionMultiplier );
		mexPrintf( "errvalThreshold:\t%g\n", rom.errvalThreshold );
		mexPrintf( "solverID:\t\t%i\n", rom.solverID );
		mexPrintf( "gpuDeviceID:\t\t%i\n", rom.gpuDeviceID );
		mexPrintf( "\n" );
		mexEvalString( "drawnow;" );
	}

	//comp is used internally
	rom.errvalThreshold = 1 - rom.errvalThreshold;

	//save arguments to file
	if( this->configuration->saveArgumentsToFileEnabled() ) {
		size_t bytesWritten = MexSimulator<T>::saveRomToFile( ROM_FILE_PATH, rom );
		double fbytesWritten = static_cast< double >( bytesWritten );

		//break down in Mebi / Gibi byte representation
		const char* multiStr[] = { "B", "kiB", "MiB", "GiB", "TiB" };
		int multi = 0;
		while( fbytesWritten > 1024 ) {
			fbytesWritten /= 1024;
			multi++;
		}

		mexPrintf( "saved %.2f%s to %s\n", fbytesWritten, multiStr[multi], ROM_FILE_PATH );
		mexEvalString( "drawnow;" );
	}

	return retv;
}

/*
check bbo\motioncorrection\align_scansequence_to_template for up-to-date argument list
[block_dx_knots, block_dy_knots, frame_corr, block_iter_used, block_mask, block_errval]*/
template<class T>
bool MexGateway<T>::convertOutputArguments( TaskOutput<T>& src, mxArray* plhs[], int nlhs ) {
	bool retv = true;
	ConversionResult result;
	char buff[128];

	switch( nlhs ) {
	case 15: //allParams
		nlhs--;
		result = MexConverter<T>::mxFromNeutral( src.allParams, plhs[nlhs], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
		this->conversionValidator->validate( result, plhs[nlhs], nlhs, "allParams" );
		//no break
	case 14: //rgparams
		nlhs--;
		result = MexConverter<T>::mxFromNeutral( src.rgParams, plhs[nlhs], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
		this->conversionValidator->validate( result, plhs[nlhs], nlhs, "rgParams" );
		//no break
	case 13://raw parameters
		nlhs--;
		if( this->configuration->copyRawParametersEnabled() ) {
			result = MexConverter<T>::mxFromNeutral( src.rawParameters, plhs[nlhs], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
			this->conversionValidator->validate( result, plhs[nlhs], nlhs, "rawParameters" );
		} else plhs[nlhs] = MexConverter<T>::mxFromEmpty();
		//no break
	case 12://iterationErrvals
		nlhs--;
		if( this->configuration->saveIterationErrvalsEnabled() ) {
			result = MexConverter<T>::mxFromNeutral( src.iterationErrvals, plhs[10], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
			this->conversionValidator->validate( result, plhs[nlhs], nlhs, "iterationErrvals" );
		} else plhs[nlhs] = MexConverter<T>::mxFromEmpty();
		//no break
	case 11://frame correlations
		nlhs--;
		if( this->configuration->saveFrameCorrelationsEnabled() ) {
			result = MexConverter<T>::mxFromNeutral( src.frameCorrelations, plhs[nlhs], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
			this->conversionValidator->validate( result, plhs[nlhs], nlhs, "frameCorrelations" );
		} else plhs[nlhs] = MexConverter<T>::mxFromEmpty();
		//no break
	case 10://function times
		nlhs--;
		if( this->configuration->subFunctionTimingEnabled()
			|| this->configuration->executionTimingEnabled() ) {
			result = MexConverter<T>::mxFromNeutral( src.functionTimes, plhs[nlhs], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
			this->conversionValidator->validate( result, plhs[nlhs], nlhs, "subfunctionTimes" );
		} else plhs[nlhs] = MexConverter<T>::mxFromEmpty();
		//no break
	case 9://iteration times
		nlhs--;
		if( this->configuration->executionTimingEnabled() ) {
			result = MexConverter<T>::mxFromNeutral( src.iterationTimes, plhs[nlhs], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
			this->conversionValidator->validate( result, plhs[nlhs], nlhs, "iterationTimes" );
		} else plhs[nlhs] = MexConverter<T>::mxFromEmpty();
		//no break
	case 8://iteration correlations
		nlhs--;
		if( this->configuration->saveIterationCorrelationsEnabled() ) {
			result = MexConverter<T>::mxFromNeutral( src.iterationCorrelations, plhs[nlhs], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
			this->conversionValidator->validate( result, plhs[nlhs], nlhs, "iterationCorrelations" );
		} else plhs[nlhs] = MexConverter<T>::mxFromEmpty();
		//no break
	case 7://deviceInitTime
		nlhs--;
		result = MexConverter<double>::mxFromScalar( ( double )src.deviceInitTime, plhs[nlhs], mxClassID::mxDOUBLE_CLASS, ConversionMode::ConvertTypesOnMismatch );
		this->conversionValidator->validate( result, plhs[nlhs], nlhs, "deviceInitTime" );
		//no break
	case 6://block_errval
		nlhs--;
		result = MexConverter<T>::mxFromNeutral( src.errval, plhs[nlhs], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
		this->conversionValidator->validate( result, plhs[nlhs], nlhs, "residrms" );
		//no break
	case 5://block_mask
		nlhs--;
		result = MexConverter<bool>::mxFromNeutral( src.mask, plhs[nlhs], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
		this->conversionValidator->validate( result, plhs[nlhs], nlhs, "mask" );
		//no break
	case 4://block_iter_used
		nlhs--;
		result = MexConverter<int>::mxFromScalar( src.iterations, plhs[nlhs], mxClassID::mxINT32_CLASS, ConversionMode::ConvertTypesOnMismatch );
		this->conversionValidator->validate( result, plhs[nlhs], nlhs, "iterations" );
		//no break
	case 3://frame_corr
		nlhs--;
		result = MexConverter<T>::mxFromNeutral( src.finalFrameCorrelations, plhs[nlhs], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
		this->conversionValidator->validate( result, plhs[nlhs], nlhs, "finalFrameCorrelations" );
		//no break
	case 2://block_dy_knots
		nlhs--;
		result = MexConverter<T>::mxFromNeutral( src.blockParametersY, plhs[nlhs], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
		this->conversionValidator->validate( result, plhs[nlhs], nlhs, "blockParameterY" );
		//no break
	case 1: //block_dx_knots
		nlhs--;
		result = MexConverter<T>::mxFromNeutral( src.blockParametersX, plhs[nlhs], mxClassID::mxUNKNOWN_CLASS, ConversionMode::ConvertTypesOnMismatch );
		this->conversionValidator->validate( result, plhs[nlhs], nlhs, "blockParameterX" );
		break;
	case 0:
		Reporter::error(ReportID::InvalidArgument, "invalid number of lhs arguments specified\n" );
		retv = false;
		break;
	default:
		//cap return value count > 11
		if( nlhs > 15 ) {
			sprintf(buff, "too many return values (%i) specified.\nreturn values with nlhs > 15 will be unassigned.\n", nlhs );
			Reporter::warn(ReportID::InvalidArgument, buff);
			if( this->configuration->treatWarningsAsErrorsEnabled() ) return false;
			return this->convertOutputArguments( src, plhs, 15 );
		}
	}

	return retv;
}

template<class T>
void MexGateway<T>::execute( int nlhs, mxArray* plhs[], int nrhs, mxArray* prhs[] ) {
	TaskOutput<T> out;
	RealTimer<T> timer;
	char buff[128];

	if( this->configuration->parameterInformationPrintEnabled() ) {
		sprintf(buff, "nlhs: %i\nnrhs: %i\n", nlhs, nrhs );
		Reporter::inform(buff);
	}

	//preallocate memory
	timer.start();
	if( !this->preparePersistentMemory( prhs, nrhs ) ) return;
	if( !CudaHelper<T>::initDevice( this->rom.gpuDeviceID ) ) {
		Reporter::error(ReportID::CudaError, "device initialization failed");
	}

	out.deviceInitTime = timer.stop();

	this->task.initialize( rom );
	this->task.allocateOutput( out );

	//store allocation time
	if( this->configuration->executionTimingEnabled() ) {
		out.functionTimes[12] = timer.stop() - out.deviceInitTime;
		timer.start();
	}

	//execute alignment
	StopCriterion result = this->task.alignFrames( out );

	if( this->configuration->showStopCriterionEnabled() ) {
		mexPrintf( "Execution terminated:%s\n", StopCriterion_toString( result ) );
		mexPrintf( "iterations: %i\n", out.iterations );
		mexEvalString( "drawnow;" );
	}

	//store execution time
	if( this->configuration->executionTimingEnabled() ) {
		out.functionTimes[13] = timer.stop();
		timer.start();
	}

	if( this->configuration->subFunctionTimingEnabled() ) {
		const char* timeNames[] = {
			"Initialize\t",
			"ApplyParameters\t",
			"CalcErrval\t",
			"UpdateGradients\t",
			"SubSparseOffset\t",
			"FillDiagonals\t",
			"FillJTr\t\t",
			"Solve\t\t",
			"ExpandResults\t",
			"Linesearch\t",
			"Correlations\t",
			"MaskInactive\t"
		};

		mexPrintf( "\nsubfunctiontimes:\n" );
		for( int i = 0; i < 12; i++ ) {
			mexPrintf( "%s%.3f\t( %.3f )\n", timeNames[i], out.functionTimes[i] / out.iterations, out.functionTimes[i] );
		}
	}

	//get parameters
	this->task.getParameters( out );

	if( !this->convertOutputArguments( out, plhs, nlhs ) ) {
		Reporter::error(ReportID::Unknown, "output conversion failed");
	}

}


template class MexGateway<float>;
template class MexGateway<double>;

#endif /* _MATLAB */

