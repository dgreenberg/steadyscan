#include "mexsimulator.h"
#include "../../helper/timer/realtimer.h"
#include "../../helper/cudahelper/cudahelper.cuh"

template<class T, class Major>
size_t WriteMatrix( NeutralMatrix<T, int, Major>& mat, FILE* fptr ) {
	int rows, cols;
	size_t written = 0;

	rows = mat.rows();
	cols = mat.cols();

	written += fwrite( &rows, sizeof( int ), 1, fptr ) * sizeof(int);
	written += fwrite( &cols, sizeof( int ), 1, fptr ) * sizeof(int);

	written += fwrite( mat.data(), sizeof( T ), mat.rows() * mat.cols(), fptr ) * sizeof(T);

	return written;
}

template<class T, class Major>
size_t ReadMatrix( NeutralMatrix<T, int, Major>& mat, FILE* fptr ) {
	int rows, cols;
	size_t read = 0;

	read += fread( &rows, sizeof( int ), 1, fptr ) * sizeof(int);
	read += fread( &cols, sizeof( int ), 1, fptr ) * sizeof(int);

	mat = NeutralMatrix<T, int, Major>( rows, cols );
	read += fread( mat.data(), sizeof( T ), rows*cols, fptr ) * sizeof(T);

	return read;
}

//retv:
//-1: dimension mismatch
// 0: equal matrices
//>0: no of mismatches
template<class T, class Major>
int CompareMatrix( NeutralMatrix<T, int, Major>& a, NeutralMatrix<T, int, Major>& b ) {
	if(a.cols() != b.cols()) return -1;
	if(a.rows() != b.rows()) return -1;
	unsigned long long ec = 0;

	for( int i = 0; i < a.rows() * a.cols() * sizeof( T ); i++ ) {
		if( reinterpret_cast<unsigned char*>( a.data() )[i] != reinterpret_cast<unsigned char*>( b.data() )[i] ) ec++;
	}

	return ec / sizeof( T );
}

template<class T>
size_t MexSimulator<T>::saveRomToFile( const char* path, PreAllocatedMemory<T>& data ) {
	FILE* fptr = fopen( path, "wb+" );

	size_t written = fwrite(&data, sizeof( PreAllocatedMemory<T> ), 1, fptr) * sizeof(PreAllocatedMemory<T>);

	written += WriteMatrix<T>( data.templateImage, fptr );
	written += WriteMatrix<T>( data.xGradients, fptr );
	written += WriteMatrix<T>( data.yGradients, fptr );
	written += WriteMatrix<T>( data.frac, fptr );
	written += WriteMatrix<T>( data.baseX, fptr );
	written += WriteMatrix<T>( data.baseY, fptr );
	written += WriteMatrix<bool>( data.baseMask, fptr );
	written += WriteMatrix<T>( data.image, fptr );
	written += WriteMatrix<T>( data.initParameterX, fptr );
	written += WriteMatrix<T>( data.initParameterY, fptr );
#ifdef MINPOINTSPERBLOCK_AS_VECTOR
	written += WriteMatrix<int>( data.minPointsPerBlock, fptr );
#endif
	fclose( fptr );
	return written;
}

template<class T>
size_t MexSimulator<T>::loadRomFromFile( const char* path, PreAllocatedMemory<T>& data ) {
	FILE* fptr = fopen( path, "rb+" );

	size_t read = fread( &data, sizeof( PreAllocatedMemory<T> ), 1, fptr ) * sizeof(PreAllocatedMemory<T>);

	read += ReadMatrix<T>( data.templateImage, fptr );
	read += ReadMatrix<T>( data.xGradients, fptr );
	read += ReadMatrix<T>( data.yGradients, fptr );
	read += ReadMatrix<T>( data.frac, fptr );
	read += ReadMatrix<T>( data.baseX, fptr );
	read += ReadMatrix<T>( data.baseY, fptr );
	read += ReadMatrix<bool>( data.baseMask, fptr );
	read += ReadMatrix<T>( data.image, fptr );
	read += ReadMatrix<T>( data.initParameterX, fptr );
	read += ReadMatrix<T>( data.initParameterY, fptr );
#ifdef MINPOINTSPERBLOCK_AS_VECTOR
	read += ReadMatrix<int>( data.minPointsPerBlock, fptr );
#endif

	fclose( fptr );

	return read;
}

template<class T>
bool MexSimulator<T>::compareRomToFile( const char* path, PreAllocatedMemory<T>& data ) {
	PreAllocatedMemory<T> buf;
	MexSimulator<T>::loadRomFromFile( path, buf );
	int result;
	bool retv = true;

	result = CompareMatrix<bool, RowMajor<int>>( buf.baseMask, data.baseMask );
	if( result != 0 ) {
		printf("baseMask:: %i mismatches\n", result);
		retv = false;
	}

	result = CompareMatrix<T, RowMajor<int>>( buf.baseX, data.baseX );
	if( result != 0 ) {
		printf("baseX:: %i mismatches\n", result);
		retv = false;
	}

	result = CompareMatrix<T, RowMajor<int>>( buf.baseY, data.baseY );
	if( result != 0 ) {
		printf("baseY:: %i mismatches\n", result);
		retv = false;
	}

	result = CompareMatrix<T, RowMajor<int>>( buf.frac, data.frac );
	if( result != 0 ) {
		printf("frac:: %i mismatches\n", result);
		retv = false;
	}

	result = CompareMatrix<T, RowMajor<int>>( buf.image, data.image );
	if( result != 0 ) {
		printf("image:: %i mismatches\n", result);
		retv = false;
	}

	result = CompareMatrix<T>( buf.initParameterX, data.initParameterX );
	if( result != 0 ) {
		printf("initParameterX:: %i mismatches\n", result);
		retv = false;
	}

	result = CompareMatrix<T>( buf.initParameterY, data.initParameterY );
	if( result != 0 ) {
		printf("initParameterY:: %i mismatches\n", result);
		retv = false;
	}

#ifdef MINPOINTSPERBLOCK_AS_VECTOR
	result = CompareMatrix<int, RowMajor<int>>( buf.minPointsPerBlock, data.minPointsPerBlock );
	if( result != 0 ) {
		printf("minPointsPerBlock:: %i mismatches\n", result);
		retv = false;
	}
#else
	if( data.minPointsPerBlock != buf.minPointsPerBlock ) {
		printf("minPointsPerBlock mismatch\n");
		retv = false;
	}
#endif

	result = CompareMatrix<T, RowMajor<int>>( buf.templateImage, data.templateImage );
	if( result != 0 ) {
		printf("templateImage:: %i mismatches\n", result);
		retv = false;
	}

	result = CompareMatrix<T, RowMajor<int>>( buf.xGradients, data.xGradients );
	if( result != 0 ) {
		printf("xGradients:: %i mismatches\n", result);
		retv = false;
	}

	result = CompareMatrix<T, RowMajor<int>>( buf.yGradients, data.yGradients );
	if( result != 0 ) {
		printf("yGradients:: %i mismatches\n", result);
		retv = false;
	}

	if( data.frameH != buf.frameH ) {
		printf("frameH mismatch\n");
		retv = false;
	}
	if( data.frameW != buf.frameW ) {
		printf("frameW mismatch\n");
		retv = false;
	}
	if( data.frameW2 != buf.frameW2 ) {
		printf("frameW2 mismatch\n");
		retv = false;
	}
	if( data.templateH != buf.templateH ) {
		printf("templateH mismatch\n");
		retv = false;
	}
	if( data.templateW != buf.templateW ) {
		printf("templateW mismatch\n");
		retv = false;
	}
	if( data.fframeW != buf.fframeW ) {
		printf("fframeW mismatch\n");
		retv = false;
	}
	if( data.nParamPoints != buf.nParamPoints ) {
		printf("nParamPoints mismatch\n");
		retv = false;
	}
	if( data.nParams != buf.nParams ) {
		printf("nParams mismatch\n");
		retv = false;
	}
	if( data.nFrames != buf.nFrames ) {
		printf("nFrames mismatch\n");
		retv = false;
	}
	if( data.groupSize != buf.groupSize ) {
		printf("groupSize mismatch\n");
		retv = false;
	}
	if( data.groupSize2 != buf.groupSize2 ) {
		printf("groupSize2 mismatch\n");
		retv = false;
	}
	if( data.nGroups != buf.nGroups ) {
		printf("nGroups mismatch\n");
		retv = false;
	}
	if( data.maxIterations != buf.maxIterations ) {
		printf("maxIterations mismatch\n");
		retv = false;
	}
	if( data.moveThreshold != buf.moveThreshold ) {
		printf("moveThreshold mismatch\n");
		retv = false;
	}
	if( data.haltCorrelation != buf.haltCorrelation ) {
		printf("haltCorrelation mismatch\n");
		retv = false;
	}
	if( data.filterStrength != buf.filterStrength ) {
		printf("filterStrength mismatch\n");
		retv = false;
	}
	if( data.nFilterIterations != buf.nFilterIterations ) {
		printf("nFilterIterations mismatch\n");
		retv = false;
	}
	if( data.filterCorrelationThreshold != buf.filterCorrelationThreshold ) {
		printf("filterCorrelationThreshold mismatch\n");
		retv = false;
	}
	if( data.correlationIncreaseThreshold != buf.correlationIncreaseThreshold ) {
		printf("correlationIncreaseThreshold mismatch\n");
		retv = false;
	}
	if( data.lambda != buf.lambda ) {
		printf("lambda mismatch\n");
		retv = false;
	}
	if( data.linesearchReductionMultiplier != buf.linesearchReductionMultiplier ) {
		printf("linesearchReductionMultiplier mismatch\n");
		retv = false;
	}
	if( data.solverID != buf.solverID ) {
		printf("solverID mismatch\n");
		retv = false;
	}
	if( data.filterID != buf.filterID ) {
		printf("filterID mismatch\n");
		retv = false;
	}
	if( data.gpuDeviceID != buf.gpuDeviceID ) {
		printf("gpuDeviceID mismatch\n");
		retv = false;
	}
	if( data.errvalThreshold != buf.errvalThreshold ) {
		printf("errvalThreshold mismatch\n");
		retv = false;
	}

	if(retv) printf("rom equals file\n");
	return retv;
}



template<typename T, typename Tconv, class Major>
NeutralMatrix<Tconv> ConvertNeutralMatrix( NeutralMatrix<T, int, Major>& base )  {
	NeutralMatrix<Tconv> retv( base.rows(), base.cols() );

	for( int i = base.rows() * base.cols() - 1; i >= 0; i-- ) {
		retv[i] = static_cast<Tconv>( base[i] );
	}

	return retv;
}

template<typename Tin, typename Tout>
void lcrff( const char* path, PreAllocatedMemory<Tout>& data ) {
	PreAllocatedMemory<Tin> buf;

	MexSimulator<Tin>::loadRomFromFile( path, buf );

	data.baseMask = buf.baseMask;
	data.baseX = ConvertNeutralMatrix<Tin, Tout, RowMajor<int>>( buf.baseX );
	data.baseY = ConvertNeutralMatrix<Tin, Tout, RowMajor<int>>( buf.baseY );
	data.frac = ConvertNeutralMatrix<Tin, Tout, RowMajor<int>>( buf.frac );
	data.image = ConvertNeutralMatrix<Tin, Tout, RowMajor<int>>( buf.image );
	data.initParameterX = ConvertNeutralMatrix<Tin, Tout>( buf.initParameterX );
	data.initParameterY = ConvertNeutralMatrix<Tin, Tout>( buf.initParameterY );
	data.templateImage = ConvertNeutralMatrix<Tin, Tout, RowMajor<int>>( buf.templateImage );
	data.xGradients = ConvertNeutralMatrix<Tin, Tout, RowMajor<int>>( buf.xGradients );
	data.yGradients = ConvertNeutralMatrix<Tin, Tout, RowMajor<int>>( buf.yGradients );
	data.minPointsPerBlock = buf.minPointsPerBlock;

	//copy each parameter
	//using implicit type conversion
	data.frameH = buf.frameH;
	data.frameW = buf.frameW;
	data.frameW2 = buf.frameW2;
	data.templateH = buf.templateH;
	data.templateW = buf.templateW;
	data.fframeW = buf.fframeW;
	data.nParamPoints = buf.nParamPoints;
	data.nParams = buf.nParams;
	data.nFrames = buf.nFrames;

	data.groupSize = buf.groupSize;
	data.groupSize2 = buf.groupSize2;
	data.nGroups = buf.nGroups;

	data.maxIterations = buf.maxIterations;
	data.moveThreshold = buf.moveThreshold;
	data.haltCorrelation = buf.haltCorrelation;
	data.filterStrength = buf.filterStrength;

	data.nFilterIterations = buf.nFilterIterations;
	data.filterCorrelationThreshold = buf.filterCorrelationThreshold;
	data.correlationIncreaseThreshold = buf.correlationIncreaseThreshold;

	data.lambda = buf.lambda;
	data.linesearchReductionMultiplier = buf.linesearchReductionMultiplier;

	data.solverID = buf.solverID;
	data.filterID = buf.filterID;
	data.gpuDeviceID = buf.gpuDeviceID;
	data.errvalThreshold = buf.errvalThreshold;
}

template<typename Tload, typename Texec>
double exec() {
	PreAllocatedMemory<Texec> rom;
	TaskOutput<Texec> to;
	Configuration conf;
	Task<Texec> task( &conf );
	RealTimer<Texec> t;
	StopCriterion result = StopCriterion::Error;

	ConfigurationParser::parseFlags( "DE", &conf );

	printf( "load start\n" );
	t.start();
	lcrff<Tload, Texec>( ROM_FILE_PATH, rom );
	printf( "load: %fs\n", t.stop() );

	rom.filterID = 0;
	printf("simulator> fixed filter to %i\n", rom.filterID);

	printf( "given scalar arguments:\n" );
	printf( "frameH:\t\t\t%i\n", rom.frameH );
	printf( "frameW:\t\t\t%i\n", rom.frameW );
	printf( "frameW2:\t\t%i\n", rom.frameW2 );
	printf( "nFrames:\t\t%i\n", rom.nFrames );
	printf( "fframeW:\t\t%i\n", rom.fframeW );
	printf( "nParamPoints:\t\t%i\n", rom.nParamPoints );
	printf( "nParams:\t\t%i\n", rom.nParams );
	printf( "templateH:\t\t%i\n", rom.templateH );
	printf( "templateW:\t\t%i\n", rom.templateW );
	printf( "groupSize:\t\t%i\n", rom.groupSize );
	printf( "groupSize2:\t\t%i\n", rom.groupSize2 );
	printf( "nGroups:\t\t%i\n", rom.nGroups );
#ifndef MINPOINTSPERBLOCK_AS_VECTOR
	printf( "minPointsPerBlock:\t%i\n", rom.minPointsPerBlock );
#endif
	printf( "maxIterations:\t\t%i\n", rom.maxIterations );
	printf( "moveThreshold:\t\t%g\n", rom.moveThreshold );
	printf( "haltCorrelation\t\t%g\n", rom.haltCorrelation );
	printf( "filterID:\t\t%i\n", rom.filterID );
	printf( "filterStrength:\t\t%g\n", rom.filterStrength );
	printf( "nFilterIterations:\t%i\n", rom.nFilterIterations );
	printf( "filterCorrThreshold:\t%g\n", rom.filterCorrelationThreshold );
	printf( "CorrIncreaseThreshold:\t%g\n", rom.correlationIncreaseThreshold );
	printf( "lambda:\t\t\t%g\n", rom.lambda );
	printf( "linesearch reduction:\t%g\n", rom.linesearchReductionMultiplier );
	printf( "errvalThreshold:\t%g\n", rom.errvalThreshold );
	printf( "solverID:\t\t%i\n", rom.solverID );
	printf( "gpuDeviceID:\t\t%i\n", rom.gpuDeviceID );
	printf( "\n" );

	t.start();
	if( !CudaHelper<Texec>::initDevice( rom.gpuDeviceID ) ) return t.stop();
	double deviceInitTime = t.stop();
	printf("device init: %gs\n", deviceInitTime );

	t.start();
	bool loadOk = task.initialize( rom );
	if( loadOk ) {
		task.allocateOutput( to );
		printf( "alloc: %fs\n", t.stop() );

		t.start();
		result = task.alignFrames( to );
		printf( "execute: %fs\n", t.stop() );
	} else {
		return deviceInitTime;
	}

	if( conf.showStopCriterionEnabled() ) {
		mexPrintf( "Execution terminated:%s\n", StopCriterion_toString( result ) );
		mexPrintf( "iterations: %i\n", to.iterations );
		mexEvalString( "drawnow;" );
	}

	if( conf.subFunctionTimingEnabled() ) {
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
		printf( "\nsubfunctiontimes:\n" );
		for( int i = 0; i < 12; i++ ) {
			printf( "%s%.3f\t( %.3f )\n", timeNames[i], to.functionTimes[i] / to.iterations, to.functionTimes[i] );
		}
	}

	task.getParameters( to );

	t.start();
	task.freePreAllocatedMemory( nullptr );
	printf( "free: %fs\n", t.stop() );

	return deviceInitTime;
}

#ifndef _MATLAB
int main() {
	RealTimer<float> t;

	t.start();
 	double deviceInitTime = exec<float, float>();
 	printf( "exec<double, double>: %fs\n", t.stop() - deviceInitTime );

	//getchar();
	return 0;
}
#endif

template class MexSimulator<float>;
template class MexSimulator<double>;
