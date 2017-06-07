#include "task.h"
#include "task_allocate.cuh"
#include "task_correlation.cuh"
#include "task_maskinactive.cuh"
#include "task_calcerrval.cuh"

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\helper\\cudahelper\\cudahelper.cuh"
#include "..\\helper\\eps.h"
#include "solver\\allsolver.cuh"
#include "mainiteration\\allmainiterations.h"
#include "..\\interface\\parameterorganizer\\parameterorganizer.h"
#include "..\\helper\\minmax.h"
#include "..\\mex\\reporter\\reporter.h"
#else
#include "../helper/cudahelper/cudahelper.cuh"
#include "../helper/eps.h"
#include "solver/allsolver.cuh"
#include "mainiteration/allmainiterations.h"
#include "../interface/parameterorganizer/parameterorganizer.h"
#include "../helper/minmax.h"
#include "../mex/reporter/reporter.h"
#endif

#include "../helper/helper.h"
#include "../mex/mexsimulator/mexsimulator.h"

#ifdef EIGENINCLUDE
#include EIGENINCLUDE
#endif

#ifdef MEXINCLUDE
#include MEXINCLUDE
#endif

template<class T>
Task<T>::Task( Configuration* config ) {
	memset( this, 0, sizeof( Task<T> ) );
	this->configuration = config;
}

template<class T>
Task<T>::~Task() {
	this->freePreAllocatedMemory( nullptr );
}

template<class T>
bool Task<T>::initialize( PreAllocatedMemory<T>& rom ) {
	size_t freeGpuMemory, freeGpuMemory2, totalGpuMemory;

	IFilter<T>* filter;

	//create selected filter
	switch( rom.filterID & ( 0 - ( rom.nFilterIterations != 0 ) ) ) {
	case 0:
		filter = NoFilter<T>::Static();
		rom.nFilterIterations = 0;
		if( this->configuration->notifyOnFilterDisabledEnabled() ) {
			Reporter::inform( "filter disabled before alignment started" );
		}
		break;
	case 1:
		filter = new SmearingFilter<T>( rom.filterStrength );
		break;
	case 2:
		filter = new RangeFilter<T>( rom.filterStrength );
		break;
	default:
		Reporter::error( ReportID::InvalidArgument, "invalid filter ID\n0: disable filter\n1: smearingfiter\n2:rangefilter\n" );
	}

	ISolver<T>* solver;
	//create selected solver
	switch( rom.solverID ) {
	case 0: //CuSparse GPU Solver
		solver = new SolverCuSparse<T>( filter );
		break;
	case 1: //CuSolver GPU Solver
		solver = new SolverCuSolver<T>( filter );
		break;
	case 2: //Eigen CPU Solver
		solver = new SolverEigen<T>( filter );
		break;
	default:
		Reporter::error( ReportID::InvalidArgument, "invalid solver id\n0:CuSparse solver\n1:CuSolver solver\n2:Eigen solver\n" );
		if( nullptr != filter ) delete filter;
		return false;
	}

	//set the filter that is always executed
	if( this->configuration->medianFilterAfterSolveEnabled() ) filter = new MedianFilter<T>();
	else filter = NoFilter<T>::Static();

	//set the filter that is executed after converging
	if( this->configuration->medianFilterBeforeReturnEnabled() ) this->endFilter = new MedianFilter<T>();
	else this->endFilter = NoFilter<T>::Static();

	//select Type of MainIteration based on timing mode
	if( this->configuration->subFunctionTimingEnabled() ) {
		this->mainIteration = new TimedMainIteration<T>( solver, filter, this->configuration );
	} else {
		this->mainIteration = new MainIteration<T>( solver, filter, this->configuration );
	}

	//get the amount of free memory before allocation
	//( CUDA has no method to read the amount of allocated memory by process )
	if( this->configuration->showGpuMemoryUsageEnabled() ) {
		if( cudaSuccess != cudaMemGetInfo( &freeGpuMemory, &totalGpuMemory ) ) {
			//only print error message as success is irrelevant for alignment execution
			Reporter::warn( ReportID::CudaWarning, "unable to call cudaMemGetInfo before allocation\n" );
			freeGpuMemory = totalGpuMemory = 0;
		}
	}

	//preallocate memory
	switch( h_allocateDeviceMemory( rom, this->mem ) ) {
	case 0:
		break;
	case -1:
		cudaDeviceReset();
		Reporter::error( ReportID::CudaError, "unable to allocate device memory" );
		return false;
	case -2:
		cudaDeviceReset();
		Reporter::error( ReportID::CudaError, "unable to initialize cusparse" );
		return false;
	case -3:
		cudaDeviceReset();
		Reporter::error( ReportID::CudaError, "unable to initialize cusolver\n" );
		return false;
	}

	//get the amount of free memory after allocation
	if( this->configuration->showGpuMemoryUsageEnabled() ) {
		char buff[128];
		if( cudaSuccess != cudaMemGetInfo( &freeGpuMemory2, &totalGpuMemory ) ) {
			Reporter::warn( ReportID::CudaWarning, "unable to call cudaMemGetInfo after allocation\n" );
			freeGpuMemory2 = 0;
		}

		//allocated memory is:
		//free memory before allocation - free memory after allocation
		//if no other process was able to allocate during this process's allocation
		//the difference in free bytes should be the amount of process GPU memory
		float usedGpuMemory = freeGpuMemory - freeGpuMemory2;

		//break down in Mebi / Gibi byte representation
		const char* multiStr[] = { "B", "kiB", "MiB", "GiB", "TiB" };
		int multi = 0;
		while( usedGpuMemory > 1024 ) {
			usedGpuMemory /= 1024;
			multi++;
		}

		sprintf( buff, "gpu memory usage is approx. %.2f%s\n", usedGpuMemory, multiStr[multi] );
		Reporter::inform( buff );
	}

	return true;
}

//frees all resources allocated on the gpu
//and pointed to by DeviceMemory<T>* or this->d_mem
template<class T>
void Task<T>::freePreAllocatedMemory( DeviceMemory<T>* mem ) {
	//if cleanup is not called on a specific struct
	//try to get the instance from gpu memory
	bool freeMem = false;
	if( nullptr == mem ) {
		if( nullptr == this->mem.d_mem ) return;
		mem = new DeviceMemory<T>();
		freeMem = true;
		if( cudaSuccess != cudaMemcpy( mem, this->mem.d_mem, sizeof( DeviceMemory<T> ), cudaMemcpyDeviceToHost ) ) {
			delete mem;
			return;
		}
	}

	::h_freeDeviceMemory( *mem );

	if( freeMem ) delete mem;
	else mem->d_mem = nullptr;

	this->mem.d_mem = nullptr;

	if( nullptr != this->mainIteration ) {
		zDelete( this->mainIteration );
	}

	if( cudaDeviceReset() == cudaSuccess ) {
#ifdef _DEBUG
		Reporter::inform( "cleanup ok" );
#endif
	} else {
		Reporter::warn( ReportID::CudaWarning, "error reported by cudaDeviceReset\n" );
	}
}

//executed last fitlering of parameters
//and converts the parameter array to the matlab format
template<class T>
void Task<T>::getParameters( TaskOutput<T>& output ) {
	;
	NeutralMatrix<T> buff( mem.nParams, 1 );

	//get data from device
	cudaMemcpy( buff.data(), this->mem.pLast, sizeof( T ) * this->mem.nParams, cudaMemcpyDeviceToHost );


	if( this->configuration->frameOrderedParametersEnabled() ) {
		ParameterOrganizer<T>::organize1D2( buff, output.frameParameters, this->mem.nFrames, this->mem.frameW, this->mem.nParamPoints );
	} else {
		ParameterOrganizer<T>::organize1D2( buff, output.blockParametersX, output.blockParametersY, this->mem.nFrames, this->mem.frameW, this->mem.nParamPoints );
	}

	if( this->configuration->copyRawParametersEnabled() ) {
		cudaMemcpy( output.rawParameters.data(), this->mem.pLast, sizeof( T ) * this->mem.nParams, cudaMemcpyDeviceToHost );
	}
}

//allocates matrices in the TaskOutput struct
template<class T>
void Task<T>::allocateOutput( TaskOutput<T>& output ) {
	//allocate memory for default return matrices
	output.errval = NeutralMatrix<T>( this->mem.nFrames, 1 );
	output.finalFrameCorrelations = NeutralMatrix<T>( this->mem.nFrames, 1 );

	if( this->configuration->frameOrderedParametersEnabled() ) {
		output.frameParameters = NeutralMatrix<T>( this->mem.nFrames, ( this->mem.frameW + 1 ) * 2 );
	} else {
		output.blockParametersX = NeutralMatrix<T>( ( this->mem.frameW + 1 ) * this->mem.nFrames, 1 );
		output.blockParametersY = NeutralMatrix<T>( ( this->mem.frameW + 1 ) * this->mem.nFrames, 1 );
	}

	//allocate memory for additional return matrices
	if( this->configuration->saveIterationCorrelationsEnabled() ) {
		output.iterationCorrelations = NeutralMatrix<T>( mem.maxIterations, 1, 0 );
	}
	if( this->configuration->saveFrameCorrelationsEnabled() ) {
		output.frameCorrelations = NeutralMatrix<T>( mem.nFrames, mem.maxIterations, 0 );
	}
	if( this->configuration->executionTimingEnabled() ||
		this->configuration->subFunctionTimingEnabled() ) {
		output.functionTimes = NeutralMatrix<T>( TimedMainIteration<T>::nTimedFunctions + 4, 1, 0 );
	}
	if( this->configuration->subFunctionTimingEnabled() ) {
		output.iterationTimes = NeutralMatrix<T>( mem.maxIterations, 1, 0 );
	}
	if( this->configuration->copyRawParametersEnabled() ) {
		output.rawParameters = NeutralMatrix<T>( mem.nParams, 1, 0 );
	}
	if( this->configuration->saveIterationErrvalsEnabled() ) {
		output.iterationErrvals = NeutralMatrix<T>( mem.maxIterations * 2, mem.nFrames );
	}

	output.allParams = NeutralMatrix<T, int, ColMajor<int>>( mem.nParams, mem.maxIterations );
}

template<class T>
StopCriterion Task<T>::alignFrames( TaskOutput<T>& output ) {
	T correlation = 0, corr;
	int iteration = 0;
	StopCriterion retv;
	RealTimer<T> iterationTimer, subFunctionTimer;

	do {
		//start iteration timing
		if( this->configuration->subFunctionTimingEnabled() ) {
			iterationTimer.start();
		}

		//execute one iteration
		//(red & green groups)
		iteration++;
		bool improved = this->mainIteration->execute( this->mem, iteration );

		if( this->configuration->subFunctionTimingEnabled() ) {
			subFunctionTimer.start();
		}

		cudaMemcpy( &output.allParams.data()[mem.nParams * ( iteration - 1 )], mem.p, sizeof( T ) * mem.nParams, cudaMemcpyDeviceToHost );

		//calculate the global correlation
		hd_globalCorrelation( mem, &corr );

		//skip per frame correlation calculation unless enabled by config flag
		if( this->configuration->saveFrameCorrelationsEnabled() ) {
			//calculate the correlation for each frame
			hd_frameCorrelation( mem );
		}

		//save time for correlation
		if( this->configuration->subFunctionTimingEnabled() ) {
			cudaDeviceSynchronize();
			output.functionTimes[TimedMainIteration<T>::nTimedFunctions] += subFunctionTimer.stop();
		}

		//save global correlation
		if( this->configuration->saveIterationCorrelationsEnabled() ) {
			output.iterationCorrelations[iteration - 1] = corr;
		}

		//save frame correlations
		if( this->configuration->saveFrameCorrelationsEnabled() ) {
			cudaMemcpy( &( output.frameCorrelations.data()[( iteration - 1 ) * mem.nFrames] ), mem.corr, sizeof( T ) * mem.nFrames, cudaMemcpyDeviceToHost );
		}

		//if correlation did not increase with activated filter
		//deactivate the filter and retry
		if( corr - correlation < mem.correlationIncreaseThreshold ) {
			retv = StopCriterion::CorrelationIncreaseThreshold;
			break;
		} else if( corr - correlation < mem.filterCorrelationThreshold ) {
			if( mem.filterID != 0 ) {
				this->mainIteration->disableFilter();
				if( this->configuration->notifyOnFilterDisabledEnabled() ) {
					char buff[128];
					sprintf( buff, "filter stage disabled in iteration %i\n", iteration );
					Reporter::inform( buff );
				}

				mem.nFilterIterations = 0;
				mem.filterID = 0;
				continue;
			}
		}


		//break if linesearch did not improve errval
		if( !improved ) {
			retv = StopCriterion::MovementThreshold;
			break;
		} else if( mem.maxIterations == iteration ) {
			retv = StopCriterion::IterationLimit;
			break;
		} else if( correlation >= mem.haltCorrelation ) {
			retv = StopCriterion::HaltCorrelationReached;
			break;
		}

		correlation = corr;
		CudaHelper<T>::copy( mem.pLast, mem.p, mem.nParams );

		if( this->configuration->subFunctionTimingEnabled() ) {
			subFunctionTimer.start();
		}

		//deactivate groups where all frames reached haltCorrelation
		if( this->configuration->maskInactiveGroupsEnabled() ) {
			hd_maskInactiveGroups( mem );
		}

		if( this->configuration->subFunctionTimingEnabled() ) {
			cudaDeviceSynchronize();
			output.functionTimes[TimedMainIteration<T>::nTimedFunctions + 1] += subFunctionTimer.stop();
		}

		if( this->configuration->subFunctionTimingEnabled() ) {
			output.iterationTimes[iteration - 1] = iterationTimer.stop();
		}

		if( this->configuration->saveIterationErrvalsEnabled() ) {
			T* buff = new T[mem.nFrames * 2];
			cudaMemcpy( buff, mem.errval, sizeof( T ) * mem.nFrames * 2, cudaMemcpyDeviceToHost );

			for( int i = 0; i < mem.nFrames; i++ ) {
				output.iterationErrvals( ( iteration - 1 ) * 2, i ) = buff[i];
				output.iterationErrvals( ( iteration - 1 ) * 2 + 1, i ) = buff[i + mem.nFrames];
			}

			delete[] buff;
		}

	} while( true );

	//save the time of the iteration
	if( this->configuration->subFunctionTimingEnabled() ) {
		output.iterationTimes[iteration - 1] = iterationTimer.stop();
	}

	if( iteration == 1 ) {
		CudaHelper<T>::copy( mem.pLast, mem.p, mem.nParams );
	}

	//get subfunction times from mainiteration
	this->mainIteration->copyOutput( output );

	//copy data from device to output struct
	hd_calculateGroupErrvals( mem, 0 );

	//make sure frame correlations are calculated at least once
	if( !this->configuration->saveFrameCorrelationsEnabled() ) {
		hd_frameCorrelation( mem );
	}

	output.iterations = iteration;
	cudaMemcpy( output.finalFrameCorrelations.data(), mem.corr, sizeof( T ) * mem.nFrames, cudaMemcpyDeviceToHost );
	cudaMemcpy( output.errval.data(), mem.errval, sizeof( T ) * mem.nFrames, cudaMemcpyDeviceToHost );
	for( int i = 0; i < mem.nFrames; i++ ) output.errval[i] = sqrt( output.errval.data()[i] );

	//convert mask from row major to col major
	bool* bbuff = new bool[mem.fframeW * mem.frameH];
	cudaMemcpy( bbuff, mem.mask, sizeof( bool ) * mem.fframeW * mem.frameH, cudaMemcpyDeviceToHost );

	output.mask = NeutralMatrix<bool, int, RowMajor<int>>( bbuff, mem.frameH, mem.fframeW, false );

	delete[] bbuff;

	output.rgParams = NeutralMatrix<T, int, ColMajor<int>>( mem.h_rgParams, mem.nParams * mem.maxIterations * 2, 1, false );

	return retv;
}

template class Task<float>;
template class Task<double>;
























