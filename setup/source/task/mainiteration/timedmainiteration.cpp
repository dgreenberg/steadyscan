#include "timedmainiteration.h"


#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\task_calcerrval.cuh"
#include "..\\task_updategradients.cuh"
#include "..\\task_generatesubsparseoffset.cuh"
#include "..\\task_filldiagonals.cuh"
#include "..\\task_filljtr.cuh"
#include "..\\task_expandresults.cuh"
#include "..\\task_linesearch.cuh"
#include "..\\task_applyparameters_hd.cuh"
#include "..\\..\\helper\\cudahelper\\cudahelper.cuh"
#else
#include "../task_calcerrval.cuh"
#include "../task_updategradients.cuh"
#include "../task_generatesubsparseoffset.cuh"
#include "../task_filldiagonals.cuh"
#include "../task_filljtr.cuh"
#include "../task_expandresults.cuh"
#include "../task_linesearch.cuh"
#include "../task_applyparameters_hd.cuh"
#include "../../helper/cudahelper/cudahelper.cuh"
#endif

#ifdef MEXINCLUDE
#include MEXINCLUDE
#endif

template<class T>
TimedMainIteration<T>::TimedMainIteration( ISolver<T>* solver, IFilter<T>* filter2, Configuration* config ) : MainIteration<T>( solver, filter2, config ) {
	this->functionTimes = new T[TimedMainIteration<T>::nTimedFunctions];
	for( int i = 0; i < TimedMainIteration<T>::nTimedFunctions; i++ ) {
		this->functionTimes[i] = 0;
	}

}
template<class T>
TimedMainIteration<T>::~TimedMainIteration() {
	if( nullptr != this->functionTimes ) {
		delete[] this->functionTimes;
		this->functionTimes = nullptr;
	}
}

template<class T>
bool TimedMainIteration<T>::execute( DeviceMemory<T>& mem, int iteration ) {

	bool improved = true;
	for( int shift = 0; shift < 2; shift++ ) {
		this->timer.start();

		CudaHelper<bool>::setArray( mem.blocksPresent, false, mem.nParamPoints );
		CudaHelper<bool>::setArray( mem.paramsUsed, false, mem.nParamPoints );
		CudaHelper<bool>::setArray( mem.mask, false, mem.fframeW * mem.frameH );
		cudaDeviceSynchronize();

		this->functionTimes[0] += this->timer.stop();
		this->timer.start();

		hd_applyParameters( mem );
		cudaDeviceSynchronize();

		this->functionTimes[1] += this->timer.stop();
		this->timer.start();

		hd_calculateGroupErrvals( mem, shift );
		cudaDeviceSynchronize();

		this->functionTimes[2] += this->timer.stop();
		this->timer.start();

		hd_updateGradients( mem );
		cudaDeviceSynchronize();

		this->functionTimes[3] += this->timer.stop();
		this->timer.start();

		hd_generateSubSparseOffsets( mem );
		cudaDeviceSynchronize();

		this->functionTimes[4] += this->timer.stop();
		this->timer.start();

		hd_fillDiagonals( mem );
		cudaDeviceSynchronize();

		this->functionTimes[5] += this->timer.stop();
		this->timer.start();

		hd_fillJtr( mem );
		cudaDeviceSynchronize();

		this->functionTimes[6] += this->timer.stop();
		this->timer.start();

		if( mem.nFilterIterations != 0 && iteration > mem.nFilterIterations ) {
			this->disableFilter();
			if( this->configuration->notifyOnFilterDisabledEnabled() ) {
				char buff[128];
				sprintf( buff, "filter stage disabled in iteration %i\n", iteration );
				Reporter::inform( buff );
			}
			mem.nFilterIterations = 0;
		}

		this->solver->solve( mem );
		this->filter->apply( mem.newDeltaP, mem.nParamsUsed / 2 );
		this->filter->apply( &mem.newDeltaP[mem.nParamsUsed / 2], mem.nParamsUsed / 2 );
		cudaDeviceSynchronize();

		this->functionTimes[7] += this->timer.stop();
		this->timer.start();

		hd_expandResults( mem, shift );
		cudaDeviceSynchronize();

		this->functionTimes[8] += this->timer.stop();
		this->timer.start();

		improved &= hd_linesearch( mem, shift );
		cudaDeviceSynchronize();

		cudaMemcpy( &mem.h_rgParams[mem.nParams * ( shift + 2 * ( iteration - 1 ) )], mem.p, sizeof( T ) * mem.nParams, cudaMemcpyDeviceToHost );

		this->functionTimes[9] += this->timer.stop();
	}
	if( !improved ) {
		if( mem.nFilterIterations != 0 ) {
			if( this->configuration->notifyOnFilterDisabledEnabled() ) {
				char buff[128];
				sprintf( buff, "filter stage disabled in iteration %i\n", iteration );
				Reporter::inform( buff );
			}
			this->disableFilter();
			mem.filterID = 0;
			mem.nFilterIterations = 0;

			//run this again without filter
			return this->execute( mem, iteration + 1 );
		} else return false;
	}

	return true;
}

template<class T>
void TimedMainIteration<T>::copyOutput( TaskOutput<T>& out ) {
	for( int i = 0; i < TimedMainIteration<T>::nTimedFunctions; i++ ) {
		out.functionTimes[i] = this->functionTimes[i];
	}
}

template class TimedMainIteration<float>;
template class TimedMainIteration<double>;
