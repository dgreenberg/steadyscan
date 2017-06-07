#include "mainiteration.h"

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

#include "../../helper/helper.h"

template<class T>
MainIteration<T>::MainIteration( ISolver<T>* solver, IFilter<T>* filter2, Configuration* config ) {
	this->configuration = config;

	this->solver = solver;
	if( nullptr == filter2 ) {
		this->filter = NoFilter<T>::Static();
	} else {
		this->filter = filter2;
	}
}

template<class T>
MainIteration<T>::~MainIteration() {
	if( nullptr != this->solver ) {
		zDelete( this->solver );
	}

	if( nullptr != this->filter ) {
		if( this->filter != NoFilter<T>::Static() ) {
			delete this->filter;
			this->filter = nullptr;
		}
	}

	return;
}

template<class T>
void MainIteration<T>::disableFilter() {
	IFilter<T>* previousFilter = this->solver->setFilter( NoFilter<T>::Static() );
	if( previousFilter != NoFilter<T>::Static() ) delete previousFilter;
}

template<class T>
bool MainIteration<T>::execute( DeviceMemory<T>& mem, int iteration ) {
	bool improved = true;

	for( int shift = 0; shift < 2; shift++ ) {
		CudaHelper<bool>::setArray( mem.blocksPresent, false, mem.nParamPoints );

		hd_applyParameters( mem );

		hd_calculateGroupErrvals( mem, shift );

		hd_updateGradients( mem );

		hd_generateSubSparseOffsets( mem );

		hd_fillDiagonals( mem );

		hd_fillJtr( mem );

		if( mem.nFilterIterations != 0 && iteration > mem.nFilterIterations ) {
			if( this->configuration->notifyOnFilterDisabledEnabled() ) {
				char buff[128];
				sprintf( buff, "filter stage disabled in iteration %i\n", iteration );
				Reporter::inform( buff );
			}

			this->disableFilter();
			mem.filterID = 0;
			mem.nFilterIterations = 0;
		}

		this->solver->solve( mem );

		this->filter->apply( mem.newDeltaP, mem.nParamsUsed / 2 );
		this->filter->apply( &mem.newDeltaP[mem.nParamsUsed / 2], mem.nParamsUsed / 2 );


		hd_expandResults( mem, shift );

		improved &= hd_linesearch( mem, shift );
	}

	if( !improved ) {
		if( mem.nFilterIterations != 0 ) {
			if( this->configuration->notifyOnFilterDisabledEnabled() ) {
				char buff[128];
				sprintf( buff, "filter stage disabled in iteration %i\n", iteration );
				Reporter::inform( buff );
			}
			this->disableFilter();
			mem.nFilterIterations = 0;

			//run this again without filter
			return this->execute(mem, iteration);
		} else return false;
	}

	return true;
}

template<class T>
void MainIteration<T>::copyOutput( TaskOutput<T>& out ) {
	return;
}

template class MainIteration<float>;
template class MainIteration<double>;
