#include "cusparsesolver.cuh"

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\helper\\minmax.h"
#include "..\\..\\cudaincludes.h"
#include "..\\..\\helper\\cudahelper\\cudahelper.cuh"
#include "..\\..\\helper\\cudahelper\\cusparseTemplates.cuh"
#include "..\\filter\\allfilter.h"
#include "..\\..\\mex\\reporter\\reporter.h"
#else
#include "../../helper/minmax.h"
#include "../../includes.h"
#include "../../cudaincludes.h"
#include "../../helper/cudahelper/cudahelper.cuh"
#include "../../helper/cudahelper/cusparsetemplates.cuh"
#include "../filter/allfilter.h"
#include "../../mex/reporter/reporter.h"
#endif

template<class T>
SolverCuSparse<T>::SolverCuSparse( IFilter<T>* filter ) : ISolver<T>() {
	if( filter == nullptr ) filter = NoFilter<T>::Static();
	this->filter = filter;
}
template<class T>
SolverCuSparse<T>::~SolverCuSparse() {
	if( nullptr == this->filter ) return;
	if( this->filter == NoFilter<T>::Static() ) return;

	delete this->filter;
}

template<class T>
IFilter<T>* SolverCuSparse<T>::setFilter( IFilter<T>* filter ) {
	if( nullptr == filter ) return filter;
	IFilter<T>* retv = this->filter;
	this->filter = filter;
	return retv;
}

template<class T>
void SolverCuSparse<T>::solve( DeviceMemory<T>& mem ) {
	DeviceMemory<T> d_mem;
	CUCHECKRC( cudaMemcpy( &d_mem, mem.d_mem, sizeof( DeviceMemory<T> ), cudaMemcpyDeviceToHost ) );

	int nnz = d_mem.H.nnz;
	int m = d_mem.nParamsUsed;

	//convert coo to csr
	int* d_csrRowInd = mem.H.csrRowInd;
	int* d_csrColInd = mem.H.colInd;
	T* d_csrVal = mem.H.values;
	T* d_x = mem.jtr;
	T* d_y = mem.newDeltaP;
	T* d_z = mem.H.intermediateBuffer;

	CUCHECKRC( cusparseXcoo2csr( mem.cuSparseHandle, mem.H.rowInd, nnz, m, d_csrRowInd, CUSPARSE_INDEX_BASE_ZERO ) );

	// Suppose that A is m x m sparse matrix represented by CSR format,
	// Assumption:
	// - mem.cuSparseHandle is already created by cusparseCreate(),
	// - (d_csrRowInd, d_csrColInd, d_csrVal) is CSR of A on device memory,
	// - d_x is right hand side vector on device memory,
	// - d_y is solution vector on device memory.
	// - d_z is intermediate result on device memory.

	cusparseMatDescr_t descr_M = 0;
	cusparseMatDescr_t descr_L = 0;
	cusparseMatDescr_t descr_U = 0;
	csrilu02Info_t info_M = 0;
	csrsv2Info_t info_L = 0;
	csrsv2Info_t info_U = 0;
	cusparseStatus_t status;
	int pBufferSize_M;
	int pBufferSize_L;
	int pBufferSize_U;
	int pBufferSize;
	void *pBuffer = 0;
	int structural_zero;
	int numerical_zero;
	const T alpha = 1.0;
	const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
	const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

	// step 1: create a descriptor which contains
	// - matrix M is base-0
	// - matrix L is base-0
	// - matrix L is lower triangular
	// - matrix L has unit diagonal
	// - matrix U is base-0
	// - matrix U is upper triangular
	// - matrix U has non-unit diagonal
	CUCHECKRC( cusparseCreateMatDescr( &descr_M ) );
	CUCHECKRC( cusparseSetMatIndexBase( descr_M, CUSPARSE_INDEX_BASE_ZERO ) );
	CUCHECKRC( cusparseSetMatType( descr_M, CUSPARSE_MATRIX_TYPE_GENERAL ) );

	CUCHECKRC( cusparseCreateMatDescr( &descr_L ) );
	CUCHECKRC( cusparseSetMatIndexBase( descr_L, CUSPARSE_INDEX_BASE_ZERO ) );
	CUCHECKRC( cusparseSetMatType( descr_L, CUSPARSE_MATRIX_TYPE_GENERAL ) );
	CUCHECKRC( cusparseSetMatFillMode( descr_L, CUSPARSE_FILL_MODE_LOWER ) );
	CUCHECKRC( cusparseSetMatDiagType( descr_L, CUSPARSE_DIAG_TYPE_UNIT ) );

	CUCHECKRC( cusparseCreateMatDescr( &descr_U ) );
	CUCHECKRC( cusparseSetMatIndexBase( descr_U, CUSPARSE_INDEX_BASE_ZERO ) );
	CUCHECKRC( cusparseSetMatType( descr_U, CUSPARSE_MATRIX_TYPE_GENERAL ) );
	CUCHECKRC( cusparseSetMatFillMode( descr_U, CUSPARSE_FILL_MODE_UPPER ) );
	CUCHECKRC( cusparseSetMatDiagType( descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT ) );

	// step 2: create a empty info structure
	// we need one info for csrilu02 and two info's for csrsv2
	CUCHECKRC( cusparseCreateCsrilu02Info( &info_M ) );
	CUCHECKRC( cusparseCreateCsrsv2Info( &info_L ) );
	CUCHECKRC( cusparseCreateCsrsv2Info( &info_U ) );

	// step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
	CUCHECKRC( cusparseTcsrilu02_bufferSize<T>( mem.cuSparseHandle, m, nnz, descr_M, d_csrVal, d_csrRowInd, d_csrColInd, info_M, &pBufferSize_M ) );
	CUCHECKRC( cusparseTcsrsv2_bufferSize<T>( mem.cuSparseHandle, trans_L, m, nnz, descr_L, d_csrVal, d_csrRowInd, d_csrColInd, info_L, &pBufferSize_L ) );
	CUCHECKRC( cusparseTcsrsv2_bufferSize<T>( mem.cuSparseHandle, trans_U, m, nnz, descr_U, d_csrVal, d_csrRowInd, d_csrColInd, info_U, &pBufferSize_U ) );

	pBufferSize = max( pBufferSize_M, max(pBufferSize_L, pBufferSize_U) );

	// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
	CUCHECKRC( cudaMalloc( &pBuffer, pBufferSize ) );

	// step 4: perform analysis of incomplete Cholesky on M
	//         perform analysis of triangular solve on L
	//         perform analysis of triangular solve on U
	// The lower(upper) triangular part of M has the same sparsity pattern as L(U),
	// we can do analysis of csrilu0 and csrsv2 simultaneously.

	CUCHECKRC( cusparseTcsrilu02_analysis<T>( mem.cuSparseHandle, m, nnz, descr_M, d_csrVal, d_csrRowInd, d_csrColInd, info_M, policy_M, pBuffer ) );
	status = cusparseXcsrilu02_zeroPivot( mem.cuSparseHandle, info_M, &structural_zero );
	if( CUSPARSE_STATUS_ZERO_PIVOT == status ) {
		char buff[128];
		sprintf(buff, "A(%d,%d) is missing\n", structural_zero, structural_zero );
		Reporter::warn(ReportID::CuSparseSolver, buff);
	}

	CUCHECKRC( cusparseTcsrsv2_analysis<T>( mem.cuSparseHandle, trans_L, m, nnz, descr_L, d_csrVal, d_csrRowInd, d_csrColInd, info_L, policy_L, pBuffer ) );
	CUCHECKRC( cusparseTcsrsv2_analysis<T>( mem.cuSparseHandle, trans_U, m, nnz, descr_U, d_csrVal, d_csrRowInd, d_csrColInd, info_U, policy_U, pBuffer ) );

	// step 5: M = L * U
	CUCHECKRC( cusparseTcsrilu02<T>( mem.cuSparseHandle, m, nnz, descr_M, d_csrVal, d_csrRowInd, d_csrColInd, info_M, policy_M, pBuffer ) );
	status = cusparseXcsrilu02_zeroPivot( mem.cuSparseHandle, info_M, &numerical_zero );
	if( CUSPARSE_STATUS_ZERO_PIVOT == status ) {
		char buff[128];
		sprintf(buff, "U(%d,%d) is zero\n", numerical_zero, numerical_zero );
		Reporter::warn(ReportID::CuSparseSolver, buff);
	}

	// step 6: solve L*z = x
	CUCHECKRC( cusparseTcsrsv2_solve<T>( mem.cuSparseHandle, trans_L, m, nnz, &alpha, descr_L, d_csrVal, d_csrRowInd, d_csrColInd, info_L, d_x, d_z, policy_L, pBuffer ) );

	// step 7: solve U*y = z
	CUCHECKRC( cusparseTcsrsv2_solve<T>( mem.cuSparseHandle, trans_U, m, nnz, &alpha, descr_U, d_csrVal, d_csrRowInd, d_csrColInd, info_U, d_z, d_y, policy_U, pBuffer ) );

	// step 8: free resources
	CUCHECKRC( cusparseDestroyMatDescr( descr_M ) );
	CUCHECKRC( cusparseDestroyMatDescr( descr_L ) );
	CUCHECKRC( cusparseDestroyMatDescr( descr_U ) );

	CUCHECKRC( cusparseDestroyCsrilu02Info( info_M ) );
	CUCHECKRC( cusparseDestroyCsrsv2Info( info_L ) );
	CUCHECKRC( cusparseDestroyCsrsv2Info( info_U ) );

	CUCHECKRC( cudaFree( pBuffer ) );


	this->filter->applyGPU( mem.newDeltaP, d_mem.nParamsUsed / 2 );
	this->filter->applyGPU( &mem.newDeltaP[d_mem.nParamsUsed / 2], d_mem.nParamsUsed / 2 );
}

template class SolverCuSparse<float>;
template class SolverCuSparse<double>;
