#ifndef CUSOLVERTEMPLATES_CUH_
#define CUSOLVERTEMPLATES_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\cudaincludes.h"
#else
#include "../../cudaincludes.h"
#endif

template<class T>
cusolverStatus_t cusolverSpTcsrlsvchol(cusolverSpHandle_t handle,
	    int m,
	    int nnz,
	    const cusparseMatDescr_t descrA,
	    const T *csrVal,
	    const int *csrRowPtr,
	    const int *csrColInd,
	    const T *b,
	    T tol,
	    int reorder,
	    T *x,
	    int *singularity) {
	return cusolverStatus_t::CUSOLVER_STATUS_MAPPING_ERROR;
}

template<>
cusolverStatus_t cusolverSpTcsrlsvchol(
	    cusolverSpHandle_t handle,
	    int m,
	    int nnz,
	    const cusparseMatDescr_t descrA,
	    const double *csrVal,
	    const int *csrRowPtr,
	    const int *csrColInd,
	    const double *b,
	    double tol,
	    int reorder,
	    double *x,
	    int *singularity) {
	return cusolverSpDcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}
template<>
cusolverStatus_t cusolverSpTcsrlsvchol(
	    cusolverSpHandle_t handle,
	    int m,
	    int nnz,
	    const cusparseMatDescr_t descrA,
	    const float *csrVal,
	    const int *csrRowPtr,
	    const int *csrColInd,
	    const float *b,
	    float tol,
	    int reorder,
	    float *x,
	    int *singularity) {
	return cusolverSpScsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

#endif
