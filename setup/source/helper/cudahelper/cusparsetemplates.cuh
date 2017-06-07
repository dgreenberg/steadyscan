#ifndef CUSPARSETEMPLATES_CUH_
#define CUSPARSETEMPLATES_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\cudaincludes.h"
#else
#include "../../cudaincludes.h"
#endif


template<class T>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrilu02_bufferSize(cusparseHandle_t handle,
        int m,
        int nnz,
        const cusparseMatDescr_t descrA,
        T *csrSortedValA,
        const int *csrSortedRowPtrA,
        const int *csrSortedColIndA,
        csrilu02Info_t info,
        int *pBufferSizeInBytes) {
	return CUSPARSE_STATUS_MAPPING_ERROR;
}
template<>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrilu02_bufferSize(cusparseHandle_t handle,
        int m,
        int nnz,
        const cusparseMatDescr_t descrA,
        double *csrSortedValA,
        const int *csrSortedRowPtrA,
        const int *csrSortedColIndA,
        csrilu02Info_t info,
        int *pBufferSizeInBytes) {
	return cusparseDcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}
template<>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrilu02_bufferSize(cusparseHandle_t handle,
        int m,
        int nnz,
        const cusparseMatDescr_t descrA,
        float *csrSortedValA,
        const int *csrSortedRowPtrA,
        const int *csrSortedColIndA,
        csrilu02Info_t info,
        int *pBufferSizeInBytes) {
	return cusparseScsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

template<class T>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrsv2_bufferSize(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        T *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        int *pBufferSizeInBytes) {
	return CUSPARSE_STATUS_MAPPING_ERROR;
}
template<>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrsv2_bufferSize(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        double *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        int *pBufferSizeInBytes) {
	return cusparseDcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}
template<>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrsv2_bufferSize(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        float *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        int *pBufferSizeInBytes) {
	return cusparseScsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

template<class T>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrilu02_analysis(cusparseHandle_t handle,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        const T *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy,
                                                        void *pBuffer) {
	return CUSPARSE_STATUS_MAPPING_ERROR;
}
template<>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrilu02_analysis(cusparseHandle_t handle,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        const double *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy,
                                                        void *pBuffer) {
	return cusparseDcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}
template<>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrilu02_analysis(cusparseHandle_t handle,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        const float *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrilu02Info_t info,
                                                        cusparseSolvePolicy_t policy,
                                                        void *pBuffer) {
	return cusparseScsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

template<class T>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrsv2_analysis(cusparseHandle_t handle,
                                                      cusparseOperation_t transA,
                                                      int m,
                                                      int nnz,
                                                      const cusparseMatDescr_t descrA,
                                                      const T *csrSortedValA,
                                                      const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA,
                                                      csrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer) {
	return CUSPARSE_STATUS_MAPPING_ERROR;
}
template<>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrsv2_analysis(cusparseHandle_t handle,
                                                      cusparseOperation_t transA,
                                                      int m,
                                                      int nnz,
                                                      const cusparseMatDescr_t descrA,
                                                      const double *csrSortedValA,
                                                      const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA,
                                                      csrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer) {
	return cusparseDcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}
template<>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrsv2_analysis(cusparseHandle_t handle,
                                                      cusparseOperation_t transA,
                                                      int m,
                                                      int nnz,
                                                      const cusparseMatDescr_t descrA,
                                                      const float *csrSortedValA,
                                                      const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA,
                                                      csrsv2Info_t info,
                                                      cusparseSolvePolicy_t policy,
                                                      void *pBuffer) {
	return cusparseScsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

template<class T>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrilu02(cusparseHandle_t handle,
                                               int m,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               T *csrSortedValA_valM,
                                               /* matrix A values are updated inplace
                                                  to be the preconditioner M values */
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info,
                                               cusparseSolvePolicy_t policy,
                                               void *pBuffer) {
	return CUSPARSE_STATUS_MAPPING_ERROR;
}
template<>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrilu02(cusparseHandle_t handle,
                                               int m,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               double *csrSortedValA_valM,
                                               /* matrix A values are updated inplace
                                                  to be the preconditioner M values */
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info,
                                               cusparseSolvePolicy_t policy,
                                               void *pBuffer) {
	return cusparseDcsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}
template<>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrilu02(cusparseHandle_t handle,
                                               int m,
                                               int nnz,
                                               const cusparseMatDescr_t descrA,
                                               float *csrSortedValA_valM,
                                               /* matrix A values are updated inplace
                                                  to be the preconditioner M values */
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info,
                                               cusparseSolvePolicy_t policy,
                                               void *pBuffer) {
	return cusparseScsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

template<class T>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrsv2_solve(cusparseHandle_t handle,
                                                   const cusparseOperation_t transA,
                                                   int m,
                                                   int nnz,
                                                   const T *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const T *csrSortedValA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   csrsv2Info_t info,
                                                   const T *f,
                                                   T *x,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer) {
	return CUSPARSE_STATUS_MAPPING_ERROR;
}
template<>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrsv2_solve(cusparseHandle_t handle,
                                                   const cusparseOperation_t transA,
                                                   int m,
                                                   int nnz,
                                                   const double *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const double *csrSortedValA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   csrsv2Info_t info,
                                                   const double *f,
                                                   double *x,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer) {
	return cusparseDcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer);
}
template<>
inline cusparseStatus_t CUSPARSEAPI cusparseTcsrsv2_solve(cusparseHandle_t handle,
                                                   const cusparseOperation_t transA,
                                                   int m,
                                                   int nnz,
                                                   const float *alpha,
                                                   const cusparseMatDescr_t descrA,
                                                   const float *csrSortedValA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   csrsv2Info_t info,
                                                   const float *f,
                                                   float *x,
                                                   cusparseSolvePolicy_t policy,
                                                   void *pBuffer) {
	return cusparseScsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer);
}




#endif
