#include "eigensolve.h"

#include EIGENINCLUDE

template<typename T>
int h_EigenSolve( T* values, int* rowInd, int* colInd, T* jtr, int ctParamsUsed, int nValues, T* result ) {
	static Eigen::SparseMatrix<T> subSparse;
	static int subSparseSize = -1;
	int nnz = 0;

	if( subSparseSize != ctParamsUsed ) {
		subSparse = Eigen::SparseMatrix<T>( ctParamsUsed, ctParamsUsed );
		subSparse.reserve( Eigen::VectorXi::Constant( ctParamsUsed, 6 ) );
		subSparseSize = ctParamsUsed;

		for( int i = 0; i < nValues; i++ ) {
			if( rowInd[i] != INT_MAX ) {
				subSparse.insert( rowInd[i], colInd[i] ) = values[i];
				nnz++;
			}
		}

		subSparse.makeCompressed();
	} else {
		for( int i = 0; i < nValues; i++ ) {
			if( rowInd[i] != INT_MAX ) {
				subSparse.coeffRef( rowInd[i], colInd[i] ) = values[i];
				nnz++;
			}
		}

		subSparse.makeCompressed();
	}

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> solver( subSparse );
	solver.compute( subSparse );

	Eigen::Map<Eigen::Matrix<T, -1, 1>>( result, ctParamsUsed, 1 ) = solver.solve(( Eigen::Matrix<T, -1, 1> ) Eigen::Map<Eigen::Matrix<T, -1, 1>>( jtr, ctParamsUsed, 1 ) );

	return nnz;
}

template int h_EigenSolve( float* values, int* rowInd, int* colInd, float* jtr, int ctParamsUsed, int nnz, float* result );
template int h_EigenSolve( double* values, int* rowInd, int* colInd, double* jtr, int ctParamsUsed, int nnz, double* result );
