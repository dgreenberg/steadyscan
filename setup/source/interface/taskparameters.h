#ifndef TASKPARAMETERS_H_
#define TASKPARAMETERS_H_

#include "neutralmatrix/neutralmatrix.h"

//output type for the Task class
//format defined by the MATLAB lhs
template<class T>
struct TaskOutput {
	int iterations;
	T deviceInitTime;

	//default
	NeutralMatrix<T> errval;
	NeutralMatrix<bool> mask;
	NeutralMatrix<T> blockParametersY;
	NeutralMatrix<T> blockParametersX;
	NeutralMatrix<T> frameParameters;
	NeutralMatrix<T> finalFrameCorrelations;

	//additional return values
	NeutralMatrix<T> iterationCorrelations;
	NeutralMatrix<T> iterationErrvals;
	NeutralMatrix<T> frameCorrelations;
	NeutralMatrix<T> functionTimes;
	NeutralMatrix<T> iterationTimes;
	NeutralMatrix<T> rawParameters;

	NeutralMatrix<T, int, ColMajor<int>> rgParams;
	NeutralMatrix<T, int, ColMajor<int>> allParams;
};

#endif /* TASKPARAMETERS_H_ */
