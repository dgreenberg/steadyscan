#ifndef MEMORY_H_
#define MEMORY_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\cudaincludes.h"
#else
#include "../cudaincludes.h"
#endif

#include "neutralmatrix/neutralmatrix.h"

template<typename T>
struct D_Sparse {
	T* values;
	int* rowInd;
	int* colInd;
	int* csrRowInd;

	int nnz;
	int m, n;

	T* intermediateBuffer;
};

template<typename T>
struct ReadOnlyVariables {
	int frameH;
	int frameW;
	int frameW2;
	int templateH;
	int templateW;
	int fframeW;
	int nParamPoints;
	int nParams;
	int nFrames;

	int groupSize;
	int groupSize2;
	int nGroups;

	int maxIterations;
	T moveThreshold;
	T haltCorrelation;

	int nFilterIterations;
	T filterStrength;
	T filterCorrelationThreshold;
	T correlationIncreaseThreshold;

	T lambda;
	T linesearchReductionMultiplier;
	T errvalThreshold;

	int solverID;
	int filterID;
	int gpuDeviceID;

#ifndef MINPOINTSPERBLOCK_AS_VECTOR
	int minPointsPerBlock;
#endif
};

template<typename T>
struct PreAllocatedMemory : ReadOnlyVariables<T> {
	NeutralMatrix<T, int, RowMajor<int>> templateImage;
	NeutralMatrix<T, int, RowMajor<int>> xGradients;
	NeutralMatrix<T, int, RowMajor<int>> yGradients;

	NeutralMatrix<T, int, RowMajor<int>> frac;

	NeutralMatrix<T, int, RowMajor<int>> baseX;
	NeutralMatrix<T, int, RowMajor<int>> baseY;

	NeutralMatrix<T, int, RowMajor<int>> image;
	NeutralMatrix<T> initParameters;
	NeutralMatrix<T> initParameterX;
	NeutralMatrix<T> initParameterY;

#ifdef MINPOINTSPERBLOCK_AS_VECTOR
	NeutralMatrix<int, int, RowMajor<int>> minPointsPerBlock;
#endif
	NeutralMatrix<bool, int, RowMajor<int>> baseMask;
};

template<typename T>
struct DeviceMemory : ReadOnlyVariables<T> {
	DeviceMemory<T>* d_mem;

	T* templateImage;
	T* xGradients;
	T* yGradients;

	T* frac;

	T* baseX;
	T* baseY;

	bool* baseMask;

#ifdef MINPOINTSPERBLOCK_AS_VECTOR
	int* minPointsPerBlock;
#endif

	T* deltaP;
	T* pnew;
	T* x;
	T* y;
	bool* mask;
	//T* wI;
	T* differenceImage;
	T* jtr;

	T* newDeltaP;
	bool* paramsUsed;
	bool* blocksPresent;

	bool* anyBlocksPresent;
	T* corrNew;
	T* corr;

	int nParamsUsed;

	T* wxgrad;
	T* wygrad;

	T* errval;
	T* h_errval;
	bool* movementAboveThreshold; 
	bool* improved;

	bool* groupActive;

	T* p;
	T* pLast;

	T* image;

	int* subSparseOffsets;

	D_Sparse<T> H;


	cusparseHandle_t cuSparseHandle;
	cusolverSpHandle_t cuSolverHandle;
	cusparseMatDescr_t cuSparseDescription;

	T* qcAssq, *qcBssq, *qcq, *qcImg;

	T* diagonalSortedValues;
	int* diagonalSortedIndices;
	int* diagonalIndices;

	T* h_penaltyBuffer;
	T* h_slvValues;
	T* h_slvNewDeltaP;
	T* h_slvJTr;
	int* h_slvRowInd;
	int* h_slvColInd;
	T* h_rgParams;

	bool* h_paramsUsedBuffer;
	int* h_subSparseOffsetBuffer;
};


#endif /* MEMORY_H_ */
