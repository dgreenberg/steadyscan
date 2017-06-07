#ifndef TASK_ORGANIZEPARAMETERS_H_
#define TASK_ORGANIZEPARAMETERS_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\interface\\neutralmatrix\\neutralmatrix.h"
#else
#include "../../interface/neutralmatrix/neutralmatrix.h"
#endif

//converts the parameter matrices between singleframe and multiframe format
template<class T>
class ParameterOrganizer {
public:
	static bool organize2D1( NeutralMatrix<T>& pxy_in, NeutralMatrix<T, int, RowMajor<int>>& p_out, int ctFrames, int frameW, int nParamPoints );
	static bool organize2D1( NeutralMatrix<T>& px_in, NeutralMatrix<T>& py_in, NeutralMatrix<T, int, RowMajor<int>>& p_out, int ctFrames, int frameW, int nParamPoints );

	static bool organize1D2( NeutralMatrix<T>& p_in, NeutralMatrix<T>& pxy_out, int ctFrames, int frameW, int nParamPoints );
	static bool organize1D2( NeutralMatrix<T>& p_in, NeutralMatrix<T>& px_out, NeutralMatrix<T>& py_out, int ctFrames, int frameW, int nParamPoints );
};

#endif
