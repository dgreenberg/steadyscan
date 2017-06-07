#include "parameterorganizer.h"

template<class T>
bool ParameterOrganizer<T>::organize2D1( NeutralMatrix<T>& pxy_in, NeutralMatrix<T, int, RowMajor<int>>& p_out, int ctFrames, int frameW, int nParamPoints ) {
	NeutralMatrix<T> retv( nParamPoints * 2, 1 );

	int d2Index = 0;
	for( int frame = 0; frame < ctFrames; frame++ ) {
		int j = frame * frameW + ( frame != 0 );

		for( int i = ( frame != 0 ); i < frameW; i++, j++ ) {
			retv[j] = pxy_in(frame, i);
			retv[j + nParamPoints] = pxy_in(frame, i + frameW + 1);
		}

		//average last from current frame and first from next frame
		//(except on last frame)
		if( frame < ctFrames - 1 ) {
			retv[j] = ( pxy_in( frame, frameW) + pxy_in( frame + 1, 0 ) ) / 2;
			retv[j + nParamPoints] = ( pxy_in( frame, frameW + frameW + 1) + pxy_in( frame + 1, frameW + 1 ) ) / 2;
		} else {
			retv[j] = pxy_in( frame, frameW );
			retv[j + nParamPoints] = pxy_in( frame, frameW + frameW + 1 );
		}
	}

	p_out = retv;
	return true;
}
template<class T>
bool ParameterOrganizer<T>::organize2D1( NeutralMatrix<T>& px_in, NeutralMatrix<T>& py_in, NeutralMatrix<T, int, RowMajor<int>>& p_out, int ctFrames, int frameW, int nParamPoints ) {
	NeutralMatrix<T, int, RowMajor<int>> retv( nParamPoints * 2, 1 );

	for( int frame = 0; frame < ctFrames; frame++ ) {
		int j = frame * frameW + ( frame != 0 );

		for( int i = ( frame != 0 ); i < frameW; i++, j++ ) {
			retv[j] = px_in[frame * ( frameW + 1 ) + i];
			retv[j + nParamPoints] = py_in[frame * ( frameW + 1 ) + i];
		}

		//average last from current frame and first from next frame
		//(except on last frame)
		if( frame < ctFrames - 1 ) {
			retv[j] = ( px_in[frame * ( frameW + 1 ) + frameW] + px_in[( frame + 1 ) * ( frameW + 1 ) + 0] ) / 2;
			retv[j + nParamPoints] = ( py_in[frame * ( frameW + 1 ) + frameW] + py_in[( frame + 1 ) * ( frameW + 1 ) + 0] ) / 2;
		} else {
			retv[j] = px_in[frame * ( frameW + 1 ) + frameW];
			retv[j + nParamPoints] = py_in[frame * ( frameW + 1 ) + frameW];
		}
	}

	p_out = retv;
	return true;
}

template<class T>
bool ParameterOrganizer<T>::organize1D2( NeutralMatrix<T>& p_in, NeutralMatrix<T>& pxy_out, int ctFrames, int frameW, int nParamPoints ) {

	NeutralMatrix<T> retv( ctFrames, ( frameW + 1 ) * 2 );

	for( int f = 0; f < ctFrames; f++ ) {
		int fi = f * frameW;

		int i;
		for( i = fi; i < fi + frameW; i++ ) {
			retv( f, i - fi ) = p_in[i];
			retv( f, i - fi + frameW + 1 ) = p_in[i + nParamPoints];
		}

		//repeat last frame parameter into first parameter of next frame
		retv( f, i - fi ) = p_in[i - 1];
		retv( f, i - fi + frameW + 1 ) = p_in[i - 1 + nParamPoints];
	}

	pxy_out = retv;
	return true;
}
template<class T>
bool ParameterOrganizer<T>::organize1D2( NeutralMatrix<T>& p_in, NeutralMatrix<T>& px_out, NeutralMatrix<T>& py_out, int ctFrames, int frameW, int nParamPoints ) {
	int ct = 0;

	for( int f = 0; f < ctFrames; f++ ) {
		int fi = f * frameW;

		int i;
		for( i = fi; i < fi + frameW; i++ ) {
			px_out[ct] = p_in[i];
			py_out[ct] = p_in[i+nParamPoints];
			ct++;
		}

		px_out[ct] = p_in[i-1];
		py_out[ct] = p_in[i-1+nParamPoints];
		ct++;
	}

	return true;
}

template class ParameterOrganizer<bool>;
template class ParameterOrganizer<char>;
template class ParameterOrganizer<short>;
template class ParameterOrganizer<int>;
template class ParameterOrganizer<long long>;
template class ParameterOrganizer<float>;
template class ParameterOrganizer<double>;
