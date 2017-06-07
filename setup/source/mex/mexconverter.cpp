#include "mexconverter.h"

#ifdef _MATLAB

template<typename T>
static inline bool MxClassIdMatchesType( mxClassID mx ) {
	switch( mx ) {
	case mxLOGICAL_CLASS:
		return typeid( T ) == typeid( bool );
	case mxCHAR_CLASS:
#if defined( UNICODE ) || defined( _UNICODE )
		return typeid( T ) == typeid( wchar_t );
#else
		return typeid( T ) == typeid( char );
#endif
	case mxDOUBLE_CLASS:
		return typeid( T ) == typeid( double );
	case mxSINGLE_CLASS:
		return typeid( T ) == typeid( float );
	case mxINT8_CLASS:
		return typeid( T ) == typeid( char );
	case mxUINT8_CLASS:
		return typeid( T ) == typeid( unsigned char );
	case mxINT16_CLASS:
		return typeid( T ) == typeid( short );
	case mxUINT16_CLASS:
		return typeid( T ) == typeid( unsigned short );
	case mxINT32_CLASS:
		return typeid( T ) == typeid( int );
	case mxUINT32_CLASS:
		return typeid( T ) == typeid( unsigned int );
	case mxINT64_CLASS:
		return typeid( T ) == typeid( long long int );
	case mxUINT64_CLASS:
		return typeid( T ) == typeid( unsigned long long int );
	default:
		return false;
	}
}
template<typename Tin, typename Tout, class Major>
static ConversionResult ConvertMxToNeutralType( mxArray* mx, NeutralMatrix<Tout, int, Major>& dest ) {
	ConversionResult retv = ConversionResult::Success;

	Tin* mxPtr = reinterpret_cast<Tin*>( mxGetData( mx ) );
	int m = mxGetM( mx );
	int n = mxGetN( mx );

	dest = NeutralMatrix<Tout, int, Major>( m, n );

	for( int j = 0; j < n; j++ ) {
		for( int i = 0; i < m; i++ ) {
			dest(i, j) = static_cast<Tout>( mxPtr[i + j * m] );
		}
	}

	if( ( Tin(-1) < Tin(0) ) ^ (Tout( -1 ) < Tout( 0 ) ) ) retv = retv | ConversionResult::WarningSignedMismatch;
	if( sizeof( Tin ) > sizeof( Tout ) ) retv = retv | ConversionResult::WarningOutputSmallerThanInput;
	if( static_cast<Tin>(1.5) != static_cast<Tout>(1.5)) retv = retv | ConversionResult::WarningDecimalMismatch;
	return retv;
}
template<typename Tin, typename Tout, class Major>
static ConversionResult ConvertNeutralToMx( NeutralMatrix<Tin, int, Major>& mat, mxArray*& dest, mxClassID outputType ) {
	ConversionResult retv = ConversionResult::Success;

	int m = mat.rows();
	int n = mat.cols();

	mxArray* mx = mxCreateNumericMatrix( m, n, outputType, mxREAL );
	Tout* mxPtr = reinterpret_cast<Tout*>( mxGetData( mx ) );
	for( int j = 0; j < n; j++ ) {
		for( int i = 0; i < m; i++ ) {
			mxPtr[i + j * m] = static_cast<Tout>( mat( i, j ) );
		}
	}

	if( ( Tin(-1) < Tin(0) ) ^ (Tout( -1 ) < Tout( 0 ) ) ) retv = retv | ConversionResult::WarningSignedMismatch;
	if( sizeof( Tin ) > sizeof( Tout ) ) retv = retv | ConversionResult::WarningOutputSmallerThanInput;
	if( static_cast<Tin>( 1.5 ) != static_cast<Tout>( 1.5 ) ) retv = retv | ConversionResult::WarningDecimalMismatch;
	return retv;
}
template<typename Tin, typename Tout>
static ConversionResult ConvertScalarToMx( Tin scalar, mxArray*& dest, mxClassID outputType ) {
	ConversionResult retv = ConversionResult::Success;

	dest = mxCreateNumericMatrix( 1, 1, outputType, mxREAL );
	*reinterpret_cast<Tout*>( mxGetData( dest ) ) = static_cast<Tout>( scalar );

	if( ( Tin(-1) < Tin(0) ) ^ (Tout( -1 ) < Tout( 0 ) ) ) retv = retv | ConversionResult::WarningSignedMismatch;
	if( sizeof( Tin ) > sizeof( Tout ) ) retv = retv | ConversionResult::WarningOutputSmallerThanInput;
	if( static_cast<Tin>( 1.5 ) != static_cast<Tout>( 1.5 ) ) retv = retv | ConversionResult::WarningDecimalMismatch;
	return retv;
}
template<typename Tin, typename Tout>
static ConversionResult ConvertMxToScalar( mxArray* mx, Tout& scalar ) {
	ConversionResult retv = ConversionResult::Success;

	scalar = static_cast<Tout>( *reinterpret_cast<Tin*>( mxGetData( mx ) ) );

	if( ( Tin(-1) < Tin(0) ) ^ (Tout( -1 ) < Tout( 0 ) ) ) retv = retv | ConversionResult::WarningSignedMismatch;
	if( sizeof( Tin ) > sizeof( Tout ) ) retv = retv | ConversionResult::WarningOutputSmallerThanInput;
	if( static_cast<Tin>(1.5) != static_cast<Tout>(1.5)) retv = retv | ConversionResult::WarningDecimalMismatch;
	return retv;
}

//c++ -> matlab
//creates a mxn matlab matrix (mxArray*) from an interface matrix (Neutralmatrix<T>)
template<class T>
ConversionResult MexConverter<T>::mxFromNeutral( NeutralMatrix<T, int, RowMajor<int>>& mat, mxArray*& dest, mxClassID outputType, ConversionMode mode ) {
	if(outputType == mxClassID::mxUNKNOWN_CLASS) {
		return mxFromNeutral( mat, dest, getMatlabClassIdFromType(), mode );
	} else if( ::MxClassIdMatchesType<T>( outputType ) ) {
		int m = mat.rows(), n = mat.cols();

		dest = mxCreateNumericMatrix( m, n, outputType, mxREAL );
		T* mxPtr = reinterpret_cast<T*>( mxGetData( dest ) );
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				mxPtr[i + j * m] = mat(i, j);
			}
		}

		return ConversionResult::Success;
	} else if( mode & ConversionMode::ConvertTypesOnMismatch ) {
		dest = mxCreateNumericMatrix( mat.cols(), mat.rows(), outputType, mxREAL );

		switch( outputType ) {
		case mxClassID::mxCHAR_CLASS:
#if defined( UNICODE ) || defined( _UNICODE )
			return ConvertNeutralToMx<T, wchar_t, RowMajor<int>>( mat, dest, outputType );
#else
			return ConvertNeutralToMx<T, char, RowMajor<int>>( mat, dest, outputType );
#endif
		case mxClassID::mxDOUBLE_CLASS:
			return ConvertNeutralToMx<T, double, RowMajor<int>>( mat, dest, outputType );
		case mxClassID::mxSINGLE_CLASS:
			return ConvertNeutralToMx<T, float, RowMajor<int>>( mat, dest, outputType );
		case mxClassID::mxINT8_CLASS:
			return ConvertNeutralToMx<T, char, RowMajor<int>>( mat, dest, outputType );
		case mxClassID::mxINT16_CLASS:
			return ConvertNeutralToMx<T, short, RowMajor<int>>( mat, dest, outputType );
		case mxClassID::mxINT32_CLASS:
			return ConvertNeutralToMx<T, int, RowMajor<int>>( mat, dest, outputType );
		case mxClassID::mxINT64_CLASS:
			return ConvertNeutralToMx<T, long long, RowMajor<int>>( mat, dest, outputType );
		case mxClassID::mxLOGICAL_CLASS:
			return ConvertNeutralToMx<T, bool, RowMajor<int>>( mat, dest, outputType );
		case mxClassID::mxUINT8_CLASS:
			return ConvertNeutralToMx<T, unsigned char, RowMajor<int>>( mat, dest, outputType );
		case mxClassID::mxUINT16_CLASS:
			return ConvertNeutralToMx<T, unsigned short, RowMajor<int>>( mat, dest, outputType );
		case mxClassID::mxUINT32_CLASS:
			return ConvertNeutralToMx<T, unsigned int, RowMajor<int>>( mat, dest, outputType );
		case mxClassID::mxUINT64_CLASS:
			return ConvertNeutralToMx<T, unsigned long long, RowMajor<int>>( mat, dest, outputType );
		default:
			mxFree( dest );
			return ConversionResult::ErrorNotSupported;
		}
	}

	return ConversionResult::ErrorTypeMismatch;
}
template<class T>
ConversionResult MexConverter<T>::mxFromNeutral( NeutralMatrix<T>& mat, mxArray*& dest, mxClassID outputType, ConversionMode mode ) {

	if(outputType == mxClassID::mxUNKNOWN_CLASS) {
		return mxFromNeutral( mat, dest, getMatlabClassIdFromType(), mode );
	} else if( ::MxClassIdMatchesType<T>( outputType ) ) {
		int m = mat.rows(), n = mat.cols();

		dest = mxCreateNumericMatrix( m, n, outputType, mxREAL );
		T* mxPtr = reinterpret_cast<T*>( mxGetData( dest ) );
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				mxPtr[i + j * m] = mat(i, j);
			}
		}

		return ConversionResult::Success;
	} else if( mode & ConversionMode::ConvertTypesOnMismatch ) {
		dest = mxCreateNumericMatrix( mat.cols(), mat.rows(), outputType, mxREAL );

		switch( outputType ) {
		case mxClassID::mxCHAR_CLASS:
#if defined( UNICODE ) || defined( _UNICODE )
			return ConvertNeutralToMx<T, wchar_t, ColMajor<int>>( mat, dest, outputType );
#else
			return ConvertNeutralToMx<T, char, ColMajor<int>>( mat, dest, outputType );
#endif
		case mxClassID::mxDOUBLE_CLASS:
			return ConvertNeutralToMx<T, double, ColMajor<int>>( mat, dest, outputType );
		case mxClassID::mxSINGLE_CLASS:
			return ConvertNeutralToMx<T, float, ColMajor<int>>( mat, dest, outputType );
		case mxClassID::mxINT8_CLASS:
			return ConvertNeutralToMx<T, char, ColMajor<int>>( mat, dest, outputType );
		case mxClassID::mxINT16_CLASS:
			return ConvertNeutralToMx<T, short, ColMajor<int>>( mat, dest, outputType );
		case mxClassID::mxINT32_CLASS:
			return ConvertNeutralToMx<T, int, ColMajor<int>>( mat, dest, outputType );
		case mxClassID::mxINT64_CLASS:
			return ConvertNeutralToMx<T, long long, ColMajor<int>>( mat, dest, outputType );
		case mxClassID::mxLOGICAL_CLASS:
			return ConvertNeutralToMx<T, bool, ColMajor<int>>( mat, dest, outputType );
		case mxClassID::mxUINT8_CLASS:
			return ConvertNeutralToMx<T, unsigned char, ColMajor<int>>( mat, dest, outputType );
		case mxClassID::mxUINT16_CLASS:
			return ConvertNeutralToMx<T, unsigned short, ColMajor<int>>( mat, dest, outputType );
		case mxClassID::mxUINT32_CLASS:
			return ConvertNeutralToMx<T, unsigned int, ColMajor<int>>( mat, dest, outputType );
		case mxClassID::mxUINT64_CLASS:
			return ConvertNeutralToMx<T, unsigned long long, ColMajor<int>>( mat, dest, outputType );
		default:
			mxFree(dest);
			return ConversionResult::ErrorNotSupported;
		}
	}

	return ConversionResult::ErrorTypeMismatch;
}

//c++ <- matlab
//creates an interface matrix (NeutralMatrix<T>) from a matlab matrix (mxArray*)
template<class T>
ConversionResult MexConverter<T>::neutralFromMx( mxArray* mx, NeutralMatrix<T, int, RowMajor<int>>& dest, ConversionMode mode ) {
	mxClassID mxClassId = mxGetClassID( mx );

	if( MxClassIdMatchesType<T>( mxClassId ) ) {
		ConversionResult retv =  ConvertMxToNeutralType<T, T, RowMajor<int>>( mx, dest );
		return retv;
	} else if( mode & ConversionMode::ConvertTypesOnMismatch ) {
		ConversionResult retv = ConversionResult::Success;

		if( mode & ConversionMode::ShareData ) {
			retv = retv | ConversionResult::WarningDataNotShared;
		}
		switch( mxClassId ) {
		case mxClassID::mxCHAR_CLASS:
#if defined( UNICODE ) || defined( _UNICODE )
			retv = retv | ConvertMxToNeutralType<wchar_t, T, RowMajor<int>>( mx, dest );
#else
			retv = retv | ConvertMxToNeutralType<char, T, RowMajor<int>>( mx, dest );
#endif
			break;
		case mxClassID::mxDOUBLE_CLASS:
			retv = retv | ConvertMxToNeutralType<double, T, RowMajor<int>>( mx, dest );
			break;
		case mxClassID::mxSINGLE_CLASS:
			retv = retv | ConvertMxToNeutralType<float, T, RowMajor<int>>( mx, dest );
			break;
		case mxClassID::mxINT8_CLASS:
			retv = retv | ConvertMxToNeutralType<char, T, RowMajor<int>>( mx, dest );
			break;
		case mxClassID::mxINT16_CLASS:
			retv = retv | ConvertMxToNeutralType<short, T, RowMajor<int>>( mx, dest );
			break;
		case mxClassID::mxINT32_CLASS:
			retv = retv | ConvertMxToNeutralType<int, T, RowMajor<int>>( mx, dest );
			break;
		case mxClassID::mxINT64_CLASS:
			retv = retv | ConvertMxToNeutralType<long long, T, RowMajor<int>>( mx, dest );
			break;
		case mxClassID::mxLOGICAL_CLASS:
			retv = retv | ConvertMxToNeutralType<bool, T, RowMajor<int>>( mx, dest );
			break;
		case mxClassID::mxUINT8_CLASS:
			retv = retv | ConvertMxToNeutralType<unsigned char, T, RowMajor<int>>( mx, dest );
			break;
		case mxClassID::mxUINT16_CLASS:
			retv = retv | ConvertMxToNeutralType<unsigned short, T, RowMajor<int>>( mx, dest );
			break;
		case mxClassID::mxUINT32_CLASS:
			retv = retv | ConvertMxToNeutralType<unsigned int, T, RowMajor<int>>( mx, dest );
			break;
		case mxClassID::mxUINT64_CLASS:
			retv = retv | ConvertMxToNeutralType<unsigned long long, T, RowMajor<int>>( mx, dest );
			break;
		default:
			return ConversionResult::ErrorNotSupported;
		}

		return retv;
	}

	return ConversionResult::ErrorTypeMismatch;
}

template<class T>
ConversionResult MexConverter<T>::neutralFromMx( mxArray* mx, NeutralMatrix<T>& dest, ConversionMode mode ) {
	mxClassID mxClassId = mxGetClassID( mx );

	if( MxClassIdMatchesType<T>( mxClassId ) ) {
		if( mode & ConversionMode::ShareData ) {
			dest = NeutralMatrix<T>( reinterpret_cast<T*>( mxGetData( mx ) ), mxGetM( mx ), mxGetN( mx ), false );
			return ConversionResult::Success;
		}

		return ConvertMxToNeutralType<T, T, ColMajor<int>>( mx, dest );
	} else if( mode & ConversionMode::ConvertTypesOnMismatch ) {
		ConversionResult retv = ConversionResult::Success;

		if( mode & ConversionMode::ShareData ) {
			retv = retv | ConversionResult::WarningDataNotShared;
		}

		switch( mxClassId ) {
		case mxClassID::mxCHAR_CLASS:
#if defined( UNICODE ) || defined( _UNICODE )
			retv = retv | ConvertMxToNeutralType<wchar_t, T, ColMajor<int>>( mx, dest );
#else
			retv = retv | ConvertMxToNeutralType<char, T, ColMajor<int>>( mx, dest );
#endif
			break;
		case mxClassID::mxDOUBLE_CLASS:
			retv = retv | ConvertMxToNeutralType<double, T, ColMajor<int>>( mx, dest );
			break;
		case mxClassID::mxSINGLE_CLASS:
			retv = retv | ConvertMxToNeutralType<float, T, ColMajor<int>>( mx, dest );
			break;
		case mxClassID::mxINT8_CLASS:
			retv = retv | ConvertMxToNeutralType<char, T, ColMajor<int>>( mx, dest );
			break;
		case mxClassID::mxINT16_CLASS:
			retv = retv | ConvertMxToNeutralType<short, T, ColMajor<int>>( mx, dest );
			break;
		case mxClassID::mxINT32_CLASS:
			retv = retv | ConvertMxToNeutralType<int, T, ColMajor<int>>( mx, dest );
			break;
		case mxClassID::mxINT64_CLASS:
			retv = retv | ConvertMxToNeutralType<long long, T, ColMajor<int>>( mx, dest );
			break;
		case mxClassID::mxLOGICAL_CLASS:
			retv = retv | ConvertMxToNeutralType<bool, T, ColMajor<int>>( mx, dest );
			break;
		case mxClassID::mxUINT8_CLASS:
			retv = retv | ConvertMxToNeutralType<unsigned char, T, ColMajor<int>>( mx, dest );
			break;
		case mxClassID::mxUINT16_CLASS:
			retv = retv | ConvertMxToNeutralType<unsigned short, T, ColMajor<int>>( mx, dest );
			break;
		case mxClassID::mxUINT32_CLASS:
			retv = retv | ConvertMxToNeutralType<unsigned int, T, ColMajor<int>>( mx, dest );
			break;
		case mxClassID::mxUINT64_CLASS:
			retv = retv | ConvertMxToNeutralType<unsigned long long, T, ColMajor<int>>( mx, dest );
			break;
		default:
			return ConversionResult::ErrorNotSupported;
		}

		return retv;
	}

	return ConversionResult::ErrorTypeMismatch;
}

template<class T>
ConversionResult MexConverter<T>::mxFromScalar( T scalar, mxArray*& dest, mxClassID outputType, ConversionMode mode ) {
	if( MxClassIdMatchesType<T>( outputType ) ) {
		dest = mxCreateNumericMatrix( 1, 1, outputType, mxREAL );
		*reinterpret_cast<T*>( mxGetData( dest ) ) = scalar;
		return ConversionResult::Success;
	} else if( mode & ConversionMode::ConvertTypesOnMismatch ) {
		//if outputType is unknown, try to find a class id from class template
		if( outputType == mxClassID::mxUNKNOWN_CLASS ) {
			outputType = MexConverter<T>::getMatlabClassIdFromType();
		}

		switch( outputType ) {
		case mxClassID::mxCHAR_CLASS:
#if defined( UNICODE ) || defined( _UNICODE )
		return ConvertScalarToMx<T, wchar_t>( scalar, dest, outputType );
#else
		return ConvertScalarToMx<T, char>( scalar, dest, outputType );
#endif
		case mxClassID::mxLOGICAL_CLASS:
			return ConvertScalarToMx<T, bool>( scalar, dest, outputType );
		case mxClassID::mxSINGLE_CLASS:
			return ConvertScalarToMx<T, float>( scalar, dest, outputType );
		case mxClassID::mxDOUBLE_CLASS:
			return ConvertScalarToMx<T, double>( scalar, dest, outputType );
		case mxClassID::mxINT8_CLASS:
			return ConvertScalarToMx<T, char>( scalar, dest, outputType );
		case mxClassID::mxINT16_CLASS:
			return ConvertScalarToMx<T, short>( scalar, dest, outputType );
		case mxClassID::mxINT32_CLASS:
			return ConvertScalarToMx<T, int>( scalar, dest, outputType );
		case mxClassID::mxINT64_CLASS:
			return ConvertScalarToMx<T, long long>( scalar, dest, outputType );
		case mxClassID::mxUINT8_CLASS:
			return ConvertScalarToMx<T, unsigned char>( scalar, dest, outputType );
		case mxClassID::mxUINT16_CLASS:
			return ConvertScalarToMx<T, unsigned short>( scalar, dest, outputType );
		case mxClassID::mxUINT32_CLASS:
			return ConvertScalarToMx<T, unsigned int>( scalar, dest, outputType );
		case mxClassID::mxUINT64_CLASS:
			return ConvertScalarToMx<T, unsigned long long>( scalar, dest, outputType );
		default:
			return ConversionResult::ErrorNotSupported;
		}

	}

	return ConversionResult::ErrorTypeMismatch;
}

template<class T>
ConversionResult MexConverter<T>::scalarFromMx( mxArray* mx, T& dest, ConversionMode mode ) {
	if( MxClassIdMatchesType<T>( mxGetClassID( mx ) ) ) {
		return ConvertMxToScalar<T, T>( mx, dest );
	} else if( mode & ConversionMode::ConvertTypesOnMismatch ) {
		switch( mxGetClassID( mx ) ) {
		case mxClassID::mxCHAR_CLASS:
#if defined( UNICODE ) || defined( _UNICODE )
		return ConvertMxToScalar<wchar_t, T>( mx, dest );
#else
		return ConvertMxToScalar<char, T>( mx, dest );
#endif
		case mxClassID::mxLOGICAL_CLASS:
			return ConvertMxToScalar<bool, T>( mx, dest );
		case mxClassID::mxSINGLE_CLASS:
			return ConvertMxToScalar<float, T>( mx, dest );
		case mxClassID::mxDOUBLE_CLASS:
			return ConvertMxToScalar<double, T>( mx, dest );
		case mxClassID::mxINT8_CLASS:
			return ConvertMxToScalar<char, T>( mx, dest );
		case mxClassID::mxINT16_CLASS:
			return ConvertMxToScalar<short, T>( mx, dest );
		case mxClassID::mxINT32_CLASS:
			return ConvertMxToScalar<int, T>( mx, dest );
		case mxClassID::mxINT64_CLASS:
			return ConvertMxToScalar<long long, T>( mx, dest );
		case mxClassID::mxUINT8_CLASS:
			return ConvertMxToScalar<unsigned char, T>( mx, dest );
		case mxClassID::mxUINT16_CLASS:
			return ConvertMxToScalar<unsigned short, T>( mx, dest );
		case mxClassID::mxUINT32_CLASS:
			return ConvertMxToScalar<unsigned int, T>( mx, dest );
		case mxClassID::mxUINT64_CLASS:
			return ConvertMxToScalar<unsigned long long, T>( mx, dest );
		default:
			return ConversionResult::ErrorNotSupported;
		}
	}

	return ConversionResult::ErrorTypeMismatch;
}

template<class T>
mxClassID MexConverter<T>::getMatlabClassIdFromType() {
	if(typeid(T) == typeid(double)) {
		return mxClassID::mxDOUBLE_CLASS;
	} else if(typeid(T) == typeid(float)) {
		return mxClassID::mxSINGLE_CLASS;
	} else if(typeid(T) == typeid(char)) {
#if defined(UNICODE) || defined(_UNICODE)
		return mxClassID::mxINT8_CLASS;
#else
		return mxClassID::mxCHAR_CLASS;
#endif
	} else if(typeid(T) == typeid(wchar_t)) {
		return mxClassID::mxCHAR_CLASS;
	} else if(typeid(T) == typeid(unsigned char)) {
		return mxClassID::mxUINT8_CLASS;
	} else if(typeid(T) == typeid(short)) {
		return mxClassID::mxINT16_CLASS;
	} else if(typeid(T) == typeid(unsigned short)) {
		return mxClassID::mxUINT16_CLASS;
	} else if(typeid(T) == typeid(int)) {
		return mxClassID::mxINT32_CLASS;
	} else if(typeid(T) == typeid(unsigned int)) {
		return mxClassID::mxUINT32_CLASS;
	} else if(typeid(T) == typeid(long long)) {
		return mxClassID::mxINT64_CLASS;
	} else if(typeid(T) == typeid(unsigned long long)) {
		return mxClassID::mxUINT64_CLASS;
	} else if(typeid(T) == typeid(bool)) {
		return mxClassID::mxLOGICAL_CLASS;
	}
	return mxClassID::mxUNKNOWN_CLASS;
}

template<class T>
mxArray* MexConverter<T>::mxFromEmpty() {
	return mxCreateNumericMatrix( 0, 0, getMatlabClassIdFromType(), mxREAL );
}





template class MexConverter<bool>;
template class MexConverter<char>;
template class MexConverter<wchar_t>;
template class MexConverter<unsigned char>;
template class MexConverter<short>;
template class MexConverter<unsigned short>;
template class MexConverter<int>;
template class MexConverter<unsigned int>;
template class MexConverter<long long>;
template class MexConverter<unsigned long long>;
template class MexConverter<float>;
template class MexConverter<double>;

#endif /* _MATLAB */
