#include "cudahelper.cuh"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

template<class T>
bool CudaHelper<T>::initDevice( int device ) {
	cudaError_t rc;
	void* ptr;
	cudaDeviceProp prop;

	rc = cudaGetDeviceProperties( &prop, device );
	if( rc != cudaSuccess ) {
		std::cout << "unable to query device info for device " << device << std::endl;
		std::cout << "error info: " << rc << "> " << cudaGetErrorName( rc ) << std::endl;
	}

	rc = cudaSetDevice( device );
	if( rc != cudaSuccess ) {
		std::cout << "unable to select " << prop.name << "(dev " << device << ")" << std::endl;
		std::cout << "error info: " << rc << "> " << cudaGetErrorName( rc ) << std::endl;
		return false;
	}

	//this initializes the device
	//which takes about 1 sec
	rc = cudaMalloc( &ptr, 1 );
	if( rc != cudaSuccess ) {
		std::cout << "unable to allocate any memory on device" << std::endl;
		std::cout << "error info: " << rc << "> " << cudaGetErrorName( rc ) << std::endl;
		return false;
	}

	cudaFree( ptr );

	return true;
}

bool cudaCheckReturnCode( cudaError_t rc, int line, const char* module ) {
	if( rc == cudaError_t::cudaSuccess ) return true;

	printf( "%s\nLine %i in %s\n", cudaGetErrorName( rc ), line, module );
	return false;
}
bool cudaCheckReturnCode( cusparseStatus_t rc, int line, const char* module ) {
	switch( rc ) {
	case CUSPARSE_STATUS_SUCCESS:
		return true;
	case CUSPARSE_STATUS_ALLOC_FAILED:
		printf( "CUSPARSE_STATUS_ALLOC_FAILED\n" );
		break;
	case CUSPARSE_STATUS_ARCH_MISMATCH:
		printf( "CUSPARSE_STATUS_ARCH_MISMATCH\n" );
		break;
	case CUSPARSE_STATUS_EXECUTION_FAILED:
		printf( "CUSPARSE_STATUS_EXECUTION_FAILED\n" );
		break;
	case CUSPARSE_STATUS_INTERNAL_ERROR:
		printf( "CUSPARSE_STATUS_INTERNAL_ERROR\n" );
		break;
	case CUSPARSE_STATUS_INVALID_VALUE:
		printf( "CUSPARSE_STATUS_INVALID_VALUE\n" );
		break;
	case CUSPARSE_STATUS_MAPPING_ERROR:
		printf( "CUSPARSE_STATUS_MAPPING_ERROR\n" );
		break;
	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		printf( "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED\n" );
		break;
	case CUSPARSE_STATUS_NOT_INITIALIZED:
		printf( "CUSPARSE_STATUS_NOT_INITIALIZED\n" );
		break;
	case CUSPARSE_STATUS_ZERO_PIVOT:
		printf( "CUSPARSE_STATUS_ZERO_PIVOT\n" );
		break;
	default:
		printf( "CUSPARSE UNHANDLED ERROR\n" );
	}
	return false;
}
bool cudaCheckReturnCode( cusolverStatus_t rc, int line, const char* module ) {
	switch( rc ) {
	case CUSOLVER_STATUS_SUCCESS:
		return true;
	case CUSOLVER_STATUS_ALLOC_FAILED:
		printf( "CUSOLVER_STATUS_ALLOC_FAILED\n" );
		break;
	case CUSOLVER_STATUS_ARCH_MISMATCH:
		printf( "CUSOLVER_STATUS_ARCH_MISMATCH\n" );
		break;
	case CUSOLVER_STATUS_EXECUTION_FAILED:
		printf( "CUSOLVER_STATUS_EXECUTION_FAILED\n" );
		break;
	case CUSOLVER_STATUS_INTERNAL_ERROR:
		printf( "CUSOLVER_STATUS_INTERNAL_ERROR\n" );
		break;
	case CUSOLVER_STATUS_INVALID_LICENSE:
		printf( "CUSOLVER_STATUS_INVALID_LICENSE\n" );
		break;
	case CUSOLVER_STATUS_INVALID_VALUE:
		printf( "CUSOLVER_STATUS_INVALID_VALUE\n" );
		break;
	case CUSOLVER_STATUS_MAPPING_ERROR:
		printf( "CUSOLVER_STATUS_MAPPING_ERROR\n" );
		break;
	case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		printf( "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED\n" );
		break;
	case CUSOLVER_STATUS_NOT_INITIALIZED:
		printf( "CUSOLVER_STATUS_NOT_INITIALIZED\n" );
		break;
	case CUSOLVER_STATUS_NOT_SUPPORTED:
		printf( "CUSOLVER_STATUS_NOT_SUPPORTED\n" );
		break;
	case CUSOLVER_STATUS_ZERO_PIVOT:
		printf( "CUSOLVER_STATUS_ZERO_PIVOT\n" );
		break;
	}
	printf( "Line %i in %s\n", line, module);
	return false;
}


template<class T>
bool CudaHelper<T>::initCuSolver( DeviceMemory<T>& mem ) {
	//init cuSolver
	if( !CUCHECKRC( cusolverSpCreate( &mem.cuSolverHandle ) ) ) {
		return false;
	}
	if( !CUCHECKRC( cusparseCreateMatDescr( &mem.cuSparseDescription ) ) ) {
		return false;
	}
	if( !CUCHECKRC( cusparseSetMatIndexBase( mem.cuSparseDescription, CUSPARSE_INDEX_BASE_ZERO ) ) ) {
		return false;
	}
	if( !CUCHECKRC( cusparseSetMatType( mem.cuSparseDescription, CUSPARSE_MATRIX_TYPE_GENERAL ) ) ) {
		return false;
	}
	return true;
}
template<class T>
bool CudaHelper<T>::initCuSparse( DeviceMemory<T>& mem ) {
	//init suSparse
	if( !CUCHECKRC( cusparseCreate( &mem.cuSparseHandle ) ) ) {
		return false;
	}

	return true;
}
template<class T>
void CudaHelper<T>::freeCuSolver( DeviceMemory<T>& mem ) {
	if( nullptr != mem.cuSolverHandle ) {
		CUCHECKRC( cusolverSpDestroy( mem.cuSolverHandle ) );
	}

	if( nullptr != mem.cuSparseDescription ) {
		CUCHECKRC( cusparseDestroyMatDescr( mem.cuSparseDescription ) );
	}
}

template<class T>
void CudaHelper<T>::freeCuSparse( DeviceMemory<T>& mem ) {
	if( nullptr != mem.cuSparseHandle ) {
		CUCHECKRC( cusparseDestroy( mem.cuSparseHandle ) );
	}
}



template<class T>
__global__ void d_setArray( T* d_dest, int numElements, T value ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= numElements ) return;
	d_dest[idx] = value;
}

template<class T>
void CudaHelper<T>::setArray( T* d_dest, T value, int numElements, int threadsPerBlock ) {
	if( threadsPerBlock < 0 ) threadsPerBlock = THREADS_PER_BLOCK;

	int numBlocks = ( numElements + threadsPerBlock - 1 ) / threadsPerBlock;
	d_setArray<T><<< numBlocks, threadsPerBlock >>>( d_dest, numElements, value );
}

template<class T>
void CudaHelper<T>::printArray( T* src, int numElements ) {
	T* arr = new T[numElements];
	cudaMemcpy( arr, src, sizeof( T ) * numElements, cudaMemcpyDeviceToHost );

	for( int i = 0; i < numElements; i++ ) {
		printf( "[%i]: %f\n", i, static_cast<float>( arr[i] ) );
	}

	delete[] arr;
}

template<class T>
__global__ void cpykernel( T* d_dest, T* d_src, int numElements ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( idx >= numElements ) return;

	d_dest[idx] = d_src[idx];
}

template<class T>
void CudaHelper<T>::copy( T* d_dest, T* d_src, int numElements ) {
	int numBlocks = ( numElements + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
	cpykernel<T><<<numBlocks, THREADS_PER_BLOCK>>>( d_dest, d_src, numElements );
}


template<class T>
bool CudaHelper<T>::hAlloc( T** h_dest, size_t numElements, unsigned int flags, const char* name ) {
	cudaError_t rc;

	rc = cudaHostAlloc( h_dest, sizeof(T) * numElements, flags );
	if( rc != cudaSuccess ) {
		std::cout << "unable to allocate host memory for '" << name << "'" << std::endl;
		std::cout << "tried to allocate " << sizeof(T) * numElements << " bytes" << std::endl;
		std::cout << "error info: " << rc << "> " << cudaGetErrorName( rc ) << std::endl;

		//allocate using default new
		*h_dest = new T[numElements];
		if( *h_dest == nullptr ){
			std::cout << "unable to allocate host memory for '" << name << "'" << std::endl;
			std::cout << "tried to allocate " << sizeof(T) * numElements << " bytes" << std::endl;
			return false;
		}

		//register memory
		rc = cudaHostRegister( *h_dest, sizeof(T) * numElements, cudaHostRegisterDefault );

		//continue as unpinned if not successfull
		if( rc != cudaSuccess ) {
			std::cout << "unable to pin host memory for '" << name << "'" << std::endl;
			std::cout << "tried to pin " << sizeof(T) * numElements << " bytes" << std::endl;
			std::cout << "error info: " << rc << "> " << cudaGetErrorName( rc ) << std::endl;
		}

		return true;
	}
	return true;
}
template<class T>
bool CudaHelper<T>::dAlloc( T** d_dest, size_t numElements, const char* name ) {
	cudaError_t rc;

	rc = cudaMalloc( d_dest, sizeof(T) * numElements );
	if( rc != cudaSuccess ) {
		std::cout << "unable to allocate device memory for '" << name << "'" << std::endl;
		std::cout << "tried to allocate " << sizeof(T) * numElements << " bytes" << std::endl;
		std::cout << "error info: " << rc << "> " << cudaGetErrorName( rc ) << std::endl;

		return false;
	}

	return true;
}
template<class T>
bool CudaHelper<T>::dAlloc( T** d_dest, size_t numElements, const char* name, const T init ) {
	if( !dAlloc( d_dest, numElements, name ) ) return false;

	setArray( *d_dest, init, numElements );

	return true;
}
template<class T>
bool CudaHelper<T>::h2d( T** d_dest, NeutralMatrix<T, int, RowMajor<int>>& h_mat, const char* name ) {
	int numElements = h_mat.rows() * h_mat.cols();

	if( !dAlloc( d_dest, numElements, name ) ) return false;
	bool unregisterHost = true;
	bool retv = true;

	//try to pin host memory
	cudaError_t rc = cudaHostRegister( h_mat.data(), sizeof( T ) * numElements, cudaHostRegisterDefault );

	//well if that doesn't work show warning and
	//make sure not to call cudaHostUnregister
	if( rc != cudaSuccess ) {
		std::cout << "unable to register host memory for '" << name << "'" << std::endl;
		std::cout << "tried to register " << sizeof( T ) * numElements << " bytes" << std::endl;
		std::cout << "error info: " << rc << "> " << cudaGetErrorName( rc ) << std::endl;

		unregisterHost = false;
	}

	//cuda host memory to device
	rc = cudaMemcpy( *d_dest, h_mat.data(), sizeof( T ) * numElements, cudaMemcpyHostToDevice );

	//if that fail show error and
	//release all allocated resources
	if( rc != cudaSuccess ) {
		std::cout << "unable to copy host memory to device for '" << name << "'" << std::endl;
		std::cout << "tried to copy " << sizeof( T ) * numElements << " bytes" << std::endl;
		std::cout << "error info: " << rc << "> " << cudaGetErrorName( rc ) << std::endl;

		zCudaFree( *d_dest );
		retv = false;
	}

	//unpin host memory if it's pinned
	if( unregisterHost ) cudaHostUnregister( h_mat.data() );

	return retv;
}








template class CudaHelper<float>;
template class CudaHelper<double>;
template class CudaHelper<bool>;
template class CudaHelper<char>;
template class CudaHelper<short>;
template class CudaHelper<int>;
template class CudaHelper<long long>;

