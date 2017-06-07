#ifndef CUDAHELPER_CUH_
#define CUDAHELPER_CUH_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\cudaincludes.h"
#include "..\\..\\interface\\memory.h"
#include "..\\..\\interface\\neutralmatrix\\neutralmatrix.h"
#else
#include "../../cudaincludes.h"
#include "../../interface/memory.h"
#include "../../interface/neutralmatrix/neutralmatrix.h"
#endif

bool cudaCheckReturnCode( cudaError_t rc, int line, const char* module );
bool cudaCheckReturnCode( cusparseStatus_t rc, int line, const char* module );
bool cudaCheckReturnCode( cusolverStatus_t rc, int line, const char* module );

#define CUCHECKRC( rc ) cudaCheckReturnCode( rc, __LINE__, __FILE__ )

template<class T>
class CudaHelper {
public:
	static bool initDevice( int device );

	static bool initCuSolver( DeviceMemory<T>& mem );
	static bool initCuSparse( DeviceMemory<T>& mem );
	static void freeCuSolver( DeviceMemory<T>& mem );
	static void freeCuSparse( DeviceMemory<T>& mem );

	static void setArray( T* d_dest, T value, int numElements, int threadsPerBlock = -1 );
	static void printArray( T* src, int numElements );
	static void copy( T* d_dest, T* d_src, int numElements );

	static bool hAlloc( T** h_dest, size_t numElements, unsigned int flags = cudaHostAllocDefault, const char* name = "" );
	static bool dAlloc( T** d_dest, size_t numElements, const char* name );
	static bool dAlloc( T** d_dest, size_t numElements, const char* name, const T init );
	static bool h2d( T** d_dest, NeutralMatrix<T, int, RowMajor<int>>& h_mat, const char* name );
};



#endif
