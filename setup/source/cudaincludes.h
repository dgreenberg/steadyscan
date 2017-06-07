#ifndef CUDAINCLUDES_H_
#define CUDAINCLUDES_H_

//cuda
#include <cuda.h>
#include <cuda_runtime.h>

//thrust
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

//cusparse
#include <cusparse_v2.h>
#include <cusolverSp.h>

#if defined( _WIN32 ) || defined( _WIN64 )
#pragma comment(lib, "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\cusolver.lib")
#pragma comment(lib, "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\cusparse.lib")
#else
#endif

//helper
#define zCudaFree( dev_ptr ) if( dev_ptr != nullptr ) { cudaFree( dev_ptr ); dev_ptr = nullptr; }else
#define zCudaFreeHost( h_ptr ) if( h_ptr != nullptr ) { cudaFreeHost( h_ptr ); h_ptr = nullptr; }else
#define IDX2C( i, j, ld ) ( ( ( j ) * ( ld ) ) + ( i ) )
#define IDX2R( i, j, ld ) ( ( ( i ) * ( ld ) ) + ( j ) )


#define THREADS_PER_BLOCK 256

#endif
