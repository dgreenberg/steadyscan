#ifndef DEBUG_H_
#define DEBUG_H_

#define VERSION "4-10-1"

//uncomment defines to enable debug options

//debug versions are compiled to executables
//to allow debugging in the current ide using MexSimulator.
//release builds are compiled into a mex file
//(broken on linux)
#if !defined( _DEBUG )
#define _MATLAB
#endif

#ifndef _MATLAB
#define mexPrintf printf
#define mexEvalString(a)
#endif

//tell Eigen to skip debug extensions
#define EIGEN_NO_DEBUG

//tell NeutralMatrix to skip debug extensions
//#define NEUTRALMATRIX_NO_DEBUG

//enable debug information for thrust library
//#define THRUST_DEBUG 0

//uncomment to enable support for the bob-pipeline
#define BOB_SUPPORT_ENABLED

//uncomment to sort diagonals of sparse linear system on the gpu using thrust
//as this can cause problems on some devices, it is disabled by default
//#define ALLOW_THRUST_DIAGONAL_SORTING

//minPointsPerBlock is passed as vector containing
//a value for each block (in a single frame)
#ifdef BOB_SUPPORT_ENABLED
#define MINPOINTSPERBLOCK_AS_VECTOR
#define FRAC_AS_MAT
#endif

#define ROM_FILE_NAME "rom_" VERSION
#define ROM_FILE_TYPE ".bin"

#if defined( _WIN32 ) || defined( _WIN64 )
#define ROM_ROOT ".\\"
#else
#define ROM_ROOT "./"
#endif

//#define ROM_FILE_PATH ROM_ROOT ROM_FILE_NAME ROM_FILE_TYPE
#define ROM_FILE_PATH "/media/greenberg/USB128/motioncorrection/rom_4-10-1.bin"
#endif /* DEBUG_H_ */
