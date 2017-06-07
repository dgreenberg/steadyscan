#ifndef MEXSIMULATOR_H_
#define MEXSIMULATOR_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\includes.h"
#include "..\\..\\interface\memory.h"
#include "..\\..\\task\\task.h"
#include "..\\..\\interface\\taskparameters.h"
#else
#include "../../includes.h"
#include "../../interface/memory.h"
#include "../../task/task.h"
#include "../../interface/taskparameters.h"
#endif

//only used for debugging without matlab
//requires data in ROM_FILE_PATH
template<class T>
class MexSimulator {
public:
	static size_t saveRomToFile( const char* path, PreAllocatedMemory<T>& data );
	static size_t loadRomFromFile( const char* path, PreAllocatedMemory<T>& data );
	static bool compareRomToFile( const char* path, PreAllocatedMemory<T>& data );
};

#endif /* MEXSIMULATOR_H_ */
