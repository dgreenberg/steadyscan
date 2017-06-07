#ifndef INCLUDES_H_
#define INCLUDES_H_

#include "debug.h"

#include "localdependencies.h"

#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <typeinfo>
#include <math.h>
#include <limits.h>
#define __STDC_LIMIT_MACROS
#include <cfloat>
#include <cstdint>
#include <limits>
#include <thread>
#include <string.h>

#if defined( _WIN32 ) || defined( _WIN64 )
#include <cinttypes>
#else
#include <inttypes.h>
#endif
using namespace std;

#if defined( _WIN32 ) || defined( _WIN64 )
#include <Windows.h>
#else
#include <unistd.h>
#endif

#define zDelete( ptr ) if( nullptr != ptr ) { delete ptr; ptr = nullptr; }else

#endif /* INCLUDES_H_ */
