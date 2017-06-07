#ifndef REALTIMER_H_
#define REALTIMER_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\..\\includes.h"
#else
#include "../../includes.h"
#endif

//stops elapsed time between the last start() call and stop()
//returns elapsed time in total seconds
template<class T>
class RealTimer {
public:

#if defined( _WIN32 ) || defined( _WIN64 )
public:

	void start() {
		QueryPerformanceCounter(&this->startTime);
	}

	T stop() {
		LARGE_INTEGER ts, f;
		QueryPerformanceCounter(&ts);
		QueryPerformanceFrequency(&f);

		return (ts.QuadPart - this->startTime.QuadPart) / static_cast<T>( f.QuadPart );
	}

private:
	LARGE_INTEGER startTime;
#else
	void start() {
		clock_gettime(CLOCK_REALTIME, &this->startTime);
	}

	T stop() {
		timespec ts;
		clock_gettime(CLOCK_REALTIME, &ts);

		auto sec  = ts.tv_sec - this->startTime.tv_sec;
		auto nsec = ts.tv_nsec - this->startTime.tv_nsec;

		return static_cast<T>( sec ) + ( nsec / 1E9 );
	}

private:
	timespec startTime;
#endif
};

#endif
