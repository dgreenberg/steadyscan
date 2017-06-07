#ifndef HELPER_H_
#define HELPER_H_

template<class T>
class Helper {
public:
	static void Print( T* values, int nValues );
	static void PrintGPU( T* d_values, int nValues );
	static void Print2D( T* values, int width, int height, bool isColMajor );
	static void Print2DGPU( T* d_values, int width, int height, bool isColMajor );

	static void PrintAsGraph( T* values, int nValues );
	static void PrintAsGraphGPU( T* d_values, int nValues );

	static void SaveToBitmap( const char* path, T* b, T* g, T* r, int width, int height, bool isColMajor, bool absolute );
	static void SaveToBitmapGPU( const char* path, T* d_b, T* d_g, T* d_r, int width, int height, bool isColMajor, bool absolute );

private:
	static void WriteBitmap( const char* path, unsigned char* b, unsigned char* g, unsigned char* r, int width, int height );
};

#endif
