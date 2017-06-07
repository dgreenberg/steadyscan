#include "helper.h"
#include "../cudaincludes.h"
#include "../includes.h"
#include "minmax.h"
#include "commonhelper.h"

#ifdef MEXINCLUDE
#include MEXINCLUDE
#endif

template<class T>
void Helper<T>::Print( T* values, int nValues ) {
	for( int i = 0; i < nValues; i++ ) {
		PrintValue( values[i] );
	}

	mexPrintf("\n");
}
template<class T>
void Helper<T>::PrintGPU( T* d_values, int nValues ) {
	T* hostValues = new T[nValues];

	cudaError_t err = cudaMemcpy( hostValues, d_values, nValues * sizeof(T), cudaMemcpyDeviceToHost );

	Helper<T>::Print( hostValues, nValues );

	delete[] hostValues;
}

template<class T>
void Helper<T>::Print2D( T* values, int width, int height, bool isColMajor ) {
	for( int i = 0; i < height; i++ ) {
		for( int j = 0; j < width; j++ ) {
			if(isColMajor) PrintValue( values[IDX2C( i, j, height )] );
			else PrintValue( values[IDX2R( i, j, width )] );
		}
	}

	mexPrintf("\n");
}
template<class T>
void Helper<T>::Print2DGPU( T* d_values, int width, int height, bool isColMajor ) {
	int nValues = width * height;
	T* hostValues = new T[nValues];

	cudaMemcpy( hostValues, d_values, sizeof( T ) * nValues, cudaMemcpyDeviceToHost );

	Helper<T>::Print2D( hostValues, width, height, isColMajor );

	delete[] hostValues;
}

template<class T>
void Helper<T>::PrintAsGraph( T* params, int nParams ) {
	const int width = 200, height = 41;
	T bins[width] = {0};
	T max = 0, min = maxValue<T>();
	int zeropos;

	if(nParams < 0) {
		mexPrintf("no graph;nParams must be > 0 (it is %i)\n", nParams);
		return;
	}

	int binstep = nParams / width;

	for(int i = 0; i < nParams; i++) {
		int bin = i / binstep;
		bins[bin] += params[i];
		if(bins[bin] > max) {
			max = bins[bin];
		} else if(bins[bin] < min) {
			min = bins[bin];
		}
	}

	if( min == max ) {
		mexPrintf("no graph;all values are equal to %g\n", (double)params[0]);
		return;
	}

	mexPrintf("\n\n\n\n\n\nmin: %g; max: %g; vres: %g/c; hres: %g/c\n", (double)min / binstep, (double)max / binstep, (double)( ( ( max - min ) / binstep) / height ), (double)( nParams / width ));

	int width2 = width + 3;
	int height2 = height + 2;
	char* print = new char[width2 * height2 + 1];
	print[width2 * height2] = '\0';

	//shift max to display negativ values
	max -= min;

	//fill whitespace
	for(int i = 1; i < height2 - 1; i++) {
		for(int j = 1; j < width2 - 2; j++) {
			print[IDX2R(i, j, width2)] = ' ';
		}
	}
	//draw border
	for(int j = 0; j < width2-1; j++) {
		print[IDX2R(0, j, width2)] = '-';
		print[IDX2R(height2 - 1, j, width2)] = '-';

		//draw zero line
		int i = height - static_cast<int>(((0 - min) / max) * (height-1) + 1.5);
		print[IDX2R(i+1, j, width2)] = '-';
	}
	for(int i = 0; i < height2; i++) {
		if(i == 0 || i == height2 - 1) {
			print[IDX2R(i, 0, width2)] = '+';
			print[IDX2R(i, width2-2, width2)] = '+';
		} else {
			print[IDX2R(i, 0, width2)] = '|';
			print[IDX2R(i, width2-2, width2)] = '|';
		}
		print[IDX2R(i, width2-1, width2)] = '\n';
	}
	//fill *
	for(int j = 0; j < width; j++) {
		int i = height - static_cast<int>(((bins[j] - min) / max) * (height-1) + 1.5);

		print[IDX2R(i+1, j+1, width2)] = '+';
	}

//print graph
	mexPrintf("%s\n", print);

	delete[] print;
}
template<class T>
void Helper<T>::PrintAsGraphGPU(T* d_params, int nParams) {
	T* buffer = new T[nParams];
	cudaMemcpy(buffer, d_params, sizeof(T) * nParams, cudaMemcpyDeviceToHost);
	Helper<T>::PrintAsGraph(buffer, nParams);
	delete[] buffer;
}

template<class T>
void Helper<T>::WriteBitmap( const char* path, unsigned char* b, unsigned char* g, unsigned char* r, int width, int height ) {
	unsigned int headers[13];
	FILE * outfile;

	//blue channel is 'copied'
	//to red and green if they are 0
	if( b == nullptr ) return;
	if( g == nullptr ) g = b;
	if( r == nullptr ) r = b;

	//How many bytes of padding to add to each
	//horizontal line - the size of which must
	//be a multiple of 4 bytes.
	int extrabytes = 4 - ( ( width * 3 ) % 4 );
	if( extrabytes == 4 ) extrabytes = 0;

	int paddedsize = ( ( width * 3 ) + extrabytes ) * height;

	//Headers...
	//Note that the "BM" identifier in bytes 0 and 1 is NOT included in these "headers".

	headers[0] = paddedsize + 54;      // bfSize (whole file size)
	headers[1] = 0;                    // bfReserved (both)
	headers[2] = 54;                   // bfOffbits
	headers[3] = 40;                   // biSize
	headers[4] = width;  // biWidth
	headers[5] = height; // biHeight
	headers[6] = 0;
	headers[7] = 0;             // biCompression
	headers[8] = paddedsize;	// biSizeImage
	headers[9] = 0;             // biXPelsPerMeter
	headers[10] = 0;            // biYPelsPerMeter
	headers[11] = 0;            // biClrUsed
	headers[12] = 0;            // biClrImportant

	outfile = fopen( path, "wb" );

	//
	// Headers
	// When writing ints and shorts, write out 1 character at a time to avoid endian issues.
	fprintf( outfile, "BM" );

	for( int n = 0; n <= 5; n++ ) {
		fprintf( outfile, "%c", headers[n] & 0x000000FF );
		fprintf( outfile, "%c", ( headers[n] & 0x0000FF00 ) >> 8 );
		fprintf( outfile, "%c", ( headers[n] & 0x00FF0000 ) >> 16 );
		fprintf( outfile, "%c", ( headers[n] & static_cast<unsigned int>( 0xFF000000 ) ) >> 24 );
	}

	//biPlanes and biBitCount fields.
	fprintf( outfile, "%c%c", 1, 0 );
	fprintf( outfile, "%c%c", 24, 0 );

	for( int n = 7; n <= 12; n++ ) {
		fprintf( outfile, "%c", headers[n] & 0x000000FF );
		fprintf( outfile, "%c", ( headers[n] & 0x0000FF00 ) >> 8 );
		fprintf( outfile, "%c", ( headers[n] & 0x00FF0000 ) >> 16 );
		fprintf( outfile, "%c", ( headers[n] & static_cast<unsigned int>( 0xFF000000 ) ) >> 24 );
	}

	//Headers done, now write the data
	//BMP image format is written from bottom to top
	//in blue green red format
	for( int y = height - 1; y >= 0; y-- ) {
		for( int x = 0; x <= width - 1; x++ ) {
			fprintf( outfile, "%c", b[y * width + x] );
			fprintf( outfile, "%c", g[y * width + x] );
			fprintf( outfile, "%c", r[y * width + x] );
		}

		// in case BMP lines lengths are not divisible by 4.
		if( extrabytes ) {
			for( int n = 1; n <= extrabytes; n++ ) {
				fprintf( outfile, "%c", 0 );
			}
		}
	}

	fclose( outfile );
}
template<class T>
void Helper<T>::SaveToBitmap( const char* path, T* b, T* g, T* r, int width, int height, bool isColMajor, bool absolute ) {
	unsigned char* cb;
	unsigned char* cg;
	unsigned char* cr;

	cb = new unsigned char[width * height];
	cg = new unsigned char[width * height];
	cr = new unsigned char[width * height];

	if( b == nullptr ) return;
	if( g == nullptr ) g = b;
	if( r == nullptr ) r = b;

	if( isColMajor ) {
		if( absolute ) {
			for(int i = 0; i < height; i++) {
				for(int j = 0; j < width; j++) {
					cb[IDX2R(i, j, width)] = static_cast<unsigned char>( b[IDX2C(i, j, height)] );
					cg[IDX2R(i, j, width)] = static_cast<unsigned char>( g[IDX2C(i, j, height)] );
					cr[IDX2R(i, j, width)] = static_cast<unsigned char>( r[IDX2C(i, j, height)] );
				}
			}
		} else {
			for( int i = 0; i < height; i++ ) {
				for( int j = 0; j < width; j++ ) {
					cb[IDX2R(i, j, width)] = static_cast<unsigned char>( max(0, min(255, b[IDX2C(i, j, height)] * 255 ) ) );
					cg[IDX2R(i, j, width)] = static_cast<unsigned char>( max(0, min(255, g[IDX2C(i, j, height)] * 255 ) ) );
					cr[IDX2R(i, j, width)] = static_cast<unsigned char>( max(0, min(255, r[IDX2C(i, j, height)] * 255 ) ) );
				}
			}
		}
	} else {
		if( absolute ) {
			for( int i = 0; i < height * width; i++ ) {
				cb[i] = static_cast<unsigned char>( b[i] );
				cg[i] = static_cast<unsigned char>( g[i] );
				cr[i] = static_cast<unsigned char>( r[i] );
			}
		} else {
			for( int i = 0; i < height * width; i++ ) {
				cb[i] = static_cast<unsigned char>( max(0, min(255, b[i] * 255 ) ) );
				cg[i] = static_cast<unsigned char>( max(0, min(255, g[i] * 255 ) ) );
				cr[i] = static_cast<unsigned char>( max(0, min(255, r[i] * 255 ) ) );
			}
		}
	}

	Helper<T>::WriteBitmap( path, cb, cg, cr, width, height );

	delete[] cb;
	delete[] cg;
	delete[] cr;
}
template<class T>
void Helper<T>::SaveToBitmapGPU( const char* path, T* d_b, T* d_g, T* d_r, int width, int height, bool isColMajor, bool absolute ) {
	T* bb;
	T* bg;
	T* br;

	bb = new T[width * height];
	bg = new T[width * height];
	br = new T[width * height];

	if( d_b == nullptr ) return;
	if( d_g == nullptr ) d_g = d_b;
	if( d_r == nullptr ) d_r = d_b;

	cudaMemcpy( bb, d_b, sizeof( T ) * width * height, cudaMemcpyDeviceToHost );
	cudaMemcpy( bg, d_g, sizeof( T ) * width * height, cudaMemcpyDeviceToHost );
	cudaMemcpy( br, d_r, sizeof( T ) * width * height, cudaMemcpyDeviceToHost );

	Helper<T>::SaveToBitmap( path, bb, bg, br, width, height, isColMajor, absolute );

	delete[] bb;
	delete[] bg;
	delete[] br;
}

template class Helper<bool>;
template class Helper<char>;
template class Helper<short>;
template class Helper<int>;
template class Helper<long long>;
template class Helper<float>;
template class Helper<double>;
