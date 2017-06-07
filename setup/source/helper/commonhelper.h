#ifndef COMMONHELPER_H_
#define COMMONHELPER_H_

#if defined( _WIN32 ) || defined( _WIN64 )
#include "..\\includes.h"
#else
#include "../includes.h"
#endif

#ifdef EIGENINCLUDE
#include EIGENINCLUDE
#endif

#ifdef MEXINCLUDE
#include MEXINCLUDE
#endif

inline Eigen::SparseMatrix<float>::InnerIterator GetSparseIterator(Eigen::SparseMatrix<float>& sp, int col) {
	return Eigen::SparseMatrix<float>::InnerIterator(sp, col);
}
inline Eigen::SparseMatrix<double>::InnerIterator GetSparseIterator(Eigen::SparseMatrix<double>& sp, int col) {
	return Eigen::SparseMatrix<double>::InnerIterator(sp, col);
}
inline Eigen::SparseMatrix<long long int>::InnerIterator GetSparseIterator(Eigen::SparseMatrix<long long int>& sp, int col) {
	return Eigen::SparseMatrix<long long int>::InnerIterator(sp, col);
}
inline Eigen::SparseMatrix<int>::InnerIterator GetSparseIterator(Eigen::SparseMatrix<int>& sp, int col) {
	return Eigen::SparseMatrix<int>::InnerIterator(sp, col);
}
inline Eigen::SparseMatrix<unsigned int>::InnerIterator GetSparseIterator(Eigen::SparseMatrix<unsigned int>& sp, int col) {
	return Eigen::SparseMatrix<unsigned int>::InnerIterator(sp, col);
}
inline Eigen::SparseMatrix<short>::InnerIterator GetSparseIterator(Eigen::SparseMatrix<short>& sp, int col) {
	return Eigen::SparseMatrix<short>::InnerIterator(sp, col);
}
inline Eigen::SparseMatrix<char>::InnerIterator GetSparseIterator(Eigen::SparseMatrix<char>& sp, int col) {
	return Eigen::SparseMatrix<char>::InnerIterator(sp, col);
}
inline Eigen::SparseMatrix<bool>::InnerIterator GetSparseIterator(Eigen::SparseMatrix<bool>& sp, int col) {
	return Eigen::SparseMatrix<bool>::InnerIterator(sp, col);
}

inline void PrintValue(float value) {
	mexPrintf("%f ", value);
}
inline void PrintValue(double value) {
	mexPrintf("%f ", value);
}
inline void PrintValue(long long int value) {
	mexPrintf("%lli ", value);
}
inline void PrintValue(int value) {
	mexPrintf("%i ", value);
}
inline void PrintValue(unsigned int value) {
	mexPrintf("%i ", value);
}
inline void PrintValue(char value) {
	mexPrintf("%c ", value);
}
inline void PrintValue(bool value) {
	mexPrintf("%i ", static_cast<int>(value));
}

inline void PrintValue(float value, int x, int y) {
	mexPrintf("(%i, %i)\t\t%f\n", y, x, value);
}
inline void PrintValue(double value, int x, int y) {
	mexPrintf("(%i, %i)\t\t%f\n", y, x, value);
}
inline void PrintValue(long long int value, int x, int y) {
	mexPrintf("(%i, %i)\t\t%lli\n", y, x, value);
}
inline void PrintValue(int value, int x, int y) {
	mexPrintf("(%i, %i)\t\t%i\n", y, x, value);
}
inline void PrintValue(unsigned int value, int x, int y) {
	mexPrintf("(%i, %i)\t\t%i\n", y, x, value);
}
inline void PrintValue(char value, int x, int y) {
	mexPrintf("(%i, %i)\t\t%c\n", y, x, value);
}
inline void PrintValue(bool value, int x, int y) {
	mexPrintf("(%i, %i)\t\t%i\n", y, x, value);
}

#endif



















































