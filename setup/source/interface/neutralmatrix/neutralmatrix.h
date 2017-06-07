#ifndef NEUTRALMATRIX_H_
#define NEUTRALMATRIX_H_

#include "major.h"
#include "../../helper/memtracker/memtracker.h"

#include "../../includes.h"

#ifdef MEXINCLUDE
#include MEXINCLUDE
#endif
//uncomment to activate deep copy using the asignment operator even if Parameters match
//#define NEUTRALMATRIX_ASSIGN_DEEP


#define NEUTRALMATRIX_TMPCONSTR(Major) \
	NeutralMatrix() : NeutralMatrixBase<ValueType, IndexType, Major<IndexType>>() {} \
	NeutralMatrix( IndexType rows, IndexType cols ) : NeutralMatrixBase<ValueType, IndexType, Major<IndexType>>(rows, cols) {} \
	NeutralMatrix( ValueType* data, IndexType rows, IndexType cols, bool trackData = false ) : NeutralMatrixBase<ValueType, IndexType, Major<IndexType>>(data, rows, cols, trackData) {} \
	NeutralMatrix( IndexType rows, IndexType cols, ValueType constVal ) : NeutralMatrixBase<ValueType, IndexType, Major<IndexType>>( rows, cols, constVal ){} \
	virtual ~NeutralMatrix() { }

//basic matrix
//containing m, n and data pointer
//used by the Task class as input parameters
//remarks:
//- caller needs to take care of memory deallocation!
//- IndexType needs to support * and + operators and be valid value for indexing operators
//- Major has to contain accessible function:
//  static IndexType index( IndexType, IndexType, IndexType, IndexType );
template<typename ValueType, typename IndexType = int, class MajorType = ColMajor<IndexType>>
class NeutralMatrixBase {
public:
	NeutralMatrixBase() {
		this->_rows = this->_cols = 0;
		this->_data = nullptr;
	}
	NeutralMatrixBase( IndexType rows, IndexType cols ) {
		this->_data = new ValueType[rows * cols];
		this->_rows = rows;
		this->_cols = cols;
		MemTracker::addBlock( this->_data );
	}
	NeutralMatrixBase( ValueType* data, IndexType rows, IndexType cols, bool trackData = false ) {
		if( trackData ) MemTracker::addBlock( data );
		this->_data = data;
		this->_rows = rows;
		this->_cols = cols;
	}
	NeutralMatrixBase( IndexType rows, IndexType cols, ValueType constVal ) {
		this->_data = MemTracker::alloc<ValueType, IndexType>( rows * cols );
		this->_rows = rows;
		this->_cols = cols;

		for( IndexType i = rows * cols - 1; i >= 0; i = i + -1 ) this->_data[i] = constVal;
	}
	NeutralMatrixBase( const NeutralMatrixBase<ValueType, IndexType, MajorType>& obj ) {
		this->_data = obj._data;
		this->_rows = obj._rows;
		this->_cols = obj._cols;
		MemTracker::inc( this->_data );
	}
	virtual ~NeutralMatrixBase() {
		MemTracker::dec( this->_data );
		this->_data = nullptr;
		this->_rows = 0;
		this->_cols = 0;
	}


	NeutralMatrixBase<ValueType, IndexType, MajorType> deepcpy() const {
		return this->as<MajorType>();
	}

	NeutralMatrixBase<ValueType, IndexType, MajorType> operator=( const NeutralMatrixBase<ValueType, IndexType, MajorType>& obj ) {
		this->_data = obj._data;
		this->_rows = obj._rows;
		this->_cols = obj._cols;
		MemTracker::inc( this->_data );

		return *this;
	}

	ValueType& operator()( IndexType i, IndexType j ) {
#ifndef NEUTRALMATRIX_NO_DEBUG
		assert(i < this->_rows && j < this->_cols);
#endif
		return this->_data[MajorType::index( i, j, this->_rows, this->_cols )];
	}
	ValueType& operator[]( IndexType index ) {
#ifndef NEUTRALMATRIX_NO_DEBUG
		assert(index < this->_cols * this->_rows);
#endif
		//assert(index < _rows * _cols);
		return this->_data[index];
	}

	IndexType rows() const { return this->_rows; }
	IndexType cols() const { return this->_cols; }

	IndexType size() const { return this->_rows * this->_cols; }

	ValueType* data() const { return this->_data; }

	void setZero() {
		for( IndexType i = this->_rows * this->_cols - 1; i >= 0; i = i + -1 ) this->_data[i] = static_cast<ValueType>( 0 );
	}

	template<typename NewMajor>
	NeutralMatrixBase<ValueType, IndexType, NewMajor> as() {
		NeutralMatrixBase<ValueType, IndexType, NewMajor> retv( this->_cols, this->_rows );
		for( IndexType i = 0; i < this->_rows; i = i + 1 ) {
			for( IndexType j = 0; j < this->_cols; j = j + 1 ) {
				retv(i, j) = this->_data[MajorType::index(i, j, this->_rows, this->_cols)];
			}
		}
		return retv;
	}
protected:
	ValueType* _data;
	IndexType _rows, _cols;
};

template<typename ValueType, typename IndexType = int, class MajorType = ColMajor<IndexType>>
class NeutralMatrix {};

template<typename ValueType, typename IndexType>
class NeutralMatrix<ValueType, IndexType, RowMajor<IndexType>> : public NeutralMatrixBase<ValueType, IndexType, RowMajor<IndexType>> {
public:
	NEUTRALMATRIX_TMPCONSTR( RowMajor )

	NeutralMatrix<ValueType, IndexType, RowMajor<IndexType>> operator=( const NeutralMatrixBase<ValueType, IndexType, ColMajor<IndexType>>& rhs ) {
		MemTracker::dec( this->_data );

		this->_rows = rhs.rows();
		this->_cols = rhs.cols();

		this->_data = MemTracker::alloc<ValueType, IndexType>( this->_rows * this->_cols );

		for( IndexType i = 0; i < this->_rows; i = i + 1 ) {
			for( IndexType j = 0; j < this->_cols; j = j + 1 ) {
				this->_data[RowMajor<IndexType>::index(i, j, this->_rows, this->_cols)] = rhs.data()[ColMajor<IndexType>::index(i, j, this->_rows, this->_cols)];
			}
		}

		return *this;
	}
};

template<typename ValueType, typename IndexType>
class NeutralMatrix<ValueType, IndexType, ColMajor<IndexType>> : public NeutralMatrixBase<ValueType, IndexType, ColMajor<IndexType>> {
public:
	NEUTRALMATRIX_TMPCONSTR( ColMajor )

	NeutralMatrix<ValueType, IndexType, ColMajor<IndexType>> operator=( const NeutralMatrixBase<ValueType, IndexType, RowMajor<IndexType>>& rhs ) {
		MemTracker::dec( this->_data );

		this->_rows = rhs.rows();
		this->_cols = rhs.cols();

		this->_data = MemTracker::alloc<ValueType, IndexType>( this->_rows * this->_cols );

		for( IndexType i = 0; i < this->_rows; i = i + 1 ) {
			for( IndexType j = 0; j < this->_cols; j = j + 1 ) {
				this->_data[ColMajor<IndexType>::index(i, j, this->_rows, this->_cols)] = rhs.data()[RowMajor<IndexType>::index(i, j, this->_rows, this->_cols)];
			}
		}

		return *this;
	}
};

#endif /* NEUTRALMATRIX_H_ */
