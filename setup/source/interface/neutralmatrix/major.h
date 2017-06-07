#ifndef MAJOR_H_
#define MAJOR_H_

// In a row-major order, the consecutive elements of a row reside next to each other
template<class IndexType>
class RowMajor {
public:
	static IndexType index( IndexType i, IndexType j, IndexType rows, IndexType cols );
};

// In a col-major order, the consecutive elements of a column reside next to each other
template<class IndexType>
class ColMajor {
public:
	static IndexType index( IndexType i, IndexType j, IndexType rows, IndexType cols );
};

template<class IndexType>
IndexType RowMajor<IndexType>::index( IndexType i, IndexType j, IndexType rows, IndexType cols ) {
	return i * cols + j;
}

template<class IndexType>
IndexType ColMajor<IndexType>::index( IndexType i, IndexType j, IndexType rows, IndexType cols ) {
	return i + j * rows;
}


#endif
