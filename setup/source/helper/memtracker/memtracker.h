#ifndef MEMTRACKER_H_
#define MEMTRACKER_H_

#include <set>

class MemTracker {
private:
	template<class T>
	class Chunk {
	public:
		Chunk( T* instance, bool block = false ) {
			this->_instance = instance;
			this->_ic = 1;
			this->_blockAllocated = block;
		}

		//const cheats to allow modification of ic in std::set
		void inc() const { this->add1(); }
		bool dec() const {
			if( !this->sub1() ) {
				this->free();
				return false;
			}
			return true;
		}
		T* instance() const { return this->_instance; }
		int icc() const { return this->_ic; }

		bool operator< (const Chunk<T>& c) const {	return this->_instance < c._instance;	}
	private:
		//const cheats to allow modification of ic in std::set
		int add1() const { return ++(*const_cast<int*>(reinterpret_cast<const int*>(this))); }
		int sub1() const { return --(*const_cast<int*>(reinterpret_cast<const int*>(this))); }

		void free() const {
			if( this->_blockAllocated ) delete[] this->instance();
			else delete this->instance();
		}

		int _ic;
		T* _instance;
		bool _blockAllocated;
	};

public:
	//adds object ptr to tracking
	//uses delete to clear memory when refcount reaches 0
	template<class T>
	static void addObject( T* ptr );

	//adds block ptr to tracking
	//uses delete[] to clear memory when refcount reaches 0
	template<class T>
	static void addBlock( T* ptr );

	//removes pointer from tracking
	//does not free allocated memory at [ptr]
	template<class T>
	static void remove( T* ptr );

	//increases the reference count for given address
	template<class T>
	static void inc( T* ptr );

	//decreases reference count for given address
	//deletes ptr when reference count reaches zero
	//make sure that T matches the actual type behind ptr
	//> (e.g. calling dec<int>( inst ) does not call the destructor for inst )
	template<class T>
	static void dec( T* ptr );

	//allocates memory block and starts tracking
	//to track allocated memory use >add or >inc
	template<class ValueType, class SizeType = int>
	static ValueType* alloc( SizeType nElements );

	static int countChunks();
	static int countPointers();
private:
	static void* getAllocationTracker();
};

template<class T>
void MemTracker::addObject( T* ptr ) {
	if(ptr == nullptr) return;

	std::set<Chunk<T>>* chunks = reinterpret_cast<std::set<Chunk<T>>*>( getAllocationTracker() );
	Chunk<T> s = Chunk<T>( ptr, false );
	chunks->insert( s );
}
template<class T>
void MemTracker::addBlock( T* ptr ) {
	if(ptr == nullptr) return;

	std::set<Chunk<T>>* chunks = reinterpret_cast<std::set<Chunk<T>>*>( getAllocationTracker() );
	Chunk<T> s = Chunk<T>( ptr, true );
	chunks->insert( s );
}

template<class T>
void MemTracker::remove( T* ptr ) {
	if(ptr == nullptr) return;

	std::set<Chunk<T>>* chunks = reinterpret_cast<std::set<Chunk<T>>*>( getAllocationTracker() );
	chunks->remove( Chunk<T>( ptr ) );
}

template<class T>
void MemTracker::inc( T* ptr ) {
	if(ptr == nullptr) return;

	std::set<Chunk<T>>* chunks = reinterpret_cast<std::set<Chunk<T>>*>( getAllocationTracker() );
	Chunk<T> s = Chunk<T>( ptr );
	typename std::set<Chunk<T>>::iterator it = chunks->find( s );
	if( it != chunks->end() ) it->inc();
}

template<class T>
void MemTracker::dec( T* ptr ) {
	if(ptr == nullptr) return;

	std::set<Chunk<T>>* chunks = reinterpret_cast<std::set<Chunk<T>>*>( getAllocationTracker() );
	Chunk<T> s = Chunk<T>( ptr );
	auto it = chunks->find( s );
	if( it == chunks->end() ) return;
	if( !it->dec() ) chunks->erase( s );
}

template<class ValueType, class SizeType>
ValueType* MemTracker::alloc( SizeType nElements ) {
	ValueType* ptr = new ValueType[nElements];
	if(nullptr == ptr) return nullptr;

	Chunk<ValueType> s = Chunk<ValueType>( ptr, true );

	std::set<Chunk<ValueType>>* chunks = reinterpret_cast<std::set<Chunk<ValueType>>*>( getAllocationTracker() );

	chunks->insert( s );

	return ptr;
}


#endif
