#include "memtracker.h"


void* MemTracker::getAllocationTracker() {
	static std::set<Chunk<int>> allocations;
	return &allocations;
}

int MemTracker::countChunks() {
	std::set<Chunk<int>>* set = reinterpret_cast<std::set<Chunk<int>>*>(getAllocationTracker());
	return set->size();
}
int MemTracker::countPointers() {
	int counts = 0;
	std::set<Chunk<int>>* set = reinterpret_cast<std::set<Chunk<int>>*>(getAllocationTracker());

	for(auto& elem : *set) {
		counts += elem.icc();
	}

	return counts;
}
