#include "stopcriterion.h"

const char* StopCriterion_toString( StopCriterion val ) {
	static const char* names[] = {
		"CorrelationIncreaseThreshold",
		"HaltCorrelation",
		"MovementThreshold",
		"IterationLimit",
		"Error"
	};
	return names[static_cast<int>( val )];
}
