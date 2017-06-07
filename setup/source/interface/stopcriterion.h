#ifndef STOPCRITERION_H_
#define STOPCRITERION_H_

enum StopCriterion {
	CorrelationIncreaseThreshold,
	HaltCorrelationReached,
	MovementThreshold,
	IterationLimit,
	Error
};

const char* StopCriterion_toString( StopCriterion val );

#endif
