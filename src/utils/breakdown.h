#ifndef UTILS_BREAKDOWN_HEADER
#define UTILS_BREAKDOWN_HEADER
#include "timer.h"
/**
 * Number of maxximum subdivision allowed in a single breakdown.
 */
#define BREAKDOWN_RESOLUTION 10

typedef struct{
	char * id;
	utils_timer_s part[BREAKDOWN_RESOLUTION+1];
}utils_breakdown_s;
#endif
