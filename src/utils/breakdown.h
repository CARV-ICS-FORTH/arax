#ifndef UTILS_BREAKDOWN_HEADER
#define UTILS_BREAKDOWN_HEADER
#ifdef BREAKS_ENABLE
#include "timer.h"
/**
 * Number of max parts allowed in a single breakdown.
 */
#define BREAKDOWN_PARTS 10

typedef struct{
	unsigned long long samples;					//< Number of breakdowns
	unsigned long long part[BREAKDOWN_PARTS];	//< Duration in ns of each part
}utils_breakdown_stats_s;

#define UTILS_BREAKDOWN_STATS(name) utils_breakdown_stats_s name

typedef struct{
	utils_timer_s timer;				//< Timer used for counting duration
	utils_breakdown_stats_s * stats;	//< Aggregate statistics
	int current_part;					//<
}utils_breakdown_instance_s;

#define UTILS_BREAKDOWN_INSTANCE(name) utils_breakdown_instance_s name

void utils_breakdown_init_stats(utils_breakdown_stats_s * stats);

void utils_breakdown_begin(utils_breakdown_instance_s * bdown,utils_breakdown_stats_s * stats);

void utils_breakdown_advance(utils_breakdown_instance_s * bdown);

void utils_breakdown_end(utils_breakdown_instance_s * bdown);
#else

#define UTILS_BREAKDOWN_STATS(name)

#define UTILS_BREAKDOWN_INSTANCE(name)

#define utils_breakdown_init_stats(stats)

#define utils_breakdown_begin(bdown,stats)

#define utils_breakdown_advance(bdown)

#define utils_breakdown_end(bdown)
#endif /* ifdef BREAKS_ENABLE */

#endif /* ifdef UTILS_BREAKDOWN_HEADER */
