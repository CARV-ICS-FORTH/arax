#ifndef UTILS_BREAKDOWN_HEADER
#define UTILS_BREAKDOWN_HEADER
#include "conf.h"
#ifdef BREAKS_ENABLE
#include "timer.h"
/**
 * Number of max parts allowed in a single breakdown.
 */
#define BREAKDOWN_PARTS 32

typedef struct{
	unsigned long long samples;					//< Number of breakdowns
	unsigned long long part[BREAKDOWN_PARTS];	//< Duration in ns of each part
#ifdef BREAKS_HEADS
	const char * desc[BREAKDOWN_PARTS];			//< Description fo each part
	char heads[BREAKDOWN_PARTS*64];				//< Storage for headers.
	char * head_ptr;							//< Header pointer.
#endif
}utils_breakdown_stats_s;

#define UTILS_BREAKDOWN_STATS(name) utils_breakdown_stats_s name

typedef struct{
	utils_timer_s timer;				//< Timer used for counting duration
	utils_breakdown_stats_s * stats;	//< Aggregate statistics
	int current_part;					//<
}utils_breakdown_instance_s;

#define UTILS_BREAKDOWN_INSTANCE(name) utils_breakdown_instance_s name

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

void utils_breakdown_init_stats(utils_breakdown_stats_s * stats);

void utils_breakdown_begin(utils_breakdown_instance_s * bdown,utils_breakdown_stats_s * stats,const char * description);

void utils_breakdown_advance(utils_breakdown_instance_s * bdown,const char * description);

void utils_breakdown_end(utils_breakdown_instance_s * bdown);

void utils_breakdown_write(const char *file,vine_accel_type_e type,const char * description,utils_breakdown_stats_s * stats);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#else

#define UTILS_BREAKDOWN_STATS(name)

#define UTILS_BREAKDOWN_INSTANCE(name)

#define utils_breakdown_init_stats(stats)

#define utils_breakdown_begin(bdown,stats,description)

#define utils_breakdown_advance(bdown,description)

#define utils_breakdown_end(bdown)

#define utils_breakdown_write(file,type,description,stats)

#endif /* ifdef BREAKS_ENABLE */

#endif /* ifdef UTILS_BREAKDOWN_HEADER */
