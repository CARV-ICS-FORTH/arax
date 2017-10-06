#ifndef UTILS_BREAKDOWN_HEADER
#define UTILS_BREAKDOWN_HEADER
#include "conf.h"
#ifdef BREAKS_ENABLE
#include "timer.h"
/**
 * Number of max parts allowed in a single breakdown.
 */
#define BREAKDOWN_PARTS 32

/**
 * Keeps aggregate time of numerous utils_breakdown_instance_s.
 */
typedef struct{
	unsigned long long samples;					//< Number of breakdowns
	utils_timer_s interval;						//< Interval Timer (Task 2 Task Gap)
	unsigned long long part[BREAKDOWN_PARTS+2];	//< Duration in ns of each part(sum)
	const char * desc[BREAKDOWN_PARTS];			//< Description for each part
	char heads[BREAKDOWN_PARTS*64];				//< Storage for headers.
	char * head_ptr;							//< Header pointer.
}utils_breakdown_stats_s __attribute__((aligned(CONF_CACHE_LINE)));

/**
 * Keeps duration of an individual operation.
 */
typedef struct{
	utils_timer_s timer;						//< Timer used for counting duration
	unsigned long long part[BREAKDOWN_PARTS+1];	//< Duration in ns of each part,sum at the end
	utils_breakdown_stats_s * stats;			//< Aggregate statistics
	int current_part;							//< Currently measured part
	int first;									//< 0 if the first instance of a kernel
}utils_breakdown_instance_s;

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Initialize utils_breakdown_stats_s \c stats.
 *
 * \param stats utils_breakdown_stats_s to be initialized.
 */
void utils_breakdown_init_stats(utils_breakdown_stats_s * stats);

/**
 * Begin counting time from this point on.
 *
 * \param bdown utils_breakdown_instance_s to hold this operations breakdown.
 * \param stats Breakdown will be added to this utils_breakdown_stats_s
 * \param description Description of the current breakdown part
 */
void utils_breakdown_begin(utils_breakdown_instance_s * bdown,utils_breakdown_stats_s * stats,const char * description);

/**
 * Mark the end on a part and the start of a new part in \c dbown breakdown.
 *
 * \param bdown utils_breakdown_instance_s to hold this operations breakdown.
 * \param description Description of the current breakdown part
 */
void utils_breakdown_advance(utils_breakdown_instance_s * bdown,const char * description);

/**
 * Mark the end of the final part in \c dbown breakdown.
 * Add this breakdown to the utils_breakdown_stats_s provided in utils_breakdown_begin.
 *
 * \param bdown utils_breakdown_instance_s to hold this operations breakdown.
 * \param description Description of the current breakdown part
 */
void utils_breakdown_end(utils_breakdown_instance_s * bdown);

/**
 * Write \c stats to \c file, with additional info.
 *
 * \param file Append to this file
 * \param type Operation accelerator type
 * \param description Description of the operation
 * \param stats Breakdown to be written
 */
void utils_breakdown_write(const char *file,vine_accel_type_e type,const char * description,utils_breakdown_stats_s * stats);

/**
 * Return duration of current operation in ns, up until the last utils_breakdown_advance or utils_breakdown_end.
 *
 * \param bdown Breakdown whose duration we return.
 * \return Duration in nanoseconds.
 */
unsigned long long utils_breakdown_duration(utils_breakdown_instance_s * bdown);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#else

#include "compat.h"

typedef utils_compat_empty_s utils_breakdown_stats_s;

typedef utils_compat_empty_s utils_breakdown_instance_s;

#define utils_breakdown_init_stats(stats)

#define utils_breakdown_begin(bdown,stats,description)

#define utils_breakdown_advance(bdown,description)

#define utils_breakdown_end(bdown)

#define utils_breakdown_write(file,type,description,stats)

#endif /* ifdef BREAKS_ENABLE */

#endif /* ifdef UTILS_BREAKDOWN_HEADER */
