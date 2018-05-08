#ifndef UTILS_BREAKDOWN_HEADER
#define UTILS_BREAKDOWN_HEADER
#include "conf.h"
#ifdef BREAKS_ENABLE
#include "timer.h"
/**
 * Number of max parts allowed in a single breakdown.
 */
#define BREAKDOWN_PARTS 16

/**
 * Keeps aggregate time of numerous utils_breakdown_instance_s.
 */
typedef struct{
	unsigned long long samples;					//< Number of breakdowns
	unsigned long long last;					//< time of last task instance.
	unsigned long long part[BREAKDOWN_PARTS+2];	//< Duration in ns of each part(+sum+iat)
	const char * desc[BREAKDOWN_PARTS];			//< Description for each part
	char heads[BREAKDOWN_PARTS*64];				//< Storage for headers.
	int head_append;							//< Append position in heads.
}utils_breakdown_stats_s __attribute__((aligned(CONF_CACHE_LINE)));

/**
 * Keeps duration of an individual operation.
 */
typedef struct{
	utils_timer_s timer;						//< Timer used for counting duration
	void * vaccel;								//< Owner VAQ(job)
	void * paccel;								//< Physical accelerator(Place of Execution)
	unsigned long long start;					//< Start time of this task/instance
	unsigned long long part[BREAKDOWN_PARTS+1];	//< Duration in ns of each part,sum at the end
	utils_breakdown_stats_s * stats;			//< Aggregate statistics
	int current_part;							//< Currently measured part
	int first;									//< 0 if the first instance of a kernel
}utils_breakdown_instance_s;

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

#ifdef VINE_TELEMETRY
void utils_breakdown_init_telemetry(char * conf);

inline static void utils_breakdown_instance_set_vaccel(utils_breakdown_instance_s * bdown,void * accel)
{
	bdown->vaccel = accel;
}

inline static void utils_breakdown_instance_set_paccel(utils_breakdown_instance_s * bdown,void * accel)
{
	bdown->paccel = accel;
}
#else

#define utils_breakdown_init_telemetry(CONF)

#define utils_breakdown_instance_set_vaccel(BDOWN,ACCEL)

#define utils_breakdown_instance_set_paccel(BDOWN,ACCEL)

#endif

/**
 * Initialize utils_breakdown_stats_s \c stats.
 *
 * \param stats utils_breakdown_stats_s to be initialized.
 */
void utils_breakdown_init_stats(utils_breakdown_stats_s * stats);

/**
 * Initialize/reset \c bdown instance.
 */
void utils_breakdown_instance_init(utils_breakdown_instance_s * bdown);

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

#define utils_breakdown_instance_init(bdown)

#define utils_breakdown_begin(bdown,stats,description)

#define utils_breakdown_advance(bdown,description)

#define utils_breakdown_end(bdown)

#define utils_breakdown_write(file,type,description,stats)

#define utils_breakdown_init_telemetry(CONF)

#define utils_breakdown_instance_set_vaccel(BDOWN,ACCEL)

#define utils_breakdown_instance_set_paccel(BDOWN,ACCEL)

#endif /* ifdef BREAKS_ENABLE */

#endif /* ifdef UTILS_BREAKDOWN_HEADER */
