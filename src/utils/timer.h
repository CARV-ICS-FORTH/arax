#ifndef UTILS_TIMER_H
#define  UTILS_TRACE_H
#include <vine_talk.h>
#include <unistd.h>
#include <sys/time.h>

typedef struct
{
	struct timespec start;
	struct timespec stop;
}utils_timer_s;

/**
 * Set the start/stop time of \c NAME timer to the current time.
 *
 * \param NAME Name of a utils_timer variable
 * \param WHAT Can be start or stop
 */
#define utils_timer_set(NAME,WHAT) clock_gettime(0,&((NAME).WHAT))

/**
 * Get the start/stop time in microseconds of the \c NAME timer.
 *
 * \param NAME Name of a utils_timer variable
 * \param WHAT Can be start or stop
 * \return The requested timestamp in microseconds
 */
#define utils_timer_get_time_us(NAME,WHAT) \
	((NAME).WHAT.tv_sec*1000000+(NAME).WHAT.tv_nsec/1000)

/**
 * Get the start/stop time in nanoseconds of the \c NAME timer.
 *
 * \param NAME Name of a utils_timer variable
 * \param WHAT Can be start or stop
 * \return The requested timestamp in nanoseconds
 */
#define utils_timer_get_time_ns(NAME,WHAT) \
	((NAME).WHAT.tv_sec*1000000000+(NAME).WHAT.tv_nsec)

/**
 * Get the duration in microseconds of the \c NAME timer.
 *
 * \param NAME Name of a utils_timer variable
 */
#define utils_timer_get_duration_us(NAME) \
	(utils_timer_get_time_us(NAME,stop)-utils_timer_get_time_us(NAME,start))

/**
 * Get the duration in nanoseconds of the \c NAME timer.
 *
 * \param NAME Name of a utils_timer variable
 */
#define utils_timer_get_duration_ns(NAME) \
	(utils_timer_get_time_us(NAME,stop)-utils_timer_get_time_us(NAME,start))

#endif
