#ifndef UTILS_TIMER_H
#define  UTILS_TRACE_H
#include <vine_talk.h>
#include <unistd.h>
#include <sys/time.h>

typedef struct
{
	struct timeval start;
	struct timeval stop;
}utils_timer_s;

/**
 * Set the start/stop time of \c NAME timer to the current time.
 *
 * \param NAME Name of a utils_timer variable
 * \param WHAT Can be start or stop
 */
#define utils_timer_set(NAME,WHAT) gettimeofday(&((NAME).WHAT), NULL)

/**
 * Get the start/stop time in microseconds of the \c NAME timer.
 *
 * \param NAME Name of a utils_timer variable
 * \param WHAT Can be start or stop
 * \return The requested timestamp in microseconds
 */
#define utils_timer_get_time(NAME,WHAT) ((NAME).WHAT.tv_sec*1000000+(NAME).WHAT.tv_usec)

/**
 * Get the duration in microseconds of the \c NAME timer.
 *
 * \param NAME Name of a utils_timer variable
 */
#define utils_timer_get_duration(NAME) (utils_timer_get_time(NAME,stop)-utils_timer_get_time(NAME,start))
#endif
