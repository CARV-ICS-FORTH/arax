#ifndef UTILS_TIMER_H
#define  UTILS_TIMER_H

#include <vine_talk.h>
#include <unistd.h>

/**
 * Set the start/stop time of \c NAME timer to the current time.
 *
 * \param NAME Name of a utils_timer variable
 * \param WHAT Can be start or stop
 */
#define utils_timer_set(NAME,WHAT) clock_gettime(CLOCK_REALTIME,&((NAME).WHAT))

/**
 * Get the raw values of start/stop time of \c NAME timer.
 *
 * \param NAME Name of a utils_timer variable
 * \param WHAT Can be start or stop
 */
#define utils_timer_get_raw(NAME,WHAT) ((NAME).WHAT)

/**
 * Set the start/stop time of \c NAME timer for the raw values RAW.
 *
 * \param NAME Name of a utils_timer variable
 * \param WHAT Can be start or stop
 * \param RAW  Raw value of timer as returned by utils_timer_get_raw
 */
#define utils_timer_set_raw(NAME,WHAT,RAW) (NAME).WHAT=RAW

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
	(((NAME).stop.tv_sec-(NAME).start.tv_sec)*1000000+((NAME).stop.tv_nsec-(NAME).start.tv_nsec)/1000)

/**
 * Get the duration in nanoseconds of the \c NAME timer.
 *
 * \param NAME Name of a utils_timer variable
 */
#define utils_timer_get_duration_ns(NAME) \
	(((NAME).stop.tv_sec-(NAME).start.tv_sec)*1000000000+((NAME).stop.tv_nsec-(NAME).start.tv_nsec))

#endif
