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

/** Use with utils_timer_tv_time() to get time in milliseconds */
#define UTILS_TIMER_MS 1000000
/** Use with utils_timer_tv_time() to get time in microseconds */
#define UTILS_TIMER_US 1000
/** Use with utils_timer_tv_time() to get time in nanoseconds */
#define UTILS_TIMER_NS 1

/**
 * Convert \c TV struct to time in \c SCALE units.
 *
 * SCALE = 1000 => microseconds
 * SCALE = 1    => nanoseconds
 */
#define utils_timer_tv_time(TV,SCALE) \
	(((TV).tv_sec*(1000000000/SCALE))+(TV).tv_nsec/(SCALE))

/**
 * Get the start/stop time in microseconds of the \c NAME timer.
 *
 * \param NAME Name of a utils_timer variable
 * \param WHAT Can be start or stop
 * \return The requested timestamp in microseconds
 */
#define utils_timer_get_time_us(NAME,WHAT) \
	utils_timer_tv_time((NAME).WHAT,UTILS_TIMER_US)

/**
 * Get the start/stop time in nanoseconds of the \c NAME timer.
 *
 * \param NAME Name of a utils_timer variable
 * \param WHAT Can be start or stop
 * \return The requested timestamp in nanoseconds
 */
#define utils_timer_get_time_ns(NAME,WHAT) \
	utils_timer_tv_time((NAME).WHAT,UTILS_TIMER_NS)

/**
 * Get the duration in microseconds of the \c NAME timer.
 *
 * \param NAME Name of a utils_timer variable
 */
#define utils_timer_get_duration_us(NAME) 				\
	(utils_timer_tv_time((NAME).stop,UTILS_TIMER_US)	\
	-utils_timer_tv_time((NAME).start,UTILS_TIMER_US))

/**
 * Get the duration in nanoseconds of the \c NAME timer.
 *
 * \param NAME Name of a utils_timer variable
 */
#define utils_timer_get_duration_ns(NAME) \
	(utils_timer_tv_time((NAME).stop,UTILS_TIMER_NS)	\
	-utils_timer_tv_time((NAME).start,UTILS_TIMER_NS))	\

	/**
	 * Get the current elapsed time since timer start time in microseconds.
	 *
	 * \param NAME Name of a utils_timer variable
	 */
#define utils_timer_get_elapsed_us(NAME)		\
	({											\
		struct timespec now;					\
		clock_gettime(CLOCK_REALTIME,&now);		\
		utils_timer_tv_time(now,UTILS_TIMER_US)\
		-utils_timer_get_time_us(NAME,start);	\
	})

	/**
	 * Get the current elapsed time since timer start time in nanoseconds.
	 *
	 * \param NAME Name of a utils_timer variable
	 */
#define utils_timer_get_elapsed_ns(NAME)		\
	({											\
		struct timespec now;					\
		clock_gettime(CLOCK_REALTIME,&now);		\
		utils_timer_tv_time(now,UTILS_TIMER_NS)\
		-utils_timer_get_time_ns(NAME,start);	\
	})

#endif
