/**
 * This api tracing  vine_talk interface.
 * More specific for every function of
 * vine_talk interface we create a log enty (== line to csv file)
 * in a trace_buffer that holds infos about arguments of each function.
 *
 * When the buffer is full or the programm comes to end
 * the contents of trace_buffer are save to a csv file.
 *
 * One log entry has the following form:
 * <
 *   Timestamp,Core Id,Thread Id,Function Id,Task Duration,Return Value,
 *	info_about(arg_1_of_vine_function),..,info_about(arg_n_of_vine_function)
 * >
 *
 * For example a log entry for function vine_data_alloc is the following:
 * 251,5,7f78abb65740,vine_data_alloc,0,0x7f5f3b517e40,5,3
 *
 */
#ifndef UTILS_TRACE_H
#define  UTILS_TRACE_H
#include <vine_talk.h>
#include "timer.h"
#include <pthread.h>
#include <conf.h>
/**
 * We need to define trace enable before
 * include inorder to enable tracer.
 * Otherwise all calls to tracer they will skipped.
 */
#ifdef TRACE_ENABLE

/**
 * Called at program termination, flushes buffers to disk and releases buffers.
 */
void trace_exit();

/**
 * Initialization of tracer do the following:
 * 1) Starts the clock (usefull for timestamps).
 * 2) Init mutexes.
 * 3) Allocates place for log buffer.
 * 4) Opens log/trace File.
 */
void trace_init();

/**
 * Returns size of trace_buffer in Bytes.
 * Value is read from config key 'trace_buffer_size'.
 * @return size of trace_buffer in Bytes.
 */
int get_trace_buffer_size();

/**
 * @return 0 in case log buffer is not full
 *		 and 1 otherwise
 */
unsigned int is_trace_buffer_full();

/**
 * Returns the name of log buffer
 * which is in the following format.
 *  < trace_hostname_pid_date.csv >
 * @return  log file name
 */
char* get_trace_file_name();

/**
 * Opens log file.
 */
void open_trace_file();

/**
 * Update log file when buffer is full
 */
void update_trace_file();

/**
 * Creates a log entry for function vine_accel_list.
 *
 * @param type
 * @param accels
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void trace_vine_accel_list(vine_accel_type_e type, int physical,
						   vine_accel ***accels,const char *func_id,
						   utils_timer_s timing,int return_value);

/**
 * Creates a log entry for function vine_accel_location.
 *
 * @param accel
 * @param func_id
 * @param return_value
 * @param task_duration
 */
void trace_vine_accel_location(vine_accel *accel, const char *func_id,
                             vine_accel_loc_s return_value, utils_timer_s timing);

/**
 * Creates a log entry for function vine_accel_type.
 *
 * @param accel
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void trace_vine_accel_type(vine_accel *accel, const char *func_id,
                         utils_timer_s timing, int return_value);

/**
 * Creates a log entry for function vine_accel_stat.
 *
 * @param accel
 * @param stat
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void trace_vine_accel_stat(vine_accel *accel, vine_accel_stats_s *stat,
                         const char *func_id, utils_timer_s timing,
						 vine_accel_state_e return_value);


/**
 * Creates a log entry for function vine_accel_acquire.
 *
 * @param accel
 * @param func_id
 * @param task_duration
 */
void trace_vine_accel_acquire_phys(vine_accel *accel, const char *func_id,
                            utils_timer_s timing);

/**
 * Creates a log entry for function vine_accel_acquire.
 *
 * @param type
 * @param func_id
 * @param return_value
 * @param task_duration
 */
void trace_vine_accel_acquire_type(vine_accel_type_e type,
										   const char *func_id,
										   vine_accel * return_value,
										   utils_timer_s timing);

/**
 * Creates a log entry for function vine_accel_release.
 *
 * @param accel
 * @param func_id
 * @param task_duration
 */
void trace_vine_accel_release(vine_accel *accel, const char *func_id,
                            utils_timer_s timing);

/**
 * Creates a log entry for function vine_proc_register.
 *
 * @param type
 * @param proc_name
 * @param func_bytes
 * @param func_bytes_size
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void trace_vine_proc_register(vine_accel_type_e type, const char *proc_name,
                            const void *func_bytes, size_t func_bytes_size,
                            const char *func_id, utils_timer_s timing,
                            void *return_value);

/**
 * Creates a log entry for function vine_proc_get.
 *
 * @param type
 * @param func_name
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void trace_vine_proc_get(vine_accel_type_e type, const char *func_name,
                       const char *func_id, utils_timer_s timing,
                       vine_proc *return_value);


/**
 * Creates a log entry for function vine_proc_put.
 *
 * @param func
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void trace_vine_proc_put(vine_proc *func, const char *func_id, utils_timer_s timing,
                       int return_value);

/**
 * Creates a log entry for function vine_data_deref.
 *
 * @param data
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void trace_vine_data_deref(vine_data *data, const char *func_id,
                         utils_timer_s timing, void *return_value);

/**
 * Creates a log entry for function vine_task_issue.
 *
 * @param accel
 * @param proc
 * @param args
 * @param input
 * @param output
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void trace_vine_task_issue(vine_accel *accel, vine_proc *proc, vine_data *args,
                         size_t in_cnt, size_t out_cnt, vine_buffer_s *input,
						 vine_buffer_s *output, const char *func_id,
                         utils_timer_s timing, vine_task *return_value);

/**
 * @brief
 *
 * @param task
 * @param stats
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void trace_vine_task_stat(vine_task *task, vine_task_stats_s *stats,
                        const char *func_id, utils_timer_s timing,
                        vine_task_state_e return_value);

/**
 * Creates a log entry for function vine_task_wait.
 *
 * @param task
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void trace_vine_task_wait(vine_task *task, const char *func_id, utils_timer_s timing,
                        vine_task_state_e return_value);

/**
 * Creates a log entry for function vine_task_free.
 */
void trace_vine_task_free(vine_task * task,const char *func_id, utils_timer_s timing);

#define trace_timer_start(NAME) utils_timer_set(NAME ## _timer,start)

#define trace_timer_stop(NAME) utils_timer_set(NAME ## _timer,stop)

#define trace_timer_value(NAME) (NAME ## _timer)

#define TRACER_TIMER(NAME) utils_timer_s NAME ## _timer;

#else /* ifdef TRACE_ENABLE */

#define trace_exit()

#define trace_init()

#define trace_vine_accel_list(...)

#define trace_vine_accel_location(...)

#define trace_timer_start(...)
#define trace_timer_stop(...)

#define trace_vine_accel_type(...)

#define trace_vine_accel_stat(...)


#define trace_vine_accel_acquire_phys(...)
#define trace_vine_accel_acquire_type(...)

#define trace_vine_data_mark_ready(...)
#define trace_vine_accel_release(...)
#define trace_vine_proc_register(...)
#define trace_vine_proc_get(...)
#define trace_vine_proc_put(...)
#define trace_vine_data_alloc(...)
#define trace_vine_data_deref(...)
#define trace_vine_data_free(...)
#define trace_vine_task_issue(...)
#define trace_vine_task_stat(...)
#define trace_vine_task_wait(...)
#define trace_vine_task_free(...)

#define TRACER_TIMER(NAME)

#endif /* ifdef TRACE_ENABLE */
#endif /* ifndef UTILS_TRACE_H */
