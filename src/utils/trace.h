/**
 * This api tracing  vine_talk interface.
 * More specific for every function of
 * vine_talk interface we create a log enty (== line to csv file)
 * in a log_buffer that holds infos about arguments of each function.
 *
 * When the buffer is full or the programm comes to end
 * the contents of log_buffer are save to a csv file.
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
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>

/**
 * We need to define trace enable before
 * include inorder to enable profiler.
 * Otherwise all calls to profiler they will skipped.
 */
#ifdef TRACE_ENABLE

/**
 * One log entry contains information
 * for one subset of those values.
 **/
typedef struct Entry {
	size_t     timestamp;
	int        core_id;
	pthread_t  thread_id;
	const char *func_id;
	size_t     task_duration;

	union {
		void *p;
		int  i;
	} return_value;

	vine_accel              ***accels;
	vine_accel              *accel;
	vine_accel_stats_s      *accel_stat;
	vine_accel_type_e       accel_type;
	const char              *func_name;
	const void              *func_bytes;
	size_t                  func_bytes_size;
	vine_proc               *func;
	vine_data_alloc_place_e accel_place;
	vine_data               *data;
	size_t                  data_size;
	size_t                  in_cnt;
	size_t                  out_cnt;
	vine_data               *args;
	vine_data               **in_data;
	vine_data               **out_data;
	vine_task               *task;
	vine_task_stats_s       *task_stats;
} log_entry;

int             curr_entry_pos;
int             log_buffer_size;
log_entry       *log_buffer_start_ptr;
int             log_file;
unsigned int    log_buffer_is_full;
pthread_mutex_t lock;
size_t          start_of_time;

/**
 * This function working like a destructor.
 * More specific when profiler ends we need
 * to write log_buffer to csv file and free log_buffer,
 * so destructor calls close_profiler in order to do that.
 */
void profiler_destructor();

/**
 * This function working like a constructor.
 * More specific when profiler start we need
 * to do some initialiazations,
 * so constructor calls init_profiler in order to do that.
 */
void profiler_constructor();

/**
 * This function is necessary in case that
 * user does ctrl-c.If that happend we
 * call destructor inorder to close properly.
 */
void signal_callback_handler(int signum);

/**
 * Every logging function(log_vine_*) calls
 * get_log_bugger_ptr in order to get
 * a ptr in log_buffer.
 *
 * If log_buffer is full ,update trace file
 * and flushes buffer and then return the
 * the first position of buffer.
 *
 * @return a pointer	to a empty position in log_buffer
 */
log_entry* get_log_buffer_ptr();


/**
 * Sets log_entry values to be empty.
 * And initialized the folowing values of log_enty:
 * timestamp,core_id,thread_id.
 *
 * @param entry
 */
void init_log_entry(log_entry *entry);

/**
 * Reads from vine_profiler.conf
 * size of log_buffer in Bytes.
 * @return size of log_buffer in Bytes.
 */
int get_log_buffer_size();

/**
 * @return 0 in case log buffer is not full
 *		 and 1 otherwise
 */
unsigned int is_log_buffer_full();

/**
 * Returns the name of log buffer
 * which is in the following format.
 *  < trace_hostname_pid_date.csv >
 * @return  log file name
 */
char* get_log_file_name();

/**
 * Initialization of profiler do the following:
 * 1) Starts the clock (usefull for timestamps).
 * 2) Init mutexes.
 * 3) Allocates place for log buffer.
 * 4) Opens log/trace File.
 */
void init_profiler();

/**
 * Opens log file.
 */
void open_log_file();

/**
 * Update log file when buffer is full
 */
void update_log_file();

/**
 * Creates a log entry for function vine_accel_list.
 *
 * @param type
 * @param accels
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void log_vine_accel_list(vine_accel_type_e type, vine_accel ***accels,
                         const char *func_id, int task_duration,
                         void *return_value);

/**
 * Creates a log entry for function vine_accel_location.
 *
 * @param accel
 * @param func_id
 * @param return_val
 * @param task_duration
 */
void log_vine_accel_location(vine_accel *accel, const char *func_id,
                             vine_accel_loc_s return_val, int task_duration);

/**
 * Creates a log entry for function vine_accel_type.
 *
 * @param accel
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void log_vine_accel_type(vine_accel *accel, const char *func_id,
                         int task_duration, int return_value);

/**
 * Creates a log entry for function vine_accel_stat.
 *
 * @param accel
 * @param stat
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void log_vine_accel_stat(vine_accel *accel, vine_accel_stats_s *stat,
                         const char *func_id, int task_duration,
                         void *return_value);


/**
 * Creates a log entry for function vine_accel_acquire.
 *
 * @param accel
 * @param func_id
 * @param return_val
 * @param task_duration
 */
void log_vine_accel_acquire(vine_accel *accel, const char *func_id,
                            int return_val, int task_duration);

/**
 * Creates a log entry for function vine_accel_release.
 *
 * @param accel
 * @param func_id
 * @param return_val
 * @param task_duration
 */
void log_vine_accel_release(vine_accel *accel, const char *func_id,
                            int return_val, int task_duration);

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
void log_vine_proc_register(vine_accel_type_e type, const char *proc_name,
                            const void *func_bytes, size_t func_bytes_size,
                            const char *func_id, int task_duration,
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
void log_vine_proc_get(vine_accel_type_e type, const char *func_name,
                       const char *func_id, int task_duration,
                       vine_proc *return_value);


/**
 * Creates a log entry for function vine_proc_put.
 *
 * @param func
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void log_vine_proc_put(vine_proc *func, const char *func_id, int task_duration,
                       int return_value);

/**
 * Creates a log entry for function vine_data_alloc.
 *
 * @param size
 * @param place
 * @param task_duration
 * @param func_id
 * @param return_val
 */
void log_vine_data_alloc(size_t size, vine_data_alloc_place_e place,
                         int task_duration, const char *func_id,
                         vine_data *return_val);


/**
 * Create a log entry for function vine_data_mark_ready
 *
 * @param data
 * @param func_id
 * @param task_duration
 */
void log_vine_data_mark_ready(vine_data *data, const char *func_id,
                              int task_duration);

/**
 * Create a log entry for function vine_data_free
 *
 * @param data
 * @param func_id
 * @param task_duration
 */
void log_vine_data_free(vine_data *data, const char *func_id,
                        int task_duration);

/**
 * Creates a log entry for function vine_data_deref.
 *
 * @param data
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void log_vine_data_deref(vine_data *data, const char *func_id,
                         int task_duration, void *return_value);

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
void log_vine_task_issue(vine_accel *accel, vine_proc *proc, vine_data *args,
                         size_t in_cnt, size_t out_cnt, vine_data **input,
                         vine_data **output, const char *func_id,
                         int task_duration, vine_task *return_value);

/**
 * @brief
 *
 * @param task
 * @param stats
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void log_vine_task_stat(vine_task *task, vine_task_stats_s *stats,
                        const char *func_id, int task_duration,
                        vine_task_state_e return_value);

/**
 * Creates a log entry for function vine_task_wait.
 *
 * @param task
 * @param func_id
 * @param task_duration
 * @param return_value
 */
void log_vine_task_wait(vine_task *task, const char *func_id, int task_duration,
                        vine_task_state_e return_value);

/**
 * Usefull for debugging,print log_buffer.
 *
 * @param FILE
 */
void debug_print_log_buffer(FILE*);

/**
 * Usefull for debugging,prints log_entry.
 *
 * @param FILE
 * @param entry
 */
void debug_print_log_entry(FILE*, log_entry *entry);

/**
 *	Prints log buffer to file descriptor.
 */
void print_log_buffer_to_fd();

/**
 * Prints log entry to file descriptor.
 * @param fd
 * @param entry
 */
void print_log_entry_to_fd(int fd, log_entry *entry);

/**
 * this fuction called from destructor,
 * and do the folowings:
 * 1)Writes log_buffer to trace file
 * 2)Deallocates log_buffer
 */
void close_profiler();

/**
 * Takes time and save it at given value t1.
 * This is usefull inorder to start timer.
 * @param t1
 */
void _log_timer_start(struct timeval *t1);

#define log_timer_start(NAME) _log_timer_start( &(NAME ## _start) )
/**
 * Returns time in ms.
 * @param t1: takes argument that function log_timer_start initialize.
 * @param t2
 *
 * @return: duration between calls log_timer_start and log_timer_stop
 */
int _log_timer_stop(struct timeval *t1, struct timeval *t2);

#define log_timer_stop(NAME)                                \
	task_duration = _log_timer_stop( &(NAME ## _start), \
	                                 &(NAME ## _stop) )

#define TRACER_TIMER(NAME)                            \
	struct timeval NAME ## _start, NAME ## _stop; \
	int            NAME ## _duration;

#else /* ifdef TRACE_ENABLE */

#define log_vine_accel_list(...)

#define log_vine_accel_location(...)

#define log_timer_start(...)
#define log_timer_stop(...)

#define log_vine_accel_type(...)

#define log_vine_accel_stat(...)


#define log_vine_accel_acquire(...)

#define log_vine_data_mark_ready(...)
#define log_vine_accel_release(...)
#define log_vine_proc_register(...)
#define log_vine_proc_get(...)
#define log_vine_proc_put(...)
#define log_vine_data_alloc(...)
#define log_vine_data_deref(...)
#define log_vine_data_free(...)
#define log_vine_task_issue(...)
#define log_vine_task_stat(...)
#define log_vine_task_wait(...)

#define TRACER_TIMER(NAME)

#endif /* ifdef TRACE_ENABLE */
#endif /* ifndef UTILS_TRACE_H */
