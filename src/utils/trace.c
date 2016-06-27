#define _GNU_SOURCE
#ifndef TRACE_ENABLE
#define TRACE_ENABLE
#endif /* TRACE_ENABLE */
#include "trace.h"
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>
#include <stdio.h>
#include <math.h>
#include <linux/limits.h>
#include <sched.h>
#include <signal.h>
#include  "config.h"

/**
 * One log entry contains in	formation
 * for one subset of those values.
 **/
struct Entry {
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
};

int             curr_entry_pos;
int             trace_buffer_size;
trace_entry       *trace_buffer_start_ptr;
int             trace_file;
pthread_mutex_t lock;
size_t          start_of_time;
sighandler_t    prev_sighandler;


void signal_callback_handler(int signum)
{
	tracer_exit();
	/* Call previous signal handler */
	prev_sighandler(signum);
}

void tracer_exit()
{
	if (trace_buffer_start_ptr != NULL) {
		/* locks here is usefull if user stops programm using C-c */
		pthread_mutex_lock(&lock);

		update_trace_file();
		free(trace_buffer_start_ptr);
		trace_buffer_start_ptr = NULL;
		close(trace_file);

		pthread_mutex_unlock(&lock);

		pthread_mutex_destroy(&lock);
	}
}

void tracer_init()
{
	struct timeval tv;

	gettimeofday(&tv, NULL);

	if (pthread_mutex_init(&lock, NULL) != 0) {
		fprintf(stderr, "TRACER: mutex init failed\n");
		exit(-1);
	}

	prev_sighandler = signal(SIGINT, signal_callback_handler);

	trace_buffer_size      = get_trace_buffer_size();
	trace_buffer_start_ptr = malloc(trace_buffer_size);
	curr_entry_pos       = -1;
	open_trace_file();
	start_of_time      = tv.tv_sec*1000000+tv.tv_usec;
}

int get_trace_buffer_size()
{
	int trace_buffer_size;

	util_config_get_int("trace_buffer_size", &trace_buffer_size,
	                    sizeof(trace_entry)*100);
	assert(trace_buffer_size);

	return trace_buffer_size;
}

char* get_trace_file_name()
{
	char           hostname[1024];
	char           buffer[30];
	char           trace_path[1024];
	struct timeval tv;
	time_t         curtime;
	char           fileName[2078];

	/* after log file is created , we must not call this function*/
	/* assert(trace_file == 0); */

	hostname[1023] = '\0';
	gethostname(hostname, 1023);

	gettimeofday(&tv, NULL);
	curtime = tv.tv_sec;

	if(!util_config_get_str("trace_path",trace_path,1024))
	{
		trace_path[0] = '.';
		trace_path[1] = '\0';
	}

	strftime( buffer, 30, "%m-%d-%Y-%T", localtime(&curtime) );
	snprintf(fileName,2078, "%s/trace_%s_%d_%s.csv",trace_path, hostname, getpid(), buffer);

	return strdup(fileName);
}

void open_trace_file()
{
	char *fileName = get_trace_file_name();

	trace_file = open(fileName, O_CREAT|O_RDWR, 0600); /*global var*/
	if (trace_file < 0) {
		perror("TRACER: open syscall failed ");
		exit(-1);
	}
	free(fileName);
	dprintf(trace_file, "%s\n\n",
	        "Timestamp,Core Id,Thread Id,Function Id,Task Duration,Return Value");
}

void update_trace_file()
{
	print_trace_buffer_to_fd(trace_file);
	fsync(trace_file);
}

unsigned int is_trace_buffer_full()
{
	int total_trace_entries = trace_buffer_size/sizeof(trace_entry);

	return curr_entry_pos >= (total_trace_entries-1);
}

/** One log entry has the following form:
 * <
 *   Timestamp,Core Id,Thread Id,Function Id,Task Duration,Return Value,
 *	info_about(arg_1_of_vine_function),..,info_about(arg_n_of_vine_function)
 * >
 *
 * For example a log entry for function vine_data_alloc is the following:
 * 251,5,7f78abb65740,vine_data_alloc,0,0x7f5f3b517e40,5,3
 */
void print_trace_entry_to_fd(int fd, trace_entry *entry)
{
	int i = 0;

	dprintf(fd, "%zu,%d,%lx,%s,%zu", entry->timestamp, entry->core_id,
	        entry->thread_id, entry->func_id, entry->task_duration);

	/*
	 *  in those functions that return value is int
	 *  prints to trace file its value otherwise
	 *  prints adress of pointer.
	 */
	if ( !strcmp(entry->func_id, "vine_accel_list")		||
	     !strcmp(entry->func_id, "vine_accel_type")		||
	     !strcmp(entry->func_id, "vine_accel_location")	||
	     !strcmp(entry->func_id, "vine_accel_stat")		||
	     !strcmp(entry->func_id, "vine_accel_acquire")	||
	     !strcmp(entry->func_id, "vine_proc_put")		||
	     !strcmp(entry->func_id, "vine_task_stat")		||
	     !strcmp(entry->func_id, "vine_task_wait")		||
		 !strcmp(entry->func_id, "trace_vine_data_check_ready")
	) {
		int ret_val = entry->return_value.i;

		dprintf(fd, ",%d", ret_val);
	} else {
		dprintf(fd, ",%p", entry->return_value.p);
	}


	if (entry->accel)
		dprintf(fd, ",%p", entry->accel);
	if (entry->accel_stat)
		dprintf(fd, ",%p", entry->accel_stat);
	if (entry->accel_type != -1)
		dprintf(fd, ",%d", entry->accel_type);
	if (entry->func_name)
		dprintf(fd, ",%p", entry->func_name);
	if (entry->func_bytes)
		dprintf(fd, ",%p", entry->func_bytes);
	if (entry->func_bytes_size)
		dprintf(fd, ",%zu", entry->func_bytes_size);
	if (entry->func)
		dprintf(fd, ",%p", entry->func);

	if ( entry->data_size && (entry->data == 0) )
		dprintf(fd, ",%zu", entry->data_size);
	if (entry->accel_place != -1)
		dprintf(fd, ",%d", entry->accel_place);
	if (entry->accels)
		dprintf(fd, ",%p", entry->accels);

	if (entry->data)
		dprintf(fd, ",%p", entry->data);
	if (entry->data_size && entry->data)
		dprintf(fd, ":%zu", entry->data_size);

	if (entry->args)
		dprintf(fd, ",%p", entry->args);
	if (entry->in_cnt)
		dprintf(fd, ",%zu", entry->in_cnt);
	if (entry->in_data)
		dprintf(fd, ",%p", entry->in_data);
	for (i = 0; i < entry->in_cnt; ++i) {
		dprintf(fd, ",%p", entry->in_data[i]);
	}
	if (entry->out_cnt)
		dprintf(fd, ",%zu", entry->out_cnt);
	if (entry->out_data)
		dprintf(fd, ",%p", entry->out_data);
	for (i = 0; i < entry->out_cnt; ++i) {
		dprintf(fd, ",%p", entry->out_data[i]);
	}
	if (entry->task)
		dprintf(fd, ",%p", entry->task);
	if (entry->task_stats)
		dprintf(fd, ",%p", entry->task_stats);
	dprintf(fd, "\n");
}                  /* print_trace_entry_to_fd */

void print_trace_buffer_to_fd()
{
	int i;

	for (i = 0; i <= curr_entry_pos; i++) {
		print_trace_entry_to_fd(trace_file, &trace_buffer_start_ptr[i]);
	}
}

void init_trace_entry(trace_entry *entry)
{
	memset( entry, 0, sizeof(trace_entry) );
	entry->accel_type  = -1;
	entry->accel_place = -1;

	struct timeval tv;

	gettimeofday(&tv, NULL);
	entry->timestamp = (tv.tv_sec*1000000+tv.tv_usec) - start_of_time;
	entry->core_id   = sched_getcpu();
	entry->thread_id = pthread_self();
}

void _trace_timer_start(struct timeval *t1)
{
	gettimeofday(t1, NULL);
}

useconds_t _trace_timer_stop(struct timeval *t2, struct timeval *t1)
{
	gettimeofday(t2, NULL);

	int elapsedTime;

	elapsedTime  = (t2->tv_sec - t1->tv_sec) * 1000000;
	elapsedTime += (t2->tv_usec - t1->tv_usec);

	return elapsedTime;
}

trace_entry* get_trace_buffer_ptr()
{
	trace_entry* entry;

	pthread_mutex_lock(&lock);
	if ( is_trace_buffer_full() ) {
		update_trace_file();
		curr_entry_pos = -1;
	}
	entry = &trace_buffer_start_ptr[++curr_entry_pos];
	pthread_mutex_unlock(&lock);

	init_trace_entry(entry);

	return entry;
}

void trace_vine_accel_list(vine_accel_type_e type, vine_accel ***accels,
						 const char *func_id, int task_duration,
						 int return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->accel_type     = type;
	entry->accels         = accels;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_value;
}

void trace_vine_accel_stat(vine_accel *accel, vine_accel_stats_s *stat,
						 const char *func_id, int task_duration,
						 vine_accel_state_e return_val)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->accel          = accel;
	entry->accel_stat     = stat;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_val;
}

void trace_vine_accel_location(vine_accel *accel, const char *func_id,
							 vine_accel_loc_s return_val, int task_duration)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->accel         = accel;
	entry->func_id       = func_id;
	entry->task_duration = task_duration;
	/*	entry->return_value  = &return_val; // Reference of stack value */
}

void trace_vine_accel_type(vine_accel *accel, const char *func_id,
						 int task_duration, int return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->accel          = accel;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_value;
}

void trace_vine_task_stat(vine_task *task, vine_task_stats_s *stats,
						const char *func_id, int task_duration,
						vine_task_state_e return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->task           = task;
	entry->task_stats     = stats;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_value;
}

void trace_vine_accel_acquire(vine_accel *accel, const char *func_id,
							int return_val, int task_duration)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->accel          = accel;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_val;
}

void trace_vine_accel_release(vine_accel *accel, const char *func_id,
							int return_val, int task_duration)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->accel          = accel;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_val;
}

void trace_vine_proc_register(vine_accel_type_e type, const char *proc_name,
							const void *func_bytes, size_t func_bytes_size,
							const char *func_id, int task_duration,
							void *return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->func_id         = func_id;
	entry->task_duration   = task_duration;
	entry->accel_type      = type;
	entry->func_name       = proc_name;
	entry->func_bytes      = func_bytes;
	entry->func_bytes_size = func_bytes_size;
	entry->return_value.p  = return_value;
}

void trace_vine_proc_get(vine_accel_type_e type, const char *func_name,
                       const char *func_id, int task_duration,
                       vine_proc *return_val)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->accel_type     = type;
	entry->func_name      = func_name;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.p = return_val;
}

void trace_vine_proc_put(vine_proc *func, const char *func_id, int task_duration,
                       int return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->func           = func;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_value;
}

void trace_vine_data_alloc(size_t size, vine_data_alloc_place_e place,
                         int task_duration, const char *func_id,
                         void *return_val)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->data_size      = size;
	entry->accel_place    = place;
	entry->task_duration  = task_duration;
	entry->func_id        = func_id;
	entry->return_value.p = return_val;
}

void trace_vine_data_mark_ready(vine_data *data, const char *func_id,
                              int task_duration)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->data           = data;
	entry->data_size      = vine_data_size(data);
	entry->task_duration  = task_duration;
	entry->func_id        = func_id;
	entry->return_value.p = NULL;
}

void trace_vine_data_check_ready(vine_data *data, const char *func_id,
								 int task_duration,int return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->data           = data;
	entry->data_size      = vine_data_size(data);
	entry->task_duration  = task_duration;
	entry->func_id        = func_id;
	entry->return_value.i = return_value;
}

void trace_vine_data_deref(vine_data *data, const char *func_id,
                         int task_duration, void *return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->data           = data;
	entry->data_size      = vine_data_size(data);
	entry->task_duration  = task_duration;
	entry->func_id        = func_id;
	entry->return_value.p = return_value;
}

void trace_vine_data_free(vine_data *data, const char *func_id, int task_duration)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->data          = data;
	entry->task_duration = task_duration;
	entry->func_id       = func_id;
}

void trace_vine_task_issue(vine_accel *accel, vine_proc *proc, vine_data *args,
                         size_t in_cnt, size_t out_cnt, vine_data **input,
                         vine_data **output, const char *func_id,
                         int task_duration, vine_task *return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();

	entry->accel          = accel;
	entry->func           = proc;
	entry->args           = args;
	entry->in_data        = input;
	entry->out_data       = output;
	entry->in_cnt         = in_cnt;
	entry->out_cnt        = out_cnt;
	entry->task_duration  = task_duration;
	entry->func_id        = func_id;
	entry->return_value.p = return_value;
}

void trace_vine_task_wait(vine_task *task, const char *func_id, int task_duration,
                        vine_task_state_e return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->task           = task;
	entry->task_duration  = task_duration;
	entry->func_id        = func_id;
	entry->return_value.i = return_value;
}
