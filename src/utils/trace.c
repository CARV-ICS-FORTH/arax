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
 * One log entry contains information
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
int             log_buffer_size;
log_entry       *log_buffer_start_ptr;
int             log_file;
pthread_mutex_t lock;
size_t          start_of_time;
sighandler_t    prev_sighandler;


void signal_callback_handler(int signum)
{
	profiler_destructor();
	/* Call previous signal handler */
	prev_sighandler(signum);
}

void profiler_constructor(void)
{
	/* Store old signal handler */
	prev_sighandler = signal(SIGINT, signal_callback_handler);
	init_profiler();
}

void profiler_destructor(void)
{
	close_profiler();
	pthread_mutex_destroy(&lock);
}

void init_profiler()
{
	struct timeval tv;

	gettimeofday(&tv, NULL);

	if (pthread_mutex_init(&lock, NULL) != 0) {
		fprintf(stderr, "PROFILER: mutex init failed\n");
		exit(-1);
	}

	log_buffer_size      = get_log_buffer_size();
	log_buffer_start_ptr = malloc(log_buffer_size);
	curr_entry_pos       = -1;
	open_log_file();
	start_of_time      = tv.tv_sec*1000000+tv.tv_usec;
}

int get_log_buffer_size()
{
	int log_buffer_size;

	util_config_get_int("log_buffer_size", &log_buffer_size,
	                    sizeof(log_entry)*100);
	assert(log_buffer_size);

	return log_buffer_size;
}

char* get_log_file_name()
{
	char           hostname[1024];
	char           buffer[30];
	struct timeval tv;
	time_t         curtime;
	char           fileName[100];

	/* after log file is created , we must not call this function*/
	/* assert(log_file == 0); */

	hostname[1023] = '\0';
	gethostname(hostname, 1023);

	gettimeofday(&tv, NULL);
	curtime = tv.tv_sec;

	strftime( buffer, 30, "%m-%d-%Y-%T", localtime(&curtime) );
	sprintf(fileName, "trace_%s_%d_%s.csv", hostname, getpid(), buffer);

	return strdup(fileName);
}

void open_log_file()
{
	char *fileName = get_log_file_name();

	log_file = open(fileName, O_CREAT|O_RDWR, 0600); /*global var*/
	if (log_file < 0) {
		perror("PROFILER: open syscall failed ");
		exit(-1);
	}
	free(fileName);
	dprintf(log_file, "%s\n\n",
	        "Timestamp,Core Id,Thread Id,Function Id,Task Duration,Return Value");
}

void close_profiler()
{
	if (log_buffer_start_ptr != NULL) {
		/* locks here is usefull if user stops programm using C-c */
		pthread_mutex_lock(&lock);

		update_log_file();
		free(log_buffer_start_ptr);
		log_buffer_start_ptr = NULL;
		close(log_file);

		pthread_mutex_unlock(&lock);
	}
}

void update_log_file()
{
	print_log_buffer_to_fd(log_file);
	fsync(log_file);
}

void log_vine_accel_list(vine_accel_type_e type, vine_accel ***accels,
                         const char *func_id, int task_duration,
                         int return_value)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);

	init_log_entry(entry);

	entry->accel_type     = type;
	entry->accels         = accels;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_value;
}

void log_vine_accel_stat(vine_accel *accel, vine_accel_stats_s *stat,
                         const char *func_id, int task_duration,
						 vine_accel_state_e return_val)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->accel          = accel;
	entry->accel_stat     = stat;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_val;
}

void log_vine_accel_location(vine_accel *accel, const char *func_id,
                             vine_accel_loc_s return_val, int task_duration)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->accel         = accel;
	entry->func_id       = func_id;
	entry->task_duration = task_duration;
/*	entry->return_value  = &return_val; // Reference of stack value */
}

void log_vine_accel_type(vine_accel *accel, const char *func_id,
                         int task_duration, int return_value)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->accel          = accel;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_value;
}

void log_vine_task_stat(vine_task *task, vine_task_stats_s *stats,
                        const char *func_id, int task_duration,
                        vine_task_state_e return_value)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->task           = task;
	entry->task_stats     = stats;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_value;
}

void log_vine_accel_acquire(vine_accel *accel, const char *func_id,
                            int return_val, int task_duration)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->accel          = accel;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_val;
}

void log_vine_accel_release(vine_accel *accel, const char *func_id,
                            int return_val, int task_duration)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->accel          = accel;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_val;
}

void log_vine_proc_register(vine_accel_type_e type, const char *proc_name,
                            const void *func_bytes, size_t func_bytes_size,
                            const char *func_id, int task_duration,
                            void *return_value)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->func_id         = func_id;
	entry->task_duration   = task_duration;
	entry->accel_type      = type;
	entry->func_name       = proc_name;
	entry->func_bytes      = func_bytes;
	entry->func_bytes_size = func_bytes_size;
	entry->return_value.p  = return_value;
}

unsigned int is_log_buffer_full()
{
	int total_log_entries = log_buffer_size/sizeof(log_entry);

	return curr_entry_pos >= (total_log_entries-1);
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
void print_log_entry_to_fd(int fd, log_entry *entry)
{
	int i = 0;

	dprintf(fd, "%zu,%d,%lx,%s,%zu", entry->timestamp, entry->core_id,
	        entry->thread_id, entry->func_id, entry->task_duration);

	/*
	 *  in those functions that return value is int
	 *  prints to trace file its value otherwise
	 *  prints adress of pointer.
	 */
	if ( !strcmp(entry->func_id,
	             "vine_accel_list") ||
	     !strcmp(entry->func_id,
	             "vine_accel_type") ||
	     !strcmp(entry->func_id,
	             "vine_accel_location") ||
	     !strcmp(entry->func_id,
	             "vine_accel_stat") ||
	     !strcmp(entry->func_id,
	             "vine_accel_acquire") ||
	     !strcmp(entry->func_id,
	             "vine_proc_put") ||
	     !strcmp(entry->func_id,
	             "vine_task_stat") ||
	     !strcmp(entry->func_id, "vine_task_wait") ) {
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
}                  /* print_log_entry_to_fd */

void print_log_buffer_to_fd()
{
	int i;

	for (i = 0; i <= curr_entry_pos; i++) {
		print_log_entry_to_fd(log_file, &log_buffer_start_ptr[i]);
	}
}

void debug_print_log_entry(FILE *fstream, log_entry *entry)
{
	fprintf(fstream, "%zu,%d,%lu,%s,%zu,%p", entry->timestamp,
	        entry->core_id, entry->thread_id, entry->func_id,
	        entry->task_duration, entry->return_value.p);
	if (entry->accels)
		printf(",%p", entry->accels);

	if (entry->accel)
		fprintf(fstream, ",%p", entry->accel);
	if (entry->accel_stat)
		fprintf(fstream, ",%p", entry->accel_stat);
	if (entry->accel_type != -1)
		fprintf(fstream, ",%d", entry->accel_type);
	if (entry->func_name)
		fprintf(fstream, ",%p", entry->func_name);
	if (entry->func_bytes)
		fprintf(fstream, ",%p", entry->func_bytes);
	if (entry->func_bytes_size)
		fprintf(fstream, ",%zu", entry->func_bytes_size);
	if (entry->func)
		fprintf(fstream, ",%p", entry->func);

	if (entry->accel_place != -1)
		fprintf(fstream, ",%d", entry->accel_place);

	if (entry->data)
		fprintf( fstream, ",%p:%zu", entry->data,
		         vine_data_size(entry->data) );
	if (entry->in_data)
		fprintf(fstream, ",%p", entry->in_data);
	if (entry->out_data)
		fprintf(fstream, ",%p", entry->out_data);
	if (entry->task)
		fprintf(fstream, ",%p", entry->task);
	if (entry->task_stats)
		fprintf(fstream, ",%p", entry->task_stats);
	fprintf(fstream, "\n");
}

void debug_print_log_buffer(FILE *file)
{
	int i;

	for (i = 0; i <= curr_entry_pos; i++) {
		debug_print_log_entry(file, &log_buffer_start_ptr[i]);
	}
}

void init_log_entry(log_entry *entry)
{
	memset( entry, 0, sizeof(log_entry) );
	entry->accel_type  = -1;
	entry->accel_place = -1;

	struct timeval tv;

	gettimeofday(&tv, NULL);
	entry->timestamp = (tv.tv_sec*1000000+tv.tv_usec) - start_of_time;
	entry->core_id   = sched_getcpu();
	entry->thread_id = pthread_self();
}

log_entry* get_log_buffer_ptr()
{
	if ( is_log_buffer_full() ) {
		update_log_file();
		memset(log_buffer_start_ptr, 0, log_buffer_size);
		curr_entry_pos = -1;
	}
	curr_entry_pos++;

	return &(log_buffer_start_ptr[curr_entry_pos]);
}

void log_vine_proc_get(vine_accel_type_e type, const char *func_name,
                       const char *func_id, int task_duration,
                       vine_proc *return_val)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);

	init_log_entry(entry);

	entry->accel_type     = type;
	entry->func_name      = func_name;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.p = return_val;
}

void log_vine_proc_put(vine_proc *func, const char *func_id, int task_duration,
                       int return_value)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->func           = func;
	entry->func_id        = func_id;
	entry->task_duration  = task_duration;
	entry->return_value.i = return_value;
}

void log_vine_data_alloc(size_t size, vine_data_alloc_place_e place,
                         int task_duration, const char *func_id,
                         void *return_val)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);

	init_log_entry(entry);

	entry->data_size      = size;
	entry->accel_place    = place;
	entry->task_duration  = task_duration;
	entry->func_id        = func_id;
	entry->return_value.p = return_val;
}

void log_vine_data_mark_ready(vine_data *data, const char *func_id,
                              int task_duration)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->data           = data;
	entry->data_size      = vine_data_size(data);
	entry->task_duration  = task_duration;
	entry->func_id        = func_id;
	entry->return_value.p = NULL;
}

void log_vine_data_deref(vine_data *data, const char *func_id,
                         int task_duration, void *return_value)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->data           = data;
	entry->data_size      = vine_data_size(data);
	entry->task_duration  = task_duration;
	entry->func_id        = func_id;
	entry->return_value.p = return_value;
}

void log_vine_data_free(vine_data *data, const char *func_id, int task_duration)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->data          = data;
	entry->task_duration = task_duration;
	entry->func_id       = func_id;
}

void log_vine_task_issue(vine_accel *accel, vine_proc *proc, vine_data *args,
                         size_t in_cnt, size_t out_cnt, vine_data **input,
                         vine_data **output, const char *func_id,
                         int task_duration, vine_task *return_value)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

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

void log_vine_task_wait(vine_task *task, const char *func_id, int task_duration,
                        vine_task_state_e return_value)
{
	log_entry *entry;

	pthread_mutex_lock(&lock);
	entry = get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->task           = task;
	entry->task_duration  = task_duration;
	entry->func_id        = func_id;
	entry->return_value.i = return_value;
}

void _log_timer_start(struct timeval *t1)
{
	gettimeofday(t1, NULL);
}

int _log_timer_stop(struct timeval *t2, struct timeval *t1)
{
	gettimeofday(t2, NULL);

	int elapsedTime;

	elapsedTime  = (t2->tv_sec - t1->tv_sec) * 1000.0;
	elapsedTime += (t2->tv_usec - t1->tv_usec) / 1000.0;

	return elapsedTime;
}
