#define _GNU_SOURCE
#ifndef TRACE_ENABLE
#define TRACE_ENABLE
#endif /* TRACE_ENABLE */
#include "vine_pipe.h"
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
 * One log entry contains in formation
 * for one subset of those values.
 **/
typedef struct Entry {
	size_t       timestamp;
	int          core_id;
	volatile int isvalid;
	pthread_t    thread_id;
	const char   *func_id;
	size_t       task_duration;

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
	vine_data               *data;
	size_t                  data_size;
	size_t                  in_cnt;
	size_t                  out_cnt;
	vine_task_msg_s         *task;
	vine_task_stats_s       *task_stats;
}trace_entry;

int             curr_entry_pos;
int             trace_buffer_size;
trace_entry       *trace_buffer_start_ptr;
int             trace_file;
pthread_mutex_t lock;
size_t          start_of_time;
sighandler_t    prev_sighandler;


void signal_callback_handler(int signum)
{
	trace_exit();
	/* Call previous signal handler */
	prev_sighandler(signum);
}

void trace_exit()
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

void trace_init()
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

	char * conf = utils_config_alloc_path(VINE_CONFIG_FILE);

	utils_config_get_int(conf,"trace_buffer_size", &trace_buffer_size,
	                    sizeof(trace_entry)*100);
	assert(trace_buffer_size);

	utils_config_free_path(conf);

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
	char * conf = utils_config_alloc_path(VINE_CONFIG_FILE);

	/* after log file is created , we must not call this function*/
	/* assert(trace_file == 0); */

	hostname[1023] = '\0';
	gethostname(hostname, 1023);

	gettimeofday(&tv, NULL);
	curtime = tv.tv_sec;

	utils_config_get_str(conf,"trace_path",trace_path,1024,".");

	strftime( buffer, 30, "%m-%d-%Y-%T", localtime(&curtime) );
	snprintf(fileName,2078, "%s/trace_%s_%d_%s.csv",trace_path, hostname, getpid(), buffer);

 	utils_config_free_path(conf);
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

static inline void wait_trace_entry_valid(trace_entry* entry)
{
	if(!entry->isvalid)
		do {} while(!entry->isvalid);
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
	char line[1024];
	char * w = line;
	w += sprintf(w, "%zu,%d,%lx,%s,%zu", entry->timestamp, entry->core_id,
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
		!strcmp(entry->func_id, "vine_task_wait")
	) {
		int ret_val = entry->return_value.i;

		w += sprintf(w, ",%d", ret_val);
	} else {
		w += sprintf(w, ",%p", entry->return_value.p);
	}


	if (entry->accel)
		w += sprintf(w, ",%p", entry->accel);
	if (entry->accel_stat)
		w += sprintf(w, ",%p", entry->accel_stat);
	if (entry->accel_type != -1)
		w += sprintf(w, ",%d", entry->accel_type);
	if (entry->func_name)
		w += sprintf(w, ",%p", entry->func_name);
	if (entry->func_bytes)
		w += sprintf(w, ",%p", entry->func_bytes);
	if (entry->func_bytes_size)
		w += sprintf(w, ",%zu", entry->func_bytes_size);
	if (entry->func)
		w += sprintf(w, ",%p", entry->func);

	if ( entry->data_size && (entry->data == 0) )
		w += sprintf(w, ",%zu", entry->data_size);

	if (entry->accels)
		w += sprintf(w, ",%p", entry->accels);

	if (entry->data)
		w += sprintf(w, ",%p", entry->data);
	if (entry->data_size && entry->data)
		w += sprintf(w, ":%zu", entry->data_size);

	if (entry->task)
	{
		w += sprintf(w, ",%p", entry->task->args.vine_data);

		if (entry->in_cnt)
			w += sprintf(w, ",%zu", entry->in_cnt);
		for (i = 0; i < entry->in_cnt; ++i) {
			w += sprintf(w, ",%p", (void*)entry->task->io[i].vine_data);
		}
		if (entry->out_cnt)
			w += sprintf(w, ",%zu", entry->out_cnt);
		for (i = 0; i < entry->out_cnt; ++i) {
			w += sprintf(w, ",%p", entry->task->io[i+entry->in_cnt].vine_data);
		}
	}
	w += sprintf(w, ",%p", entry->task);
	if (entry->task_stats)
		w += sprintf(w, ",%p", entry->task_stats);
	w += sprintf(w, "\n");
	write(fd,line,w-line);
}                  /* print_trace_entry_to_fd */

void print_trace_buffer_to_fd()
{
	int i;

	for (i = 0; i <= curr_entry_pos; i++) {
		wait_trace_entry_valid(&trace_buffer_start_ptr[i]);
		print_trace_entry_to_fd(trace_file, &trace_buffer_start_ptr[i]);
	}
}

void update_trace_file()
{
	print_trace_buffer_to_fd(trace_file);
}

unsigned int is_trace_buffer_full()
{
	int total_trace_entries = trace_buffer_size/sizeof(trace_entry);

	return curr_entry_pos >= (total_trace_entries-1);
}

void init_trace_entry(trace_entry *entry)
{
	memset( entry, 0, sizeof(trace_entry) );
	entry->accel_type  = -1;

	entry->core_id   = sched_getcpu();
	entry->thread_id = pthread_self();
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

static inline void put_trace_buffer_ptr(trace_entry* entry)
{
	entry->isvalid = 1;
}

void trace_vine_accel_list(vine_accel_type_e type, int physical,
						   vine_accel ***accels, const char *func_id,
						   utils_timer_s timing, int return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);
	entry->accel_type     = type;
	entry->accels         = accels;
	entry->func_id        = func_id;
	entry->task_duration  = utils_timer_get_duration_us(timing);
	entry->return_value.i = return_value;
	put_trace_buffer_ptr(entry);
}

void trace_vine_accel_stat(vine_accel *accel, vine_accel_stats_s *stat,
						 const char *func_id, utils_timer_s timing,
						 vine_accel_state_e return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);
	entry->accel          = accel;
	entry->accel_stat     = stat;
	entry->func_id        = func_id;
	entry->task_duration  = utils_timer_get_duration_us(timing);
	entry->return_value.i = return_value;
	put_trace_buffer_ptr(entry);
}

void trace_vine_accel_location(vine_accel *accel, const char *func_id,
							 vine_accel_loc_s return_value, utils_timer_s timing)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);
	entry->accel         = accel;
	entry->func_id       = func_id;
	entry->task_duration = utils_timer_get_duration_us(timing);
	/*	entry->return_value  = &return_value; // Reference of stack value */
	put_trace_buffer_ptr(entry);
}

void trace_vine_accel_type(vine_accel *accel, const char *func_id,
						 utils_timer_s timing, int return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);
	entry->accel          = accel;
	entry->func_id        = func_id;
	entry->task_duration  = utils_timer_get_duration_us(timing);
	entry->return_value.i = return_value;
	put_trace_buffer_ptr(entry);
}

void trace_vine_task_stat(vine_task *task, vine_task_stats_s *stats,
						const char *func_id, utils_timer_s timing,
						vine_task_state_e return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);
	entry->task           = task;
	entry->task_stats     = stats;
	entry->func_id        = func_id;
	entry->task_duration  = utils_timer_get_duration_us(timing);
	entry->return_value.i = return_value;
	put_trace_buffer_ptr(entry);
}

void trace_vine_accel_acquire_phys(vine_accel *accel, const char *func_id,
							utils_timer_s timing)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);
	entry->accel          = accel;
	entry->func_id        = func_id;
	entry->task_duration  = utils_timer_get_duration_us(timing);
	put_trace_buffer_ptr(entry);
}

void trace_vine_accel_acquire_type(vine_accel_type_e type,
										   const char *func_id,
										   vine_accel * return_value,
										   utils_timer_s timing)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);
	entry->accel_type     = type;
	entry->func_id        = func_id;
	entry->task_duration  = utils_timer_get_duration_us(timing);
	entry->return_value.p = return_value;
	put_trace_buffer_ptr(entry);
}


void trace_vine_accel_release(vine_accel *accel, const char *func_id,
							utils_timer_s timing)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);
	entry->accel          = accel;
	entry->func_id        = func_id;
	entry->task_duration  = utils_timer_get_duration_us(timing);
	put_trace_buffer_ptr(entry);
}

void trace_vine_proc_register(vine_accel_type_e type, const char *proc_name,
							const void *func_bytes, size_t func_bytes_size,
							const char *func_id, utils_timer_s timing,
							void *return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp       = utils_timer_get_time_us(timing,start);
	entry->func_id         = func_id;
	entry->task_duration   = utils_timer_get_duration_us(timing);
	entry->accel_type      = type;
	entry->func_name       = proc_name;
	entry->func_bytes      = func_bytes;
	entry->func_bytes_size = func_bytes_size;
	entry->return_value.p  = return_value;
	put_trace_buffer_ptr(entry);
}

void trace_vine_proc_get(vine_accel_type_e type, const char *func_name,
                       const char *func_id, utils_timer_s timing,
                       vine_proc *return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);
	entry->accel_type     = type;
	entry->func_name      = func_name;
	entry->func_id        = func_id;
	entry->task_duration  = utils_timer_get_duration_us(timing);
	entry->return_value.p = return_value;
	put_trace_buffer_ptr(entry);
}

void trace_vine_proc_put(vine_proc *func, const char *func_id, utils_timer_s timing,
                       int return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);
	entry->func           = func;
	entry->func_id        = func_id;
	entry->task_duration  = utils_timer_get_duration_us(timing);
	entry->return_value.i = return_value;
	put_trace_buffer_ptr(entry);
}

void trace_vine_task_issue(vine_accel *accel, vine_proc *proc, vine_data *args,
						   size_t in_cnt, size_t out_cnt, vine_buffer_s *input,
						   vine_buffer_s *output, const char *func_id,
						   utils_timer_s timing, vine_task *return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);

	entry->accel          = accel;
	entry->func           = proc;
	entry->in_cnt         = in_cnt;
	entry->out_cnt        = out_cnt;
	entry->task_duration  = utils_timer_get_duration_us(timing);
	entry->func_id        = func_id;
	entry->return_value.p = return_value;
	put_trace_buffer_ptr(entry);
}

void trace_vine_task_wait(vine_task *task, const char *func_id, utils_timer_s timing,
                        vine_task_state_e return_value)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);
	entry->task           = task;
	entry->task_duration  = utils_timer_get_duration_us(timing);
	entry->func_id        = func_id;
	entry->return_value.i = return_value;
	put_trace_buffer_ptr(entry);
}

void trace_vine_task_free(vine_task * task,const char *func_id, utils_timer_s timing)
{
	trace_entry *entry;

	entry = get_trace_buffer_ptr();
	entry->timestamp      = utils_timer_get_time_us(timing,start);
	entry->task           = task;
	entry->task_duration  = utils_timer_get_duration_us(timing);
	entry->func_id        = func_id;
	put_trace_buffer_ptr(entry);
}
