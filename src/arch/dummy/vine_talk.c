#include <vine_talk.h>
#include <vine_pipe.h>
#include "utils/trace.h"
#include <stdlib.h>
#include <errno.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <fcntl.h>
#include <stdio.h>
#include <math.h>

vine_pipe_s* vine_pipe_get()
{
	return NULL;
}

int vine_accel_list(vine_accel_type_e type, vine_accel ***accels)
{
	int        task_duration = 0;
	int        num_of_accels = 1;
	vine_accel **a;

	a       = (vine_accel**)malloc(sizeof(vine_accel*)*num_of_accels);
	a[0]    = malloc( sizeof(vine_accel) );
	*accels = a;

	log_vine_accel_list(type, accels, __FUNCTION__, task_duration,
	                    &num_of_accels);
	return num_of_accels;
}

vine_accel_loc_s vine_accel_location(vine_accel *accel)
{
	vine_accel_loc_s return_val;
	int              task_duration = 0;

	log_vine_accel_location(accel, __FUNCTION__, return_val, task_duration);
	return return_val;
}

vine_accel_type_e vine_accel_type(vine_accel *accel)
{
	int               task_duration = 0;
	vine_accel_type_e return_value  = 0;

	log_vine_accel_type(accel, __FUNCTION__, return_value, &task_duration);
	return return_value;
}

vine_accel_state_e vine_accel_stat(vine_accel *accel, vine_accel_stats_s *stat)
{
	int                task_duration = 0;
	vine_accel_state_e return_value  = 0;

	log_vine_accel_stat(accel, stat, __FUNCTION__, task_duration,
	                    &return_value);
	return return_value;
}

int vine_accel_acquire(vine_accel *accel)
{
	int return_value  = 0;
	int task_duration = 0;

	log_vine_accel_acquire(accel, __FUNCTION__, return_value,
	                       task_duration);
	return 0;
}

void vine_accel_release(vine_accel *accel)
{
	int task_duration = 0;

	log_vine_accel_release(accel, __FUNCTION__, task_duration);
}

vine_proc* vine_proc_register(vine_accel_type_e type, const char *func_name,
                              const void *func_bytes, size_t func_bytes_size)
{
	int       task_duration = 0;
	vine_proc *return_value = NULL;

	log_vine_proc_register(type, func_name, func_bytes, func_bytes_size,
	                       __FUNCTION__, task_duration, return_value);

	return return_value;
}

vine_proc* vine_proc_get(vine_accel_type_e type, const char *proc_name)
{
	vine_proc *ret_proc     = NULL;
	int       task_duration = 0;

	log_vine_proc_get(type, proc_name, __FUNCTION__, task_duration,
	                  ret_proc);
	return ret_proc;
}

int vine_proc_put(vine_proc *func)
{
	int return_value  = 0;
	int task_duration = 0;

	log_vine_proc_put(func, __FUNCTION__, task_duration, return_value);
	return return_value;
}

vine_data* vine_data_alloc(size_t size, vine_data_alloc_place_e place)
{
	int       task_duration = 0;
	vine_data *return_val   = NULL;

	log_vine_data_alloc(size, place, task_duration, __FUNCTION__,
	                    return_val);

	return (vine_data*)malloc(size);
}

size_t vine_data_size(vine_data *data)
{
	return 0;
}

void* vine_data_deref(vine_data *data)
{
	int  task_duration = 0;
	void *return_val   = NULL;

	log_vine_data_deref(data, __FUNCTION__, task_duration, return_val);

	return return_val;
}

void vine_data_mark_ready(vine_data *data)
{
	/* TODO : implement? */
}

void vine_data_free(vine_data *data)
{
	int task_duration = 0;

	log_vine_data_free(data, __FUNCTION__, task_duration);
	free(data);
}

vine_task* vine_task_issue(vine_accel *accel, vine_proc *proc, vine_data *args,
                           size_t in_count, vine_data **input, size_t out_count,
                           vine_data **output)
{
	int       task_duration = 0;
	vine_task *return_value = NULL;

	log_vine_task_issue(accel, proc, args, in_count, out_count, input,
	                    output, __FUNCTION__, task_duration, return_value);
	return return_value;
}

vine_task_state_e vine_task_stat(vine_task *task, vine_task_stats_s *stats)
{
	int               task_duration = 0;
	vine_task_state_e return_value  = -1;

	log_vine_task_stat(task, stats, __FUNCTION__, task_duration,
	                   return_value);
	return return_value;
}

vine_task_state_e vine_task_wait(vine_task *task)
{
	int               task_duration = 0;
	vine_task_state_e return_val    = 0;

	log_vine_task_wait(task, __FUNCTION__, task_duration, return_val);
	return return_val;
}
