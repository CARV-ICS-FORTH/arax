#include "vine_talk.h"

int vine_accel_list(vine_accel_type_e type,vine_accel *** accels)
{
	return 0;
}

vine_accel_loc_s vine_accel_location(vine_accel * accel)
{
	vine_accel_loc_s loc;
	return loc;
}

vine_accel_type_e vine_accel_type(vine_accel * accel)
{
	return 0;
}

vine_accel_state_e vine_accel_stat(vine_accel * accel,vine_accel_stats_s * stat)
{
	return 0;
}

int vine_accel_acquire(vine_accel * accel)
{
	return 0;
}

void vine_accel_release(vine_accel * accel)
{
}

vine_proc * vine_proc_register(vine_accel_type_e type,const char * func_name,const void * func_bytes,size_t func_bytes_size)
{
	return 0;
}

vine_proc * vine_proc_get(vine_accel_type_e type,const char * func_name)
{
	return 0;
}

int vine_proc_put(vine_proc * func)
{
	return 0;
}

vine_data * vine_data_alloc(size_t size,vine_data_alloc_place_e place)
{
	return 0;
}

size_t vine_data_size(vine_data * data)
{
	return 0;
}

void * vine_data_deref(vine_data * data)
{
	return 0;
}

void vine_data_mark_ready(vine_data * data)
{
}

void vine_data_free(vine_data * data)
{
}

vine_task * vine_task_issue(vine_accel * accel,vine_proc * proc,vine_data * args,size_t in_count,vine_data ** input,size_t out_count,vine_data ** output)
{
	return 0;
}

vine_task_state_e vine_task_stat(vine_task * task,vine_task_stats_s * stats)
{
	return 0;
}

vine_task_state_e vine_task_wait(vine_task * task)
{
	return 0;
}

