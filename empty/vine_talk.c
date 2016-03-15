#include "vine_talk.h"

int vine_accel_list(vine_accel_type_e type,vine_accel *** accels)
{

}

vine_accel_loc_s vine_accel_location(vine_accel * accel)
{

}

vine_accel_type_e vine_accel_type(vine_accel * accel)
{

}

vine_accel_state_e vine_accel_stat(vine_accel * accel,vine_accel_stats_s * stat)
{

}

int vine_accel_acquire(vine_accel * accel)
{

}

void vine_accel_release(vine_accel * accel)
{

}

vine_proc * vine_proc_register(vine_accel_type_e type,const char * func_name,const void * func_bytes,size_t func_bytes_size)
{

}

vine_proc * vine_proc_get(vine_accel_type_e type,const char * func_name)
{

}

int vine_proc_put(vine_proc * func)
{

}

vine_data * vine_data_alloc(size_t size,vine_data_alloc_place_e place)
{

}

size_t vine_data_size(vine_data * data)
{

}

void * vine_data_deref(vine_data * data)
{

}

void vine_data_free(vine_data * data)
{

}

vine_task * vine_task_issue(vine_accel * accel,vine_proc * proc,vine_data * args,vine_data ** input,vine_data ** output)
{

}

vine_task_state_e vine_task_stat(vine_task * task,vine_task_stats_s * stats)
{

}

vine_task_state_e vine_task_wait(vine_task * task)
{

}

