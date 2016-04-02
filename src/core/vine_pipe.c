#include "vine_pipe.h"
#include <stdio.h>
#include <string.h>

vine_pipe_s* vine_pipe_init(void *mem, size_t size, size_t ring_size)
{
	vine_pipe_s *pipe = mem;

	__sync_val_compare_and_swap(&(pipe->self), 0, pipe); /** Will only
	                                                      * succed once */
	if ( __sync_fetch_and_add(&(pipe->mapped), 1) )
		return pipe;
	utils_list_init( &(pipe->accelerator_list) );
	utils_list_init( &(pipe->process_list) );
	pipe->allocator =
	        utils_alloc_init( &(pipe->allocator)+1, size-sizeof(*pipe) );
	pipe->queue =
	        utils_alloc_allocate( pipe->allocator,
	                          queue_calc_bytes(ring_size) );
	if (!pipe->queue)
		return 0;
	pipe->queue = queue_init( pipe->queue, queue_calc_bytes(ring_size) );
	return pipe;
}

int vine_pipe_register_accel(vine_pipe_s *pipe, vine_accel_s *accel)
{
	utils_list_add( &(pipe->accelerator_list), &(accel->list) );
	return 0;
}

vine_accel_s* vine_proc_find_accel(vine_pipe_s *pipe, const char *name,
                                   vine_accel_type_e type)
{
	utils_list_node_s *itr;
	vine_accel_s     *accel;

	utils_list_for_each(pipe->accelerator_list, itr) {
		accel = (vine_accel_s*)itr;
		if (type && type != accel->type != type)
			continue;
		if (strcmp(name, accel->name) == 0)
			return accel;
	}
	return 0;
}

int vine_pipe_register_proc(vine_pipe_s *pipe, vine_proc_s *proc)
{
	utils_list_add( &(pipe->process_list), &(proc->list) );
	return 0;
}

vine_proc_s* vine_proc_find_proc(vine_pipe_s *pipe, const char *name,
                                 vine_accel_type_e type)
{
	utils_list_node_s *itr;
	vine_proc_s      *proc;

	utils_list_for_each(pipe->process_list, itr) {
		proc = (vine_proc_s*)itr;
		if (type && type != proc->type != type)
			continue;
		if (strcmp(name, proc->name) == 0)
			return proc;
	}
	return 0;
}

/**
 * Destroy vine_pipe.
 */
int vine_pipe_exit(vine_pipe_s *pipe)
{
	return __sync_fetch_and_add(&(pipe->mapped), -1) == 1;
}
