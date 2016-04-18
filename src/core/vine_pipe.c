#include <vine_pipe.h>
#include <stdio.h>
#include <string.h>

vine_pipe_s* vine_pipe_init(void *mem, size_t size, size_t queue_size)
{
	vine_pipe_s *pipe = mem;

	if ( __sync_bool_compare_and_swap(&(pipe->self), 0, pipe) )
		pipe->shm_size = size;

	if ( __sync_fetch_and_add(&(pipe->mapped), 1) )
		return pipe;
	utils_spinlock_init( &(pipe->accelerator_lock) );
	utils_list_init( &(pipe->accelerator_list) );
	utils_spinlock_init( &(pipe->process_lock) );
	utils_list_init( &(pipe->process_list) );
	pipe->allocator =
	        arch_alloc_init( &(pipe->allocator)+1, size-sizeof(*pipe) );
	pipe->queue = arch_alloc_allocate( pipe->allocator, utils_queue_calc_bytes(
	                                           queue_size) );
	if (!pipe->queue)
		return 0;
	pipe->queue =
	        utils_queue_init( pipe->queue, utils_queue_calc_bytes(
	                                  queue_size) );
	return pipe;
}

int vine_pipe_register_accel(vine_pipe_s *pipe, vine_accel_s *accel)
{
	utils_spinlock_lock( &(pipe->accelerator_lock) );
	utils_list_add( &(pipe->accelerator_list), &(accel->list) );
	utils_spinlock_unlock( &(pipe->accelerator_lock) );
	return 0;
}

vine_accel_s* vine_proc_find_accel(vine_pipe_s *pipe, const char *name,
                                   vine_accel_type_e type)
{
	utils_list_node_s *itr;
	vine_accel_s      *accel;

	utils_spinlock_lock( &(pipe->accelerator_lock) );
	utils_list_for_each(pipe->accelerator_list, itr) {
		accel = (vine_accel_s*)itr;
		if ( type && (type != accel->type) )
			continue;
		if ( !name || (strcmp(name, accel->name) == 0) ) {
			utils_spinlock_unlock( &(pipe->accelerator_lock) );
			return accel;
		}
	}
	utils_spinlock_unlock( &(pipe->accelerator_lock) );
	return 0;
}

int vine_pipe_register_proc(vine_pipe_s *pipe, vine_proc_s *proc)
{
	utils_spinlock_lock( &(pipe->process_lock) );
	utils_list_add( &(pipe->process_list), &(proc->list) );
	utils_spinlock_unlock( &(pipe->process_lock) );
	return 0;
}

vine_proc_s* vine_proc_find_proc(vine_pipe_s *pipe, const char *name,
                                 vine_accel_type_e type)
{
	utils_list_node_s *itr;
	vine_proc_s       *proc;

	utils_spinlock_lock( &(pipe->process_lock) );
	utils_list_for_each(pipe->process_list, itr) {
		proc = (vine_proc_s*)itr;
		if (type && type != proc->type)
			continue;
		if (strcmp(name, proc->name) == 0) {
			utils_spinlock_unlock( &(pipe->process_lock) );
			return proc;
		}
	}
	utils_spinlock_unlock( &(pipe->process_lock) );
	return 0;
}

/**
 * Destroy vine_pipe.
 */
int vine_pipe_exit(vine_pipe_s *pipe)
{
	return __sync_fetch_and_add(&(pipe->mapped), -1) == 1;
}
