#include <vine_pipe.h>
#include <stdio.h>
#include <string.h>

vine_pipe_s* vine_pipe_init(void *mem, size_t size)
{
	vine_pipe_s *pipe = mem;
	uint64_t    value;

	value = __sync_bool_compare_and_swap(&(pipe->self), 0, pipe);
	if (value)
		pipe->shm_size = size;

	value = __sync_fetch_and_add(&(pipe->mapped), 1);
	if (value)
		return pipe;
	vine_object_repo_init( &(pipe->objs) );

	arch_alloc_init( &(pipe->allocator),&(pipe->allocator)+1, size-sizeof(*pipe) );
	pipe->queue = arch_alloc_allocate( &(pipe->allocator), sizeof(*(pipe->queue)));
	if (!pipe->queue)
		return 0;
	pipe->queue = utils_queue_init( pipe->queue );
	async_meta_init_once( &(pipe->async) );
	return pipe;
}

int vine_pipe_delete_accel(vine_pipe_s *pipe, vine_accel_s *accel)
{
	if ( !vine_pipe_find_accel(pipe, vine_accel_get_name(accel),
	                           accel->type) )
		return 1;
	vine_accel_erase(&(pipe->objs),accel);
	return 0;
}

vine_accel_s* vine_pipe_find_accel(vine_pipe_s *pipe, const char *name,
                                   vine_accel_type_e type)
{
	utils_list_node_s *itr;
	utils_list_s      *list;
	vine_accel_s      *accel = 0;

	list = vine_object_list_lock(&(pipe->objs), VINE_TYPE_PHYS_ACCEL);
	utils_list_for_each(*list, itr) {
		accel = (vine_accel_s*)itr->owner;
		if ( type && (type != accel->type) )
			continue;
		if ( !name ||
		     (strcmp( name, vine_accel_get_name(accel) ) == 0) ) {
			vine_object_list_unlock(&(pipe->objs),
			                        VINE_TYPE_PHYS_ACCEL);
			return accel;
		}
	}
	accel = 0;
	vine_object_list_unlock(&(pipe->objs), VINE_TYPE_PHYS_ACCEL);
	return accel;
}

vine_proc_s* vine_pipe_find_proc(vine_pipe_s *pipe, const char *name,
                                 vine_accel_type_e type)
{
	utils_list_node_s *itr;
	utils_list_s      *list;
	vine_proc_s       *proc;

	list = vine_object_list_lock(&(pipe->objs), VINE_TYPE_PROC);
	utils_list_for_each(*list, itr) {
		proc = (vine_proc_s*)itr->owner;
		if (type && type != proc->type)
			continue;
		if (strcmp(name, proc->obj.name) == 0) {
			vine_object_list_unlock(&(pipe->objs), VINE_TYPE_PROC);
			return proc;
		}
	}
	proc = 0;
	vine_object_list_unlock(&(pipe->objs), VINE_TYPE_PROC);
	return proc;
}

int vine_pipe_delete_proc(vine_pipe_s *pipe, vine_proc_s *proc)
{
	if ( !vine_pipe_find_proc(pipe, proc->obj.name,
		proc->type) )
		return 1;
	utils_breakdown_write(proc->obj.name,proc->type,proc->obj.name,&(proc->breakdown));
	vine_object_remove( &(pipe->objs), &(proc->obj) );
	return 0;
}

/**
 * Destroy vine_pipe.
 */
int vine_pipe_exit(vine_pipe_s *pipe)
{
	int ret = __sync_fetch_and_add(&(pipe->mapped), -1) == 1;
	if(ret)	// Last user
	{
		async_meta_exit( &(pipe->async) );
		arch_alloc_exit( &(pipe->allocator) );
		memset(pipe,0,sizeof(*pipe));
	}
	return ret;
}
