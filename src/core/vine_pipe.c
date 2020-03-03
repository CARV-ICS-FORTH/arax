#include <vine_pipe.h>
#include <stdio.h>
#include <string.h>

vine_pipe_s* vine_pipe_init(void *mem, size_t size,int enforce_version)
{
	vine_pipe_s *pipe = mem;
	uint64_t    value;

	value = vine_pipe_add_process(pipe);

	if (value)	// Not first so assume initialized
	{
		while(!pipe->sha[0]);
		if(strcmp(pipe->sha,VINE_TALK_GIT_REV))
		{
			fprintf(stderr,"Vinetalk revision mismatch(%s vs %s)!",VINE_TALK_GIT_REV,pipe->sha);
			if(enforce_version)
				return 0;
		}
		return pipe;
	}

	printf("Initializing pipe.\n");

	pipe->shm_size = size;
	/**
	 * Write sha sum except first byte
	 */
	sprintf(pipe->sha+1,"%s",VINE_TALK_GIT_REV+1);
	pipe->sha[0] = VINE_TALK_GIT_REV[0];

	vine_object_repo_init( &(pipe->objs), &(pipe->allocator) );

	if(arch_alloc_init( &(pipe->allocator), size-sizeof(*pipe) ))
		return 0;

	pipe->queue = arch_alloc_allocate( &(pipe->allocator), sizeof(*(pipe->queue)));

	if (!pipe->queue)
		return 0;

	pipe->queue = utils_queue_init( pipe->queue );

	async_meta_init_once( &(pipe->async), &(pipe->allocator) );
	async_condition_init(&(pipe->async), &(pipe->tasks_cond));

	vine_throttle_init(&(pipe->async),&(pipe->throttle),size,size);

	for(value = 0 ; value < VINE_ACCEL_TYPES ; value++)
		pipe->tasks[value] = 0;

	utils_kv_init(&(pipe->ass_kv));

	return pipe;
}

const char * vine_pipe_get_revision(vine_pipe_s * pipe)
{
	return pipe->sha;
}

uint64_t vine_pipe_add_process(vine_pipe_s * pipe)
{
	return __sync_fetch_and_add(&(pipe->processes), 1);
}

uint64_t vine_pipe_del_process(vine_pipe_s * pipe)
{
	return __sync_fetch_and_add(&(pipe->processes), -1);
}

void * vine_pipe_mmap_address(vine_pipe_s * pipe)
{
	int value = __sync_bool_compare_and_swap(&(pipe->self), 0, pipe);

	if(value)
		return pipe;
	else
		return pipe->self;
}

int vine_pipe_delete_accel(vine_pipe_s *pipe, vine_accel_s *accel)
{
	if ( !vine_pipe_find_accel(pipe, vine_accel_get_name(accel),
	                           accel->type) )
		return 1;
	vine_object_ref_dec(&(accel->obj));
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

void vine_pipe_add_task(vine_pipe_s *pipe,vine_accel_type_e type,void * assignee)
{
	async_condition_lock(&(pipe->tasks_cond));
	pipe->tasks[type]++;
	if(assignee)
	{
		size_t * tasks = (size_t*)utils_kv_get(&(pipe->ass_kv),assignee);
		(*tasks)++;
	}
	async_condition_notify(&(pipe->tasks_cond));
	async_condition_unlock(&(pipe->tasks_cond));
}

void vine_pipe_wait_for_task(vine_pipe_s *pipe,vine_accel_type_e type)
{
	async_condition_lock(&(pipe->tasks_cond));
	while(!pipe->tasks[type])	// Spurious wakeup
		async_condition_wait(&(pipe->tasks_cond));
	pipe->tasks[type]--;
	async_condition_unlock(&(pipe->tasks_cond));
}

vine_accel_type_e vine_pipe_wait_for_task_type_or_any_assignee(vine_pipe_s *pipe,vine_accel_type_e type,void * assignee)
{
	vine_assert(type != ANY);
	async_condition_lock(&(pipe->tasks_cond));
	size_t zero = 0;
	size_t * tasks = (size_t*)utils_kv_get(&(pipe->ass_kv),assignee);

	if(!tasks)	// Unassigned
	{
		fprintf(stderr,"WARNING:%s() with unregistered assignee\n",__func__);
		tasks = &zero;
	}

	while(
		!pipe->tasks[type] &&	// Dont have the type i want
		!pipe->tasks[ANY] && 	// Dont have any type
		(!*tasks)				// No assigned tasks (if this crashes, assigne was invalid)
	)	// Spurious wakeup
		async_condition_wait(&(pipe->tasks_cond));

	if(*tasks)
	{
		(*tasks)--;
	}
	if(pipe->tasks[type])
	{
		pipe->tasks[type]--;
	}
	else if(pipe->tasks[ANY])
	{
		pipe->tasks[ANY]--;
		type = ANY;
	}
	async_condition_unlock(&(pipe->tasks_cond));
	return type;
}

void vine_pipe_register_assignee(vine_pipe_s *pipe,void * assignee)
{
	utils_kv_set(&(pipe->ass_kv),assignee,0);
}

/**
 * Destroy vine_pipe.
 */
int vine_pipe_exit(vine_pipe_s *pipe)
{
	int ret = vine_pipe_del_process(pipe) == 1;
	if(ret)	// Last user
	{
		arch_alloc_free( &(pipe->allocator), pipe->queue );
		async_meta_exit( &(pipe->async) );
		arch_alloc_exit( &(pipe->allocator) );
		memset(pipe,0,sizeof(*pipe));
	}
	return ret;
}


void vine_pipe_size_inc(vine_pipe_s *pipe,size_t sz){
    //error check
    vine_assert(pipe);
    //notify exdw
    vine_throttle_size_inc(&pipe->throttle,sz);
}


void vine_pipe_size_dec(vine_pipe_s *pipe,size_t sz){
    //error check
    vine_assert(pipe);
    //wait exdw
    vine_throttle_size_dec(&pipe->throttle,sz);
}


size_t vine_pipe_get_available_size(vine_pipe_s *pipe){
    //error check
    vine_assert(pipe);
    return vine_throttle_get_available_size(&pipe->throttle);
}


size_t vine_pipe_get_total_size(vine_pipe_s *pipe){
    //error check
    vine_assert(pipe);
    return vine_throttle_get_total_size(&pipe->throttle);
}
