#include <vine_talk.h>
#include <vine_pipe.h>
#include "arch/alloc.h"
#include "utils/queue.h"
#include "utils/config.h"
#include "utils/trace.h"
#include "utils/system.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

static void        *shm = 0;
static vine_pipe_s *_vpipe = 0;
static char        shm_file[1024];
static uint64_t    instance_uid = 0;
static uint64_t    task_uid = 0;
vine_pipe_s* vine_pipe_get()
{
	if (!_vpipe)
	{
		/* This will be (hopefully) be removed at a letter time,
		 * when users learn to call vine_talk_init(). */
		fprintf(stderr,
		"WARNING:Using vine_talk without prior call to vine_talk_init()!\n");
		vine_talk_init();
	}
	return _vpipe;
}

#define RING_SIZE 128
#define MY_ID     1

void vine_talk_init()
{
	int    err         = 0;
	size_t shm_size    = 0;
	size_t shm_off     = 0;
	int    shm_trunc   = 0;
	int    shm_ivshmem = 0;
	int    remap       = 0;
	int    fd          = 0;

	if (_vpipe) /* Already initialized */
		return;

	#ifdef TRACE_ENABLE
	tracer_init();
	#endif

	/* Required Confguration Keys */
	if ( !utils_config_get_str("shm_file", shm_file, 1024,0) ) {
		err = __LINE__;
		goto FAIL;
	}

	/* Default /4 of system memory*/
	shm_size = system_total_memory()/4;

	utils_config_get_size("shm_size", &shm_size, shm_size);

	if ( !shm_size || shm_size > system_total_memory() ) {
		err = __LINE__;
		goto FAIL;
	}

	/* Optional Confguration Keys */
	utils_config_get_size("shm_off", &shm_off, 0);
	utils_config_get_bool("shm_trunc", &shm_trunc, 1);
	utils_config_get_bool("shm_ivshmem", &shm_ivshmem, 0);

	if (shm_file[0] == '/')
		fd = open(shm_file, O_CREAT|O_RDWR, 0644);
	else
		fd = shm_open(shm_file, O_CREAT|O_RDWR, S_IRWXU);

	if (fd < 0) {
		err = __LINE__;
		goto FAIL;
	}

	if (shm_ivshmem) {
		shm_off  += 4096; /* Skip register section */
		shm_trunc = 0; /* Don't truncate ivshm  device */
	}

	if (shm_trunc) /* If shm_trunc */
		if ( ftruncate(fd, shm_size) ) {
			err = __LINE__;
			goto FAIL;
		}



	do {
		shm = mmap(shm, shm_size, PROT_READ|PROT_WRITE|PROT_EXEC,
		           MAP_SHARED|(shm ? MAP_FIXED : 0), fd, shm_off);

		if (!shm || shm == MAP_FAILED) {
			err = __LINE__;
			goto FAIL;
		}

		if(_vpipe) // Already initialized, so just remaped
			_vpipe = (vine_pipe_s*)shm;
		else
			_vpipe = vine_pipe_init(shm, shm_size, RING_SIZE);
		shm    = _vpipe->self; /* This is where i want to go */

		if (_vpipe != _vpipe->self) {
			printf("Remapping from %p to %p.\n", _vpipe, shm);
			remap = 1;
		}

		if (shm_size != _vpipe->shm_size) {
			printf("Resizing from %lu to %lu.\n", shm_size,
			       _vpipe->shm_size);
			shm_size = _vpipe->shm_size;
			remap    = 1;
		}
	} while (remap--); /* Not where i want */
	printf("ShmFile:%s\n", shm_file);
	printf("ShmLocation:%p\n", shm);
	printf("ShmSize:%zu\n", shm_size);
	instance_uid = __sync_fetch_and_add(&(_vpipe->last_uid),1);
	printf("InstanceUID:%zu\n", instance_uid);
	return;

FAIL:   printf("prepare_vine_talk Failed on line %d (file:%s,shm:%p)\n", err,
	       shm_file, shm);
	exit(0);
}                  /* vine_task_init */

uint64_t vine_talk_instance_uid()
{
	return instance_uid;
}

void vine_talk_exit()
{
	int last;

	if(_vpipe)
	{

		#ifdef TRACE_ENABLE
		tracer_exit();
		#endif


		last = vine_pipe_exit(_vpipe);

		_vpipe = 0;
		printf("%s", __func__);
		printf("vine_pipe_exit() = %d\n", last);
		if (last)
			if ( shm_unlink(shm_file) )
				printf("Could not delete \"%s\"\n", shm_file);
	}
	else
		fprintf(stderr,
		"WARNING:vine_talk_exit() called with no mathcing\
		call to vine_talk_init()!\n");
}

int vine_accel_list(vine_accel_type_e type, vine_accel ***accels)
{
	vine_pipe_s       *vpipe;
	utils_list_node_s *itr;
	utils_list_s      *acc_list;
	vine_accel_s      *accel      = 0;
	vine_accel_s      **acl       = 0;
	int               accel_count = 0;

	TRACER_TIMER(task);
	trace_timer_start(task);

	vpipe = vine_pipe_get();

	acc_list =
	        vine_object_list_lock(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL);

	if (accels) { /* Want the accels */
		*accels = malloc( acc_list->length*sizeof(vine_accel*) );
		acl     = (vine_accel_s**)*accels;
	}

	utils_list_for_each(*acc_list, itr) {
		accel = (vine_accel_s*)itr->owner;
		if (!type || accel->type == type) {
			accel_count++;
			if (acl) {
				*acl = accel;
				acl++;
			}
		}
	}
	vine_object_list_unlock(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL);

	trace_timer_stop(task);

	trace_vine_accel_list(type, accels, __FUNCTION__, task_duration,
	                    accel_count);

	return accel_count;
}

vine_accel_loc_s vine_accel_location(vine_accel *accel)
{
	vine_accel_loc_s ret;

	TRACER_TIMER(task);

	trace_timer_start(task);

	/*
	 * TODO: Implement
	 */
	trace_timer_stop(task);

	trace_vine_accel_location(accel, __FUNCTION__, ret, task_duration);
	return ret;
}

vine_accel_type_e vine_accel_type(vine_accel *accel)
{
	vine_accel_s *_accel;

	TRACER_TIMER(task);

	trace_timer_start(task);
	_accel = accel;

	trace_timer_stop(task);

	trace_vine_accel_type(accel, __FUNCTION__, task_duration, _accel->type);
	return _accel->type;
}

vine_accel_state_e vine_accel_stat(vine_accel *accel, vine_accel_stats_s *stat)
{
	vine_accel_s *_accel;
	vine_accel_state_e ret;
	TRACER_TIMER(task);

	trace_timer_start(task);
	_accel = accel;

	switch(_accel->obj.type)
	{
		case VINE_TYPE_PHYS_ACCEL:
			ret = vine_accel_get_stat(_accel,stat);
			break;
		case VINE_TYPE_VIRT_ACCEL:
			ret = vine_vaccel_get_stat((vine_vaccel_s*)_accel,stat);
			break;
		default:
			ret = accel_failed;	/* Not very 'corrent' */
	}
	trace_timer_stop(task);

	trace_vine_accel_stat(accel, stat, __FUNCTION__, task_duration,ret);

	return ret;
}

int vine_accel_acquire_phys(vine_accel **accel)
{
	vine_pipe_s  *vpipe;
	vine_accel_s *_accel;
	int          return_value = 0;

	TRACER_TIMER(task);

	vpipe = vine_pipe_get();

	trace_timer_start(task);
	_accel = *accel;

	if (_accel->obj.type == VINE_TYPE_PHYS_ACCEL) {
		void *accel_mem =
		        arch_alloc_allocate(vpipe->allocator, 4096);

		*accel = vine_vaccel_init(&(vpipe->objs), accel_mem, 4096,
		                          "FILL",_accel->type, _accel);
		return_value = 1;
	}

	trace_timer_stop(task);

	trace_vine_accel_acquire_phys(*accel, __FUNCTION__, return_value,
	                       task_duration);

	return return_value;
}

vine_accel * vine_accel_acquire_type(vine_accel_type_e type)
{
	vine_pipe_s  *vpipe;
	vine_accel_s *_accel = 0;
	void *accel_mem = 0;
	TRACER_TIMER(task);

	vpipe = vine_pipe_get();

	accel_mem =	arch_alloc_allocate(vpipe->allocator, 4096);
	_accel = (vine_accel_s*)vine_vaccel_init(&(vpipe->objs), accel_mem, 4096,
							  "FILL",type, 0);

	trace_timer_stop(task);

	trace_vine_accel_acquire_type(type, __FUNCTION__, _accel,task_duration);
	return (vine_accel *)_accel;
}

int vine_accel_release(vine_accel **accel)
{
	vine_pipe_s   *vpipe;
	vine_vaccel_s *_accel;
	int           return_value = 0;

	TRACER_TIMER(task);

	vpipe = vine_pipe_get();

	trace_timer_start(task);
	_accel = *accel;

	if (_accel->obj.type == VINE_TYPE_VIRT_ACCEL) {
		vine_vaccel_erase(&(vpipe->objs), _accel);
		arch_alloc_free(vpipe->allocator, _accel);
		*accel       = 0;
		return_value = 1;
	}


	trace_timer_stop(task);

	trace_vine_accel_release(*accel, __FUNCTION__, return_value,
	                       task_duration);
	return return_value;
}

vine_proc* vine_proc_register(vine_accel_type_e type, const char *func_name,
                              const void *func_bytes, size_t func_bytes_size)
{
	vine_pipe_s *vpipe;
	vine_proc_s *proc = 0;

	TRACER_TIMER(task);

	trace_timer_start(task);

	if(type) // Can not create an ANY procedure.
	{
		vpipe = vine_pipe_get();
		proc  = vine_pipe_find_proc(vpipe, func_name, type);

		if (!proc) { /* Proc has not been declared */
			proc = arch_alloc_allocate( vpipe->allocator, vine_proc_calc_size(
												func_name,
												func_bytes_size) );
			proc = vine_proc_init(&(vpipe->objs), proc, func_name, type,
								func_bytes, func_bytes_size);
		} else {
			/* Proc has been re-declared */
			if ( !vine_proc_match_code(proc, func_bytes, func_bytes_size) )
				return 0; /* Different than before */
		}
		vine_proc_mod_users(proc, +1); /* Increase user count */
	}

	trace_timer_stop(task);

	trace_vine_proc_register(type, func_name, func_bytes, func_bytes_size,
	                       __FUNCTION__, task_duration, proc);

	return proc;
}

vine_proc* vine_proc_get(vine_accel_type_e type, const char *func_name)
{
	TRACER_TIMER(task);

	trace_timer_start(task);

	vine_pipe_s *vpipe = vine_pipe_get();
	vine_proc_s *proc  = vine_pipe_find_proc(vpipe, func_name, type);

	if (proc)
		vine_proc_mod_users(proc, +1); /* Increase user count */

	trace_timer_stop(task);

	trace_vine_proc_get(type, func_name, __FUNCTION__, task_duration,
	                  (void*)proc);

	return proc;
}

int vine_proc_put(vine_proc *func)
{
	TRACER_TIMER(task);

	trace_timer_start(task);

	vine_proc_s *proc = func;
	/* Decrease user count */
	int return_value = vine_proc_mod_users(proc, -1);

	trace_timer_stop(task);

	trace_vine_proc_put(func, __FUNCTION__, task_duration, return_value);

	return return_value;
}

vine_data* vine_data_alloc(size_t size, vine_data_alloc_place_e place)
{
	void *mem;
	vine_data *return_val = 0;
	TRACER_TIMER(task);

	trace_timer_start(task);

	vine_pipe_s *vpipe = vine_pipe_get();

	/* Not valid place */
	if(!place || place>>2)
		return 0;

	mem = arch_alloc_allocate( vpipe->allocator, size+sizeof(vine_data_s) );

	if(mem)
	{
		return_val =
			vine_data_init(&(vpipe->objs),&(vpipe->async), mem, size, place);
	}

	trace_timer_stop(task);
	trace_vine_data_alloc(size, place, task_duration, __FUNCTION__,
	                    return_val);

	return return_val;
}

size_t vine_data_size(vine_data *data)
{
	vine_data_s *vdata;

	vdata = data;
	return vdata->size;
}

void* vine_data_deref(vine_data *data)
{
	vine_data_s *vdata;

	TRACER_TIMER(task);

	trace_timer_start(task);

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);

	trace_timer_stop(task);

	if (!(vdata->place&HostOnly)) {
		trace_vine_data_deref(data, __FUNCTION__, task_duration, 0);
		return 0;
	}

	trace_vine_data_deref( data, __FUNCTION__, task_duration,
	                     (void*)(vdata+1) );
	return (void*)(vdata+1);
}

void vine_data_mark_ready(vine_data *data)
{
	vine_data_s *vdata;

	TRACER_TIMER(task);

	trace_timer_start(task);

	vine_pipe_s *vpipe = vine_pipe_get();

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);
	async_completion_complete(&(vpipe->async),&(vdata->ready));

	trace_timer_stop(task);

	trace_vine_data_mark_ready(data, __FUNCTION__, task_duration);
}

int vine_data_check_ready(vine_data *data)
{
	vine_data_s *vdata;
	int return_val;
	TRACER_TIMER(task);

	trace_timer_start(task);

	vine_pipe_s *vpipe = vine_pipe_get();

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);
	return_val = async_completion_check(&(vpipe->async),&(vdata->ready));

	trace_timer_stop(task);

	trace_vine_data_check_ready(data, __FUNCTION__, task_duration,return_val);

	return return_val;
}

void vine_data_free(vine_data *data)
{
	vine_data_s *vdata;

	TRACER_TIMER(task);

	trace_timer_start(task);

	vine_pipe_s *vpipe = vine_pipe_get();

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);
	vine_data_erase(&(vpipe->objs), vdata);
	arch_alloc_free(vpipe->allocator, vdata);

	trace_timer_stop(task);

	trace_vine_data_free(data, __FUNCTION__, task_duration);
}

vine_task* vine_task_issue(vine_accel *accel, vine_proc *proc, vine_data *args,
                           size_t in_count, vine_data **input, size_t out_count,
                           vine_data **output)
{
	TRACER_TIMER(task);

	trace_timer_start(task);

	vine_pipe_s     *vpipe = vine_pipe_get();
	vine_task_msg_s *task  =
	        arch_alloc_allocate( vpipe->allocator,
	                             sizeof(vine_task_msg_s)+sizeof(vine_data*)*
	                             (in_count+out_count) );
	vine_data_s **dest = (vine_data_s**)task->io;
	utils_queue_s * queue;
	int         cnt;

	task->accel    = accel;
	task->proc     = proc;
	task->args     = args;
	task->in_count = in_count;
	task->stats.task_id = __sync_fetch_and_add(&(task_uid),1);
	for (cnt = 0; cnt < in_count; cnt++) {
		*dest          = *(input++);
		(*dest)->flags = VINE_INPUT;
		dest++;
	}
	task->out_count = out_count;
	for (cnt = 0; cnt < out_count; cnt++) {
		*dest           = *(output++);
		(*dest)->flags |= VINE_OUTPUT;
		async_completion_init(&(vpipe->async),&(*dest)->ready); /* Data might have been used previously */
		dest++;
	}

	/* FIX IT PROPERLY */
	if(((vine_object_s*)accel)->type == VINE_TYPE_PHYS_ACCEL)
		queue = vpipe->queue;
	else
	{
		if(((vine_vaccel_s*)accel)->phys)
			queue = vine_vaccel_queue((vine_vaccel_s*)accel);
		else	// Not yet bound to a physical accel
			queue = vpipe->queue;
	}

	/* Push it or spin */
	while ( !utils_queue_push( queue,task ) )
		;
	task->state = task_issued;

	trace_timer_stop(task);

	trace_vine_task_issue(accel, proc, args, in_count, out_count, input-in_count,
	                    output-out_count, __FUNCTION__, task_duration, task);

	return task;
}

vine_task_state_e vine_task_stat(vine_task *task, vine_task_stats_s *stats)
{
	TRACER_TIMER(task);

	trace_timer_start(task);
	vine_task_msg_s *_task = task;
	vine_task_state_e ret = 0;
	ret = _task->state;

	if(stats)
		memcpy(stats,&(_task->stats),sizeof(*stats));

	trace_timer_stop(task);

	trace_vine_task_stat(task, stats, __FUNCTION__, task_duration,ret);
	return ret;
}

vine_task_state_e vine_task_wait(vine_task *task)
{
	TRACER_TIMER(task);

	trace_timer_start(task);

	vine_task_msg_s *_task = task;
	int             start  = _task->in_count;
	int             end    = start + _task->out_count;
	int             out;
	vine_data_s     *vdata;
	vine_pipe_s     *vpipe = vine_pipe_get();

	for (out = start; out < end; out++) {
		vdata = offset_to_pointer(vine_data_s*, vpipe, _task->io[out]);
		async_completion_wait(&(vpipe->async),&(vdata->ready));
	}

	trace_timer_stop(task);

	trace_vine_task_wait(task, __FUNCTION__, task_duration, _task->state);

	return _task->state;
}

void vine_task_free(vine_task * task)
{
	TRACER_TIMER(task);

	trace_timer_start(task);

	vine_pipe_s     *vpipe = vine_pipe_get();
	arch_alloc_free(vpipe->allocator,task);

	trace_timer_stop(task);

	trace_vine_task_free(task,__FUNCTION__, task_duration);
}
