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

static void        *shm = 0;
static vine_pipe_s *_vpipe;
static char        shm_file[1024];

void prepare_vine_talk();

vine_pipe_s* vine_pipe_get()
{
	if (!_vpipe)
		prepare_vine_talk();
	return _vpipe;
}

#define RING_SIZE 128
#define MY_ID     1

void prepare_vine_talk()
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

	/* Required Confguration Keys */
	if ( !util_config_get_str("shm_file", shm_file, 1024) ) {
		err = __LINE__;
		goto FAIL;
	}

	/* Default /4 of system memory*/
	shm_size = system_total_memory()/4;

	util_config_get_size("shm_size", &shm_size, shm_size);

	if ( !shm_size || shm_size > system_total_memory() ) {
		err = __LINE__;
		goto FAIL;
	}

	/* Optional Confguration Keys */
	util_config_get_size("shm_off", &shm_off, 0);
	util_config_get_bool("shm_trunc", &shm_trunc, 1);
	util_config_get_bool("shm_ivshmem", &shm_ivshmem, 0);

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

	return;

FAIL:   printf("prepare_vine_talk Failed on line %d (file:%s,shm:%p)\n", err,
	       shm_file, shm);
	exit(0);
}                  /* prepare_vine_talk */

void destroy_vine_talk() __attribute__( (destructor) );

void destroy_vine_talk()
{
	vine_pipe_s *vpipe = vine_pipe_get();
	int         last   = vine_pipe_exit(vpipe);

	vpipe = 0;
	printf("%s", __func__);
	printf("vine_pipe_exit() = %d\n", last);
	if (last)
		if ( shm_unlink(shm_file) )
			printf("Could not delete \"%s\"\n", shm_file);

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
	log_timer_start(task);

	vpipe = vine_pipe_get();

	acc_list =
	        vine_object_list_locked(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL);

	if (accels) { /* Want the accels */
		*accels = malloc( acc_list->length*sizeof(vine_accel*) );
		acl     = (vine_accel_s**)*accels;
	}

	utils_list_for_each(*acc_list, itr) {
		accel = (vine_accel_s*)itr;
		if (!type || accel->type == type) {
			accel_count++;
			if (acl) {
				*acl = accel;
				acl++;
			}
		}
	}
	vine_object_list_unlock(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL);

	log_timer_stop(task);

	log_vine_accel_list(type, accels, __FUNCTION__, task_duration,
	                    &accel_count);

	return accel_count;
}

vine_accel_loc_s vine_accel_location(vine_accel *accel)
{
	vine_accel_loc_s ret;

	TRACER_TIMER(task);

	log_timer_start(task);

	/*
	 * TODO: Implement
	 */
	log_timer_stop(task);

	log_vine_accel_location(accel, __FUNCTION__, ret, task_duration);
	return ret;
}

vine_accel_type_e vine_accel_type(vine_accel *accel)
{
	vine_accel_s *_accel;

	TRACER_TIMER(task);

	log_timer_start(task);
	_accel = accel;

	log_timer_stop(task);

	log_vine_accel_type(accel, __FUNCTION__, task_duration, _accel->type);
	return _accel->type;
}

vine_accel_state_e vine_accel_stat(vine_accel *accel, vine_accel_stats_s *stat)
{
	vine_accel_s *_accel;

	TRACER_TIMER(task);

	log_timer_start(task);
	_accel = accel;

	log_timer_stop(task);

	log_vine_accel_stat(accel, stat, __FUNCTION__, task_duration,
	                    (void*)_accel->state);

	return _accel->state;
}

int vine_accel_acquire(vine_accel **accel)
{
	vine_pipe_s  *vpipe;
	vine_accel_s *_accel;
	int          return_value = 0;

	TRACER_TIMER(task);

	vpipe = vine_pipe_get();

	log_timer_start(task);
	_accel = *accel;

	if (_accel->obj.type == VINE_TYPE_PHYS_ACCEL) {
		void *accel_mem =
		        arch_alloc_allocate(&(vpipe->allocator), 4096);

		*accel       = vine_vaccel_init(&(vpipe->objs), accel_mem, 4096,
		                                "FILL", _accel);
		return_value = 1;
	}

	log_timer_stop(task);

	log_vine_accel_acquire(*accel, __FUNCTION__, return_value,
	                       task_duration);

	return return_value;
}

int vine_accel_release(vine_accel **accel)
{
	vine_pipe_s   *vpipe;
	vine_vaccel_s *_accel;
	int           return_value = 0;

	TRACER_TIMER(task);

	vpipe = vine_pipe_get();

	log_timer_start(task);
	_accel = *accel;

	if (_accel->obj.type == VINE_TYPE_VIRT_ACCEL) {
		vine_vaccel_erase(&(vpipe->objs), _accel);
		arch_alloc_free(&(vpipe->allocator), _accel);
		*accel       = 0;
		return_value = 1;
	}


	log_timer_stop(task);

	log_vine_accel_release(*accel, __FUNCTION__,return_value, task_duration);
	return return_value;
}

vine_proc* vine_proc_register(vine_accel_type_e type, const char *func_name,
                              const void *func_bytes, size_t func_bytes_size)
{
	vine_pipe_s *vpipe;
	vine_proc_s *proc;

	TRACER_TIMER(task);

	log_timer_start(task);
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

	log_timer_stop(task);

	log_vine_proc_register(type, func_name, func_bytes, func_bytes_size,
	                       __FUNCTION__, task_duration, proc);

	return proc;
}

vine_proc* vine_proc_get(vine_accel_type_e type, const char *func_name)
{
	TRACER_TIMER(task);

	log_timer_start(task);

	vine_pipe_s *vpipe = vine_pipe_get();
	vine_proc_s *proc  = vine_pipe_find_proc(vpipe, func_name, type);

	if (proc)
		vine_proc_mod_users(proc, +1); /* Increase user count */

	log_timer_stop(task);

	log_vine_proc_get(type, func_name, __FUNCTION__, task_duration,
	                  (void*)proc);

	return proc;
}

int vine_proc_put(vine_proc *func)
{
	TRACER_TIMER(task);

	log_timer_start(task);

	vine_proc_s *proc = func;
	/* Decrease user count */
	int return_value = vine_proc_mod_users(proc, -1);

	log_timer_stop(task);

	log_vine_proc_put(func, __FUNCTION__, task_duration, return_value);

	return return_value;
}

vine_data* vine_data_alloc(size_t size, vine_data_alloc_place_e place)
{
	void *mem;

	TRACER_TIMER(task);

	log_timer_start(task);

	vine_pipe_s *vpipe = vine_pipe_get();

	mem = arch_alloc_allocate( vpipe->allocator, size+sizeof(vine_data_s) );

	vine_data *new_data =
	        vine_data_init(&(vpipe->objs), mem, size, place);
	vine_data *return_val = pointer_to_offset(vine_data*, vpipe, new_data);

	log_timer_stop(task);
	log_vine_data_alloc(size, place, task_duration, __FUNCTION__,
	                    return_val);

	return return_val;
}

size_t vine_data_size(vine_data *data)
{
	vine_data_s *vdata;

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);
	return vdata->size;
}

void* vine_data_deref(vine_data *data)
{
	vine_data_s *vdata;

	TRACER_TIMER(task);

	log_timer_start(task);

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);

	log_timer_stop(task);

	if (!vdata->place&HostOnly) {
		log_vine_data_deref(data, __FUNCTION__, task_duration, 0);
		return 0;
	}

	log_vine_data_deref( data, __FUNCTION__, task_duration,
	                     (void*)(vdata+1) );
	return (void*)(vdata+1);
}

void vine_data_mark_ready(vine_data *data)
{
	vine_data_s *vdata;

	TRACER_TIMER(task);

	log_timer_start(task);

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);
	__sync_bool_compare_and_swap(&(vdata->ready), 0, 1);

	log_timer_stop(task);

	log_vine_data_mark_ready(data, __FUNCTION__, task_duration);
}

void vine_data_free(vine_data *data)
{
	vine_data_s *vdata;

	TRACER_TIMER(task);

	log_timer_start(task);

	vine_pipe_s *vpipe = vine_pipe_get();

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);
	vine_data_erase(&(vpipe->objs), vdata);
	arch_alloc_free(vpipe->allocator, vdata);

	log_timer_stop(task);

	log_vine_data_free(data, __FUNCTION__, task_duration);
}

vine_task* vine_task_issue(vine_accel *accel, vine_proc *proc, vine_data *args,
                           size_t in_count, vine_data **input, size_t out_count,
                           vine_data **output)
{
	TRACER_TIMER(task);

	log_timer_start(task);

	vine_pipe_s     *vpipe = vine_pipe_get();
	vine_task_msg_s *task  =
	        arch_alloc_allocate( vpipe->allocator,
	                             sizeof(vine_task_msg_s)+sizeof(vine_data*)*
	                             (in_count+out_count) );
	vine_data_s **dest = (vine_data_s**)task->io;
	int         cnt;

	task->accel    = accel;
	task->proc     = proc;
	task->args     = args;
	task->in_count = in_count;
	for (cnt = 0; cnt < in_count; cnt++) {
		*dest          = *(input++);
		(*dest)->flags = VINE_INPUT;
		dest++;
	}
	task->out_count = out_count;
	for (cnt = 0; cnt < out_count; cnt++) {
		*dest           = *(output++);
		(*dest)->flags |= VINE_OUTPUT;
		dest++;
	}
	/* Push it or spin */
	while ( !utils_queue_push( vpipe->queue,
	                           pointer_to_offset(void*, vpipe, task) ) )
		;
	task->state = task_issued;

	log_timer_stop(task);

	/*TODO PROFILER incnt outcnt*/
	log_vine_task_issue(accel, proc, args, in_count, out_count, input,
	                    output, __FUNCTION__, task_duration, task);

	return task;
}

vine_task_state_e vine_task_stat(vine_task *task, vine_task_stats_s *stats)
{
	TRACER_TIMER(task);

	log_timer_start(task);

	log_timer_stop(task);

	log_vine_task_stat(task, stats, __FUNCTION__, task_duration,
	                   task_failed);
	return task_failed;
}

vine_task_state_e vine_task_wait(vine_task *task)
{
	TRACER_TIMER(task);

	log_timer_start(task);

	vine_task_msg_s *_task = task;
	int             start  = _task->in_count;
	int             end    = start + _task->out_count;
	int             out;
	vine_data_s     *vdata;

	for (out = start; out < end; out++) {
		vdata = offset_to_pointer(vine_data_s*, vpipe, _task->io[out]);
		while (!vdata->ready)
			;
	}

	log_timer_stop(task);

	log_vine_task_wait(task, __FUNCTION__, task_duration, _task->state);

	return _task->state;
}
