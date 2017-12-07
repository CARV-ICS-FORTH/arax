#include <vine_talk.h>
#include <vine_pipe.h>
#include "arch/alloc.h"
#include "core/vine_data.h"
#include "utils/queue.h"
#include "utils/config.h"
#include "utils/trace.h"
#include "utils/system.h"
#include "utils/btgen.h"
#include "utils/timer.h"
#include "utils/breakdown.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

struct
{
	void              *shm;
	vine_pipe_s       *vpipe;
	char              shm_file[1024];
	uint64_t          threads;
	uint64_t          instance_uid;
	uint64_t          task_uid;
	volatile uint64_t initialized;
	char              *config_path;
	int               fd;
} vine_state =
{NULL,NULL,{'\0'},0,0,0,0,NULL};

#define vine_pipe_get() vine_state.vpipe

vine_pipe_s * vine_talk_init()
{
	int    err         = 0;
	size_t shm_size    = 0;
	size_t shm_off     = 0;
	size_t old_shm_size = 0;
	int    shm_trunc   = 0;
	int    shm_ivshmem = 0;
	int    enforce_version = 0;
	int    remap       = 0;
	int    mmap_prot   = PROT_READ|PROT_WRITE|PROT_EXEC;
	int    mmap_flags  = MAP_SHARED;
	const char * err_msg = "No Error Set";
#ifdef MMAP_POPULATE
	mmap_flags |= MAP_POPULATE;
#endif

	printf("%s Thread id:%lu\n",__func__,vine_state.threads);

	if( __sync_fetch_and_add(&(vine_state.threads),1) != 0)
	{	// I am not the first but stuff might not yet be initialized
		while(!vine_state.initialized); // wait for initialization
		return vine_state.vpipe;
	}

	utils_bt_init();

	vine_state.config_path = utils_config_alloc_path(VINE_CONFIG_FILE);

	utils_breakdown_init_telemetry(vine_state.config_path);

	#ifdef TRACE_ENABLE
	trace_init();
	#endif

	printf("Config:%s\n",VINE_CONFIG_FILE);

	/* Required Confguration Keys */
	if ( !utils_config_get_str(vine_state.config_path,"shm_file", vine_state.shm_file, 1024,0) ) {
		err = __LINE__;
		err_msg = "No shm_file set in config";
		goto FAIL;
	}

	/* Default /4 of system memory*/
	shm_size = system_total_memory()/4;

	utils_config_get_size(vine_state.config_path,"shm_size", &shm_size, shm_size);

	if ( !shm_size || shm_size > system_total_memory() ) {
		err = __LINE__;
		err_msg = "shm_size exceeds system memory";
		goto FAIL;
	}

	/* Optional Confguration Keys */
	utils_config_get_size(vine_state.config_path,"shm_off", &shm_off, 0);
	utils_config_get_bool(vine_state.config_path,"shm_trunc", &shm_trunc, 1);
	utils_config_get_bool(vine_state.config_path,"shm_ivshmem", &shm_ivshmem, 0);
	utils_config_get_bool(vine_state.config_path,"enforce_version", &enforce_version, 1);

	if (vine_state.shm_file[0] == '/')
		vine_state.fd = open(vine_state.shm_file, O_CREAT|O_RDWR, 0644);
	else
		vine_state.fd = shm_open(vine_state.shm_file, O_CREAT|O_RDWR, S_IRWXU);

	if (vine_state.fd < 0) {
		err = __LINE__;
		err_msg = "Could not open shm_file";
		goto FAIL;
	}

	if (shm_ivshmem) {
		shm_off  += 4096; /* Skip register section */
		shm_trunc = 0; /* Don't truncate ivshm  device */
	}

	if (shm_trunc) /* If shm_trunc */
	{
		if(system_file_size(vine_state.shm_file) != shm_size)
		{		/* If not the correct size */
			if ( ftruncate(vine_state.fd, shm_size) ) {
				err = __LINE__;
				err_msg = "Could not truncate shm_file";
				goto FAIL;
			}
		}
	}

#if CONF_VINE_MMAP_BASE!=0
	vine_state.shm = (void*)CONF_VINE_MMAP_BASE;
#endif
	do {
		vine_state.shm = mmap(vine_state.shm, shm_size, mmap_prot, mmap_flags,
							  vine_state.fd, shm_off);

		if (!vine_state.shm || vine_state.shm == MAP_FAILED) {
			err = __LINE__;
			err_msg = "Could not mmap shm_file";
			goto FAIL;
		}

		if(vine_state.vpipe) // Already initialized, so just remaped
			vine_state.vpipe = (vine_pipe_s*)vine_state.shm;
		else
			vine_state.vpipe = vine_pipe_init(vine_state.shm, shm_size, enforce_version);
		vine_state.shm    = vine_state.vpipe->self; /* This is where i want to go */

		if(!vine_state.vpipe)
		{
			err = __LINE__;
			err_msg = "Could initialize vine_pipe";
			goto FAIL;
		}

		if (vine_state.vpipe != vine_state.vpipe->self) {
			printf("Remapping from %p to %p.\n", vine_state.vpipe, vine_state.shm);
			remap = 1;
		}

		old_shm_size = shm_size;
		if (shm_size != vine_state.vpipe->shm_size) {
			printf("Resizing from %lu to %lu.\n", shm_size,
			       vine_state.vpipe->shm_size);
			shm_size = vine_state.vpipe->shm_size;
			remap    = 1;
		}

		if(remap)
		{
			printf("Unmaping address %p size: %lu\n",vine_state.vpipe,old_shm_size);
			if(munmap(vine_state.vpipe,old_shm_size) == -1)
			{
				err_msg = "Unmap operation failed";
				goto FAIL;
			}
		}


	} while (remap--); /* Not where i want */
	async_meta_init_always( &(vine_state.vpipe->async) );
	printf("ShmFile:%s\n", vine_state.shm_file);
	printf("ShmLocation:%p\n", vine_state.shm);
	printf("ShmSize:%zu\n", shm_size);
	vine_state.instance_uid = __sync_fetch_and_add(&(vine_state.vpipe->last_uid),1);
	printf("InstanceUID:%zu\n", vine_state.instance_uid);
	vine_state.initialized = 1;
	return vine_state.vpipe;

	FAIL:
	printf("%c[31mprepare_vine_talk Failed on line %d (conf:%s,file:%s,shm:%p)\n\
			Why:%s%c[0m\n",27, err,VINE_CONFIG_FILE,vine_state.shm_file,
			vine_state.shm,err_msg,27);
	shm_unlink(vine_state.shm_file);
	exit(0);
}                  /* vine_task_init */

uint64_t vine_talk_instance_uid()
{
	return vine_state.instance_uid;
}

void vine_talk_exit()
{
	int last;

	if(vine_state.vpipe)
	{

		#ifdef TRACE_ENABLE
		trace_exit();
		#endif
		printf("%s Thread id:%lu\n",__func__,vine_state.threads);
		if( __sync_fetch_and_add(&(vine_state.threads),-1) == 1)
		{	// Last thread of process

			last = vine_pipe_exit(vine_state.vpipe);

			munmap(vine_state.shm,vine_state.vpipe->shm_size);

			vine_state.vpipe = 0;

			utils_config_free_path(vine_state.config_path);
			printf("vine_pipe_exit() = %d\n", last);
			close(vine_state.fd);
			if (last)
				if ( shm_unlink(vine_state.shm_file) )
					printf("Could not delete \"%s\"\n", vine_state.shm_file);
		}
	}
	else
		fprintf(stderr,
		"WARNING:vine_talk_exit() called with no matching\
		call to vine_talk_init()!\n");
	utils_bt_exit();
}

int vine_accel_list(vine_accel_type_e type, int physical, vine_accel ***accels)
{
	vine_pipe_s        *vpipe;
	utils_list_node_s  *itr;
	utils_list_s       *acc_list;

	vine_accel_s       **acl       = 0;
	int                accel_count = 0;
	vine_object_type_e ltype;

	TRACER_TIMER(task);
	trace_timer_start(task);

	if(physical)
		ltype = VINE_TYPE_PHYS_ACCEL;
	else
		ltype = VINE_TYPE_VIRT_ACCEL;

	vpipe = vine_pipe_get();

	acc_list =
	        vine_object_list_lock(&(vpipe->objs), ltype);

	if (accels) { /* Want the accels */
		*accels = malloc( acc_list->length*sizeof(vine_accel*) );
		acl     = (vine_accel_s**)*accels;
	}

	if(physical)
	{
		vine_accel_s *accel = 0;
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
	}
	else
	{
		vine_vaccel_s *accel = 0;
		utils_list_for_each(*acc_list, itr) {
			accel = (vine_vaccel_s*)itr->owner;
			if (!type || accel->type == type) {
				accel_count++;
				if (acl) {
					*acl = (vine_accel_s*)accel;
					acl++;
				}
			}
		}
	}
	vine_object_list_unlock(&(vpipe->objs), ltype);

	trace_timer_stop(task);

	trace_vine_accel_list(type, physical, accels, __FUNCTION__, trace_timer_value(task),
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

	trace_vine_accel_location(accel, __FUNCTION__, ret, trace_timer_value(task));
	return ret;
}

vine_accel_type_e vine_accel_type(vine_accel *accel)
{
	vine_accel_s *_accel;

	TRACER_TIMER(task);

	trace_timer_start(task);
	_accel = accel;

	trace_timer_stop(task);

	trace_vine_accel_type(accel, __FUNCTION__, trace_timer_value(task), _accel->type);
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

	trace_vine_accel_stat(accel, stat, __FUNCTION__, trace_timer_value(task),ret);

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
		*accel = vine_vaccel_init(&(vpipe->objs), "FILL",_accel->type, _accel);
		return_value = 1;
	}

	trace_timer_stop(task);

	trace_vine_accel_acquire_phys(*accel, __FUNCTION__, trace_timer_value(task));

	return return_value;
}

vine_accel * vine_accel_acquire_type(vine_accel_type_e type)
{
	vine_pipe_s  *vpipe;
	vine_accel_s *_accel = 0;
	TRACER_TIMER(task);

	vpipe = vine_pipe_get();

	_accel = (vine_accel_s*)vine_vaccel_init(&(vpipe->objs), "FILL",type, 0);

	trace_timer_stop(task);

	trace_vine_accel_acquire_type(type, __FUNCTION__, _accel,trace_timer_value(task));
	return (vine_accel *)_accel;
}

void vine_accel_release(vine_accel **accel)
{
	vine_vaccel_s *_accel;

	TRACER_TIMER(task);

	trace_timer_start(task);
	_accel = *accel;

	vine_object_ref_dec(&(_accel->obj));

	*accel = 0;

	trace_timer_stop(task);

	trace_vine_accel_release(*accel, __FUNCTION__, trace_timer_value(task));
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
			proc = vine_proc_init(&(vpipe->objs), func_name, type,
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
	                       __FUNCTION__, trace_timer_value(task), proc);

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

	trace_vine_proc_get(type, func_name, __FUNCTION__, trace_timer_value(task),
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

	trace_vine_proc_put(func, __FUNCTION__, trace_timer_value(task), return_value);

	return return_value;
}

vine_task* vine_task_issue(vine_accel *accel, vine_proc *proc, vine_buffer_s *args,
                           size_t in_count, vine_buffer_s *input, size_t out_count,
						   vine_buffer_s *output)
{
	TRACER_TIMER(task);

	trace_timer_start(task);

	vine_pipe_s     *vpipe = vine_pipe_get();
	vine_task_msg_s *task  =
	        arch_alloc_allocate( &(vpipe->allocator),
	                             sizeof(vine_task_msg_s)+sizeof(vine_buffer_s)*
	                             (in_count+out_count) );
	vine_buffer_s*dest = (vine_buffer_s*)task->io;
	vine_data_s * data;
	utils_queue_s * queue;
	int         in_cnt;
	int         out_cnt;

	utils_breakdown_instance_init(&(task->breakdown));

	utils_breakdown_instance_set_vaccel(&(task->breakdown),accel);

	utils_breakdown_begin(&(task->breakdown),&(((vine_proc_s*)proc)->breakdown),"Inp_Cpy");

	task->accel    = accel;
	task->proc     = proc;
	if(args)
	{
		data = vine_data_init(vpipe,args->user_buffer_size,HostOnly);
		vine_buffer_init(&(task->args),args->user_buffer,args->user_buffer_size,data,1);
	}
	else
		task->args.vine_data = 0;
	task->in_count = in_count;
	task->stats.task_id = __sync_fetch_and_add(&(vine_state.task_uid),1);
	for (in_cnt = 0; in_cnt < in_count; in_cnt++) {
		data = vine_data_init(vpipe,input->user_buffer_size,Both);
		vine_buffer_init(dest,input->user_buffer,input->user_buffer_size,data,1);
		data->flags = VINE_INPUT;
		input++;
		dest++;
	}
	utils_breakdown_advance(&(task->breakdown),"Out_Cpy");
	task->out_count = out_count;
	input = task->io; // Reset input pointer
	for (out_cnt = 0; out_cnt < out_count; out_cnt++) {
		data = 0;
		for(in_cnt = 0 ; in_cnt < in_count ; in_cnt++)
		{
			if(input[in_cnt].user_buffer == output->user_buffer)
			{	// Buffer is I&O
				data = input[in_cnt].vine_data;
				break;
			}
		}
		if(!data)
			data = vine_data_init(vpipe,output->user_buffer_size,Both);
		data->flags |= VINE_OUTPUT;
		async_completion_init(&(vpipe->async),&(data->ready)); /* Data might have been used previously */
		vine_buffer_init(dest,output->user_buffer,output->user_buffer_size,data,0);
		output++;
		dest++;
	}
	utils_breakdown_advance(&(task->breakdown),"Issue");

	if(((vine_object_s*)accel)->type == VINE_TYPE_PHYS_ACCEL)
	{
		task->type = ((vine_accel_s*)accel)->type;
		queue = vpipe->queue;
	}
	else
	{
		task->type = ((vine_vaccel_s*)accel)->type;
		queue = vine_vaccel_queue((vine_vaccel_s*)accel);
	}
	utils_timer_set(task->stats.task_duration,start);
	/* Push it or spin */
	while ( !utils_queue_push( queue,task ) )
		;
	task->state = task_issued;
	vine_pipe_add_task(vpipe,task->type);

	trace_timer_stop(task);


	trace_vine_task_issue(accel, proc, args, in_count, out_count, input-in_count,
	                    output-out_count, __FUNCTION__, trace_timer_value(task), task);

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

	trace_vine_task_stat(task, stats, __FUNCTION__, trace_timer_value(task),ret);
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

	utils_breakdown_advance(&(_task->breakdown),"Wait_For_Cntrlr");
	for (out = start; out < end; out++) {
		vdata = offset_to_pointer(vine_data_s*, vpipe, _task->io[out].vine_data);
		async_completion_wait(&(vpipe->async),&(vdata->ready));
	}


	utils_breakdown_advance(&(_task->breakdown),"Copy_Out_Buffs");
	for (out = start; out < end; out++) {
		vdata = offset_to_pointer(vine_data_s*, vpipe, _task->io[out].vine_data);
		memcpy(_task->io[out].user_buffer,vine_data_deref(vdata),_task->io[out].user_buffer_size);
	}

	trace_timer_stop(task);

	trace_vine_task_wait(task, __FUNCTION__, trace_timer_value(task), _task->state);

	utils_breakdown_advance(&(_task->breakdown),"Gap_To_Free");
	return _task->state;
}

void vine_task_free(vine_task * task)
{
	TRACER_TIMER(task);

	trace_timer_start(task);
	vine_task_msg_s *_task = task;
	void * prev;
 	int cnt;

	utils_breakdown_advance(&(_task->breakdown),"TaskFree");

	vine_pipe_s     *vpipe = vine_pipe_get();

	if(_task->args.vine_data)
		vine_data_free(vpipe, _task->args.vine_data);

	// Sort them pointers
	qsort(_task->io,_task->in_count+_task->out_count,sizeof(vine_buffer_s),vine_buffer_compare);
	prev = 0;
	for(cnt = 0 ; cnt < _task->in_count+_task->out_count ; cnt++)
	{
		if(prev != _task->io[cnt].vine_data)
		{
			prev = _task->io[cnt].vine_data;
			vine_data_free(vpipe, _task->io[cnt].vine_data);
		}
	}

	arch_alloc_free(&(vpipe->allocator),task);
	utils_breakdown_end(&(_task->breakdown));

	trace_timer_stop(task);

	trace_vine_task_free(task,__FUNCTION__, trace_timer_value(task));
}
