#include <vine_talk.h>
#include <vine_pipe.h>
#include "arch/alloc.h"
#include "core/vine_data.h"
#include "core/vine_accel.h"
#include "utils/queue.h"
#include "utils/config.h"
#include "utils/trace.h"
#include "utils/system.h"
#include "utils/btgen.h"
#include "utils/timer.h"
#include "utils/vine_assert.h"
#include "utils/breakdown.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

const char * vine_talk_version = "VT_VERSION " VINE_TALK_GIT_REV " - " VINE_TALK_GIT_BRANCH;

struct
{
	vine_pipe_s       *vpipe;
	char              shm_file[1024];
	uint64_t          threads;
	uint64_t          instance_uid;
	uint64_t          task_uid;
	volatile uint64_t initialized;
	char              *config_path;
	int               fd;
} vine_state =
{(void*)CONF_VINE_MMAP_BASE,{'\0'},0,0,0,0,NULL};

#define vine_pipe_get() vine_state.vpipe

#define GO_FAIL(MSG) ({err = __LINE__;err_msg = MSG;goto FAIL;})

vine_pipe_s * vine_talk_init()
{
	vine_pipe_s * shm_addr = 0;
	int    err         = 0;
	size_t shm_size    = 0;
	size_t shm_off     = 0;
	int    shm_trunc   = 0;
	int    shm_ivshmem = 0;
	int    enforce_version = 0;
	int    mmap_prot   = PROT_READ|PROT_WRITE|PROT_EXEC;
	int    mmap_flags  = MAP_SHARED;
	const char * err_msg = "No Error Set";
#ifdef MMAP_POPULATE
	mmap_flags |= MAP_POPULATE;
#endif

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
	if ( !utils_config_get_str(vine_state.config_path,"shm_file", vine_state.shm_file, 1024,0) )
		GO_FAIL("No shm_file set in config");

	/* Default /4 of system memory*/
	shm_size = system_total_memory()/4;

	utils_config_get_size(vine_state.config_path,"shm_size", &shm_size, shm_size);

	if ( !shm_size || shm_size > system_total_memory() )
		GO_FAIL("shm_size exceeds system memory");

	/* Optional Confguration Keys */
	utils_config_get_size(vine_state.config_path,"shm_off", &shm_off, 0);
	utils_config_get_bool(vine_state.config_path,"shm_trunc", &shm_trunc, 1);
	utils_config_get_bool(vine_state.config_path,"shm_ivshmem", &shm_ivshmem, 0);
	utils_config_get_bool(vine_state.config_path,"enforce_version", &enforce_version, 1);

	if (vine_state.shm_file[0] == '/')
		vine_state.fd = open(vine_state.shm_file, O_CREAT|O_RDWR, 0644);
	else
		vine_state.fd = shm_open(vine_state.shm_file, O_CREAT|O_RDWR, S_IRWXU);

	if (vine_state.fd < 0)
		GO_FAIL("Could not open shm_file");

	if (shm_ivshmem) {
		shm_off  += 4096; /* Skip register section */
		shm_trunc = 0; /* Don't truncate ivshm  device */
	}

	if (shm_trunc) /* If shm_trunc */
	{
		if(system_file_size(vine_state.shm_file) != shm_size)
		{		/* If not the correct size */
			if ( ftruncate(vine_state.fd, shm_size) )
				GO_FAIL("Could not truncate shm_file");
		}
	}

	vine_state.vpipe = mmap(vine_state.vpipe, shm_size, mmap_prot, mmap_flags,
							vine_state.fd, shm_off);

	if (!vine_state.vpipe || vine_state.vpipe == MAP_FAILED)
		GO_FAIL("Could not first mmap shm_file");

	shm_addr =  vine_pipe_mmap_address(vine_state.vpipe);

	if(shm_addr != vine_state.vpipe)
	{

		munmap(vine_state.vpipe,vine_state.vpipe->shm_size);	// unmap misplaced map.

		vine_state.vpipe = mmap(shm_addr, shm_size, mmap_prot, mmap_flags,
							vine_state.fd, shm_off);

	}

	if (!vine_state.vpipe || vine_state.vpipe == MAP_FAILED || vine_state.vpipe != shm_addr)
		GO_FAIL("Could not mmap shm_file in proper address");

	vine_state.vpipe = vine_pipe_init(vine_state.vpipe, shm_size, enforce_version);

	if(!vine_state.vpipe)
		GO_FAIL("Could not initialize vine_pipe");

	async_meta_init_always( &(vine_state.vpipe->async) );
	printf("ShmFile:%s\n", vine_state.shm_file);
	printf("ShmLocation:%p\n", vine_state.vpipe);
	printf("ShmSize:%zu\n", vine_state.vpipe->shm_size);
	vine_state.instance_uid = __sync_fetch_and_add(&(vine_state.vpipe->last_uid),1);
	printf("InstanceUID:%zu\n", vine_state.instance_uid);
	vine_state.initialized = 1;
	return vine_state.vpipe;

	FAIL:
	printf("%c[31mprepare_vine_talk Failed on line %d (conf:%s,file:%s,shm:%p)\n\
			Why:%s%c[0m\n",27, err,VINE_CONFIG_FILE,vine_state.shm_file,
			vine_state.vpipe,err_msg,27);
	munmap(vine_state.vpipe,vine_state.vpipe->shm_size);
	exit(1);
}                  /* vine_task_init */

#undef GO_FAIL

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
		if( __sync_fetch_and_add(&(vine_state.threads),-1) == 1)
		{	// Last thread of process

			last = vine_pipe_exit(vine_state.vpipe);

			if( last )
			{
				size_t  available 	= vine_pipe_get_available_size(vine_state.vpipe);
				size_t  total		= vine_pipe_get_total_size(vine_state.vpipe);
				#ifdef VINE_THROTTLE_DEBUG
				vine_assert( available == total);
				#endif

				if( available != total ){
					printf("\033[1;31mERROR : shm LEAK !!\n\033[0m");
				}else{
					printf("\033[1;32mSHM GOOD !!\n\033[0m");
				}
			}

			munmap(vine_state.vpipe,vine_state.vpipe->shm_size);

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

void vine_accel_set_physical(vine_accel* vaccel,vine_accel* phys){
	vine_assert(phys);
	vine_assert(vaccel);
	vine_vaccel_s*    acl    = (vine_vaccel_s*)vaccel;
	vine_assert(acl);
	acl->phys = phys ;
}

void vine_accel_list_free_pre_locked(vine_accel **accels);

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
		if(*accels)
			vine_accel_list_free_pre_locked(*accels);
		*accels = malloc( (acc_list->length+1)*sizeof(vine_accel*) );
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
					vine_object_ref_inc(&(accel->obj));
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
					vine_object_ref_inc(&(accel->obj));
					*acl = (vine_accel_s*)accel;
					acl++;
				}
			}
		}
	}
	if(acl)
		*acl = 0;	// delimiter
	vine_object_list_unlock(&(vpipe->objs), ltype);

	trace_timer_stop(task);

	trace_vine_accel_list(type, physical, accels, __FUNCTION__, trace_timer_value(task),
	                    accel_count);

	return accel_count;
}

void vine_accel_list_free(vine_accel **accels)
{
	vine_object_s ** itr = (vine_object_s **)accels;

	while(*itr)
	{
		vine_object_ref_dec(*itr);
		itr++;
	}
	free(accels);
}

void vine_accel_list_free_pre_locked(vine_accel **accels)
{
	vine_object_s ** itr = (vine_object_s **)accels;

	while(*itr)
	{
		vine_object_ref_dec_pre_locked(*itr);
		itr++;
	}
	free(accels);
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
			ret = accel_failed;	/* Not very 'correct' */
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
		*accel = vine_vaccel_init(vpipe, "FILL",_accel->type, _accel);
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

	_accel = (vine_accel_s*)vine_vaccel_init(vpipe, "FILL",type, 0);

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

	if(
		type && 					// Can not create an ANY procedure.
		type < VINE_ACCEL_TYPES		// type is a valid vine_accel_type_e.
	)
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
		vine_object_ref_inc(&(proc->obj));
	else
		fprintf(stderr,"Proc %s not found!\n",func_name);
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
	int return_value = vine_object_ref_dec(&(proc->obj));

	trace_timer_stop(task);

	trace_vine_proc_put(func, __FUNCTION__, trace_timer_value(task), return_value);

	return return_value;
}

int check_semantics(size_t in_count,vine_data **input, size_t out_count,
					vine_data **output){
	size_t      io_cnt;
	size_t      dup_cnt;
	size_t      all_io = out_count + in_count;
	vine_data_s *temp_data_1 = 0;
	vine_data_s *temp_data_2 = 0;

	for(io_cnt = 0 ; io_cnt < all_io ; io_cnt++)
	{
		//Choose from input or output
		if(io_cnt<in_count)
			temp_data_1 = input[io_cnt];
		else
			temp_data_1 = output[io_cnt-in_count];
		//check Validity temp_data_1
		if(!temp_data_1)
		{
			fprintf(stderr,"NULL input #%lu\n",io_cnt);
			return 0;
		}
		if( temp_data_1->obj.type != VINE_TYPE_DATA )
		{
			fprintf(stderr,"Input #%lu not valid data\n",io_cnt);
			return 0;
		}
		//Check duplicates
		for(dup_cnt = 0 ; dup_cnt < all_io ; dup_cnt++)
		{
			//Choose from input or output
			if(dup_cnt<in_count)
				temp_data_2 = input[dup_cnt];
			else
				temp_data_2 = output[dup_cnt-in_count];
			//check Validity temp_data_2
			if(!temp_data_2)
			{
				fprintf(stderr,"NULL input #%lu\n",dup_cnt);
				return 0;
			}
			if(temp_data_2->obj.type != VINE_TYPE_DATA)
			{
				fprintf(stderr,"Input #%lu not valid data\n",dup_cnt);
				return 0;
			}
			//check dup
			if( temp_data_1 != temp_data_2 )
			{
				if((temp_data_1->user || temp_data_2->user) &&
				   (temp_data_1->user == temp_data_2->user) )
				{
					fprintf(stderr,"vine_data(%p,%p) point to one user (%p)\n",temp_data_1,temp_data_2,temp_data_1->user);
					return 0;
				}
			}
		}
	}
	return 1;
}

int vine_data_remote_check(vine_data_s* data){
	vine_assert(data!=NULL);
	return (data->remote == NULL ? 1 : 0) ;
}

void check_accel_size_and_sync(vine_accel *accel, vine_proc *proc ,size_t in_count,
						   vine_data **input, size_t out_count,vine_data **output
						   , void *args, size_t args_size){

	vine_object_s * proc_obj = (vine_object_s *)proc;
	if( !strcmp(proc_obj->name,"free") )
	{
		vine_pipe_size_dec( vine_pipe_get() ,_VINE_DATA_CALC_SIZE (args_size,1) );
		return;
	}

	int i,j;
	size_t sync_size_accel = 0;
	size_t sync_size_pipe = 0;
		//Sum sync size to phys
	for( i = 0 ;  i < in_count;  i++){
		if(vine_data_remote_check((vine_data_s*)input[i])){
			#ifdef VINE_THROTTLE_DEBUG
			printf("size of (%p)->input[%d]: %lu\n",input[i],i,vine_data_size((vine_data_s*)input[i]));
			#endif
			sync_size_accel += vine_data_size((vine_data_s*)input[i]);
			sync_size_pipe  += VINE_DATA_CALC_SIZE ((vine_data_s*)input[i] );
		}
	}
		#ifdef VINE_THROTTLE_DEBUG
	size_t tmp1 = sync_size_accel;
	size_t tmp2 = sync_size_pipe;
	printf("In_count: %lu Out_count: %lu \n",in_count,out_count);
	printf("Accel Input size : %lu \n",tmp1);
	printf("Shm   Input size : %lu \n",tmp2);
	#endif
		for( i = 0 ;  i < out_count;  i++){
		int dup_flag=0;
		//first check for same  in-out
		for( j = 0 ;  j < in_count && dup_flag==0 ;  j++){
			if(  output[i] == input[j] )
			{
				dup_flag = 1;
			}
		}
		//if not same
		if(!dup_flag){
			if(vine_data_remote_check((vine_data_s*)output[i])){
				#ifdef VINE_THROTTLE_DEBUG
				printf("size of (%p)->output[%d]: %lu\n",output[i],i,vine_data_size((vine_data_s*)input[i]));
				#endif
				sync_size_accel += vine_data_size((vine_data_s*)output[i]);
				sync_size_pipe  += VINE_DATA_CALC_SIZE( (vine_data_s*)output[i] );
			}
		}
	}

	#ifdef VINE_THROTTLE_DEBUG
	printf("Accel Output size : %lu sum: %lu\n",sync_size_accel-tmp1,sync_size_accel);
	printf("Shm   Output size : %lu sum: %lu\n",sync_size_pipe-tmp2,sync_size_pipe);
	if( args_size > 0)
		printf("Shm   Args  size  : %lu sum: %lu +aling:%lu\n",args_size,sync_size_pipe ,  _VINE_DATA_CALC_SIZE (args_size,1));
	printf("SHM task_issue\t");
	#endif
		//add arguments size
	//align is always 0 here because it allocates only in task issue
	if( args_size > 0)
	{
		sync_size_pipe += _VINE_DATA_CALC_SIZE (args_size,1);
	}
		//Check if phys exists if not init
	if( ((vine_vaccel_s*)accel)->phys == NULL ){
		((vine_vaccel_s*)accel)->phys = (void*)0xBAADF00D;
		vine_proc_s * init_phys = vine_proc_get(((vine_vaccel_s*)accel)->type,"init_phys");
		vine_task_msg_s * task = vine_task_issue(accel,init_phys,0,0,0,0,0,0);
		vine_assert( vine_task_wait(task) == task_completed );
		vine_proc_put(init_phys);
		vine_task_free(task);
	}

	//Dec pipe size
	if(sync_size_pipe)
		vine_pipe_size_dec( vine_pipe_get() ,sync_size_pipe );

	#ifdef VINE_THROTTLE_DEBUG
	printf("VACCEL throttle\t");
	#endif
	//Dec accel size
	if(sync_size_accel)
		vine_accel_size_dec(((vine_vaccel_s*)accel)->phys,sync_size_accel);
}

vine_task* vine_task_issue(vine_accel *accel, vine_proc *proc, void *args,size_t args_size,
                           size_t in_count, vine_data **input, size_t out_count,
						   vine_data **output)
{
	TRACER_TIMER(task);
	//printf("%s %s\n",__func__, ((vine_proc_s*)proc)->obj.name) ;
	trace_timer_start(task);

	vine_pipe_s     *vpipe = vine_pipe_get();
	vine_task_msg_s *task;
	vine_data **dest;
	int cnt;

	vine_assert(check_semantics(in_count,input,out_count,output));

	task  = vine_task_alloc(vpipe,in_count,out_count);

	vine_assert(task);
	utils_breakdown_instance_set_vaccel(&(task->breakdown),accel);

	utils_breakdown_begin(&(task->breakdown),&(((vine_proc_s*)proc)->breakdown),"Inp_Cpy");

	vine_assert(accel);
	vine_assert(proc);

	check_accel_size_and_sync(accel,proc,in_count,input,out_count,output,args,args_size);

	task->accel    = accel;
	task->proc     = proc;

	if(args && args_size)
	{
		//printf("ARGs allocation \n");
		task->args = vine_data_init(vpipe,args,args_size);
		//check vine_data_s* initialization
		vine_assert(task->args!=0);
		vine_data_arg_init(task->args,accel);
		//check vine_data_s->buffer initlizations
		vine_assert(vine_data_deref(task->args)!=0);
		vine_data_modified(task->args,USER_SYNC|SHM_SYNC);
		/*I clean buffer inside vine_data_allocate.*/
		memcpy(vine_data_deref(task->args),args,args_size);
		vine_data_annotate(task->args,"%s:Args",((vine_proc_s*)proc)->obj.name);
	}
	else
		task->args = 0;

	task->stats.task_id = __sync_fetch_and_add(&(vine_state.task_uid),1);

	dest = task->io;

	for(cnt = 0 ; cnt < in_count; cnt++,dest++)
	{
		if(!input[cnt])
		{
			fprintf(stderr,"Invalid input #%d\n",cnt);
			return 0;
		}
		*dest = input[cnt];
		if(((vine_data_s*)*dest)->obj.type != VINE_TYPE_DATA)
		{
			fprintf(stderr,"Input #%d not valid data\n",cnt);
			return 0;
		}
		//printf("Input allocation \n");
		vine_data_input_init(*dest,accel);
		vine_data_annotate(*dest,"%s:in[%d]",((vine_proc_s*)proc)->obj.name,cnt);
		// Sync up to shm if neccessary
		vine_data_modified(*dest, USER_SYNC);
		vine_data_sync_to_remote(accel,*dest,0);
	}


	for(cnt = 0 ; cnt < out_count; cnt++,dest++)
	{
		*dest = output[cnt];
		if(!*dest)
		{
			fprintf(stderr,"Invalid output #%d\n",cnt);
			return 0;
		}
		if(((vine_data_s*)*dest)->obj.type != VINE_TYPE_DATA)
		{
			fprintf(stderr,"Input #%d not valid data\n",cnt);
			return 0;
		}
		//printf("Output allocation \n" );
		vine_data_output_init(*dest,accel);
		vine_data_annotate(*dest,"%s:out[%d]",((vine_proc_s*)proc)->obj.name,cnt);
	}

	vine_task_submit(task);

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

	utils_breakdown_advance(&(_task->breakdown),"Wait_For_Cntrlr");
	vine_task_wait_done(_task);

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

	vine_object_ref_dec(&(_task->obj));

	utils_breakdown_end(&(_task->breakdown));

	trace_timer_stop(task);

	trace_vine_task_free(task,__FUNCTION__, trace_timer_value(task));
}

vine_buffer_s VINE_BUFFER(void * user_buffer,size_t size)
{
	vine_pipe_s     *vpipe = vine_pipe_get();

	vine_data_s * vd = vine_data_init(vpipe,user_buffer,size);

	return vd;

}
