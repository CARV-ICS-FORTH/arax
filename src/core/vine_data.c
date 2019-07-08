#include "vine_data.h"
#include "vine_pipe.h"
#include <string.h>
#include <stdlib.h>

//#define printd(...) fprintf(__VA_ARGS__)
#define  printd(...)

#define VDFLAG(DATA,FLAG) (DATA->flags&FLAG)
#define VD_BUFF_OWNER(BUFF) *(vine_data_s**)(BUFF-8)

vine_data_s* vine_data_init(vine_pipe_s * vpipe,void * user, size_t size)
{
	vine_data_s *data;

	data = (vine_data_s*)vine_object_register(&(vpipe->objs),
											  VINE_TYPE_DATA,
										   "",size+sizeof(vine_data_s));

	if(!data)
		return 0;

	data->vpipe = vpipe;
	data->user = user;
	data->remote = 0;
	data->size  = size;
	async_completion_init(&(data->vpipe->async),&(data->ready));
	data->buffer = data+1;
	VD_BUFF_OWNER(data->buffer) = data;

	return data;
}

vine_data_s* vine_data_init_aligned(vine_pipe_s * vpipe,void * user, size_t size,size_t align)
{
	vine_data_s *data;

	if(!align)
		return 0;

	data = (vine_data_s*)vine_object_register(&(vpipe->objs),
											  VINE_TYPE_DATA,
										   "",size+sizeof(vine_data_s)+align-1);

	if(!data)
		return 0;

	data->vpipe = vpipe;
	data->user = user;
	data->size  = size;
	async_completion_init(&(data->vpipe->async),&(data->ready));
	data->buffer = data+1;

	if( ((size_t)data->buffer) % align )
		data->buffer = data->buffer + align - ((size_t)data->buffer) % align;

	VD_BUFF_OWNER(data->buffer) = data;

	return data;
}

void vine_data_check_flags(vine_data_s * data)
{
	switch(data->flags)
	{
		case NONE_SYNC:
		case USER_SYNC:
		case SHM_SYNC:
		case USER_SYNC|SHM_SYNC:
		case REMT_SYNC:
		case REMT_SYNC|SHM_SYNC:
		case ALL_SYNC:
			return;
		default:
			fprintf(stderr,"%s(%p): Inconsistent data flags %lu\n",__func__,data,data->flags);
			abort();
	}
}

void vine_data_memcpy(vine_accel * accel,vine_data_s * dst,vine_data_s * src,int block)
{
	if(dst == src)
		return;

	if(vine_data_size(dst) != vine_data_size(src))
		fprintf(stderr,"%s(%p,%p): Size mismatch (%lu,%lu)\n",__func__,dst,src,vine_data_size(dst),vine_data_size(src));
	fprintf(stderr,"%s(%p,%p)[%lu,%lu]\n",__func__,dst,src,dst->flags,src->flags);

	vine_data_sync_from_remote(accel,src,block);

	memcpy(vine_data_deref(dst),vine_data_deref(src),vine_data_size(src));

	vine_data_modified(dst,SHM_SYNC);

	vine_data_sync_to_remote(accel,dst,block);
}

void vine_data_set_arch(vine_data_s* data,vine_accel_type_e arch)
{
	if( data->arch && (data->arch!=arch) )
	{
		abort();
	}
	data->arch = arch;
}

vine_accel_type_e vine_data_get_arch(vine_data_s* data)
{
	return data->arch;
}

void vine_data_arg_init(vine_data_s* data,vine_accel_type_e arch)
{
	vine_data_set_arch(data,arch);
	async_completion_init(&(data->vpipe->async),&(data->ready));
}

void vine_data_input_init(vine_data_s* data,vine_accel_type_e arch)
{
	vine_object_ref_inc(&(data->obj));
	vine_data_set_arch(data,arch);
	async_completion_init(&(data->vpipe->async),&(data->ready));
}

void vine_data_output_init(vine_data_s* data,vine_accel_type_e arch)
{
	vine_object_ref_inc(&(data->obj));
	vine_data_set_arch(data,arch);
	async_completion_init(&(data->vpipe->async),&(data->ready));
}

void vine_data_output_done(vine_data_s* data)
{
	// Invalidate on all levels except accelerator memory.
	vine_data_modified(data,REMT_SYNC);
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

	vdata = (vine_data_s*)data;

	return vdata->buffer;
}

vine_data * vine_data_ref(void * data)
{
  if(!data)
    return 0;

  vine_data_s *vdata = VD_BUFF_OWNER(data);

  if(!vdata)
    return 0;

  if(vdata->obj.type != VINE_TYPE_DATA)
    return 0;

  return vdata;
}

void vine_data_mark_ready(vine_pipe_s *vpipe, vine_data *data)
{
	vine_data_s *vdata;

	vdata = (vine_data_s*)data;
	async_completion_complete(&(vdata->ready));
}

int vine_data_check_ready(vine_pipe_s *vpipe, vine_data *data)
{
	vine_data_s *vdata;
	int return_val;

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);
	return_val = async_completion_check(&(vdata->ready));

	return return_val;
}

void vine_data_free(vine_data *data)
{
	vine_data_s *vdata;

	vdata = (vine_data_s*)data;
	vine_object_ref_dec(&(vdata->obj));
}

int vine_data_valid(vine_object_repo_s *repo, vine_data *data)
{
	return 0;
}

void rs_sync(vine_accel * accel, int sync_dir,const char * func,vine_data_s * data,int block)
{
	if(data->remote == vine_data_deref(data))	// Remote points to shm buffer
		return;

	void * args[2] = {data,(void*)(size_t)block};
	data->sync_dir = sync_dir;

	vine_accel_type_e type = ((vine_vaccel_s*)accel)->type;
	vine_proc_s * proc = vine_proc_get(type,func);

	if(!vine_proc_get_functor(proc))
		return;

	vine_task_msg_s * task = vine_task_issue(accel,proc,args,sizeof(void*)*2,0,0,0,0);

	if(block)
	{
		vine_task_wait(task);
		vine_task_free(task);
	}
}

/*
 * Send user data to the remote
 */
void vine_data_sync_to_remote(vine_accel * accel,vine_data * data,int block)
{
	vine_data_s *vdata;

	vdata = (vine_data_s*)data;

	vine_data_check_flags(data);	// Ensure flags are consistent

	switch(vdata->flags)
	{
		case NONE_SYNC:
			fprintf(stderr,"%s(%p) called with uninitialized buffer!\n",__func__,data);
			abort();
		case USER_SYNC:	// usr->shm
			if(vdata->user)
				memcpy(vine_data_deref(vdata),vdata->user,vdata->size);
			vdata->flags |= USER_SYNC;
		case USER_SYNC|SHM_SYNC:
		case SHM_SYNC:
			rs_sync(accel,TO_REMOTE,"syncTo",vdata,block);
			vdata->flags |= SHM_SYNC;
		case REMT_SYNC|SHM_SYNC:
		case REMT_SYNC:
			vdata->flags |= REMT_SYNC;
		case ALL_SYNC:
			break;	// All set
		default:
			fprintf(stderr,"%s(%p) unexpected flags %lu!\n",__func__,data,vdata->flags);
			abort();
			break;
	}

	vine_data_check_flags(data);	// Ensure flags are consistent
}

/*
 * Get remote data to user
 */
void vine_data_sync_from_remote(vine_accel * accel,vine_data * data,int block)
{
	vine_data_s *vdata;

	vdata = (vine_data_s*)data;

	vine_data_check_flags(data);	// Ensure flags are consistent

	switch(vdata->flags)
	{
		case NONE_SYNC:
			fprintf(stderr,"%s(%p) called with uninitialized buffer!\n",__func__,data);
			abort();
		case REMT_SYNC: // rmt->shm
			rs_sync(accel,FROM_REMOTE,"syncFrom",vdata,block);
			vdata->flags |= REMT_SYNC;
		case REMT_SYNC|SHM_SYNC:
		case SHM_SYNC:
			if(vdata->user)
				memcpy(vdata->user,vine_data_deref(vdata),vdata->size);
			vdata->flags |= SHM_SYNC;
		case USER_SYNC|SHM_SYNC:
		case USER_SYNC:	// usr->shm
			vdata->flags |= USER_SYNC;
		case ALL_SYNC:
			break;
		default:
			fprintf(stderr,"%s(%p) unexpected flags %lu!\n",__func__,data,vdata->flags);
			abort();
			break;
	}

	vine_data_check_flags(data);	// Ensure flags are consistent
}

void vine_data_modified(vine_data * data,vine_data_flags_e where)
{
	vine_data_s *vdata;

	vdata = (vine_data_s*)data;
	vdata->flags = where;
}

#undef vine_data_stat

void vine_data_stat(vine_data * data,const char * file,size_t line)
{
        vine_data_s *vdata;

        vdata = (vine_data_s*)data;

	file += strlen(file);
	while(*file != '/')
		file--;

	int scsum = 0;
	int ucsum = 0;
	int cnt;
	char * bytes = vine_data_deref(data);

	for(cnt = 0 ; cnt < vine_data_size(data) ; cnt++)
	{
		scsum += *bytes;
		bytes++;
	}

	bytes = vdata->user;

	if(bytes)
	{

		for(cnt = 0 ; cnt < vine_data_size(data) ; cnt++)
		{
			ucsum += *bytes;
			bytes++;
		}
	}

	fprintf(stderr,"%s(%p)[%lu]:Flags(%s%s%s%s) %08x %08x ?????? @%lu:%s\n",__func__,vdata,vine_data_size(vdata),
		(vdata->flags&USER_SYNC)?"U":" ",
		(vdata->flags&SHM_SYNC)?"S":" ",
		(vdata->flags&REMT_SYNC)?"R":" ",
		(vdata->flags&FREE)?"F":" ",
		ucsum,
		scsum,
		line,file
	);
}

VINE_OBJ_DTOR_DECL(vine_data_s)
{
	vine_data_s * data = (vine_data_s *)obj;

	if(data->remote)
	{
		fprintf(stderr,"vine_data(%p) dtor called, with dangling remote, leak!\n",data);
	}
	else
		arch_alloc_free(obj->repo->alloc,data);
}
