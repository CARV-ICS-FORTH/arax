#include "vine_data.h"
#include "vine_pipe.h"
#include <string.h>
#include <stdlib.h>

//#define printd(...) fprintf(__VA_ARGS__)
#define  printd(...)

#define VDFLAG(DATA,FLAG) (DATA->flags&FLAG)

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
	data->size  = size;

	async_completion_init(&(data->vpipe->async),&(data->ready));

	return data;
}

void vine_data_memcpy(vine_data_s * dst,vine_data_s * src)
{
	if(vine_data_size(dst) != vine_data_size(src))
		fprintf(stderr,"%s(%p,%p): Size mismatch (%lu,%lu)\n",__func__,dst,src,vine_data_size(dst),vine_data_size(src));
	fprintf(stderr,"%s(%p,%p)[%lu,%lu]\n",__func__,dst,src,dst->flags,src->flags);
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
	data->flags = REMT_OWNED;
	vine_data_mark_ready(data->vpipe ,data);
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

	return (void*)(vdata+1);
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

/*
 * Send user data to the remote
 */
void vine_data_sync_to_remote(vine_data * data,vine_data_flags_e upto)
{
	vine_data_s *vdata;

	vdata = (vine_data_s*)data;

	if( vdata->flags & REMT_OWNED )
	{
		printd(stderr,"%s(%p):REMT_OWNED %lu\n",__func__,data,vdata->flags);
		return;
	}

	if(!(vdata->flags & USER_IN_SYNC) && ( upto & USER_IN_SYNC) )
	{
		memcpy(vine_data_deref(vdata),vdata->user,vdata->size);
		vdata->flags = USER_IN_SYNC;	// Only user data is in sync now
		printd(stderr,"%s(%p):USER_IN_SYNC %lu\n",__func__,data,vdata->flags);
	}
	if(!(vdata->flags & REMT_IN_SYNC) && ( upto & REMT_IN_SYNC) )
	{
		async_completion_init(&(vdata->vpipe->async),&(vdata->ready));
		vdata->sync_dir = TO_REMOTE;
		async_condition_lock(&(vdata->vpipe->sync_cond));
		utils_queue_push(vdata->vpipe->sync_queue,data);
		async_condition_notify(&(vdata->vpipe->sync_cond));
		async_condition_unlock(&(vdata->vpipe->sync_cond));

		async_completion_wait(&(vdata->ready));

		printd(stderr,"%s(%p):REMT_IN_SYNC %lu\n",__func__,data,vdata->flags);
	}
}

/*
 * Get remote data to user
 */
void vine_data_sync_from_remote(vine_data * data,vine_data_flags_e upto)
{
	vine_data_s *vdata;

	vdata = (vine_data_s*)data;

	if( !(vdata->flags & REMT_OWNED) )
	{
		fprintf(stderr,"%s(%p):REMT_OWNED not set! %lu\n",__func__,data,vdata->flags);
	}

	if(!(vdata->flags & REMT_IN_SYNC) && ( upto & REMT_IN_SYNC) )
	{
		async_completion_init(&(vdata->vpipe->async),&(vdata->ready));
		vdata->sync_dir = FROM_REMOTE;
		async_condition_lock(&(vdata->vpipe->sync_cond));
		utils_queue_push(vdata->vpipe->sync_queue,data);
		async_condition_notify(&(vdata->vpipe->sync_cond));
		async_condition_unlock(&(vdata->vpipe->sync_cond));

		async_completion_wait(&(vdata->ready));

		printd(stderr,"%s(%p):REMT_IN_SYNC %lu\n",__func__,data,vdata->flags);
	}
	if(!(vdata->flags & USER_IN_SYNC) && ( upto & USER_IN_SYNC) )
	{
		memcpy(vdata->user,vine_data_deref(vdata),vdata->size);
		vdata->flags |= USER_IN_SYNC;
		printd(stderr,"%s(%p):USER_IN_SYNC %lu\n",__func__,data,vdata->flags);
	}
}

void vine_data_modified(vine_data * data,vine_data_flags_e where)
{
	vine_data_s *vdata;

	vdata = (vine_data_s*)data;
	vdata->flags &= ~(ALL_IN_SYNC);	// Invalidata all other locations
	vdata->flags |= where;
}

VINE_OBJ_DTOR_DECL(vine_data_s)
{
	vine_data_s * data = (vine_data_s *)obj;

	if(data->remote)
	{
		async_completion_init(&(data->vpipe->async),&(data->ready));
		data->sync_dir = 0;
		data->flags = FREE;
		async_condition_lock(&(data->vpipe->sync_cond));
		utils_queue_push(data->vpipe->sync_queue,data);
		async_condition_notify(&(data->vpipe->sync_cond));
		async_condition_unlock(&(data->vpipe->sync_cond));

		async_completion_wait(&(data->ready));
	}
	arch_alloc_free(obj->repo->alloc,data);
}
