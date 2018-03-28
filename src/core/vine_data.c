#include "vine_data.h"
#include "vine_pipe.h"
#include <string.h>

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
	data->flags = 0;

	return data;
}

void vine_data_input_init(vine_data_s* data)
{
	vine_object_ref_inc(&(data->obj));
}

void vine_data_output_init(vine_data_s* data)
{
	vine_object_ref_inc(&(data->obj));
	async_completion_init(&(data->vpipe->async),&(data->ready));
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
	if(!(vdata->flags & USER_IN_SYNC) && ( upto & USER_IN_SYNC) )
	{
		memcpy(vine_data_deref(vdata),vdata->user,vdata->size);
	}
	if(!(vdata->flags & REMT_IN_SYNC) && ( upto & REMT_IN_SYNC) )
	{

	}
	vdata->flags |= USER_IN_SYNC | REMT_IN_SYNC;
}

/*
 * Get remote data to user
 */
void vine_data_sync_from_remote(vine_data * data,vine_data_flags_e upto)
{
	vine_data_s *vdata;

	vdata = (vine_data_s*)data;
	if(!(vdata->flags & REMT_IN_SYNC) && ( upto & REMT_IN_SYNC) )
	{

	}
	if(!(vdata->flags & USER_IN_SYNC) && ( upto & USER_IN_SYNC) )
	{
		memcpy(vdata->user,vine_data_deref(vdata),vdata->size);
	}
	vdata->flags |= USER_IN_SYNC | REMT_IN_SYNC;
}

void vine_data_modified(vine_data * data)
{
}

VINE_OBJ_DTOR_DECL(vine_data_s)
{
	vine_data_s * data = (vine_data_s *)obj;
	arch_alloc_free(obj->repo->alloc,data);
}
