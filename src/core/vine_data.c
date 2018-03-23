#include "vine_data.h"
#include "vine_pipe.h"

vine_data_s* vine_data_init(vine_pipe_s * vpipe,void * src, size_t size)
{
	vine_data_s *data;

	data = (vine_data_s*)vine_object_register(&(vpipe->objs),
											  VINE_TYPE_DATA,
										   "",size+sizeof(vine_data_s));

	if(!data)
		return 0;

	data->src = src;
	data->size  = size;
	data->flags = 0;
	async_completion_init(&(vpipe->async),&(data->ready));
	return data;
}

size_t vine_data_size(vine_data *data)
{
	vine_data_s *vdata;

	vdata = data;
	return vdata->size;
}

void * vine_data_src_ptr(vine_data *data)
{
	vine_data_s *vdata;

	vdata = data;

	return vdata->src;
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

	vdata = (vine_data_s*)data;
	return_val = async_completion_check(&(vdata->ready));

	return return_val;
}

void vine_data_free(vine_pipe_s *vpipe, vine_data *data)
{
	vine_data_s *vdata;

	vdata = (vine_data_s*)data;
	vine_object_ref_dec(&(vdata->obj));
}

int vine_data_valid(vine_object_repo_s *repo, vine_data_s *data)
{
	return 0;
}

VINE_OBJ_DTOR_DECL(vine_data_s)
{
	vine_data_s * data = (vine_data_s *)obj;
	arch_alloc_free(obj->repo->alloc,data);
}
