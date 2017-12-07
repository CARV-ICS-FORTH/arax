#include "vine_data.h"
#include "vine_pipe.h"

vine_data_s* vine_data_init(vine_pipe_s * vpipe, size_t size,
                            vine_data_alloc_place_e place)
{
	vine_data_s *data;

	/* Not valid place */
	if(!place || place>>2)
		return 0;


	data = (vine_data_s*)vine_object_register(&(vpipe->objs),
											  VINE_TYPE_DATA,
										   "",size+sizeof(vine_data_s));

	if(!data)
		return 0;

	data->place = place;
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

void* vine_data_deref(vine_data *data)
{
	vine_data_s *vdata;

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);

	if (!(vdata->place&HostOnly)) {
		return 0;
	}

	return (void*)(vdata+1);
}

void vine_data_mark_ready(vine_pipe_s *vpipe, vine_data *data)
{
	vine_data_s *vdata;

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);
	async_completion_complete(&(vpipe->async),&(vdata->ready));
}

int vine_data_check_ready(vine_pipe_s *vpipe, vine_data *data)
{
	vine_data_s *vdata;
	int return_val;

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);
	return_val = async_completion_check(&(vpipe->async),&(vdata->ready));

	return return_val;
}

void vine_data_free(vine_pipe_s *vpipe, vine_data *data)
{
	vine_data_s *vdata;

	vdata = offset_to_pointer(vine_data_s*, vpipe, data);
	vine_object_ref_dec(&(vdata->obj));
}
/*
void vine_data_erase(vine_object_repo_s *repo, vine_data_s *data)
{
	vine_object_remove( repo, &(data->obj) );
}
*/
int vine_data_valid(vine_object_repo_s *repo, vine_data_s *data)
{
	return 0;
}

VINE_OBJ_DTOR_DECL(vine_data_s)
{
}
