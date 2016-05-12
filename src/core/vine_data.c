#include "vine_data.h"

vine_data_s* vine_data_init(vine_object_repo_s * repo,void *mem, size_t size,
                            vine_data_alloc_place_e place)
{
	vine_data_s *data;

	data        = (vine_data_s*)mem;
	vine_object_register(repo,&(data->obj),VINE_TYPE_DATA,"");/* Use it for leak detection */
	data->place = place;
	data->size  = size;
	data->flags = 0;
	data->ready = 0;
	return data;
}

void vine_data_erase(vine_object_repo_s * repo,vine_data_s* data)
{
	vine_object_remove(repo,&(data->obj));
}

int vine_data_valid(vine_object_repo_s * repo,vine_data_s* data)
{
	return 0;
}
