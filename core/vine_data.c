#include "vine_data.h"

vine_data_s* vine_data_init(void *base, void *mem, size_t size,
                            vine_data_alloc_place_e place)
{
	vine_data_s *data;

	data        = (vine_data_s*)mem;
	data->place = place;
	data->size  = size;
	data->ready = 0;
	return data;
}
