#ifndef VINE_DATA_HEADER
#define VINE_DATA_HEADER
#include <vine_talk.h>
typedef struct vine_data_s {
	vine_data_alloc_place_e place;
	size_t                  size;
	size_t                  ready;

	/* Add status variables */
} vine_data_s;

vine_data_s* vine_data_init(void *mem, size_t size,
                            vine_data_alloc_place_e place);

#endif /* ifndef VINE_DATA_HEADER */
