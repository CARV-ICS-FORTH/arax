#ifndef VINE_DATA_HEADER
#define VINE_DATA_HEADER
#include <vine_talk.h>
typedef struct vine_data_s {
	vine_data_alloc_place_e place;
	size_t                  size;
	size_t                  flags;
	volatile size_t         ready;

	/* Add status variables */
} vine_data_s;

typedef enum vine_data_io {
	VINE_INPUT  = 1,
	VINE_OUTPUT = 2
}vine_data_io_e;

vine_data_s* vine_data_init(void *mem, size_t size,
                            vine_data_alloc_place_e place);

#endif /* ifndef VINE_DATA_HEADER */
