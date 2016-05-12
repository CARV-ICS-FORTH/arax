#ifndef VINE_DATA_HEADER
#define VINE_DATA_HEADER
#include <vine_talk.h>
#include "core/vine_object.h"
typedef struct vine_data_s {
	vine_object_s obj; /* Might make this optional (for perf reasons) */
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

vine_data_s* vine_data_init(vine_object_repo_s * repo,void *mem, size_t size,
                            vine_data_alloc_place_e place);

void vine_data_erase(vine_object_repo_s * repo,vine_data_s* data);

int vine_data_valid(vine_object_repo_s * repo,vine_data_s* data);

#endif /* ifndef VINE_DATA_HEADER */
