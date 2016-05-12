#ifndef VINE_VACCEL_HEADER
#define VINE_VACCEL_HEADER
#include <vine_talk.h>
#include "utils/queue.h"
#include "core/vine_object.h"
#include "core/vine_accel.h"

typedef struct {
	vine_object_s obj;
	utils_spinlock lock;
	vine_accel_s * phys;
} vine_vaccel_s;

vine_vaccel_s* vine_vaccel_init(vine_object_repo_s * repo,void *mem,size_t mem_size, char *name, vine_accel_s * accel);

utils_queue_s * vine_vaccel_queue(vine_vaccel_s* vaccel);

void vine_vaccel_erase(vine_object_repo_s * repo,vine_vaccel_s * accel);
#endif
