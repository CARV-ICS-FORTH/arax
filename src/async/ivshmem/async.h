#ifndef ASYNC_HEADER
#define ASYNC_HEADER
#include "core/vine_object.h"
#include "utils/list.h"
#include "utils/spinlock.h"
#include "pthread.h"

typedef struct ivshmem ivshmem_s;

typedef struct
{
	utils_spinlock lock;
	utils_list_s outstanding;
	ivshmem_s * regs;
	int fd;
}async_meta_s;

typedef struct
{
	utils_list_node_s outstanding;
	volatile size_t counter;
	pthread_mutex_t mutex;
	pthread_mutexattr_t attr;
}
async_completion_s;

#include "async_api.h"
#endif
