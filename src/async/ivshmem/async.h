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
	pthread_t thread;
	volatile int fd;
}async_meta_s;

typedef struct
{
	utils_list_node_s outstanding;
	size_t vm_id;
	volatile size_t completed;
	pthread_mutex_t mutex;
	pthread_mutexattr_t attr;
}
async_completion_s;

#include "async_api.h"
#endif
