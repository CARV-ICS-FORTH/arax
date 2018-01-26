#ifndef ASYNC_HEADER
#define ASYNC_HEADER
#include "core/vine_object.h"
#include "utils/list.h"
#include "utils/spinlock.h"
#include "pthread.h"
#include "arch/alloc.h"

typedef struct ivshmem ivshmem_s;

typedef struct async_meta_s
{
	utils_spinlock lock;
	utils_list_s outstanding;
	ivshmem_s * regs;
	arch_alloc_s * alloc;
	pthread_t thread;
	volatile int fd;
}async_meta_s;

typedef struct async_completion_s
{
	async_meta_s * meta;
	utils_list_node_s outstanding;
	size_t vm_id;
	volatile size_t completed;
	pthread_mutex_t mutex;
	pthread_mutexattr_t attr;
}async_completion_s;

typedef struct
{
	async_meta_s * meta;
	utils_list_s pending_list;
	utils_spinlock pending_lock;
	volatile size_t value;
}
async_semaphore_s;

typedef struct
{
	async_meta_s * meta;
	async_completion_s mutex;
	async_semaphore_s semaphore;
}async_condition_s;

#include "async_api.h"
#endif
