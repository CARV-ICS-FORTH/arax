#ifndef ARCH_ASYNC_HEADER
#define ARCH_ASYNC_HEADER
#include "core/vine_object.h"
#include "pthread.h"
typedef struct
{
	pthread_mutex_t mutex;
	pthread_mutexattr_t attr;
}
arch_async_completion_s;

#include "../async_api.h"
#endif
