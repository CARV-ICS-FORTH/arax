#include "async.h"

void arch_async_completion_init(arch_async_completion_s * completion)
{
	pthread_mutexattr_init(&(completion->attr));
	pthread_mutexattr_setpshared(&(completion->attr), PTHREAD_PROCESS_SHARED);
	pthread_mutex_init(&(completion->mutex),&(completion->attr));
	pthread_mutex_lock(&(completion->mutex));
}

void arch_async_completion_complete(arch_async_completion_s * completion)
{
	pthread_mutex_unlock(&(completion->mutex));
}

void arch_async_completion_wait(arch_async_completion_s * completion)
{
	pthread_mutex_lock(&(completion->mutex));

}
