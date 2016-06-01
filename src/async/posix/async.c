#include "async.h"

void arch_async_completion_init(arch_async_completion_s * compl)
{
	pthread_mutexattr_init(&(compl->attr));
	pthread_mutexattr_setpshared(&(compl->attr), PTHREAD_PROCESS_SHARED);
	pthread_mutex_init(&(compl->mutex),&(compl->attr));
	pthread_mutex_lock(&(compl->mutex));
}

void arch_async_completion_complete(arch_async_completion_s * compl)
{
	pthread_mutex_unlock(&(compl->mutex));
}

void arch_async_completion_wait(arch_async_completion_s * compl)
{
	pthread_mutex_lock(&(compl->mutex));

}
