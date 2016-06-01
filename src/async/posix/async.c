#include "async.h"

void arch_async_completion_init(arch_async_completion_s * compl)
{
	pthread_mutex_init(&(compl->mutex),0);
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
