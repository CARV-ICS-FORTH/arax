#include "async.h"

void async_meta_init_once(async_meta_s * meta)
{}

void async_meta_init_always(async_meta_s * meta)
{}

void async_completion_init(async_meta_s * meta,async_completion_s * completion)
{
	pthread_mutexattr_init(&(completion->attr));
	pthread_mutexattr_setpshared(&(completion->attr), PTHREAD_PROCESS_SHARED);
	pthread_mutex_init(&(completion->mutex),&(completion->attr));
	pthread_mutex_lock(&(completion->mutex));
}

void async_completion_complete(async_meta_s * meta,async_completion_s * completion)
{
	pthread_mutex_trylock(&(completion->mutex)); // Hack to avoid double unlocking
	pthread_mutex_unlock(&(completion->mutex));
}

void async_completion_wait(async_meta_s * meta,async_completion_s * completion)
{
	pthread_mutex_lock(&(completion->mutex));
}

int async_completion_check(async_meta_s * meta,async_completion_s * completion)
{
	int ret_val;
	if( !(ret_val = pthread_mutex_trylock(&(completion->mutex))) )
		pthread_mutex_unlock(&(completion->mutex));
	return !ret_val;
}

void async_meta_exit(async_meta_s * meta)
{}
