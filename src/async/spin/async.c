#include "async.h"

void async_meta_init(async_meta_s * meta)
{}

void async_completion_init(async_meta_s * meta,async_completion_s * completion)
{
	completion->completed = 0;
}

void async_completion_complete(async_meta_s * meta,async_completion_s * completion)
{
	__sync_bool_compare_and_swap(&(completion->completed), 0, 1);
}

void async_completion_wait(async_meta_s * meta,async_completion_s * completion)
{
	while (!__sync_bool_compare_and_swap(&(completion->completed), 1, 0));
}

int async_completion_check(async_meta_s * meta,async_completion_s * completion)
{
	return completion->completed;
}

void async_meta_exit(async_meta_s * meta)
{}
