#include "async.h"

void async_completion_init(async_completion_s * completion)
{
	completion->counter = 0;
}

void async_completion_complete(async_completion_s * completion)
{
	__sync_bool_compare_and_swap(&(completion->counter), 0, 1);
}

void async_completion_wait(async_completion_s * completion)
{
	while (!completion->counter);
}
