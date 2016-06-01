#include "async.h"

void arch_async_completion_init(arch_async_completion_s * completion)
{
	completion->counter = 0;
}

void arch_async_completion_complete(arch_async_completion_s * completion)
{
	__sync_bool_compare_and_swap(&(completion->counter), 0, 1);
}

void arch_async_completion_wait(arch_async_completion_s * completion)
{
	while (!completion->counter);
}
