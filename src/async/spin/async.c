#include "async.h"

void arch_async_completion_init(arch_async_completion_s * compl)
{
	compl->counter = 0;
}

void arch_async_completion_complete(arch_async_completion_s * compl)
{
	__sync_bool_compare_and_swap(&(compl->counter), 0, 1);
}

void arch_async_completion_wait(arch_async_completion_s * compl)
{
	while (!compl->counter);
}
