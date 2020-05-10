#include "arch/alloc.h"
#include <stdlib.h>

arch_alloc_s arch_alloc_init(void *shm, size_t size)
{
	return NULL;
}

void* arch_alloc_allocate(arch_alloc_s alloc, size_t size)
{
	return malloc(size);
}

void _arch_alloc_free(arch_alloc_s alloc, void *mem)
{
	free(mem);
}

void arch_alloc_exit(arch_alloc_s alloc) {}
