#include "arch/alloc.h"
#define ONLY_MSPACES   1
#define USE_SPIN_LOCKS 1
#define MSPACES        1
#include "3rdparty/dlmalloc/malloc.h"

arch_alloc_s arch_alloc_init(void *shm, size_t size)
{
	arch_alloc_s ret = create_mspace_with_base(shm, size, 1);

	return ret;
}

void* arch_alloc_allocate(arch_alloc_s alloc, size_t size)
{
	return mspace_malloc(alloc, size);
}

void arch_alloc_free(arch_alloc_s alloc, void *mem)
{
	mspace_free(alloc, mem);
}

void arch_alloc_exit(arch_alloc_s alloc) {}
