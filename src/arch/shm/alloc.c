#include "arch/alloc.h"
#include <string.h>
#define ONLY_MSPACES   1
#define USE_SPIN_LOCKS 1
#define MSPACES        1
#include "3rdparty/dlmalloc/malloc.h"

int arch_alloc_init(arch_alloc_s * alloc,void *shm, size_t size)
{
	memset(alloc,0,sizeof(arch_alloc_s));
	alloc->state = create_mspace_with_base(shm, size, 1);

	return 0;
}

void* arch_alloc_allocate(arch_alloc_s * alloc, size_t size)
{
	return mspace_malloc(alloc->state, size);
}

void arch_alloc_free(arch_alloc_s * alloc, void *mem)
{
	mspace_free(alloc->state, mem);
}

void arch_alloc_exit(arch_alloc_s * alloc)
{
	destroy_mspace(alloc->state);
}

arch_alloc_stats_s arch_alloc_stats(arch_alloc_s * alloc)
{
	arch_alloc_stats_s stats;
	return stats;
}
