#include "arch/alloc.h"
#include <string.h>
#include <malloc.h>
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
	void * data = mspace_malloc(alloc->state, size);
	#ifdef ALLOC_STATS
	__sync_fetch_and_add(&(alloc->allocs[!!data]),1);
	#endif
	return data;
}

void arch_alloc_free(arch_alloc_s * alloc, void *mem)
{
	#ifdef ALLOC_STATS
	__sync_fetch_and_add(&(alloc->frees),1);
	#endif
	mspace_free(alloc->state, mem);
}

void arch_alloc_exit(arch_alloc_s * alloc)
{
	destroy_mspace(alloc->state);
}

arch_alloc_stats_s arch_alloc_stats(arch_alloc_s * alloc)
{
	arch_alloc_stats_s stats;
	struct mallinfo minfo;

	minfo  = mspace_mallinfo(alloc->state);
	stats.total_bytes = minfo.arena;
	stats.used_bytes = minfo.uordblks;
#ifdef ALLOC_STATS
	stats.allocs[0] = alloc->allocs[0];
	stats.allocs[1] = alloc->allocs[1];
	stats.frees = alloc->frees;
#endif
	return stats;
}
