#include "arch/alloc.h"
#include "utils/timer.h"
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
	void * data;
	#ifdef ALLOC_STATS
	utils_timer_s dt;
	utils_timer_set(dt,start);
	#endif

	data = mspace_malloc(alloc->state, size);

	#ifdef ALLOC_STATS
	utils_timer_set(dt,stop);
	__sync_fetch_and_add(&(alloc->alloc_ns[!!data]),
						 utils_timer_get_duration_ns(dt));
	__sync_fetch_and_add(&(alloc->allocs[!!data]),1);
	#endif
	return data;
}

void arch_alloc_free(arch_alloc_s * alloc, void *mem)
{
	#ifdef ALLOC_STATS
	utils_timer_s dt;
	utils_timer_set(dt,start);
	#endif

	mspace_free(alloc->state, mem);

	#ifdef ALLOC_STATS
	utils_timer_set(dt,stop);
	__sync_fetch_and_add(&(alloc->free_ns),
						 utils_timer_get_duration_ns(dt));
	__sync_fetch_and_add(&(alloc->frees),1);
	#endif
}

void arch_alloc_exit(arch_alloc_s * alloc)
{
	destroy_mspace(alloc->state);
}

arch_alloc_stats_s arch_alloc_stats(arch_alloc_s * alloc)
{
	arch_alloc_stats_s stats;
	struct mallinfo minfo = {0};

	minfo  = mspace_mallinfo(alloc->state);
	stats.total_bytes = minfo.arena;
	stats.used_bytes = minfo.uordblks;
#ifdef ALLOC_STATS
	stats.allocs[0] = alloc->allocs[0];
	stats.allocs[1] = alloc->allocs[1];
	stats.frees = alloc->frees;
	stats.alloc_ns[0] = alloc->alloc_ns[0];
	stats.alloc_ns[1] = alloc->alloc_ns[1];
	stats.free_ns = alloc->free_ns;
#endif
	return stats;
}
