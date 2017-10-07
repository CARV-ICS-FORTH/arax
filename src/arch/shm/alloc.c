#include "arch/alloc.h"
#include "utils/timer.h"
#include <string.h>
#include <malloc.h>
#define ONLY_MSPACES   1
#define USE_SPIN_LOCKS 1
#define MSPACES        1
#define HAVE_USR_INCLUDE_MALLOC_H
#include "3rdparty/dlmalloc/malloc.h"

#define MB (1024ul*1024ul)
int arch_alloc_init(arch_alloc_s * alloc,void *shm, size_t size)
{
	memset(alloc,0,sizeof(arch_alloc_s));

	if(size > ARCH_ALLOC_MAX_SPACE*1024*MB)
	{
		fprintf(stderr,"%s(): Allocator size exceeds ARCH_ALLOC_MAX_SPACE!",__func__);
		return -1;
	}

	alloc->base = shm;

	while(size > 512*MB)
	{
		alloc->states[alloc->mspaces++] = create_mspace_with_base(shm, 512*MB, 1);
		size -= 512*MB;
		shm += 512*MB;
	}

	if(size)
		alloc->states[alloc->mspaces++] = create_mspace_with_base(shm, size, 1);

	return 0;
}

void* arch_alloc_allocate(arch_alloc_s * alloc, size_t size)
{
	void * data;
	int pool;
	#ifdef ALLOC_STATS
	utils_timer_s dt;
	utils_timer_set(dt,start);
	#endif

	for(pool = 0 ; pool < alloc->mspaces ; pool++)
	{
		data = mspace_malloc(alloc->states[pool], size);
		if(data)
			break;
	}

	if(pool == alloc->mspaces)
	{
		fprintf(stderr,"%s(): Could not allocate %lu, avaiable space exceeded%lu!\n",__func__,size,alloc->mspaces);
	}
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
	int mspace;
	#ifdef ALLOC_STATS
	utils_timer_s dt;
	utils_timer_set(dt,start);
	#endif

	mspace = ((size_t)mem-(size_t)alloc->base)/(512*MB);

	mspace_free(alloc->states[mspace], mem);

	#ifdef ALLOC_STATS
	utils_timer_set(dt,stop);
	__sync_fetch_and_add(&(alloc->free_ns),
						 utils_timer_get_duration_ns(dt));
	__sync_fetch_and_add(&(alloc->frees),1);
	#endif
}

void arch_alloc_exit(arch_alloc_s * alloc)
{
	while(alloc->mspaces--)
		destroy_mspace(alloc->states[alloc->mspaces]);
}

arch_alloc_stats_s arch_alloc_stats(arch_alloc_s * alloc)
{
	arch_alloc_stats_s stats = {0};
	struct mallinfo minfo = {0};
	int mspace = 0;

	for(mspace = 0 ; mspace < alloc->mspaces ; mspace++)
	{
		minfo  = mspace_mallinfo(alloc->states[mspace]);
		stats.total_bytes += minfo.arena;
		stats.used_bytes += minfo.uordblks;
	}

	stats.mspaces = alloc->mspaces;
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

arch_alloc_stats_s arch_alloc_mspace_stats(arch_alloc_s * alloc,size_t mspace)
{
	arch_alloc_stats_s stats = {0};
	struct mallinfo minfo = {0};

	if(mspace < alloc->mspaces)
	{
		stats.mspaces = mspace+1;
		minfo  = mspace_mallinfo(alloc->states[mspace]);
		stats.total_bytes += minfo.arena;
		stats.used_bytes += minfo.uordblks;
	}

	return stats;
}
