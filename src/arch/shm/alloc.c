#include "arch/alloc.h"
#include "utils/timer.h"
#include <string.h>
#define MALLOC_INSPECT_ALL 1
#include <malloc.h>
#define ONLY_MSPACES   1
#define USE_SPIN_LOCKS 1
#define MSPACES        1
#define HAVE_USR_INCLUDE_MALLOC_H
#include "3rdparty/dlmalloc/malloc.h"

#define MB (1024ul*1024ul)

#define PART_DATA_SIZE (4096*MB)
typedef struct {mspace mspace;char data[PART_DATA_SIZE-sizeof(mspace)];} PARTITION;


int arch_alloc_init(arch_alloc_s * alloc, size_t size)
{
	PARTITION * part = (PARTITION*)(alloc+1);
	size_t part_size;
	size_t prev_size = -1;

	memset(alloc,0,sizeof(arch_alloc_s));

	size -= sizeof(arch_alloc_s);

	while(size < prev_size)
	{
		part_size = (size > sizeof(part->data))?sizeof(part->data):size;
		part->mspace = create_mspace_with_base(part->data, part_size , 1);
		alloc->mspaces++;
		prev_size = size;
		size -= part_size+sizeof(mspace);
		part++;
	}

	return 0;
}

void* arch_alloc_allocate(arch_alloc_s * alloc, size_t size)
{
	void * data;
	PARTITION * part;
	int pool;
	#ifdef ALLOC_STATS
	utils_timer_s dt;
	utils_timer_set(dt,start);
	#endif

	part = (PARTITION*)(alloc+1);

	for(pool = 0 ; pool < alloc->mspaces ; pool++,part++)
	{
		data = mspace_malloc(part->mspace, size);
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
	PARTITION * part;
	#ifdef ALLOC_STATS
	utils_timer_s dt;
	utils_timer_set(dt,start);
	#endif

	part = (PARTITION*)(alloc+1);

	mspace = ((size_t)mem-(size_t)part)/(512*MB);

	part += mspace;

	mspace_free(part->mspace, mem);

	#ifdef ALLOC_STATS
	utils_timer_set(dt,stop);
	__sync_fetch_and_add(&(alloc->free_ns),
						 utils_timer_get_duration_ns(dt));
	__sync_fetch_and_add(&(alloc->frees),1);
	#endif
}

void arch_alloc_exit(arch_alloc_s * alloc)
{
	PARTITION * part = (PARTITION*)(alloc+1);
	while(alloc->mspaces--)
	{
		destroy_mspace(part->mspace);
		part++;
	}
}

static void _arch_alloc_mspace_mallinfo(mspace * mspace,arch_alloc_stats_s * stats)
{
	struct mallinfo minfo = mspace_mallinfo(mspace);
	stats->total_bytes += minfo.arena;
	stats->used_bytes += minfo.uordblks;
}

arch_alloc_stats_s arch_alloc_stats(arch_alloc_s * alloc)
{
	PARTITION * part = (PARTITION*)(alloc+1);
	arch_alloc_stats_s stats = {0};
	int mspace = 0;

	for(mspace = 0 ; mspace < alloc->mspaces ; mspace++)
	{
		_arch_alloc_mspace_mallinfo(part->mspace,&stats);
		part++;
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
	PARTITION * part = (PARTITION*)(alloc+1);
	arch_alloc_stats_s stats = {0};

	if(mspace < alloc->mspaces)
	{
		stats.mspaces = mspace+1;
		_arch_alloc_mspace_mallinfo(part[mspace].mspace,&stats);
	}

	return stats;
}

void arch_alloc_inspect(arch_alloc_s * alloc,void (*inspector)(void * start,void * end, size_t size, void * arg),void * arg)
{
	PARTITION * part = (PARTITION*)(alloc+1);
	int mspace;
	for(mspace = 0 ; mspace < alloc->mspaces ; mspace++)
	{
		mspace_inspect_all(part->mspace,inspector,arg);
		part++;
	}
}
