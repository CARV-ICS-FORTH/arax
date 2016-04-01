#include "alloc.h"
#include "../dlmalloc/malloc.h"

vine_alloc_s vine_alloc_init(void * shm,size_t size)
{
	vine_alloc_s ret = create_mspace_with_base(shm,size,1);
	return ret;
}

void * vine_alloc_alloc(vine_alloc_s alloc,size_t size)
{
	return mspace_malloc(alloc,size);
}

void vine_alloc_free(vine_alloc_s alloc,void * mem)
{
	mspace_free(alloc,mem);
}

void vine_alloc_exit(vine_alloc_s alloc)
{
}
