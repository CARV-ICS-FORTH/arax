#include "alloc.h"
#include <3rdparty/dlmalloc/malloc.h>

utils_alloc_s utils_alloc_init(void *shm, size_t size)
{
	utils_alloc_s ret = create_mspace_with_base(shm, size, 1);

	return ret;
}

void* utils_alloc_allocate(utils_alloc_s alloc, size_t size)
{
	return mspace_malloc(alloc, size);
}

void utils_alloc_free(utils_alloc_s alloc, void *mem)
{
	mspace_free(alloc, mem);
}

void utils_alloc_exit(utils_alloc_s alloc) {}
