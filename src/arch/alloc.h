#ifndef ARCH_ALLOCATOR_HEADER
#define ARCH_ALLOCATOR_HEADER
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

typedef void *arch_alloc_state;

/*
 * TODO:Add canarry
 */
typedef struct
{
	arch_alloc_state * state;
#ifdef ALLOC_STATS
	size_t allocs[2];	//< Number of arch_alloc_allocate(failed/successfull).
	size_t frees;		//< Number of arch_alloc_free.
#endif
}arch_alloc_s;

/**
 * Initialize a arch_alloc_s instance on a mapped shared memory segment.
 *
 * @param alloc Pointer to be filled with initialized instance.
 * @param shm The mapped shared memory segment.
 * @param size The size of the shared memory segment in bytes.
 * @return An initialized arch_alloc_s instance, or NULL on failure.
 */
int arch_alloc_init(arch_alloc_s * alloc,void *shm, size_t size);

/**
 * Allocate contiguous memory from the alloc arch_alloc_s instance.
 *
 * @param alloc An initialized arch_alloc_s instance.
 * @param size The size of the allocation.
 * @return Pointer to the beginning of size usable bytes,
 * or NULL on failure.
 */
void* arch_alloc_allocate(arch_alloc_s * alloc, size_t size);

/**
 * Free previously allocated memory from a arch_alloc_s instance.
 *
 * @param alloc An initialized arch_alloc_s instance.
 * @param mem A pointer returned from arch_alloc_allocate.
 */
void arch_alloc_free(arch_alloc_s * alloc, void *mem);

/**
 * Release any resources claimed by the alloc arch_alloc_s instance.
 * @note The shared memory segment (shm in arch_alloc_init) must be freed
 * by the user.
 */
void arch_alloc_exit(arch_alloc_s * alloc);

typedef struct
{
	size_t total_bytes; //<Bytes available for user data AND allocator metadata.
	size_t used_bytes;  //<Bytes used for user data AND allocator metadata.
#ifdef ALLOC_STATS
	size_t allocs[2];	//< Number of arch_alloc_allocate(failed/successfull).
	size_t frees;		//< Number of arch_alloc_free.
#endif
}arch_alloc_stats_s;

arch_alloc_stats_s arch_alloc_stats(arch_alloc_s * alloc);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef ARCH_ALLOCATOR_HEADER */
