#ifndef VINE_ALLOCATOR_HEADER
#define VINE_ALLOCATOR_HEADER
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*
 * TODO:Add canarry
 */
typedef void * vine_alloc_s;
/**
 * Initialize a vine_alloc_s instance on a mapped shared memory segment.
 *
 * @param shm The mapped shared memory segment.
 * @param size The size of the shared memory segment in bytes.
 * @return An initialized vine_alloc_s instance, or NULL on failure.
 */
vine_alloc_s vine_alloc_init(void * shm,size_t size);
/**
 * Allocate contigous memory from the alloc vine_alloc_s instance.
 *
 * @param alloc An initialized vine_alloc_s instance.
 * @param size The size of the allocation.
 * @return Pointer to the begining of size usable bytes,
 * or NULL on failure.
 */
void * vine_alloc_alloc(vine_alloc_s alloc,size_t size);
/**
 * Free previously allocated memory from a vine_alloc_s instance.
 *
 * @param alloc An initialized vine_alloc_s instance.
 * @param mem A pointer returned from vine_alloc_alloc.
 */
void vine_alloc_free(vine_alloc_s alloc,void * mem);
/**
 * Release any resources claimed by the alloc vine_alloc_s instance.
 * @note The shared memory segment (shm in vine_alloc_init) must be freed
 * by the user.
 */
void vine_alloc_exit(vine_alloc_s alloc);

#ifdef __cplusplus
}
#endif

#endif
