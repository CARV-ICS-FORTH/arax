#ifndef UTILS_ALLOCATOR_HEADER
#define UTILS_ALLOCATOR_HEADER
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/*
 * TODO:Add canarry
 */
typedef void *utils_alloc_s;

/**
 * Initialize a utils_alloc_s instance on a mapped shared memory segment.
 *
 * @param shm The mapped shared memory segment.
 * @param size The size of the shared memory segment in bytes.
 * @return An initialized utils_alloc_s instance, or NULL on failure.
 */
utils_alloc_s utils_alloc_init(void *shm, size_t size);

/**
 * Allocate contigous memory from the alloc utils_alloc_s instance.
 *
 * @param alloc An initialized utils_alloc_s instance.
 * @param size The size of the allocation.
 * @return Pointer to the begining of size usable bytes,
 * or NULL on failure.
 */
void* utils_alloc_allocate(utils_alloc_s alloc, size_t size);

/**
 * Free previously allocated memory from a utils_alloc_s instance.
 *
 * @param alloc An initialized utils_alloc_s instance.
 * @param mem A pointer returned from utils_alloc_allocate.
 */
void utils_alloc_free(utils_alloc_s alloc, void *mem);

/**
 * Release any resources claimed by the alloc utils_alloc_s instance.
 * @note The shared memory segment (shm in utils_alloc_init) must be freed
 * by the user.
 */
void utils_alloc_exit(utils_alloc_s alloc);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef UTILS_ALLOCATOR_HEADER */
