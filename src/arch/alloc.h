#ifndef ARCH_ALLOCATOR_HEADER
#define ARCH_ALLOCATOR_HEADER
#include <stddef.h>
#include "conf.h"
#include "vine_talk_types.h"
#include <utils/bitmap.h>

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

typedef void *arch_alloc_state;

/*
 * TODO:Add canarry
 */
struct arch_alloc_s
{
    #ifdef ALLOC_STATS
    size_t allocs[2];   // < Number of arch_alloc_allocate(failed/successfull).
    size_t frees;       // < Number of arch_alloc_free.
    size_t alloc_ns[2]; // < Cumulative nanoseconds spend in alloc(failed/successful).
    size_t free_ns;     // < Cumulative nanoseconds spend in free.
    #endif
};

/**
 * Initialize a arch_alloc_s instance on a mapped shared memory segment.
 *
 * \note This only has to be called by the first process, to initialize
 * global state
 *
 * @param alloc Pointer to be filled with initialized instance.
 * @param size The size of the shared memory segment in bytes.
 * @return 0 on success.
 */
int arch_alloc_init_once(arch_alloc_s *alloc, size_t size);

/**
 * Perform necessary initialization for every vine_talk application.
 *
 * \note This has to be called on new processes, that have not called
 * \c arch_alloc_init_once
 *
 * @param alloc
 */
void arch_alloc_init_always(arch_alloc_s *alloc);

/**
 * Allocate contiguous memory from the alloc arch_alloc_s instance.
 *
 * @param alloc An initialized arch_alloc_s instance.
 * @param size The size of the allocation.
 * @return Pointer to the beginning of size usable bytes,
 * or NULL on failure.
 */
void* arch_alloc_allocate(arch_alloc_s *alloc, size_t size);

void _arch_alloc_free(arch_alloc_s *alloc, void *mem);

/**
 * Free previously allocated memory from a arch_alloc_s instance.
 *
 * @param alloc An initialized arch_alloc_s instance.
 * @param mem A pointer returned from arch_alloc_allocate.
 */
#define arch_alloc_free(ALLOC, MEM)  \
    ({                              \
        _arch_alloc_free(ALLOC, MEM); \
        MEM = 0;                    \
    })

/**
 * Release any resources claimed by the alloc arch_alloc_s instance.
 * @note The shared memory segment (shm in arch_alloc_init) must be freed
 * by the user.
 */
void arch_alloc_exit(arch_alloc_s *alloc);

typedef struct
{
    size_t total_bytes; // <Bytes available for user data AND allocator metadata.
    size_t used_bytes;  // <Bytes used for user data AND allocator metadata.
    #ifdef ALLOC_STATS
    size_t allocs[2];   // < Number of arch_alloc_allocate(failed/successful).
    size_t frees;       // < Number of arch_alloc_free.
    size_t alloc_ns[2]; // < Cumulative nanoseconds spend in alloc(failed/successful).
    size_t free_ns;     // < Cumulative nanoseconds spend in free.
    #endif
} arch_alloc_stats_s;

arch_alloc_stats_s arch_alloc_stats(arch_alloc_s *alloc);

void arch_alloc_inspect(arch_alloc_s *alloc, void (*inspector)(void *start, void *end, size_t size,
  void *arg), void *arg);

utils_bitmap_s* arch_alloc_get_bitmap();
#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef ARCH_ALLOCATOR_HEADER */
