#ifndef ARCH_ASYNC_HEADER
#define ARCH_ASYNC_HEADER
#include <stddef.h>
#include "core/vine_object.h"

/**
 * Object allowing (a)synchronous operations.
 */
typedef vine_object_s arch_async_completion_s;

/**
 * Object holding all necessary state for implementing the arch_async_* API.
 */
typedef vine_object_s arch_async_provider_s;

/**
 * Create and register arch_async_completion_s objects created in \c buff.
 *
 * @return Number of objects created, should be buff_size/arch_async_completion_size().
 */
arch_async_provider_s * arch_async_completion_create(void * buff,int buff_size);

/**
 * Get a arch_async_completion_s object from internall pool.
 *
 * @return A valid arch_async_completion_s object or NULL.
 */
arch_async_completion_s * arch_async_completion_get(arch_async_provider_s * prov);

/**
 * Return \c compl to the \c repo pool of arch_async_completion_s *
 */
void arch_async_completion_put(arch_async_provider_s * prov,arch_async_completion_s * compl);

/**
 * Mark \c compl as completed and notify pending arch_async_completion_wait() callers.
 */
void arch_async_completion_complete(arch_async_completion_s * compl);

/**
 * Wait for \c compl to be completed with arch_async_completion_complete().
 */
void arch_async_completion_wait(arch_async_completion_s * compl);

#endif
