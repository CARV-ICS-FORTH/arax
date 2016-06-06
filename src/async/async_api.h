#ifndef ARCH_ASYNC_API_HEADER
#define ARCH_ASYNC_API_HEADER
#include <stddef.h>

/**
 * Create and register arch_async_completion_s objects created in \c buff.
 *
 * @param completion Completion to be initialized
 * @return Number of objects created, should be buff_size/arch_async_completion_size().
 */
void arch_async_completion_init(arch_async_completion_s * completion);

/**
 * Mark \c compl as completed and notify pending arch_async_completion_wait() callers.
 *
 * @param completion Completion to be marked as completed.
 */
void arch_async_completion_complete(arch_async_completion_s * completion);

/**
 * Wait for \c compl to be completed with arch_async_completion_complete().
 *
 * @param completion Sleep untill it has been completed with arch_async_completion_complete.
 */
void arch_async_completion_wait(arch_async_completion_s * completion);

#endif
