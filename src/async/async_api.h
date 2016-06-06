#ifndef ARCH_ASYNC_API_HEADER
#define ARCH_ASYNC_API_HEADER
#include <stddef.h>

/**
 * Create and register async_completion_s objects created in \c buff.
 *
 * @param completion Completion to be initialized
 * @return Number of objects created, should be buff_size/async_completion_size().
 */
void async_completion_init(async_completion_s * completion);

/**
 * Mark \c compl as completed and notify pending async_completion_wait() callers.
 *
 * @param completion Completion to be marked as completed.
 */
void async_completion_complete(async_completion_s * completion);

/**
 * Wait for \c compl to be completed with async_completion_complete().
 *
 * @param completion Sleep untill it has been completed with async_completion_complete.
 */
void async_completion_wait(async_completion_s * completion);

#endif
