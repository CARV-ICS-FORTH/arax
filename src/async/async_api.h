#ifndef ASYNC_API_HEADER
#define ASYNC_API_HEADER
#include "arch/alloc.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Initialize a async_meta_s object once.
 *
 * This will be called only once, on the first node.
 *
 * @param meta An uninitialized async_meta_s object.
 * @param alloc Allocator instance to be used for internall allocations.
 */
void async_meta_init_once(async_meta_s * meta, arch_alloc_s * alloc);

/**
 * Initialize a async_meta_s object on every node.
 *
 * This will be called multiple times, once for every node.
 *
 * @param meta An uninitialized async_meta_s object.
 */
void async_meta_init_always(async_meta_s * meta);

/**
 * Create and register async_completion_s objects created in \c buff.
 *
 * @param meta Pointer to async_meta_s that will 'own' this completion.
 * @param completion Completion to be initialized
 * @return Number of objects created, should be buff_size/async_completion_size().
 */
void async_completion_init(async_meta_s * meta, async_completion_s * completion);

/**
 * Mark \c compl as completed and notify pending async_completion_wait() callers.
 *
 * @param completion Completion to be marked as completed.
 */
void async_completion_complete(async_completion_s * completion);

/**
 * Check if completion has been marked as completed.
 *
 * @param completion Completion to be checked.
 * @return 0 if not completed, !0 if completed.
 */
int async_completion_check(async_completion_s * completion);

/**
 * Wait for \c compl to be completed with async_completion_complete().
 *
 * @param completion Sleep untill it has been completed with async_completion_complete.
 */
void async_completion_wait(async_completion_s * completion);

/**
 * Initialize semaphore.
 *
 * @param meta Pointer to async_meta_s that will 'own' this semaphore.
 * @param sem Semaphore to be initialized
 */
void async_semaphore_init(async_meta_s * meta, async_semaphore_s * sem);

/**
 * Return value of \c sem.
 * @param sem Semaphore to be initialized
 */
int async_semaphore_value(async_semaphore_s * sem);

/**
 * Increase semaphore.
 *
 * Increase(ie produce) \c sem by one.
 * This function will never block.
 *
 * @param sem Semaphore to be increased.
 */
void async_semaphore_inc(async_semaphore_s * sem);

/**
 * Decrease semaphore.
 *
 * Decrease(ie consume) \c sem by one.
 * This function will block if async_semaphore_value() == 0.
 *
 * @param sem Semaphore to be increased.
 */
void async_semaphore_dec(async_semaphore_s * sem);

/**
 * Initialize semaphore.
 *
 * @param meta Pointer to async_meta_s that will 'own' this semaphore.
 * @param cond Condition to be initialized
 */
void async_condition_init(async_meta_s * meta, async_condition_s * cond);

/**
 * Lock on condition \c cond.
 *
 * @param cond Condition to be read.
 */
void async_condition_lock(async_condition_s * cond);

/**
 * Wait on \c cond.
 *
 * \note Prior to this call \c cond must be locked using async_condition_lock().
 *
 * @param cond Condition to be read.
 */
void async_condition_wait(async_condition_s * cond);

/**
 * Notify \c cond.
 *
 * \note Prior to this call \c cond must be locked using async_condition_lock().
 *
 * @param cond Condition to be notified.
 */
void async_condition_notify(async_condition_s * cond);

/**
 * Lock on condition \c cond.
 *
 * \note Prior to this call \c cond must be locked using async_condition_lock().
 *
 * @param cond Condition to be read.
 */
void async_condition_unlock(async_condition_s * cond);

/**
 * De initialize an async_meta_s object.
 *
 * @param meta The async_meta_s object to be uninitialized.
 */
void async_meta_exit(async_meta_s * meta);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */


#endif // ifndef ASYNC_API_HEADER
