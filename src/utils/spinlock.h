#ifndef UTILS_SPINLOCK_HEADER
#define UTILS_SPINLOCK_HEADER
#include <stdint.h>
#include "utils/vine_assert.h"

#if __x86_64__
typedef volatile uint64_t utils_spinlock;

#else /* if __x86_64__ */
typedef volatile uint32_t utils_spinlock;

#endif /* if __x86_64__ */

/**
 * Initialize \c lock as unlocked.
 *
 * @param lock utils_spinlock to be initialized as unlocked.
 */
static inline void utils_spinlock_init(utils_spinlock *lock)
{
    *lock = 0;
}

/**
 * Lock \c lock.
 *
 * Will attempt to lock \c lock.
 * Will spin until succesfull.
 *
 * @param lock utils_spinlock instance to be locked.
 */
static inline void utils_spinlock_lock(utils_spinlock *lock)
{
    do {
        while (*lock)
            ;  /* Maybe add relax()? */
        if (__sync_bool_compare_and_swap(lock, 0, 1) )
            break;  /* We got it */
    } while (1);    /* Try again */
}

/**
 * Will unlock \c lock that was previously locked.
 * \note Calling utils_spinlock_unlock on an unlocked utils_spinlock
 * instance is an error.
 * @param lock utils_spinlock instance to be unlocked.
 */
static inline void utils_spinlock_unlock(utils_spinlock *lock)
{
    vine_assert(*lock); /* Attempting to unlock twice */
    __sync_fetch_and_and(lock, 0);
}

#endif /* ifndef UTILS_SPINLOCK_HEADER */
