#ifndef VINE_SPIN_HEADER
#define VINE_SPIN_HEADER
#include <assert.h>

typedef volatile uint64_t util_spinlock;

/**
 * Initialize \c lock as unlocked.
 *
 * \param lock util_spinlock to be initialized as unlocked.
 */
static inline void util_spinlock_init(util_spinlock * lock)
{
	*lock = 0;
}

/**
 * Lock \c lock.
 *
 * Will attempt to lock \c lock.
 * Will spin until succesfull.
 *
 * \param lock util_spinlock instance to be locked.
 */
static inline void util_spinlock_lock(util_spinlock * lock)
{
	TRY_AGAIN:
	while(lock);			/* Maybe add relax()? */
	if( !__sync_bool_compare_and_swap(lock,0,1) )
		goto TRY_AGAIN;		/* Someone was faster */
}

/**
 * Will unlock \c lock that was previously locked.
 *\note Calling util_spinlock_unlock on an unlocked util_spinlock
 * instance is an error.
 * \param lock util_spinlock instance to be unlocked.
 */
static inline void util_spinlock_unlock(util_spinlock * lock)
{
	assert(*lock);	/* Attempting to unlock twice */
	__sync_fetch_and_and(lock,0);
}

#endif
