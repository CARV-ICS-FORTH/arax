#include "vine_throttle.h"
#include "utils/vine_assert.h"
#include "stdio.h"

#ifdef VINE_THROTTLE_DEBUG
void PRINT_THR(vine_throttle_s *thr, long int delta, const char *func, const char *parent)
{
    utils_spinlock_lock(&(thr->lock));
    vine_assert(thr->print_cnt < 100000);
    printf("#%05ld %30s->%30s(%p) ,sz: %6ld ,was: %11lu => is: %11lu, cap: %11ld, used:%11ld)\n",
      thr->print_cnt,
      parent,
      func,
      thr,
      delta,
      thr->available,
      thr->available + delta,
      thr->capacity,
      thr->capacity - (thr->available + delta)
    );
    thr->print_cnt++;
    utils_spinlock_unlock(&(thr->lock));
}

#else /* ifdef VINE_THROTTLE_DEBUG */
#define PRINT_THR(OBJ, DELTA, FUNC, PARENT)
#endif /* ifdef VINE_THROTTLE_DEBUG */

void vine_throttle_init(async_meta_s *meta, vine_throttle_s *thr, size_t a_sz, size_t t_sz)
{
    // error check
    vine_assert(meta);
    vine_assert(thr);
    vine_assert(a_sz > 0);
    vine_assert(t_sz > 0);
    vine_assert(t_sz >= a_sz);

    // init sizes
    thr->available = a_sz;
    thr->capacity  = t_sz;
    // init async
    async_condition_init(meta, &thr->ready);

    #ifdef VINE_THROTTLE_DEBUG
    thr->print_cnt = 0;
    utils_spinlock_init(&(thr->lock));
    #endif
}

void VINE_THROTTLE_DEBUG_FUNC(vine_throttle_size_inc)(vine_throttle_s * thr, size_t sz VINE_THROTTLE_DEBUG_PARAMS){
    // error check
    vine_assert(thr);

    if (!sz)
        return;

    // lock critical section
    async_condition_lock(&(thr->ready));

    PRINT_THR(thr, +sz, func, parent);

    // inc available size
    #ifdef VINE_THROTTLE_ENABLE
    thr->available += sz;
    #endif

    // check bad use of api
    vine_assert(thr->capacity >= thr->available);

    // notify to stop async_condition_wait
    async_condition_notify(&(thr->ready));

    // unlock critical section
    async_condition_unlock(&(thr->ready));
}


void VINE_THROTTLE_DEBUG_FUNC(vine_throttle_size_dec)(vine_throttle_s * thr, size_t sz VINE_THROTTLE_DEBUG_PARAMS){
    // error check
    vine_assert(thr);

    if (!sz)
        return;

    // lock critical section
    async_condition_lock(&(thr->ready));

    // wait till there is space to dec coutner
    while (thr->available < sz)
        async_condition_wait(&(thr->ready));

    PRINT_THR(thr, -sz, func, parent);

    // dec available size
    #ifdef VINE_THROTTLE_ENABLE
    thr->available -= sz;
    #endif

    // check bad use of api
    vine_assert(thr->capacity >= thr->available);

    // unlock critical section
    async_condition_unlock(&(thr->ready));
}


size_t vine_throttle_get_available_size(vine_throttle_s *thr)
{
    // error check
    vine_assert(thr);
    return thr->available;
}

size_t vine_throttle_get_total_size(vine_throttle_s *thr)
{
    // error check
    vine_assert(thr);
    return thr->capacity;
}
