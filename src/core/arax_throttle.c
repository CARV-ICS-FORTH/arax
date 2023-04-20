#include "arax_throttle.h"
#include "utils/arax_assert.h"
#include "stdio.h"

#ifdef ARAX_THROTTLE_DEBUG
void PRINT_THR(arax_throttle_s *thr, long int delta, const char *func)
{
    utils_spinlock_lock(&(thr->lock));
    arax_assert(thr->print_cnt < 100000);
    printf("#%05ld %30s(%p) ,sz: %6ld ,was: %11lu => is: %11lu, cap: %11ld, used:%11ld)\n",
      thr->print_cnt,
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

#else /* ifdef ARAX_THROTTLE_DEBUG */
#define PRINT_THR(OBJ, DELTA, FUNC)
#endif /* ifdef ARAX_THROTTLE_DEBUG */

void arax_throttle_init(async_meta_s *meta, arax_throttle_s *thr, size_t a_sz, size_t t_sz)
{
    // error check
    arax_assert(meta);
    arax_assert(thr);
    arax_assert(a_sz > 0);
    arax_assert(t_sz > 0);
    arax_assert(t_sz >= a_sz);

    // init sizes
    thr->available = a_sz;
    thr->capacity  = t_sz;
    // init async
    async_condition_init(meta, &thr->ready);

    #ifdef ARAX_THROTTLE_DEBUG
    thr->print_cnt = 0;
    utils_spinlock_init(&(thr->lock));
    #endif
}

void ARAX_THROTTLE_DEBUG_FUNC(arax_throttle_size_inc)(arax_throttle_s * thr, size_t sz ARAX_THROTTLE_DEBUG_PARAMS){
    // error check
    arax_assert(thr);

    if (!sz)
        return;

    // lock critical section
    async_condition_lock(&(thr->ready));

    PRINT_THR(thr, +sz, func);

    // inc available size
    thr->available += sz;

    #ifdef ARAX_THROTTLE_ENFORCE
    // check bad use of api
    arax_assert(thr->capacity >= thr->available);
    #endif

    // notify to stop async_condition_wait
    async_condition_notify(&(thr->ready));

    // unlock critical section
    async_condition_unlock(&(thr->ready));
}


void ARAX_THROTTLE_DEBUG_FUNC(arax_throttle_size_dec)(arax_throttle_s * thr, size_t sz ARAX_THROTTLE_DEBUG_PARAMS){
    // error check
    arax_assert(thr);

    if (!sz)
        return;

    // lock critical section
    async_condition_lock(&(thr->ready));

    #ifdef ARAX_THROTTLE_ENFORCE
    // wait till there is space to dec coutner
    while (thr->available < sz)
        async_condition_wait(&(thr->ready));
    #endif

    PRINT_THR(thr, -sz, func);

    // dec available size
    thr->available -= sz;

    #ifdef ARAX_THROTTLE_ENFORCE
    // check bad use of api
    arax_assert(thr->capacity >= thr->available);
    #endif

    // unlock critical section
    async_condition_unlock(&(thr->ready));
}


size_t arax_throttle_get_available_size(arax_throttle_s *thr)
{
    // error check
    arax_assert(thr);
    return thr->available;
}

size_t arax_throttle_get_total_size(arax_throttle_s *thr)
{
    // error check
    arax_assert(thr);
    return thr->capacity;
}
