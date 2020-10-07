#ifndef VINE_THROTTLE_HEADER
#define VINE_THROTTLE_HEADER

typedef struct vine_throttle_s vine_throttle_s;

#include "async.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

struct vine_throttle_s
{
    size_t            available;
    size_t            capacity;
    async_condition_s ready;
    #ifdef VINE_THROTTLE_DEBUG
    volatile size_t   print_cnt;
    utils_spinlock    lock;
    #endif
};

#ifdef VINE_THROTTLE_DEBUG
#define VINE_THROTTLE_DEBUG_PARAMS , const char *func, const char *parent
#define VINE_THROTTLE_DEBUG_FUNC(FUNC)  __ ## FUNC
#define VINE_THROTTLE_DEBUG_PRINT(...)  fprintf(stderr, __VA_ARGS__)
#define vine_throttle_size_inc(thr, sz) __vine_throttle_size_inc(thr, sz, __func__, parent)
#define vine_throttle_size_dec(thr, sz) __vine_throttle_size_dec(thr, sz, __func__, parent)
#else
#define VINE_THROTTLE_DEBUG_PARAMS
#define VINE_THROTTLE_DEBUG_FUNC(FUNC) FUNC
#define VINE_THROTTLE_DEBUG_PRINT(...) ({ })
#endif

/**
 * Increments available size by sz
 *
 * @param meta   async meta for cond wait
 * @param thr    vine_throttle_s instance to inc
 * @return       Nothing .
 */
void vine_throttle_init(async_meta_s *meta, vine_throttle_s *thr, size_t a_sz, size_t t_sz);

/**
 * Increments available size by sz
 *
 * @param thr    vine_throttle_s instance to inc
 * @param sz     Size of added data
 * @return       Nothing .
 */
void VINE_THROTTLE_DEBUG_FUNC(vine_throttle_size_inc)(vine_throttle_s * thr, size_t sz VINE_THROTTLE_DEBUG_PARAMS);

/**
 * Decrements available size by sz
 *
 * @param thr    vine_throttle_s instance to dec
 * @param sz     size of removed data
 * @return       Nothing .
 */
void VINE_THROTTLE_DEBUG_FUNC(vine_throttle_size_dec)(vine_throttle_s * thr, size_t sz VINE_THROTTLE_DEBUG_PARAMS);

/**
 * Gets available size
 *
 * @param thr    vine_throttle_s instance
 * @return       Avaliable size.
 */
size_t vine_throttle_get_available_size(vine_throttle_s *thr);

/**
 * Gets available size
 *
 * @param thr    vine_throttle_s instance
 * @return       Total size
 */
size_t vine_throttle_get_total_size(vine_throttle_s *thr);


#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_THROTTLE_HEADER */
