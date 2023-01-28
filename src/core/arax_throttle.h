#ifndef ARAX_THROTTLE_HEADER
#define ARAX_THROTTLE_HEADER

typedef struct arax_throttle_s arax_throttle_s;

#include "async.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

struct arax_throttle_s
{
    size_t            available;
    size_t            capacity;
    async_condition_s ready;
    #ifdef ARAX_THROTTLE_DEBUG
    volatile size_t   print_cnt;
    utils_spinlock    lock;
    #endif
};

#ifdef ARAX_THROTTLE_DEBUG
#define ARAX_THROTTLE_DEBUG_PARAMS , const char *func
#define ARAX_THROTTLE_DEBUG_FUNC(FUNC)  __ ## FUNC
#define ARAX_THROTTLE_DEBUG_PRINT(...)  fprintf(stderr, __VA_ARGS__)
#define arax_throttle_size_inc(thr, sz) __arax_throttle_size_inc(thr, sz, __func__)
#define arax_throttle_size_dec(thr, sz) __arax_throttle_size_dec(thr, sz, __func__)
#else
#define ARAX_THROTTLE_DEBUG_PARAMS
#define ARAX_THROTTLE_DEBUG_FUNC(FUNC) FUNC
#define ARAX_THROTTLE_DEBUG_PRINT(...) ({ })
#endif

/**
 * Increments available size by sz
 *
 * @param meta   async meta for cond wait
 * @param thr    arax_throttle_s instance to inc
 * @return       Nothing .
 */
void arax_throttle_init(async_meta_s *meta, arax_throttle_s *thr, size_t a_sz, size_t t_sz);

/**
 * Increments available size by sz
 *
 * @param thr    arax_throttle_s instance to inc
 * @param sz     Size of added data
 * @return       Nothing .
 */
void ARAX_THROTTLE_DEBUG_FUNC(arax_throttle_size_inc)(arax_throttle_s * thr, size_t sz ARAX_THROTTLE_DEBUG_PARAMS);

/**
 * Decrements available size by sz
 *
 * @param thr    arax_throttle_s instance to dec
 * @param sz     size of removed data
 * @return       Nothing .
 */
void ARAX_THROTTLE_DEBUG_FUNC(arax_throttle_size_dec)(arax_throttle_s * thr, size_t sz ARAX_THROTTLE_DEBUG_PARAMS);

/**
 * Gets available size
 *
 * @param thr    arax_throttle_s instance
 * @return       Avaliable size.
 */
size_t arax_throttle_get_available_size(arax_throttle_s *thr);

/**
 * Gets available size
 *
 * @param thr    arax_throttle_s instance
 * @return       Total size
 */
size_t arax_throttle_get_total_size(arax_throttle_s *thr);


#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef ARAX_THROTTLE_HEADER */
