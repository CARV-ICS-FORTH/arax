#ifndef VINE_THROTTLE_HEADER
#define VINE_THROTTLE_HEADER

typedef struct vine_throttle_s vine_throttle_s;

#include "async.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

struct vine_throttle_s {
    size_t             AvaliableSize;
    size_t             totalSize;
    async_condition_s  sz_ready;
};

/**
 * Increments avaliable size by sz
 *
 * @param thr    vine_throttle_s instance to inc
 * @param sz     Size of added data
 * @return       Nothing .
 */
void vine_throttle_size_inc(vine_throttle_s* thr,size_t sz);

/**
 * Decrements avaliable size by sz
 *
 * @param thr    vine_throttle_s instance to dec
 * @param sz     size of removed data
 * @return       Nothing .
 */
void vine_throttle_size_dec(vine_throttle_s* thr,size_t sz);

/**
 * Gets avaliable size
 *
 * @param thr    vine_throttle_s instance
 * @return       Avaliable size.
 */
size_t vine_throttle_get_avaliable_size(vine_throttle_s* thr);

/**
 * Gets avaliable size
 *
 * @param thr    vine_throttle_s instance
 * @return       Total size
 */
size_t vine_throttle_get_total_size(vine_throttle_s* thr);


#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_THROTTLE_HEADER */ 
