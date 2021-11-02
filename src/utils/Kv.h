#ifndef VINEYARD_KV_HEADER
#define VINEYARD_KV_HEADER
#include <conf.h>
#include <stddef.h>
#include "spinlock.h"

/**
 * Basic key-value object.
 */
typedef struct
{
    struct Pair
    {
        void *key;
        void *value;
    }              kv[VINE_KV_CAP];
    size_t         pairs;
    utils_spinlock lock;
} utils_kv_s;

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Initialize \c kv.
 */
void utils_kv_init(utils_kv_s *kv);

/**
 * Set kv[key] = value.
 *
 * Overwrites existing value.
 */
void utils_kv_set(utils_kv_s *kv, void *key, void *value);

/**
 * Return &(kv[key]) if found.
 *
 * Returns NULL if \c key not found.
 */
void** utils_kv_get(utils_kv_s *kv, void *key);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif // ifndef VINEYARD_KV_HEADER
