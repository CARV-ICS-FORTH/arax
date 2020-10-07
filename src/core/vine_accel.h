#ifndef VINE_ACCEL_HEADER
#define VINE_ACCEL_HEADER
#include <vine_talk.h>
typedef struct vine_accel_s vine_accel_s;

#include "core/vine_vaccel.h"
#include "core/vine_throttle.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

struct vine_accel_s
{
    vine_object_s      obj;
    vine_accel_type_e  type;
    utils_spinlock     lock;
    utils_list_s       vaccels;
    vine_accel_loc_s   location;
    vine_accel_stats_s stats;
    vine_accel_state_e state;
    size_t             revision;
    vine_throttle_s    throttle;
    /* To add more as needed */
};

/**
 * Allocate and initialize a vine_accel descriptor with the provided arguements.
 * @pipe A valid vine_pipe_s* instance.
 * @pipe Name of new accelerator.
 * @type Accelerator type/architecture.
 * @size Avaliable accelerator memory in bytes.
 * @return An initialized vine_accel instance on success, or NULL on failure.
 */
vine_accel_s* vine_accel_init(vine_pipe_s *pipe, const char *name,
  vine_accel_type_e type, size_t size, size_t capacity);

/**
 * Get name.
 */
const char* vine_accel_get_name(vine_accel_s *accel);

/**
 * Get stats.
 *
 * @param accel A physsical accelerator
 */
vine_accel_state_e vine_accel_get_stat(vine_accel_s *accel, vine_accel_stats_s *stat);

/**
 * Increase 'revision' of accelerator.
 *
 * @param accel A physsical accelerator
 */
void vine_accel_inc_revision(vine_accel_s *accel);

/**
 * Get 'revision' of accelerator.
 *
 * @param accel A physsical accelerator
 * @return      Revision
 */
size_t vine_accel_get_revision(vine_accel_s *accel);


#ifdef VINE_THROTTLE_DEBUG
#define VINE_THROTTLE_DEBUG_ACCEL_PARAMS , const char *parent
#define VINE_THROTTLE_DEBUG_ACCEL_FUNC(FUNC) __ ## FUNC
#define vine_accel_size_inc(vac, sz)         __vine_accel_size_inc(vac, sz, __func__)
#define vine_accel_size_dec(vac, sz)         __vine_accel_size_dec(vac, sz, __func__)
#else
#define VINE_THROTTLE_DEBUG_ACCEL_PARAMS
#define VINE_THROTTLE_DEBUG_ACCEL_FUNC(FUNC) FUNC
#endif

/**
 * Increments available size of accelerator by sz
 *
 * @param accel A physsical accelerator
 * @param sz     Size of added data
 */
void VINE_THROTTLE_DEBUG_ACCEL_FUNC(vine_accel_size_inc)(vine_accel * accel,
  size_t sz VINE_THROTTLE_DEBUG_ACCEL_PARAMS);

/**
 * Decrements available size of gpu by sz
 *
 * @param accel A physsical accelerator
 * @param sz    size of removed data
 */
void VINE_THROTTLE_DEBUG_ACCEL_FUNC(vine_accel_size_dec)(vine_accel * accel,
  size_t sz VINE_THROTTLE_DEBUG_ACCEL_PARAMS);

/**
 * Gets available size of GPU
 *
 * @param accel A physsical accelerator
 * @return       Avaliable size of accelerator
 */
size_t vine_accel_get_available_size(vine_accel *accel);

/**
 * Gets available size of GPU
 *
 * @param accel A physsical accelerator
 * @return       Total size of accelerator
 */
size_t vine_accel_get_total_size(vine_accel *accel);

/**
 * Add (register) a virtual accell \c vaccel to physical accelerator \c accel.
 *
 * @param accel A physsical accelerator
 * @param vaccel A virtual accelerator to be linked with \c accel
 */
void vine_accel_add_vaccel(vine_accel_s *accel, vine_vaccel_s *vaccel);

/**
 * Delete (unregister) a virtual accell \c vaccel from physical accelerator \c accel.
 *
 * @param accel A physsical accelerator
 * @param vaccel A virtual accelerator to be unlinked from \c accel
 */
void vine_accel_del_vaccel(vine_accel_s *accel, vine_vaccel_s *vaccel);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_ACCEL_HEADER */
