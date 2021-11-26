#ifndef VINE_ACCEL_HEADER
#define VINE_ACCEL_HEADER
#include <vine_talk.h>
typedef struct vine_accel_s vine_accel_s;

#include "async.h"
#include "core/vine_vaccel.h"
#include "core/vine_throttle.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

struct vine_accel_s
{
    vine_object_s      obj;
    vine_accel_type_e  type;
    vine_accel_state_e state;
    async_condition_s  lock; /* protect vaccels & tasks */
    utils_list_s       vaccels;
    size_t             tasks; /**< Number of pending tasks */
    size_t             revision;
    vine_throttle_s    throttle;
    vine_accel_stats_s stats;
    vine_vaccel_s *    free_vaq;
    /* To add more as needed */
};

/**
 * Allocate and initialize a vine_accel descriptor with the provided arguments.
 * @pipe A valid vine_pipe_s* instance.
 * @pipe Name of new accelerator.
 * @type Accelerator type/architecture.
 * @size Available accelerator memory in bytes.
 * @return An initialized vine_accel instance on success, or NULL on failure.
 */
vine_accel_s* vine_accel_init(vine_pipe_s *pipe, const char *name,
  vine_accel_type_e type, size_t size, size_t capacity);

/**
 * Block until a task is added to any of the vine_vaccel_s assigned to \c accel.
 *
 * This function reduces the number of pending tasks of this vine_accel_s
 * (\c vine_accel_s::tasks).
 */
void vine_accel_wait_for_task(vine_accel_s *accel);

/**
 * Increase the number of tasks of \c accel and notify blocked
 * \c vine_accel_wait_for_task() callers.
 *
 * This function increases the number of pending tasks of this vine_accel_s
 * (\c vine_accel_s::tasks).
 */
void vine_accel_add_task(vine_accel_s *accel);

/**
 * Return pending tasks for \c accel.
 */
size_t vine_accel_pending_tasks(vine_accel_s *accel);

/**
 * Get name.
 */
const char* vine_accel_get_name(vine_accel_s *accel);

/**
 * Get stats.
 *
 * @param accel A physical accelerator
 */
vine_accel_state_e vine_accel_get_stat(vine_accel_s *accel, vine_accel_stats_s *stat);

/**
 * Increase 'revision' of accelerator.
 *
 * @param accel A physical accelerator
 */
void vine_accel_inc_revision(vine_accel_s *accel);

/**
 * Get 'revision' of accelerator.
 *
 * @param accel A physical accelerator
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
 * @param accel A physical accelerator
 * @param sz     Size of added data
 */
void VINE_THROTTLE_DEBUG_ACCEL_FUNC(vine_accel_size_inc)(vine_accel * accel,
  size_t sz VINE_THROTTLE_DEBUG_ACCEL_PARAMS);

/**
 * Decrements available size of gpu by sz
 *
 * @param accel A physical accelerator
 * @param sz    size of removed data
 */
void VINE_THROTTLE_DEBUG_ACCEL_FUNC(vine_accel_size_dec)(vine_accel * accel,
  size_t sz VINE_THROTTLE_DEBUG_ACCEL_PARAMS);

/**
 * Gets available size of GPU
 *
 * @param accel A physical accelerator
 * @return       Avaliable size of accelerator
 */
size_t vine_accel_get_available_size(vine_accel *accel);

/**
 * Gets available size of GPU
 *
 * @param accel A physical accelerator
 * @return       Total size of accelerator
 */
size_t vine_accel_get_total_size(vine_accel *accel);

/**
 * Add (register) a virtual accell \c vaccel to physical accelerator \c accel.
 *
 * If \c vaccel is already assigned to \c accel, the function is no-op.
 * If \c vaccel is not yet assigned to any accel, it will be assigned to \c accel.
 * In any other behaviour is undefined.
 *
 * \note This call should be matched to calls of vine_accel_del_vaccel()
 *
 * @param accel A physical accelerator
 * @param vaccel A virtual accelerator to be linked with \c accel
 */
void vine_accel_add_vaccel(vine_accel_s *accel, vine_vaccel_s *vaccel);

/**
 * Return all vine_vaccel_s objects 'assigned' to \c accel.
 *
 * The initial value of \c vaccel does not matter.
 * The value of \c vaccel will be overwriten by a malloc call.
 * After the call, the user is responsible for freeing \c vaccel using \c free().
 *
 * @param vaccel Pointer to unallocated array that will contain assigned vine_vaccel_sobjects.
 * @return Size of vaccel array, in number of objects/pointers.
 */
size_t vine_accel_get_assigned_vaccels(vine_accel_s *accel, vine_vaccel_s ***vaccel);

/**
 * Delete (unregister) a virtual accell \c vaccel from physical accelerator \c accel.
 *
 * \note This call should be matched to calls of vine_accel_add_vaccel()
 *
 * @param accel A physical accelerator
 * @param vaccel A virtual accelerator to be unlinked from \c accel
 */
void vine_accel_del_vaccel(vine_accel_s *accel, vine_vaccel_s *vaccel);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_ACCEL_HEADER */
