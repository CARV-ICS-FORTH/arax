#ifndef ARAX_ACCEL_HEADER
#define ARAX_ACCEL_HEADER
#include <arax.h>
typedef struct arax_accel_s arax_accel_s;

#include "async.h"
#include "core/arax_vaccel.h"
#include "core/arax_throttle.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

struct arax_accel_s
{
    arax_object_s      obj;
    arax_accel_type_e  type;
    arax_accel_state_e state;
    async_condition_s  lock; /* protect vaccels & tasks */
    utils_list_s       vaccels;
    size_t             tasks; /**< Number of pending tasks */
    size_t             revision;
    arax_throttle_s    throttle;
    arax_accel_stats_s stats;
    arax_vaccel_s *    free_vaq;
    /* To add more as needed */
};

/**
 * Allocate and initialize a arax_accel descriptor with the provided arguments.
 * @pipe A valid arax_pipe_s* instance.
 * @pipe Name of new accelerator.
 * @type Accelerator type/architecture.
 * @size Available accelerator memory in bytes.
 * @return An initialized arax_accel instance on success, or NULL on failure.
 */
arax_accel_s* arax_accel_init(arax_pipe_s *pipe, const char *name,
  arax_accel_type_e type, size_t size, size_t capacity);

/**
 * Block until a task is added to any of the arax_vaccel_s assigned to \c accel.
 *
 * This function reduces the number of pending tasks of this arax_accel_s
 * (\c arax_accel_s::tasks).
 */
void arax_accel_wait_for_task(arax_accel_s *accel);

/**
 * Increase the number of tasks of \c accel and notify blocked
 * \c arax_accel_wait_for_task() callers.
 *
 * This function increases the number of pending tasks of this arax_accel_s
 * (\c arax_accel_s::tasks).
 */
void arax_accel_add_task(arax_accel_s *accel);

/**
 * Return pending tasks for \c accel.
 */
size_t arax_accel_pending_tasks(arax_accel_s *accel);

/**
 * Get name.
 */
const char* arax_accel_get_name(arax_accel_s *accel);

/**
 * Get stats.
 *
 * @param accel A physical accelerator
 */
arax_accel_state_e arax_accel_get_stat(arax_accel_s *accel, arax_accel_stats_s *stat);

/**
 * Increase 'revision' of accelerator.
 *
 * @param accel A physical accelerator
 */
void arax_accel_inc_revision(arax_accel_s *accel);

/**
 * Get 'revision' of accelerator.
 *
 * @param accel A physical accelerator
 * @return      Revision
 */
size_t arax_accel_get_revision(arax_accel_s *accel);


#ifdef ARAX_THROTTLE_DEBUG
#define ARAX_THROTTLE_DEBUG_ACCEL_PARAMS , const char *parent
#define ARAX_THROTTLE_DEBUG_ACCEL_FUNC(FUNC) __ ## FUNC
#define arax_accel_size_inc(vac, sz)         __arax_accel_size_inc(vac, sz, __func__)
#define arax_accel_size_dec(vac, sz)         __arax_accel_size_dec(vac, sz, __func__)
#else
#define ARAX_THROTTLE_DEBUG_ACCEL_PARAMS
#define ARAX_THROTTLE_DEBUG_ACCEL_FUNC(FUNC) FUNC
#endif

/**
 * Increments available size of accelerator by sz
 *
 * @param accel A physical accelerator
 * @param sz     Size of added data
 */
void ARAX_THROTTLE_DEBUG_ACCEL_FUNC(arax_accel_size_inc)(arax_accel * accel,
  size_t sz ARAX_THROTTLE_DEBUG_ACCEL_PARAMS);

/**
 * Decrements available size of gpu by sz
 *
 * @param accel A physical accelerator
 * @param sz    size of removed data
 */
void ARAX_THROTTLE_DEBUG_ACCEL_FUNC(arax_accel_size_dec)(arax_accel * accel,
  size_t sz ARAX_THROTTLE_DEBUG_ACCEL_PARAMS);

/**
 * Gets available size of GPU
 *
 * @param accel A physical accelerator
 * @return       Avaliable size of accelerator
 */
size_t arax_accel_get_available_size(arax_accel *accel);

/**
 * Gets available size of GPU
 *
 * @param accel A physical accelerator
 * @return       Total size of accelerator
 */
size_t arax_accel_get_total_size(arax_accel *accel);

/**
 * Add (register) a virtual accell \c vaccel to physical accelerator \c accel.
 *
 * If \c vaccel is already assigned to \c accel, the function is no-op.
 * If \c vaccel is not yet assigned to any accel, it will be assigned to \c accel.
 * In any other behaviour is undefined.
 *
 * \note This call should be matched to calls of arax_accel_del_vaccel()
 *
 * @param accel A physical accelerator
 * @param vaccel A virtual accelerator to be linked with \c accel
 */
void arax_accel_add_vaccel(arax_accel_s *accel, arax_vaccel_s *vaccel);

/**
 * Return all arax_vaccel_s objects 'assigned' to \c accel.
 *
 * The initial value of \c vaccel does not matter.
 * The value of \c vaccel will be overwriten by a malloc call.
 * After the call, the user is responsible for freeing \c vaccel using \c free().
 *
 * @param vaccel Pointer to unallocated array that will contain assigned arax_vaccel_sobjects.
 * @return Size of vaccel array, in number of objects/pointers.
 */
size_t arax_accel_get_assigned_vaccels(arax_accel_s *accel, arax_vaccel_s ***vaccel);

/**
 * Delete (unregister) a virtual accell \c vaccel from physical accelerator \c accel.
 *
 * \note This call should be matched to calls of arax_accel_add_vaccel()
 *
 * @param accel A physical accelerator
 * @param vaccel A virtual accelerator to be unlinked from \c accel
 */
void arax_accel_del_vaccel(arax_accel_s *accel, arax_vaccel_s *vaccel);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef ARAX_ACCEL_HEADER */
