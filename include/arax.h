#ifndef ARAX_TALK
#define ARAX_TALK

#include <stdio.h>
#include <stddef.h>
#include "arax_types.h"
#include <core/arax_accel_types.h>
#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/** @defgroup init_clean Initialization/Cleanup
 *  Functions responsible for Arax initialzation and cleanup.
 *  @{
 */

/**
 * Initialize Arax.
 *
 * This should be called by all applications prior to using any other Arax
 * function.
 *
 * @return A arax_pipe_s instance
 */
arax_pipe_s* arax_init();

/**
 * Exit and cleanup Arax.
 */
void arax_exit();

/**
 * Clean/delete shared segment.
 * \note This should only be called when there are no uses of the shared segment.
 * \note Behaviour undefined if called with processes accessing the shared segment.
 * @return Returns true if the shared semgent file was succesfully deleted.
 */
int arax_clean();

/** @} */

/** @defgroup pub_accel_api Public Accelerator user API
 *  Functions usable from applications for manipulating Accelerators.
 *  @{
 */

/**
 * Return number of accelerators of provided type
 * If zero is returned no matching devices were found.
 * If accels is not null an array with all matching accelerator
 * descriptors is allocated and passed to the user.
 * If *accels is not null, it will be freed by \c arax_accel_list_free.
 * \note The *accels pointer must be freed by the user using free().
 *
 * @param type Count only accelerators of specified arax_accel_type_e
 * @param physical Boolean value (0,1), if true return physical accelerators,
 *                 if false return virtual accelerators.
 * @param accels pointer to array with available matching accelerator
 * descriptors.
 * @return Number of available accelerators of specified type.
 */
int arax_accel_list(arax_accel_type_e type, int physical, arax_accel ***accels);

/**
 * Set physical accelator to arax_accel_s
 *
 * @param vaccel Virtual accelator to set physical accelerator
 * @param phys   Physical accelerator to set on arax_accel_s
 * @return       Nothing .
 */
void arax_accel_set_physical(arax_accel *vaccel, arax_accel *phys);

/**
 * Free memory of accelerator array returned by arax_accel_list
 *
 * @param accels pointer acquired through a arax_accel_list call.
 */
void arax_accel_list_free(arax_accel **accels);

/**
 * Return the type of accelerator specified by accel.
 *
 * @param accel A valid arax_accel descriptor returned by arax_accel_list().
 * @return A value from arax_accel_type_e.
 */
arax_accel_type_e arax_accel_type(arax_accel *accel);

/**
 * Return statistics of accelerator specified by accel.
 *
 * @param accel A valid arax_accel descriptor returned by arax_accel_list().
 * @param stat A pointer to a arax_accel_stats_s struct, to be filled with the
 * accel statistics.
 * @return The state of the accelerator at the time of the call.
 */
arax_accel_state_e arax_accel_stat(arax_accel *accel, arax_accel_stats_s *stat);

/**
 * Acquire specific physical accelerator specified by accel for exclusive use.
 *
 * \note By default all accelerators are 'shared'
 * \note Every call to arax_accel_acquire must have a
 * matching arax_accel_release call.
 *
 * @param accel Accelerator to be acquired for exclusive use.
 * @return Return 1 if successful, 0 on failure.
 *
 */
int arax_accel_acquire_phys(arax_accel **accel);

/**
 * Acquire a virtual accelerator of the given \c type.
 *
 * \note Every call to arax_accel_acquire must have a
 * matching arax_accel_release call.
 *
 * @param type Accelerator type to be acquired.
 * @return Return acquired virtual accelerator, NULL on failure.
 *
 */
arax_accel* arax_accel_acquire_type(arax_accel_type_e type);

/**
 * Release previously acquired accelerator.
 *
 * \note By default all accelerators are 'shared'
 * \note Every call to arax_accel_acquire must have a
 * matching arax_accel_release call.
 *
 * @param accel A previously acquired accelerator to be released.
 * @return Return 1 if successful, 0 on failure.
 *
 */
void arax_accel_release(arax_accel **accel);

/** @} */

/** @defgroup pub_proc_api Public Procedure user API
 *  Functions usable from applications for manipulating Procedures.
 *  @{
 */

/**
 * Register a new process 'func_name'.
 * Processes are accelerator agnostic and initially have no 'implementations'/functors.
 * Returned arax_proc * identifies given function globally.
 *
 * \note For every arax_proc_get()/arax_proc_register() there should be a
 * matching call to arax_proc_put()
 *
 * \note To add a functor/implementation see/use \c arax_proc_set_functor().
 *
 * @param func_name Descriptive name of function, has to be unique for given
 * type.
 * @return arax_proc * corresponding to the registered function, NULL on
 * failure.
 */
arax_proc* arax_proc_register(const char *func_name);

/**
 * Retrieve a previously registered arax_proc pointer.
 *
 * \note For every arax_proc_get()/arax_proc_register() there should be a
 * matching call to arax_proc_put()
 *
 * @param func_name Descriptive name of function, as provided to
 * arax_proc_register.
 * @return arax_proc * corresponding to the requested function, NULL on failure.
 */
arax_proc* arax_proc_get(const char *func_name);

/**
 * Delete registered arax_proc pointer.
 *
 * \note For every arax_proc_get()/arax_proc_register() there should be a
 * matching call to arax_proc_put()
 *
 * @param func arax_proc to be deleted.
 */
int arax_proc_put(arax_proc *func);

/** @} */

/** @defgroup pub_task_api Public Task user API
 *  Functions usable from applications for manipulating Tasks.
 *  @{
 */

/**
 * Issue a new arax_task.
 *
 * This call must be followed by calls to arax_task_wait() and arax_task_free().
 *
 * After \c arax_task_wait() and before \c arax_task_free(), \c arax_task_host_data()
 * can be called to read updated host values.
 *
 * @param accel The accelerator responsible for executing the task.
 * @param proc arax_proc to be dispatched on accelerator.
 * @param host_init Host accesible data initial values. May be null. Will not be modified.
 * @param host_size Size of \c host_init data.
 * @param in_count size of input array (elements).
 * @param dev_in array of arax_data pointers with input data.
 * @param out_count size of output array (elements).
 * @param dev_out array of arax_data pointers with output data.
 * @return arax_task * corresponding to the issued function invocation.
 */
arax_task* arax_task_issue(arax_accel *accel, arax_proc *proc, const void *host_init,
  size_t host_size, size_t in_count, arax_data **dev_in, size_t out_count,
  arax_data **dev_out);

/**
 * Helper function for issueing,waiting and freeing a task.
 *
 * @param accel The accelerator responsible for executing the task.
 * @param proc arax_proc to be dispatched on accelerator.
 * @param host_init Host accesible data initial values. May be null. Will not be modified.
 * @param host_size Size of \c host_init data.
 * @param in_count size of input array (elements).
 * @param dev_in array of arax_data pointers with input data.
 * @param out_count size of output array (elements).
 * @param dev_out array of arax_data pointers with output data.
 * @return Returs the status as returned from arax_task_wait().
 */
arax_task_state_e arax_task_issue_sync(arax_accel *accel, arax_proc *proc, void *host_init,
  size_t host_size, size_t in_count, arax_data **dev_in, size_t out_count,
  arax_data **dev_out);


/**
 * Get arax_task status and statistics.
 * If stats is not NULL, copy task statistics to stats.
 *
 * \note This function does not call arax_task_wait(), so the user must call
 *       it before accessing related arax_buffers.
 *
 * @param task The arax_task of interest.
 * @param stats Pointer to an allocated arax_task_stats struct to be filled with
 * statistics.
 * @return The current arax_task_state of the task.
 */
arax_task_state_e arax_task_stat(arax_task *task, arax_task_stats_s *stats);

/**
 * Wait for an issued task to complete or fail.
 *
 * When provided task is successfully completed, user buffers are synchronized
 * with up to date data from araxs internal buffers.
 *
 * @param task The task to wait for.
 * @return The arax_task_state of the given arax_task.
 */
arax_task_state_e arax_task_wait(arax_task *task);

/**
 * Decrease ref counter of task
 *
 * @param task The task to wait for.
 * @return Nothing.
 */
void arax_task_free(arax_task *task);

/** @} */

/** @defgroup pub_buffer_api Public Buffer user API
 *  Functions usable from applications for manipulating arax_buffers.
 *  @{
 */

/**
 * ARAX_BUFFER create a arax_buffer_s object.
 *
 * @param  size          Size of user_buffer.
 * @return arax_buffer_s.
 */
arax_buffer_s ARAX_BUFFER(size_t size);

/** @} */

/**
 * Define a handler for a function named \c FN, for the \c ARCH architecture.
 */
#define ARAX_HANDLER(FN, ARCH) \
    extern "C" arax_task_state_e FN ## _ARAX_FN_ ## ARCH(arax_task_msg_s \
      * task) __attribute__((section(".ARAX_HANDLERS"))); \
    extern "C" arax_task_state_e FN ## _ARAX_FN_ ## ARCH(arax_task_msg_s * task)

/**
 * Define a handler for a function named \c FN, for the \c ARCH architecture.
 * Extended version allows additional arguements.
 */
#define ARAX_HANDLER_EX(FN, ARCH, EX) \
    extern "C" arax_task_state_e FN ## _ARAX_FN_ ## ARCH(arax_task_msg_s \
      * task, EX) __attribute__((section(".ARAX_HANDLERS"))); \
    extern "C" arax_task_state_e FN ## _ARAX_FN_ ## ARCH(arax_task_msg_s * task, EX)


#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef ARAX_TALK */
