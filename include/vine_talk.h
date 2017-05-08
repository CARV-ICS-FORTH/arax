/**
 * @file
 * Example use of the VineTalk API:
 * \include producer.c
 */

#ifndef VINE_TALK
#define VINE_TALK

#include <stdio.h>
#include <stddef.h>
#include "conf.h"
#include "core/vine_buffer.h"
#include "core/vine_accel_types.h"
#include <utils/timer.h>

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * vine_accel: Accelerator descriptor.
 */
typedef void vine_accel;

/**
 * vine_proc: Process descriptor.
 */
typedef void vine_proc;

/**
 * Location of a vine_accel.
 */
typedef struct vine_accel_loc {
	/**< To be filled */
} vine_accel_loc_s;

/**
 * Initialize VineTalk.
 */
void vine_talk_init();

/**
 * Exit and cleanup VineTalk.
 */
void vine_talk_exit();

/**
 * Accelerator Statistics
 */
typedef struct vine_accel_stats {} vine_accel_stats_s;

/**
 * Return number of accelerators of provided type
 * If zero is returned no matching devices were found.
 * If accels is not null an array with all matching accelerator
 * descriptors is allocated and passed to the user.
 * \note The *accels pointer must be freed by the user using free().
 *
 * @param type Count only accelerators of specified vine_accel_type_e
 * @param physical Boolean value (0,1), if true return physical accelerators,
 *                 if false return virtual accelerators.
 * @param accels pointer to array with available matching accelerator
 * descriptors.
 * @return Number of available accelerators of specified type.
 */
int vine_accel_list(vine_accel_type_e type, int physical, vine_accel ***accels);

/**
 * Return location object for accelerator specified by accel.
 *
 * @param accel A valid vine_accel descriptor returned by vine_accel_list().
 * @return vine_accel_loc object specifying the location of an accelerator.
 */
vine_accel_loc_s vine_accel_location(vine_accel *accel);

/**
 * Return the type of accelerator specified by accel.
 *
 * @param accel A valid vine_accel descriptor returned by vine_accel_list().
 * @return A value from vine_accel_type_e.
 */
vine_accel_type_e vine_accel_type(vine_accel *accel);

/**
 * Accelerator State enumeration.
 */
typedef enum vine_accel_state {
	accel_failed, /**< Accelerator has failed. */
	accel_idle, /**< Accelerator is idle. */
	accel_busy /**< Accelerator is busy. */
} vine_accel_state_e;

/**
 * Return statistics of accelerator specified by accel.
 *
 * @param accel A valid vine_accel descriptor returned by vine_accel_list().
 * @param stat A pointer to a vine_accel_stats_s struct, to be filled with the
 * accel statistics.
 * @return The state of the accelerator at the time of the call.
 */
vine_accel_state_e vine_accel_stat(vine_accel *accel, vine_accel_stats_s *stat);

/**
 * Acquire specific physical accelerator specified by accel for exclusive use.
 *
 * \note By default all accelerators are 'shared'
 * \note Every call to vine_accel_acquire must have a
 * matching vine_accel_release call.
 *
 * @param accel Accelerator to be acquired for exclusive use.
 * @return Return 1 if successful, 0 on failure.
 *
 */
int vine_accel_acquire_phys(vine_accel **accel);

/**
 * Acquire a virtual accelerator of the given \c type.
 *
 * \note Every call to vine_accel_acquire must have a
 * matching vine_accel_release call.
 *
 * @param type Accelerator type to be acquired.
 * @return Return acquired virtual accelerator, NULL on failure.
 *
 */
vine_accel * vine_accel_acquire_type(vine_accel_type_e type);

/**
 * Release previously acquired accelerator.
 *
 * \note By default all accelerators are 'shared'
 * \note Every call to vine_accel_acquire must have a
 * matching vine_accel_release call.
 *
 * @param accel A previously acquired accelerator to be released.
 * @return Return 1 if successful, 0 on failure.
 *
 */
int vine_accel_release(vine_accel **accel);

/**
 * Register a new process 'func_name' for vine_accel_type_e type accelerators.
 * Returned vine_proc * identifies given function globally.
 * func_bytes contains executable for the given vine_accel_type_e.
 * (e.g. for Fpga bitstream, for GPU CUDA binaries or source(?), for CPU binary
 * code)
 *
 * In case a function is already registered(same type and func_name), func_bytes
 * will be compared.
 * If func_bytes are equal the function will return the previously registered
 * vine_proc.
 * If func_bytes don't match, the second function will return NULL denoting
 * failure.
 *
 * \note For every vine_proc_get()/vine_proc_register() there should be a
 * matching call to vine_proc_put()
 *
 * @param type Provided binaries work for this type of accelerators.
 * @param func_name Descriptive name of function, has to be unique for given
 * type.
 * @param func_bytes Binary containing executable of the appropriate format.
 * @param func_bytes_size Size of provided func_bytes array in bytes.
 * @return vine_proc * corresponding to the registered function, NULL on
 * failure.
 */
vine_proc* vine_proc_register(vine_accel_type_e type, const char *func_name,
                              const void *func_bytes, size_t func_bytes_size);

/**
 * Retrieve a previously registered vine_proc pointer.
 *
 * \note For every vine_proc_get()/vine_proc_register() there should be a
 * matching call to vine_proc_put()
 *
 * @param type Provided binaries work for this type of accelerators.
 * @param func_name Descriptive name of function, as provided to
 * vine_proc_register.
 * @return vine_proc * corresponding to the requested function, NULL on failure.
 */
vine_proc* vine_proc_get(vine_accel_type_e type, const char *func_name);

/**
 * Delete registered vine_proc pointer.
 *
 * \note For every vine_proc_get()/vine_proc_register() there should be a
 * matching call to vine_proc_put()
 *
 * @param func vine_proc to be deleted.
 */
int vine_proc_put(vine_proc *func);

/**
 * Vineyard Task Descriptor
 */
typedef void vine_task;

/**
 * Issue a new vine_task.
 *
 * This call must be followed by calls to vine_task_wait() and vine_task_free().
 *
 * @param accel The accelerator responsible for executing the task.
 * @param proc vine_proc to be dispatched on accelerator.
 * @param args vine_data pointing to packed function arguments.
 * @param in_count size of input array (elements).
 * @param input array of vine_data pointers with input data.
 * @param out_count size of output array (elements).
 * @param output array of vine_data pointers with output data.
 * @return vine_task * corresponding to the issued function invocation.
 */
vine_task* vine_task_issue(vine_accel *accel, vine_proc *proc, vine_buffer_s *args,
                           size_t in_count, vine_buffer_s *input, size_t out_count,
                           vine_buffer_s *output);

/**
 * Vine Task State enumeration.
 */
typedef enum vine_task_state_e {
	task_failed, /**< Task execution failed. */
	task_issued, /**< Task has been issued. */
	task_completed /**< Task has been completed. */
} vine_task_state_e;

/**
 * Vine Task Statistics
 */
typedef struct vine_task_stats {
	int task_id; /**< Unique among tasks of this instance */
	utils_timer_s task_duration;
} vine_task_stats_s;

/**
 * Get vine_task status and statistics.
 * If stats is not NULL, copy task statistics to stats.
 *
 * \note This function does not call vine_task_wait(), so the user must call
 *       it before accessing related vine_buffers.
 *
 * @param task The vine_task of interest.
 * @param stats Pointer to an allocated vine_task_stats struct to be filled with
 * statistics.
 * @return The current vine_task_state of the task.
 */
vine_task_state_e vine_task_stat(vine_task *task, vine_task_stats_s *stats);

/**
 * Wait for an issued task to complete or fail.
 *
 * When provided task is successfully completed, user buffers are synchronized
 * with up to date data from vine_talks internal buffers.
 *
 * @param task The task to wait for.
 * @return The vine_task_state of the given vine_task.
 */
vine_task_state_e vine_task_wait(vine_task *task);

void vine_task_free(vine_task * task);
#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_TALK */
