/**
 * @file
 * Example use of the VinePipe API:
 * \include consumer.c
 */

#ifndef VINE_PIPE_HEADER
#define VINE_PIPE_HEADER
#include <vine_talk.h>
#include "async.h"
#include "arch/alloc.h"
#include "utils/list.h"
#include "utils/queue.h"
#include "utils/spinlock.h"
#include "utils/breakdown.h"
#include "core/vine_accel.h"
#include "core/vine_vaccel.h"
#include "core/vine_task.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */
/**
 * Shared Memory segment layout
 */
typedef struct vine_pipe {
	char               sha[48]; /**< Git revision  */
	void               *self; /**< Pointer to myself */
	uint64_t           shm_size; /**< Size in bytes of shared region */
	uint64_t           processes; /**< Process counter - Processes using this */
	uint64_t           last_uid; /**< Last instance UID */
	vine_object_repo_s objs; /**< Vine object repository  */
	async_meta_s       async; /**< Async related metadata  */
	async_condition_s  tasks_cond;
	int                tasks[VINE_ACCEL_TYPES]; /**< Semaphore tracking number of inflight tasks */
	utils_queue_s      *queue; /**< Queue */
	arch_alloc_s       allocator; /**< Allocator for this shared memory */
} vine_pipe_s;

/**
 * Get VineTalk revision
 *
 * @param pipe vine_pipe instance.
 * @return const string with VineTalk revision.
 */
const char * vine_pipe_get_revision(vine_pipe_s * pipe);

/**
 * Increase process counter for \c pipe.
 *
 * @param pipe vine_pipe instance.
 * @return Number of active processes before adding issuer.
 */
uint64_t vine_pipe_add_process(vine_pipe_s * pipe);

/**
 * Decrease process counter for \c pipe.
 *
 * @param pipe vine_pipe instance.
 * @return Number of active processes before removing issuer.
 */
uint64_t vine_pipe_del_process(vine_pipe_s * pipe);

/**
 * Return (and set if needed) the mmap location for \c pipe.
 *
 * @param pipe vine_pipe instance.
 */
void * vine_pipe_mmap_address(vine_pipe_s * pipe);

/**
 * Initialize a vine_pipe.
 *
 * \note This function must be called from all end points in order to
 * initialize a vine_pipe instance.Concurrent issuers will be serialized
 * and the returned vine_pipe instance will be initialized by the 'first'
 * issuer. All subsequent issuers will receive the already initialized
 * instance.
 *
 *
 * @param mem Shared memory pointer.
 * @param size Size of the shared memory in bytes.
 * @param enforce_version Set to 0 to make version mismatch non fatal.
 * @return An initialized vine_pipe_s instance.
 */
vine_pipe_s* vine_pipe_init(void *mem, size_t size,int enforce_version);

/**
 * Remove \c accel from the \c pipe accelerator list.
 *
 * @param pipe The pipe instance where the accelerator belongs.
 * @param accel The accelerator to be removed.
 * @return Returns 0 on success.
 */
int vine_pipe_delete_accel(vine_pipe_s *pipe, vine_accel_s *accel);

/**
 * Find an accelerator matching the user specified criteria.
 *
 * @param pipe vine_pipe instance.
 * @param name The cstring name of the accelerator, \
 *                              NULL if we dont care for the name.
 * @param type Type of the accelerator, see vine_accel_type_e.
 * @return An vine_accel_s instance, NULL on failure.
 */
vine_accel_s* vine_pipe_find_accel(vine_pipe_s *pipe, const char *name,
                                   vine_accel_type_e type);

/**
 * Find a procedure matching the user specified criteria.
 *
 * @param pipe vine_pipe instance.
 * @param name The cstring name of the procedure.
 * @param type Type of the procedure, see vine_accel_type_e.
 * @return An vine_proc_s instance, NULL on failure.
 */
vine_proc_s* vine_pipe_find_proc(vine_pipe_s *pipe, const char *name,
                                 vine_accel_type_e type);

/**
 * Remove \c proc from the \c pipe procedure list.
 *
 * @param pipe The pipe instance where the procedure belongs.
 * @param proc The accelerator to be removed.
 * @return Returns 0 on success.
 */
int vine_pipe_delete_proc(vine_pipe_s *pipe, vine_proc_s *proc);

/**
 * Notify \c pipe that a new task of \c type has been added.
 */
void vine_pipe_add_task(vine_pipe_s *pipe,vine_accel_type_e type);

/**
 * Wait until a task of \c type has been added.
 */
void vine_pipe_wait_for_task(vine_pipe_s *pipe,vine_accel_type_e type);

/**
 * Wait until a task of any type or \c type is available..
 *
 * @param pipe A pipe instance.
 * @return type of task available
 */
vine_accel_type_e vine_pipe_wait_for_task_type_or_any(vine_pipe_s *pipe,vine_accel_type_e type);

/**
 * Destroy vine_pipe.
 *
 * \note Ensure you perform any cleanup(e.g. delete shared segment)
 * when return value becomes 0.
 *
 * @param pipe vine_pipe instance to be destroyed.
 * @return Number of remaining users of this shared segment.
 */
int vine_pipe_exit(vine_pipe_s *pipe);

#ifdef MMAP_FIXED
#define pointer_to_offset(TYPE, BASE, \
	                  VD) ( (TYPE)( (void*)(VD)-(void*)(BASE) ) )
#define offset_to_pointer(TYPE, BASE, \
	                  VD) ( (TYPE)( (char*)(BASE)+(size_t)(VD) ) )
#else /* ifdef MMAP_FIXED */
#define pointer_to_offset(TYPE, BASE, VD) ( (TYPE)VD )
#define offset_to_pointer(TYPE, BASE, VD) ( (TYPE)VD )
#endif /* ifdef MMAP_FIXED */

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_PIPE_HEADER */
