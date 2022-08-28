#ifndef ARAX_PIPE_HEADER
#define ARAX_PIPE_HEADER
#include <arax.h>
#include <utils/Kv.h>
#include "utils/queue.h"
#include "core/arax_accel.h"
#include "core/arax_task.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

typedef struct
{
    int pid;
    int threads;
} arax_process_tracker_s;

#define ARAX_PIPE_SHA_SIZE 48

/**
 * Shared Memory segment layout
 */
typedef struct arax_pipe
{
    char               sha[ARAX_PIPE_SHA_SIZE + 1];  /**< Git revision 48+1 for \0 */
    void               *self;                        /**< Pointer to myself */
    uint64_t           shm_size;                     /**< Size in bytes of shared region */
    uint64_t           processes;                    /**< Process counter - Processes using this */
    utils_spinlock     proc_lock;                    /**< Protect the proc_map */
    uint64_t           proc_map[ARAX_PROC_MAP_SIZE]; /**< Array contains PIDs of all mmaped processes */
    uint64_t           last_uid;                     /**< Last instance UID */
    arax_object_repo_s objs;                         /**< Arax object repository  */
    async_meta_s       async;                        /**< Async related metadata  */
    arax_throttle_s    throttle;

    int                cntrl_ready;      /**< Flag if != 0 means, controller is fully initialized*/
    async_condition_s  cntrl_ready_cond; /**< Condition for cntrl_ready */

    async_condition_s  orphan_cond; /**< Notify orphan changes */
    utils_list_s       orphan_vacs; /**< Unassigned virtual accels */

    utils_kv_s         ass_kv;     /**< Assignees KV, <assigne_id,task_count>*/
    utils_kv_s         metrics_kv; /**< Name, Metric */

    arch_alloc_s       allocator; /**< Allocator for this shared memory */
} arax_pipe_s;

/**
 * Similar to \c arax_init().
 *
 * As this should be called only by the controller, prior to any other Arax
 * function.
 *
 * After the controller process is initialized and ready to recieve tasks
 * \c arax_controller_init_done should be called.
 */
arax_pipe_s* arax_controller_init_start();

/**
 * Should only be called by the controller process, after it is ready to
 * recieve tasks. See \c arax_controller_init_start().
 */
void arax_controller_init_done();

/**
 * Get Arax revision
 *
 * @param pipe arax_pipe instance.
 * @return const string with Arax revision.
 */
const char* arax_pipe_get_revision(arax_pipe_s *pipe);

/**
 * Add \c vac to the list of orphan_vacs/ unassigned accels.
 *
 * @param pipe arax_pipe instance.
 * @param vac Unassigned/Orphan Virtual Acceleator instance.
 */
void arax_pipe_add_orphan_vaccel(arax_pipe_s *pipe, arax_vaccel_s *vac);

/**
 * Will return != 0 if there are orphan vaccels.
 *
 * @Note: This function may return old values.
 *
 * @param pipe arax_pipe instance.
 * @return 0 if no orphans, may return any non zero value if orphans exist
 */
int arax_pipe_have_orphan_vaccels(arax_pipe_s *pipe);

/**
 * Return an orphan/unassigned virtual accelerator or null.
 * Function will sleep if no orphans exist at the time of the call.
 * Returned arax_vaccel_s should either be assigned to a arax_accel_s using
 * \c arax_accel_add_vaccel(), or should be marked again as orphan using
 * \c arax_pipe_add_orphan_vaccel().
 *
 * @return Unassigned/Orphan Virtual Acceleator instance.
 */
arax_vaccel_s* arax_pipe_get_orphan_vaccel(arax_pipe_s *pipe);

/**
 * Remove specific \c vac for list of orphan vacs.
 */
void arax_pipe_remove_orphan_vaccel(arax_pipe_s *pipe, arax_vaccel_s *vac);

/**
 * This will return null to a blocked caller thread of \c arax_pipe_get_orphan_vaccel().
 * That should be used to signal thread termination.
 *
 * This function should be called once for every thread using arax_pipe_get_orphan_vaccel().
 *
 * @param pipe arax_pipe instance.
 */
void arax_pipe_orphan_stop(arax_pipe_s *pipe);

/**
 * Increase process counter for \c pipe.
 *
 * @param pipe arax_pipe instance.
 * @return Number of active processes before adding issuer.
 */
uint64_t arax_pipe_add_process(arax_pipe_s *pipe);

/**
 * Decrease process counter for \c pipe.
 *
 * @param pipe arax_pipe instance.
 * @return Number of active processes before removing issuer.
 */
uint64_t arax_pipe_del_process(arax_pipe_s *pipe);

/**
 * Return if we have to mmap, for the given pid.
 * This will return 1, only the first time it is callled with
 * a specific \c pid.
 */
int arax_pipe_have_to_mmap(arax_pipe_s *pipe, int pid);

/**
 * This should be called after munmap'ing \c pipe, in \c pid process.
 */
void arax_pipe_mark_unmap(arax_pipe_s *pipe, int pid);

/**
 * Return (and set if needed) the mmap location for \c pipe.
 *
 * @param pipe arax_pipe instance.
 */
void* arax_pipe_mmap_address(arax_pipe_s *pipe);

/**
 * Initialize a arax_pipe.
 *
 * \note This function must be called from all end points in order to
 * initialize a arax_pipe instance.Concurrent issuers will be serialized
 * and the returned arax_pipe instance will be initialized by the 'first'
 * issuer. All subsequent issuers will receive the already initialized
 * instance.
 *
 *
 * @param mem Shared memory pointer.
 * @param size Size of the shared memory in bytes.
 * @param enforce_version Set to 0 to make version mismatch non fatal.
 * @return An initialized arax_pipe_s instance.
 */
arax_pipe_s* arax_pipe_init(void *mem, size_t size, int enforce_version);

/**
 * Remove \c accel from the \c pipe accelerator list.
 *
 * @param pipe The pipe instance where the accelerator belongs.
 * @param accel The accelerator to be removed.
 * @return Returns 0 on success.
 */
int arax_pipe_delete_accel(arax_pipe_s *pipe, arax_accel_s *accel);

/**
 * Find an accelerator matching the user specified criteria.
 *
 * @param pipe arax_pipe instance.
 * @param name The cstring name of the accelerator, \
 *                              NULL if we dont care for the name.
 * @param type Type of the accelerator, see arax_accel_type_e.
 * @return An arax_accel_s instance, NULL on failure.
 */
arax_accel_s* arax_pipe_find_accel(arax_pipe_s *pipe, const char *name,
  arax_accel_type_e type);

/**
 * Find a procedure matching the user specified criteria.
 *
 * @param pipe arax_pipe instance.
 * @param name The cstring name of the procedure.
 * @return An arax_proc_s instance, NULL on failure.
 */
arax_proc_s* arax_pipe_find_proc(arax_pipe_s *pipe, const char *name);

/**
 * Destroy arax_pipe.
 *
 * \note Ensure you perform any cleanup(e.g. delete shared segment)
 * when return value becomes 0.
 *
 * @param pipe arax_pipe instance to be destroyed.
 * @return Number of remaining users of this shared segment.
 */
int arax_pipe_exit(arax_pipe_s *pipe);

#ifdef ARAX_THROTTLE_DEBUG
#define ARAX_PIPE_THOTTLE_DEBUG_PARAMS , const char *parent
#define ARAX_PIPE_THOTTLE_DEBUG_FUNC(FUNC) __ ## FUNC
#define arax_pipe_size_inc(PIPE, SZ)       __arax_pipe_size_inc(PIPE, SZ, __func__)
#define arax_pipe_size_dec(PIPE, SZ)       __arax_pipe_size_dec(PIPE, SZ, __func__)
#else
#define ARAX_PIPE_THOTTLE_DEBUG_PARAMS
#define ARAX_PIPE_THOTTLE_DEBUG_FUNC(FUNC) FUNC
#endif

/**
 * Increments available size of gpu by sz
 *
 * @param pipe   pipe for shm
 * @param sz     Size of added data
 * @return       Nothing .
 */
void ARAX_PIPE_THOTTLE_DEBUG_FUNC(arax_pipe_size_inc)(arax_pipe_s * pipe, size_t sz ARAX_PIPE_THOTTLE_DEBUG_PARAMS);

/**
 * Decrements available size of gpu by sz
 *
 * @param pipe   pipe for shm
 * @param sz     size of removed data
 * @return       Nothing .
 */
void ARAX_PIPE_THOTTLE_DEBUG_FUNC(arax_pipe_size_dec)(arax_pipe_s * pipe, size_t sz ARAX_PIPE_THOTTLE_DEBUG_PARAMS);

/**
 * Gets available size of shm
 *
 * @param pipe   pipe for shm
 * @return       Avaliable size of shm
 */
size_t arax_pipe_get_available_size(arax_pipe_s *pipe);

/**
 * Gets available total size of shm
 *
 * @param pipe   pipe for shm
 * @return       Total size of shm
 */
size_t arax_pipe_get_total_size(arax_pipe_s *pipe);

#ifdef MMAP_FIXED
#define pointer_to_offset(TYPE, BASE, \
      VD)                                 ( (TYPE) ( (void *) (VD) -(void *) (BASE) ) )
#define offset_to_pointer(TYPE, BASE, \
      VD)                                 ( (TYPE) ( (char *) (BASE) + (size_t) (VD) ) )
#else /* ifdef MMAP_FIXED */
#define pointer_to_offset(TYPE, BASE, VD) ( (TYPE) VD )
#define offset_to_pointer(TYPE, BASE, VD) ( (TYPE) VD )
#endif /* ifdef MMAP_FIXED */

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef ARAX_PIPE_HEADER */
